/*
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ffi::OsString,
    fmt::Debug,
    os::unix::ffi::OsStrExt,
    sync::{Arc, LazyLock},
};

use anyhow::Result;
use bytes::{BufMut, Bytes};
use parking_lot::{Mutex, RwLock};

use crate::{
    command::AutoCommandVar,
    error, error_loc,
    eval::Frame,
    loc::Loc,
    strutil::{WordWriter, has_path_prefix},
    symtab::{Symtab, with_symtab},
    warn_loc,
};
use crate::{
    eval::Evaluator,
    expr::{Evaluable, Value},
    stmt::AssignOp,
    symtab::Symbol,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VarOrigin {
    Default,
    Environment,
    EnvironmentOverride,
    File,
    CommandLine,
    Override,
    Automatic,
}

pub fn get_origin_str(origin: VarOrigin) -> &'static str {
    match origin {
        VarOrigin::Default => "default",
        VarOrigin::Environment => "environment",
        VarOrigin::EnvironmentOverride => "environment override",
        VarOrigin::File => "file",
        VarOrigin::CommandLine => "command line",
        VarOrigin::Override => "override",
        VarOrigin::Automatic => "automatic",
    }
}

pub type Var = Arc<RwLock<Variable>>;

#[derive(Debug)]
pub struct Variable {
    loc: Option<Loc>,

    definition: Option<Arc<Frame>>,

    origin: VarOrigin,

    pub assign_op: Option<AssignOp>,
    pub readonly: bool,
    pub deprecated: Option<Arc<String>>,
    obsolete: Option<Arc<String>>,

    visibility_prefix: Option<Vec<OsString>>,

    value: InnerVar,
}

#[derive(Debug)]
pub enum InnerVar {
    Simple(Vec<u8>),
    Recursive { v: Arc<Value>, orig: Bytes },
    AutoCommand(Symbol, AutoCommandVar),
    ShellStatus,
    VariableNames { name: Bytes, all: bool },
}

impl Variable {
    pub fn loc(&self) -> &Option<Loc> {
        &self.loc
    }
    pub fn origin(&self) -> VarOrigin {
        self.origin
    }
    pub fn definition(&self) -> &Option<Arc<Frame>> {
        &self.definition
    }
    pub fn obsolete(&self) -> bool {
        self.obsolete.is_some()
    }
    pub fn set_obsolete(&mut self, message: Arc<String>) {
        self.obsolete = Some(message);
    }
    pub fn flavor(&self) -> &'static str {
        match &self.value {
            InnerVar::Simple(_) => "simple",
            InnerVar::Recursive { .. } => "recursive",
            InnerVar::AutoCommand(_, _) => "undefined",
            InnerVar::ShellStatus => "simple",
            InnerVar::VariableNames { .. } => "kati_variable_names",
        }
    }
    pub fn used(&self, ev: &Evaluator, sym: &Symbol) -> Result<()> {
        if let Some(obsolete) = &self.obsolete {
            error_loc!(ev.loc.as_ref(), "*** {sym} is obsolete{obsolete}.");
        }
        if let Some(deprecated) = &self.deprecated {
            warn_loc!(ev.loc.as_ref(), "{sym} has been deprecated{deprecated}.");
        }
        Ok(())
    }
    pub fn set_visibility_prefix(&mut self, prefixes: Vec<OsString>, name: &Symbol) -> Result<()> {
        if self.visibility_prefix.is_none() {
            self.visibility_prefix = Some(prefixes);
        } else if self.visibility_prefix != Some(prefixes) {
            error!("Visibility prefix conflict on variable: {name}");
        }
        Ok(())
    }
    pub fn immediate_eval(&self) -> bool {
        matches!(&self.value, InnerVar::Simple(_))
    }
    pub fn append_var(
        &mut self,
        v: Arc<Value>,
        frame: Arc<Frame>,
        loc: Option<&Loc>,
    ) -> Result<()> {
        match &mut self.value {
            InnerVar::Simple(_) => {
                panic!("append_var should not be used when immediate_eval returns true")
            }
            InnerVar::Recursive { v: prev, .. } => {
                *prev = Arc::new(Value::List(
                    prev.loc(),
                    vec![
                        prev.clone(),
                        Arc::new(Value::Literal(None, Bytes::from_static(b" "))),
                        v,
                    ],
                ));
                self.definition = Some(frame);
            }
            InnerVar::AutoCommand(sym, _) => {
                error_loc!(loc, "appending to ${sym} is not supported");
            }
            InnerVar::ShellStatus => panic!(),
            InnerVar::VariableNames { .. } => panic!(),
        }
        Ok(())
    }
    pub fn append_str(&mut self, buf: &Bytes, frame: Arc<Frame>) -> Result<()> {
        match &mut self.value {
            InnerVar::Simple(s) => {
                s.push(b' ');
                s.extend_from_slice(buf);
                self.definition = Some(frame);
            }
            InnerVar::Recursive { v: prev, .. } => {
                *prev = Arc::new(Value::List(
                    prev.loc(),
                    vec![
                        prev.clone(),
                        Arc::new(Value::Literal(None, Bytes::from_static(b" "))),
                        Arc::new(Value::Literal(None, buf.clone())),
                    ],
                ));
                self.definition = Some(frame);
            }
            InnerVar::AutoCommand(sym, _) => {
                error!("appending to ${sym} is not supported");
            }
            InnerVar::ShellStatus => panic!(),
            InnerVar::VariableNames { .. } => panic!(),
        }
        Ok(())
    }
    pub fn check_current_referencing_file(&self, loc: &Option<Loc>, sym: Symbol) -> Result<()> {
        let Some(prefixes) = &self.visibility_prefix else {
            return Ok(());
        };
        let loc = loc.clone().unwrap_or_default();
        let mut valid = false;
        for prefix in prefixes {
            if has_path_prefix(&loc.filename.as_bytes(), prefix.as_bytes()) {
                valid = true;
                break;
            }
        }
        if !valid {
            let s = prefixes
                .iter()
                .map(|s| s.to_string_lossy())
                .collect::<Vec<Cow<str>>>()
                .join("\n");
            error!(
                "{} is not a valid file to reference variable {sym}. Line #{}.\nValid file prefixes:\n{s}",
                loc.filename, loc.line
            );
        }
        Ok(())
    }
    pub fn string(&self) -> Result<Cow<'_, [u8]>> {
        Ok(match &self.value {
            InnerVar::Simple(s) => Cow::Borrowed(s.as_slice()),
            InnerVar::Recursive { v: _, orig } => Cow::Borrowed(orig),
            InnerVar::AutoCommand(sym, _) => {
                error!("$(value {sym}) is not implemented yet");
            }
            InnerVar::ShellStatus => {
                Cow::Owned(if let Some(status) = SHELL_STATUS.lock().as_ref() {
                    status.to_string().as_bytes().to_vec()
                } else {
                    Vec::new()
                })
            }
            InnerVar::VariableNames { name, .. } => Cow::Borrowed(name),
        })
    }

    pub fn new_simple(
        origin: VarOrigin,
        frame: Option<Arc<Frame>>,
        loc: Option<Loc>,
    ) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc,
            definition: frame,
            origin,
            assign_op: None,
            readonly: false,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::Simple(Vec::new()),
        }))
    }

    pub fn with_simple_string(
        value: Bytes,
        origin: VarOrigin,
        frame: Option<Arc<Frame>>,
        loc: Option<Loc>,
    ) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc,
            definition: frame,
            origin,
            assign_op: None,
            readonly: false,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::Simple(value.to_vec()),
        }))
    }

    pub fn with_simple_value(
        origin: VarOrigin,
        frame: Option<Arc<Frame>>,
        loc: Option<Loc>,
        ev: &mut Evaluator,
        v: &Value,
    ) -> Result<Arc<RwLock<Self>>> {
        let value = v.eval_to_buf(ev)?;
        Ok(Arc::new(RwLock::new(Self {
            loc,
            definition: frame,
            origin,
            assign_op: None,
            readonly: false,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::Simple(value.to_vec()),
        })))
    }

    pub fn new_recursive(
        v: Arc<Value>,
        origin: VarOrigin,
        frame: Option<Arc<Frame>>,
        loc: Option<Loc>,
        orig: Bytes,
    ) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc,
            definition: frame,
            origin,
            assign_op: None,
            readonly: false,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::Recursive { v, orig },
        }))
    }

    pub fn new_autocommand(sym: Symbol, a: AutoCommandVar) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc: None,
            definition: None,
            origin: VarOrigin::Automatic,
            assign_op: None,
            readonly: false,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::AutoCommand(sym, a),
        }))
    }

    pub fn new_shell_status_var() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc: None,
            definition: None,
            origin: VarOrigin::Override,
            assign_op: Some(AssignOp::ColonEq),
            readonly: true,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::ShellStatus,
        }))
    }

    pub fn new_variable_names(name: &'static [u8], all: bool) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            loc: None,
            definition: None,
            origin: VarOrigin::Override,
            assign_op: Some(AssignOp::ColonEq),
            readonly: true,
            deprecated: None,
            obsolete: None,
            visibility_prefix: None,
            value: InnerVar::VariableNames {
                name: Bytes::from_static(name),
                all,
            },
        }))
    }
}

impl Evaluable for Variable {
    fn eval(&self, ev: &mut crate::eval::Evaluator, out: &mut dyn BufMut) -> Result<()> {
        match &self.value {
            InnerVar::Simple(v) => {
                out.put_slice(v);
            }
            InnerVar::Recursive { v, .. } => {
                v.eval(ev, out)?;
            }
            InnerVar::AutoCommand(_, a) => {
                a.eval(ev, out)?;
            }
            InnerVar::ShellStatus => {
                if ev.is_evaluating_command {
                    error_loc!(
                        ev.loc.as_ref(),
                        "Kati does not support using .SHELLSTATUS inside of a rule"
                    );
                }

                if let Some(status) = SHELL_STATUS.lock().as_ref() {
                    out.put_slice(format!("{status}").as_bytes());
                }
            }
            InnerVar::VariableNames { all, .. } => {
                let mut ww = WordWriter::new(out);
                let symbols = global_var_names(|var| !var.read().obsolete());
                for (sym, entry) in symbols {
                    if !*all
                        && let Some(var) = peek_global_var(sym)
                        && var.read().is_func()
                    {
                        continue;
                    }
                    ww.write(&entry);
                }
            }
        }
        Ok(())
    }
    fn is_func(&self) -> bool {
        match &self.value {
            InnerVar::Simple(_) => false,
            InnerVar::Recursive { v, .. } => v.is_func(),
            InnerVar::AutoCommand(_, _) => true,
            InnerVar::ShellStatus => false,
            InnerVar::VariableNames { .. } => false,
        }
    }
}

static SHELL_STATUS: LazyLock<Mutex<Option<i32>>> = LazyLock::new(|| Mutex::new(None));

pub fn set_shell_status_var(status: i32) {
    *SHELL_STATUS.lock() = Some(status)
}

pub static USED_ENV_VARS: LazyLock<Mutex<HashSet<Symbol>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Make's global variable scope: the bindings of the outermost scope, keyed by
/// interned [`Symbol`].
///
/// It stores no names, only bindings, so it can be constructed, replaced, or
/// dropped without disturbing the interner that produced its keys, and
/// interning a name does not create a binding here.
// [spec:ronin:req:make.scope-separation]
pub struct GlobalVars {
    /// Indexed by [`Symbol::index`]. Sparse: most interned names never become
    /// global variables.
    vars: Vec<Option<Var>>,
}

impl Default for GlobalVars {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalVars {
    /// An empty scope. No interner is needed to make one.
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    /// A scope carrying the bindings kati defines before any makefile is read.
    ///
    /// This is the one place the two halves meet, and only because the builtin
    /// names have to be interned somewhere: the scope keeps no reference to
    /// `symtab` afterwards.
    pub fn with_builtins(symtab: &mut Symtab) -> Self {
        let mut vars = Self::new();
        vars.define(
            symtab.intern(".SHELLSTATUS"),
            Variable::new_shell_status_var(),
        );
        vars.define(
            symtab.intern(".VARIABLES"),
            Variable::new_variable_names(b".VARIABLES", true),
        );
        vars.define(
            symtab.intern(".KATI_SYMBOLS"),
            Variable::new_variable_names(b".KATI_SYMBOLS", false),
        );
        vars
    }

    /// The binding for `sym`, without recording the read.
    pub fn peek(&self, sym: Symbol) -> Option<Var> {
        self.vars.get(sym.index())?.clone()
    }

    /// Bind `sym` unconditionally, returning what it was bound to before.
    pub fn define(&mut self, sym: Symbol, var: Var) -> Option<Var> {
        self.replace(sym, Some(var))
    }

    /// Set or clear the binding for `sym` unconditionally, returning what it
    /// was bound to before. Bypasses the precedence rules in [`Self::assign`],
    /// which is what makes it right for save-and-restore.
    pub fn replace(&mut self, sym: Symbol, var: Option<Var>) -> Option<Var> {
        let idx = sym.index();
        if idx >= self.vars.len() {
            self.vars.resize(idx + 1, None);
        }
        std::mem::replace(&mut self.vars[idx], var)
    }

    /// Assign to `sym` under GNU Make's readonly and origin precedence rules,
    /// which can decline the assignment silently.
    pub fn assign(
        &mut self,
        sym: Symbol,
        var: Var,
        is_override: bool,
        readonly: Option<&mut bool>,
    ) -> Result<()> {
        let idx = sym.index();
        if idx >= self.vars.len() {
            self.vars.resize(idx + 1, None);
        }
        let entry = self.vars.get_mut(idx).unwrap();
        if let Some(orig) = entry {
            if orig.read().readonly {
                if let Some(readonly) = readonly {
                    *readonly = true;
                } else {
                    error!("*** cannot assign to readonly variable: {sym}");
                }
                return Ok(());
            } else if let Some(readonly) = readonly {
                *readonly = false;
            }
            let origin = orig.read().origin();
            if !is_override
                && (origin == VarOrigin::Override || origin == VarOrigin::EnvironmentOverride)
            {
                return Ok(());
            }
            if origin == VarOrigin::CommandLine && var.read().origin() == VarOrigin::File {
                return Ok(());
            }
            if origin == VarOrigin::Automatic {
                error!("overriding automatic variable is not implemented yet");
            }
        }
        *entry = Some(var);
        Ok(())
    }

    /// Every binding satisfying `filter`, in symbol order.
    pub fn matching<F: Fn(&Var) -> bool>(&self, filter: F) -> Vec<(Symbol, Var)> {
        self.vars
            .iter()
            .enumerate()
            .filter_map(|(idx, var)| {
                let var = var.clone()?;
                let sym = Symbol::from_index(idx)?;
                filter(&var).then_some((sym, var))
            })
            .collect()
    }
}

/// The process-global variable scope.
///
/// Temporary: `kati-session-value` moves this into the session. It is a
/// separate lock from the interner's, so a diagnostic that renders a symbol may
/// be raised while it is held.
static GLOBAL_VARS: LazyLock<Mutex<GlobalVars>> =
    LazyLock::new(|| Mutex::new(with_symtab(GlobalVars::with_builtins)));

/// Read a global variable without recording the read.
pub fn peek_global_var(sym: Symbol) -> Option<Var> {
    GLOBAL_VARS.lock().peek(sym)
}

/// Read a global variable, recording it if it came from the environment.
pub fn get_global_var(sym: Symbol) -> Option<Var> {
    let var = GLOBAL_VARS.lock().peek(sym)?;
    match var.read().origin() {
        VarOrigin::Environment | VarOrigin::EnvironmentOverride => {
            USED_ENV_VARS.lock().insert(sym);
        }
        _ => {}
    }
    Some(var)
}

/// Assign to a global variable under Make's precedence rules.
pub fn set_global_var(
    sym: Symbol,
    var: Var,
    is_override: bool,
    readonly: Option<&mut bool>,
) -> Result<()> {
    GLOBAL_VARS.lock().assign(sym, var, is_override, readonly)
}

/// The names of the global variables satisfying `filter`, in symbol order.
pub fn global_var_names<F: Fn(&Var) -> bool>(filter: F) -> Vec<(Symbol, Bytes)> {
    // The scope lock is released before the interner is consulted: these are
    // two structures with two locks, and nothing takes both.
    let matched = GLOBAL_VARS.lock().matching(filter);
    with_symtab(|symtab| {
        matched
            .into_iter()
            .map(|(sym, _)| (sym, symtab.name(sym)))
            .collect()
    })
}

/// Binds a global variable for as long as it is held, then restores whatever
/// the symbol was bound to before, including nothing.
pub struct ScopedGlobalVar {
    sym: Symbol,
    orig: Option<Var>,
}

impl ScopedGlobalVar {
    pub fn new(sym: Symbol, var: Var) -> Result<Self> {
        let orig = GLOBAL_VARS.lock().replace(sym, Some(var));
        Ok(Self { sym, orig })
    }
}

impl Drop for ScopedGlobalVar {
    fn drop(&mut self) {
        GLOBAL_VARS.lock().replace(self.sym, self.orig.take());
    }
}

pub struct Vars(pub Mutex<HashMap<Symbol, Var>>);

impl Default for Vars {
    fn default() -> Self {
        Self::new()
    }
}

impl Vars {
    pub fn new() -> Self {
        Vars(Mutex::new(HashMap::new()))
    }

    pub fn lookup(&self, sym: Symbol) -> Option<Var> {
        let ret = self.0.lock().get(&sym).cloned()?;
        match ret.read().origin() {
            VarOrigin::Environment | VarOrigin::EnvironmentOverride => {
                USED_ENV_VARS.lock().insert(sym);
            }
            _ => {}
        }
        Some(ret)
    }

    pub fn peek(&self, sym: Symbol) -> Option<Var> {
        self.0.lock().get(&sym).cloned()
    }

    pub fn assign(&self, sym: Symbol, var: Var, readonly: &mut bool) -> Result<()> {
        *readonly = false;
        let mut vars = self.0.lock();
        if let Some(orig) = vars.get_mut(&sym) {
            if orig.read().readonly {
                *readonly = true;
                return Ok(());
            }
            match orig.read().origin() {
                VarOrigin::Override | VarOrigin::EnvironmentOverride => return Ok(()),
                VarOrigin::Automatic => {
                    error!("overriding automatic variable is not implemented yet");
                }
                _ => {}
            }
            *orig = var;
        } else {
            vars.insert(sym, var);
        }
        Ok(())
    }

    pub fn merge_from(&self, vars: &Vars) {
        let mut to = self.0.lock();
        let from = vars.0.lock();

        for (sym, var) in from.iter() {
            to.insert(*sym, var.clone());
        }
    }
}

impl Clone for Vars {
    fn clone(&self) -> Self {
        let m = self.0.lock();
        Self(Mutex::new(m.clone()))
    }
}

impl Debug for Vars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vars = self.0.lock();
        f.debug_map().entries(vars.iter()).finish()
    }
}

pub struct ScopedVar {
    vars: Arc<Vars>,
    sym: Symbol,
    orig: Option<Var>,
}

impl ScopedVar {
    pub fn new(vars: Arc<Vars>, sym: Symbol, var: Var) -> Self {
        let orig = {
            let mut vars = vars.0.lock();
            vars.insert(sym, var)
        };
        Self { vars, sym, orig }
    }
}

impl Drop for ScopedVar {
    fn drop(&mut self) {
        let mut vars = self.vars.0.lock();
        if let Some(orig) = self.orig.clone() {
            vars.insert(self.sym, orig);
        } else {
            vars.remove(&self.sym);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interning a name must not create, read, or modify a binding.
    // [spec:ronin:req:make.scope-separation/test]
    #[test]
    fn test_interning_defines_nothing() {
        let mut symtab = Symtab::new();
        let scope = GlobalVars::new();
        let sym = symtab.intern("SOME_VARIABLE");
        assert!(scope.peek(sym).is_none());
        assert!(scope.matching(|_| true).is_empty());
    }

    /// A scope can be replaced wholesale without reinterning its symbols.
    // [spec:ronin:req:make.scope-separation/test]
    #[test]
    fn test_scope_replaced_without_reinterning() {
        let mut symtab = Symtab::new();
        let sym = symtab.intern("CFLAGS");

        let mut first = GlobalVars::new();
        first
            .assign(
                sym,
                Variable::with_simple_string(
                    Bytes::from_static(b"-O2"),
                    VarOrigin::File,
                    None,
                    None,
                ),
                false,
                None,
            )
            .unwrap();
        assert_eq!(
            first.peek(sym).unwrap().read().string().unwrap().as_ref(),
            b"-O2"
        );

        // A fresh scope over the same interner starts empty, and the symbol
        // still resolves to its name without being interned again.
        let second = GlobalVars::new();
        assert!(second.peek(sym).is_none());
        assert_eq!(symtab.name(sym), Bytes::from_static(b"CFLAGS"));
        assert_eq!(symtab.intern("CFLAGS"), sym);

        // The replaced scope is untouched by the new one.
        assert!(first.peek(sym).is_some());
    }

    /// The builtins live in the scope, not the interner.
    #[test]
    fn test_builtins_are_scope_state() {
        let mut symtab = Symtab::new();
        let scope = GlobalVars::with_builtins(&mut symtab);
        let shell_status = symtab.intern(".SHELLSTATUS");
        assert!(scope.peek(shell_status).is_some());
        // Same interner, a scope without builtins: the name interns to the same
        // handle but has no binding.
        assert!(GlobalVars::new().peek(shell_status).is_none());
    }

    #[test]
    fn test_replace_restores_absence() {
        let mut symtab = Symtab::new();
        let sym = symtab.intern("TMP");
        let mut scope = GlobalVars::new();
        let var = Variable::new_simple(VarOrigin::Automatic, None, None);
        assert!(scope.replace(sym, Some(var)).is_none());
        assert!(scope.replace(sym, None).is_some());
        assert!(scope.peek(sym).is_none());
    }
}
