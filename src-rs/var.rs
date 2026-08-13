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
    sync::Arc,
};

use anyhow::Result;
use bytes::{BufMut, Bytes};
use parking_lot::{Mutex, RwLock};

use crate::{
    command::AutoCommandVar,
    error, error_loc,
    eval::Frame,
    loc::Loc,
    session::{Context, Session},
    strutil::{WordWriter, has_path_prefix},
    symtab::{Interner, Symtab},
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

#[derive(Clone, Debug)]
pub struct Variable {
    loc: Option<Loc>,

    definition: Option<Arc<Frame>>,

    origin: VarOrigin,

    pub assign_op: Option<AssignOp>,
    pub readonly: bool,
    /// `private`: reachable from the scope that defined it and from nothing
    /// that reaches that scope through a parent.
    pub is_private: bool,
    pub deprecated: Option<Arc<String>>,
    obsolete: Option<Arc<String>>,

    visibility_prefix: Option<Vec<OsString>>,

    value: InnerVar,
}

#[derive(Clone, Debug)]
pub enum InnerVar {
    Simple(Vec<u8>),
    Recursive { v: Arc<Value>, orig: Bytes },
    AutoCommand(Symbol, AutoCommandVar),
    ShellStatus,
    VariableNames { name: Bytes, all: bool },
}

/// The pieces of a recursive value that `+=` has appended to, in order.
///
/// The space `+=` writes is a separator, so it belongs between two values and
/// nowhere else. A variable whose text is empty has no left-hand value for it
/// to separate, and takes the appended expression on its own.
///
/// GNU Make asks this of the text as written rather than of what the text
/// expands to, which is why `V = $(EMPTY)` still counts as something to append
/// to: `V += x` reads back as `$(EMPTY) x`, and expands with the space.
fn appended_values(prev: Arc<Value>, prev_text: &Bytes, added: Arc<Value>) -> Vec<Arc<Value>> {
    if prev_text.is_empty() {
        return vec![prev, added];
    }
    vec![
        prev,
        Arc::new(Value::Literal(None, Bytes::from_static(b" "))),
        added,
    ]
}

/// The same join over the text those values were written as.
///
/// A recursive variable keeps its text beside its expression because `$(value)`
/// reads it back, and because the next `+=` asks this of it in turn.
fn appended_text(prev: &Bytes, added: &Bytes) -> Bytes {
    if prev.is_empty() {
        return added.clone();
    }
    let mut joined = Vec::with_capacity(prev.len() + 1 + added.len());
    joined.extend_from_slice(prev);
    joined.push(b' ');
    joined.extend_from_slice(added);
    Bytes::from(joined)
}

impl Variable {
    /// Replace only this variable's recursive expression.
    ///
    /// GNU Make canonicalises `MAKEFLAGS` after assigning it while preserving
    /// the assignment's origin, location, `private`, and `override` metadata.
    pub fn replace_recursive_value(&mut self, value: Arc<Value>, original: Bytes) {
        self.value = InnerVar::Recursive {
            v: value,
            orig: original,
        };
    }

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
    /// Whether the variable's text — what was written, before any expansion —
    /// holds nothing.
    ///
    /// A simple variable's text is its value; a recursive one's is the
    /// expression rather than what the expression produces. GNU Make reads
    /// `.DEFAULT_GOAL` this way when it decides whether a recorded target
    /// should become the default goal, so `.DEFAULT_GOAL = $(EMPTY)` keeps
    /// selection disarmed while expanding to nothing at all.
    ///
    /// A variable whose value is computed rather than written — a shell
    /// status, the name list — has no text and is never empty in this sense.
    pub fn text_is_empty(&self) -> bool {
        match &self.value {
            InnerVar::Simple(value) => value.is_empty(),
            InnerVar::Recursive { orig, .. } => orig.is_empty(),
            InnerVar::AutoCommand(_, _)
            | InnerVar::ShellStatus
            | InnerVar::VariableNames { .. } => false,
        }
    }
    pub fn used(&self, ev: &Evaluator, sym: &Symbol) -> Result<()> {
        if let Some(obsolete) = &self.obsolete {
            error_loc!(
                ev,
                ev.loc.as_ref(),
                "*** {} is obsolete{obsolete}.",
                sym.display(ev)
            );
        }
        if let Some(deprecated) = &self.deprecated {
            warn_loc!(
                ev,
                ev.loc.as_ref(),
                "{} has been deprecated{deprecated}.",
                sym.display(ev)
            );
        }
        Ok(())
    }
    pub fn set_visibility_prefix(
        &mut self,
        names: &impl Interner,
        prefixes: Vec<OsString>,
        name: &Symbol,
    ) -> Result<()> {
        if self.visibility_prefix.is_none() {
            self.visibility_prefix = Some(prefixes);
        } else if self.visibility_prefix != Some(prefixes) {
            error!(
                "Visibility prefix conflict on variable: {}",
                name.display(names)
            );
        }
        Ok(())
    }
    pub fn immediate_eval(&self) -> bool {
        matches!(&self.value, InnerVar::Simple(_))
    }

    /// Copy an existing value before appending, changing only the provenance
    /// of the assignment that will replace it if precedence permits.
    pub fn clone_for_assignment(
        &self,
        origin: VarOrigin,
        definition: Option<Arc<Frame>>,
        loc: Option<Loc>,
    ) -> Var {
        let mut variable = self.clone();
        variable.origin = origin;
        variable.definition = definition;
        variable.loc = loc;
        Arc::new(RwLock::new(variable))
    }

    /// Copy a command-line value into the recursive environment form recipes
    /// receive. Recursive expressions stay deferred; simple values become
    /// literal recursive expressions without being evaluated again.
    pub fn clone_for_recipe_environment(&self) -> Var {
        let mut variable = self.clone();
        if let InnerVar::Simple(value) = &variable.value {
            let value = Bytes::from(value.clone());
            variable.value = InnerVar::Recursive {
                v: Arc::new(Value::Literal(None, value.clone())),
                orig: value,
            };
        }
        variable.origin = VarOrigin::Environment;
        variable.assign_op = None;
        Arc::new(RwLock::new(variable))
    }
    /// Append an unexpanded expression, `+=` onto a recursive variable.
    ///
    /// `text` is that expression as it was written, which the variable keeps
    /// alongside it and which decides where the separator goes.
    pub fn append_var(
        &mut self,
        ctx: &impl Context,
        v: Arc<Value>,
        text: &Bytes,
        frame: Arc<Frame>,
        loc: Option<&Loc>,
    ) -> Result<()> {
        match &mut self.value {
            InnerVar::Simple(_) => {
                panic!("append_var should not be used when immediate_eval returns true")
            }
            InnerVar::Recursive { v: prev, orig } => {
                *prev = Arc::new(Value::List(
                    prev.loc(),
                    appended_values(prev.clone(), orig, v),
                ));
                *orig = appended_text(orig, text);
                self.definition = Some(frame);
            }
            InnerVar::AutoCommand(sym, _) => {
                error_loc!(
                    ctx,
                    loc,
                    "appending to ${} is not supported",
                    sym.display(ctx)
                );
            }
            InnerVar::ShellStatus => panic!(),
            InnerVar::VariableNames { .. } => panic!(),
        }
        Ok(())
    }
    /// Append text that needs no further expansion, `+=` onto a simple
    /// variable and the way the reader grows `MAKEFILE_LIST`.
    pub fn append_str(
        &mut self,
        names: &impl Interner,
        buf: &Bytes,
        frame: Arc<Frame>,
    ) -> Result<()> {
        match &mut self.value {
            InnerVar::Simple(s) => {
                if !s.is_empty() {
                    s.push(b' ');
                }
                s.extend_from_slice(buf);
                self.definition = Some(frame);
            }
            InnerVar::Recursive { v: prev, orig } => {
                let added = Arc::new(Value::Literal(None, buf.clone()));
                *prev = Arc::new(Value::List(
                    prev.loc(),
                    appended_values(prev.clone(), orig, added),
                ));
                *orig = appended_text(orig, buf);
                self.definition = Some(frame);
            }
            InnerVar::AutoCommand(sym, _) => {
                error!("appending to ${} is not supported", sym.display(names));
            }
            InnerVar::ShellStatus => panic!(),
            InnerVar::VariableNames { .. } => panic!(),
        }
        Ok(())
    }
    pub fn check_current_referencing_file(
        &self,
        names: &impl Interner,
        loc: &Option<Loc>,
        sym: Symbol,
    ) -> Result<()> {
        let Some(prefixes) = &self.visibility_prefix else {
            return Ok(());
        };
        let loc = loc.clone().unwrap_or_default();
        let mut valid = false;
        for prefix in prefixes {
            if has_path_prefix(&loc.filename.as_bytes(names), prefix.as_bytes()) {
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
                "{} is not a valid file to reference variable {}. Line #{}.\nValid file prefixes:\n{s}",
                loc.filename.display(names),
                sym.display(names),
                loc.line
            );
        }
        Ok(())
    }
    pub fn string(&self, session: &Session) -> Result<Cow<'_, [u8]>> {
        Ok(match &self.value {
            InnerVar::Simple(s) => Cow::Borrowed(s.as_slice()),
            InnerVar::Recursive { v: _, orig } => Cow::Borrowed(orig),
            InnerVar::AutoCommand(sym, _) => {
                error!("$(value {}) is not implemented yet", sym.display(session));
            }
            InnerVar::ShellStatus => Cow::Owned(if let Some(status) = session.shell_status {
                status.to_string().as_bytes().to_vec()
            } else {
                Vec::new()
            }),
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
            is_private: false,
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
            is_private: false,
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
            is_private: false,
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
            is_private: false,
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
            is_private: false,
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
            is_private: false,
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
            is_private: false,
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
                if let Some(status) = ev.session.shell_status {
                    out.put_slice(format!("{status}").as_bytes());
                }
            }
            InnerVar::VariableNames { all, .. } => {
                let mut ww = WordWriter::new(out);
                let symbols = ev.session.global_var_names(|var| !var.read().obsolete());
                for (sym, entry) in symbols {
                    if !*all
                        && let Some(var) = ev.session.peek_global_var(sym)
                        && var.read().is_func(&ev.session.symtab)
                    {
                        continue;
                    }
                    ww.write(&entry);
                }
            }
        }
        Ok(())
    }
    fn is_func(&self, names: &Symtab) -> bool {
        match &self.value {
            InnerVar::Simple(_) => false,
            InnerVar::Recursive { v, .. } => v.is_func(names),
            InnerVar::AutoCommand(_, _) => true,
            InnerVar::ShellStatus => false,
            InnerVar::VariableNames { .. } => false,
        }
    }
}

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
    /// No interner is needed: the builtin names are among the ones every
    /// interner preloads, so their handles are `const`.
    pub fn with_builtins() -> Self {
        let mut vars = Self::new();
        vars.define(Symbol::SHELLSTATUS, Variable::new_shell_status_var());
        vars.define(
            Symbol::RECIPEPREFIX,
            Variable::with_simple_string(Bytes::new(), VarOrigin::Default, None, None),
        );
        // GNU Make binds this one in `main.c` rather than in the tool
        // catalogue, and the difference is observable: `-R` withdraws the
        // catalogue but leaves `.SHELLFLAGS` at `-c`, because a switch asking
        // for a clean namespace is not asking for an unusable shell. Undefining
        // it deliberately does leave the shell no flags, exactly as it does
        // there.
        vars.define(
            Symbol::SHELLFLAGS,
            Variable::with_simple_string(Bytes::from_static(b"-c"), VarOrigin::Default, None, None),
        );
        vars.define(
            Symbol::VARIABLES,
            Variable::new_variable_names(b".VARIABLES", true),
        );
        vars.define(
            Symbol::KATI_SYMBOLS,
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
        names: &impl Interner,
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
                    error!(
                        "*** cannot assign to readonly variable: {}",
                        sym.display(names)
                    );
                }
                return Ok(());
            } else if let Some(readonly) = readonly {
                *readonly = false;
            }
            let origin = orig.read().origin();
            let assigning = var.read().origin();
            // `-e` lifts the environment above the makefile and no higher: a
            // command-line assignment still outranks it, as it outranks
            // everything an `override` directive did not write.
            if !is_override
                && (origin == VarOrigin::Override
                    || (origin == VarOrigin::EnvironmentOverride
                        && assigning != VarOrigin::CommandLine))
            {
                return Ok(());
            }
            if origin == VarOrigin::CommandLine && assigning == VarOrigin::File {
                return Ok(());
            }
            if origin == VarOrigin::Automatic {
                error!("overriding automatic variable is not implemented yet");
            }
        }
        *entry = Some(var);
        Ok(())
    }

    /// Remove the binding for `sym`, if the `undefine` outranks what defined it.
    ///
    /// A makefile's `undefine` reaches the environment and its own assignments
    /// and stops there; `override undefine` reaches what the command line and
    /// `override` set as well. An automatic variable is out of reach of both.
    pub fn undefine(
        &mut self,
        names: &impl Interner,
        sym: Symbol,
        is_override: bool,
    ) -> Result<()> {
        let Some(var) = self.peek(sym) else {
            return Ok(());
        };
        let (readonly, origin) = {
            let var = var.read();
            (var.readonly, var.origin())
        };
        if readonly {
            error!(
                "*** cannot undefine readonly variable: {}",
                sym.display(names)
            );
        }
        let outranks = match origin {
            VarOrigin::Automatic => false,
            VarOrigin::EnvironmentOverride | VarOrigin::CommandLine | VarOrigin::Override => {
                is_override
            }
            _ => true,
        };
        if outranks {
            self.replace(sym, None);
        }
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

    /// The binding for `sym`, recording the read in `used_env_vars` if it came
    /// from the environment.
    pub fn lookup(&self, used_env_vars: &mut HashSet<Symbol>, sym: Symbol) -> Option<Var> {
        let ret = self.0.lock().get(&sym).cloned()?;
        match ret.read().origin() {
            VarOrigin::Environment | VarOrigin::EnvironmentOverride => {
                used_env_vars.insert(sym);
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
            let assigning = var.read().origin();
            match orig.read().origin() {
                VarOrigin::Override if assigning != VarOrigin::Override => return Ok(()),
                VarOrigin::EnvironmentOverride
                    if !matches!(assigning, VarOrigin::CommandLine | VarOrigin::Override) =>
                {
                    return Ok(());
                }
                VarOrigin::CommandLine if assigning == VarOrigin::File => return Ok(()),
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
    use crate::symtab::Symtab;

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
                &symtab,
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
        let session = Session::new();
        assert_eq!(
            first
                .peek(sym)
                .unwrap()
                .read()
                .string(&session)
                .unwrap()
                .as_ref(),
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
        let scope = GlobalVars::with_builtins();
        assert!(scope.peek(Symbol::SHELLSTATUS).is_some());
        assert!(scope.peek(Symbol::RECIPEPREFIX).is_some());
        // A scope without builtins: the name is interned all the same but has
        // no binding.
        assert!(GlobalVars::new().peek(Symbol::SHELLSTATUS).is_none());
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
