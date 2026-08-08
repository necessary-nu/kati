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

//! The unit of Make evaluation.
//!
//! Everything the front end used to keep in a process global lives here:
//! command-line flags, the symbol interner, Make's global variable scope, the
//! glob, makefile, and find caches, recorded command results, used-variable
//! tracking, the shell status behind `.SHELLSTATUS`, and the statistics
//! registry. Two sessions can be evaluated in one process without either
//! observing the other.
//!
//! See `[spec:ronin:req:make.no-ambient-state]` and
//! `plan/decisions/session-owned-evaluation.md`.

use std::{
    collections::HashSet,
    ffi::{OsStr, OsString},
    sync::{OnceLock, atomic::AtomicUsize},
};

use anyhow::Result;
use bytes::Bytes;

use crate::{
    file_cache::MakefileCache,
    fileutil::{GlobCache, GlobResults},
    find::FindEmulator,
    flags::Flags,
    func::CommandResult,
    stats::StatsRegistry,
    symtab::{Interner, Symbol, Symtab},
    var::{GlobalVars, Var, VarOrigin},
};

/// What a diagnostic, a statistics site, or a flag test needs to be reachable
/// from: an interner to render symbols with, the flags, and the statistics
/// registry.
///
/// [`Session`] is the value that has all three. [`crate::eval::Evaluator`]
/// implements it by delegating to the session it owns, so the diagnostic macros
/// can be given whichever of the two is in scope.
pub trait Context: Interner {
    fn flags(&self) -> &Flags;
    fn stats(&self) -> &StatsRegistry;
}

/// One Make evaluation's state, in one owned value.
// [spec:ronin:req:make.no-ambient-state]
pub struct Session {
    /// The parsed command line. A value, not a read of the process arguments.
    pub flags: Flags,
    /// Byte strings to [`Symbol`] handles and back.
    pub symtab: Symtab,
    /// Make's outermost variable scope, keyed by interned symbol.
    pub globals: GlobalVars,
    /// Named timing and count collection sites.
    pub stats: StatsRegistry,

    /// `$(wildcard)` and `include` results, keyed by pattern.
    ///
    /// Behind a lock because the regeneration check globs from a worker thread
    /// with only a `&Session`; it is session state either way.
    pub glob_cache: GlobCache,
    /// Parsed makefiles, keyed by filename, and the extra file dependencies
    /// `$(KATI_extra_file_deps)` adds.
    pub makefiles: MakefileCache,
    /// The find emulator's directory tree, built on first use because building
    /// it walks the source tree.
    find_emulator: OnceLock<FindEmulator>,
    /// How many directory entries the find emulator has read, for `--kati_stats`.
    pub find_node_count: AtomicUsize,

    /// Everything `$(shell)`, `$(file)`, and the find emulator did, for the
    /// regeneration stamp.
    pub command_results: Vec<CommandResult>,
    /// Environment variables an evaluation read.
    pub used_env_vars: HashSet<Symbol>,
    /// Variables an evaluation read without finding a binding.
    pub used_undefined_vars: HashSet<Symbol>,
    /// The exit status of the last `$(shell)`, which is what `.SHELLSTATUS`
    /// reads.
    pub shell_status: Option<i32>,
    /// Where to look for a prerequisite that is not in the current directory,
    /// in the order the `vpath` directives declared it.
    ///
    /// A list rather than a map because order is the semantics: the first
    /// pattern that matches a name decides which directories are searched, and
    /// a later directive for the same pattern extends it rather than replacing
    /// it.
    pub vpaths: Vec<(crate::strutil::Pattern, Vec<Bytes>)>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// A session with default flags.
    pub fn new() -> Self {
        Self::with_flags(Flags::default())
    }

    /// A session whose flags come from `args`, a whole `argv`.
    pub fn from_args(args: Vec<OsString>) -> Self {
        let mut symtab = Symtab::new();
        let flags = Flags::from_args(args, &mut symtab);
        Self::from_parts(flags, symtab)
    }

    pub fn with_flags(flags: Flags) -> Self {
        Self::from_parts(flags, Symtab::new())
    }

    fn from_parts(flags: Flags, symtab: Symtab) -> Self {
        Self {
            flags,
            symtab,
            globals: GlobalVars::with_builtins(),
            stats: StatsRegistry::new(),
            glob_cache: GlobCache::default(),
            makefiles: MakefileCache::new(),
            find_emulator: OnceLock::new(),
            find_node_count: AtomicUsize::new(0),
            command_results: Vec::new(),
            used_env_vars: HashSet::new(),
            used_undefined_vars: HashSet::new(),
            shell_status: None,
            vpaths: Vec::new(),
        }
    }

    pub fn intern<T: Into<Bytes> + AsRef<[u8]>>(&mut self, s: T) -> Symbol {
        self.symtab.intern(s)
    }

    /// Read a global variable without recording the read.
    pub fn peek_global_var(&self, sym: Symbol) -> Option<Var> {
        self.globals.peek(sym)
    }

    /// Read a global variable, recording it if it came from the environment.
    pub fn get_global_var(&mut self, sym: Symbol) -> Option<Var> {
        let var = self.globals.peek(sym)?;
        match var.read().origin() {
            VarOrigin::Environment | VarOrigin::EnvironmentOverride => {
                self.used_env_vars.insert(sym);
            }
            _ => {}
        }
        Some(var)
    }

    /// Assign to a global variable under Make's precedence rules.
    pub fn set_global_var(
        &mut self,
        sym: Symbol,
        var: Var,
        is_override: bool,
        readonly: Option<&mut bool>,
    ) -> Result<()> {
        // Disjoint fields: the scope is written while the interner is read for
        // the readonly diagnostic. They were two locks before, and taking both
        // was the hazard this removes.
        self.globals
            .assign(&self.symtab, sym, var, is_override, readonly)
    }

    /// Remove a global variable, if the `undefine` outranks what defined it.
    pub fn undefine_global_var(&mut self, sym: Symbol, is_override: bool) -> Result<()> {
        self.globals.undefine(&self.symtab, sym, is_override)
    }

    /// The names of the global variables satisfying `filter`, in symbol order.
    pub fn global_var_names<F: Fn(&Var) -> bool>(&self, filter: F) -> Vec<(Symbol, Bytes)> {
        self.globals
            .matching(filter)
            .into_iter()
            .map(|(sym, _)| (sym, self.symtab.name(sym)))
            .collect()
    }

    /// Record that a variable was read without a binding.
    pub fn note_undefined_var(&mut self, sym: Symbol) {
        self.used_undefined_vars.insert(sym);
    }

    /// Glob `pat`, memoising the result for the rest of the session.
    pub fn glob(&self, pat: Bytes) -> GlobResults {
        crate::fileutil::glob(&self.glob_cache, pat)
    }

    pub fn clear_glob_cache(&self) {
        self.glob_cache.lock().clear();
    }

    /// The find emulator, built on first use.
    pub fn find_emulator(&self) -> &FindEmulator {
        self.find_emulator.get_or_init(FindEmulator::new)
    }

    /// A parsed makefile, read and parsed on first use.
    pub fn get_makefile(
        &mut self,
        filename: &OsStr,
    ) -> Result<Option<std::sync::Arc<crate::file::Makefile>>> {
        crate::file_cache::get_makefile(self, filename)
    }
}

impl Interner for Session {
    fn symtab(&self) -> &Symtab {
        &self.symtab
    }
}

impl Context for Session {
    fn flags(&self) -> &Flags {
        &self.flags
    }
    fn stats(&self) -> &StatsRegistry {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Evaluator;
    use crate::expr::Evaluable;
    use bytes::BytesMut;

    fn eval_in(ev: &mut Evaluator, src: &str) -> Result<Bytes> {
        let stmts = crate::parser::parse_buf(
            &mut ev.session,
            &Bytes::from(src.as_bytes().to_vec()),
            crate::loc::Loc::default(),
        )?;
        let stmts = stmts.lock().clone();
        for stmt in stmts {
            stmt.eval(ev)?;
        }
        Ok(Bytes::new())
    }

    fn expand(ev: &mut Evaluator, src: &str) -> Result<Bytes> {
        let expr = crate::expr::parse_expr(
            &mut ev.session,
            &mut crate::loc::Loc::default(),
            Bytes::from(src.as_bytes().to_vec()),
            crate::expr::ParseExprOpt::Normal,
        )?;
        let mut out = BytesMut::new();
        expr.eval(ev, &mut out)?;
        Ok(out.freeze())
    }

    /// Two sessions evaluating different content in one process must not see
    /// each other's symbols, variables or caches.
    // [spec:ronin:req:make.no-ambient-state/test]
    #[test]
    fn test_two_sessions_do_not_observe_each_other() {
        let mut a = Evaluator::new(Session::new());
        let mut b = Evaluator::new(Session::new());

        eval_in(&mut a, "ONLY_IN_A := first\nSHARED := from-a\n").unwrap();
        eval_in(&mut b, "ONLY_IN_B := second\nSHARED := from-b\n").unwrap();

        // Variables: each session sees its own binding and not the other's.
        assert_eq!(expand(&mut a, "$(SHARED)").unwrap().as_ref(), b"from-a");
        assert_eq!(expand(&mut b, "$(SHARED)").unwrap().as_ref(), b"from-b");
        assert_eq!(expand(&mut a, "$(ONLY_IN_A)").unwrap().as_ref(), b"first");
        assert_eq!(expand(&mut a, "$(ONLY_IN_B)").unwrap().as_ref(), b"");
        assert_eq!(expand(&mut b, "$(ONLY_IN_B)").unwrap().as_ref(), b"second");
        assert_eq!(expand(&mut b, "$(ONLY_IN_A)").unwrap().as_ref(), b"");

        // Symbols: a name interned in one session is absent from the other's
        // interner, and each renders its own handles correctly.
        let a_only = a.session.symtab.intern("ONLY_IN_A");
        assert_eq!(
            a_only.display(&a.session).to_string(),
            "ONLY_IN_A",
            "a session must render its own symbol"
        );
        let mut fresh = Session::new();
        assert!(
            fresh.symtab.count() < a.session.symtab.count(),
            "a fresh interner must not carry another session's names"
        );
        assert_eq!(fresh.intern("ONLY_IN_A").index(), a_only.index());

        // Caches and observation state start empty in a new session, whatever
        // the others have accumulated.
        assert!(fresh.used_env_vars.is_empty());
        assert!(fresh.used_undefined_vars.is_empty());
        assert!(fresh.command_results.is_empty());
        assert!(fresh.glob_cache.lock().is_empty());
        assert!(fresh.shell_status.is_none());
        assert!(fresh.peek_global_var(a_only).is_none());
    }

    /// The undefined-variable set is per session: reading an unbound name in
    /// one must not make the other think it was read.
    // [spec:ronin:req:make.no-ambient-state/test]
    #[test]
    fn test_used_variable_tracking_is_per_session() {
        let mut a = Evaluator::new(Session::new());
        let b = Evaluator::new(Session::new());
        expand(&mut a, "$(NEVER_DEFINED_ANYWHERE)").unwrap();
        let sym = a.session.symtab.intern("NEVER_DEFINED_ANYWHERE");
        assert!(a.session.used_undefined_vars.contains(&sym));
        assert!(b.session.used_undefined_vars.is_empty());
    }

    /// A `$(shell)` in one session sets only that session's `.SHELLSTATUS` and
    /// records only that session's command results.
    // [spec:ronin:req:make.no-ambient-state/test]
    #[test]
    fn test_shell_state_is_per_session() {
        let mut a = Evaluator::new(Session::new());
        let mut b = Evaluator::new(Session::new());
        eval_in(&mut a, "SHELL := /bin/sh\n").unwrap();
        eval_in(&mut b, "SHELL := /bin/sh\n").unwrap();
        expand(&mut a, "$(shell exit 3)").unwrap();
        assert_eq!(a.session.shell_status, Some(3));
        assert_eq!(expand(&mut a, "$(.SHELLSTATUS)").unwrap().as_ref(), b"3");
        assert_eq!(b.session.shell_status, None);
        assert_eq!(expand(&mut b, "$(.SHELLSTATUS)").unwrap().as_ref(), b"");
        assert!(!a.session.command_results.is_empty());
        assert!(b.session.command_results.is_empty());
    }
}
