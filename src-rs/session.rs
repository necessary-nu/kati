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
    path::PathBuf,
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

/// Which of a read's questions the ground answers, rather than the text.
///
/// Every one of these takes a value from outside the Makefile and hands it to
/// the expansion that asked, so a second read over the same text can neither
/// skip it nor be given nothing. `$(abspath)` is deliberately absent: it is
/// spelling, not a question — it never touches the filesystem, and it answers
/// the same on every pass by construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroundQuestion {
    /// `$(shell)`, `!=` and `KATI_shell_no_rerun`.
    Shell,
    /// `$(wildcard)`.
    Wildcard,
    /// `$(realpath)`.
    RealPath,
    /// `$(file < name)`.
    FileRead,
    /// One target or prerequisite word holding `?`, `*` or `[`, matched
    /// against the filesystem where GNU Make's `parse_file_seq` matches it.
    Glob,
    /// Which files one word of an `include` line reaches.
    ///
    /// Here for a different reason than the rest: this is not a value handed to
    /// an expansion, it is what the read IS. GNU Make reads once, so a makefile
    /// that only appears once a staged child has written it was never part of
    /// the read at all, and a repeated read that opened it would be compiling a
    /// makefile the build never had.
    Include,
}

/// One question a read asked the ground, and the answer it was given.
#[derive(Clone, Debug)]
pub struct GroundAnswer {
    pub question: GroundQuestion,
    /// What was asked, already expanded — the command, the pattern, the words.
    /// Kept so a replay can tell that the read it is answering is still the
    /// read it recorded.
    pub asked: Bytes,
    /// The bytes the call put into the expansion that asked for it.
    pub answer: Bytes,
    /// `.SHELLSTATUS`, for the one question that leaves one.
    pub status: Option<i32>,
}

/// The answers one read got from outside itself, in the order it asked for
/// them.
///
/// A front end that compiles a recursive child into its parent's graph reads
/// the parent again once the child's inputs are on the ground, and by then the
/// ground has moved — the staged work is what moved it. GNU Make asks once,
/// before any recipe has run, so its answers are the answers to the ground the
/// build started with. Holding the questions back is not available the way it
/// is for `$(info)`: the expansion that asked has to be handed something. So
/// the first read's answers are kept and handed back.
///
/// KEYED BY POSITION, not by text. The same command written twice in one
/// makefile is two questions and runs twice on the first read; a `$(shell)`
/// inside a `$(foreach)` is one question per iteration. Only the order they
/// were asked in identifies them, and the order is the same because the text
/// is the same and every answer that could have changed it is replayed too.
/// What is recorded beside each answer is the question, and that is a check
/// rather than a key: if the read asks something else where the record says it
/// asked this, the sequence has stopped meaning anything.
///
/// WHEN IT RUNS OUT, or when that check fails, the replay stops for the rest
/// of this read and every remaining question goes to the ground. Resyncing
/// would mean handing back an answer from a call site that is not the one
/// asking. Stopping is exactly what the front end did before it recorded
/// anything, so a read that diverges is no worse off than it was.
#[derive(Default)]
pub struct GroundJournal {
    recorded: Vec<GroundAnswer>,
    replaying: Vec<GroundAnswer>,
    at: usize,
    diverged: bool,
    suspended: bool,
}

impl GroundJournal {
    /// Hand this read the answers an earlier read of the same text was given.
    pub fn replay(&mut self, answers: Vec<GroundAnswer>) {
        self.replaying = answers;
        self.at = 0;
        self.diverged = false;
    }

    /// End the read: hand back what it asked and was told, for the read after
    /// it, and stop replaying.
    ///
    /// Stopping matters as much as handing back. The session outlives the read
    /// when a recipe is expanded as its edge launches, and a `$(shell)` in a
    /// recipe is a command that runs — it is not part of any read, so nothing
    /// left in the journal may answer for it.
    pub fn close_read(&mut self) -> Vec<GroundAnswer> {
        self.replaying = Vec::new();
        self.at = 0;
        self.diverged = false;
        std::mem::take(&mut self.recorded)
    }

    /// Whether the replay stopped short of the end of what it was given.
    pub const fn diverged(&self) -> bool {
        self.diverged
    }

    /// Set while a recipe is expanded, which is not part of any read.
    ///
    /// GNU Make expands a recipe when it runs it, so what the recipe asks is
    /// asked of the ground as it stands then. This front end has one case
    /// where that matters and it is the reason recursive Make can be compiled
    /// at all: a `$(MAKE)` line whose argument is `$(shell cat stamp)` cannot
    /// be read until `stamp` has been staged, so the pass after the staging is
    /// the pass that asks — and it must get the new answer, not the empty one
    /// the pass before it got. Neither answered nor recorded, because a
    /// recorded one would also move every later question out of its place.
    pub fn suspend(&mut self, suspended: bool) {
        self.suspended = suspended;
    }

    /// The answer an earlier read got to this same question, if it is still
    /// the same question in the same place.
    pub(crate) fn answered(
        &mut self,
        question: GroundQuestion,
        asked: &Bytes,
    ) -> Option<GroundAnswer> {
        if self.diverged || self.suspended {
            return None;
        }
        let recorded = self.replaying.get(self.at)?;
        if recorded.question != question || recorded.asked != asked {
            self.diverged = true;
            return None;
        }
        self.at += 1;
        let answered = recorded.clone();
        self.recorded.push(answered.clone());
        Some(answered)
    }

    /// What the ground has just said, for the reads after this one.
    pub(crate) fn record(
        &mut self,
        question: GroundQuestion,
        asked: Bytes,
        answer: Bytes,
        status: Option<i32>,
    ) {
        if self.suspended {
            return;
        }
        self.recorded.push(GroundAnswer {
            question,
            asked,
            answer,
            status,
        });
    }
}

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
    /// Where a non-fatal diagnostic raised against this context is written.
    fn diagnostics(&self) -> &crate::diagnostics::Diagnostics;
}

/// One Make evaluation's state, in one owned value.
// [spec:ronin:req:make.no-ambient-state]
pub struct Session {
    /// The parsed command line. A value, not a read of the process arguments.
    pub flags: Flags,
    /// The directories an `include` is searched for in, in order.
    ///
    /// Not the same list as `flags.include_dirs`, and deliberately so: this is
    /// what `construct_include_path` (read.c) builds out of it — every `-I`
    /// directory tilde-expanded and stat'ed, the ones that are not there left
    /// out and trailing slashes discarded, with the built-in default
    /// directories on the end. `.INCLUDE_DIRS` publishes this list, and the
    /// search reads this list, so the variable is an answer about the search
    /// rather than a second opinion.
    pub include_path: Vec<PathBuf>,
    /// The environment this invocation imports, when its caller is another
    /// compiler session rather than the ambient process.
    ///
    /// `None` preserves the standalone API: evaluation snapshots the process
    /// environment. A semantic submake supplies `Some` so its parent exports
    /// reach Make evaluation without launching another process or mutating the
    /// compiler's own environment.
    pub invocation_environment: Option<Vec<(OsString, OsString)>>,
    /// Byte strings to [`Symbol`] handles and back.
    pub symtab: Symtab,
    /// Make's outermost variable scope, keyed by interned symbol.
    pub globals: GlobalVars,
    /// Command-line values as recipes receive them through Make's exported
    /// environment, retained even when `override undefine` removes the
    /// makefile-scope binding.
    pub(crate) recipe_command_line: GlobalVars,
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
    /// What this read asked the ground, and what an earlier read of the same
    /// text was told.
    pub ground_journal: GroundJournal,
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
    /// Where this session's non-fatal diagnostics are written.
    ///
    /// Shared with every other session of the same compilation, because a
    /// recursive `$(MAKE)` composed into its parent's graph is another session
    /// and what it says belongs to the one invocation that asked. The default
    /// writes each diagnostic to standard error where it is raised, which is
    /// what the fork's own binary does; a front end that collects them replaces
    /// it before the read.
    pub diagnostics: std::sync::Arc<crate::diagnostics::Diagnostics>,
    /// Where this session records what it decided about each recursive
    /// invocation it classified.
    ///
    /// Shared with every other session of the same compilation for the reason
    /// the diagnostics descriptor is: a composed child is another session, and
    /// what it classified belongs to the invocation that asked. The default
    /// records nothing, because a build acts on each classification and has no
    /// use for it afterwards.
    pub census: std::sync::Arc<crate::census::Census>,
    /// Where this unit's Makefiles sit relative to the compilation's root,
    /// for a census that has to say which `Makefile` a line is in.
    ///
    /// A recursive child is read from its own directory, so the name it opens
    /// its Makefile under is the same `Makefile` its parent opened — and a
    /// report naming both `Makefile:44` would be pointing at two files with
    /// one name. Empty for the root, which is already where it says it is.
    pub unit_prefix: Vec<u8>,
    /// `.SUFFIXES` as the whole read left it, in the order it was written, each
    /// entry keeping its leading dot.
    ///
    /// Settled once the last Makefile is closed, because `.SUFFIXES:` clears
    /// the list and a later line adds to what is left. It decides which suffix
    /// rules exist, and it is also what `$*` reads for an explicit rule: GNU
    /// Make's `set_file_variables` walks this list rather than a fixed table,
    /// so a Makefile that rewrote it moves that answer too.
    pub suffixes: Vec<Bytes>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// Supply the bytes read for a named makefile.
    ///
    /// The ordinary cache still parses and records it on first use.  This is
    /// how an embedding frontend implements GNU Make's `-f -` while retaining
    /// `-` as the source name in diagnostics and `MAKEFILE_LIST`.
    pub fn supply_makefile(&mut self, filename: OsString, contents: Vec<u8>) {
        self.makefiles
            .supply(filename, bytes::Bytes::from(contents));
    }

    /// Every makefile this session read, with the bytes it read, for a read
    /// that repeats this one.
    ///
    /// A staging pass re-reads a unit over text that has not moved while the
    /// ground under it has — the staged work is what moved it — so a makefile a
    /// staged child has rewritten, or removed, must still read as the text GNU
    /// Make's one read had. Handing these back to `supply_makefile` on the next
    /// pass is how that is said.
    pub fn read_sources(&self) -> Vec<(OsString, Vec<u8>)> {
        self.makefiles
            .sources()
            .map(|(name, contents)| (name.clone(), contents.to_vec()))
            .collect()
    }

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
            include_path: Vec::new(),
            invocation_environment: None,
            symtab,
            globals: GlobalVars::with_builtins(),
            recipe_command_line: GlobalVars::new(),
            stats: StatsRegistry::new(),
            glob_cache: GlobCache::default(),
            makefiles: MakefileCache::new(),
            find_emulator: OnceLock::new(),
            find_node_count: AtomicUsize::new(0),
            command_results: Vec::new(),
            ground_journal: GroundJournal::default(),
            used_env_vars: HashSet::new(),
            used_undefined_vars: HashSet::new(),
            shell_status: None,
            vpaths: Vec::new(),
            diagnostics: std::sync::Arc::new(crate::diagnostics::Diagnostics::to_stderr()),
            census: std::sync::Arc::new(crate::census::Census::ignored()),
            unit_prefix: Vec::new(),
            suffixes: Vec::new(),
        }
    }

    pub fn intern<T: Into<Bytes> + AsRef<[u8]>>(&mut self, s: T) -> Symbol {
        self.symtab.intern(s)
    }

    /// Record what a `$(shell)` exited with, and publish `.SHELLSTATUS`.
    ///
    /// GNU Make's `func_shell_base` ends with
    /// `define_variable_cname (".SHELLSTATUS", buf, o_override, 0)`, so the
    /// name does not exist at all until a `$(shell)` has run — a makefile is
    /// free to write it before that and loses the precedence contest after —
    /// and every later `$(shell)` redefines it at override origin over
    /// whatever an `override` directive put there. Defining it here rather
    /// than in the builtin catalogue is what says both.
    pub fn record_shell_status(&mut self, status: Option<i32>) -> Result<()> {
        self.shell_status = status;
        self.set_global_var(
            Symbol::SHELLSTATUS,
            crate::var::Variable::new_shell_status_var(),
            true,
            None,
        )
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
        if self.flags.environment_overrides {
            self.globals.note_environment_outranks_the_makefile(sym);
        }
        // Disjoint fields: the scope is written while the interner is read for
        // the readonly diagnostic. They were two locks before, and taking both
        // was the hazard this removes.
        self.globals
            .assign(&self.symtab, sym, var, is_override, readonly)
    }

    /// Remove a global variable, if the `undefine` outranks what defined it.
    pub fn undefine_global_var(&mut self, sym: Symbol, is_override: bool) -> Result<()> {
        if self.flags.environment_overrides {
            self.globals.note_environment_outranks_the_makefile(sym);
        }
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

    /// Glob `pat`, memoising the result until something runs.
    pub fn glob(&self, pat: Bytes) -> GlobResults {
        self.glob_cache.glob(pat)
    }

    /// Note that a command ran, so what the filesystem said before it did is
    /// no longer what it says.
    ///
    /// GNU Make counts commands for exactly this — `dir.c` believes a
    /// directory it read only while `command_count` is unchanged — and every
    /// way a makefile has of changing the filesystem is a command, so calling
    /// this wherever one runs is the whole of the coherence rule.
    pub fn note_command_ran(&self) {
        self.glob_cache.invalidate();
    }

    pub fn clear_glob_cache(&self) {
        self.glob_cache.clear();
    }

    /// The find emulator, built on first use.
    pub fn find_emulator(&self) -> &FindEmulator {
        self.find_emulator.get_or_init(FindEmulator::new)
    }

    /// A parsed makefile, read and parsed on first use, or why it was not.
    pub fn get_makefile(&mut self, filename: &OsStr) -> Result<crate::file::Source> {
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
    fn diagnostics(&self) -> &crate::diagnostics::Diagnostics {
        &self.diagnostics
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
        assert!(fresh.glob_cache.is_empty());
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

    /// A repeated read is answered by position, and the question recorded
    /// beside each answer is the check that the positions still mean anything.
    // [spec:ronin:req:make.semantics+1/test]
    #[test]
    fn a_repeated_read_is_answered_in_order() {
        let mut ev = Evaluator::new(Session::new());
        eval_in(&mut ev, "SHELL := /bin/sh\n").unwrap();
        // The same command written twice runs twice, so the record has two
        // entries for it and they are told apart by where they stand.
        expand(
            &mut ev,
            "$(shell echo one)$(shell echo one)$(shell echo two)",
        )
        .unwrap();
        let first = ev.session.ground_journal.close_read();
        assert_eq!(first.len(), 3);

        ev.session.ground_journal.replay(first.clone());
        assert_eq!(
            expand(&mut ev, "$(shell echo one)").unwrap().as_ref(),
            b"one"
        );
        // Asked out of order, the second question does not match what stands
        // in its place, so the replay stops and the ground answers the rest.
        assert_eq!(
            expand(&mut ev, "$(shell echo three)").unwrap().as_ref(),
            b"three"
        );
        assert!(ev.session.ground_journal.diverged());
        // And it stays stopped: resyncing would hand back an answer belonging
        // to a call site that is not the one asking.
        assert_eq!(
            expand(&mut ev, "$(shell echo one)").unwrap().as_ref(),
            b"one"
        );
        let second = ev.session.ground_journal.close_read();
        assert_eq!(second.len(), 3);

        // Closing the read ends the replay. A recipe expanded as its edge
        // launches uses this same session, and a `$(shell)` there is a command
        // that runs rather than part of any read.
        ev.session.ground_journal.replay(first);
        ev.session.ground_journal.close_read();
        assert_eq!(
            expand(&mut ev, "$(shell echo fresh)").unwrap().as_ref(),
            b"fresh"
        );
        assert!(!ev.session.ground_journal.diverged());
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
