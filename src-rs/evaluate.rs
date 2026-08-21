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

//! Makefile to dependency graph.
//!
//! Everything between a [`Session`] and the [`DepNode`](crate::dep::DepNode)s a
//! Makefile describes: the built-in variables and suffix rules, the environment,
//! the command line's own assignments, the makefile itself, and then dependency
//! analysis over what all of that defined.
//!
//! This is the half of a kati run that has nothing to do with where the graph
//! goes afterwards. `rkati` follows it with [`crate::ninja`] or
//! [`crate::exec`]; a front end embedding kati follows it with its own
//! [`BuildSink`](crate::build_sink::BuildSink).

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;

use crate::dep::{NamedDepNode, RegenerationRoot, make_dep};
use crate::eval::{Evaluator, FrameType};
use crate::expr::{ParseExprOpt, Value, parse_expr};
use crate::file::Source;
use crate::loc::Loc;
use crate::session::Session;
use crate::stmt::Stmt;
use crate::symtab::{Symbol, join_symbols};
use crate::timeutil::ScopedTimeReporter;
use crate::var::{VarExport, VarOrigin, Variable};
use crate::{error_loc, log, warn_loc};

/// The GNU Make features this evaluator actually has, in `.FEATURES`' spelling.
///
/// Each one was established differentially against GNU Make 4.4.1 rather than
/// copied from its list, and the list is short because most of what GNU Make
/// reports here is genuinely absent: `second-expansion`, `oneshell` and
/// `notparallel` are warned about as unsupported at `dep.rs`, `undefine` and
/// `load` are not directives the parser knows, `shell-export` does not reach
/// the shell, and `grouped-target` parses but runs the recipe once per target
/// rather than once for the group.
///
/// `archives` is absent from *this* list although the evaluator now has every
/// front-end half of it — the `lib.a(member.o)` name shape, the built-in
/// `(%): %` rule, `$%`, `ar_glob` over the archive's index, the `.X.a`
/// conversion's second pattern rule, and `lib.a(a.o b.o)` naming two members.
/// It is absent because the feature is not the evaluator's alone to claim: a
/// member's timestamp comes out of the archive's index rather than off a file
/// of that name, and reading it is the business of whoever stats the graph.
/// The evaluator emitting these names to a front end that stats them as
/// filenames would get a working parse and a wrong build.
///
/// So `archives` arrives through [`Flags::extra_features`], declared by the
/// front end that has both halves — Ronin's Make mode, which sets
/// `archive_members` on its disk interface. `jobserver` is here for the same
/// reason and by the same rule: build-side features belong to whoever runs the
/// recipes.
///
/// What is not implemented is `-t` on a member, and it is worth being exact
/// about why, because it is not an archive gap. `-t` is not implemented on
/// anything: it is accepted and ignored, per the `Accept without emulation`
/// disposition in docs/make-compiler-boundary-audit.md. GNU Make's own
/// `features/archives` script — the measure for this feature — never invokes
/// it.
pub const EVALUATOR_FEATURES: &[&str] =
    &["target-specific", "order-only", "else-if", "shortest-stem"];

/// A Makefile that has been read, expanded, and reduced to a graph.
pub struct Evaluated {
    /// The evaluator the graph was produced by, which still holds the session,
    /// the exported variables, and everything a stamp or a command needs.
    pub ev: Evaluator,
    /// The roots of the dependency graph, in the order the targets asked for
    /// them.
    pub nodes: Vec<NamedDepNode>,
    /// Missing included Makefiles that have rules in this provisional graph.
    ///
    /// They are graph roots like the goals, and a frontend that emits the graph
    /// has to emit them too, but they are not goals: a build that was not asked
    /// for one does not produce it. An embedding frontend may build them and
    /// evaluate the Makefile again, just as Ninja rebuilds and reloads its own
    /// manifest. Missing includes with no rule are not here at all, because
    /// GNU Make forgets an optional one it cannot remake and dies on a required
    /// one.
    pub regeneration_nodes: Vec<RegenerationRoot>,
    /// A required Makefile the read could not open and no rule can make.
    ///
    /// GNU Make refuses over one of these from inside the update that brings
    /// the makefiles up to date, so the makefiles it reached before this one
    /// are remade first and the run ends afterwards. The refusal travels with
    /// the plan rather than in place of it, so a frontend can do that work in
    /// between — and the located complaint about the file travels with the
    /// refusal, because GNU Make prints that from the same place.
    pub refusals: Vec<crate::dep::Refusal>,
}

/// The Makefile kati reads before the real one.
///
/// Half of it is GNU Make's built-in suffix rules, and half is kati telling the
/// Makefile about the invocation: what `$(MAKE)` re-runs, what goals were asked
/// for, and where the run started. The tool defaults `-R` withholds are not
/// here — they are a catalogue with origins of their own, installed by
/// [`crate::builtins::install_default_variables`].
fn read_bootstrap_makefile(session: &mut Session) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    let mut bootstrap = BytesMut::new();
    bootstrap.put_slice(b"KATI?=ckati\n");
    // Three names that used to be lines here are not any more, because no
    // makefile line says what GNU Make's own call for each of them says:
    // `SHELL` (see [`stand_the_shell`]), `.FEATURES` and `MAKE` (see
    // [`install_worked_out_variables`]). `KATI` stays, because it is kati's own
    // name and a `?=` is exactly what it means: a default the makefile may
    // prefer to set.
    // TODO: Add more builtin vars.

    // GNU Make's `set_default_suffixes`, which is the whole of the built-in
    // rule catalogue that has to be in scope while a Makefile is read: the
    // rules themselves are derived from this list once the read is over, so
    // that a Makefile's `.SUFFIXES:` decides which of them exist. The manual's
    // catalogue of rules disagrees with `src/default.c`, and `default.c` is the
    // one that runs — see [`crate::builtin_rules`].
    if !session.flags.no_builtin_rules {
        bootstrap.put_slice(b".SUFFIXES: ");
        bootstrap.put_slice(crate::builtin_rules::default_suffix_list().as_bytes());
        bootstrap.put_u8(b'\n');
    }
    let filename = session.intern("*bootstrap*");
    crate::parser::parse_buf(session, &bootstrap.freeze(), Loc { filename, line: 0 })
}

/// Define the names GNU Make works out for itself, at the rank and the flavour
/// its own `define_variable_cname` calls give them.
///
/// These are not bootstrap makefile lines and cannot be, because no makefile
/// line says what those calls say. All but one are SIMPLE — the last argument
/// to `define_variable_cname` is 0, and `MAKE` is the one given a 1 — and
/// Make's syntax has no spelling for "at this rank, and only where nothing
/// outranks it, and promoting the environment's binding if `-e` lifted it":
/// `?=` declines a bound name without promoting anything, and `:=` claims the
/// name whatever holds it, at file rank. A Makefile sees all three differences
/// through `$(flavor)`, `$(value)` and `$(origin)`, and for these names each is
/// branch-worthy: `$(value MAKE_VERSION)` is the version when the binding is
/// simple and the same text only by coincidence when it is recursive.
///
/// # Errors
///
/// Returns the working directory the process could not read, which is the same
/// failure GNU Make dies on before it reads anything.
fn install_worked_out_variables(ev: &mut Evaluator, targets: &[Symbol]) -> Result<()> {
    // `define_variable_cname ("MAKE_VERSION", buf, o_default, 0)` in
    // `define_automatic_variables`. The one place a GNU Make version is claimed,
    // because Makefiles branch on it: it names the version this front end is
    // measured against rather than the one the vendored Go harness pinned, or a
    // Makefile testing `$(MAKE_VERSION)` for a feature takes a branch neither
    // tool would have taken.
    claim_at_default(
        &mut ev.session,
        "MAKE_VERSION",
        Bytes::from_static(b"4.4.1"),
    );
    // `define_variable_cname ("MAKECMDGOALS", value, o_default, 0)`, reached
    // from inside the branch of `handle_non_switch_argument` (main.c) that
    // enters a goal target — so an invocation that named no goal defines no
    // such name, and `ifeq ($(origin MAKECMDGOALS),undefined)` is how a
    // Makefile asks whether it was invoked bare. (`ifdef` cannot ask: it tests
    // for a non-empty value, which an absent name and an empty one both fail.)
    if !targets.is_empty() {
        let goals = join_symbols(&ev.session, targets, b" ");
        claim_at_default(&mut ev.session, "MAKECMDGOALS", goals);
    }
    // `define_variable_cname ("MAKE_HOST", make_host, o_default, 0)`, beside
    // MAKE_VERSION in the same function. See [`make_host`] for what Ronin
    // answers there and why, since GNU's value is baked by its configure run
    // and there is nothing to copy.
    claim_at_default(&mut ev.session, "MAKE_HOST", Bytes::from(make_host()));
    // `define_variable_cname (".LOADED", "", o_default, 0)` in `main`, before
    // anything is read and whether or not anything is ever loaded. It is empty
    // here for good rather than for now: `load` is not a directive this parser
    // knows, so no object can join the list. GNU's is empty on any makefile
    // that loads nothing, which is every makefile in the corpus.
    //
    // main.c:1436, which is ahead of `decode_switches` — so `-e` is not yet in
    // force when this one is written and an inherited `.LOADED` keeps its plain
    // `environment` rank.
    claim_before_the_switches(&mut ev.session, ".LOADED", Bytes::new());
    // What a Makefile is allowed to assume, and no more. Claiming a feature
    // that is not there is worse than claiming none: a Makefile branches on
    // this to decide whether it may use a construct, and GNU Make's test suite
    // skips a case it names. An honest short list makes a build take the
    // portable path; a generous one makes it take a path that then misbehaves.
    //
    // `define_variable_cname (".FEATURES", features, o_default, 0)` at
    // main.c:1475, which is ahead of `decode_switches` as well — and at DEFAULT
    // rank, so an inherited `.FEATURES` outranks it and keeps its value, which
    // is what the bootstrap's `:=` line could not say: that assignment stood at
    // file rank and took the name from the environment.
    let features = EVALUATOR_FEATURES
        .iter()
        .map(|feature| (*feature).to_owned())
        .chain(ev.session.flags.extra_features.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ");
    claim_before_the_switches(&mut ev.session, ".FEATURES", Bytes::from(features));
    // `define_variable_cname ("MAKE_COMMAND", argv[0], o_default, 0)` in
    // `main`, immediately before `MAKE` is defined as the REFERENCE
    // `$(MAKE_COMMAND)` — which is why a makefile replacing MAKE_COMMAND
    // changes what `$(MAKE)` runs, and why the `MAKE` claim below names the
    // reference rather than the path.
    let command = Bytes::from(
        ev.session
            .flags
            .subkati_args
            .join(OsStr::new(" "))
            .as_bytes()
            .to_vec(),
    );
    claim_at_default(&mut ev.session, "MAKE_COMMAND", command);
    // `define_variable_cname ("MAKE", "$(MAKE_COMMAND)", o_default, 1)`, the
    // one name here whose trailing flag is a 1: `MAKE` is RECURSIVE, and that
    // is what makes it a reference rather than a copy. It was a bootstrap
    // `MAKE?=` line, and a `?=` is the one assignment GNU Make declines
    // without promoting — `do_variable_definition` returns on a name already
    // bound before it reaches `define_variable_in_set` — so an inherited `MAKE`
    // read `environment` under `-e` where GNU says `environment override`.
    let make = if ev.session.flags.generate_ninja {
        // Generating a build file rather than running one: there is no
        // `MAKE_COMMAND` a child could re-run, so `$(MAKE)` names the tool the
        // generated file will be built with, at this run's job budget.
        Bytes::from(format!("make -j{}", ev.session.flags.num_jobs.max(1)))
    } else {
        Bytes::from_static(b"$(MAKE_COMMAND)")
    };
    claim_recursive_at_default(&mut ev.session, "MAKE", make)?;
    // `define_variable_cname ("CURDIR", current_directory, o_file, 0)` in
    // `main`, at FILE rank and not default. The distinction is the whole point
    // of the call: a Makefile's own `CURDIR = x` is a peer replacing it rather
    // than an override standing over it, and `$(origin CURDIR)` — which is what
    // a Makefile asks before deciding whether to trust the answer — says so.
    // Going through the ladder rather than defining outright is what keeps an
    // `-e` environment and a command-line write above it, as they are there.
    let curdir = Bytes::from(std::env::current_dir()?.as_os_str().as_bytes().to_vec());
    let sym = ev.session.intern("CURDIR");
    let var = Variable::with_simple_string(curdir, VarOrigin::File, None, None);
    ev.session.set_global_var(sym, var, false, None)
}

/// The host this Make is running on, in the triple shape `MAKE_HOST` carries.
///
/// GNU Make answers with the triple its own configure run recorded —
/// `x86_64-pc-linux-gnu` on the oracle — so there is no value to copy, only a
/// decision to take. A makefile reads this name to find out what platform it is
/// on: `$(findstring mingw32,...)`, `cygwin`, `darwin`, `linux`. So the fields
/// that have to be true are the architecture and the system, and the decision
/// recorded here is that they name the host the run is ON rather than the host
/// the binary was built on. Nothing else would be true of the run, and a
/// makefile branching on it is asking about the run.
///
/// The vendor field is autoconf's own guess and nothing branches on it. It is
/// spelled the way `config.guess` spells it — `apple` on Apple systems, `pc`
/// for the x86 family elsewhere, `unknown` otherwise — so that a makefile
/// pattern written against a real triple still matches one from here.
fn make_host() -> String {
    let arch = std::env::consts::ARCH;
    let vendor = if cfg!(target_vendor = "apple") {
        "apple"
    } else if matches!(arch, "x86" | "x86_64") {
        "pc"
    } else {
        "unknown"
    };
    // autoconf's name for each system is Rust's, except for Apple's.
    let system = match std::env::consts::OS {
        "macos" | "ios" => "darwin",
        other => other,
    };
    // The C library joins the triple only where more than one is usual.
    let abi = if system == "linux" {
        if cfg!(target_env = "musl") {
            "-musl"
        } else {
            "-gnu"
        }
    } else {
        ""
    };
    format!("{arch}-{vendor}-{system}{abi}")
}

/// The directories an `include` falls back to when no `-I` named one, and no
/// makefile ever asks for.
///
/// `default_include_directories` (read.c:103) is the configure-time
/// `INCLUDEDIR` followed by these three, and `construct_include_path` keeps
/// whichever of them are on disk — which is why the oracle answers with two of
/// them on this host.
///
/// THE DECISION, and it is one rather than a value to copy, the way `MAKE_HOST`
/// was. GNU Make's `INCLUDEDIR` is `$(includedir)` from its own `configure`
/// run: the place that installation puts headers. Ronin has no configure step
/// and installs no include tree of its own, so there is nothing honest to
/// prepend — a path named after this program would name a directory nothing
/// ever puts a makefile fragment in. What is left is the convention, which is
/// exactly the list GNU Make ships when it is built without an `INCLUDEDIR`,
/// and that is what the oracle is: its `-DINCLUDEDIR` is commented out of the
/// build, so it answers `/usr/local/include /usr/include` for the same reason
/// this does rather than by being matched.
const DEFAULT_INCLUDE_DIRECTORIES: &[&str] =
    &["/usr/gnu/include", "/usr/local/include", "/usr/include"];

/// The include search path, in the order and the spelling
/// `construct_include_path` (read.c) builds it.
///
/// Every `-I` directory is tilde-expanded, stat'ed, and kept only if it is
/// there, with trailing slashes discarded — `-I nosuchdir` leaves the list
/// unchanged and `-I inc/` joins it as `inc`. Then the built-in default
/// directories go on the END, under the same test, so a `-I` always wins over
/// them; unless an `-I -` turned them off.
///
/// `-` IS AN ENTRY IN THE LIST rather than a state the switch table remembers,
/// which is what makes its position readable here: it empties the path built so
/// far and turns the built-in directories off for good, and a `-I` after it
/// starts the list again with the defaults still off. The distinction is
/// observable because the switch table de-duplicates what it stores — a second
/// `-I -` is dropped as a duplicate and so resets nothing, and a `-I inc`
/// repeated across a `-I -` is dropped too and so does not come back.
///
/// Nothing is de-duplicated HERE: `make -I /usr/include` names that directory
/// twice, once from the switch and once from the default path, and GNU Make
/// lists it twice.
pub(crate) fn construct_include_path(session: &mut Session) {
    session.include_path = search_path(session);
    // The same list, published. It is one list rather than two — the search
    // reads it and `.INCLUDE_DIRS` publishes it — so the variable is an answer
    // about the search rather than a second opinion about it. At DEFAULT rank,
    // as `do_variable_definition (... o_default, f_simple)` puts it, so a
    // makefile that assigned the name itself keeps its own value through every
    // rebuild of the path.
    let dirs = include_directories(session);
    claim_at_default(session, ".INCLUDE_DIRS", dirs);
}

/// The search path itself, without publishing it.
fn search_path(session: &Session) -> Vec<PathBuf> {
    let home = own_home(session);
    let mut path = Vec::new();
    let mut disable = false;
    for dir in &session.flags.include_dirs {
        if dir.as_os_str().as_bytes() == b"-" {
            disable = true;
            path.clear();
            continue;
        }
        push_directory(&mut path, tilde_expand(home.as_deref(), dir));
    }
    if !disable {
        for dir in DEFAULT_INCLUDE_DIRECTORIES {
            push_directory(&mut path, PathBuf::from(dir));
        }
    }
    path
}

/// Keep `dir` if it is a directory, in the spelling GNU Make records: trailing
/// slashes discarded, and nothing else touched.
fn push_directory(path: &mut Vec<PathBuf>, dir: PathBuf) {
    if !dir.is_dir() {
        return;
    }
    let mut bytes = dir.as_os_str().as_bytes();
    while bytes.len() > 1 && bytes.ends_with(b"/") {
        bytes = &bytes[..bytes.len() - 1];
    }
    path.push(PathBuf::from(OsStr::from_bytes(bytes).to_owned()));
}

/// `~` as GNU Make's `tilde_expand` (read.c) expands it.
///
/// GNU Make reaches this function from two places and so does this one: the
/// switch table canonicalises every file name a switch gave through
/// `expand_command_line_file`, and `construct_include_path` expands a search
/// directory again before it stats it. Both are here rather than one being
/// copied, because a second spelling of this rule is a second answer.
///
/// `~` and `~/...` become `home` and the rest of the path. Anything else — and
/// a `~` with no home behind it — is left exactly as it stands, which then
/// fails the directory test and drops out of the search path, so it is not an
/// error but an entry that was never there.
///
/// GNU reads `$(HOME)` as a variable and falls back to `getenv`; here it is the
/// environment either way, because both callers run before any makefile could
/// have written the name — measured, with a makefile assigning `HOME` that GNU
/// Make also ignores.
///
/// THE ONE FORM THAT IS NOT HERE is `~user`. GNU Make answers it out of the
/// passwd database; this does not, and leaves the word alone — which is what
/// GNU Make itself does for a user IT cannot resolve, so the word fails the
/// directory test and drops out of the search path rather than erroring.
///
/// It is unimplemented rather than unimplementable. An earlier note here read
/// that `getpwnam` on a miss reaches `libnss_systemd`, which glibc has to
/// `dlopen`, and that the resulting segfault made `~user` impossible under
/// `+crt-static`. The crash is real on a static-glibc build and is glibc's own
/// documented NSS limitation; static glibc is not a configuration this ships
/// in. The shipped static link is musl, whose `getpwnam` reads the files
/// directly and cannot fail this way. So the recorded blocker was an artifact
/// of one development host, and whoever wants `~user` is free to add it.
#[must_use]
pub fn tilde_expand(home: Option<&[u8]>, dir: &Path) -> PathBuf {
    let bytes = dir.as_os_str().as_bytes();
    let Some(rest) = bytes.strip_prefix(b"~") else {
        return dir.to_owned();
    };
    if !rest.is_empty() && !rest.starts_with(b"/") {
        return dir.to_owned();
    }
    let Some(home) = home.filter(|home| !home.is_empty()) else {
        return dir.to_owned();
    };
    let mut expanded = home.to_vec();
    expanded.extend_from_slice(rest);
    PathBuf::from(OsString::from_vec(expanded))
}

/// `HOME` as this invocation was given it.
fn own_home(session: &Session) -> Option<Vec<u8>> {
    match &session.invocation_environment {
        Some(environment) => environment
            .iter()
            .rev()
            .find(|(name, _)| name.as_bytes() == b"HOME")
            .map(|(_, value)| value.as_bytes().to_vec()),
        None => std::env::var_os("HOME").map(std::ffi::OsString::into_vec),
    }
}

/// The search path as `.INCLUDE_DIRS` spells it: the directories, space
/// separated.
fn include_directories(session: &Session) -> Bytes {
    let mut list = BytesMut::new();
    for dir in &session.include_path {
        if !list.is_empty() {
            list.put_u8(b' ');
        }
        list.put_slice(dir.as_os_str().as_bytes());
    }
    list.freeze()
}

/// Stand `SHELL` where GNU Make stands it, which is not where the environment
/// put it.
///
/// `define_automatic_variables` (variable.c) offers the built-in shell at
/// default rank and then refuses to let the environment's answer survive:
///
/// ```text
/// v = define_variable_cname ("SHELL", default_shell, o_default, 0);
/// /* Don't let SHELL come from the environment.  */
/// if (*v->value == '\0' || v->origin == o_env || v->origin == o_env_override)
///   { v->origin = o_file; v->value = xstrdup (default_shell); }
/// ```
///
/// Two different pairs of answers come out of that one call site, and a
/// makefile can see both. With no `SHELL` in the environment the offer lands
/// and the name stands `default` and SIMPLE. With one, the offer is declined
/// and the block rewrites the rank and the text while leaving the import's
/// recursive flavour alone, so the name stands `file` and RECURSIVE — at a rank
/// a makefile's own `SHELL =` is a peer of and may replace, and one that `-e`
/// cannot lift, because the rank `-e` promotes an environment binding to is a
/// rank this block then overwrites.
///
/// The empty test is not about the environment at all. A binding of any rank
/// whose text is empty — `make SHELL=` — stands back at the built-in shell,
/// which is what keeps a recipe running when one arrives.
///
/// Called where GNU Make calls it: after the command line has been read, so a
/// command-line `SHELL=` is the binding this sees, and before the first
/// makefile, so a makefile's own assignment is a peer replacing it. What sits
/// between those two points in GNU Make sits between them here too — an
/// expansion during the command line reads the environment's shell in both
/// tools, because neither has stood this one yet.
fn stand_the_shell(session: &mut Session) {
    let shell = Bytes::from_static(crate::simple_command::DEFAULT_SHELL);
    claim_at_default(session, "SHELL", shell.clone());
    let sym = session.intern("SHELL");
    let Some(var) = session.peek_global_var(sym) else {
        return;
    };
    let came_from_the_environment = {
        let held = var.read();
        held.string(session).is_ok_and(|text| text.is_empty())
            || matches!(
                held.origin(),
                VarOrigin::Environment | VarOrigin::EnvironmentOverride
            )
    };
    if came_from_the_environment {
        var.write().restate_at(VarOrigin::File, shell);
    }
}

/// Bind `name` to a simple value at default rank, leaving whatever outranks a
/// default exactly as it stands.
///
/// [`crate::builtins::claimable`] is the same test GNU Make's
/// `define_variable_cname` applies with `o_default`: another default may be
/// replaced, and an environment, command-line, `override` or Makefile binding
/// may not.
fn claim_at_default(session: &mut Session, name: &str, value: Bytes) {
    let Some(sym) = crate::builtins::claimable(session, name) else {
        return;
    };
    session.globals.define(
        sym,
        Variable::with_simple_string(value, VarOrigin::Default, None, None),
    );
}

/// The same, for a name GNU Make defines before it has read its switches, where
/// `-e` is not yet in force and the environment's binding is not promoted.
///
/// See [`crate::builtins::claimable_before_the_switches`] for which names those
/// are and why the distinction is a fact about the write rather than the name.
fn claim_before_the_switches(session: &mut Session, name: &str, value: Bytes) {
    let Some(sym) = crate::builtins::claimable_before_the_switches(session, name) else {
        return;
    };
    session.globals.define(
        sym,
        Variable::with_simple_string(value, VarOrigin::Default, None, None),
    );
}

/// Bind `name` to a RECURSIVE expression at default rank, on the same terms.
///
/// The trailing argument to `define_variable_cname` is the recursive flag, and
/// `MAKE` is the one name here that is given a 1. A Makefile can see the
/// difference through `$(flavor)` and `$(value)`, and for this name it is the
/// whole point: `$(value MAKE)` reads back `$(MAKE_COMMAND)`, so a Makefile
/// that replaces `MAKE_COMMAND` changes the program `$(MAKE)` runs.
///
/// # Errors
///
/// Returns a parse failure for the expression, which is a defect here.
fn claim_recursive_at_default(session: &mut Session, name: &str, text: Bytes) -> Result<()> {
    let Some(sym) = crate::builtins::claimable(session, name) else {
        return Ok(());
    };
    let mut loc = Loc::default();
    let parsed = parse_expr(session, &mut loc, text.clone(), ParseExprOpt::Normal)?;
    session.globals.define(
        sym,
        Variable::new_recursive(parsed, VarOrigin::Default, None, None, text),
    );
    Ok(())
}

/// Read one Makefile the command line named, into the session already open.
///
/// A Makefile that is not there is not the end of the read: GNU Make says so
/// where it failed to open the file, goes on to the ones after it, and only
/// then treats the missing name as a target it must reach — which is a rule a
/// later Makefile is still allowed to supply.
fn read_named_makefile(ev: &mut Evaluator, makefile: &OsStr) -> Result<()> {
    let name = Bytes::from(makefile.as_bytes().to_vec());
    let _file_frame = ev.enter(FrameType::Parse, name.clone(), Loc::default());
    let mk = match ev.session.get_makefile(makefile)? {
        Source::Read(mk) => mk,
        // A file that is not there and a file that would not open are one
        // answer here, and GNU Make reports both from the same line: it is
        // `perror_with_name ("", *makefiles)` after `eval_makefile` returns
        // (read.c:219), which quotes whatever errno the open left. So the
        // complaint is made here, under Make's own name because no `include`
        // line asked for the file, and the name still goes on to the update as
        // a target it must reach — which a later Makefile may yet supply a rule
        // for.
        source @ (Source::Absent | Source::Unopened(_)) => {
            let reason = match &source {
                Source::Unopened(err) => crate::strerror(err),
                _ => crate::strerror(&std::io::Error::from_raw_os_error(libc::ENOENT)),
            };
            warn_loc!(ev, None, "{}: {reason}", makefile.to_string_lossy());
            ev.note_unread_include(name, true, None, &reason);
            return Ok(());
        }
        // Opened and then unreadable, or Make itself out of descriptors: GNU
        // Make defers neither. `readline` finds `ferror` and calls
        // `pfatal_with_name` (read.c:2744), and the three exhaustion errnos are
        // fatal where the open happened (read.c:347). No `include` asked for
        // this one, so there is no line to point at; the diagnostic still has to
        // say which file and why.
        Source::Unreadable(err) | Source::Exhausted(err) => error_loc!(
            ev,
            None,
            "*** {}: {}",
            makefile.to_string_lossy(),
            crate::strerror(&err)
        ),
    };
    ev.note_read_makefile(name.clone(), true);
    ev.note_makefile_list(name)?;
    let stmts = mk.stmts.lock().clone();
    for stmt in stmts {
        log!("{stmt:?}");
        stmt.eval(ev)?;
    }
    Ok(())
}

/// Seed the evaluator with `MAKEFILE_LIST` and the process environment.
fn read_invocation_state(ev: &mut Evaluator) -> Result<()> {
    // Empty, and grown a name at a time as each Makefile opens. GNU Make binds
    // it before it reads anything so that a Makefile asking `$(origin
    // MAKEFILE_LIST)` is told `file`, and so that the first name to arrive has
    // somewhere to be appended.
    let frame = ev.current_frame();
    let loc = ev.loc.clone();
    let makefile_list_sym = ev.session.intern("MAKEFILE_LIST");
    ev.session.set_global_var(
        makefile_list_sym,
        Variable::with_simple_string(Bytes::new(), VarOrigin::File, Some(frame), loc),
        false,
        None,
    )?;
    // GNU Make reads the environment before it decodes the switches, so every
    // variable it finds there is recorded as `environment` and none of them can
    // have been affected by a `-e` it has not seen yet. `-e` is a question of
    // precedence, and the origin says so only once something tries to redefine
    // the name and is refused — see `Session::set_global_var`.
    let origin = VarOrigin::Environment;
    let environment = ev
        .session
        .invocation_environment
        .clone()
        .unwrap_or_else(|| std::env::vars_os().collect());
    for (k, v) in environment {
        let v = Bytes::from(v.as_bytes().to_vec());
        let frame = ev.current_frame();
        let sym = ev.session.intern(k.as_bytes().to_vec());
        // Three of these names are ones GNU Make writes into the environment
        // and then defines a second time, once the environment is in scope, at
        // the environment's OWN rank: `define_variable_cname (MAKELEVEL_NAME,
        // buf, o_env, 0)` in `define_automatic_variables`, writing back the
        // depth it parsed with the recursive flag off; `define_variable_cname
        // ("MFLAGS", flagstring, o_env, 1)` in `define_makeflags`, writing back
        // the switches spelled as a command line; and `define_variable_cname
        // (GNUMAKEFLAGS_NAME, "", o_env, 0)` in `main`, emptying the stream
        // whose switches have just been folded into `MAKEFLAGS`.
        //
        // That second define is why these two read `environment override` under
        // `-e` while an ordinary imported name still reads `environment`.
        // `define_variable_in_set` lifts an INCOMING `o_env` define to
        // `o_env_override` when `-e` is in force — and lifts the binding
        // already in the slot with it, so the two ranks are equal and the write
        // lands. Only a name Make defines over again is ever incoming. Saying
        // that here rather than defining twice is the same thing said once.
        let defined_over_again = matches!(k.as_bytes(), b"MAKELEVEL" | b"MFLAGS" | b"GNUMAKEFLAGS");
        let origin = if defined_over_again && ev.session.flags.environment_overrides {
            VarOrigin::EnvironmentOverride
        } else {
            origin
        };
        // The environment's own entry is RECURSIVE, and that is the whole of
        // what a value holding a `$` means here: `define_variable_in_set (name,
        // len, value, o_env, 1)` in `main`, the trailing 1 being the recursive
        // flag. So an environment value is makefile text, parsed here and
        // expanded at every reference like any `NAME = text` — which is where
        // `$(words a b c)` becomes `3`, `$$` halves to one `$`, and a name the
        // makefile defines is reachable from a value the invocation supplied.
        //
        // Parsing is not expanding. The expression is built once and evaluated
        // only when something reads the name, so an environment value carrying
        // `$(shell)` runs its command per reference and not at all if nothing
        // refers to it, and one whose call is left unterminated is a held
        // `Unreadable` that raises where it is read rather than at startup.
        // GNU Make is lazy for the same reason and by the same construction.
        //
        // Two names are the exception, and they are GNU Make's: the two of the
        // three above whose second define carries a recursive flag of 0.
        // `MAKELEVEL` is written back from the depth `main` parsed, so the one
        // environment name whose value is a number stays simple even when the
        // invocation put an expression there; `GNUMAKEFLAGS` is written back
        // empty, so what a `$` in it could have meant never arises. `MFLAGS`
        // is not among them — its second define passes 1 — and it is the
        // reason this is a list of names rather than a rule about switches.
        let var = if matches!(k.as_bytes(), b"MAKELEVEL" | b"GNUMAKEFLAGS") {
            Variable::with_simple_string(v, origin, Some(frame), None)
        } else {
            let mut loc = Loc::default();
            let parsed = parse_expr(&mut ev.session, &mut loc, v.clone(), ParseExprOpt::Normal)?;
            Variable::new_recursive(parsed, origin, Some(frame), None, v)
        };
        // Everything culled from the environment is exported by default, and
        // GNU Make records that on the variable rather than deriving it from
        // the origin — which is why a makefile that replaces the name keeps
        // handing it to its children, and why `SHELL` never is: POSIX says a
        // makefile's SHELL must not change the one subprocesses are given, so
        // the import marks it withheld and the invocation's own value is what
        // reaches them.
        var.write().export = if k.as_bytes() == b"SHELL" {
            VarExport::NoExport
        } else {
            VarExport::Export
        };
        ev.session.set_global_var(sym, var, false, None)?;
    }
    Ok(())
}

/// Read `GNUMAKEFLAGS` a second time, once the last makefile has been read.
///
/// GNU Make's `main` does this immediately after `read_all_makefiles`: whatever
/// the name holds by then is decoded as switches at the makefile's own rank,
/// and the name is emptied again so nothing reads it twice — this time at
/// `o_override`, which is why `$(origin GNUMAKEFLAGS)` answers `override` to a
/// recipe whatever the invocation was given and whether or not a makefile ever
/// wrote the name. The emptying is unconditional where the startup one is
/// guarded, so the name exists from here on even in a run that was handed none.
///
/// A frontend that supplies no switch grammar has no second stream to read, and
/// this leaves the name exactly as the makefiles left it.
fn decode_gnumakeflags_after_read(ev: &mut Evaluator) -> Result<()> {
    if ev.session.flags.makeflags_assignment.is_none() {
        return Ok(());
    }
    let name = ev.session.intern("GNUMAKEFLAGS");
    let value = ev.eval_var(name)?;
    if !value.is_empty() {
        ev.fold_switches_into_makeflags(&value, None)?;
    }
    // The export attribute travels with the name rather than with the value:
    // GNU Make redefines the binding in place, so one that arrived from the
    // environment keeps handing its children an empty value and one this line
    // invented reaches no child at all.
    let exported = ev
        .session
        .globals
        .peek(name)
        .map_or(VarExport::Default, |var| var.read().export);
    let emptied = Variable::with_simple_string(Bytes::new(), VarOrigin::Override, None, None);
    emptied.write().export = exported;
    ev.session.globals.define(name, emptied);
    Ok(())
}

/// Install the Make interface variables an embedding frontend already parsed.
///
/// `MAKEFLAGS` is a recursive file-origin variable whose raw value refers to
/// `MAKEOVERRIDES`; the latter is a default-origin proxy for an automatic
/// simple variable. Keeping that relationship instead of importing a flattened
/// environment string means a Makefile can inspect origins and can deliberately
/// replace `MAKEOVERRIDES`, just as it can under GNU Make.
fn install_compiler_invocation_variables(ev: &mut Evaluator) {
    let Some(makeflags) = ev.session.flags.makeflags.clone() else {
        return;
    };
    let make_overrides = ev.session.flags.make_overrides.clone().unwrap_or_default();
    let inherited_overrides =
        ev.session
            .invocation_environment
            .as_ref()
            .and_then(|environment| {
                environment
                    .iter()
                    .rev()
                    .find(|(name, _)| name.as_bytes() == b"MAKEOVERRIDES")
                    .map(|(_, value)| !value.as_bytes().is_empty())
            })
            .or_else(|| {
                ev.session.invocation_environment.is_none().then(|| {
                    std::env::var_os("MAKEOVERRIDES").is_some_and(|value| !value.is_empty())
                })
            })
            .unwrap_or(false);

    let eval_flags_text = ev.session.flags.eval_flags.clone();
    let has_evals = !eval_flags_text.is_empty();
    let command_variables = ev.session.intern("-*-command-variables-*-");
    let eval_flags = ev.session.intern(crate::eval::EVAL_FLAGS_NAME);
    let overrides = ev.session.intern("MAKEOVERRIDES");
    // Both names exist only where a command-line assignment put one there. GNU
    // Make defines the pair inside `if (command_variables != 0)` in
    // `define_makeflags` (main.c), so `$(origin MAKEOVERRIDES)` answering
    // `undefined` is how a Makefile asks whether anything outranked it — an
    // assignment arriving in an inherited `MAKEFLAGS` counts, because that is
    // where the table it fills comes from in a child.
    if !make_overrides.is_empty() {
        ev.session.globals.define(
            command_variables,
            Variable::with_simple_string(make_overrides.clone(), VarOrigin::Automatic, None, None),
        );
        // Through the claim rather than a bare `is_none`, so that an inherited
        // `MAKEOVERRIDES` is promoted under `-e` on the way to declining this
        // write — as GNU Make's `define_variable_cname` inside the same
        // `if (command_variables != 0)` does. Without a command-line assignment
        // there is no define at all, and an inherited one stays `environment`
        // in both tools.
        if crate::builtins::claimable(&mut ev.session, "MAKEOVERRIDES").is_some() {
            ev.session.globals.define(
                overrides,
                Variable::new_recursive(
                    Arc::new(Value::SymRef(Loc::default(), command_variables)),
                    VarOrigin::Default,
                    None,
                    None,
                    Bytes::from_static(b"${-*-command-variables-*-}"),
                ),
            );
        }
    }

    // The `--eval` fragments live in a variable of their own, which
    // `MAKEFLAGS` names rather than contains. `o_automatic`, as GNU Make's
    // `define_variable_cname ("-*-eval-flags-*-", value, o_automatic, 0)` is,
    // and SIMPLE — which is the whole reason a fragment's `$$` survives being
    // read back through `$(MAKEFLAGS)`: substituting a simple variable's bytes
    // does not expand them a second time. The variable exists only where the
    // invocation carried a fragment, exactly as GNU Make's `if (eval_strings)`
    // guard has it, and it is never written again.
    //
    // After the command-line pair rather than before, because that is the order
    // GNU Make defines them in — `define_makeflags` at main.c:2036, then the
    // fragments at 2072 — and `$(.VARIABLES)` is a list in the order names
    // arrived.
    if has_evals {
        ev.session.globals.define(
            eval_flags,
            Variable::with_simple_string(eval_flags_text, VarOrigin::Automatic, None, None),
        );
    }

    let has_overrides = !make_overrides.is_empty() || inherited_overrides;
    if let Some(state) = &mut ev.session.flags.makeflags_assignment {
        state.has_overrides = has_overrides;
        state.has_evals = has_evals;
        // Before a Makefile has written to it, the accumulated table is exactly
        // what argv and the environment supplied — which is `protected`, and
        // not the published `MAKEFLAGS`: the two differ by the switches the
        // table carries without publishing.
        state.effective = state.protected.clone();
    }
    let (value, original) =
        crate::eval::makeflags_value(makeflags, has_evals, has_overrides, eval_flags, overrides);
    let makeflags = ev.session.intern("MAKEFLAGS");
    // GNU Make defines this one at the rank `-e` gives the environment rather
    // than at the makefile's (main.c, `env_overrides ? o_env_override :
    // o_file`), which is what keeps its own answer in place: a makefile writing
    // `MAKEFLAGS += -r` under `-e` is outranked and the flag never arrives.
    let origin = if ev.session.flags.environment_overrides {
        VarOrigin::EnvironmentOverride
    } else {
        VarOrigin::File
    };
    ev.session.globals.define(
        makeflags,
        Variable::new_recursive(value, origin, None, None, original),
    );
}

/// Bind `.DEFAULT_GOAL` to the empty selection every read starts from.
///
/// The variable exists before any Makefile is read, which is what lets one be
/// asked `$(origin .DEFAULT_GOAL)` and be told `file` rather than `undefined`.
/// The origin is not decoration: it is the rank the binding assigns at, so an
/// exported `.DEFAULT_GOAL=x` in the environment is outranked and discarded
/// here — while under `-e`, where the environment outranks the Makefile, the
/// same assignment survives and chooses the goal.
///
/// GNU Make does this in `main`, between the default variables and the first
/// line of any Makefile, and so does this.
fn install_default_goal(ev: &mut Evaluator) -> Result<()> {
    ev.session.set_global_var(
        Symbol::DEFAULT_GOAL,
        Variable::with_simple_string(Bytes::new(), VarOrigin::File, None, None),
        false,
        None,
    )
}

/// Bind `MAKEFILES` to the empty default it holds before anything sets it.
///
/// GNU Make gives it the weakest origin and the one export attribute nothing
/// else has — `define_variable_cname ("MAKEFILES", "", o_default, 0)` then
/// `v->export = v_ifset` in variable.c `define_automatic_variables`. Being
/// defined is observable on its own: `$(origin MAKEFILES)` answers `default`
/// rather than `undefined`, and the value is simple and empty.
///
/// It is not part of the catalogue `-R` withholds, because it is not in the
/// catalogue: `make -R` still answers `default` here and still reads what the
/// variable names.
fn install_makefiles_variable(ev: &mut Evaluator) -> Result<()> {
    // The environment has already been imported, and GNU Make's write is an
    // ordinary ranked one that a stronger origin declines — so an inherited
    // `MAKEFILES` keeps its value and its origin, and under `-e` is promoted on
    // the way to being declined, like any other name Make defines over. The
    // attribute is set either way, because `define_variable_cname` hands back
    // whichever variable now holds the name and `v->export = v_ifset` is
    // written on that one.
    let claim = crate::builtins::claimable(&mut ev.session, "MAKEFILES");
    let sym = ev.session.intern("MAKEFILES");
    if claim.is_none() {
        if let Some(existing) = ev.session.peek_global_var(sym) {
            existing.write().export = VarExport::IfSet;
        }
        return Ok(());
    }
    let var = Variable::with_simple_string(Bytes::new(), VarOrigin::Default, None, None);
    var.write().export = VarExport::IfSet;
    ev.session.set_global_var(sym, var, false, None)
}

/// Read the makefiles `MAKEFILES` names, before the ones the invocation asked
/// for.
///
/// GNU Make does this at the top of `read_all_makefiles` with
/// `RM_NO_DEFAULT_GOAL|RM_INCLUDED|RM_DONTCARE`, and every word of that matters:
/// a name that will not open is passed over without a word, a target one of
/// these files declares never becomes the default goal, and each file is
/// appended to `MAKEFILE_LIST` as it opens, so they stand in front of the
/// makefile the invocation named.
///
/// A makefile writing to `MAKEFILES` is too late to be read — this runs before
/// any of them — which is why the variable is only useful from the environment
/// or the command line.
fn read_makefiles_variable(ev: &mut Evaluator) -> Result<()> {
    let sym = ev.session.intern("MAKEFILES");
    if ev.session.peek_global_var(sym).is_none() {
        return Ok(());
    }
    let named = ev.eval_var(sym)?;
    let names: Vec<Bytes> = crate::strutil::word_scanner(&named)
        .map(|word| named.slice_ref(word))
        .collect();
    for name in names {
        read_makefiles_entry(ev, &name)?;
    }
    Ok(())
}

/// One name from `MAKEFILES`: read if it opens, remade if a rule says how, and
/// never allowed to choose the default goal.
///
/// `RM_DONTCARE` forgives the whole of the open, not absence alone. GNU Make
/// reads these with `eval_makefile (name, RM_NO_DEFAULT_GOAL|RM_INCLUDED|
/// RM_DONTCARE)` (read.c:204) and never looks at `errno` afterwards the way the
/// `-f` loop does, so a name with no permission is as quiet as a name nothing is
/// at — `MAKEFILES=secret.mk` says nothing at all.
///
/// Quiet is not the same as absent from the update. The goaldep `eval_makefile`
/// returns joins `read_files` like any other, so the makefile update considers
/// the name, remakes it if a rule says how, and starts the read over on what the
/// recipe wrote: `RM_DONTCARE` forgives the failure, it does not excuse the
/// attempt. `MAKEFILES=gen.mk` over a Makefile holding a `gen.mk:` rule is a
/// fragment that bootstraps itself, which is most of what the variable is for.
///
/// So the name is noted whichever way the open went — as read when it opened,
/// and as a Makefile the read wanted and did not get when it did not. Forgiven
/// either way, which is the shape `-include` already travels: no complaint is
/// held for it, and a name with no rule behind it passes without a word.
///
/// A read that fails after the open succeeded is not forgiven by anything: it is
/// `pfatal_with_name` from inside `readline` (read.c:2744), which is why
/// `MAKEFILES=<a directory>` stops the run under Make's own name.
fn read_makefiles_entry(ev: &mut Evaluator, name: &Bytes) -> Result<()> {
    let filename = OsString::from_vec(name.to_vec());
    let _file_frame = ev.enter(FrameType::Parse, name.clone(), Loc::default());
    let mk = match ev.session.get_makefile(&filename)? {
        Source::Read(mk) => mk,
        source @ (Source::Absent | Source::Unopened(_)) => {
            let reason = match &source {
                Source::Unopened(err) => crate::strerror(err),
                _ => crate::strerror(&std::io::Error::from_raw_os_error(libc::ENOENT)),
            };
            ev.note_unread_include(name.clone(), false, None, &reason);
            return Ok(());
        }
        Source::Unreadable(err) | Source::Exhausted(err) => error_loc!(
            ev,
            None,
            "*** {}: {}",
            filename.to_string_lossy(),
            crate::strerror(&err)
        ),
    };
    ev.note_read_makefile(name.clone(), false);
    ev.note_makefile_list(name.clone())?;
    let stmts = mk.stmts.lock().clone();
    ev.withhold_the_default_goal(true);
    let read = stmts.into_iter().try_for_each(|stmt| {
        log!("{stmt:?}");
        stmt.eval(ev)
    });
    ev.withhold_the_default_goal(false);
    read
}

/// Evaluate the Makefile `session` names into the graph it describes.
///
/// # Errors
///
/// Returns whatever Make evaluation or dependency analysis rejected: a syntax
/// error, a `$(error)`, a rule with no way to make one of its prerequisites.
pub fn evaluate(session: Session) -> Result<Evaluated> {
    let targets = session.flags.targets.clone();
    let cl_vars = session.flags.cl_vars.clone();

    let mut ev = Evaluator::new(session);
    ev.start()?;
    read_invocation_state(&mut ev)?;
    install_compiler_invocation_variables(&mut ev);

    // GNU Make's `define_default_variables`, in its place: after the
    // environment, so an inherited `CC` outranks the catalogue, and before any
    // Makefile, so the Makefile outranks it in turn.
    let catalogue_installed = !ev.session.flags.no_builtin_variables;
    if catalogue_installed {
        crate::builtins::install_default_variables(&mut ev.session)?;
    }
    // GNU Make's `define_automatic_variables`, which runs whatever `-R` says:
    // the path forms of the automatic variables are part of the language, and
    // a Makefile can read their origin, flavor and text before any rule has
    // been chosen.
    crate::builtins::install_path_automatic_variables(&mut ev.session)?;

    let rules_installed = !ev.session.flags.no_builtin_rules;
    crate::builtin_rules::install_suffixes_variable(&mut ev.session, !rules_installed);

    install_default_goal(&mut ev)?;
    install_makefiles_variable(&mut ev)?;

    // GNU Make's `construct_include_path` (main.c:1831), in its place: after
    // the environment, because a `-I ~` reads `HOME` out of it, and before
    // anything can include a file.
    construct_include_path(&mut ev.session);

    // The names GNU Make works out for itself, before the bootstrap Makefile so
    // that a line in it could still outrank one — none does today, and the
    // ordering is what makes that a fact about the text rather than an accident
    // of where the call sits.
    install_worked_out_variables(&mut ev, &targets)?;

    let bootstrap_asts = read_bootstrap_makefile(&mut ev.session)?;
    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*bootstrap*"),
            Loc::default(),
        );
        ev.in_bootstrap();
        let stmts = bootstrap_asts.lock().clone();
        for stmt in stmts {
            log!("{stmt:?}");
            stmt.eval(&mut ev)?;
        }
    }

    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*command line*"),
            Loc::default(),
        );
        ev.in_command_line();
        for l in &cl_vars {
            let filename = ev.session.intern("*bootstrap*");
            let asts = crate::parser::parse_buf(&mut ev.session, l, Loc { filename, line: 0 })?;
            let asts = asts.lock().clone();
            assert!(asts.len() == 1);
            asts[0].eval(&mut ev)?;
        }
        ev.capture_command_line_environment();
    }
    // GNU Make's `define_automatic_variables` runs after `decode_switches` has
    // entered the command line's own variables and before `read_all_makefiles`,
    // and this is that point: the command line has been read, no makefile has.
    stand_the_shell(&mut ev.session);
    ev.in_toplevel_makefile();

    {
        let _eval_frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*parse*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "eval time");

        // What `MAKEFILES` names comes first, so an assignment one of those
        // files makes is in scope while the invocation's own Makefile is read.
        read_makefiles_variable(&mut ev)?;

        // Every Makefile the invocation named, in the order it named them —
        // which GNU Make reads as though they had been concatenated. Reading
        // them in one session is what makes an earlier file's variables visible
        // to a later one and leaves the default goal with the first file that
        // declared a target.
        for makefile in ev.session.flags.makefiles.clone() {
            read_named_makefile(&mut ev, &makefile)?;
        }
    }

    // The environment's second option stream is read a second time here, over
    // whatever the makefiles left in it, and emptied again: `decode_env_switches
    // (STRING_SIZE_TUPLE (GNUMAKEFLAGS_NAME), o_env)` followed by
    // `define_variable_cname (GNUMAKEFLAGS_NAME, "", o_override, 0)`, main.c
    // just after `read_all_makefiles` and before the catalogue is withdrawn —
    // so a `-R` written here still takes the catalogue away below.
    //
    // `MAKEFLAGS` has no second read of its own because it never needed one:
    // GNU Make's `set_special_var` intercepts every write to that name as it
    // happens, which is what `normalize_makeflags_assignment` is. `GNUMAKEFLAGS`
    // is not a special variable, so its whole effect is decided once, here, by
    // what it holds when the last makefile has been read.
    decode_gnumakeflags_after_read(&mut ev)?;
    // A Makefile's own `MAKEFLAGS += -rR` is decoded where it is written, but
    // GNU Make withdraws the catalogue only once the whole read is over. The
    // difference is visible: `$(origin CC)` on the next line still answers
    // `default`, and the recipe that runs afterwards expands to nothing.
    if catalogue_installed && ev.session.flags.no_builtin_variables {
        crate::builtins::undefine_default_variables(&mut ev.session);
    }
    // The rules go the same way, and the list they are derived from goes with
    // them. Dependency analysis takes `.SUFFIXES` away where it can see whether
    // the Makefile wrote a list of its own; `SUFFIXES` is the readable half and
    // is emptied here.
    if rules_installed && ev.session.flags.no_builtin_rules {
        crate::builtin_rules::install_suffixes_variable(&mut ev.session, true);
    }

    if let Some(filename) = ev.session.flags.dump_include_graph.clone() {
        ev.dump_include_json(&filename)?;
    }

    let plan;
    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*dependency analysis*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "make dep time");
        let missing_includes = std::mem::take(&mut ev.missing_includes);
        let read_makefiles = std::mem::take(&mut ev.read_makefiles);
        plan = make_dep(&mut ev, targets, &read_makefiles, &missing_includes)?;
    }

    Ok(Evaluated {
        ev,
        nodes: plan.nodes,
        regeneration_nodes: plan.regenerations,
        refusals: plan.refusals,
    })
}
