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
use std::sync::Arc;

use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;

use crate::dep::{NamedDepNode, RegenerationRoot, make_dep};
use crate::eval::{Evaluator, FrameType};
use crate::expr::Value;
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
/// `archives` is deliberately absent though `lib.a(member.o)` is now read as a
/// member of an archive: the shape, the built-in `(%): %` rule, `$%` and
/// member freshness off the archive index are here, and `ar_glob`, the
/// `.X.a` suffix conversion, `lib.a(a.o b.o)` and `-t` are not. GNU Make's own
/// suite is the measure and says so plainly — claiming the feature runs its
/// `features/archives` script, whose sixteen cases leave eight compiler-class
/// differences today. Those eight are what the remaining work is worth, and
/// the entry belongs here when they are gone.
///
/// Build-side features belong to whoever runs the recipes, not to the
/// evaluator, and arrive through [`Flags::extra_features`].
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
fn read_bootstrap_makefile(
    session: &mut Session,
    targets: &[Symbol],
) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    let mut bootstrap = BytesMut::new();
    // The one place a GNU Make version is claimed, because Makefiles branch on
    // it. It names the version this front end is measured against rather than
    // the one the vendored Go harness pinned: a Makefile that tests
    // `$(MAKE_VERSION)` for a feature must get the answer the oracle would
    // give, or it takes a branch neither tool would have taken.
    bootstrap.put_slice(b"MAKE_VERSION?=4.4.1\n");
    // What a Makefile is allowed to assume, and no more. Claiming a feature
    // that is not there is worse than claiming none: a Makefile branches on
    // this to decide whether it may use a construct, and GNU Make's test suite
    // skips a case it names. An honest short list makes a build take the
    // portable path; a generous one makes it take a path that then misbehaves.
    // Simple assignment rather than `?=`, because it is a statement about the
    // program and not a default the makefile may prefer to set.
    let features = EVALUATOR_FEATURES
        .iter()
        .map(|feature| (*feature).to_owned())
        .chain(session.flags.extra_features.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ");
    bootstrap.put_slice(b".FEATURES := ");
    bootstrap.put_slice(features.as_bytes());
    bootstrap.put_u8(b'\n');
    bootstrap.put_slice(b"KATI?=ckati\n");
    // Overwrite $SHELL environment variable.
    bootstrap.put_slice(b"SHELL=/bin/sh\n");
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
    if session.flags.generate_ninja {
        bootstrap.put_slice(format!("MAKE?=make -j{}\n", session.flags.num_jobs.max(1)).as_bytes());
    } else {
        bootstrap.put_slice(b"MAKE?=");
        bootstrap.put_slice(session.flags.subkati_args.join(OsStr::new(" ")).as_bytes());
        bootstrap.put_u8(b'\n');
    }
    bootstrap.put_slice(b"MAKECMDGOALS?=");
    bootstrap.put(join_symbols(&*session, targets, b" "));
    bootstrap.put_u8(b'\n');

    bootstrap.put_slice(b"CURDIR:=");
    bootstrap.put_slice(std::env::current_dir()?.as_os_str().as_bytes());
    bootstrap.put_u8(b'\n');

    let filename = session.intern("*bootstrap*");
    crate::parser::parse_buf(session, &bootstrap.freeze(), Loc { filename, line: 0 })
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
        let val = Arc::new(Value::Literal(None, v.clone()));
        let frame = ev.current_frame();
        let sym = ev.session.intern(k.as_bytes().to_vec());
        let var = Variable::new_recursive(val, origin, Some(frame), None, v);
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

    let command_variables = ev.session.intern("-*-command-variables-*-");
    ev.session.globals.define(
        command_variables,
        Variable::with_simple_string(make_overrides.clone(), VarOrigin::Automatic, None, None),
    );

    let overrides = ev.session.intern("MAKEOVERRIDES");
    if ev.session.peek_global_var(overrides).is_none() {
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

    let has_overrides = !make_overrides.is_empty() || inherited_overrides;
    if let Some(state) = &mut ev.session.flags.makeflags_assignment {
        state.has_overrides = has_overrides;
        // Before a Makefile has written to it, the accumulated table is exactly
        // what argv and the environment supplied — which is `protected`, and
        // not the published `MAKEFLAGS`: the two differ by the switches the
        // table carries without publishing.
        state.effective = state.protected.clone();
    }
    let (value, original) = if has_overrides {
        let mut prefix = BytesMut::from(makeflags.as_ref());
        prefix.put_slice(b" -- ");
        let mut original = prefix.clone();
        original.put_slice(b"$(MAKEOVERRIDES)");
        (
            Arc::new(Value::List(
                None,
                vec![
                    Arc::new(Value::Literal(None, prefix.freeze())),
                    Arc::new(Value::SymRef(Loc::default(), overrides)),
                ],
            )),
            original.freeze(),
        )
    } else {
        (Arc::new(Value::Literal(None, makeflags.clone())), makeflags)
    };
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
    let sym = ev.session.intern("MAKEFILES");
    // The environment has already been imported, and GNU Make's write is an
    // ordinary ranked one that a stronger origin declines — so an inherited
    // `MAKEFILES` keeps its value and its origin. The attribute is set either
    // way, because `define_variable_cname` hands back whichever variable now
    // holds the name and `v->export = v_ifset` is written on that one.
    if let Some(existing) = ev.session.peek_global_var(sym) {
        existing.write().export = VarExport::IfSet;
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

/// One name from `MAKEFILES`: read if it opens, passed over in silence if it
/// does not, and never allowed to choose the default goal.
///
/// `RM_DONTCARE` forgives the whole of the open, not absence alone. GNU Make
/// reads these with `eval_makefile (name, RM_NO_DEFAULT_GOAL|RM_INCLUDED|
/// RM_DONTCARE)` (read.c:204) and never looks at `errno` afterwards the way the
/// `-f` loop does, so a name with no permission is as quiet as a name nothing is
/// at — `MAKEFILES=secret.mk` builds the goals and says nothing at all.
///
/// A read that fails after the open succeeded is not forgiven by anything: it is
/// `pfatal_with_name` from inside `readline` (read.c:2744), which is why
/// `MAKEFILES=<a directory>` stops the run under Make's own name.
fn read_makefiles_entry(ev: &mut Evaluator, name: &Bytes) -> Result<()> {
    let filename = OsString::from_vec(name.to_vec());
    let _file_frame = ev.enter(FrameType::Parse, name.clone(), Loc::default());
    let mk = match ev.session.get_makefile(&filename)? {
        Source::Read(mk) => mk,
        Source::Absent | Source::Unopened(_) => return Ok(()),
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

    let bootstrap_asts = read_bootstrap_makefile(&mut ev.session, &targets)?;
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
