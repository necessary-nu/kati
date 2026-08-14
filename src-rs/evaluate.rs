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

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
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
        Source::Absent => {
            warn_loc!(
                ev,
                None,
                "{}: No such file or directory",
                makefile.to_string_lossy()
            );
            ev.note_missing_include(name, true, None);
            return Ok(());
        }
        // No `include` asked for this one, so there is no line to point
        // at; the diagnostic still has to say which file and why.
        Source::Unreadable(err) => error_loc!(
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
    // Where an environment variable sits in Make's precedence order is the
    // whole of `-e`: normally the makefile's own assignment wins, and under
    // `-e` the environment does, which is what an overriding origin says.
    let origin = if ev.session.flags.environment_overrides {
        VarOrigin::EnvironmentOverride
    } else {
        VarOrigin::Environment
    };
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
        state.effective = makeflags.clone();
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
    ev.session.globals.define(
        makeflags,
        Variable::new_recursive(value, VarOrigin::File, None, None, original),
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
    let rules_installed = !ev.session.flags.no_builtin_rules;
    crate::builtin_rules::install_suffixes_variable(&mut ev.session, !rules_installed);

    install_default_goal(&mut ev)?;

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

    let nodes;
    let regeneration_nodes;
    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*dependency analysis*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "make dep time");
        let missing_includes = std::mem::take(&mut ev.missing_includes);
        let read_makefiles = std::mem::take(&mut ev.read_makefiles);
        (nodes, regeneration_nodes) =
            make_dep(&mut ev, targets, &read_makefiles, &missing_includes)?;
    }

    Ok(Evaluated {
        ev,
        nodes,
        regeneration_nodes,
    })
}
