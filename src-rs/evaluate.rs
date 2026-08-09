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

use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;

use crate::dep::{NamedDepNode, make_dep_with_additional_targets};
use crate::error_loc;
use crate::eval::{Evaluator, FrameType};
use crate::expr::Value;
use crate::loc::Loc;
use crate::log;
use crate::session::Session;
use crate::stmt::Stmt;
use crate::symtab::{Symbol, join_symbols};
use crate::timeutil::ScopedTimeReporter;
use crate::var::{VarOrigin, Variable};

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
    /// They are ordinary graph roots. An embedding frontend may build them and
    /// evaluate the Makefile again, just as Ninja rebuilds and reloads its own
    /// manifest.
    pub regeneration_nodes: Vec<NamedDepNode>,
}

/// The Makefile kati reads before the real one.
///
/// Half of it is GNU Make's built-in variables and suffix rules, and half is
/// kati telling the Makefile about the invocation: what `$(MAKE)` re-runs, what
/// goals were asked for, and where the run started.
fn read_bootstrap_makefile(
    session: &mut Session,
    targets: &[Symbol],
) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    let mut bootstrap = BytesMut::new();
    // The tool defaults, and the only part `-R` withholds. Everything below is
    // Make describing itself, which it goes on doing.
    if !session.flags.no_builtin_variables {
        bootstrap.put_slice(b"CC?=cc\n");
        if cfg!(target_os = "macos") {
            bootstrap.put_slice(b"CXX?=c++\n");
        } else {
            bootstrap.put_slice(b"CXX?=g++\n");
        }
        bootstrap.put_slice(b"AR?=ar\n");
    }
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

    if !session.flags.no_builtin_rules {
        // http://www.gnu.org/software/make/manual/make.html#Catalogue-of-Rules
        // The document above is actually not correct. See default.c:
        // http://git.savannah.gnu.org/cgit/make.git/tree/default.c?id=4.1
        bootstrap.put_slice(b".SUFFIXES: .o .c .cc\n");
        bootstrap.put_slice(b".c.o:\n");
        bootstrap.put_slice(b"\t$(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -o $@ $<\n");
        bootstrap.put_slice(b".cc.o:\n");
        bootstrap.put_slice(b"\t$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -o $@ $<\n");
        // TODO: Add more builtin rules.
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

/// Seed the evaluator with `MAKEFILE_LIST` and the process environment.
fn read_invocation_state(ev: &mut Evaluator) -> Result<()> {
    let mut makefile_list = BytesMut::new();
    makefile_list.put_u8(b' ');
    makefile_list.put_slice(ev.session.flags.makefile.clone().unwrap().as_bytes());
    let frame = ev.current_frame();
    let loc = ev.loc.clone();
    let makefile_list_sym = ev.session.intern("MAKEFILE_LIST");
    ev.session.set_global_var(
        makefile_list_sym,
        Variable::with_simple_string(makefile_list.freeze(), VarOrigin::File, Some(frame), loc),
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
        ev.session.set_global_var(
            sym,
            Variable::new_recursive(val, origin, Some(frame), None, v),
            false,
            None,
        )?;
    }
    Ok(())
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
    }
    ev.in_toplevel_makefile();

    {
        let _eval_frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*parse*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "eval time");

        let makefile = ev.session.flags.makefile.clone().unwrap();
        let _file_frame = ev.enter(
            FrameType::Parse,
            Bytes::from(makefile.as_bytes().to_vec()),
            Loc::default(),
        );
        let Some(mk) = ev.session.get_makefile(&makefile)? else {
            bail!("makefile not found")
        };
        let stmts = mk.stmts.lock().clone();
        for stmt in stmts {
            log!("{stmt:?}");
            stmt.eval(&mut ev)?;
        }
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
        let additional_targets = missing_includes
            .iter()
            .map(|include| include.filename)
            .collect();
        nodes = make_dep_with_additional_targets(&mut ev, targets, additional_targets)?;

        regeneration_nodes = {
            let mut regeneration_nodes = Vec::new();
            for include in missing_includes {
                let found = nodes
                    .iter()
                    .find(|(name, _)| *name == include.filename)
                    .cloned();
                if let Some(node) = found.filter(|node| node.1.lock().has_rule) {
                    regeneration_nodes.push(node);
                } else if include.required {
                    let filename = include.filename;
                    let loc = include.loc;
                    error_loc!(
                        &ev,
                        Some(&loc),
                        "{}: No such file or directory",
                        filename.display(&ev)
                    );
                }
            }
            regeneration_nodes
        };
    }

    Ok(Evaluated {
        ev,
        nodes,
        regeneration_nodes,
    })
}
