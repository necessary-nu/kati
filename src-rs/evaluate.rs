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

use crate::dep::{NamedDepNode, make_dep};
use crate::eval::{Evaluator, FrameType};
use crate::expr::Value;
use crate::loc::Loc;
use crate::log;
use crate::session::Session;
use crate::stmt::Stmt;
use crate::symtab::{Symbol, join_symbols};
use crate::timeutil::ScopedTimeReporter;
use crate::var::{VarOrigin, Variable};

/// A Makefile that has been read, expanded, and reduced to a graph.
pub struct Evaluated {
    /// The evaluator the graph was produced by, which still holds the session,
    /// the exported variables, and everything a stamp or a command needs.
    pub ev: Evaluator,
    /// The roots of the dependency graph, in the order the targets asked for
    /// them.
    pub nodes: Vec<NamedDepNode>,
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
    bootstrap.put_slice(b"CC?=cc\n");
    if cfg!(target_os = "macos") {
        bootstrap.put_slice(b"CXX?=c++\n");
    } else {
        bootstrap.put_slice(b"CXX?=g++\n");
    }
    bootstrap.put_slice(b"AR?=ar\n");
    // The one place a GNU Make version is claimed, because Makefiles branch on
    // it. It names the version this front end is measured against rather than
    // the one the vendored Go harness pinned: a Makefile that tests
    // `$(MAKE_VERSION)` for a feature must get the answer the oracle would
    // give, or it takes a branch neither tool would have taken.
    bootstrap.put_slice(b"MAKE_VERSION?=4.4.1\n");
    bootstrap.put_slice(b"KATI?=ckati\n");
    // Overwrite $SHELL environment variable.
    bootstrap.put_slice(b"SHELL=/bin/sh\n");
    // TODO: Add more builtin vars.

    if !session.flags.no_builtin_rules {
        // http://www.gnu.org/software/make/manual/make.html#Catalogue-of-Rules
        // The document above is actually not correct. See default.c:
        // http://git.savannah.gnu.org/cgit/make.git/tree/default.c?id=4.1
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
    for (k, v) in std::env::vars_os() {
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
    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*dependency analysis*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "make dep time");
        nodes = make_dep(&mut ev, targets)?;
    }

    Ok(Evaluated { ev, nodes })
}
