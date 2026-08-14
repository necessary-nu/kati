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

// TODO: Add docs
#![allow(missing_docs)]
// These are the lints enabled by default in Android
// #![deny(missing_docs)]
// `#![deny(warnings)]` removed here for the reason given in `lib.rs`.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::ffi::{OsStr, OsString};
use std::io::{Write, stdout};
use std::os::unix::ffi::OsStrExt;

use anyhow::Result;
use bytes::Bytes;

#[cfg(feature = "gperf")]
use gperftools::{HEAP_PROFILER, PROFILER};

use kati::evaluate::{Evaluated, evaluate};
use kati::log;
use kati::ninja::generate_ninja;
use kati::regen::needs_regen;
use kati::regen_dump::stamp_dump_main;

use kati::eval::FrameType;
use kati::loc::Loc;

use kati::session::Session;
use kati::timeutil::ScopedTimeReporter;

#[cfg(all(feature = "jemalloc", not(feature = "gperf"), target_os = "linux"))]
use tikv_jemallocator::Jemalloc;

// Use jemalloc for better performance, but gperftools will use tcmalloc for
// heap debugging.
//
// Behind the `jemalloc` feature, which is off by default so that a consumer of
// the library does not compile the allocator's bundled C sources for a binary
// it is not building; see the feature's comment in `Cargo.toml`.
// no-globals-gate: the global allocator, a Ronin-level choice in the binary
// crate and not evaluation state.
#[cfg(all(feature = "jemalloc", not(feature = "gperf"), target_os = "linux"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn run(session: Session, orig_args: OsString) -> Result<i32> {
    let start_time = std::time::SystemTime::now();

    if session.flags.generate_ninja && (session.flags.regen || session.flags.dump_kati_stamp) {
        let _tr = ScopedTimeReporter::new(&session, "regen_check_time");
        if !needs_regen(&session, start_time, &orig_args) {
            eprintln!("No need to regenerate ninja file");
            return Ok(0);
        }
        if session.flags.dump_kati_stamp {
            println!("Need to regenerate ninja file");
            return Ok(0);
        }
        session.clear_glob_cache();
    }

    let Evaluated { mut ev, nodes, .. } = evaluate(session)?;

    if ev.session.flags.is_syntax_check_only {
        return Ok(0);
    }

    if ev.session.flags.generate_ninja {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*ninja generation*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "generate ninja time");
        generate_ninja(&nodes, &mut ev, orig_args.as_bytes(), start_time)?;
        ev.finish()?;
        kati::stats::report_all_stats(&ev.session);
        return Ok(0);
    }

    // This executor runs recipes as children of this process, so the exported
    // set is applied to this process's own environment once, before any of them
    // starts. Target-specific exports belong to a scope this loop does not
    // have; the graph sinks that can express one carry it per edge instead.
    for (name, value) in
        kati::export::exported_environment(&mut ev, None, kati::export::ChildKind::Recipe)?
    {
        let name = OsStr::from_bytes(&name);
        match value {
            Some(value) => {
                log!("setenv({name:?}, {})", String::from_utf8_lossy(&value));
                // SAFETY: we're single threaded here. If that changes, we could pass the
                // expected environment to the children explicitly.
                unsafe {
                    std::env::set_var(name, OsStr::from_bytes(&value));
                }
            }
            None => {
                log!("unsetenv({name:?})");
                // SAFETY: we're single threaded here. If that changes, we could pass the
                // expected environment to the children explicitly.
                unsafe {
                    std::env::remove_var(name);
                }
            }
        }
    }

    {
        let _frame = ev.enter(
            FrameType::Phase,
            Bytes::from_static(b"*execution*"),
            Loc::default(),
        );
        let _tr = ScopedTimeReporter::new(&ev.session, "exec time");
        kati::exec::exec(nodes, &mut ev)?;
    }

    ev.finish()?;
    kati::stats::report_all_stats(&ev.session);

    Ok(0)
}

fn find_first_makefile(session: &mut Session) {
    if !session.flags.makefiles.is_empty() {
        return;
    }
    if std::fs::exists("GNUMakefile").unwrap_or(false) {
        session.flags.makefiles.push(OsString::from("GNUMakefile"));
    } else if !cfg!(target_os = "macos") && std::fs::exists("makefile").unwrap_or(false) {
        session.flags.makefiles.push(OsString::from("makefile"));
    } else if std::fs::exists("Makefile").unwrap_or(false) {
        session.flags.makefiles.push(OsString::from("Makefile"));
    }
}

/// Bracket a `-C` run with the directory it moved to, as GNU Make does.
///
/// GNU Make turns its `-w` on whenever `-C` moved it, because the relative
/// paths a recipe is about to print stop resolving against the caller's
/// directory; `-s` withdraws that. Every error parser that inherited the
/// convention reads the pair, so the wording, the quoting and the flush are
/// GNU Make's: without the flush a redirected run would order the line against
/// the recipe output that has already gone out.
fn announce_directory(verb: &str) {
    let Ok(directory) = std::env::current_dir() else {
        return;
    };
    let mut stdout = stdout();
    let _ = writeln!(stdout, "kati: {verb} directory '{}'", directory.display());
    let _ = stdout.flush();
}

fn handle_realpath(args: Vec<String>) {
    for arg in args {
        if let Ok(path) = std::fs::canonicalize(&arg) {
            let _ = stdout().write_all(path.as_os_str().as_bytes());
            println!();
        }
    }
}

/// What GNU Make exits with when it abandons a build instead of finishing one.
const ABANDONED: i32 = 2;

fn main() {
    kati::logging::init("KATI_LOG", log::LevelFilter::Warn);

    if std::env::args().len() >= 2 {
        let arg = std::env::args().nth(1).unwrap();
        if arg == "--realpath" {
            handle_realpath(std::env::args().skip(2).collect());
            return;
        } else if arg == "--dump_stamp_tool" {
            // Unfortunately, this can easily be confused with --dump_kati_stamp,
            // which prints debug info about the stamp while executing a normal kati
            // run. This tool flag only dumps information, and doesn't run the rest of
            // kati.
            if let Err(err) = stamp_dump_main() {
                eprintln!("{err}");
                std::process::exit(1);
            }
            return;
        }
    }

    // Everything that used to be a process global now hangs off this value.
    // [spec:ronin:req:make.no-ambient-state]
    let mut session = Session::from_args(std::env::args_os().collect());

    #[cfg(feature = "gperf")]
    {
        if let Some(path) = &session.flags.cpu_profile_path {
            PROFILER
                .lock()
                .unwrap()
                .start(std::ffi::CString::new(path.as_bytes()).unwrap())
                .unwrap();
        }
        if let Some(path) = &session.flags.memory_profile_path {
            HEAP_PROFILER
                .lock()
                .unwrap()
                .start(std::ffi::CString::new(path.as_bytes()).unwrap())
                .unwrap();
        }
    }

    if let Some(working_dir) = &session.flags.working_dir
        && let Err(e) = std::env::set_current_dir(working_dir)
    {
        eprintln!(
            "{}*** {}: {e}  Stop.",
            kati::diagnostic_prefix(&session),
            working_dir.to_string_lossy()
        );
        std::process::exit(ABANDONED);
    }
    let announcing = session.flags.working_dir.is_some() && !session.flags.is_silent_mode;
    if announcing {
        announce_directory("Entering");
    }
    let orig_args = std::env::args_os()
        .collect::<Vec<OsString>>()
        .join(OsStr::new(" "));
    find_first_makefile(&mut session);
    if session.flags.makefiles.is_empty() {
        eprintln!(
            "{}*** No targets specified and no makefile found.  Stop.",
            kati::diagnostic_prefix(&session)
        );
        if announcing {
            announce_directory("Leaving");
        }
        std::process::exit(ABANDONED);
    }
    let gperf_cpu = session.flags.cpu_profile_path.is_some();
    let gperf_mem = session.flags.memory_profile_path.is_some();
    let _ = (gperf_cpu, gperf_mem);
    let ret = match run(session, orig_args) {
        Ok(ret) => ret,
        Err(err) => {
            for cause in err.chain() {
                eprintln!("{cause}");
            }
            // GNU Make exits 2 when it abandons a build rather than finishing
            // one, and scripts branch on the difference: 1 is the answer `-q`
            // gives to a question, not the way a run fails.
            ABANDONED
        }
    };
    if announcing {
        announce_directory("Leaving");
    }
    #[cfg(feature = "gperf")]
    {
        if gperf_cpu {
            PROFILER.lock().unwrap().stop().unwrap();
        }
        if gperf_mem {
            HEAP_PROFILER.lock().unwrap().stop().unwrap();
        }
    }
    std::process::exit(ret);
}
