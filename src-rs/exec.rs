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

use std::{ffi::OsStr, os::unix::ffi::OsStrExt, path::Path, sync::Arc, time::SystemTime};

use crate::fasthash::FastMap;
use anyhow::Result;
use parking_lot::Mutex;

use crate::{
    build_sink::{FileEvaluation, NewInputsTiming, OutputEvaluation, ShellEvaluation},
    command::CommandEvaluator,
    dep::{DepNode, NamedDepNode},
    error, error_loc,
    eval::{Evaluator, FrameType},
    fileutil::{RedirectStderr, get_timestamp, run_command},
    log,
    symtab::Symbol,
    warn,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecStatus {
    Processing,
    Timestamp(Option<SystemTime>),
}

impl PartialOrd for ExecStatus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ExecStatus::Processing, ExecStatus::Processing) => Some(std::cmp::Ordering::Equal),
            (ExecStatus::Processing, ExecStatus::Timestamp(Some(_))) => {
                Some(std::cmp::Ordering::Less)
            }
            (ExecStatus::Timestamp(None), ExecStatus::Timestamp(None)) => {
                Some(std::cmp::Ordering::Equal)
            }
            (ExecStatus::Timestamp(None), _) => Some(std::cmp::Ordering::Less),
            (_, ExecStatus::Timestamp(None)) => Some(std::cmp::Ordering::Greater),
            (ExecStatus::Timestamp(Some(a)), ExecStatus::Timestamp(Some(b))) => Some(a.cmp(b)),
            (ExecStatus::Timestamp(Some(_)), _) => Some(std::cmp::Ordering::Greater),
        }
    }
}

struct Executor<'a> {
    ce: CommandEvaluator<'a>,
    done: FastMap<Symbol, ExecStatus>,
    num_commands: u64,
}

impl<'a> Executor<'a> {
    fn new(ev: &'a mut Evaluator) -> Result<Self> {
        Ok(Executor {
            ce: CommandEvaluator::new(
                ev,
                NewInputsTiming::RecipeShell,
                ShellEvaluation::Expansion,
                // kati's own executor runs the build in this process, so it
                // performs a recipe's file operations, and prints a recipe's
                // `$(info)`, where GNU Make does.
                FileEvaluation::Expansion,
                OutputEvaluation::Expansion,
            )?,
            done: FastMap::default(),
            num_commands: 0,
        })
    }

    fn exec_node(
        &mut self,
        n: &Arc<Mutex<DepNode>>,
        needed_by: Option<&[u8]>,
    ) -> Result<ExecStatus> {
        let output = n.lock().output;
        let output_str = output.as_bytes(&self.ce.ev.session);
        if let Some(found) = self.done.get(&output) {
            if found == &ExecStatus::Processing {
                warn!(
                    "Circular {} <- {} dependency dropped.",
                    String::from_utf8_lossy(needed_by.unwrap_or(b"(null)")),
                    output.display(&self.ce.ev.session)
                )
            }
            return Ok(*found);
        }
        let loc = n.lock().loc;
        let _frame = self
            .ce
            .ev
            .enter(FrameType::Exec, output_str.clone(), loc.unwrap_or_default());

        self.done.insert(output, ExecStatus::Processing);
        let output_timestamp = get_timestamp(&output_str)?;
        let output_ts = ExecStatus::Timestamp(output_timestamp);

        log!(
            "ExecNode: {} for {}",
            output.display(&self.ce.ev.session),
            String::from_utf8_lossy(needed_by.unwrap_or(b"(null)"))
        );

        if !n.lock().has_rule && output_timestamp.is_none() && !n.lock().is_phony {
            if let Some(needed_by) = needed_by {
                error_loc!(
                    &self.ce.ev.session,
                    None,
                    "*** No rule to make target '{}', needed by '{}'.",
                    output.display(&self.ce.ev.session),
                    String::from_utf8_lossy(needed_by)
                );
            } else {
                // GNU Make ends the sentence here too; kati left the period off
                // only in this one of the pair.
                error_loc!(
                    &self.ce.ev.session,
                    None,
                    "*** No rule to make target '{}'.",
                    output.display(&self.ce.ev.session)
                );
            }
        }

        let mut latest = ExecStatus::Processing;
        let order_onlys = n.lock().order_onlys.clone();
        for (_, d) in order_onlys {
            let dep_out = d.lock().output.as_bytes(&self.ce.ev.session);
            let dep_path = Path::new(OsStr::from_bytes(&dep_out));
            match std::fs::exists(dep_path) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(err) => return Err(crate::io_failure(dep_path, &err)),
            }
            let ts = self.exec_node(&d, Some(&output_str))?;
            if latest < ts {
                latest = ts;
            }
        }

        let deps = n.lock().deps.clone();
        for (_, d) in deps {
            let ts = self.exec_node(&d, Some(&output_str))?;
            if latest < ts {
                latest = ts;
            }
        }

        if output_ts >= latest && !n.lock().is_phony {
            self.done.insert(output, output_ts);
            return Ok(output_ts);
        }

        let commands = self.ce.launches(n)?;
        for command in commands {
            self.num_commands += 1;
            if command.echo {
                println!("{}", String::from_utf8_lossy(&command.cmd));
            }
            if !self.ce.ev.session.flags.is_dry_run {
                // The shell this recipe's own scope named. GNU Make expands
                // `$(SHELL)` in `construct_command_argv` (job.c) as
                // `allocated_variable_expand_for_file ("$(SHELL)", file)`, so
                // a `SHELL` one target set governs that target's launches and
                // descends to the prerequisites built for it. The evaluation
                // above did that read with this node's scope in front of it
                // and left the answer here; asking again would be too late,
                // because the scope is taken down when the evaluation ends.
                //
                // Lazy for the same reason it is per node: the answer is only
                // filled in for a node that has a recipe to run, so a build
                // with nothing to do never asks — and a `SHELL` whose own
                // value has to start a shell to expand is a makefile GNU Make
                // runs to completion when nothing needs one.
                let shell = self.ce.recipe_shell.clone();
                let (status, output) = run_command(
                    crate::fileutil::ShellToReadWith {
                        program: &shell,
                        flag: &command.shell_flag,
                        stand_in: self.ce.ev.session.flags.default_shell_program.as_deref(),
                        // `.ONESHELL` made this one script out of the recipe's
                        // lines, so the newlines in it separate commands and
                        // there is no single command line to exec directly.
                        // The flags travelling with it are what GNU Make's own
                        // recursion would default to while it re-reads
                        // `.SHELLFLAGS` for this launch: the recipe's prefixes
                        // are the first line's, and so is the `-` this asks
                        // about.
                        one_script: self
                            .ce
                            .ev
                            .session
                            .flags
                            .one_shell
                            .then(|| self.ce.ev.default_shell_flag(command.dash_prefixed)),
                    },
                    &command.cmd,
                    // This executor applied the exported set to its own
                    // environment before the first recipe started.
                    &[],
                    RedirectStderr::Stdout,
                    &crate::diagnostic_prefix(&self.ce.ev.session),
                    &self.ce.ev.session.diagnostics,
                    self.ce.ev.session.interrupts.as_deref(),
                )?;
                // The command was waited for, so whatever it did to the
                // filesystem is what the next recipe's expansion has to see.
                self.ce.ev.session.note_command_ran();
                print!("{}", String::from_utf8_lossy(&output));
                if !status.success() {
                    if command.ignore_error {
                        eprintln!(
                            "[{}] Error {} (ignored)",
                            command.output.display(&self.ce.ev.session),
                            status.code().unwrap_or(1)
                        )
                    } else {
                        error!(
                            "{}*** [{}] Error {}",
                            crate::diagnostic_prefix(&self.ce.ev.session),
                            command.output.display(&self.ce.ev.session),
                            status.code().unwrap_or(1)
                        );
                    }
                }
            }
        }

        self.done.insert(output, output_ts);
        Ok(output_ts)
    }
}

pub fn exec(roots: Vec<NamedDepNode>, ev: &mut Evaluator) -> Result<()> {
    let mut executor = Executor::new(ev)?;
    for (_sym, root) in &roots {
        executor.exec_node(root, None)?;
    }
    if executor.num_commands == 0 {
        for (sym, _) in roots {
            println!(
                "kati: Nothing to be done for '{}'.",
                sym.display(&executor.ce.ev.session)
            )
        }
    }
    Ok(())
}
