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

use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;
use std::{collections::HashSet, fmt::Debug, sync::Arc};

use crate::{
    dep::DepNode,
    error_loc,
    eval::Evaluator,
    exec::ExecStatus,
    expr::Evaluable,
    fileutil::get_timestamp,
    strutil::{
        Pattern, WordWriter, basename, dirname, find_end_of_line, trim_left_space, word_scanner,
    },
    symtab::Symbol,
    var::Variable,
};

pub struct AutoCommandVar {
    typ: AutoCommand,
    sym: Symbol,
    variant: AutoCommandVariant,
    current_dep_node: Arc<Mutex<Option<Arc<Mutex<DepNode>>>>>,
}

#[derive(Clone, Debug)]
enum AutoCommand {
    At,
    Less,
    Hat,
    Plus,
    Bar,
    Star,
    Question { found_new_inputs: Arc<Mutex<bool>> },
    NotImplemented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoCommandVariant {
    None,
    D,
    F,
}

impl AutoCommandVar {
    pub fn eval(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
        match self.variant {
            AutoCommandVariant::None => self.eval_impl(ev, out)?,
            AutoCommandVariant::D => {
                let mut buf = BytesMut::new();
                self.eval_impl(ev, &mut buf)?;
                let buf = Bytes::from(buf);
                let mut ww = WordWriter::new(out);
                for tok in word_scanner(&buf) {
                    let tok = buf.slice_ref(tok);
                    ww.write(&dirname(&tok))
                }
            }
            AutoCommandVariant::F => {
                let mut buf = BytesMut::new();
                self.eval_impl(ev, &mut buf)?;
                let buf = Bytes::from(buf);
                let mut ww = WordWriter::new(out);
                for tok in word_scanner(&buf) {
                    ww.write(basename(tok))
                }
            }
        }
        Ok(())
    }

    fn eval_impl(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
        let current_dep_node = self.current_dep_node.lock();
        let current_dep_node = current_dep_node.as_ref().unwrap().lock();
        let names = &ev.session.symtab;

        match &self.typ {
            AutoCommand::At => {
                out.put_slice(&current_dep_node.output.as_bytes(names));
            }
            AutoCommand::Less => {
                if let Some(ai) = current_dep_node.actual_inputs.first() {
                    out.put_slice(&ai.as_bytes(names));
                }
            }
            AutoCommand::Hat => {
                let mut seen = HashSet::new();
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    if seen.insert(*ai) {
                        ww.write(&ai.as_bytes(names))
                    }
                }
            }
            AutoCommand::Plus => {
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    ww.write(&ai.as_bytes(names))
                }
            }
            AutoCommand::Bar => {
                let mut seen = HashSet::new();
                let mut ww = WordWriter::new(out);
                for oi in current_dep_node.actual_order_only_inputs.iter() {
                    if seen.insert(*oi) {
                        ww.write(&oi.as_bytes(names))
                    }
                }
            }
            AutoCommand::Star => {
                if let Some(output_pattern) = &current_dep_node.output_pattern {
                    let pat = Pattern::new(output_pattern.as_bytes(names));
                    out.put_slice(pat.stem(&current_dep_node.output.as_bytes(names)))
                }
            }
            AutoCommand::Question { found_new_inputs } => {
                let mut seen: HashSet<Symbol> = HashSet::new();

                if ev.avoid_io {
                    let mut delayed = None;
                    // Check timestamps using the shell at the start of rule execution
                    // instead.
                    out.put_slice(b"${KATI_NEW_INPUTS}");
                    if !*found_new_inputs.lock() {
                        let mut def = BytesMut::new();

                        let mut ww = WordWriter::new(&mut def);
                        ww.write(b"KATI_NEW_INPUTS=$(find");
                        for ai in current_dep_node.actual_inputs.iter() {
                            if seen.insert(*ai) {
                                ww.write(&ai.as_bytes(names));
                            }
                        }
                        ww.write(b"$(test -e");
                        ww.write(&current_dep_node.output.as_bytes(names));
                        ww.write(b"&& echo -newer");
                        ww.write(&current_dep_node.output.as_bytes(names));
                        ww.write(b")) && export KATI_NEW_INPUTS");
                        delayed = Some(def.freeze());
                        *found_new_inputs.lock() = true;
                    }
                    if let Some(def) = delayed {
                        ev.delayed_output_commands.push(def);
                    }
                } else {
                    let mut ww = WordWriter::new(out);
                    let target_age = ExecStatus::Timestamp(get_timestamp(
                        &current_dep_node.output.as_bytes(names),
                    )?);
                    for ai in current_dep_node.actual_inputs.iter() {
                        let ai_str = ai.as_bytes(names);
                        if seen.insert(*ai)
                            && ExecStatus::Timestamp(get_timestamp(&ai_str)?) > target_age
                        {
                            ww.write(&ai_str)
                        }
                    }
                }
            }
            AutoCommand::NotImplemented => {
                error_loc!(
                    ev,
                    ev.loc.as_ref(),
                    "Automatic variable `${}' isn't supported yet",
                    self.sym.display(ev)
                );
            }
        }
        Ok(())
    }
}

impl Debug for AutoCommandVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AutoVar({:?})", self.sym)
    }
}

pub struct Command {
    pub output: Symbol,
    pub cmd: Bytes,
    pub echo: bool,
    pub ignore_error: bool,
    pub force_no_subshell: bool,
    /// The `+` prefix: run this line even when the run is only pretending.
    pub always_run: bool,
    /// Values produced by `MAKE` references while expanding this recipe line.
    pub recursive_make: Vec<Bytes>,
}

/// Whether one expanded `MAKE` value occupies the command position of a
/// recipe line rather than merely being printed or passed as data.
fn invokes_make(command: &[u8], make: &[u8]) -> bool {
    fn starts_with_word(command: &[u8], word: &[u8]) -> bool {
        command.starts_with(word)
            && command
                .get(word.len())
                .is_none_or(|next| next.is_ascii_whitespace())
    }

    let mut command = command.trim_ascii_start();
    if starts_with_word(command, b"exec") {
        command = command[b"exec".len()..].trim_ascii_start();
    }
    if starts_with_word(command, make) {
        return true;
    }

    let Some(and) = command.windows(2).position(|bytes| bytes == b"&&") else {
        return false;
    };
    let before = command[..and].trim_ascii();
    let mut after = command[and + 2..].trim_ascii_start();
    if !starts_with_word(before, b"cd") {
        return false;
    }
    if starts_with_word(after, b"exec") {
        after = after[b"exec".len()..].trim_ascii_start();
    }
    starts_with_word(after, make)
}

fn parse_command_prefixes(
    cmds: Bytes,
    echo: &mut bool,
    ignore_error: &mut bool,
    always_run: &mut bool,
) -> Bytes {
    let mut s = trim_left_space(&cmds);
    while !s.is_empty() {
        match s[0] {
            b'@' => {
                *echo = false;
            }
            b'-' => {
                *ignore_error = true;
            }
            b'+' => {
                *always_run = true;
            }
            _ => {
                break;
            }
        }
        s = trim_left_space(&s[1..]);
    }
    cmds.slice_ref(s)
}

pub struct CommandEvaluator<'a> {
    pub ev: &'a mut Evaluator,
    pub current_dep_node: Arc<Mutex<Option<Arc<Mutex<DepNode>>>>>,
    pub found_new_inputs: Arc<Mutex<bool>>,
}

impl<'a> CommandEvaluator<'a> {
    pub fn new(ev: &'a mut Evaluator) -> Result<Self> {
        let found_new_inputs = Arc::new(Mutex::new(false));
        let mut ret = Self {
            ev,
            current_dep_node: Arc::new(Mutex::new(None)),
            found_new_inputs: found_new_inputs.clone(),
        };
        ret.register_autocommand('@', AutoCommand::At)?;
        ret.register_autocommand('<', AutoCommand::Less)?;
        ret.register_autocommand('^', AutoCommand::Hat)?;
        ret.register_autocommand('+', AutoCommand::Plus)?;
        ret.register_autocommand('*', AutoCommand::Star)?;
        ret.register_autocommand('?', AutoCommand::Question { found_new_inputs })?;
        // TODO: Implement them.
        ret.register_bare_autocommand('|', AutoCommand::Bar)?;
        ret.register_autocommand('%', AutoCommand::NotImplemented)?;
        Ok(ret)
    }

    /// `$|` has no D or F form: GNU Make reads `$(|D)` as an ordinary variable
    /// nobody defined and expands it to nothing.
    fn register_bare_autocommand(&mut self, c: char, a: AutoCommand) -> Result<()> {
        let sym = self.ev.session.intern(c.to_string());
        let v = Variable::new_autocommand(
            sym,
            AutoCommandVar {
                typ: a,
                sym,
                variant: AutoCommandVariant::None,
                current_dep_node: self.current_dep_node.clone(),
            },
        );
        self.ev.session.set_global_var(sym, v, false, None)?;
        Ok(())
    }

    fn register_autocommand(&mut self, c: char, a: AutoCommand) -> Result<()> {
        let sym = self.ev.session.intern(c.to_string());
        let v = Variable::new_autocommand(
            sym,
            AutoCommandVar {
                typ: a.clone(),
                sym,
                variant: AutoCommandVariant::None,
                current_dep_node: self.current_dep_node.clone(),
            },
        );
        self.ev.session.set_global_var(sym, v, false, None)?;
        let sym = self.ev.session.intern(format!("{c}D"));
        let v = Variable::new_autocommand(
            sym,
            AutoCommandVar {
                typ: a.clone(),
                sym,
                variant: AutoCommandVariant::D,
                current_dep_node: self.current_dep_node.clone(),
            },
        );
        self.ev.session.set_global_var(sym, v, false, None)?;
        let sym = self.ev.session.intern(format!("{c}F"));
        let v = Variable::new_autocommand(
            sym,
            AutoCommandVar {
                typ: a,
                sym,
                variant: AutoCommandVariant::F,
                current_dep_node: self.current_dep_node.clone(),
            },
        );
        self.ev.session.set_global_var(sym, v, false, None)?;
        Ok(())
    }

    // [spec:ronin:req:make.recursive-invocation+1]
    pub fn eval(&mut self, n: &Arc<Mutex<DepNode>>) -> Result<Vec<Command>> {
        let mut result: Vec<Command> = Vec::new();
        let node_cmds;
        {
            let node = n.lock();
            self.ev.loc = node.loc.clone();
            self.ev.current_scope = node.rule_vars.clone();
            node_cmds = node.cmds.clone();
        }
        let node_ignores_errors = n.lock().is_ignore_error;
        self.ev.is_evaluating_command = true;
        *self.current_dep_node.lock() = Some(n.clone());
        *self.found_new_inputs.lock() = false;
        for v in node_cmds {
            self.ev.loc = v.loc();
            self.ev.expanded_make_in_command.clear();
            let cmds_buf = v.eval_to_buf(self.ev)?;
            let recursive_make = self.ev.expanded_make_in_command.clone();
            let mut cmds = cmds_buf.clone();
            let mut global_echo = !self.ev.session.flags.is_silent_mode;
            // `-i` is the `-` prefix asked for once instead of per line, so it
            // starts each recipe where a leading `-` would have left it.
            let mut global_ignore_error =
                self.ev.session.flags.ignore_errors || node_ignores_errors;
            let mut global_always_run = false;
            cmds = parse_command_prefixes(
                cmds,
                &mut global_echo,
                &mut global_ignore_error,
                &mut global_always_run,
            );
            if cmds.is_empty() {
                continue;
            }
            while !cmds.is_empty() {
                let eol = find_end_of_line(&cmds);
                let mut cmd = eol.line.slice_ref(trim_left_space(&eol.line));
                cmds = eol.rest;

                let mut echo = global_echo;
                let mut ignore_error = global_ignore_error;
                let mut always_run = global_always_run;
                cmd = parse_command_prefixes(cmd, &mut echo, &mut ignore_error, &mut always_run);

                if !cmd.is_empty() {
                    let recursive_make = recursive_make
                        .iter()
                        .filter(|make| invokes_make(&cmd, make))
                        .cloned()
                        .collect();
                    result.push(Command {
                        output: n.lock().output,
                        cmd,
                        echo,
                        ignore_error,
                        force_no_subshell: false,
                        always_run,
                        recursive_make,
                    })
                }
            }
        }

        if !self.ev.delayed_output_commands.is_empty() {
            let mut output_commands = Vec::new();
            let node = n.lock();
            for cmd in &self.ev.delayed_output_commands {
                output_commands.push(Command {
                    output: node.output,
                    cmd: cmd.clone(),
                    echo: false,
                    ignore_error: false,
                    force_no_subshell: true,
                    always_run: false,
                    recursive_make: Vec::new(),
                })
            }
            // Prepend |output_commands|.
            std::mem::swap(&mut result, &mut output_commands);
            result.extend(output_commands);
            self.ev.delayed_output_commands.clear();
        }

        self.ev.current_scope = None;
        self.ev.is_evaluating_command = false;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::invokes_make;

    #[test]
    fn make_must_occupy_the_invoked_command_position() {
        assert!(invokes_make(b"make -f Child.mk", b"make"));
        assert!(invokes_make(b"exec ./make child", b"./make"));
        assert!(invokes_make(b"cd sub && make child", b"make"));
        assert!(invokes_make(b"cd 'sub dir' && exec make child", b"make"));
        assert!(!invokes_make(b"printf '%s' make", b"make"));
        assert!(!invokes_make(b"echo make && true", b"make"));
        assert!(!invokes_make(b"make-believe child", b"make"));
    }
}
