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
    build_sink::{NewInputsTiming, ShellEvaluation},
    dep::DepNode,
    error_loc,
    eval::Evaluator,
    exec::ExecStatus,
    expr::{Evaluable, Value},
    fileutil::get_timestamp,
    strutil::{
        Pattern, WordWriter, basename, dirname, find_end_of_line, trim_left_space, word_scanner,
    },
    symtab::{Interner, Symbol},
    var::Variable,
};

pub(crate) const DEFERRED_NEW_INPUTS_REFERENCE: &[u8] = b"${KATI_NEW_INPUTS}";

#[derive(Clone)]
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
    Question {
        found_new_inputs: Arc<Mutex<bool>>,
        timing: NewInputsTiming,
    },
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
                out.put_slice(&current_dep_node.recipe_output.as_bytes(names));
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
                    out.put_slice(pat.stem(&current_dep_node.recipe_output.as_bytes(names)))
                }
            }
            AutoCommand::Question {
                found_new_inputs,
                timing,
            } => {
                let mut seen: HashSet<Symbol> = HashSet::new();

                if ev.avoid_io
                    && (*timing == NewInputsTiming::SchedulerBoundary
                        || current_dep_node.grouped_double_action.is_some())
                {
                    // The grouped action's comparison is deliberately made by
                    // the scheduler after its prerequisites finish.  It binds
                    // this value when the edge is launched; doing a second
                    // shell-side timestamp test would move the snapshot past
                    // the prerequisite boundary.
                    out.put_slice(DEFERRED_NEW_INPUTS_REFERENCE);
                    *found_new_inputs.lock() = true;
                } else if let Some(action) = &current_dep_node.grouped_double_action {
                    let mut oldest_member = None;
                    let mut missing_member = action.has_phony_member;
                    for member in &action.members {
                        match get_timestamp(&member.as_bytes(names))? {
                            Some(mtime) => {
                                oldest_member = Some(
                                    oldest_member
                                        .map_or(mtime, |oldest| std::cmp::min(oldest, mtime)),
                                );
                            }
                            None => missing_member = true,
                        }
                    }
                    let mut ww = WordWriter::new(out);
                    for ai in &current_dep_node.actual_inputs {
                        let ai_str = ai.as_bytes(names);
                        let input_mtime = get_timestamp(&ai_str)?;
                        if seen.insert(*ai)
                            && (missing_member
                                || action.phony_inputs.contains(ai)
                                || input_mtime.is_none()
                                || oldest_member.is_some_and(|oldest| input_mtime > Some(oldest)))
                        {
                            ww.write(&ai_str);
                        }
                    }
                } else if ev.avoid_io {
                    let mut delayed = None;
                    // Check timestamps using the shell at the start of rule execution
                    // instead.
                    out.put_slice(DEFERRED_NEW_INPUTS_REFERENCE);
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
                        ww.write(&current_dep_node.recipe_output.as_bytes(names));
                        ww.write(b"&& echo -newer");
                        ww.write(&current_dep_node.recipe_output.as_bytes(names));
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
                        &current_dep_node.recipe_output.as_bytes(names),
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

#[derive(Clone)]
pub struct Command {
    pub output: Symbol,
    pub cmd: Bytes,
    pub echo: bool,
    pub ignore_error: bool,
    /// Whether this line was written with a `-` prefix, as distinct from being
    /// ignored because of `-i` or `.IGNORE`. GNU Make's `lines_flags`, and the
    /// only one of the three that also reaches the shell's flags.
    pub dash_prefixed: bool,
    /// The flags this line's shell takes, chosen while the rule's own
    /// variables were in scope so a target-specific `.SHELLFLAGS` is seen.
    ///
    /// Per line rather than per recipe because GNU Make decides it per line:
    /// under `.POSIX:` a `-` prefix asks for a shell without `-e` while the
    /// line beside it still gets one.
    pub shell_flag: Bytes,
    pub force_no_subshell: bool,
    /// GNU Make's recursive-line classification, read from the recipe before
    /// it is expanded: the `+` prefix, or a `$(MAKE)`/`${MAKE}` reference.
    ///
    /// GNU Make 4.4.1 uses it to decide which lines run under `-n`, `-t` and
    /// `-q`, because running the child is the only way it can learn what the
    /// child would do. Here it is a compiler input and nothing else: a
    /// classified line describes recursion, so the child Makefile is compiled
    /// and composed into the graph instead. Verified against 4.4.1: with
    /// `MAKE_ALIAS = $(MAKE)`, the line `$(MAKE_ALIAS) --version` is printed
    /// and not run under `-n`, so it is the reference that classifies and not
    /// the value it expands to.
    pub recursive_line: bool,
    /// Values produced by `MAKE` references while expanding this recipe line,
    /// kept only where the expansion put one in the invoked command position.
    pub recursive_make: Vec<Bytes>,
    /// A recursion this compiler can see but cannot turn into a child
    /// compilation: the line is classified recursive and starts a Make process
    /// somewhere a shell begins a command, but not in a position
    /// [`invokes_make`] can lift out as one static invocation.
    ///
    /// A line like this carries no [`Self::recursive_make`], so a recipe that
    /// splits its other lines into child graphs would read this one as
    /// ordinary residual work and leave it to start a nested Make beside them.
    /// Naming it lets the compiler decline to split the recipe at all, which
    /// is the rule the split already claimed to follow.
    pub uncomposable_recursion: bool,
}

/// Whether an unexpanded recipe references `MAKE` the way GNU Make's own
/// classification reads it.
///
/// GNU Make scans the recipe text for the literal `$(MAKE)` or `${MAKE}`, so
/// what counts is the reference and not what it expands to. Both spellings
/// parse to the same [`Value::SymRef`], and every other way of naming the
/// variable — `$(MAKE:x=y)`, `$($(V))` with `V = MAKE`, a variable holding
/// `$(MAKE)` — parses to something else, which is exactly the set 4.4.1
/// declines to classify. Function arguments are walked because the text
/// inside them is text GNU Make scans too: `$(info $(MAKE))` is classified.
fn references_make(value: &Value, names: &impl Interner) -> bool {
    match value {
        Value::Literal(_, _) => false,
        Value::SymRef(_, sym) => sym.as_bytes(names).as_ref() == b"MAKE",
        Value::List(_, values) => values.iter().any(|value| references_make(value, names)),
        Value::VarRef(_, name) => references_make(name, names),
        // `$(MAKE:x=y)` holds the name as a literal rather than a reference,
        // and 4.4.1 does not classify it. The pattern and replacement are
        // ordinary text and can hold a reference of their own.
        Value::VarSubst {
            loc: _,
            name: _,
            pat,
            subst,
        } => references_make(pat, names) || references_make(subst, names),
        Value::Func {
            loc: _,
            fi: _,
            args,
        } => args.iter().any(|arg| references_make(arg, names)),
    }
}

fn starts_with_word(command: &[u8], word: &[u8]) -> bool {
    command.starts_with(word)
        && command
            .get(word.len())
            .is_none_or(|next| next.is_ascii_whitespace())
}

/// Split an expanded recipe line where a shell would begin a new command.
///
/// Quoting is respected, so the `;` in `echo "a; make b"` is text rather than
/// a separator and the line keeps its one segment. Redirections and
/// substitutions are left alone: they change what a segment reads or writes,
/// not where the next one starts.
fn command_segments(command: &[u8]) -> Vec<&[u8]> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut index = 0;
    while index < command.len() {
        let byte = command[index];
        match quote {
            Some(delimiter) => {
                if byte == delimiter {
                    quote = None;
                } else if byte == b'\\' && delimiter == b'"' {
                    index += 1;
                }
            }
            None => match byte {
                b'\'' | b'"' => quote = Some(byte),
                b'\\' => index += 1,
                b';' | b'&' | b'|' | b'(' | b')' | b'\n' => {
                    segments.push(&command[start..index]);
                    // A two-byte operator must not leave its second byte to be
                    // read as the head of the next segment.
                    if command.get(index + 1) == Some(&byte) && matches!(byte, b'&' | b'|') {
                        index += 1;
                    }
                    start = index + 1;
                }
                _ => {}
            },
        }
        index += 1;
    }
    segments.push(&command[start..]);
    segments
}

/// Whether one expanded `MAKE` value starts a process anywhere in the line.
///
/// Wider than [`invokes_make`], which asks the narrower question of whether
/// the line is *one* invocation that can be lifted out as a child
/// compilation. This asks only whether a nested Make would be started at all,
/// so `test -d sub && $(MAKE) -C sub` answers yes while `echo "run $(MAKE)"`
/// answers no.
fn spawns_make(command: &[u8], make: &[u8]) -> bool {
    command_segments(command).into_iter().any(|segment| {
        let mut segment = segment.trim_ascii_start();
        // A leading `VAR=value` sequence, and `env` or `exec` before the
        // program, are the shell's way of saying the same command differently.
        while let Some(word) = segment
            .split(|byte| byte.is_ascii_whitespace())
            .next()
            .filter(|word| !word.is_empty())
            .filter(|word| {
                *word == b"env"
                    || *word == b"exec"
                    || word.split(|byte| *byte == b'=').next().is_some_and(|name| {
                        name.len() < word.len() && !name.is_empty() && !name.contains(&b'/')
                    })
            })
        {
            segment = segment[word.len()..].trim_ascii_start();
        }
        starts_with_word(segment, make)
    })
}

/// Whether one expanded `MAKE` value occupies the command position of a
/// recipe line rather than merely being printed or passed as data.
fn invokes_make(command: &[u8], make: &[u8]) -> bool {
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
    recursive_line: &mut bool,
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
                *recursive_line = true;
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
    pub fn new(
        ev: &'a mut Evaluator,
        new_inputs_timing: NewInputsTiming,
        shell_evaluation: ShellEvaluation,
    ) -> Result<Self> {
        ev.new_inputs_timing = new_inputs_timing;
        ev.shell_evaluation = shell_evaluation;
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
        ret.register_autocommand(
            '?',
            AutoCommand::Question {
                found_new_inputs,
                timing: new_inputs_timing,
            },
        )?;
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
        self.ev.deferred_new_inputs_filter_out.clear();
        for v in node_cmds {
            self.ev.loc = v.loc();
            self.ev.expanded_make_in_command.clear();
            let cmds_buf = v.eval_to_buf(self.ev)?;
            let make_values = self.ev.expanded_make_in_command.clone();
            let mut cmds = cmds_buf.clone();
            let mut global_echo = !self.ev.session.flags.is_silent_mode;
            // `-i` and `.IGNORE` say a failure does not count, which is what
            // the `-` prefix says too — but only the prefix also relaxes the
            // shell, so the two are carried separately and joined at the end.
            let ignored_without_prefix = self.ev.session.flags.ignore_errors || node_ignores_errors;
            let mut global_dash_prefixed = false;
            // The classification is read from the recipe as written, so it has
            // to be taken before anything is expanded away.
            let mut global_recursive_line = references_make(&v, &self.ev.session);
            cmds = parse_command_prefixes(
                cmds,
                &mut global_echo,
                &mut global_dash_prefixed,
                &mut global_recursive_line,
            );
            if cmds.is_empty() {
                continue;
            }
            while !cmds.is_empty() {
                let eol = find_end_of_line(&cmds);
                let mut cmd = eol.line.slice_ref(trim_left_space(&eol.line));
                cmds = eol.rest;

                let mut echo = global_echo;
                let mut dash_prefixed = global_dash_prefixed;
                let mut recursive_line = global_recursive_line;
                cmd =
                    parse_command_prefixes(cmd, &mut echo, &mut dash_prefixed, &mut recursive_line);

                if !cmd.is_empty() {
                    let recursive_make: Vec<Bytes> = make_values
                        .iter()
                        .filter(|make| invokes_make(&cmd, make))
                        .cloned()
                        .collect();
                    // Only a classified line is held to this. A `MAKE`-valued
                    // variable that GNU Make never classified is composed when
                    // the expansion makes that possible and otherwise left as
                    // written, exactly as 4.4.1 leaves it.
                    let uncomposable_recursion = recursive_line
                        && recursive_make.is_empty()
                        && make_values.iter().any(|make| spawns_make(&cmd, make));
                    let shell_flag = self.ev.get_shell_flag(dash_prefixed)?;
                    result.push(Command {
                        output: n.lock().recipe_output,
                        cmd,
                        echo,
                        ignore_error: ignored_without_prefix || dash_prefixed,
                        dash_prefixed,
                        shell_flag,
                        force_no_subshell: false,
                        recursive_line,
                        recursive_make,
                        uncomposable_recursion,
                    })
                }
            }
        }

        if !self.ev.delayed_output_commands.is_empty() {
            // Written by the front end rather than by the Makefile, so there is
            // no `-` prefix to read and no line for one to be on.
            let shell_flag = self.ev.get_shell_flag(false)?;
            let mut output_commands = Vec::new();
            let node = n.lock();
            for cmd in &self.ev.delayed_output_commands {
                output_commands.push(Command {
                    output: node.recipe_output,
                    cmd: cmd.clone(),
                    echo: false,
                    ignore_error: false,
                    dash_prefixed: false,
                    shell_flag: shell_flag.clone(),
                    force_no_subshell: true,
                    recursive_line: false,
                    recursive_make: Vec::new(),
                    uncomposable_recursion: false,
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
    use super::{invokes_make, references_make, spawns_make};
    use crate::expr::{ParseExprOpt, parse_expr};
    use crate::loc::Loc;
    use crate::session::Session;
    use bytes::Bytes;

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

    /// Whether a recipe line as written classifies as recursive, which is the
    /// question GNU Make 4.4.1 answers by looking for the literal `$(MAKE)` or
    /// `${MAKE}` before anything is expanded.
    fn classified(recipe: &'static [u8]) -> bool {
        let mut session = Session::new();
        let value = parse_expr(
            &mut session,
            &mut Loc::default(),
            Bytes::from_static(recipe),
            ParseExprOpt::Command,
        )
        .expect("a parsable recipe line");
        references_make(&value, &session)
    }

    /// Probed against GNU Make 4.4.1 under `-n`, which runs a classified line
    /// and prints the rest: the first three lines wrote their file and the
    /// last three did not.
    #[test]
    fn the_reference_classifies_a_recipe_line_and_not_the_value() {
        assert!(classified(b"$(MAKE) -C sub"));
        assert!(classified(b"${MAKE} --version"));
        assert!(classified(b"echo info $(info $(MAKE))"));
        // `MAKE_ALIAS = $(MAKE)` holds the same value and is not the same
        // reference, and 4.4.1 declines it.
        assert!(!classified(b"$(MAKE_ALIAS) --version"));
        assert!(!classified(b"echo subst $(MAKE:x=y)"));
        // `V = MAKE`, so the name is computed rather than written.
        assert!(!classified(b"echo indirect $($(V))"));
        assert!(!classified(b"echo plain"));
    }

    #[test]
    fn a_nested_make_is_found_wherever_a_shell_would_start_one() {
        assert!(spawns_make(b"test -d sub && make -C sub", b"make"));
        assert!(spawns_make(b"cd sub; make child", b"make"));
        assert!(spawns_make(b"true || make fallback", b"make"));
        assert!(spawns_make(b"V=1 exec make child", b"make"));
        // Mentioning it is not starting it, and a separator inside quotes is
        // text rather than a separator.
        assert!(!spawns_make(b"echo \"run make install\"", b"make"));
        assert!(!spawns_make(b"echo 'a; make b'", b"make"));
        assert!(!spawns_make(b"printf '%s' make", b"make"));
        assert!(!spawns_make(b"echo done > make", b"make"));
    }
}
