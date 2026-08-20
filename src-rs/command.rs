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
    build_sink::{FileEvaluation, NewInputsTiming, OutputEvaluation, ShellEvaluation},
    dep::DepNode,
    eval::Evaluator,
    exec::ExecStatus,
    expr::{Evaluable, Value},
    fileutil::get_timestamp,
    strutil::{
        Pattern, WordWriter, basename, dirname, find_end_of_line, trim_left_space, word_scanner,
    },
    symtab::{Interner, Symbol},
    var::{Variable, Vars},
};

/// The name a scheduler substitutes the real `$?` list for, and the two it
/// substitutes that list's directory and file halves for.
///
/// `$?` has no value while a graph is being constructed: which prerequisites
/// are newer than the target is settled after the prerequisites have been
/// made, which is later than any expansion here. So the recipe carries a name
/// and the destination binds it.
///
/// The `D` and `F` forms need names of their own rather than a `dir`/`notdir`
/// taken off the first: applied here they would read the placeholder — one
/// word, no separator in it — and answer `.` and the placeholder itself.
/// GNU Make binds all three from the same list in `set_file_variables`, so
/// all three are deferred to the one destination that has the list.
pub const NEW_INPUTS_VARIABLE: &[u8] = b"KATI_NEW_INPUTS";
/// The `$(?D)` half of the same list.
pub const NEW_INPUTS_DIRECTORIES_VARIABLE: &[u8] = b"KATI_NEW_INPUTS_D";
/// The `$(?F)` half of the same list.
pub const NEW_INPUTS_FILENAMES_VARIABLE: &[u8] = b"KATI_NEW_INPUTS_F";

pub(crate) const DEFERRED_NEW_INPUTS_REFERENCE: &[u8] = b"${KATI_NEW_INPUTS}";
pub(crate) const DEFERRED_NEW_INPUTS_DIRECTORIES_REFERENCE: &[u8] = b"${KATI_NEW_INPUTS_D}";
pub(crate) const DEFERRED_NEW_INPUTS_FILENAMES_REFERENCE: &[u8] = b"${KATI_NEW_INPUTS_F}";

/// What all three references begin with, for a reader that only needs to know
/// whether a line still holds one of them.
pub(crate) const DEFERRED_NEW_INPUTS_PREFIX: &[u8] = b"${KATI_NEW_INPUTS";

/// The date a prerequisite is compared by, which for `lib.a(member.o)` comes
/// out of the archive's index rather than off a file of that name.
///
/// GNU Make's `f_mtime` reads the shape wherever it is written, so a name with
/// parentheses in it is never handed to the filesystem — where it would always
/// miss, and every archive member would then be newer than everything.
fn prerequisite_timestamp(name: &Bytes) -> Result<Option<std::time::SystemTime>> {
    match crate::archive::split_archive_name(name) {
        Some((archive, member)) => Ok(crate::archive::member_timestamp(archive, member)),
        None => get_timestamp(name),
    }
}

/// The same question asked about the file being made rather than about one of
/// its prerequisites.
///
/// The two differ for an archive member and only for one: GNU Make marks it
/// `low_resolution_time` and rounds the date of the file it is updating up to
/// the end of its second, so a member filed in the same second as the object it
/// came from counts as current.
fn target_timestamp(name: &Bytes) -> Result<Option<std::time::SystemTime>> {
    match crate::archive::split_archive_name(name) {
        Some((archive, member)) => Ok(crate::archive::member_timestamp_as_target(archive, member)),
        None => get_timestamp(name),
    }
}

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
    /// `$%`, the archive member: the half of `lib.a(member.o)` inside the
    /// parentheses, and empty for every target that is not one.
    Percent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoCommandVariant {
    None,
    D,
    F,
}

/// Whether an automatic variable's expansion produced the value itself or a
/// reference the destination binds later.
///
/// The distinction only matters to the `D` and `F` forms: halving a name is
/// this expansion's work, and halving a reference is not work at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    /// The bytes written are the value.
    Now,
    /// The bytes written are a reference the destination substitutes.
    Late,
}

impl AutoCommand {
    /// The character GNU Make names this automatic variable with.
    fn name_char(&self) -> char {
        match self {
            AutoCommand::At => '@',
            AutoCommand::Less => '<',
            AutoCommand::Hat => '^',
            AutoCommand::Plus => '+',
            AutoCommand::Bar => '|',
            AutoCommand::Star => '*',
            AutoCommand::Question { .. } => '?',
            AutoCommand::Percent => '%',
        }
    }
}

impl AutoCommandVar {
    /// The makefile text GNU Make defined this automatic variable with, for the
    /// forms that were defined from text at all.
    ///
    /// The two kinds of automatic variable are not built the same way, and
    /// `$(value)` is where the difference becomes visible. GNU Make sets the
    /// base forms per file in `set_file_variables`, as simple variables whose
    /// value is the computed name, so there is no unexpanded text behind them
    /// and `$(value @)` reads back exactly what `$@` expands to — `None` here,
    /// leaving the caller to evaluate.
    ///
    /// The `D` and `F` forms are not computed at all. `define_automatic_variables`
    /// defines them once, at startup, as recursive variables whose text is a
    /// `dir`/`notdir` expression over the base form (`src/variable.c`). Reading
    /// one back therefore yields that expression rather than a directory or a
    /// file name: `$(value @D)` is `$(patsubst %/,%,$(dir $@))`, whatever the
    /// current target happens to be.
    pub fn definition(&self) -> Option<Bytes> {
        let base = self.typ.name_char();
        match self.variant {
            AutoCommandVariant::None => None,
            AutoCommandVariant::D => Some(Bytes::from(format!("$(patsubst %/,%,$(dir ${base}))"))),
            AutoCommandVariant::F => Some(Bytes::from(format!("$(notdir ${base})"))),
        }
    }

    /// Whether this is the base form — `$@`, `$?`, `$<` — rather than one of
    /// the `D`/`F` forms taken off it.
    ///
    /// The two are defined in different places in GNU Make and the difference
    /// is visible from `.VARIABLES`: `set_file_variables` binds the base forms
    /// in the FILE's own variable set as the recipe is prepared, while
    /// `define_automatic_variables` binds each `D` and `F` form once, at
    /// startup, in the global set. So the global name list holds `@D` and `@F`
    /// and never `@`.
    pub const fn is_base_form(&self) -> bool {
        matches!(self.variant, AutoCommandVariant::None)
    }

    /// How `$(flavor)` names this automatic variable.
    ///
    /// The same split: a base form was defined as a simple variable holding the
    /// computed name, and a `D`/`F` form as a recursive one holding the
    /// expression that computes it.
    pub fn flavor(&self) -> &'static str {
        match self.variant {
            AutoCommandVariant::None => "simple",
            AutoCommandVariant::D | AutoCommandVariant::F => "recursive",
        }
    }

    /// The reference this form leaves behind when the list it names is the
    /// destination's to bind rather than this expansion's to compute.
    const fn deferred_reference(&self) -> &'static [u8] {
        match self.variant {
            AutoCommandVariant::None => DEFERRED_NEW_INPUTS_REFERENCE,
            AutoCommandVariant::D => DEFERRED_NEW_INPUTS_DIRECTORIES_REFERENCE,
            AutoCommandVariant::F => DEFERRED_NEW_INPUTS_FILENAMES_REFERENCE,
        }
    }

    pub fn eval(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
        match self.variant {
            AutoCommandVariant::None => {
                self.eval_impl(ev, out)?;
            }
            AutoCommandVariant::D => {
                let mut buf = BytesMut::new();
                if self.eval_impl(ev, &mut buf)? == Bound::Late {
                    // The reference stands for the directory halves already.
                    // Halving it again would halve the placeholder.
                    out.put_slice(&buf);
                    return Ok(());
                }
                let buf = Bytes::from(buf);
                let mut ww = WordWriter::new(out);
                for tok in word_scanner(&buf) {
                    let tok = buf.slice_ref(tok);
                    ww.write(&dirname(&tok))
                }
            }
            AutoCommandVariant::F => {
                let mut buf = BytesMut::new();
                if self.eval_impl(ev, &mut buf)? == Bound::Late {
                    out.put_slice(&buf);
                    return Ok(());
                }
                let buf = Bytes::from(buf);
                let mut ww = WordWriter::new(out);
                for tok in word_scanner(&buf) {
                    ww.write(basename(tok))
                }
            }
        }
        Ok(())
    }

    fn eval_impl(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<Bound> {
        let current_dep_node = self.current_dep_node.lock();
        let current_dep_node = current_dep_node.as_ref().unwrap().lock();
        let names = &ev.session.symtab;

        match &self.typ {
            AutoCommand::At => {
                // For `lib.a(member.o)` the target is the archive and the
                // member is `$%`; `set_file_variables` splits them at the
                // first `(` (reference/gnumake/src/commands.c).
                let name = current_dep_node.recipe_output.as_bytes(names);
                match crate::archive::split_archive_name(&name) {
                    Some((archive, _)) => out.put_slice(&name.slice(..archive.len())),
                    None => out.put_slice(&name),
                }
            }
            AutoCommand::Percent => {
                let name = current_dep_node.recipe_output.as_bytes(names);
                if let Some((archive, _)) = crate::archive::split_archive_name(&name) {
                    out.put_slice(&name.slice(archive.len() + 1..name.len() - 1));
                }
            }
            AutoCommand::Less => {
                if let Some(ai) = current_dep_node.actual_inputs.first() {
                    out.put_slice(&crate::archive::member_or_whole(&ai.as_bytes(names)))
                }
            }
            AutoCommand::Hat => {
                let mut seen = HashSet::new();
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    if seen.insert(*ai) {
                        ww.write(&crate::archive::member_or_whole(&ai.as_bytes(names)))
                    }
                }
            }
            AutoCommand::Plus => {
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    ww.write(&crate::archive::member_or_whole(&ai.as_bytes(names)))
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
                // The implicit search records what its match read, because a
                // search that held a directory aside leaves a stem the pattern
                // and the target name cannot be made to yield between them.
                if let Some(stem) = &current_dep_node.stem {
                    out.put_slice(&stem.as_bytes(names));
                } else if let Some(output_pattern) = &current_dep_node.output_pattern {
                    let pat = Pattern::new(output_pattern.as_bytes(names));
                    // GNU Make sets the stem by substituting the target into a
                    // bare `%` rather than by reading the match out, and the
                    // two answers part company for a static pattern rule's
                    // target that missed the pattern: substitution leaves a
                    // name it could not match alone, so `$*` is the whole
                    // target rather than the empty string a non-match reads as.
                    out.put_slice(&pat.append_subst(
                        &current_dep_node.recipe_output.as_bytes(names),
                        &Bytes::from_static(b"%"),
                    ));
                } else {
                    // An explicit rule has no match to read a stem out of, and
                    // Unix Make's answer for one is the target name with a
                    // known suffix taken off. `set_file_variables`
                    // (src/commands.c:97) walks `.SUFFIXES` in the order the
                    // read left it and stops at the first entry the name ends
                    // with, so which of two suffixes is found is a question
                    // about the list's order rather than about which is
                    // longer, and a name the list does not reach reads as
                    // empty. The name has to be longer than the suffix, so a
                    // target named for the suffix itself has no stem either.
                    // An archive member's stem is read off the member name,
                    // not off the whole target: `lib.a(foo.o)` has `$*` of
                    // `foo` (src/commands.c, the same `ar_name` branch that
                    // sets `$@` and `$%`).
                    let whole = current_dep_node.recipe_output.as_bytes(names);
                    let name = match crate::archive::split_archive_name(&whole) {
                        Some((archive, _)) => whole.slice(archive.len() + 1..whole.len() - 1),
                        None => whole,
                    };
                    for suffix in &ev.session.suffixes {
                        if name.len() > suffix.len() && name.ends_with(suffix.as_ref()) {
                            out.put_slice(&name[..name.len() - suffix.len()]);
                            break;
                        }
                    }
                }
            }
            AutoCommand::Question {
                found_new_inputs,
                timing,
            } => {
                let mut seen: HashSet<Symbol> = HashSet::new();

                if *timing == NewInputsTiming::Launch {
                    // The recipe is being expanded at launch, so every
                    // prerequisite has settled and the comparison GNU Make
                    // makes here is the one the filesystem already answers.
                    let mut ww = WordWriter::new(out);
                    let target_age = ExecStatus::Timestamp(target_timestamp(
                        &current_dep_node.recipe_output.as_bytes(names),
                    )?);
                    for ai in current_dep_node.actual_inputs.iter() {
                        let ai_str = ai.as_bytes(names);
                        if seen.insert(*ai)
                            && ExecStatus::Timestamp(prerequisite_timestamp(&ai_str)?) > target_age
                        {
                            ww.write(&crate::archive::member_or_whole(&ai_str));
                        }
                    }
                } else if ev.avoid_io
                    && (*timing == NewInputsTiming::SchedulerBoundary
                        || current_dep_node.grouped_double_action.is_some())
                {
                    // The grouped action's comparison is deliberately made by
                    // the scheduler after its prerequisites finish.  It binds
                    // this value when the edge is launched; doing a second
                    // shell-side timestamp test would move the snapshot past
                    // the prerequisite boundary.
                    out.put_slice(self.deferred_reference());
                    *found_new_inputs.lock() = true;
                    return Ok(Bound::Late);
                } else if let Some(action) = &current_dep_node.grouped_double_action {
                    let mut oldest_member = None;
                    let mut missing_member = action.has_phony_member;
                    for member in &action.members {
                        match target_timestamp(&member.as_bytes(names))? {
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
                        let input_mtime = prerequisite_timestamp(&ai_str)?;
                        if seen.insert(*ai)
                            && (missing_member
                                || action.phony_inputs.contains(ai)
                                || input_mtime.is_none()
                                || oldest_member.is_some_and(|oldest| input_mtime > Some(oldest)))
                        {
                            ww.write(&crate::archive::member_or_whole(&ai_str));
                        }
                    }
                } else if ev.avoid_io {
                    let mut delayed = None;
                    // Check timestamps using the shell at the start of rule execution
                    // instead. The name here is a shell variable the prologue
                    // below exports, not one a destination binds, so this is
                    // deliberately the base reference under every form: a `D`
                    // or `F` name would be one the shell never heard of. The
                    // caller halves the reference text as it always did, which
                    // is wrong the way GNU Make counts it and is what a
                    // generated `build.ninja` has always said.
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
                    let target_age = ExecStatus::Timestamp(target_timestamp(
                        &current_dep_node.recipe_output.as_bytes(names),
                    )?);
                    for ai in current_dep_node.actual_inputs.iter() {
                        let ai_str = ai.as_bytes(names);
                        if seen.insert(*ai)
                            && ExecStatus::Timestamp(prerequisite_timestamp(&ai_str)?) > target_age
                        {
                            ww.write(&crate::archive::member_or_whole(&ai_str))
                        }
                    }
                }
            }
        }
        Ok(Bound::Now)
    }
}

impl Debug for AutoCommandVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AutoVar({:?})", self.sym)
    }
}

/// One static child invocation lifted out of a recipe line.
#[derive(Clone, Debug)]
pub struct LiftedInvocation {
    /// What the child is compiled from: the invocation alone, with any wrapper
    /// the shell would have reached through already taken off.
    pub command: Bytes,
    /// The value the `MAKE` reference in its command position produced.
    pub make: Bytes,
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
    /// The static child invocations this line names, in the order the shell
    /// would run them. Empty for a line that names none the compiler can lift.
    pub recursive_make: Vec<LiftedInvocation>,
    /// A recursion this compiler can see but cannot turn into a child
    /// compilation: the line is classified recursive and starts a Make process
    /// somewhere a shell begins a command, but not in a position
    /// [`invokes_make`] can lift out as one static invocation. `None` for
    /// every other line, recursive or not.
    ///
    /// A line like this carries no [`Self::recursive_make`], so a recipe that
    /// splits its other lines into child graphs would read this one as
    /// ordinary residual work and leave it to start a nested Make beside them.
    /// Naming it lets the compiler decline to split the recipe at all, which
    /// is the rule the split already claimed to follow.
    ///
    /// The reason travels with the fact because a report about the build has
    /// to say why this line nests, and working it out again later from the
    /// recipe text would be a second reading that could disagree with the one
    /// the build acted on.
    pub nesting: Option<crate::census::NestingReason>,
    /// Where this line was written, for a report that has to point at it. The
    /// rule's own location names where the target was defined, which is a
    /// different line as soon as a recipe has more than one.
    pub loc: Option<crate::loc::Loc>,
}

impl Command {
    /// Whether this line names a recursion the compiler could not lift out.
    #[must_use]
    pub const fn uncomposable_recursion(&self) -> bool {
        self.nesting.is_some()
    }
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
/// Whether expanding this recipe could reach a `$(MAKE)` reference.
///
/// kati classifies a recursive line by what the expansion *did*: expanding a
/// reference named `MAKE` inside a command records its value, and a line that
/// invokes one of those values is compiled as a child graph rather than run.
/// So the question a deferral has to answer before expanding anything is
/// whether the expansion could record one at all — which is the recipe's own
/// syntax tree, plus the syntax tree of every recursively expanded variable it
/// can reach through it. A simply expanded variable holds text rather than a
/// tree, and text cannot hold a reference, which is why the walk stops there
/// and why kati's own classification stops there too.
///
/// Conservative wherever a name is not knowable without expanding: a computed
/// reference, and the functions that reach a variable by a name they compute,
/// all answer yes.
pub fn expansion_can_reach_make(
    value: &Value,
    ev: &Evaluator,
    rule_vars: Option<&Vars>,
    seen: &mut HashSet<Symbol>,
) -> bool {
    match value {
        Value::Literal(_, _) => false,
        // Expanding it raises rather than producing text, so it reaches nothing.
        Value::Unreadable(_, _) => false,
        Value::SymRef(_, sym) => symbol_can_reach_make(*sym, ev, rule_vars, seen),
        Value::List(_, values) => values
            .iter()
            .any(|value| expansion_can_reach_make(value, ev, rule_vars, seen)),
        // The name is computed, so it can be `MAKE`.
        Value::VarRef(_, _) => true,
        Value::VarSubst {
            loc: _,
            name,
            pat,
            subst,
        } => {
            match name.as_ref() {
                Value::Literal(_, literal) => {
                    let Some(sym) = ev.session.symtab.peek_symbol(literal) else {
                        return false;
                    };
                    if symbol_can_reach_make(sym, ev, rule_vars, seen) {
                        return true;
                    }
                }
                _ => return true,
            }
            expansion_can_reach_make(pat, ev, rule_vars, seen)
                || expansion_can_reach_make(subst, ev, rule_vars, seen)
        }
        Value::Func { loc: _, fi, args } => {
            // `call` and `value` reach a variable by a name they are given
            // rather than by one written here, so what they reach cannot be
            // walked from this tree.
            if matches!(fi.name, b"call" | b"value") {
                return true;
            }
            // `eval` is walked like any other function. It expands to nothing,
            // so it contributes no text to the recipe; a `$(MAKE)` it could
            // reach is one written inside its own argument, and that is in this
            // tree. Answering `true` for it instead would keep every recipe
            // holding an `$(eval)` out of launch expansion — and when the
            // `$(eval)` is the point, as it is for a recipe-time assignment or
            // export, compiling it early is exactly what makes the timing
            // wrong: the write would land before the recipes that ran ahead of
            // it were expanded. A variable the `$(eval)` defines and a later
            // line then expands to `$(MAKE)` is beyond any static walk; the
            // check made after a deferred expansion catches that and refuses,
            // rather than handing the executor a nested Make.
            args.iter()
                .any(|arg| expansion_can_reach_make(arg, ev, rule_vars, seen))
        }
    }
}

/// Whether expanding a reference to `sym` could reach `$(MAKE)`.
fn symbol_can_reach_make(
    sym: Symbol,
    ev: &Evaluator,
    rule_vars: Option<&Vars>,
    seen: &mut HashSet<Symbol>,
) -> bool {
    if sym.as_bytes(&ev.session).as_ref() == b"MAKE" {
        return true;
    }
    if !seen.insert(sym) {
        return false;
    }
    let bound = rule_vars
        .and_then(|vars| vars.peek(sym))
        .or_else(|| ev.session.globals.peek(sym));
    let Some(bound) = bound else {
        return false;
    };
    let Some(definition) = bound.read().recursive_definition() else {
        return false;
    };
    expansion_can_reach_make(&definition, ev, rule_vars, seen)
}

/// Whether a recipe line as written can hold no command at all.
///
/// A recipe of nothing but whitespace has nothing to expand and nothing to
/// run: GNU Make reads it as a target with an empty recipe, which is remade by
/// doing nothing. Text is the only case that can be answered without expanding,
/// and it is the case Makefiles write — `all:;` — so it is worth answering.
pub fn is_blank_recipe_line(value: &Value) -> bool {
    match value {
        Value::Literal(_, text) => text.trim_ascii().is_empty(),
        Value::List(_, values) => values.iter().all(|value| is_blank_recipe_line(value)),
        _ => false,
    }
}

/// Whether a recipe as written can reach the `$?` automatic variable.
///
/// Deliberately conservative in both directions a scan can be wrong about: a
/// computed reference is counted, because its name is not knowable here, and
/// the `D` and `F` variants are counted by name. A recipe this answers `true`
/// for is expanded while the graph is constructed, so the scheduler binds the
/// value and any `filter-out` around it is seen.
pub fn references_new_inputs(value: &Value, names: &impl Interner) -> bool {
    match value {
        Value::Literal(_, _) => false,
        // Expanding it raises rather than producing text, so it reaches nothing.
        Value::Unreadable(_, _) => false,
        Value::SymRef(_, sym) => {
            matches!(sym.as_bytes(names).as_ref(), b"?" | b"?D" | b"?F")
        }
        Value::List(_, values) => values
            .iter()
            .any(|value| references_new_inputs(value, names)),
        // A name computed at expansion time can be `?`, and nothing here can
        // rule that out.
        Value::VarRef(_, _) => true,
        Value::VarSubst {
            loc: _,
            name,
            pat,
            subst,
        } => {
            references_new_inputs(name, names)
                || references_new_inputs(pat, names)
                || references_new_inputs(subst, names)
        }
        Value::Func {
            loc: _,
            fi: _,
            args,
        } => args.iter().any(|arg| references_new_inputs(arg, names)),
    }
}

fn references_make(value: &Value, names: &impl Interner) -> bool {
    match value {
        Value::Literal(_, _) => false,
        // Expanding it raises rather than producing text, so it reaches nothing.
        Value::Unreadable(_, _) => false,
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

/// The shell words that stand in front of a command without being one.
///
/// A segment beginning with one of these has a command after it, in the
/// position the shell reads as a command position: `then make x` starts a
/// Make and `echo then make x` does not, and the difference is that `then` is
/// a reserved word only where a command may begin — which, after a `;`, is
/// exactly where [`command_segments`] has just cut.
///
/// `fi`, `done`, `esac` and `elif`'s closing siblings are absent on purpose:
/// nothing follows them but the end of the construct.
const COMMAND_PRECEDERS: [&[u8]; 11] = [
    b"if", b"then", b"elif", b"else", b"while", b"until", b"do", b"!", b"time", b"{", b"(",
];

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
        // A leading `VAR=value` sequence, `env` or `exec` before the program,
        // and the reserved words a construct puts in front of a command are
        // all the shell's ways of saying the same command differently.
        while let Some(word) = segment
            .split(|byte| byte.is_ascii_whitespace())
            .next()
            .filter(|word| !word.is_empty())
            .filter(|word| {
                *word == b"env"
                    || *word == b"exec"
                    // A recipe line written across several source lines keeps
                    // its continuations, and a `\` alone in front of the
                    // newline is one: the command after it begins where the
                    // shell says a command begins.
                    || *word == b"\\"
                    || COMMAND_PRECEDERS.contains(word)
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

/// One command with the subshell a whole recipe line sits inside taken off.
///
/// A line that is nothing but a parenthesized sequence *is* that sequence. The
/// parentheses keep a directory change and an environment from reaching the
/// rest of the script, and a line holding only the sequence has no rest of the
/// script for them to reach: GNU Make hands the line to a shell of its own
/// either way. vim's top-level Makefile writes `(cd runtime/indent && $(MAKE)
/// clean)`, and what that starts is the same child, with the same goals, in
/// the same directory, as the line without its parentheses.
///
/// Only when the wrapper closes at the very end. `(a) && (b)` is two commands
/// and opens on the first byte, so the match has to be found rather than
/// assumed.
///
/// A brace group is the same wrapper written the other way and is taken off
/// too, but it costs more to recognise: `(` is always the operator, while `{`
/// is a reserved word and so opens a group only where a command may begin and
/// only as a word of its own. `echo a{b}` is one word and `{ a; }` is a group,
/// and reading either as the other changes what the line means.
pub fn unwrapped_command(command: &[u8]) -> &[u8] {
    let mut command = command.trim_ascii();
    while let Some(body) = subshell_body(command).or_else(|| brace_group_body(command)) {
        command = body.trim_ascii();
        // The `;` a sequence may end with belongs to the sequence, and with
        // the wrapper gone there is nothing left for it to separate. A brace
        // group must be closed by one, so this is where that one goes.
        command = command
            .strip_suffix(b";")
            .unwrap_or(command)
            .trim_ascii_end();
    }
    command
}

/// What a leading `{` encloses, when what it encloses is the whole command.
///
/// `{` and `}` are reserved words rather than operators, so each is read only
/// where a command may begin: that is what tells the group in `{ cd a; }` from
/// the brace in `echo a}`, and it is why a byte test would be wrong here where
/// it is right for `(`.
fn brace_group_body(command: &[u8]) -> Option<&[u8]> {
    // A reserved word is a word: `{cd a; }` runs a program called `{cd`.
    if !command.starts_with(b"{") || !command.get(1).is_some_and(u8::is_ascii_whitespace) {
        return None;
    }
    let mut depth = 0usize;
    let mut quote = None;
    // Whether what comes next would begin a command, which is the only place a
    // reserved word is one.
    let mut command_position = true;
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
                b'\'' | b'"' => {
                    quote = Some(byte);
                    command_position = false;
                }
                b'\\' => {
                    index += 1;
                    command_position = false;
                }
                b'{' if command_position => depth += 1,
                b'}' if command_position => {
                    depth -= 1;
                    if depth == 0 {
                        return (index + 1 == command.len()).then(|| &command[1..index]);
                    }
                }
                b';' | b'&' | b'|' | b'(' | b'\n' => command_position = true,
                byte if byte.is_ascii_whitespace() => {}
                _ => command_position = false,
            },
        }
        index += 1;
    }
    None
}

/// What a leading `(` encloses, when what it encloses is the whole command.
fn subshell_body(command: &[u8]) -> Option<&[u8]> {
    if !command.starts_with(b"(") {
        return None;
    }
    let mut depth = 0usize;
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
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return (index + 1 == command.len()).then(|| &command[1..index]);
                    }
                }
                _ => {}
            },
        }
        index += 1;
    }
    None
}

/// The static child invocations one recipe line names, in the order the shell
/// would run them, and empty for a line naming none the compiler can lift.
///
/// Two shapes are lifted, and the second is what a line naming more than one
/// child looks like. A line that IS an invocation is that invocation, with the
/// `cd` a child is entered through still on it, because the resolver reads
/// that. A line that is nothing but invocations joined by `&&` names each of
/// them: `&&` runs the next only when the last one won and in that order,
/// which is exactly what one child graph ordered after another does, so the
/// two describe the same build. No other joiner does — `;` runs the next
/// whatever the last one did, and `||` runs it only when the last one lost —
/// so a line holding one is left to run as written.
pub fn lifted_invocations(line: &Bytes, make_values: &[Bytes]) -> Vec<LiftedInvocation> {
    let lifted = |text: &[u8]| {
        let command = unwrapped_command(text);
        make_values
            .iter()
            .find(|make| invokes_make(command, make))
            .map(|make| LiftedInvocation {
                command: line.slice_ref(command),
                make: make.clone(),
            })
    };
    let whole = unwrapped_command(line);
    let conjuncts = conjuncts(whole);
    if conjuncts.len() > 1
        && let Some(invocations) = conjuncts
            .iter()
            .map(|conjunct| lifted(conjunct))
            .collect::<Option<Vec<_>>>()
    {
        return invocations;
    }
    lifted(whole).into_iter().collect()
}

/// Why a line that starts a nested Make was not lifted out as one child.
///
/// Asked where the decision is made and answered from the shape the line
/// actually has, so a report about the build carries the compiler's own reason
/// rather than a second reading of the same recipe.
///
/// The two answers are the two sides of what [`invokes_make`] refuses. A line
/// with more than one place a shell begins a command has a construct standing
/// between it and the invocation — an `if`, a `;`, a `||`, a pipeline — and
/// what the compiler lifts is a line that IS an invocation, not a line that
/// holds one somewhere. A line with one such place was refused for what is
/// written in the command position instead: an assignment or `env` prefix, a
/// redirection, a glob, an expansion the resolver will not read as an
/// argument list.
#[must_use]
pub fn nesting_reason(line: &[u8]) -> crate::census::NestingReason {
    if command_segments(unwrapped_command(line)).len() > 1 {
        crate::census::NestingReason::ThroughAConstruct
    } else {
        crate::census::NestingReason::NotAnArgumentList
    }
}

/// Split a command where a shell would run the next part only if this one won.
///
/// Quoting, escapes and wrappers are respected, so the `&&` inside `(a && b)`
/// or `"x && y"` is not a place this splits: what is wanted is the joints of
/// the line itself, not of everything written on it.
fn conjuncts(command: &[u8]) -> Vec<&[u8]> {
    let mut conjuncts = Vec::new();
    let mut start = 0;
    let mut parens = 0usize;
    let mut braces = 0usize;
    let mut quote = None;
    let mut command_position = true;
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
                b'\'' | b'"' => {
                    quote = Some(byte);
                    command_position = false;
                }
                b'\\' => {
                    index += 1;
                    command_position = false;
                }
                b'(' => {
                    parens += 1;
                    command_position = true;
                }
                b')' => {
                    parens = parens.saturating_sub(1);
                    command_position = false;
                }
                b'{' if command_position => braces += 1,
                b'}' if command_position => braces = braces.saturating_sub(1),
                b'&' if parens == 0 && braces == 0 && command.get(index + 1) == Some(&b'&') => {
                    conjuncts.push(&command[start..index]);
                    index += 1;
                    start = index + 1;
                    command_position = true;
                }
                b';' | b'&' | b'|' | b'\n' => command_position = true,
                byte if byte.is_ascii_whitespace() => {}
                _ => command_position = false,
            },
        }
        index += 1;
    }
    conjuncts.push(&command[start..]);
    conjuncts
}

/// Whether one expanded `MAKE` value occupies the command position of a
/// command the resolver can read as the argument list it is.
///
/// Two questions in one, because separating them would leave a gap that ends
/// a run. The recogniser says a line is a child invocation and the resolver
/// then splits it into the words the nested process would have received as
/// argv — and a line the recogniser claims and the resolver cannot read is a
/// build that stops, where GNU Make would have handed the line to a shell.
/// So what is lifted is a subset of what the resolver reads: ordinary bytes,
/// quotes, backslash escapes and command substitutions are an argument list;
/// every other shell operator is a program, and a line holding one runs as
/// the line it is.
fn invokes_make(command: &[u8], make: &[u8]) -> bool {
    let mut command = unwrapped_command(command);
    // `cd DIR && make …` selects where the child is read, and the resolver
    // takes the directory off itself, so it travels with the invocation.
    if let [entered, invocation] = conjuncts(command).as_slice()
        && starts_with_word(entered.trim_ascii_start(), b"cd")
        && is_argument_list(entered)
    {
        command = invocation.trim_ascii();
    }
    if starts_with_word(command, b"exec") {
        command = command[b"exec".len()..].trim_ascii_start();
    }
    starts_with_word(command, make) && is_argument_list(command)
}

/// Whether a command is words and nothing else — no operator that would make
/// it a program rather than one invocation.
///
/// The rejected set is the resolver's: `src/make/cli/subninja.rs` splits these
/// same bytes into argv and refuses everything here refuses, plus a `$` that
/// does not open a command substitution.
fn is_argument_list(command: &[u8]) -> bool {
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
                b'$' if command.get(index + 1) == Some(&b'(') => {
                    let Some(end) = substitution_end(command, index + 2) else {
                        return false;
                    };
                    index = end;
                }
                b'|' | b'&' | b';' | b'<' | b'>' | b'(' | b')' | b'{' | b'}' | b'$' | b'`'
                | b'*' | b'?' | b'[' | b']' | b'~' | b'#' | b'\n' => return false,
                _ => {}
            },
        }
        index += 1;
    }
    quote.is_none()
}

/// Where the command substitution opened before `start` closes.
fn substitution_end(command: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut quote = None;
    let mut index = start;
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
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(index);
                    }
                }
                _ => {}
            },
        }
        index += 1;
    }
    None
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

/// What the `@`, `-` and `+` on one expanded recipe line say about it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct LinePrefixes {
    echo: bool,
    dash_prefixed: bool,
    recursive_line: bool,
}

/// Read the prefixes off a recipe line as it is written, before expansion.
///
/// GNU Make's `chop_commands` runs at parse time over the unexpanded text and
/// stores what it finds in the written line's `lines_flags`, which then seeds
/// every line the expansion produces. So a prefix written in front of a
/// `$(call …)` belongs to the whole call, while a prefix that came *out* of the
/// call binds to its own line. kati's own `silent_multiline.mk` is the case
/// that tells them apart against 4.4.1: `$(call cmd2)` and `@$(call cmd2)`
/// expand to the same three lines, and Make echoes the trailing `echo bar` for
/// the first and not for the second.
///
/// Returns whether the scan reached the end of what it read without meeting
/// anything that is not a prefix — which is when the next piece of the line
/// still counts as its beginning. A reference ends the run: GNU Make's scan
/// stops at the `$` that starts one.
fn scan_written_prefixes(value: &Value, prefixes: &mut LinePrefixes) -> bool {
    match value {
        Value::Literal(_, text) => parse_command_prefixes(
            text.clone(),
            &mut prefixes.echo,
            &mut prefixes.dash_prefixed,
            &mut prefixes.recursive_line,
        )
        .is_empty(),
        Value::List(_, values) => values
            .iter()
            .all(|value| scan_written_prefixes(value, prefixes)),
        _ => false,
    }
}

/// The lines one written recipe line expanded into, each carrying the prefixes
/// that apply to it.
///
/// GNU Make expands a written recipe line once and then hands the result to
/// `start_job_command` one physical line at a time — `construct_command_argv`
/// stops at the newline and leaves `command_ptr` on the next. That function
/// reads `@`, `-` and `+` off the line in front of it, seeding them afresh from
/// the written line's own flags every time, so a prefix produced by the
/// expansion belongs to the expanded line it stands on rather than to the
/// expansion. Probed against 4.4.1: `define multi / @echo hi / echo there /
/// endef` run from one written line prints `hi`, `echo there`, `there` — Make
/// echoes the second line, so the first line's `@` never reached it.
///
/// `+` is the exception, and it is GNU Make's exception rather than one of
/// ours: the same function writes `COMMANDS_RECURSE` back into the written
/// line's flags, under a comment in job.c admitting this marks more lines
/// recursive than it should for exactly this shape. So a `+` reaches the rest
/// of this expansion and stops at the end of it. Probed: `+touch a` above
/// `touch b` in one expansion makes both files under `-n`, and the same two
/// lines written out make only the first.
struct ExpandedRecipeLines {
    rest: Bytes,
    /// What the written line already settled for every line of its expansion:
    /// `-s` and any prefix written before the expansion for the echo and the
    /// forgiveness, and those plus the unexpanded `$(MAKE)` scan for the
    /// recursion.
    written: LinePrefixes,
}

impl ExpandedRecipeLines {
    fn new(expansion: Bytes, written: LinePrefixes) -> Self {
        Self {
            rest: expansion,
            written,
        }
    }
}

impl Iterator for ExpandedRecipeLines {
    type Item = (Bytes, LinePrefixes);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let eol = find_end_of_line(&self.rest);
        let line = eol.line.slice_ref(trim_left_space(&eol.line));
        self.rest = eol.rest;
        let mut prefixes = self.written;
        let command = parse_command_prefixes(
            line,
            &mut prefixes.echo,
            &mut prefixes.dash_prefixed,
            &mut prefixes.recursive_line,
        );
        // A line that was nothing but a `+` still carries it, so the write-back
        // happens before the caller decides there is no command here.
        self.written.recursive_line = prefixes.recursive_line;
        Some((command, prefixes))
    }
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
        file_evaluation: FileEvaluation,
        output_evaluation: OutputEvaluation,
    ) -> Result<Self> {
        ev.new_inputs_timing = new_inputs_timing;
        ev.shell_evaluation = shell_evaluation;
        ev.file_evaluation = file_evaluation;
        ev.output_evaluation = output_evaluation;
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
        ret.register_autocommand('%', AutoCommand::Percent)?;
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

    // [spec:ronin:req:make.recursive-invocation+2]
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
        self.ev.session.ground_journal.suspend(true);
        *self.current_dep_node.lock() = Some(n.clone());
        *self.found_new_inputs.lock() = false;
        self.ev.deferred_new_inputs_filter_out.clear();
        for v in node_cmds {
            self.ev.loc = v.loc();
            self.ev.expanded_make_in_command.clear();
            let cmds_buf = v.eval_to_buf(self.ev)?;
            let make_values = self.ev.expanded_make_in_command.clone();
            // `-i` and `.IGNORE` say a failure does not count, which is what
            // the `-` prefix says too — but only the prefix also relaxes the
            // shell, so the two are carried separately and joined at the end.
            let ignored_without_prefix = self.ev.session.flags.ignore_errors || node_ignores_errors;
            let mut written = LinePrefixes {
                echo: !self.ev.session.flags.is_silent_mode,
                dash_prefixed: false,
                // The classification is read from the recipe as written, so it
                // has to be taken before anything is expanded away.
                recursive_line: references_make(&v, &self.ev.session),
            };
            scan_written_prefixes(&v, &mut written);
            let lines = ExpandedRecipeLines::new(cmds_buf, written);
            for (cmd, prefixes) in lines {
                if !cmd.is_empty() {
                    let recursive_make = lifted_invocations(&cmd, &make_values);
                    // Only a classified line is held to this. A `MAKE`-valued
                    // variable that GNU Make never classified is composed when
                    // the expansion makes that possible and otherwise left as
                    // written, exactly as 4.4.1 leaves it.
                    let nesting = (prefixes.recursive_line
                        && recursive_make.is_empty()
                        && make_values.iter().any(|make| spawns_make(&cmd, make)))
                    .then(|| nesting_reason(&cmd));
                    let shell_flag = self.ev.get_shell_flag(prefixes.dash_prefixed)?;
                    result.push(Command {
                        output: n.lock().recipe_output,
                        cmd,
                        echo: prefixes.echo,
                        ignore_error: ignored_without_prefix || prefixes.dash_prefixed,
                        dash_prefixed: prefixes.dash_prefixed,
                        shell_flag,
                        force_no_subshell: false,
                        recursive_line: prefixes.recursive_line,
                        recursive_make,
                        nesting,
                        loc: self.ev.loc.clone(),
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
                    nesting: None,
                    loc: node.loc.clone(),
                })
            }
            // Prepend |output_commands|.
            std::mem::swap(&mut result, &mut output_commands);
            result.extend(output_commands);
            self.ev.delayed_output_commands.clear();
        }

        self.ev.current_scope = None;
        self.ev.is_evaluating_command = false;
        self.ev.session.ground_journal.suspend(false);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AutoCommand, AutoCommandVar, AutoCommandVariant, ExpandedRecipeLines, LinePrefixes,
        invokes_make, lifted_invocations, nesting_reason, references_make, scan_written_prefixes,
        spawns_make, unwrapped_command,
    };
    use crate::expr::{ParseExprOpt, parse_expr};
    use crate::loc::Loc;
    use crate::session::Session;
    use bytes::Bytes;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// An automatic variable named `c`, in the form the `variant` selects.
    fn automatic(typ: AutoCommand, variant: AutoCommandVariant) -> AutoCommandVar {
        let mut session = Session::new();
        let sym = session.intern(typ.name_char().to_string());
        AutoCommandVar {
            typ,
            sym,
            variant,
            current_dep_node: Arc::new(Mutex::new(None)),
        }
    }

    /// GNU Make builds the two kinds of automatic variable differently, and
    /// `$(value)` is where that becomes visible. A base form is set per file to
    /// the name just computed, so there is no text behind it to read back. A
    /// `D` or `F` form was defined once from a `dir`/`notdir` expression over
    /// the base form, and reading it back yields that expression.
    ///
    /// Probed against GNU Make 4.4.1, in a recipe for a target `sub/a.o`:
    /// `$(value @)` is `sub/a.o` while `$(value @D)` is
    /// `$(patsubst %/,%,$(dir $@))` and `$(value @F)` is `$(notdir $@)`.
    #[test]
    fn only_the_path_forms_of_an_automatic_variable_have_a_definition() {
        let base = automatic(AutoCommand::At, AutoCommandVariant::None);
        assert_eq!(base.definition(), None);
        assert_eq!(base.flavor(), "simple");

        let directory = automatic(AutoCommand::At, AutoCommandVariant::D);
        assert_eq!(
            directory.definition().unwrap(),
            "$(patsubst %/,%,$(dir $@))"
        );
        assert_eq!(directory.flavor(), "recursive");

        let file = automatic(AutoCommand::At, AutoCommandVariant::F);
        assert_eq!(file.definition().unwrap(), "$(notdir $@)");
        assert_eq!(file.flavor(), "recursive");
    }

    /// The expression names whichever base form the `D` or `F` was derived
    /// from, so every automatic variable that has one reads back its own.
    #[test]
    fn a_path_form_reads_back_the_base_form_it_was_derived_from() {
        for (typ, base) in [
            (AutoCommand::Less, '<'),
            (AutoCommand::Hat, '^'),
            (AutoCommand::Plus, '+'),
            (AutoCommand::Star, '*'),
        ] {
            assert_eq!(
                automatic(typ.clone(), AutoCommandVariant::D)
                    .definition()
                    .unwrap(),
                format!("$(patsubst %/,%,$(dir ${base}))")
            );
            assert_eq!(
                automatic(typ, AutoCommandVariant::F).definition().unwrap(),
                format!("$(notdir ${base})")
            );
        }
    }

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

    /// The subshell vim's top-level Makefile writes its recursion inside, and
    /// the shapes that look like it and are not one command.
    #[test]
    fn a_subshell_around_the_whole_line_is_not_a_command() {
        assert_eq!(
            unwrapped_command(b"(cd sub && make child)"),
            b"cd sub && make child"
        );
        assert_eq!(
            unwrapped_command(b"  ( ( make child ; ) )  "),
            b"make child"
        );
        assert_eq!(
            unwrapped_command(b"(cd a && make b) && (cd c && make d)"),
            b"(cd a && make b) && (cd c && make d)"
        );
        assert_eq!(
            unwrapped_command(b"(cd a && make b) # )"),
            b"(cd a && make b) # )"
        );
        assert_eq!(unwrapped_command(b"echo '(a)'"), b"echo '(a)'");
        // Unbalanced, so there is no body to find and the line stands as
        // written for the shell to complain about.
        assert_eq!(
            unwrapped_command(b"(cd sub && make child"),
            b"(cd sub && make child"
        );

        assert!(invokes_make(b"(cd sub && make child)", b"make"));
        assert!(invokes_make(b"(make child)", b"make"));
        assert!(!invokes_make(b"(echo make)", b"make"));
        assert!(!invokes_make(
            b"(cd a && make b); (cd c && make d)",
            b"make"
        ));
    }

    /// A brace group is the subshell written the other way, and `{` is a word
    /// rather than an operator, so what tells the two apart is where it sits.
    #[test]
    fn a_brace_group_around_the_whole_line_is_not_a_command() {
        assert_eq!(
            unwrapped_command(b"{ cd sub && make child; }"),
            b"cd sub && make child"
        );
        assert_eq!(unwrapped_command(b"{ ( make child ; ) ; }"), b"make child");
        // `{` opens a group only as a word of its own, and only where a
        // command may begin.
        assert_eq!(
            unwrapped_command(b"{cd sub && make child; }"),
            b"{cd sub && make child; }"
        );
        assert_eq!(unwrapped_command(b"echo a{b}"), b"echo a{b}");
        assert_eq!(unwrapped_command(b"echo '{ a; }'"), b"echo '{ a; }'");
        // Two groups, so the first `}` is not the line's own close.
        assert_eq!(
            unwrapped_command(b"{ make a; } && { make b; }"),
            b"{ make a; } && { make b; }"
        );

        assert!(invokes_make(b"{ cd sub && make child; }", b"make"));
        assert!(!invokes_make(b"{ echo make; }", b"make"));
    }

    /// A line that is nothing but invocations joined by `&&` names each of
    /// them, and a line that merely holds one somewhere names none.
    #[test]
    fn a_conjunction_of_invocations_names_each_of_them() {
        let lifted = |line: &'static [u8]| {
            lifted_invocations(&Bytes::from_static(line), &[Bytes::from_static(b"make")])
                .into_iter()
                .map(|invocation| invocation.command)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            lifted(b"make -C a && make -C b"),
            [&b"make -C a"[..], b"make -C b"]
        );
        assert_eq!(
            lifted(b"(cd a && make) && { cd b && make; }"),
            [&b"cd a && make"[..], b"cd b && make"]
        );
        // One invocation, and the `cd` in front of it belongs to it.
        assert_eq!(
            lifted(b"cd sub && make child"),
            [&b"cd sub && make child"[..]]
        );
        // A joiner that is not `&&` does not order two children: `;` runs the
        // next whatever the last one did.
        assert!(lifted(b"make -C a ; make -C b").is_empty());
        // Work beside the invocation that the child graph cannot carry.
        assert!(lifted(b"cd a && make && echo done").is_empty());
        assert!(lifted(b"make -C a > log").is_empty());
        assert!(lifted(b"echo make && make -C b").is_empty());
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

    /// Read one written line's expansion the way `eval` does: not silent, and
    /// not classified recursive by the `$(MAKE)` scan over the written text.
    fn expanded(text: &'static [u8]) -> Vec<(String, bool, bool, bool)> {
        ExpandedRecipeLines::new(
            Bytes::from_static(text),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false,
            },
        )
        .map(|(cmd, prefixes)| {
            (
                String::from_utf8_lossy(&cmd).into_owned(),
                prefixes.echo,
                prefixes.dash_prefixed,
                prefixes.recursive_line,
            )
        })
        .collect()
    }

    /// GNU Make 4.4.1 with `define multi / @echo hi / echo there / endef` and
    /// `all: ; $(multi)` prints `hi`, `echo there`, `there` — the second line is
    /// echoed, so the first line's `@` did not reach it. `-` is the same shape
    /// and the difference is a build rather than an echo: `-false` above a bare
    /// `false` makes GNU ignore the first failure and stop on the second.
    #[test]
    fn silence_and_forgiveness_belong_to_the_expanded_line_they_are_written_on() {
        assert_eq!(
            expanded(b"@echo hi\necho there"),
            vec![
                ("echo hi".to_owned(), false, false, false),
                ("echo there".to_owned(), true, false, false),
            ]
        );
        assert_eq!(
            expanded(b"-false\nfalse"),
            vec![
                ("false".to_owned(), true, true, false),
                ("false".to_owned(), true, false, false),
            ]
        );
        // Both at once, and in either order, still bind to their own line.
        assert_eq!(
            expanded(b"@-one\ntwo\n-@three"),
            vec![
                ("one".to_owned(), false, true, false),
                ("two".to_owned(), true, false, false),
                ("three".to_owned(), false, true, false),
            ]
        );
    }

    /// `+` is GNU Make's own exception: `start_job_command` writes
    /// `COMMANDS_RECURSE` back into the written line's flags, so it reaches the
    /// rest of this expansion. Probed against 4.4.1 under `-n`, where a
    /// recursive line runs and an ordinary one does not: `+touch plus.out`
    /// above `touch noplus.out` in one expansion makes both files, and the same
    /// two lines written as two recipe lines make only `plus.out`.
    #[test]
    fn a_plus_reaches_the_rest_of_the_expansion_it_is_written_in() {
        assert_eq!(
            expanded(b"+touch plus.out\ntouch noplus.out"),
            vec![
                ("touch plus.out".to_owned(), true, false, true),
                ("touch noplus.out".to_owned(), true, false, true),
            ]
        );
        // A line that is nothing but the prefix still carries it, and leaves
        // no command behind for the caller to run.
        assert_eq!(
            expanded(b"+\ntouch after.out"),
            vec![
                (String::new(), true, false, true),
                ("touch after.out".to_owned(), true, false, true),
            ]
        );
        // Nothing carries backwards: a `+` below a line does not reach it.
        assert_eq!(
            expanded(b"touch before.out\n+touch plus.out"),
            vec![
                ("touch before.out".to_owned(), true, false, false),
                ("touch plus.out".to_owned(), true, false, true),
            ]
        );
    }

    /// Parse one recipe line as the makefile writes it, and read the prefixes
    /// `chop_commands` would take off it.
    fn written(text: &str) -> LinePrefixes {
        let mut session = Session::new();
        let value = parse_expr(
            &mut session,
            &mut Loc::default(),
            Bytes::from(text.to_owned()),
            ParseExprOpt::Command,
        )
        .expect("a parsable recipe line");
        let mut prefixes = LinePrefixes {
            echo: true,
            dash_prefixed: false,
            recursive_line: false,
        };
        scan_written_prefixes(&value, &mut prefixes);
        prefixes
    }

    /// A prefix written in front of an expansion belongs to every line the
    /// expansion produces, because GNU Make reads it at parse time and stores
    /// it on the written line. kati's own `silent_multiline.mk` is what tells
    /// this apart from a prefix the expansion produced: `$(call cmd2)` and
    /// `@$(call cmd2)` expand to the same three lines, and 4.4.1 echoes the
    /// trailing `echo bar` for the first and not for the second.
    #[test]
    fn a_prefix_written_before_an_expansion_belongs_to_the_whole_expansion() {
        assert_eq!(
            written("@$(call cmd)"),
            LinePrefixes {
                echo: false,
                dash_prefixed: false,
                recursive_line: false
            }
        );
        assert_eq!(
            written("\t-+@$(call cmd)"),
            LinePrefixes {
                echo: false,
                dash_prefixed: true,
                recursive_line: true
            }
        );
        // The scan stops at the `$`, so a prefix the expansion carries is not
        // this line's — it is read again, per line, once the text exists.
        assert_eq!(
            written("$(call cmd)"),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false
            }
        );
        // A reference is not a prefix character even where its value is one.
        assert_eq!(
            written("$(AT)echo hi"),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false
            }
        );
        assert_eq!(
            written("echo @ - +"),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false
            }
        );
    }

    /// The seeds are what the written line already settled: `-s` silences every
    /// expanded line without any of them saying so, and a written line the
    /// `$(MAKE)` scan classified is recursive throughout.
    #[test]
    fn the_written_lines_own_flags_seed_every_line_of_its_expansion() {
        let seeded = ExpandedRecipeLines::new(
            Bytes::from_static(b"one\ntwo"),
            LinePrefixes {
                echo: false,
                dash_prefixed: true,
                recursive_line: true,
            },
        )
        .map(|(_, prefixes)| prefixes)
        .collect::<Vec<_>>();
        assert_eq!(
            seeded,
            vec![
                LinePrefixes {
                    echo: false,
                    dash_prefixed: true,
                    recursive_line: true
                };
                2
            ]
        );
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

    /// The reason a line nests is the reason a report gives for it, so it has
    /// to tell the two apart: a construct standing between the line and the
    /// invocation, against a line that is one command written in a way the
    /// resolver will not read as an argument list.
    #[test]
    fn a_nesting_line_says_why_it_nests() {
        use crate::census::NestingReason::{NotAnArgumentList, ThroughAConstruct};

        assert_eq!(
            nesting_reason(b"if test x = y; then make sub; fi"),
            ThroughAConstruct
        );
        assert_eq!(
            nesting_reason(b"test -d sub && make -C sub"),
            ThroughAConstruct
        );
        assert_eq!(nesting_reason(b"true || make fallback"), ThroughAConstruct);
        assert_eq!(nesting_reason(b"cd sub; make child"), ThroughAConstruct);

        assert_eq!(nesting_reason(b"V=1 make child"), NotAnArgumentList);
        assert_eq!(nesting_reason(b"env V=1 make child"), NotAnArgumentList);
        assert_eq!(nesting_reason(b"make child > log"), NotAnArgumentList);
        assert_eq!(nesting_reason(b"make child *.o"), NotAnArgumentList);
    }
}
