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

use crate::fasthash::{FastMap, FastSet};
use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;
use std::{collections::HashSet, fmt::Debug, sync::Arc};

use crate::{
    build_sink::{
        FileEvaluation, NewInputsTiming, OutputEvaluation, SettledName, SettledNameView,
        ShellEvaluation,
    },
    dep::DepNode,
    eval::Evaluator,
    exec::ExecStatus,
    expr::{Value, ValueArena, ValueId},
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

/// What a settled-name reference is spelt with, before the number that tells
/// one prerequisite's from another's.
///
/// A recipe the compiler has to read for itself — a recursive `$(MAKE)`, an
/// automatic depfile, a grouped `::` action, a `$?` whose value the scheduler
/// binds — is expanded while the graph is being built, which is before the
/// build has decided whether a prerequisite the directory search answered
/// about has to be remade. Until that is decided the prerequisite has two
/// names and neither can be written down, so the recipe carries one of these
/// and the destination substitutes the spelling it settled on. Every other
/// prerequisite is written the way it always was.
const SETTLED_NAME_PREFIX: &[u8] = b"KATI_SETTLED_";

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

    /// Which part of a name this form reads.
    const fn settled_view(&self) -> SettledNameView {
        match self.variant {
            AutoCommandVariant::None => SettledNameView::Whole,
            AutoCommandVariant::D => SettledNameView::Directory,
            AutoCommandVariant::F => SettledNameView::Filename,
        }
    }

    /// This form's answer for one name that is where it is written.
    ///
    /// The same halving [`Self::eval`] does over the whole value, done a word
    /// at a time because the value this one is part of holds references as
    /// well as names, and a reference has no halves to take.
    fn viewed(&self, name: &Bytes) -> Bytes {
        match self.variant {
            AutoCommandVariant::None => name.clone(),
            AutoCommandVariant::D => dirname(name),
            AutoCommandVariant::F => name.slice_ref(basename(name)),
        }
    }

    /// A reference for each prerequisite this expansion is about to name that
    /// the build has not settled the spelling of yet.
    ///
    /// Empty unless the destination binds late values — a manifest cannot, so
    /// nothing crosses into one — and empty for every recipe whose
    /// prerequisites are all where they are written, which is nearly every one.
    /// One reference per prerequisite and form, reused where the same recipe
    /// reads the same prerequisite twice.
    fn settled_references(
        &self,
        ev: &mut Evaluator,
        node: &Arc<Mutex<DepNode>>,
        words: &[Symbol],
    ) -> FastMap<Symbol, Bytes> {
        let mut references = FastMap::default();
        if ev.new_inputs_timing != NewInputsTiming::SchedulerBoundary || ev.function_depth > 0 {
            return references;
        }
        let view = self.settled_view();
        // `$@` is the one automatic variable that names the TARGET, and for a
        // `::` action the target's name is settled by the walk rather than by
        // the read: an entry running after a current one writes the found path.
        // So it leaves a reference too, and is minted first so a recipe that
        // names both the target and a searched prerequisite numbers them in the
        // order it reads them.
        if matches!(self.typ, AutoCommand::At) {
            let target = {
                let node = node.lock();
                node.settled_target.map(|_| node.recipe_output)
            };
            // An archive target's `$@` is the archive rather than the whole
            // name, which is a halving of the settled spelling this side cannot
            // do. Nothing renames an archive down a search path here, so the
            // case is left where it was.
            if let Some(target) = target
                && crate::archive::split_archive_name(target.name_bytes(&ev.session)).is_none()
            {
                let reference = Self::settled_reference(ev, node, target, view);
                references.insert(target, reference);
            }
        }
        let searched = node.lock().searched_inputs.clone();
        if searched.is_empty() {
            return references;
        }
        for word in words {
            if !searched.iter().any(|(input, _)| input == word) || references.contains_key(word) {
                continue;
            }
            references.insert(*word, Self::settled_reference(ev, node, *word, view));
        }
        references
    }

    /// The reference that stands for one name the build has still to settle,
    /// minted once per name and form and reused wherever the recipe reads it
    /// again.
    fn settled_reference(
        ev: &mut Evaluator,
        node: &Arc<Mutex<DepNode>>,
        name: Symbol,
        view: SettledNameView,
    ) -> Bytes {
        let held = node
            .lock()
            .settled_names
            .iter()
            .find(|settled| settled.input == name && settled.view == view)
            .map(|settled| settled.variable);
        let variable = match held {
            Some(variable) => variable,
            None => {
                let index = node.lock().settled_names.len();
                let mut spelling = BytesMut::from(SETTLED_NAME_PREFIX);
                spelling.put_slice(index.to_string().as_bytes());
                let variable = ev.session.intern(spelling.freeze());
                node.lock().settled_names.push(SettledName {
                    variable,
                    input: name,
                    view,
                });
                variable
            }
        };
        let mut reference = BytesMut::from(&b"${"[..]);
        reference.put_slice(variable.name_bytes(&ev.session));
        reference.put_slice(b"}");
        reference.freeze()
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
        let current_dep_node = current_dep_node.as_ref().unwrap();
        // Settled before the node is read, because minting a reference needs
        // the session to write and reading a name needs it to read. The words
        // are this form's own: `$<` names one prerequisite and `$|` names none
        // of the ordinary ones, so neither mints a reference the other's list
        // would have wanted.
        let words = {
            let node = current_dep_node.lock();
            match self.typ {
                AutoCommand::Less => node.actual_inputs.first().copied().into_iter().collect(),
                AutoCommand::Hat | AutoCommand::Plus => node.actual_inputs.clone(),
                AutoCommand::Bar => node.actual_order_only_inputs.clone(),
                _ => Vec::new(),
            }
        };
        let references = self.settled_references(ev, current_dep_node, &words);
        // The destination's own `$?` for this launch, when it handed one over.
        // Taken before the field borrows below so a `$?` read can reach it
        // without a second mutable borrow of the evaluator.
        let launch_new_inputs = ev.launch_new_inputs.clone();
        let current_dep_node = current_dep_node.lock();
        let names = &ev.session.symtab;
        // How the build spells a prerequisite the directory search answered
        // about. Every automatic variable that names prerequisites reads
        // through this, because GNU Make's `$<` and its neighbours are read off
        // the file objects whose names `update_file_1` has already settled.
        let settled = &ev.settled_names;
        let spelt = |name: Symbol| settled.get(&name).copied().unwrap_or(name).as_bytes(names);
        // A prerequisite the build has still to choose a name for reaches the
        // recipe as a reference; every other one reaches it as itself. Where
        // any reference is written the halving this form asks for is done here,
        // a word at a time, because `eval` cannot halve a reference.
        let held = |name: Symbol| references.get(&name).cloned();
        // The halving is done here only where a reference was written, because
        // then `eval` is told the value is bound and does not halve it again.
        // A value of nothing but names is left whole and halved there, exactly
        // as it always was.
        let viewed = |name: Bytes| {
            if references.is_empty() {
                name
            } else {
                self.viewed(&name)
            }
        };
        let word = |name: Symbol| {
            held(name).unwrap_or_else(|| viewed(crate::archive::member_or_whole(&spelt(name))))
        };

        match &self.typ {
            AutoCommand::At => {
                // For `lib.a(member.o)` the target is the archive and the
                // member is `$%`; `set_file_variables` splits them at the
                // first `(` (reference/gnumake/src/commands.c).
                let name = current_dep_node.recipe_output.as_bytes(names);
                match crate::archive::split_archive_name(&name) {
                    Some((archive, _)) => out.put_slice(&name.slice(..archive.len())),
                    // A reference where the build settles the spelling, and the
                    // name itself where it does not. The reference already
                    // stands for the form this variable asks for, so nothing
                    // halves it afterwards -- `eval` is told by `Bound::Late`.
                    None => match held(current_dep_node.recipe_output) {
                        Some(reference) => out.put_slice(&reference),
                        None => out.put_slice(&name),
                    },
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
                    out.put_slice(&word(*ai))
                }
            }
            AutoCommand::Hat => {
                let mut seen = HashSet::new();
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    if seen.insert(*ai) {
                        ww.write(&word(*ai))
                    }
                }
            }
            AutoCommand::Plus => {
                let mut ww = WordWriter::new(out);
                for ai in current_dep_node.actual_inputs.iter() {
                    ww.write(&word(*ai))
                }
            }
            AutoCommand::Bar => {
                // A name declared both ways is an ordinary prerequisite and
                // nothing else. GNU Make holds one prerequisite chain with an
                // `ignore_mtime` bit per entry, and `set_file_variables`
                // (commands.c) upgrades both entries — "if so then we need to
                // 'upgrade' one that is order-only" — before it reads any of
                // the three lists off, so the name leaves `$|` and stays in
                // `$^`. Two lists say the same thing by leaving it out here.
                let mut seen: FastSet<Symbol> =
                    current_dep_node.actual_inputs.iter().copied().collect();
                let mut ww = WordWriter::new(out);
                for oi in current_dep_node.actual_order_only_inputs.iter() {
                    if seen.insert(*oi) {
                        ww.write(&held(*oi).unwrap_or_else(|| viewed(spelt(*oi))))
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
                    //
                    // Substituted into the name the rule was WRITTEN for, which
                    // is the one whose stem `record_files` read when it worked
                    // the prerequisites out. A `GPATH` rename moves the target
                    // and leaves the stem where it was: `out.o: %.o: %.c` made
                    // at `build/out.o` still has `$*` of `out`.
                    out.put_slice(&pat.append_subst(
                        &current_dep_node.declared_output.as_bytes(names),
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
                let mut seen: FastSet<Symbol> = FastSet::default();

                if *timing == NewInputsTiming::Launch {
                    // The recipe is being expanded at launch. When the
                    // destination handed over the list its scheduler settled,
                    // that is `$?` — it counts a prerequisite that does not
                    // exist, one remade without its mtime moving, and an
                    // archive member, none of which a stat of the tree can see.
                    // The words are already spelt where this command runs and
                    // an archive member is already reduced to its member name,
                    // so the `D`/`F` forms split this value where `eval` splits
                    // any other.
                    if let Some(new_inputs) = &launch_new_inputs {
                        out.put_slice(new_inputs);
                    } else {
                        // Nothing was handed over — a grouped action expanded
                        // here answers from the timestamps GNU Make compares,
                        // which are the ones on disk now that every
                        // prerequisite has settled.
                        let mut ww = WordWriter::new(out);
                        let target_age = ExecStatus::Timestamp(target_timestamp(
                            &current_dep_node.recipe_output.as_bytes(names),
                        )?);
                        for ai in current_dep_node.actual_inputs.iter() {
                            let ai_str = spelt(*ai);
                            if seen.insert(*ai)
                                && ExecStatus::Timestamp(prerequisite_timestamp(&ai_str)?)
                                    > target_age
                            {
                                ww.write(&crate::archive::member_or_whole(&ai_str));
                            }
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
                                ww.write(ai.name_bytes(names));
                            }
                        }
                        ww.write(b"$(test -e");
                        ww.write(current_dep_node.recipe_output.name_bytes(names));
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
        // A value holding a reference has had this form applied to it already,
        // word by word, so `eval` must not apply it a second time.
        Ok(if references.is_empty() {
            Bound::Now
        } else {
            Bound::Late
        })
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

/// What separates the lines of a recipe `.ONESHELL` made one script of.
///
/// GNU Make's own newlines, and nothing else: `chop_commands` never chopped
/// the recipe, so what the shell is handed is the text the makefile wrote.
/// Both consumers spell it with this — the one that compiles the recipe into a
/// build file and the one that runs it — because a separator that differed
/// between them would be two different scripts from one makefile.
pub const ONE_SHELL_SEPARATOR: &[u8] = b"\n";

/// The one script a `.ONESHELL` recipe is, out of the lines it was read as.
///
/// Beside [`CommandEvaluator::read_one_shell_prefixes_off_the_first_line`]
/// because the two are one reading of one flag: the prefixes are the first
/// line's BECAUSE there is only one line, and this assembles that line. Every
/// per-line flag the runner reads has already been made one answer there, so
/// the script takes the first line's and the rest carry nothing left to
/// disagree about.
fn one_shell_script(lines: Vec<Command>) -> Vec<Command> {
    let mut lines = lines.into_iter();
    let Some(mut script) = lines.next() else {
        return Vec::new();
    };
    let mut text = BytesMut::from(script.cmd.as_ref());
    for line in lines {
        text.put_slice(ONE_SHELL_SEPARATOR);
        text.put_slice(&line.cmd);
        // The invocations stay in the order the shell would reach them; the
        // script runs every line's, because it is every line.
        script.recursive_make.extend(line.recursive_make);
        script.nesting = script.nesting.take().or(line.nesting);
        script.unreached |= line.unreached;
    }
    script.cmd = text.freeze();
    vec![script]
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
    /// Whether the blanks in front of this line are the script's own text
    /// rather than the recipe syntax Make eats.
    ///
    /// True only below the first line of a `.ONESHELL` recipe whose `SHELL` is
    /// not one of the seven names GNU Make knows — see [`PrefixStripping`].
    /// Such a recipe is one script for a reader that may be counting columns:
    /// `SHELL := /usr/bin/python3` with an `if` in the recipe is the case that
    /// makes it plain, and 4.4.1 hands the indentation over untouched.
    pub keeps_indent: bool,
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
    /// A recursion the line names and never starts: the invocation stands
    /// under a condition the reading settled as false, or in a loop over no
    /// words, so no Make is composed for it and none is started. The line
    /// still runs as written, and runs nothing.
    ///
    /// A fact for the census and not for the build, which treats the line as
    /// the ordinary residual work it is: a report that dropped the line would
    /// no longer name every recursion the Makefile wrote.
    pub unreached: bool,
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
    id: ValueId,
    ev: &Evaluator,
    rule_vars: Option<&Vars>,
    seen: &mut FastSet<Symbol>,
) -> bool {
    match ev.session.values.get(id) {
        Value::Literal(_, _) => false,
        // Expanding it raises rather than producing text, so it reaches nothing.
        Value::Unreadable(_, _) => false,
        Value::SymRef(_, sym) => symbol_can_reach_make(*sym, ev, rule_vars, seen),
        Value::List(_, values) => ev
            .session
            .values
            .children(*values)
            .iter()
            .any(|value| expansion_can_reach_make(*value, ev, rule_vars, seen)),
        // The name is computed, so it can be `MAKE`.
        Value::VarRef(_, _) => true,
        Value::VarSubst {
            loc: _,
            name,
            pat,
            subst,
        } => {
            match ev.session.values.get(*name) {
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
            expansion_can_reach_make(*pat, ev, rule_vars, seen)
                || expansion_can_reach_make(*subst, ev, rule_vars, seen)
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
            ev.session
                .values
                .children(*args)
                .iter()
                .any(|arg| expansion_can_reach_make(*arg, ev, rule_vars, seen))
        }
        // Finished fold bytes hold no reference at all.
        Value::Folded { .. } => false,
    }
}

/// Whether expanding a reference to `sym` could reach `$(MAKE)`.
fn symbol_can_reach_make(
    sym: Symbol,
    ev: &Evaluator,
    rule_vars: Option<&Vars>,
    seen: &mut FastSet<Symbol>,
) -> bool {
    if sym.name_bytes(&ev.session) == b"MAKE" {
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
    expansion_can_reach_make(definition, ev, rule_vars, seen)
}

/// Whether a recipe line as written can hold no command at all.
///
/// A recipe of nothing but whitespace has nothing to expand and nothing to
/// run: GNU Make reads it as a target with an empty recipe, which is remade by
/// doing nothing. Text is the only case that can be answered without expanding,
/// and it is the case Makefiles write — `all:;` — so it is worth answering.
pub fn is_blank_recipe_line(arena: &ValueArena, id: ValueId) -> bool {
    match arena.get(id) {
        Value::Literal(_, text) => text.trim_ascii().is_empty(),
        Value::List(_, values) => arena
            .children(*values)
            .iter()
            .all(|value| is_blank_recipe_line(arena, *value)),
        _ => false,
    }
}

/// What an unexpanded recipe does with `$?`.
///
/// Two answers rather than one, because the two have different consequences.
/// Reaching the value at all is what makes an edge declare deferred freshness,
/// so the destination settles the list and hands it over when it launches the
/// edge. READING that value with a function is what takes the whole recipe out
/// of the compiler's hands: a scheduler-bound `$?` leaves a placeholder behind,
/// the placeholder is one word standing for a list nobody has settled yet, and
/// a function asked about that word answers about the word.
#[derive(Clone, Copy, Default)]
pub struct NewInputsUse {
    /// Expanding this may produce a value derived from `$?`.
    ///
    /// Conservative: a name computed at expansion time can be `?`, and nothing
    /// here can rule it out.
    pub reached: bool,
    /// Expanding this DOES produce such a value — a reference to `$?` itself,
    /// or text carrying one. Held apart from [`Self::reached`] because a name
    /// this walk cannot follow is a reason to settle the list and not a reason
    /// to believe the recipe reads it.
    certain: bool,
    /// A value certainly derived from `$?` is read by a function, so what the
    /// expansion produces depends on more than the words the list is spelt as.
    pub decides: bool,
}

impl NewInputsUse {
    const NONE: Self = Self {
        reached: false,
        certain: false,
        decides: false,
    };
    /// The list may be behind this, and nothing here can say.
    const MAY_REACH: Self = Self {
        reached: true,
        certain: false,
        decides: false,
    };
    /// The list is behind this.
    const REACHES: Self = Self {
        reached: true,
        certain: true,
        decides: false,
    };

    /// Both answers at once, for two values one expansion produces.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            reached: self.reached || other.reached,
            certain: self.certain || other.certain,
            decides: self.decides || other.decides,
        }
    }

    /// The same answer, for a value a function reads rather than copies into
    /// what it produces.
    const fn read(self) -> Self {
        Self {
            reached: self.reached,
            certain: self.certain,
            decides: self.decides || self.certain,
        }
    }
}

/// One walk of a recipe, asking what its expansion would do with `$?`.
///
/// The answer for a variable is cached rather than merely marked seen: a
/// recipe that names the same variable twice reads it in two places, and the
/// second place is as able to be the one that reads the list as the first.
/// `active` is the cycle guard the cache cannot be, since a self-referential
/// definition has no answer to cache yet.
pub struct NewInputsWalk<'a> {
    ev: &'a Evaluator,
    rule_vars: Option<&'a Vars>,
    cache: FastMap<Symbol, NewInputsUse>,
    active: FastSet<Symbol>,
}

impl<'a> NewInputsWalk<'a> {
    /// A walk over one rule's recipes. The rule's own variables come first,
    /// because a target-specific definition is the one its recipe expands.
    #[must_use]
    pub fn new(ev: &'a Evaluator, rule_vars: Option<&'a Vars>) -> Self {
        Self {
            ev,
            rule_vars,
            cache: FastMap::default(),
            active: FastSet::default(),
        }
    }

    /// What expanding `id` would do with `$?`.
    ///
    /// The same walk [`expansion_can_reach_make`] performs, asked about the
    /// other automatic variable a compiler cannot answer early: the value's own
    /// syntax tree, plus the tree of every recursively expanded variable it
    /// names, plus the body of every `$(call)` whose function is named
    /// outright. A simply expanded variable holds text, and text holds no
    /// reference, which is where the walk stops — as it does there.
    pub fn of(&mut self, id: ValueId) -> NewInputsUse {
        match self.ev.session.values.get(id) {
            Value::Literal(_, _) | Value::Folded { .. } => NewInputsUse::NONE,
            // Expanding it raises rather than producing text, so it reaches
            // nothing.
            Value::Unreadable(_, _) => NewInputsUse::NONE,
            Value::SymRef(_, sym) => self.of_symbol(*sym),
            Value::List(_, values) => {
                let values = self.ev.session.values.children(*values);
                values.iter().fold(NewInputsUse::NONE, |so_far, value| {
                    so_far.merge(self.of(*value))
                })
            }
            // The name is computed, so it can be `?`. Which variable the
            // reference stands for is not knowable here, and neither therefore
            // is what that variable does with the list — but a name spelt out
            // of the list is itself a reading of it, and that much is.
            Value::VarRef(_, name) => NewInputsUse::MAY_REACH.merge(NewInputsUse {
                decides: self.of(*name).read().decides,
                ..NewInputsUse::NONE
            }),
            // `$(NAME:pat=subst)` rewrites the value it names word by word, so
            // reaching the list through any of the three is a reading of it.
            Value::VarSubst {
                loc: _,
                name,
                pat,
                subst,
            } => {
                let named = match self.ev.session.values.get(*name) {
                    Value::Literal(_, literal) => self
                        .ev
                        .session
                        .symtab
                        .peek_symbol(literal)
                        .map_or(NewInputsUse::NONE, |sym| self.of_symbol(sym)),
                    _ => NewInputsUse::MAY_REACH,
                };
                named.merge(self.of(*pat)).merge(self.of(*subst)).read()
            }
            Value::Func { loc: _, fi, args } => {
                let args = self.ev.session.values.children(*args);
                if fi.name == b"call" {
                    return self.of_call(args);
                }
                let mut result = NewInputsUse::NONE;
                for (at, arg) in args.iter().enumerate() {
                    let arg = self.of(*arg);
                    // `$(filter-out pat,$?)` is the one reading a scheduler can
                    // perform for itself: the patterns travel with the edge and
                    // the words are struck out where the list is settled. Every
                    // other reading — the pattern side of this very function
                    // included — is a question about a list that does not exist
                    // yet.
                    result = result.merge(if fi.name == b"filter-out" && at == 1 {
                        arg
                    } else {
                        arg.read()
                    });
                }
                result
            }
        }
    }

    /// What `$(call f,...)` would do with `$?`.
    ///
    /// The body is walked when `f` is named outright, which is how a Makefile
    /// that hides `$?` behind a chain of definitions is followed to it. kbuild
    /// is the case that makes it matter: `$(call if_changed,cc_o_c)` reaches
    /// `$?` three definitions down, in the condition of an `$(if)` that chooses
    /// between the compile and doing nothing.
    ///
    /// The parameters are not bound, so a `$(1)` in the body is a reference to
    /// a variable this walk does not hold. An argument that reaches the list is
    /// therefore counted as read: the body receives it under a name nothing
    /// here can follow.
    fn of_call(&mut self, args: &[ValueId]) -> NewInputsUse {
        let mut result = NewInputsUse::NONE;
        for arg in args.iter().skip(1) {
            result = result.merge(self.of(*arg).read());
        }
        let Some(name) = args.first() else {
            return result;
        };
        // The function's own name chooses the body, so a name spelt out of the
        // list is a reading of it — and one this walk cannot follow.
        match self.ev.session.values.get(*name) {
            Value::Literal(_, literal) => match self.ev.session.symtab.peek_symbol(literal) {
                Some(sym) => result.merge(self.of_symbol(sym)),
                // A name no symbol was ever made for names no variable, and
                // `$(call)` on an undefined function expands to nothing.
                None => result,
            },
            _ => {
                let name = self.of(*name).read();
                result.merge(name).merge(NewInputsUse::MAY_REACH)
            }
        }
    }

    /// What expanding a reference to `sym` would do with `$?`.
    fn of_symbol(&mut self, sym: Symbol) -> NewInputsUse {
        if matches!(sym.name_bytes(&self.ev.session), b"?" | b"?D" | b"?F") {
            return NewInputsUse::REACHES;
        }
        if let Some(answer) = self.cache.get(&sym) {
            return *answer;
        }
        if !self.active.insert(sym) {
            return NewInputsUse::NONE;
        }
        let answer = self.definition_use(sym);
        self.active.remove(&sym);
        self.cache.insert(sym, answer);
        answer
    }

    fn definition_use(&mut self, sym: Symbol) -> NewInputsUse {
        let bound = self
            .rule_vars
            .and_then(|vars| vars.peek(sym))
            .or_else(|| self.ev.session.globals.peek(sym));
        let Some(bound) = bound else {
            return NewInputsUse::NONE;
        };
        let Some(definition) = bound.read().recursive_definition() else {
            return NewInputsUse::NONE;
        };
        self.of(definition)
    }
}

/// GNU Make's `lines_flags[i] & COMMANDS_RECURSE` for one written recipe line:
/// a `+` among the prefixes it opens with, or a `$(MAKE)` reference anywhere in
/// it.
///
/// Read off the line AS WRITTEN, which `chop_commands` (commands.c) does at
/// parse time over the unexpanded text. That is the whole of why a `$(FOO)`
/// whose value is `+` is not a recursive line and a written `+` is.
pub fn written_line_recurses(arena: &ValueArena, id: ValueId, names: &impl Interner) -> bool {
    let mut prefixes = LinePrefixes {
        echo: true,
        dash_prefixed: false,
        recursive_line: references_make(arena, id, names),
    };
    scan_written_prefixes(arena, id, &mut prefixes);
    prefixes.recursive_line
}

fn references_make(arena: &ValueArena, id: ValueId, names: &impl Interner) -> bool {
    match arena.get(id) {
        Value::Literal(_, _) => false,
        // Expanding it raises rather than producing text, so it reaches nothing.
        Value::Unreadable(_, _) => false,
        Value::SymRef(_, sym) => sym.name_bytes(names) == b"MAKE",
        Value::List(_, values) => arena
            .children(*values)
            .iter()
            .any(|value| references_make(arena, *value, names)),
        Value::VarRef(_, name) => references_make(arena, *name, names),
        // `$(MAKE:x=y)` holds the name as a literal rather than a reference,
        // and 4.4.1 does not classify it. The pattern and replacement are
        // ordinary text and can hold a reference of their own.
        Value::VarSubst {
            loc: _,
            name: _,
            pat,
            subst,
        } => references_make(arena, *pat, names) || references_make(arena, *subst, names),
        Value::Func {
            loc: _,
            fi: _,
            args,
        } => arena
            .children(*args)
            .iter()
            .any(|arg| references_make(arena, *arg, names)),
        // Finished fold bytes name nothing, `MAKE` included.
        Value::Folded { .. } => false,
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

/// Whether a line's shell flags arm errexit before the line says anything.
///
/// `-c` is GNU Make's default and `-ec` is what `.POSIX:` gives an
/// unprefixed line; `.SHELLFLAGS` can say anything. Read as `sh` reads its
/// option words: every `-` word turns on the letters it holds and every `+`
/// word turns them off, the last word to mention `e` winning. Only the
/// letter is read — `-o errexit` is not seen, and not seeing it costs a lift
/// and never a wrong one, because an errexit the reading does not know
/// about only ever makes the line stop sooner than the reading assumed.
fn arms_errexit(shell_flag: &[u8]) -> bool {
    let mut armed = false;
    for word in shell_flag.split(|byte| byte.is_ascii_whitespace()) {
        match word.split_first() {
            Some((b'-', letters)) if letters.contains(&b'e') => armed = true,
            Some((b'+', letters)) if letters.contains(&b'e') => armed = false,
            _ => {}
        }
    }
    armed
}

/// The shells GNU Make knows to be Bourne-compatible, in its own order.
///
/// `is_bourne_compatible_shell` (job.c:433) is the whole of the test: it walks
/// back to the last directory separator and compares what follows against
/// these seven names exactly. Nothing about the file is looked at — a shell
/// named `flash` is not one of these and `./sub/sh` is, whatever either one
/// turns out to be.
const BOURNE_COMPATIBLE_SHELLS: [&[u8]; 7] =
    [b"sh", b"bash", b"dash", b"ksh", b"rksh", b"zsh", b"ash"];

/// Whether GNU Make would take a `SHELL` of this spelling for a Bourne shell.
///
/// The value is `$(SHELL)` as the Makefile left it, arguments and all, because
/// that is what GNU Make hands the test. `SHELL = ./bsh/sh -x` therefore has
/// the basename `sh -x` and is not one of these — measured against 4.4.1,
/// which leaves the prefixes in place for it and strips them for `./bsh/sh`.
///
/// Asked twice, because GNU Make asks twice about the same script: here, where
/// a recipe is being compiled and the answer decides whether the text keeps its
/// interior prefixes, and again in [`crate::fileutil`], where a launch is being
/// assembled and `construct_command_argv_internal` asks it for itself.
pub fn is_bourne_compatible_shell(shell: &[u8]) -> bool {
    let basename = match shell.iter().rposition(|&byte| byte == b'/') {
        Some(separator) => &shell[separator + 1..],
        None => shell,
    };
    BOURNE_COMPATIBLE_SHELLS.contains(&basename)
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
fn scan_written_prefixes(arena: &ValueArena, id: ValueId, prefixes: &mut LinePrefixes) -> bool {
    match arena.get(id) {
        Value::Literal(_, text) => parse_command_prefixes(
            text.clone(),
            &mut prefixes.echo,
            &mut prefixes.dash_prefixed,
            &mut prefixes.recursive_line,
        )
        .is_empty(),
        Value::List(_, values) => arena
            .children(*values)
            .iter()
            .all(|value| scan_written_prefixes(arena, *value, prefixes)),
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
    /// Whether a line is still owed although the text has run out.
    ///
    /// Set where the text ended ON a newline, and set to begin with, so an
    /// expansion that came out empty is still the one line it was written as.
    /// Only [`Self::keeps_blank_lines`] ever sets it, so every other recipe
    /// stops exactly where it always did — at the first empty remainder.
    owed: bool,
    /// Whether an expansion that left a line empty leaves a BLANK LINE rather
    /// than nothing at all.
    ///
    /// `.ONESHELL` is the whole of the condition. GNU Make chops a recipe into
    /// lines BEFORE expanding it and then walks past every expanded line that
    /// came out empty — `construct_command_argv_internal` answers no argv for
    /// one ("Make sure not to bother processing an empty line") and
    /// `start_job_command` moves to the next. `chop_commands` (commands.c:335)
    /// never chops a `.ONESHELL` recipe at all: it is one line, so the same
    /// expansion leaves the newlines that were around it exactly where they
    /// were and the script has a blank line in it.
    ///
    /// That is more than a blank line in an echo, because a script can be
    /// reading its own text. Measured on 4.4.1: a here-document with a line
    /// that expands to nothing inside it writes that blank line into the file,
    /// and dropping the line writes a different file.
    keeps_blank_lines: bool,
    /// What the written line already settled for every line of its expansion:
    /// `-s` and any prefix written before the expansion for the echo and the
    /// forgiveness, and those plus the unexpanded `$(MAKE)` scan for the
    /// recursion.
    written: LinePrefixes,
    stripping: PrefixStripping,
}

/// Which of a recipe's lines Make takes the leading blanks and `[@+-]` off.
///
/// GNU Make removes them in two places and only one of them asks about the
/// shell. `start_job_command` (job.c:1198) skips them off the front of what it
/// is about to run and stops at the first newline, so the recipe's first line
/// loses them whatever the shell is. `construct_command_argv_internal`
/// (job.c:3293) removes them from *every* logical line of a `.ONESHELL`
/// recipe, and only when [`is_bourne_compatible_shell`] says so — its comment
/// gives the reason: a `SHELL` Make does not recognise may be reading those
/// characters as its own script.
#[derive(Clone, Copy)]
struct PrefixStripping {
    /// Whether a line below the recipe's first loses them too.
    interior: bool,
    /// Whether the recipe's first line has been read already.
    past_first: bool,
}

impl ExpandedRecipeLines {
    fn new(
        expansion: Bytes,
        written: LinePrefixes,
        stripping: PrefixStripping,
        keeps_blank_lines: bool,
    ) -> Self {
        Self {
            rest: expansion,
            owed: keeps_blank_lines,
            keeps_blank_lines,
            written,
            stripping,
        }
    }
}

impl Iterator for ExpandedRecipeLines {
    type Item = (Bytes, LinePrefixes);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() && !self.owed {
            return None;
        }
        let eol = find_end_of_line(&self.rest);
        // A newline was consumed exactly where the two halves do not add back
        // up to what was walked, and a text that ended on one still owes the
        // line after it.
        self.owed = self.keeps_blank_lines && eol.line.len() + eol.rest.len() != self.rest.len();
        self.rest = eol.rest;
        if self.stripping.past_first && !self.stripping.interior {
            // Script text, handed over as the expansion left it — indentation
            // included, because a shell Make does not know may be one that
            // reads it.
            return Some((eol.line, self.written));
        }
        let line = eol.line.slice_ref(trim_left_space(&eol.line));
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
        // Where a blank line is a line, reading one is being past the first:
        // GNU Make's skip-over runs off the front of the whole expanded text
        // and stops at the first newline, so an expansion that opened with one
        // leaves the line above empty and everything below it interior.
        self.stripping.past_first |= self.keeps_blank_lines || !command.is_empty();
        Some((command, prefixes))
    }
}

pub struct CommandEvaluator<'a> {
    pub ev: &'a mut Evaluator,
    pub current_dep_node: Arc<Mutex<Option<Arc<Mutex<DepNode>>>>>,
    pub found_new_inputs: Arc<Mutex<bool>>,
    /// The shell the last node read through [`Self::eval`] runs its recipe
    /// under, read with that node's own scope in hand.
    ///
    /// Said here rather than returned because it is the answer to a question
    /// about the node rather than about the lines: a recipe that expanded to
    /// nothing still has a shell, and the caller needs the same one either
    /// way. Empty before the first node is read.
    pub recipe_shell: Bytes,
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
            recipe_shell: Bytes::new(),
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

    /// Take up an evaluator again with the state an earlier command evaluator
    /// left, without registering the autocommands a second time.
    ///
    /// [`Self::new`] defines `$@`, `$<` and the rest in the session's global
    /// scope, and each of those bindings holds the very `current_dep_node` the
    /// evaluator sets before it reads a recipe. The session travels with the
    /// evaluator, so those bindings are already there — and defining them again
    /// would replace them with ones pointing at a different cell, so a recipe
    /// expanded afterwards would read `$@` from a node nobody had set. Handing
    /// the same cells back is what makes the two halves of one emission one
    /// evaluation.
    pub fn resumed(
        ev: &'a mut Evaluator,
        evaluation: crate::ninja::BuildEvaluation,
        current_dep_node: Arc<Mutex<Option<Arc<Mutex<DepNode>>>>>,
        found_new_inputs: Arc<Mutex<bool>>,
        recipe_shell: Bytes,
    ) -> Self {
        ev.new_inputs_timing = evaluation.new_inputs_timing;
        ev.shell_evaluation = evaluation.shell_evaluation;
        ev.file_evaluation = evaluation.file_evaluation;
        ev.output_evaluation = evaluation.output_evaluation;
        Self {
            ev,
            current_dep_node,
            found_new_inputs,
            recipe_shell,
        }
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
        self.ev.session.set_global_var(sym, v, false)?;
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
        self.ev.session.set_global_var(sym, v, false)?;
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
        self.ev.session.set_global_var(sym, v, false)?;
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
        self.ev.session.set_global_var(sym, v, false)?;
        Ok(())
    }

    /// Give every line of a `.ONESHELL` recipe the one set of prefixes GNU
    /// Make read for the whole of it.
    ///
    /// Under `.ONESHELL` the recipe is not chopped: `chop_commands` leaves it
    /// as a single line, so the scan that fills `lines_flags` runs over its
    /// first line's leading characters and what it finds stands for all of it.
    /// `start_job_command` then scans the *expanded* text the same way and
    /// stops at the first newline — its skip-over is blanks and prefixes, and
    /// a newline is neither — so an expansion that produced the first line
    /// carries prefixes too, and lines below it never do.
    ///
    /// A prefix character further down is script text rather than a flag.
    /// Whether the shell ever sees it is decided elsewhere and by the shell's
    /// name: `construct_command_argv_internal` strips the leading blanks and
    /// `[@+-]` off every line for a Bourne-compatible one — the basenames
    /// `sh`, `bash`, `dash`, `ksh`, `rksh`, `zsh`, `ash` — and leaves the whole
    /// text alone for any other, because there the characters could be the
    /// script's own. Measured on 4.4.1: `@echo two` on the second line prints
    /// `two` under `/bin/sh` and under `SHELL = /bin/dash`, and reports
    /// `@echo: not found` under a `SHELL` named anything else. Either way the
    /// recipe stays loud, because the flags came from the line above.
    ///
    /// `+` is the one GNU Make widens for itself: the `$(MAKE)` search
    /// `chop_commands` runs after the prefix scan reads the whole line it
    /// chopped, which here is the whole recipe. So `$(MAKE)` on any line marks
    /// the recipe recursive while a `+` on any line but the first does not —
    /// measured under `-n`, where a recursive recipe runs and an ordinary one
    /// is only printed.
    fn read_one_shell_prefixes_off_the_first_line(result: &mut [Command], references_make: bool) {
        let Some(first) = result.first() else {
            return;
        };
        let echo = first.echo;
        let dash_prefixed = first.dash_prefixed;
        let ignore_error = first.ignore_error;
        let shell_flag = first.shell_flag.clone();
        // The first line's own `+` is already in its `recursive_line`, beside
        // the reference this recipe was read for.
        let recursive_line = first.recursive_line || references_make;
        for command in result.iter_mut() {
            command.echo = echo;
            command.dash_prefixed = dash_prefixed;
            command.ignore_error = ignore_error;
            command.shell_flag = shell_flag.clone();
            command.recursive_line = recursive_line;
        }
    }

    /// The processes this recipe is, for a consumer that RUNS it rather than
    /// compiling it into a build file.
    ///
    /// One per line, which is what GNU Make gives a recipe — except under
    /// `.ONESHELL`, where the recipe is one script handed to one shell. So a
    /// variable set on one line is still set on the next, a `cd` stays made,
    /// and the status the recipe is judged on is the last line's rather than
    /// the first failure's.
    ///
    /// Held apart from [`Self::eval`] rather than folded into it because the
    /// lines are what a compiler wants: each has to be translated for the
    /// binding it lands in, classified for `$(MAKE)`, and read for whether it
    /// is work worth announcing. Only a consumer that starts processes needs
    /// them assembled, and it needs them assembled the same way.
    pub fn launches(&mut self, n: &Arc<Mutex<DepNode>>) -> Result<Vec<Command>> {
        let commands = self.eval(n)?;
        if !self.ev.session.flags.one_shell {
            return Ok(commands);
        }
        Ok(one_shell_script(commands))
    }

    // [spec:ronin:req:make.recursive-invocation+3]
    pub fn eval(&mut self, n: &Arc<Mutex<DepNode>>) -> Result<Vec<Command>> {
        let mut result: Vec<Command> = Vec::new();
        let node_cmds;
        {
            let node = n.lock();
            self.ev.loc = node.loc;
            self.ev.current_scope = node.rule_vars.clone();
            node_cmds = node.cmds.clone();
        }
        let node_ignores_errors = n.lock().is_ignore_error;
        // `.SILENT` named this target, which GNU Make carries as
        // `COMMANDS_SILENT` in the file's `command_flags` and ORs into every
        // line's own flags. Seeding it here rather than clearing it per line is
        // the same thing said once: `scan_written_prefixes` below can only turn
        // echoing off, never back on.
        let node_is_silent = n.lock().is_silent;
        // GNU Make expands `$(SHELL)` once per recipe with the target's own
        // scope — `construct_command_argv` (job.c) calls
        // `allocated_variable_expand_for_file ("$(SHELL)", file)` — so a
        // `SHELL` one target set governs that target's launches, and descends
        // to the prerequisites built for it exactly as every other
        // target-specific variable does. The scope was put in place above;
        // `.SHELLFLAGS` beside it has been read this way all along.
        //
        // A target remade by doing nothing never reaches
        // `construct_command_argv`, so it never asks what the shell is — and a
        // `SHELL` that cannot be expanded without starting one is a makefile
        // GNU Make runs to completion when nothing in it has a recipe to run.
        // Nothing reads the answer in that case either: an empty recipe
        // declares no rule.
        //
        // The boundary is the text as written rather than what it expands to,
        // and GNU Make's is the same one: `chop_commands` (commands.c) drops a
        // blank line before anything is expanded, while `all: ; $(EMPTY)` is a
        // command line that survives to be expanded and therefore does ask.
        let shell = if node_cmds
            .iter()
            .all(|cmd| is_blank_recipe_line(&self.ev.session.values, *cmd))
        {
            Bytes::new()
        } else {
            self.ev.get_shell()?
        };
        // Whether a `[@+-]` below the recipe's first line is a prefix Make eats
        // or a character the script wrote. Only `.ONESHELL` can make it the
        // latter, so only `.ONESHELL` asks what the shell is called — and the
        // name it asks about is this node's own.
        let strips_interior_prefixes =
            !self.ev.session.flags.one_shell || is_bourne_compatible_shell(&shell);
        self.recipe_shell = shell;
        // GNU Make's `$(MAKE)` search runs over the whole line it chopped, and
        // under `.ONESHELL` that line is the whole recipe.
        let mut references_make_anywhere = false;
        self.ev.is_evaluating_command = true;
        self.ev.session.ground_journal.suspend(true);
        *self.current_dep_node.lock() = Some(n.clone());
        *self.found_new_inputs.lock() = false;
        self.ev.deferred_new_inputs_filter_out.clear();
        for v in node_cmds {
            self.ev.loc = self.ev.session.values.get(v).loc();
            self.ev.expanded_make_in_command.clear();
            let cmds_buf = v.eval_to_buf(self.ev)?;
            let make_values = self.ev.expanded_make_in_command.clone();
            // `-i` and `.IGNORE` say a failure does not count, which is what
            // the `-` prefix says too — but only the prefix also relaxes the
            // shell, so the two are carried separately and joined at the end.
            let ignored_without_prefix = self.ev.session.flags.ignore_errors || node_ignores_errors;
            let mut written = LinePrefixes {
                echo: !self.ev.session.flags.is_silent_mode && !node_is_silent,
                dash_prefixed: false,
                // The classification is read from the recipe as written, so it
                // has to be taken before anything is expanded away.
                recursive_line: references_make(&self.ev.session.values, v, &self.ev.session),
            };
            references_make_anywhere |= written.recursive_line;
            scan_written_prefixes(&self.ev.session.values, v, &mut written);
            let lines = ExpandedRecipeLines::new(
                cmds_buf,
                written,
                PrefixStripping {
                    interior: strips_interior_prefixes,
                    // The recipe's first line is the first one that carried a
                    // command, whichever written line it came from — and under
                    // `.ONESHELL`, where a line that came out empty is still a
                    // line, the first one the makefile wrote.
                    past_first: !result.is_empty(),
                },
                self.ev.session.flags.one_shell,
            );
            for (cmd, prefixes) in lines {
                // A line whose expansion came out empty is a line that
                // vanished for every recipe GNU Make chopped, and a blank line
                // of the one script for the recipe it did not — see
                // [`ExpandedRecipeLines::keeps_blank_lines`].
                if !cmd.is_empty() || self.ev.session.flags.one_shell {
                    let keeps_indent = !strips_interior_prefixes && !result.is_empty();
                    let shell_flag = self.ev.get_shell_flag(prefixes.dash_prefixed)?;
                    // The line is read as the shell it runs under would read
                    // it, and whether that shell arms `-e` is part of what the
                    // reading decides: under `.POSIX:` a bare loop stops at its
                    // first failed iteration, and under `-c` it carries on.
                    let lifted = crate::lift::lift(
                        &self.ev.session,
                        &cmd,
                        &make_values,
                        arms_errexit(&shell_flag),
                    );
                    // Only a classified line is held to this. A `MAKE`-valued
                    // variable that GNU Make never classified is composed when
                    // the expansion makes that possible and otherwise left as
                    // written, exactly as 4.4.1 leaves it. And only a line
                    // that starts a Make somewhere: one that names it in an
                    // argument, or in text, has nothing to lift and nothing
                    // to explain.
                    let starts_a_make = prefixes.recursive_line
                        && make_values.iter().any(|make| spawns_make(&cmd, make));
                    let (recursive_make, nesting, unreached) = match lifted {
                        Ok(invocations) => {
                            let unreached = starts_a_make && invocations.is_empty();
                            (invocations, None, unreached)
                        }
                        Err(reason) => (Vec::new(), starts_a_make.then_some(reason), false),
                    };
                    result.push(Command {
                        output: n.lock().recipe_output,
                        cmd,
                        echo: prefixes.echo,
                        ignore_error: ignored_without_prefix || prefixes.dash_prefixed,
                        dash_prefixed: prefixes.dash_prefixed,
                        shell_flag,
                        force_no_subshell: false,
                        keeps_indent,
                        recursive_line: prefixes.recursive_line,
                        recursive_make,
                        nesting,
                        unreached,
                        loc: self.ev.loc,
                    })
                }
            }
        }

        if self.ev.session.flags.one_shell {
            // A whole recipe that expanded to blanks starts no shell at all,
            // and the target is remade by doing nothing. That is GNU Make's
            // opening test in `construct_command_argv_internal` — skip the
            // blanks, and answer no argv if the text ends there — asked of the
            // one line a `.ONESHELL` recipe is. A NEWLINE IS NOT A BLANK, so
            // two lines that both expanded to nothing are a script of one
            // newline and 4.4.1 really does start a shell on it; only a single
            // blank line is nothing to run.
            if result.len() <= 1
                && result
                    .iter()
                    .all(|c| c.cmd.iter().all(|b| matches!(b, b' ' | b'\t')))
            {
                result.clear();
            }
            Self::read_one_shell_prefixes_off_the_first_line(&mut result, references_make_anywhere);
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
                    keeps_indent: false,
                    recursive_line: false,
                    recursive_make: Vec::new(),
                    nesting: None,
                    unreached: false,
                    loc: node.loc,
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
        AutoCommand, AutoCommandVar, AutoCommandVariant, Command, CommandEvaluator,
        ExpandedRecipeLines, LinePrefixes, PrefixStripping, arms_errexit,
        is_bourne_compatible_shell, references_make, scan_written_prefixes, spawns_make,
    };
    use crate::expr::{ParseExprOpt, parse_expr};
    use crate::loc::Loc;
    use crate::session::Session;
    use bytes::Bytes;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// One line of a `.ONESHELL` recipe as the per-line reading left it, as
    /// `(text, prefixes)` — what a `@`, `-` or `+` in front of that line said
    /// about that line alone.
    fn one_shell_line(text: &'static str, prefixes: LinePrefixes) -> Command {
        let mut session = Session::new();
        Command {
            output: session.intern("out"),
            cmd: Bytes::from_static(text.as_bytes()),
            echo: prefixes.echo,
            ignore_error: prefixes.dash_prefixed,
            dash_prefixed: prefixes.dash_prefixed,
            shell_flag: Bytes::from_static(if prefixes.dash_prefixed {
                b"-c"
            } else {
                b"-ec"
            }),
            force_no_subshell: false,
            keeps_indent: false,
            recursive_line: prefixes.recursive_line,
            recursive_make: Vec::new(),
            nesting: None,
            unreached: false,
            loc: None,
        }
    }

    /// The reading every recipe gets but one: each line's leading blanks and
    /// `[@+-]` are Make's, because the recipe is not `.ONESHELL` or its shell
    /// is one of the seven.
    fn every_line() -> PrefixStripping {
        PrefixStripping {
            interior: true,
            past_first: false,
        }
    }

    fn plain() -> LinePrefixes {
        LinePrefixes {
            echo: true,
            dash_prefixed: false,
            recursive_line: false,
        }
    }

    /// What a `.ONESHELL` recipe's lines say about the recipe, measured
    /// against GNU Make 4.4.1 on each shape.
    ///
    /// `-` on the first line: `-@touch m1` above `false` makes m1 and exits 0.
    /// `-` below it: `@touch m1`, `-touch m2`, `false` makes both and exits 2.
    /// `@` on the first line silences the whole recipe; `@echo two` on the
    /// second line prints `two` and leaves every line echoed. `+` on the first
    /// line runs the whole script under `-n`; `+touch n2` on the second runs
    /// nothing. `$(MAKE)` on any line runs the whole script under `-n`,
    /// because the search `chop_commands` runs reads the whole chopped line
    /// and under `.ONESHELL` that is the whole recipe.
    #[test]
    fn a_one_shell_recipe_reads_its_prefixes_off_the_first_line() {
        let read = |lines: &[LinePrefixes], references_make: bool| {
            let mut commands: Vec<Command> = lines
                .iter()
                .map(|prefixes| one_shell_line("touch m", *prefixes))
                .collect();
            CommandEvaluator::read_one_shell_prefixes_off_the_first_line(
                &mut commands,
                references_make,
            );
            commands
                .iter()
                .map(|c| LinePrefixes {
                    echo: c.echo,
                    dash_prefixed: c.dash_prefixed,
                    recursive_line: c.recursive_line,
                })
                .collect::<Vec<_>>()
        };
        let dashed = LinePrefixes {
            dash_prefixed: true,
            ..plain()
        };
        let silent = LinePrefixes {
            echo: false,
            ..plain()
        };
        let plussed = LinePrefixes {
            recursive_line: true,
            ..plain()
        };

        // The first line's prefix reaches the whole recipe.
        assert_eq!(read(&[dashed, plain(), plain()], false), [dashed; 3]);
        assert_eq!(read(&[silent, plain(), plain()], false), [silent; 3]);
        assert_eq!(read(&[plussed, plain()], false), [plussed; 2]);
        // A prefix below it reaches nothing: it is script text, and whether
        // the shell ever sees it is the shell's name's business.
        assert_eq!(read(&[plain(), dashed, plain()], false), [plain(); 3]);
        assert_eq!(read(&[plain(), silent, plain()], false), [plain(); 3]);
        assert_eq!(read(&[plain(), plussed], false), [plain(); 2]);
        // `$(MAKE)` is the one GNU Make widens to the whole recipe.
        assert_eq!(read(&[plain(), plain()], true), [plussed; 2]);
        // The forgiveness carries the shell's flags with it, because a `-`
        // under `.POSIX:` is what asks for a shell without `-e`.
        let mut recipe = vec![
            one_shell_line("touch m1", dashed),
            one_shell_line("false", plain()),
        ];
        CommandEvaluator::read_one_shell_prefixes_off_the_first_line(&mut recipe, false);
        assert!(recipe.iter().all(|c| c.ignore_error));
        assert!(recipe.iter().all(|c| c.shell_flag == "-c"));
    }

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
        references_make(&session.values, value, &session)
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
            every_line(),
            false,
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

    /// The same reading for a `.ONESHELL` recipe, where the recipe was never
    /// chopped and a line the expansion emptied is a blank line of the script.
    fn expanded_for_one_shell(text: &'static [u8]) -> Vec<String> {
        ExpandedRecipeLines::new(
            Bytes::from_static(text),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false,
            },
            every_line(),
            true,
        )
        .map(|(cmd, _)| String::from_utf8_lossy(&cmd).into_owned())
        .collect()
    }

    /// A `.ONESHELL` recipe's lines are the text's, counted the way a shell
    /// counts them: one more than the newlines in it, and never fewer than
    /// one. GNU Make expands the whole recipe as a single line and hands the
    /// result over whole, so the blanks between are still there.
    #[test]
    fn a_one_shell_expansion_keeps_the_lines_it_emptied() {
        assert_eq!(expanded_for_one_shell(b""), vec![String::new()]);
        assert_eq!(expanded_for_one_shell(b"one"), vec!["one".to_owned()]);
        assert_eq!(
            expanded_for_one_shell(b"\none"),
            vec![String::new(), "one".to_owned()]
        );
        assert_eq!(
            expanded_for_one_shell(b"one\n"),
            vec!["one".to_owned(), String::new()]
        );
        assert_eq!(
            expanded_for_one_shell(b"one\n\ntwo"),
            vec!["one".to_owned(), String::new(), "two".to_owned()]
        );
        // The control, and the answer for every recipe GNU Make did chop: an
        // expansion that came out empty is no line at all, because
        // `start_job_command` gets no argv for it and walks on.
        assert!(expanded(b"").is_empty());
        assert_eq!(expanded(b"one\n").len(), 1);
    }

    /// Read one written line's expansion for a `.ONESHELL` recipe whose shell
    /// GNU Make does not know: the first line is Make's to read, and every
    /// line below it is the script's own text.
    ///
    /// Lines with no command left are dropped, as `eval` drops them.
    fn expanded_for_an_unknown_shell(text: &'static [u8]) -> Vec<String> {
        ExpandedRecipeLines::new(
            Bytes::from_static(text),
            LinePrefixes {
                echo: true,
                dash_prefixed: false,
                recursive_line: false,
            },
            PrefixStripping {
                interior: false,
                past_first: false,
            },
            false,
        )
        .filter(|(cmd, _)| !cmd.is_empty())
        .map(|(cmd, _)| String::from_utf8_lossy(&cmd).into_owned())
        .collect()
    }

    /// GNU Make's own list, and the only thing it asks about a shell before
    /// deciding whether a `.ONESHELL` script's interior `[@+-]` are its to
    /// remove. Measured against 4.4.1 with a `/bin/sh` copied under each name:
    /// the strip follows the name and nothing about the file.
    #[test]
    fn a_shell_is_bourne_compatible_by_its_basename_and_nothing_else() {
        for known in [
            &b"sh"[..],
            b"bash",
            b"dash",
            b"ksh",
            b"rksh",
            b"zsh",
            b"ash",
        ] {
            assert!(is_bourne_compatible_shell(known), "{known:?}");
            let path = [&b"/usr/local/bin/"[..], known].concat();
            assert!(is_bourne_compatible_shell(&path), "{path:?}");
        }
        // Named for one of the seven without being it.
        assert!(!is_bourne_compatible_shell(b"flash"));
        assert!(!is_bourne_compatible_shell(b"shell"));
        assert!(!is_bourne_compatible_shell(b"ashx"));
        assert!(!is_bourne_compatible_shell(b"sh.exe"));
        assert!(!is_bourne_compatible_shell(b"/usr/bin/python3"));
        assert!(!is_bourne_compatible_shell(b""));
        // The value is `$(SHELL)` whole, arguments included, so a shell given
        // one has a basename no name on the list can equal. Measured: 4.4.1
        // strips for `./bsh/sh` and leaves the prefixes in place for
        // `./bsh/sh -x`.
        assert!(!is_bourne_compatible_shell(b"./bsh/sh -x"));
        assert!(is_bourne_compatible_shell(b"./bsh/sh"));
        assert!(is_bourne_compatible_shell(b"./bsh/./sh"));
    }

    /// What a shell GNU Make cannot name is handed. `start_job_command`
    /// (job.c) takes the blanks and prefixes off the front of the script it is
    /// about to run and stops at the first newline, whatever the shell is;
    /// `construct_command_argv_internal` (job.c:3293) takes them off every
    /// line below that one only for a Bourne-compatible shell, because
    /// otherwise they could be the script's own.
    ///
    /// Measured against 4.4.1 with a `SHELL` that writes its argument to a
    /// file: `-@ touch m1` / `  @touch m2` / `    touch m3` arrives as
    /// `touch m1\n  @touch m2\n    touch m3`.
    #[test]
    fn an_unknown_shell_reads_a_one_shell_script_below_its_first_line() {
        assert_eq!(
            expanded_for_an_unknown_shell(b"  -@ touch m1\n  @touch m2\n    touch m3"),
            vec![
                "touch m1".to_owned(),
                "  @touch m2".to_owned(),
                "    touch m3".to_owned(),
            ]
        );
        // The same recipe for a shell on the list loses all three, which is the
        // reading every other recipe gets.
        assert_eq!(
            expanded(b"  -@ touch m1\n  @touch m2\n    touch m3")
                .into_iter()
                .map(|(text, ..)| text)
                .collect::<Vec<_>>(),
            vec![
                "touch m1".to_owned(),
                "touch m2".to_owned(),
                "touch m3".to_owned(),
            ]
        );
        // The recipe's first line is the first that carried a command: a line
        // of nothing but prefixes is not one, so the next line is still read.
        assert_eq!(
            expanded_for_an_unknown_shell(b"@\n  @touch m2\n  touch m3"),
            vec!["touch m2".to_owned(), "  touch m3".to_owned()]
        );
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
        scan_written_prefixes(&session.values, value, &mut prefixes);
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
            every_line(),
            false,
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

    /// The shell flags a line runs under say whether `-e` is armed before
    /// the line says anything, read as `sh` reads its option words.
    #[test]
    fn errexit_is_read_off_the_shell_flags() {
        assert!(!arms_errexit(b"-c"));
        assert!(arms_errexit(b"-ec"));
        assert!(arms_errexit(b"-e -c"));
        assert!(arms_errexit(b"-xec"));
        assert!(!arms_errexit(b"-ec +e"));
        assert!(!arms_errexit(b""));
        // Not seen, and not seeing it only ever refuses a lift.
        assert!(!arms_errexit(b"-o errexit -c"));
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
