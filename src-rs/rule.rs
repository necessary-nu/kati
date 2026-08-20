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

use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use memchr::memchr;

use crate::expr::Value;
use crate::loc::Loc;
use crate::session::Session;
use crate::strutil::{Pattern, makefile_word_scanner, trim_leading_curdir, trim_space};
use crate::symtab::Symbol;
use crate::{error_loc, warn_loc};

#[derive(Clone)]
pub struct Rule {
    pub outputs: Vec<Symbol>,
    pub inputs: Vec<Symbol>,
    pub order_only_inputs: Vec<Symbol>,
    pub output_patterns: Vec<Symbol>,
    pub validations: Vec<Symbol>,
    pub is_double_colon: bool,
    /// `&:` / `&::` makes every output one group produced by one recipe.
    pub is_grouped: bool,
    pub is_suffix_rule: bool,
    /// Set when `.SECONDEXPANSION` was declared before this rule was read.
    pub expand_again: bool,
    /// The dependency-name chain GNU Make's `new_pattern_rule` compares. An
    /// immediately parsed list contributes each dependency name; a deferred
    /// list contributes its entire retained text as one name.
    pub prerequisite_names: Vec<Symbol>,
    /// The prerequisite text as the first expansion left it, kept unparsed for
    /// the second one. Only a list that still has a `$` in it: everything else
    /// would expand to itself, and holding it back would hide it from the
    /// automatic variables a later list reads.
    pub deferred_prerequisites: Option<Bytes>,
    /// Set once the implicit search has filled this rule's `%` in for the name
    /// it matched, so nothing downstream substitutes into the result again.
    ///
    /// GNU Make's `pattern_search` builds the prerequisite names itself and
    /// hands the file a finished chain, which is why a stem is substituted
    /// exactly once however many `%` the pattern and the prerequisites hold
    /// between them. A rule that still carries its pattern text — an explicit
    /// or a static pattern rule — has not been through that search and is
    /// substituted where it is read instead.
    pub prerequisites_are_resolved: bool,
    pub cmds: Vec<Arc<Value>>,
    pub loc: Loc,
    pub cmd_loc: Option<Loc>,
}

impl Rule {
    pub fn new(loc: Loc, is_double_colon: bool, is_grouped: bool) -> Self {
        Self {
            outputs: Vec::new(),
            inputs: Vec::new(),
            order_only_inputs: Vec::new(),
            output_patterns: Vec::new(),
            validations: Vec::new(),
            is_double_colon,
            is_grouped,
            is_suffix_rule: false,
            expand_again: false,
            prerequisite_names: Vec::new(),
            deferred_prerequisites: None,
            prerequisites_are_resolved: false,
            cmds: Vec::new(),
            loc,
            cmd_loc: None,
        }
    }

    fn parse_inputs(&mut self, session: &mut Session, inputs_str: &Bytes) {
        let (inputs, order_only) = split_order_only(inputs_str);
        for input in file_sequence(&inputs) {
            let identity_start = self.inputs.len();
            glob_word(session, input.clone(), &mut self.inputs);
            if input.as_ref() != b".WAIT" {
                self.prerequisite_names
                    .extend_from_slice(&self.inputs[identity_start..]);
            }
        }
        for input in file_sequence(&order_only) {
            let identity_start = self.order_only_inputs.len();
            glob_word(session, input.clone(), &mut self.order_only_inputs);
            if input.as_ref() != b".WAIT" {
                self.prerequisite_names
                    .extend_from_slice(&self.order_only_inputs[identity_start..]);
            }
        }
    }

    fn defer_prerequisites(&mut self, session: &mut Session, prerequisites: Bytes) {
        self.prerequisite_names.clear();
        self.prerequisite_names
            .push(session.intern(prerequisites.clone()));
        self.deferred_prerequisites = Some(prerequisites);
    }

    pub fn parse_prerequisites(&mut self, session: &mut Session, line: &Bytes) -> Result<()> {
        // line is either
        //    prerequisites
        // or
        //    target-prerequisites : prereq-patterns
        // The evaluator has already separated an inline command at the point
        // GNU Make decides whether this is a target-specific assignment.
        let prereq_string = line.clone();

        let Some(separator_pos) = find_unescaped_colon(&prereq_string) else {
            // Simple prerequisites
            let prereq_string = normalize_prerequisites(prereq_string);
            if self.expand_again && memchr(b'$', &prereq_string).is_some() {
                self.defer_prerequisites(session, prereq_string);
            } else {
                self.parse_inputs(session, &prereq_string);
            }
            return Ok(());
        };

        // Static pattern rule. A rule whose targets are already patterns has
        // nowhere to put a second one, and GNU Make names that collision for
        // what it is rather than reusing the plain-target sentence.
        if !self.output_patterns.is_empty() {
            error_loc!(
                session,
                Some(&self.loc),
                "*** mixed implicit and static pattern rules"
            );
        }

        // Empty static patterns should not produce rules, but need to eat the
        // commands So return a rule with no outputs nor output_patterns
        if self.outputs.is_empty() {
            return Ok(());
        }

        let target_prereq = prereq_string.slice(..separator_pos);
        let prereq_patterns = normalize_prerequisites(prereq_string.slice(separator_pos + 1..));

        let patterns = makefile_word_scanner(&target_prereq)
            .map(|pattern| pattern.slice_ref(trim_leading_curdir(&pattern)))
            .collect::<Vec<_>>();
        for target_pattern in patterns {
            self.output_patterns.push(session.intern(target_pattern));
        }

        if self.output_patterns.is_empty() {
            error_loc!(session, Some(&self.loc), "*** missing target pattern.");
        }
        if self.output_patterns.len() > 1 {
            error_loc!(session, Some(&self.loc), "*** multiple target patterns.");
        }
        let target_pattern = *self.output_patterns.first().unwrap();
        // Whether the pattern holds a wildcard at all is `find_percent`'s
        // question and not "is there a percent here": a backslash escapes the
        // wildcard, so `\%.o` is the literal name `%.o` and carries none. GNU
        // Make asks it right here — `pattern_percent = find_percent_cached
        // (&target->name)` in read.c, refusing when the answer is null — and a
        // static pattern rule whose target pattern is not a pattern is not a
        // rule.
        let pat = Pattern::new(target_pattern.as_bytes(&*session));
        if !pat.is_pattern() {
            error_loc!(
                session,
                Some(&self.loc),
                "*** target pattern contains no '%'."
            );
        }

        // Whether a target matches is asked only once the rule itself is known
        // to be well formed. GNU Make settles the target pattern while parsing
        // the line and tests each target against it later, in `record_files`,
        // so a rule that names two patterns dies on that alone rather than
        // first complaining that neither of them matched anything.
        let unmatched = self
            .outputs
            .iter()
            .filter(|t| !pat.matches(&t.as_bytes(&*session)))
            .copied()
            .collect::<Vec<_>>();
        for target in unmatched {
            warn_loc!(
                session,
                Some(&self.loc),
                "target '{}' doesn't match the target pattern",
                target.display(&*session)
            );
        }
        if self.expand_again && memchr(b'$', &prereq_patterns).is_some() {
            self.defer_prerequisites(session, prereq_patterns);
        } else {
            self.parse_inputs(session, &prereq_patterns);
        }
        Ok(())
    }
}

/// Find the static-pattern separator. A colon is quoted only by an odd run of
/// immediately preceding backslashes; parentheses have no special meaning.
fn find_unescaped_colon(prerequisites: &[u8]) -> Option<usize> {
    let mut preceding_backslashes = 0;
    for (index, byte) in prerequisites.iter().enumerate() {
        if *byte == b'\\' {
            preceding_backslashes += 1;
            continue;
        }
        if *byte == b':' && preceding_backslashes % 2 == 0 {
            return Some(index);
        }
        preceding_backslashes = 0;
    }
    None
}

/// Apply the normalization GNU Make performs before either splitting an
/// immediate prerequisite list or retaining one for second expansion.
fn normalize_prerequisites(prerequisites: Bytes) -> Bytes {
    let prerequisites = prerequisites.slice_ref(trim_space(&prerequisites));
    if memchr(b'\\', &prerequisites).is_none() || memchr(b':', &prerequisites).is_none() {
        return prerequisites;
    }

    let mut normalized = Vec::with_capacity(prerequisites.len());
    let mut index = 0;
    while index < prerequisites.len() {
        if prerequisites[index] != b'\\' {
            normalized.push(prerequisites[index]);
            index += 1;
            continue;
        }

        let slash_start = index;
        while index < prerequisites.len() && prerequisites[index] == b'\\' {
            index += 1;
        }
        let slash_count = index - slash_start;
        if prerequisites.get(index) == Some(&b':') && slash_count % 2 == 1 {
            normalized
                .extend_from_slice(&prerequisites[slash_start..slash_start + slash_count / 2]);
        } else {
            normalized.extend_from_slice(&prerequisites[slash_start..index]);
        }
        if index < prerequisites.len() {
            normalized.push(prerequisites[index]);
            index += 1;
        }
    }
    normalized.into()
}

impl Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "outputs={:?} inputs={:?}", self.outputs, self.inputs)?;
        if !self.order_only_inputs.is_empty() {
            write!(f, " order_only_inputs={:?}", self.order_only_inputs)?;
        }
        if !self.output_patterns.is_empty() {
            write!(f, " output_patterns={:?}", self.output_patterns)?;
        }
        if self.is_double_colon {
            write!(f, " is_double_colon")?;
        }
        if self.is_grouped {
            write!(f, " is_grouped")?;
        }
        if self.is_suffix_rule {
            write!(f, " is_suffix_rule")?;
        }
        if !self.cmds.is_empty() {
            write!(f, " cmds={:?}", self.cmds)?;
        }
        Ok(())
    }
}

/// Split a prerequisite list at the first `|`, which ends the word it falls in
/// rather than needing space around it. After it there is no second list, so a
/// later `|` is an ordinary character.
pub fn split_order_only(inputs: &Bytes) -> (Bytes, Bytes) {
    match memchr(b'|', inputs) {
        Some(i) => (inputs.slice(..i), inputs.slice(i + 1..)),
        None => (inputs.clone(), Bytes::new()),
    }
}

/// Split a rule's target or prerequisite text into the names GNU Make reads
/// out of it.
///
/// GNU Make's `parse_file_seq` (reference/gnumake/src/read.c) in the order it
/// does the three things: split into words, strip a leading `./` from each,
/// and reopen any archive group so that `lib.a(a.o b.o)` names two members
/// rather than one whose name has a space in it.
///
/// Globbing is deliberately not here. It happens per name in [`glob_word`],
/// because an archive name globs in two places at once — the archive half
/// against the filesystem and the member half against the archive's index —
/// and the group has to be reopened before either can be asked about.
pub fn file_sequence(text: &Bytes) -> Vec<Bytes> {
    let words: Vec<Bytes> = makefile_word_scanner(text)
        .map(|word| word.slice_ref(trim_leading_curdir(&word)))
        .collect();
    // The group machinery is reached only for text holding a `(`, which is
    // GNU Make's own condition for looking and is false for almost every rule.
    if words.iter().any(|word| word.contains(&b'(')) {
        return crate::archive::reopen_groups(words);
    }
    words
}

/// Match one name of a target or prerequisite list against the filesystem, as
/// GNU Make does for any name holding `?`, `*` or `[`. A name matching nothing
/// is kept as it was written, which is how `%` survives to make a pattern rule
/// and how a refusal still names what the makefile asked for.
///
/// An archive name is matched in two halves, as `parse_file_seq` matches it:
/// the archive against the filesystem, and then the member against that
/// archive's index rather than against the filesystem again.
pub fn glob_word(session: &mut Session, word: Bytes, into: &mut Vec<Symbol>) {
    if !word.iter().any(|c| matches!(c, b'?' | b'*' | b'[')) {
        into.push(session.intern(word));
        return;
    }
    if let Some((archive, member)) = crate::archive::split_archive_name(&word) {
        let archive = word.slice_ref(archive);
        let member = word.slice_ref(member);
        // The archive half is an ordinary filename and globs like one; a
        // pattern matching nothing stands as written, so a member of an
        // archive that is not there yet is still asked for.
        let mut archives = Vec::new();
        glob_word_on_disk(session, archive, &mut archives);
        for archive in archives {
            match crate::archive::glob_members(&archive, &member) {
                Some(members) => into.extend(
                    members
                        .iter()
                        .map(|found| session.intern(crate::archive::member_name(&archive, found))),
                ),
                // No matches, or no pattern: the member name stands as
                // written, and the build refuses over it if it is not there.
                None => into.push(session.intern(crate::archive::member_name(&archive, &member))),
            }
        }
        return;
    }
    let mut matched = Vec::new();
    glob_word_on_disk(session, word, &mut matched);
    into.extend(matched.into_iter().map(|name| session.intern(name)));
}

/// The filesystem half, which is `glob()` and the rule that a pattern matching
/// nothing is kept as it was written.
fn glob_word_on_disk(session: &mut Session, word: Bytes, into: &mut Vec<Bytes>) {
    if !word.iter().any(|c| matches!(c, b'?' | b'*' | b'[')) {
        into.push(word);
        return;
    }
    // A name matched against the filesystem is a question the ground answers,
    // so a read repeating itself over the same text is told what the first
    // read was told. The names are recorded NUL-separated rather than
    // space-separated, because a filename may hold a space and this record is
    // not something a Makefile ever reads; an empty answer is no matches at
    // all, which is the pattern standing as it was written.
    if let Some(answered) = session
        .ground_journal
        .answered(crate::session::GroundQuestion::Glob, &word)
    {
        if answered.answer.is_empty() {
            into.push(word);
        } else {
            into.extend(
                answered
                    .answer
                    .split(|byte| *byte == 0)
                    .map(|name| answered.answer.slice_ref(name)),
            );
        }
        return;
    }
    let matched = match session.glob(word.clone()).as_ref() {
        Ok(paths) if !paths.is_empty() => paths.clone(),
        _ => Vec::new(),
    };
    let mut answer = bytes::BytesMut::new();
    for (position, name) in matched.iter().enumerate() {
        if position > 0 {
            bytes::BufMut::put_u8(&mut answer, 0);
        }
        bytes::BufMut::put_slice(&mut answer, name);
    }
    session.ground_journal.record(
        crate::session::GroundQuestion::Glob,
        word.clone(),
        answer.freeze(),
        None,
    );
    if matched.is_empty() {
        into.push(word);
    } else {
        into.extend(matched);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_prerequisite_normalization_matches_make() {
        assert_eq!(
            normalize_prerequisites(Bytes::from_static(b"  one  two \t")),
            Bytes::from_static(b"one  two")
        );
        assert_eq!(
            normalize_prerequisites(Bytes::from_static(br"one\:two")),
            Bytes::from_static(b"one:two")
        );
        assert_eq!(
            normalize_prerequisites(Bytes::from_static(br"one\\:two")),
            Bytes::from_static(br"one\\:two")
        );
        assert_eq!(
            normalize_prerequisites(Bytes::from_static(br"one\\\:two")),
            Bytes::from_static(br"one\:two")
        );
    }

    #[test]
    fn static_pattern_colon_only_honors_backslash_quoting() {
        assert_eq!(find_unescaped_colon(b"$(subst :,x,dep)"), Some(8));
        assert_eq!(find_unescaped_colon(br"dep\:name"), None);
        assert_eq!(find_unescaped_colon(br"dep\\:pattern"), Some(5));
        assert_eq!(find_unescaped_colon(br"dep\\\:name:pattern"), Some(11));
    }
}
