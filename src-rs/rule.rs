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
use crate::stmt::{RuleSep, RuleStmt};
use crate::strutil::{Pattern, trim_leading_curdir, word_scanner};
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
    pub is_suffix_rule: bool,
    /// Set when `.SECONDEXPANSION` was declared before this rule was read.
    pub expand_again: bool,
    /// The first-expanded prerequisite names used to identify a pattern-rule
    /// definition. The order-only marker changes how a name is scheduled, not
    /// its place in GNU Make's `new_pattern_rule` comparison.
    pub prerequisite_names: Vec<Symbol>,
    /// The prerequisite text as the first expansion left it, kept unparsed for
    /// the second one. Only a list that still has a `$` in it: everything else
    /// would expand to itself, and holding it back would hide it from the
    /// automatic variables a later list reads.
    pub deferred_prerequisites: Option<Bytes>,
    pub cmds: Vec<Arc<Value>>,
    pub loc: Loc,
    pub cmd_loc: Option<Loc>,
}

impl Rule {
    pub fn new(loc: Loc, is_double_colon: bool) -> Self {
        Self {
            outputs: Vec::new(),
            inputs: Vec::new(),
            order_only_inputs: Vec::new(),
            output_patterns: Vec::new(),
            validations: Vec::new(),
            is_double_colon,
            is_suffix_rule: false,
            expand_again: false,
            prerequisite_names: Vec::new(),
            deferred_prerequisites: None,
            cmds: Vec::new(),
            loc,
            cmd_loc: None,
        }
    }

    fn parse_inputs(&mut self, session: &mut Session, inputs_str: &Bytes) {
        let (inputs, order_only) = split_order_only(inputs_str);
        let inputs_start = self.inputs.len();
        for input in word_scanner(&inputs) {
            let word = inputs.slice_ref(trim_leading_curdir(input));
            glob_word(session, word, &mut self.inputs);
        }
        let order_only_start = self.order_only_inputs.len();
        for input in word_scanner(&order_only) {
            let word = order_only.slice_ref(trim_leading_curdir(input));
            glob_word(session, word, &mut self.order_only_inputs);
        }
        self.prerequisite_names
            .extend_from_slice(&self.inputs[inputs_start..]);
        self.prerequisite_names
            .extend_from_slice(&self.order_only_inputs[order_only_start..]);
    }

    /// Record the names in a list whose raw text must survive for second
    /// expansion. Splitting the words here makes whitespace and `|`
    /// classification independent from the bytes retained for later expansion.
    fn record_deferred_prerequisite_names(&mut self, session: &mut Session, inputs_str: &Bytes) {
        let (inputs, order_only) = split_order_only(inputs_str);
        for text in [inputs, order_only] {
            for input in word_scanner(&text) {
                let word = text.slice_ref(trim_leading_curdir(input));
                self.prerequisite_names.push(session.intern(word));
            }
        }
    }

    pub fn parse_prerequisites(
        &mut self,
        session: &mut Session,
        line: &Bytes,
        separator_pos: Option<usize>,
        rule_stmt: &RuleStmt,
    ) -> Result<()> {
        // line is either
        //    prerequisites [ ; command ]
        // or
        //    target-prerequisites : prereq-patterns [ ; command ]
        // First, separate command. At this point separator_pos should point to ';'
        // unless null.
        let mut prereq_string = line.clone();
        if let Some(separator_pos) = separator_pos
            && rule_stmt.sep != RuleSep::Semicolon
        {
            assert!(line[separator_pos] == b';');
            let value = line.slice(separator_pos + 1..);
            self.cmds.push(Arc::new(Value::Literal(None, value)));
            prereq_string = line.slice(..separator_pos);
        }

        let Some(separator_pos) = memchr(b':', &prereq_string) else {
            // Simple prerequisites
            if self.expand_again && memchr(b'$', &prereq_string).is_some() {
                self.record_deferred_prerequisite_names(session, &prereq_string);
                self.deferred_prerequisites = Some(prereq_string);
            } else {
                self.parse_inputs(session, &prereq_string);
            }
            return Ok(());
        };

        // Static pattern rule.
        if !self.output_patterns.is_empty() {
            error_loc!(
                session,
                Some(&self.loc),
                "*** mixed implicit and normal rules: deprecated syntax"
            );
        }

        // Empty static patterns should not produce rules, but need to eat the
        // commands So return a rule with no outputs nor output_patterns
        if self.outputs.is_empty() {
            return Ok(());
        }

        let target_prereq = prereq_string.slice(..separator_pos);
        let prereq_patterns = prereq_string.slice(separator_pos + 1..);

        let patterns = word_scanner(&target_prereq)
            .map(|p| target_prereq.slice_ref(trim_leading_curdir(p)))
            .collect::<Vec<_>>();
        for target_pattern in patterns {
            let pat = Pattern::new(target_pattern.clone());
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
                    "target `{}' doesn't match the target pattern",
                    target.display(&*session)
                );
            }
            self.output_patterns.push(session.intern(target_pattern));
        }

        if self.output_patterns.is_empty() {
            error_loc!(session, Some(&self.loc), "*** missing target pattern.");
        }
        if self.output_patterns.len() > 1 {
            error_loc!(session, Some(&self.loc), "*** multiple target patterns.");
        }
        if !is_pattern_rule(&self.output_patterns.first().unwrap().as_bytes(&*session)) {
            error_loc!(
                session,
                Some(&self.loc),
                "*** target pattern contains no '%'."
            );
        }
        if self.expand_again && memchr(b'$', &prereq_patterns).is_some() {
            self.record_deferred_prerequisite_names(session, &prereq_patterns);
            self.deferred_prerequisites = Some(prereq_patterns);
        } else {
            self.parse_inputs(session, &prereq_patterns);
        }
        Ok(())
    }
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

pub fn is_pattern_rule(target: &[u8]) -> bool {
    memchr(b'%', target).is_some()
}

/// Match one word of a target or prerequisite list against the filesystem, as
/// GNU Make does for any name holding `?`, `*` or `[`. A name matching nothing
/// is kept as it was written, which is how `%` survives to make a pattern rule
/// and how a refusal still names what the makefile asked for.
pub fn glob_word(session: &mut Session, word: Bytes, into: &mut Vec<Symbol>) {
    if !word.iter().any(|c| matches!(c, b'?' | b'*' | b'[')) {
        into.push(session.intern(word));
        return;
    }
    let matched = session.glob(word.clone());
    match matched.as_ref() {
        Ok(paths) if !paths.is_empty() => {
            for path in paths {
                let sym = session.intern(path.clone());
                into.push(sym);
            }
        }
        _ => into.push(session.intern(word)),
    }
}
