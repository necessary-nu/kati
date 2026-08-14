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
use bytes::{Buf, BufMut, Bytes, BytesMut};
use memchr::memchr;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    os::unix::ffi::OsStrExt,
    path::PathBuf,
    sync::Arc,
};

use crate::{
    error_loc,
    eval::{Evaluator, FrameType, MissingInclude, ReadMakefile, ScopedFrame},
    expr::{Evaluable, Value},
    loc::Loc,
    log,
    rule::{Rule, glob_word, is_pattern_rule, split_order_only},
    session::{Context, Session},
    stmt::AssignOp,
    strutil::{
        Pattern, WordWriter, get_ext, is_space_byte, makefile_word_scanner, strip_ext,
        trim_leading_curdir, word_scanner,
    },
    symtab::{Interner, Symbol},
    timeutil::ScopedTimeReporter,
    var::{ScopedVar, Var, Variable, Vars},
    warn_loc,
};

pub type NamedDepNode = (Symbol, Arc<Mutex<DepNode>>);

/// Undo scope bindings the way a scope unwinds: last installed, first removed.
///
/// `Vec`'s own drop runs front to back, which is wrong as soon as two of them
/// bind the same name — as two matching pattern scopes do. The outer binding
/// would be restored first, and the guard that shadowed it would then restore
/// what *it* replaced, leaving the inner value behind for whatever is built
/// next.
fn unbind(mut bindings: Vec<ScopedVar>) {
    while bindings.pop().is_some() {}
}

/// One Makefile the read consulted that a rule says how to remake.
///
/// `required` is what `include` said and `-include` did not. It travels with
/// the node because a frontend that builds these roots has to know which
/// failures end the run: GNU Make abandons a build over a Makefile it cares
/// about and says nothing at all about one it does not.
pub struct RegenerationRoot {
    pub node: NamedDepNode,
    pub required: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DoubleActionId {
    rule: usize,
    /// An ordinary multi-target `::` record is one action per member. A
    /// grouped record has one action for the whole record.
    trigger: Option<Symbol>,
}

#[derive(Clone, Debug)]
pub struct GroupedDoubleAction {
    /// Every real filesystem member declared by this exact `&::` record.
    pub members: Vec<Symbol>,
    /// A phony member forces the whole record whenever any member is reached.
    pub has_phony_member: bool,
    /// Normal prerequisites declared phony are always present in `$?`.
    pub phony_inputs: Vec<Symbol>,
}

/// The cycle guard cannot catch `%.a: %.b.a` against `%.b.a: %.a`, where every
/// name visited is new. The deepest chain in GNU Make's suite is three.
const MAX_IMPLICIT_CHAIN: usize = 6;

#[derive(Debug)]
pub struct DepNode {
    /// The graph edge's primary output. An exact grouped action uses a private
    /// virtual name here so independent records never compete for a member.
    pub output: Symbol,
    /// The logical Make target used by automatic variables and diagnostics.
    pub recipe_output: Symbol,
    /// Runtime freshness metadata for one exact grouped double-colon record.
    pub grouped_double_action: Option<GroupedDoubleAction>,
    /// A public member joining every independent action that declares it.
    pub grouped_double_join: bool,
    pub cmds: Vec<Arc<Value>>,
    pub deps: Vec<NamedDepNode>,
    pub order_onlys: Vec<NamedDepNode>,
    pub validations: Vec<NamedDepNode>,
    pub has_rule: bool,
    /// Whether this node is the first rule of the read. Read only where a
    /// manifest needs a `default` line and the goals cannot supply one —
    /// `--gen_all_targets`, where they are all of them.
    pub is_default_target: bool,
    pub is_phony: bool,
    /// At least one ordinary `::` recipe has no prerequisites. GNU Make runs
    /// that action whenever the target is considered, even when the file is
    /// otherwise current.
    pub unconditional_double_colon: bool,
    pub is_restat: bool,
    /// `.IGNORE` named this target: a failing recipe line is not a failure.
    pub is_ignore_error: bool,
    /// This file's absence is no reason to remake what reads it: the implicit
    /// rule search invented the name to complete a chain, or `.INTERMEDIATE`
    /// or `.SECONDARY` said so.
    pub is_intermediate: bool,
    /// The build deletes this file once it has finished with it, which every
    /// intermediate but a `.SECONDARY` one and a goal is.
    pub is_disposable: bool,
    /// The outputs of this action a failed recipe leaves half-made, which
    /// `.DELETE_ON_ERROR` says must not be left behind.
    ///
    /// Empty unless the Makefile declared `.DELETE_ON_ERROR`, and then only the
    /// outputs that survive the exclusions: `.PRECIOUS` protects a name from
    /// deletion, and a `.PHONY` name stands for no file to delete.
    pub delete_on_error_outputs: Vec<Symbol>,
    pub implicit_outputs: Vec<Symbol>,
    /// The outputs among [`Self::implicit_outputs`] that this recipe makes only
    /// on the way to making something else — GNU Make's `also_make`.
    ///
    /// A pattern rule spelling several target patterns is one recipe for all of
    /// them, but GNU Make still decides each name's freshness from that name
    /// alone: the peer of the target the search matched is entered as a target
    /// of its own (`implicit.c` sets `is_target`), which keeps it out of the
    /// intermediate sweep, and is otherwise only marked updated when the recipe
    /// runs. So a peer nothing asked for neither forces the recipe by being
    /// absent nor is swept up afterwards. A name that is later asked for in its
    /// own right stops being one of these.
    pub peer_outputs: Vec<Symbol>,
    pub actual_inputs: Vec<Symbol>,
    pub actual_order_only_inputs: Vec<Symbol>,
    pub actual_validations: Vec<Symbol>,
    pub rule_vars: Option<Arc<Vars>>,
    pub depfile_var: Option<Var>,
    pub ninja_pool_var: Option<Var>,
    pub tags_var: Option<Var>,
    pub output_pattern: Option<Symbol>,
    pub loc: Option<Loc>,
}

impl DepNode {
    fn new(
        output: Symbol,
        is_phony: bool,
        is_restat: bool,
        is_ignore_error: bool,
        is_intermediate: bool,
        is_disposable: bool,
    ) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            output,
            recipe_output: output,
            grouped_double_action: None,
            grouped_double_join: false,
            cmds: Vec::new(),
            deps: Vec::new(),
            order_onlys: Vec::new(),
            validations: Vec::new(),
            has_rule: false,
            is_default_target: false,
            is_phony,
            unconditional_double_colon: false,
            is_restat,
            is_ignore_error,
            is_intermediate,
            is_disposable,
            delete_on_error_outputs: Vec::new(),
            implicit_outputs: Vec::new(),
            peer_outputs: Vec::new(),
            actual_inputs: Vec::new(),
            actual_order_only_inputs: Vec::new(),
            actual_validations: Vec::new(),
            rule_vars: None,
            depfile_var: None,
            ninja_pool_var: None,
            tags_var: None,
            output_pattern: None,
            loc: None,
        }))
    }
}

fn replace_suffix(session: &mut Session, s: Symbol, newsuf: &Symbol) -> Symbol {
    let s = s.as_bytes(&*session);
    let s = strip_ext(&s);
    let newsuf = newsuf.as_bytes(&*session);
    let mut r = BytesMut::with_capacity(s.len() + newsuf.len() + 1);
    r.put_slice(s);
    r.put_u8(b'.');
    r.put_slice(&newsuf);
    session.intern(r.freeze())
}

/// Rewrite a deferred prerequisite's `%` to `$*` ahead of the second
/// expansion, the first one of each whitespace-separated token as GNU Make
/// does. Substituting the stem itself would expand it a third time, which is
/// wrong for a stem containing `$`.
fn stem_references(text: &Bytes) -> Bytes {
    if memchr(b'%', text).is_none() {
        return text.clone();
    }
    let mut ret = BytesMut::with_capacity(text.len() + 8);
    let mut substituted = false;
    for &c in text.iter() {
        match c {
            b'%' if !substituted => {
                ret.put_slice(b"$*");
                substituted = true;
            }
            _ => {
                if c.is_ascii_whitespace() {
                    substituted = false;
                }
                ret.put_u8(c);
            }
        }
    }
    ret.freeze()
}

/// Split the retained prerequisite text of an implicit pattern rule the way
/// GNU Make's `get_next_word` does before second expansion. A raw backslash
/// does not quote a blank at this stage. Variable references stay whole, and a
/// pipe ends the current chunk so expansion can still decide whether it is an
/// order-only separator.
fn implicit_prerequisite_words(source: &Bytes) -> impl Iterator<Item = Bytes> + '_ {
    let mut index = 0usize;
    std::iter::from_fn(move || {
        while source.get(index).is_some_and(is_space_byte) {
            index += 1;
        }
        if index == source.len() {
            return None;
        }

        let start = index;
        while let Some(&byte) = source.get(index) {
            match byte {
                b' ' | b'\t' => break,
                b'|' => {
                    index += 1;
                    break;
                }
                b'$' => {
                    index += 1;
                    let Some(&open) = source.get(index) else {
                        break;
                    };
                    index += 1;
                    if open == b'$' {
                        continue;
                    }
                    let close = match open {
                        b'(' => b')',
                        b'{' => b'}',
                        _ => continue,
                    };
                    let mut depth = 0usize;
                    while let Some(&inner) = source.get(index) {
                        index += 1;
                        if inner == open {
                            depth += 1;
                        } else if inner == close {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        }
                    }
                }
                _ => index += 1,
            }
        }
        Some(source.slice(start..index))
    })
}

/// Whether a rule's prerequisites reach `output` at all.
///
/// A static pattern rule records its prerequisites per target, and GNU Make
/// records them only for the targets the target pattern matched: `record_files`
/// reaches the copy under an `else` to the mismatch diagnostic, so a target
/// that missed the pattern keeps the recipe and gets a stem but is left with an
/// empty prerequisite chain. Deferred prerequisites are that same chain held
/// back for `.SECONDEXPANSION:`, so they are dropped on the same terms.
fn prerequisites_reach(session: &Session, r: &Rule, output: Symbol) -> bool {
    if r.is_suffix_rule {
        return true;
    }
    let Some(pattern) = r.output_patterns.first() else {
        return true;
    };
    Pattern::new(pattern.as_bytes(session)).matches(&output.as_bytes(session))
}

/// The directories a search path names.
///
/// A search path is a list of directories rather than of strings, so a lone `.`
/// says nothing and a trailing slash is not part of a directory's name — GNU
/// Make's `construct_vpath_list` drops both, and `gpath_search` then compares
/// what is left byte for byte against the directory a name was found in.
fn search_path(value: &Bytes) -> Vec<Bytes> {
    crate::strutil::word_scanner(value)
        .flat_map(|word| word.split(|byte| *byte == b':'))
        .filter(|directory| !directory.is_empty())
        .map(|directory| {
            let mut directory = value.slice_ref(directory);
            if directory.len() > 1 && directory.ends_with(b"/") {
                directory.truncate(directory.len() - 1);
            }
            directory
        })
        .filter(|directory| directory.as_ref() != b".")
        .collect()
}

/// The directory a search joined to `name` to arrive at `found`.
///
/// Measured off the end rather than by looking for the last slash in the found
/// path, which is how GNU Make measures it: what a search path is asked about
/// is the entry it looked in, so a name carrying a directory of its own is
/// answered by that entry alone and not by the whole prefix the join produced.
fn search_directory<'a>(found: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    found
        .len()
        .checked_sub(name.len() + 1)
        .map(|end| &found[..end])
}

fn apply_output_pattern(
    session: &mut Session,
    r: &Rule,
    output: Symbol,
    inputs: &[Symbol],
) -> Vec<Symbol> {
    let mut ret = Vec::new();
    if inputs.is_empty() {
        return ret;
    }
    if !prerequisites_reach(session, r, output) {
        return ret;
    }
    if r.is_suffix_rule {
        for input in inputs {
            ret.push(replace_suffix(session, output, input));
        }
        return ret;
    }
    if r.output_patterns.is_empty() {
        ret.extend(inputs);
        return ret;
    }
    assert!(r.output_patterns.len() == 1);
    let pat = Pattern::new(r.output_patterns[0].as_bytes(&*session));
    let output_str = output.as_bytes(&*session);
    for input in inputs {
        let buf = pat.append_subst(&output_str, &input.as_bytes(&*session));
        ret.push(session.intern(buf));
    }
    ret
}

/// One target pattern of one pattern rule, as the search considers it.
///
/// A rule with several target patterns is several candidates: GNU Make's
/// `pattern_search` records one `tryrule` per target that matches, so which
/// of a rule's patterns matched is part of the candidate rather than something
/// recovered afterwards.
#[derive(Clone)]
struct ImplicitCandidate {
    rule: Arc<Rule>,
    /// The rule's own target pattern this candidate was reached through.
    pattern: Symbol,
    /// Where the rule was written, counting one per target pattern, which is
    /// what breaks a tie between two rules that match a target equally well.
    order: usize,
}

struct RuleTrieEntry {
    candidate: ImplicitCandidate,
    suffix: Vec<u8>,
}

struct RuleTrie {
    rules: Vec<RuleTrieEntry>,
    children: HashMap<u8, RuleTrie>,
}

impl RuleTrie {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            children: HashMap::new(),
        }
    }

    fn add(&mut self, name: &[u8], candidate: ImplicitCandidate) {
        if name.is_empty() || name.starts_with(b"%") {
            self.rules.push(RuleTrieEntry {
                candidate,
                suffix: name.to_vec(),
            });
            return;
        }
        let c = name[0];
        self.children
            .entry(c)
            .or_insert_with(RuleTrie::new)
            .add(&name[1..], candidate)
    }

    fn get(&self, name: &[u8]) -> Vec<ImplicitCandidate> {
        let mut ret = Vec::new();
        for ent in &self.rules {
            if (ent.suffix.is_empty() && name.is_empty()) || name.ends_with(&ent.suffix[1..]) {
                ret.push(ent.candidate.clone())
            }
        }
        if name.is_empty() {
            return ret;
        }
        let c = name[0];
        if let Some(child) = self.children.get(&c) {
            ret.extend(child.get(&name[1..]));
        }
        ret
    }

    fn len(&self) -> usize {
        self.rules.len() + self.children.values().map(|c| c.len()).sum::<usize>()
    }

    fn remove_rule(&mut self, rule: &Arc<Rule>) {
        self.rules
            .retain(|entry| !Arc::ptr_eq(&entry.candidate.rule, rule));
        self.children.retain(|_, child| {
            child.remove_rule(rule);
            !child.rules.is_empty() || !child.children.is_empty()
        });
    }
}

/// GNU Make's `new_pattern_rule` compares one dependency-name chain across
/// both paths. Immediate prerequisites contribute their parsed names, while a
/// list retained for second expansion contributes its whole text as one name.
fn pattern_rule_prerequisites_match(rule: &Rule, existing: &Rule) -> bool {
    rule.prerequisite_names == existing.prerequisite_names
}

/// Whether GNU Make's `new_pattern_rule` removes `existing` for `rule`.
///
/// Its nested target loop is deliberately asymmetric: every target of the old
/// rule must equal one target of the new rule. In ordinary rules that means a
/// later grouped rule containing an older single target replaces it, while the
/// reverse does not. Replacement happens while the rule list is populated, so
/// the new rule moves to the end of that list before any target is searched.
fn replaces_pattern_rule(rule: &Rule, existing: &Rule) -> bool {
    pattern_rule_prerequisites_match(rule, existing) && pattern_rule_targets_match(rule, existing)
}

/// The target half of that comparison, on its own because the suffix-rule path
/// asks the same question of a prerequisite it has to spell out first.
fn pattern_rule_targets_match(rule: &Rule, existing: &Rule) -> bool {
    rule.output_patterns.iter().any(|target| {
        existing
            .output_patterns
            .iter()
            .all(|existing_target| existing_target == target)
    })
}

/// Whether a written pattern rule already holds the identity a suffix rule
/// would take.
///
/// GNU Make turns suffix rules into pattern rules once every makefile has been
/// read, and installs each one with `new_pattern_rule`'s override off: a rule
/// already written with that target and those prerequisites keeps the identity
/// and the suffix-derived one is thrown away. That is the other direction of
/// the same comparison, and it is how a recipe-less `%.tex: %.w` cancels
/// `.w.tex:` — the rule the search would otherwise have used never arrives.
fn pattern_rule_holds_suffix_rule(
    names: &impl Interner,
    existing: &Rule,
    suffix_rule: &Rule,
) -> bool {
    let [input] = suffix_rule.inputs.as_slice() else {
        return false;
    };
    let [prerequisite] = existing.prerequisite_names.as_slice() else {
        return false;
    };
    let input = input.as_bytes(names);
    let mut written = BytesMut::with_capacity(input.len() + 2);
    written.put_slice(b"%.");
    written.put_slice(&input);
    prerequisite.as_bytes(names) == written.freeze()
        && pattern_rule_targets_match(suffix_rule, existing)
}

fn is_suffix_rule(names: &impl Interner, output: &Symbol) -> bool {
    if !is_special_target(names, output) {
        return false;
    }
    let mut output = output.as_bytes(names);
    output.advance(1);
    let dot_index = memchr(b'.', &output);
    // If there is only a single dot or the third dot, this is not a
    // suffix rule.
    if let Some(dot_index) = dot_index {
        if memchr(b'.', &output[dot_index + 1..]).is_some() {
            return false;
        }
    } else {
        return false;
    }
    true
}

#[derive(Debug)]
struct RuleMerger {
    rules: Vec<Arc<Rule>>,
    implicit_outputs: Vec<(Symbol, Arc<Mutex<RuleMerger>>)>,
    validations: Vec<Symbol>,
    primary_rule: Option<Arc<Rule>>,
    parent: Option<Arc<Mutex<RuleMerger>>>,
    parent_sym: Option<Symbol>,
    is_double_colon: bool,
}

impl RuleMerger {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            rules: Vec::new(),
            implicit_outputs: Vec::new(),
            validations: Vec::new(),
            primary_rule: None,
            parent: None,
            parent_sym: None,
            is_double_colon: false,
        }))
    }

    fn add_implicit_output(&mut self, output: Symbol, merger: Arc<Mutex<RuleMerger>>) {
        self.implicit_outputs.push((output, merger))
    }

    fn add_validation(&mut self, validation: Symbol) {
        self.validations.push(validation)
    }

    fn set_implicit_output(
        &mut self,
        ctx: &impl Context,
        output: Symbol,
        p: Symbol,
        merger: Arc<Mutex<RuleMerger>>,
    ) -> Result<()> {
        {
            let merger = merger.lock();
            if merger.primary_rule.is_none() {
                error_loc!(
                    ctx,
                    None,
                    "*** implicit output `{}' on phony target `{}'",
                    output.display(ctx),
                    p.display(ctx)
                );
            }
            if let Some(parent) = &self.parent {
                let parent = parent.lock();
                error_loc!(
                    ctx,
                    merger
                        .primary_rule
                        .as_ref()
                        .and_then(|r| r.cmd_loc.clone())
                        .as_ref(),
                    "*** implicit output `{}' of `{}' was already defined by `{}' at {}",
                    output.display(ctx),
                    p.display(ctx),
                    self.parent_sym.unwrap().display(ctx),
                    parent
                        .primary_rule
                        .as_ref()
                        .and_then(|r| r.cmd_loc.clone())
                        .unwrap_or_default()
                        .display(ctx)
                );
            }
            if let Some(primary_rule) = &self.primary_rule {
                error_loc!(
                    ctx,
                    primary_rule.cmd_loc.as_ref(),
                    "*** implicit output `{}' may not have commands",
                    output.display(ctx)
                );
            }
        }
        self.parent = Some(merger);
        self.parent_sym = Some(p);
        Ok(())
    }

    fn add_rule(&mut self, ctx: &impl Context, output: Symbol, r: Arc<Rule>) -> Result<()> {
        if self.rules.is_empty() {
            self.is_double_colon = r.is_double_colon
        } else if self.is_double_colon != r.is_double_colon {
            error_loc!(
                ctx,
                Some(&r.loc),
                "*** target file `{}' has both : and :: entries.",
                output.display(ctx)
            );
        }

        if let Some(primary_rule) = &mut self.primary_rule
            && !r.cmds.is_empty()
            && !is_suffix_rule(ctx, &output)
            && !r.is_double_colon
        {
            if ctx.flags().werror_overriding_commands {
                error_loc!(
                    ctx,
                    r.cmd_loc.as_ref(),
                    "*** overriding commands for target `{}', previously defined at {}",
                    output.display(ctx),
                    primary_rule
                        .cmd_loc
                        .clone()
                        .unwrap_or_default()
                        .display(ctx)
                );
            } else {
                warn_loc!(
                    ctx,
                    r.cmd_loc.as_ref(),
                    "warning: overriding commands for target `{}'",
                    output.display(ctx)
                );
                warn_loc!(
                    ctx,
                    primary_rule.cmd_loc.as_ref(),
                    "warning: ignoring old commands for target `{}'",
                    output.display(ctx)
                )
            }
            *primary_rule = r.clone();
        }
        if self.primary_rule.is_none() && !r.cmds.is_empty() {
            self.primary_rule = Some(r.clone());
        }
        self.rules.push(r);
        Ok(())
    }

    fn fill_dep_node_from_rule(
        &self,
        session: &mut Session,
        output: Symbol,
        r: &Rule,
        n: &mut DepNode,
    ) {
        if self.is_double_colon {
            n.cmds.extend(r.cmds.iter().cloned());
        }

        n.actual_inputs
            .extend(apply_output_pattern(session, r, output, &r.inputs));
        n.actual_order_only_inputs.extend(apply_output_pattern(
            session,
            r,
            output,
            &r.order_only_inputs,
        ));

        if !r.output_patterns.is_empty() {
            assert!(r.output_patterns.len() == 1);
            n.output_pattern = Some(r.output_patterns[0]);
        }
    }

    fn fill_grouped_outputs(&self, output: Symbol, rule: &Rule, node: &mut DepNode) {
        if !rule.is_grouped {
            return;
        }
        for grouped_output in &rule.outputs {
            if *grouped_output != output && !node.implicit_outputs.contains(grouped_output) {
                node.implicit_outputs.push(*grouped_output);
            }
        }
    }

    fn fill_dep_node_loc(&self, r: &Rule, n: &mut DepNode) {
        n.loc = Some(r.loc.clone());
        if !r.cmds.is_empty()
            && let Some(cmd_loc) = r.cmd_loc.clone()
        {
            n.loc = Some(cmd_loc);
        }
    }

    fn fill_dep_node(
        &self,
        session: &mut Session,
        output: Symbol,
        pattern_rule: &Option<Arc<Rule>>,
        grouped_outputs: &[Symbol],
        n: &Arc<Mutex<DepNode>>,
    ) {
        let mut n = n.lock();
        if let Some(primary_rule) = &self.primary_rule {
            assert!(pattern_rule.is_none());
            self.fill_dep_node_from_rule(session, output, primary_rule, &mut n);
            if primary_rule.is_grouped && !primary_rule.is_double_colon {
                for grouped_output in grouped_outputs {
                    if *grouped_output != output && !n.implicit_outputs.contains(grouped_output) {
                        n.implicit_outputs.push(*grouped_output);
                    }
                }
            } else {
                self.fill_grouped_outputs(output, primary_rule, &mut n);
            }
            self.fill_dep_node_loc(primary_rule, &mut n);
            n.cmds = primary_rule.cmds.clone();
        } else if let Some(pattern_rule) = pattern_rule {
            self.fill_dep_node_from_rule(session, output, pattern_rule, &mut n);
            self.fill_dep_node_loc(pattern_rule, &mut n);
            n.cmds = pattern_rule.cmds.clone();
        }

        for r in &self.rules {
            if let Some(primary_rule) = &self.primary_rule
                && Arc::ptr_eq(r, primary_rule)
            {
                continue;
            }
            self.fill_dep_node_from_rule(session, output, r, &mut n);
            if self.is_double_colon {
                self.fill_grouped_outputs(output, r, &mut n);
            }
            if n.loc.is_none() {
                n.loc = Some(r.loc.clone())
            }
        }

        let mut all_outputs = HashSet::new();
        all_outputs.insert(output);

        for (sym, merger) in &self.implicit_outputs {
            n.implicit_outputs.push(*sym);
            all_outputs.insert(*sym);
            let merger = merger.lock();
            for r in &merger.rules {
                self.fill_dep_node_from_rule(session, output, r, &mut n);
            }
        }

        for validation in &self.validations {
            n.actual_validations.push(*validation)
        }
    }
}

type SuffixRuleMap = HashMap<Bytes, Vec<Arc<Rule>>>;

struct DepBuilder<'a> {
    ev: &'a mut Evaluator,
    rules: HashMap<Symbol, Arc<Mutex<RuleMerger>>>,
    rule_vars: HashMap<Symbol, Arc<Vars>>,
    /// The pattern keys of `rule_vars` in the order GNU Make would reach them:
    /// shortest pattern first, and among patterns of one length, the order they
    /// were written. Every entry matching a target applies, and a later one
    /// outranks an earlier one, so a longer pattern — which is to say the one
    /// leaving the shorter stem — wins.
    pattern_var_order: Vec<(Symbol, Pattern)>,
    cur_rule_vars: Option<Arc<Vars>>,
    /// Every explicit double-colon record is an independent action. Grouped
    /// records can share a real member, so the graph needs the full membership
    /// set before assigning producers.
    double_memberships: HashMap<Symbol, Vec<Arc<Rule>>>,
    /// One action node per exact double-colon action: one per grouped record,
    /// or one per member of an ordinary multi-target record.
    double_actions: HashMap<DoubleActionId, Arc<Mutex<DepNode>>>,
    /// Invocation-local creation order, used to serialize overlapping records
    /// in the same order GNU Make reaches them.
    double_action_creation_indices: HashMap<DoubleActionId, usize>,
    next_double_action_creation: usize,
    /// Stable evaluation-order identity for collision-free private outputs.
    double_action_indices: HashMap<DoubleActionId, usize>,
    next_double_action: usize,

    implicit_rules: RuleTrie,
    /// Pattern rules still present after GNU Make's population-time
    /// `new_pattern_rule` replacement.
    implicit_rule_defs: Vec<Arc<Rule>>,
    /// How many target patterns have been recorded, which is the next one's
    /// place in the order they were written.
    implicit_rule_order: usize,
    /// One second expansion per rule and target, as GNU Make does. The search
    /// makes two passes over the same rules and probes a rule again before
    /// using it, and the expansion is free to have side effects. Keyed by the
    /// candidate's definition order and requested output. The candidate order
    /// distinguishes two target patterns of the same rule, including duplicate
    /// patterns whose expansions can have side effects.
    expanded: HashMap<(usize, Symbol), (Vec<Symbol>, Vec<Symbol>)>,
    /// Cycle guard for the recursive implicit rule search.
    chaining: HashSet<Symbol>,
    /// Names the search invented to complete a chain, which the Makefile
    /// therefore never says.
    intermediates: HashSet<Symbol>,
    /// What `.INTERMEDIATE` and `.SECONDARY` named outright, which outranks
    /// every reason a name might have not to be intermediate.
    declared_intermediate: HashSet<Symbol>,
    /// The targets `.SECONDARY` named, which are intermediate without the
    /// deletion. Empty when it named none, which is the form that means every
    /// target and sets `all_secondary` instead.
    secondary: HashSet<Symbol>,
    all_secondary: bool,
    /// What `.NOTINTERMEDIATE` named, by name and by pattern, and whether it
    /// named nothing at all — which is every target.
    not_intermediate: HashSet<Symbol>,
    not_intermediate_patterns: Vec<Symbol>,
    no_intermediates: bool,
    /// Every name an explicit rule writes down as a prerequisite. A name the
    /// Makefile says is not intermediate however the search reached it, and a
    /// pattern is not a name.
    mentioned: HashSet<Symbol>,
    wait_sym: Symbol,
    /// Each prerequisite that followed a `.WAIT`, with what preceded it.
    wait_barriers: Vec<(Symbol, Vec<Symbol>)>,
    /// The recipe `.DEFAULT` offers for a target with no rule of its own.
    default_rule: Option<Arc<Rule>>,
    suffix_rules: SuffixRuleMap,

    /// The first target of the read that could stand for the Makefile as a
    /// whole, from before a goal could be named.
    ///
    /// `.DEFAULT_GOAL` decides what an invocation naming no goal builds, and it
    /// is answered by the evaluation rather than from here. This survives for
    /// `--gen_all_targets`, where every root is a target and the manifest still
    /// wants one of them written on its `default` line.
    first_rule: Option<Symbol>,
    done: HashMap<Symbol, Arc<Mutex<DepNode>>>,
    phony: HashSet<Symbol>,
    restat: HashSet<Symbol>,
    /// The targets `.IGNORE` named. Empty when it named none, which is the
    /// form that means every target and sets the flag instead.
    ignore_errors: HashSet<Symbol>,
    /// The Makefile declared `.DELETE_ON_ERROR`, which is one global answer:
    /// GNU Make reads the name once, as a target rather than a prerequisite,
    /// and any prerequisites it was given mean nothing.
    delete_on_error: bool,
    /// The names `.PRECIOUS` protects from deletion, and the target patterns it
    /// protects.
    ///
    /// The two are not one list under different spellings. A pattern protects a
    /// name only when an implicit rule whose target pattern is written exactly
    /// that way is the rule that made it, so `.PRECIOUS: %.bar` says nothing
    /// about a `foo.bar` an explicit rule built. Matching the pattern against
    /// the finished name instead would protect both, and GNU Make protects
    /// neither.
    precious: HashSet<Symbol>,
    precious_patterns: HashSet<Symbol>,
    depfile_var_name: Symbol,
    /// `VPATH`, the variable form of the directory search.
    vpath_var_name: Symbol,
    /// `GPATH`, which says that a directory the search looks in is also a
    /// directory a target found there is remade in.
    gpath_var_name: Symbol,
    /// The directories `GPATH` names, as GNU Make's `construct_vpath_list`
    /// leaves them.
    ///
    /// Read once, when the whole read has finished, because that is when
    /// `build_vpath_lists` reads them and nothing after it can change them.
    gpaths: Vec<Bytes>,
    /// The name a target renamed into its `GPATH` directory was written as.
    ///
    /// GNU Make's `rename_file` moves the file object rather than copying a
    /// string, so the rule the Makefile declared for the written name goes on
    /// making the found path. Kati keys rules by name, so the found path needs
    /// a way back to the name that carries its rule.
    gpath_origin: HashMap<Symbol, Symbol>,
    implicit_outputs_var_name: Symbol,
    ninja_pool_var_name: Symbol,
    validations_var_name: Symbol,
    tags_var_name: Symbol,
}

#[derive(Debug)]
struct PickedRuleInfo {
    merger: Option<Arc<Mutex<RuleMerger>>>,
    pattern_rule: Option<Arc<Rule>>,
    /// Weakest first. See `DepBuilder::applicable_rule_vars`.
    vars: Vec<Arc<Vars>>,
}

impl<'a> DepBuilder<'a> {
    fn new(ev: &'a mut Evaluator) -> Result<Self> {
        let rule_vars = std::mem::take(&mut ev.rule_vars);
        let mut pattern_var_order = std::mem::take(&mut ev.pattern_rule_var_order)
            .into_iter()
            .map(|sym| {
                let text = sym.as_bytes(&ev.session);
                (sym, Pattern::new(text))
            })
            .collect::<Vec<_>>();
        // Stable, so patterns of equal length keep the order they were written.
        pattern_var_order.sort_by_key(|(_, pattern)| pattern.as_bytes().len());
        let depfile_var_name = ev.session.intern(".KATI_DEPFILE");
        let vpath_var_name = ev.session.intern("VPATH");
        let gpath_var_name = ev.session.intern("GPATH");
        let implicit_outputs_var_name = ev.session.intern(".KATI_IMPLICIT_OUTPUTS");
        let ninja_pool_var_name = ev.session.intern(".KATI_NINJA_POOL");
        let validations_var_name = ev.session.intern(".KATI_VALIDATIONS");
        let tags_var_name = ev.session.intern(".KATI_TAGS");
        let wait_sym = ev.session.intern(".WAIT");
        let mut ret = Self {
            ev,
            rules: HashMap::new(),
            rule_vars,
            pattern_var_order,
            cur_rule_vars: None,
            double_memberships: HashMap::new(),
            double_actions: HashMap::new(),
            double_action_creation_indices: HashMap::new(),
            next_double_action_creation: 0,
            double_action_indices: HashMap::new(),
            next_double_action: 0,

            implicit_rules: RuleTrie::new(),
            implicit_rule_defs: Vec::new(),
            implicit_rule_order: 0,
            expanded: HashMap::new(),
            chaining: HashSet::new(),
            intermediates: HashSet::new(),
            declared_intermediate: HashSet::new(),
            secondary: HashSet::new(),
            all_secondary: false,
            not_intermediate: HashSet::new(),
            not_intermediate_patterns: Vec::new(),
            no_intermediates: false,
            mentioned: HashSet::new(),
            wait_sym,
            wait_barriers: Vec::new(),
            default_rule: None,
            suffix_rules: HashMap::new(),

            first_rule: None,
            done: HashMap::new(),
            phony: HashSet::new(),
            restat: HashSet::new(),
            ignore_errors: HashSet::new(),
            delete_on_error: false,
            precious: HashSet::new(),
            precious_patterns: HashSet::new(),
            depfile_var_name,
            vpath_var_name,
            gpath_var_name,
            gpaths: Vec::new(),
            gpath_origin: HashMap::new(),
            implicit_outputs_var_name,
            ninja_pool_var_name,
            validations_var_name,
            tags_var_name,
        };
        let _tr = ScopedTimeReporter::new(&ret.ev.session, "make dep (populate)");
        ret.populate_rules()?;
        if ret.ev.session.flags.enable_stat_logs {
            eprintln!("*kati*: {} explicit rules", ret.rules.len());
            eprintln!("*kati*: {} implicit rules", ret.implicit_rules.len());
            eprintln!("*kati*: {} suffix rules", ret.suffix_rules.len());
        }

        ret.handle_special_targets()?;
        ret.gpaths = ret.gpath_directories()?;

        // The rules are this builder's now. Anything the evaluator records from
        // here on — a recipe's `$(eval)`, a second expansion's — is a rule the
        // graph will never see, so the evaluator has to refuse it rather than
        // accept it and describe a different build. GNU Make raises
        // `snapped_deps` at the same point, at the end of `snap_deps`.
        ret.ev.rules_snapped = true;

        Ok(ret)
    }

    /// The directories `GPATH` names.
    ///
    /// GNU Make expands `$(strip $(GPATH))` once the read has finished, which
    /// is where this is read, and parses the answer as a search path.
    fn gpath_directories(&mut self) -> Result<Vec<Bytes>> {
        Ok(search_path(&self.ev.eval_var(self.gpath_var_name)?))
    }

    /// Whether `GPATH` names the directory the search found `found` in.
    fn gpath_holds(&self, found: &[u8], name: &[u8]) -> bool {
        !self.gpaths.is_empty()
            && search_directory(found, name)
                .is_some_and(|directory| self.gpaths.iter().any(|gpath| gpath == directory))
    }

    fn handle_special_targets(&mut self) -> Result<()> {
        let phony = self.ev.session.intern(".PHONY");
        if let Some((targets, _)) = self.get_rule_inputs(phony)? {
            for t in targets {
                self.phony.insert(t);
            }
        }
        let restat = self.ev.session.intern(".KATI_RESTAT");
        if let Some((targets, _)) = self.get_rule_inputs(restat)? {
            for t in targets {
                self.restat.insert(t);
            }
        }
        // Bare `.IGNORE:` is `-i` asked for by the Makefile; with prerequisites
        // it is the same thing for those targets alone.
        // Only the bare form. With prerequisites it says something narrower
        // that has not been established against GNU Make.
        let not_parallel = self.ev.session.intern(".NOTPARALLEL");
        if let Some((targets, _)) = self.get_rule_inputs(not_parallel)?
            && targets.is_empty()
        {
            self.ev.session.flags.not_parallel = true;
        }
        let one_shell = self.ev.session.intern(".ONESHELL");
        if self.get_rule_inputs(one_shell)?.is_some() {
            self.ev.session.flags.one_shell = true;
        }
        let export_all = self.ev.session.intern(".EXPORT_ALL_VARIABLES");
        if self.get_rule_inputs(export_all)?.is_some() {
            self.ev.session.flags.export_all_variables = true;
        }
        self.handle_intermediate_targets()?;
        self.handle_deletion_targets()?;
        let ignore = self.ev.session.intern(".IGNORE");
        if let Some((targets, _)) = self.get_rule_inputs(ignore)? {
            if targets.is_empty() {
                self.ev.session.flags.ignore_errors = true;
            } else {
                self.ignore_errors.extend(targets);
            }
        }
        // The bare `.WAIT:` form is what Makefiles write for older makes, so it
        // is not worth a word.
        if let Some(merger) = self.rules.get(&self.wait_sym).cloned() {
            let merger = merger.lock();
            for rule in &merger.rules {
                if !rule.inputs.is_empty() || !rule.order_only_inputs.is_empty() {
                    warn_loc!(
                        self.ev,
                        Some(&rule.loc),
                        ".WAIT should not have prerequisites"
                    );
                }
                if !rule.cmds.is_empty() {
                    warn_loc!(self.ev, Some(&rule.loc), ".WAIT should not have commands");
                }
            }
        }
        // The last one wins, and a `.DEFAULT:` with no recipe cancels it.
        let default = self.ev.session.intern(".DEFAULT");
        if let Some(merger) = self.rules.get(&default).cloned() {
            self.default_rule = merger
                .lock()
                .rules
                .last()
                .filter(|rule| !rule.cmds.is_empty())
                .cloned();
        }
        // In order, because `.SUFFIXES:` clears the list and a later one adds
        // to what is left. Merging them first loses the clear.
        let suffixes = self.ev.session.intern(".SUFFIXES");
        if let Some(merger) = self.rules.get(&suffixes).cloned() {
            let mut declared: Vec<Symbol> = Vec::new();
            let rules = merger.lock().rules.clone();
            for rule in &rules {
                let mut inputs = rule.inputs.clone();
                inputs.extend(self.declared_by(suffixes, rule)?);
                if inputs.is_empty() {
                    declared.clear();
                } else {
                    declared.extend(inputs);
                }
            }
            if declared.is_empty() {
                self.suffix_rules.clear();
            } else {
                self.keep_only_declared_suffix_rules(&declared);
            }
        }

        Ok(())
    }

    /// Read the two targets that decide what a failed recipe leaves behind.
    ///
    /// `.DELETE_ON_ERROR` is a switch and not a list: GNU Make asks only
    /// whether the name was written as a target, so prerequisites beside it
    /// neither narrow it nor widen it. `.PRECIOUS` is the list, and a name on
    /// it that looks like a pattern is kept apart because it is matched against
    /// the rule that made a file rather than against the file.
    fn handle_deletion_targets(&mut self) -> Result<()> {
        let delete_on_error = self.ev.session.intern(".DELETE_ON_ERROR");
        self.delete_on_error = self.get_rule_inputs(delete_on_error)?.is_some();
        let precious = self.ev.session.intern(".PRECIOUS");
        if let Some((targets, _)) = self.get_rule_inputs(precious)? {
            for t in targets {
                if is_pattern_rule(&t.as_bytes(&self.ev.session)) {
                    self.precious_patterns.insert(t);
                } else {
                    self.precious.insert(t);
                }
            }
        }
        Ok(())
    }

    /// Whether `.PRECIOUS` protects this name, given the implicit rule pattern
    /// that made it if one did.
    fn is_precious(&self, output: Symbol, output_pattern: Option<Symbol>) -> bool {
        self.precious.contains(&output)
            || output_pattern.is_some_and(|pattern| self.precious_patterns.contains(&pattern))
    }

    /// Record which of each action's outputs a failed recipe must not leave
    /// behind.
    ///
    /// Asked once the whole graph is planned rather than as each node is built,
    /// because an action's outputs are not all known when it is: a grouped
    /// record and a multi-target pattern rule both acquire the rest of theirs
    /// later, and a name protected by a pattern acquires that protection when
    /// the rule that makes it is chosen.
    ///
    /// A node with no recipe is skipped: nothing can fail, so nothing is
    /// half-made.
    fn mark_delete_on_error(&self) {
        if !self.delete_on_error {
            return;
        }
        let mut seen = HashSet::new();
        let nodes = self
            .done
            .values()
            .filter(|node| seen.insert(Arc::as_ptr(node)))
            .cloned()
            .collect::<Vec<_>>();
        for node in nodes {
            let mut node = node.lock();
            if node.cmds.is_empty() {
                continue;
            }
            // The rule's target pattern speaks for the name the search matched
            // it against and for no other. A multi-target pattern rule's other
            // names were protected by their own patterns when they were
            // invented, which is where each one's pattern was still known.
            let recipe_output = node.recipe_output;
            let output_pattern = node.output_pattern;
            let mut outputs = node
                .grouped_double_action
                .as_ref()
                .map_or_else(|| vec![recipe_output], |action| action.members.clone());
            outputs.extend(node.implicit_outputs.iter().copied());
            outputs.retain(|output| {
                let pattern = (*output == recipe_output)
                    .then_some(output_pattern)
                    .flatten();
                !self.phony.contains(output) && !self.is_precious(*output, pattern)
            });
            node.delete_on_error_outputs = outputs;
        }
    }

    /// Take back the sweeping-up from every intermediate `.PRECIOUS` protects.
    ///
    /// Being intermediate and being deleted are two answers in GNU Make, and
    /// `.PRECIOUS` gives only the second. A protected name is still intermediate
    /// — its absence is still no reason to remake what reads it — and the file
    /// the build leaves behind is simply not swept up afterwards. Saying it the
    /// other way round would make `.PRECIOUS` on a `.INTERMEDIATE` name rebuild
    /// a chain GNU Make leaves alone.
    ///
    /// Asked once the graph is planned rather than as each node is made,
    /// because a pattern protects a name only from the moment the rule that
    /// makes it has been chosen — the same reason
    /// [`Self::mark_delete_on_error`] waits.
    fn keep_precious_intermediates(&self) {
        if self.precious.is_empty() && self.precious_patterns.is_empty() {
            return;
        }
        let mut seen = HashSet::new();
        for node in self.done.values() {
            if !seen.insert(Arc::as_ptr(node)) {
                continue;
            }
            let mut node = node.lock();
            // The rule's target pattern speaks for the name the search matched
            // it against, so it is read beside that name and no other.
            if node.is_disposable && self.is_precious(node.recipe_output, node.output_pattern) {
                node.is_disposable = false;
            }
        }
    }

    /// Read the three targets that argue over which files are intermediate.
    ///
    /// In GNU Make's order, which is the order they veto each other in:
    /// `.NOTINTERMEDIATE` first, so the other two can refuse a name it already
    /// took, and last the one pair that cannot both mean everything.
    fn handle_intermediate_targets(&mut self) -> Result<()> {
        let not_intermediate = self.ev.session.intern(".NOTINTERMEDIATE");
        if let Some((targets, _)) = self.get_rule_inputs(not_intermediate)? {
            if targets.is_empty() {
                self.no_intermediates = true;
            }
            for t in targets {
                if t.as_bytes(&self.ev.session).contains(&b'%') {
                    self.not_intermediate_patterns.push(t);
                } else {
                    self.not_intermediate.insert(t);
                }
            }
        }
        let intermediate = self.ev.session.intern(".INTERMEDIATE");
        if let Some((targets, _)) = self.get_rule_inputs(intermediate)? {
            // Naming none would mean every target, and a build whose every
            // target may be skipped builds nothing. GNU Make ignores it.
            for t in targets {
                if self.not_intermediate.contains(&t) {
                    error_loc!(
                        self.ev,
                        None,
                        "*** {} cannot be both .NOTINTERMEDIATE and .INTERMEDIATE.",
                        t.display(self.ev)
                    );
                }
                self.declared_intermediate.insert(t);
            }
        }
        let secondary = self.ev.session.intern(".SECONDARY");
        if let Some((targets, _)) = self.get_rule_inputs(secondary)? {
            if targets.is_empty() {
                if self.no_intermediates {
                    error_loc!(
                        self.ev,
                        None,
                        "*** .NOTINTERMEDIATE and .SECONDARY are mutually exclusive."
                    );
                }
                self.all_secondary = true;
            }
            for t in targets {
                if self.not_intermediate.contains(&t) {
                    error_loc!(
                        self.ev,
                        None,
                        "*** {} cannot be both .NOTINTERMEDIATE and .SECONDARY.",
                        t.display(self.ev)
                    );
                }
                self.declared_intermediate.insert(t);
                self.secondary.insert(t);
            }
        }
        Ok(())
    }

    /// Whether a file's absence is no reason to remake what reads it.
    ///
    /// `.INTERMEDIATE` and `.SECONDARY` win outright: a name either of them
    /// says is intermediate however else it was reached, which is what makes
    /// them worth writing beside a `.NOTINTERMEDIATE` pattern.
    fn treat_as_intermediate(&self, output: Symbol) -> bool {
        if self.declared_intermediate.contains(&output) {
            return true;
        }
        if self.no_intermediates || self.not_intermediate.contains(&output) {
            return false;
        }
        let name = output.as_bytes(&self.ev.session);
        if self
            .not_intermediate_patterns
            .iter()
            .any(|p| Pattern::new(p.as_bytes(&self.ev.session)).matches(&name))
        {
            return false;
        }
        self.all_secondary || self.intermediates.contains(&output)
    }

    /// A `.x.y:` rule is a suffix rule only while both `.x` and `.y` are on the
    /// list, so a Makefile that clears the list and declares its own decides
    /// which rules survive.
    fn keep_only_declared_suffix_rules(&mut self, declared: &[Symbol]) {
        let declared = declared
            .iter()
            .map(|s| {
                let name = s.as_bytes(&self.ev.session);
                name.slice(usize::from(name.starts_with(b"."))..)
            })
            .collect::<HashSet<_>>();
        let names = &self.ev.session;
        self.suffix_rules.retain(|output_suffix, rules| {
            if !declared.contains(output_suffix) {
                return false;
            }
            rules.retain(|rule| declared.contains(&rule.inputs[0].as_bytes(names)));
            !rules.is_empty()
        });
    }

    /// The goal an invocation that named none builds.
    ///
    /// `.DEFAULT_GOAL` answers, and it is read here rather than remembered
    /// from the read that set it: what counts is the value the last line of
    /// the last Makefile left, whether that was the first eligible target's
    /// name or something the Makefile wrote over it with.
    ///
    /// The value names one target. Empty means nothing was ever eligible and
    /// nothing was asked for, which is a build with nothing to aim at. More
    /// than one name is a Makefile asking for something Make cannot do, and it
    /// says so rather than picking one.
    fn default_goal(&mut self) -> Result<Symbol> {
        let value = self.ev.eval_var(Symbol::DEFAULT_GOAL)?;
        let mut named = makefile_word_scanner(&value);
        let Some(goal) = named.next() else {
            // GNU Make's own wording, because its test suite matches this
            // message exactly to learn what the program under test is called.
            // The name and the `Stop.` are added on the way out.
            error_loc!(self.ev, None, "*** No targets.");
        };
        if named.next().is_none() {
            return Ok(self.ev.session.intern(goal.to_vec()));
        }
        // A name is a name before it is a list. GNU Make asks whether the whole
        // value is a target it has heard of before it reads words out of it, so
        // `a\ xb` — one target whose name holds a space — is that target rather
        // than two it has never heard of.
        let whole = self.ev.session.intern(value.to_vec());
        if self.rules.contains_key(&whole) || self.mentioned.contains(&whole) {
            return Ok(whole);
        }
        error_loc!(
            self.ev,
            None,
            "*** .DEFAULT_GOAL contains more than one target."
        );
    }

    fn build(
        &mut self,
        mut targets: Vec<Symbol>,
        read_makefiles: &[ReadMakefile],
        missing_includes: &[MissingInclude],
    ) -> Result<(Vec<NamedDepNode>, Vec<RegenerationRoot>)> {
        // Generated included Makefiles are compiler inputs rather than user
        // goals, and GNU Make remakes them before it picks a goal at all. Both
        // halves of that matter here: asking the graph for one must not change
        // what the Makefile builds once it is reread, and a required include
        // with no rule is the failure the run dies on, ahead of the complaint
        // about having nothing to aim at.
        let regeneration_nodes = self.plan_regeneration(read_makefiles, missing_includes)?;

        if !self.ev.session.flags.gen_all_targets && targets.is_empty() {
            targets.push(self.default_goal()?);
        }
        if self.ev.session.flags.gen_all_targets {
            let mut non_root_targets = HashSet::new();
            for (sym, merger) in &self.rules {
                if is_special_target(&self.ev.session, sym) {
                    continue;
                }
                for r in merger.lock().rules.iter() {
                    for t in &r.inputs {
                        non_root_targets.insert(*t);
                    }
                    for t in &r.order_only_inputs {
                        non_root_targets.insert(*t);
                    }
                }
            }

            let mut rule_keys = self.rules.keys().cloned().collect::<Vec<_>>();
            let names = &self.ev.session;
            rule_keys.sort_by_cached_key(|k| k.as_bytes(names));
            for t in rule_keys {
                if !non_root_targets.contains(&t) && !is_special_target(&self.ev.session, &t) {
                    targets.push(t);
                }
            }
        }

        // TODO: LogStats?

        // A goal is a file like any other, so `GPATH` reaches it too: one found
        // in a directory `GPATH` names is asked for, and remade, under the path
        // the search returned. The goals are what the graph is aimed at, so the
        // rename has to reach them before they are read as that.
        let targets = targets
            .into_iter()
            .map(|target| self.at_gpath(target))
            .collect::<Vec<_>>();
        self.ev.goals.clone_from(&targets);
        let mut nodes = Vec::new();
        for target in targets {
            nodes.push((target, self.plan_root(target)?));
        }
        self.apply_wait_barriers();
        self.mark_delete_on_error();
        self.keep_precious_intermediates();
        Ok((nodes, regeneration_nodes))
    }

    /// Plan one root of the graph: a goal, or a Makefile that has to be
    /// generated before the goals mean what they will mean.
    fn plan_root(&mut self, target: Symbol) -> Result<Arc<Mutex<DepNode>>> {
        let v = Arc::new(Vars::new());
        self.cur_rule_vars = Some(v.clone());
        self.ev.current_scope = Some(v.clone());
        let n = self.build_plan(target, None)?;
        // A root is asked for, so it is built and it is kept: GNU Make reaches
        // one directly rather than through the rule that wanted it, and never
        // deletes what the command line named.
        {
            let mut n = n.lock();
            n.is_intermediate = false;
            n.is_disposable = false;
        }
        self.ev.current_scope = None;
        self.cur_rule_vars = None;
        Ok(n)
    }

    /// Decide what to do about each Makefile the read reached.
    ///
    /// GNU Make looks for a rule that would make every one of them — the file
    /// the invocation named as much as the files it included — and hands the
    /// ones it finds to an ordinary update before it chooses a goal. A Makefile
    /// that is actually remade sends make back to the start to read it again,
    /// so the roots returned here are what an embedding frontend builds and
    /// then re-evaluates on.
    ///
    /// A file the read could not open is the same question with a louder
    /// answer when there is no rule: `-include` and `sinclude` forget it
    /// without a word, while `include` reports the read it could not do and
    /// then dies naming the file as a target it cannot reach.
    fn plan_regeneration(
        &mut self,
        read_makefiles: &[ReadMakefile],
        missing_includes: &[MissingInclude],
    ) -> Result<Vec<RegenerationRoot>> {
        let mut nodes = Vec::new();
        for &ReadMakefile {
            filename: makefile,
            required,
        } in read_makefiles
        {
            let node = self.plan_root(makefile)?;
            if Self::is_remakable(&node) {
                nodes.push(RegenerationRoot {
                    node: (makefile, node),
                    required,
                });
                continue;
            }
            let Some(include) = missing_includes
                .iter()
                .find(|include| include.filename == makefile)
            else {
                continue;
            };
            if !required {
                continue;
            }
            let name = include.filename.as_bytes(&self.ev.session);
            let name = String::from_utf8_lossy(&name).into_owned();
            // A Makefile the command line named carries no location, because no
            // `include` line asked for it. GNU Make reports that one where it
            // failed to open, so the read has already said so and only the
            // refusal is left.
            if let Some(loc) = &include.loc {
                warn_loc!(self.ev, Some(loc), "{name}: No such file or directory");
            }
            error_loc!(self.ev, None, "*** No rule to make target '{name}'.");
        }
        Ok(nodes)
    }

    /// Whether GNU Make would try to bring this Makefile up to date.
    ///
    /// A rule has to say how. Two shapes that have one are still refused,
    /// because each would be remade every time it was considered and so would
    /// restart the read forever: a Makefile declared `.PHONY`, and one whose
    /// `::` recipe has no prerequisites.
    fn is_remakable(node: &Arc<Mutex<DepNode>>) -> bool {
        let node = node.lock();
        node.has_rule && !node.is_phony && !node.unconditional_double_colon
    }

    fn exists(&self, target: Symbol) -> bool {
        self.rules.contains_key(&target)
            || self.phony.contains(&target)
            || std::fs::exists(OsStr::from_bytes(&target.as_bytes(&self.ev.session)))
                .is_ok_and(|v| v)
            || self.vpath_of(target).is_some()
    }

    /// Replace each prerequisite with where the directory search found it.
    ///
    /// The rewrite has to happen to the node's inputs rather than only at the
    /// point of asking whether a file exists, because the inputs are what `$<`
    /// and `$^` expand to and what the recipe is therefore handed. A search
    /// that found the file and then passed on the name as written would build
    /// with a path that is not there.
    ///
    /// A prerequisite with a rule of its own is left alone: it is going to be
    /// built here, so where an older copy of it might be lying is not a
    /// question worth asking.
    fn resolve_vpaths(&mut self, n: &Arc<Mutex<DepNode>>) {
        if self.ev.session.vpaths.is_empty() && self.vpath_variable().is_empty() {
            return;
        }
        let (inputs, order_only) = {
            let n = n.lock();
            (n.actual_inputs.clone(), n.actual_order_only_inputs.clone())
        };
        let inputs = self.at_vpaths(inputs);
        let order_only = self.at_vpaths(order_only);
        let mut n = n.lock();
        n.actual_inputs = inputs;
        n.actual_order_only_inputs = order_only;
    }

    /// The prerequisites already recorded for a target, which is what `$<` and
    /// its neighbours are worth while the rest are being worked out.
    fn recorded_prerequisites(&mut self, output: Symbol) -> (Vec<Symbol>, Vec<Symbol>) {
        let Some(merger) = self.rules.get(&output).cloned() else {
            return (Vec::new(), Vec::new());
        };
        let rules = merger.lock().rules.clone();
        let mut inputs = Vec::new();
        let mut order_only = Vec::new();
        for r in &rules {
            let session = &mut self.ev.session;
            inputs.extend(apply_output_pattern(session, r, output, &r.inputs));
            order_only.extend(apply_output_pattern(
                session,
                r,
                output,
                &r.order_only_inputs,
            ));
        }
        (inputs, order_only)
    }

    fn joined(&self, syms: &[Symbol], unique: bool) -> Bytes {
        let mut out = BytesMut::new();
        {
            let mut seen = HashSet::new();
            let mut ww = WordWriter::new(&mut out);
            for sym in syms {
                if !unique || seen.insert(*sym) {
                    ww.write(&sym.as_bytes(&self.ev.session));
                }
            }
        }
        out.freeze()
    }

    /// The second half of `.SECONDEXPANSION`: expand what the first expansion
    /// left, now that `$@` and the stem have values, and read the result as
    /// prerequisites. A stem is given for a static pattern rule and withheld
    /// for an explicit one, where `%` is an ordinary character.
    fn expand_prerequisites_again(
        &mut self,
        output: Symbol,
        stem: Option<Bytes>,
        prerequisites: (&[Symbol], &[Symbol]),
        text: &Bytes,
    ) -> Result<(Vec<Symbol>, Vec<Symbol>)> {
        self.expand_deferred_prerequisites(output, stem, prerequisites, vec![text.clone()])
    }

    /// An implicit pattern rule expands each raw prerequisite word
    /// independently. This keeps a backslash at the end of one raw word from
    /// quoting the blank before the next one, while still letting an expansion
    /// introduce an escaped blank inside its own result.
    fn expand_pattern_prerequisites_again(
        &mut self,
        output: Symbol,
        stem: Bytes,
        prerequisites: (&[Symbol], &[Symbol]),
        text: &Bytes,
    ) -> Result<(Vec<Symbol>, Vec<Symbol>)> {
        self.expand_deferred_prerequisites(
            output,
            Some(stem),
            prerequisites,
            implicit_prerequisite_words(text).collect(),
        )
    }

    fn expand_deferred_prerequisites(
        &mut self,
        output: Symbol,
        stem: Option<Bytes>,
        prerequisites: (&[Symbol], &[Symbol]),
        texts: Vec<Bytes>,
    ) -> Result<(Vec<Symbol>, Vec<Symbol>)> {
        let at = self.ev.session.intern("@");
        let star = self.ev.session.intern("*");
        let less = self.ev.session.intern("<");
        let hat = self.ev.session.intern("^");
        let plus = self.ev.session.intern("+");
        let bar = self.ev.session.intern("|");
        let automatic = |s: Bytes| {
            Variable::with_simple_string(s, crate::var::VarOrigin::Automatic, None, None)
        };
        let scope = self.cur_rule_vars.clone().unwrap_or_default();
        let texts = match &stem {
            Some(_) => texts
                .into_iter()
                .map(|text| stem_references(&text))
                .collect::<Vec<_>>(),
            None => texts,
        };
        let (recorded, recorded_order_only) = prerequisites;
        let first = recorded
            .first()
            .map(|s| s.as_bytes(&self.ev.session))
            .unwrap_or_default();
        let (hat_value, plus_value, bar_value) = (
            self.joined(recorded, true),
            self.joined(recorded, false),
            self.joined(recorded_order_only, true),
        );
        let expanded = {
            let _at = ScopedVar::new(
                scope.clone(),
                at,
                automatic(output.as_bytes(&self.ev.session)),
            );
            let _star = stem.map(|s| ScopedVar::new(scope.clone(), star, automatic(s)));
            let _less = ScopedVar::new(scope.clone(), less, automatic(first));
            let _hat = ScopedVar::new(scope.clone(), hat, automatic(hat_value));
            let _plus = ScopedVar::new(scope.clone(), plus, automatic(plus_value));
            let _bar = ScopedVar::new(scope, bar, automatic(bar_value));
            let mut expanded = Vec::with_capacity(texts.len());
            for text in texts {
                let mut loc = self.ev.loc.clone().unwrap_or_default();
                let expr = crate::expr::parse_expr(
                    &mut self.ev.session,
                    &mut loc,
                    text,
                    crate::expr::ParseExprOpt::Normal,
                )?;
                expanded.push(expr.eval_to_buf(self.ev)?);
            }
            expanded
        };

        let mut inputs = Vec::new();
        let mut order_only_inputs = Vec::new();
        let mut order_only = false;
        for expanded_word in expanded {
            let (before, after) = if order_only {
                (Bytes::new(), expanded_word)
            } else {
                let split = split_order_only(&expanded_word);
                order_only = memchr(b'|', &expanded_word).is_some();
                split
            };
            for (text, into) in [(before, &mut inputs), (after, &mut order_only_inputs)] {
                for word in makefile_word_scanner(&text) {
                    let word = word.slice_ref(trim_leading_curdir(&word));
                    glob_word(&mut self.ev.session, word, into);
                }
            }
        }
        Ok((inputs, order_only_inputs))
    }

    /// The stem of `output` under a rule's first output pattern, or None when
    /// the rule has none and `%` is therefore literal.
    fn stem_of(&self, rule: &Rule, output: &Bytes) -> Option<Bytes> {
        let pattern = rule.output_patterns.first()?;
        let pat = Pattern::new(pattern.as_bytes(&self.ev.session));
        Some(Bytes::copy_from_slice(pat.stem(output)))
    }

    /// `.WAIT` names no file, so it goes before build_plan descends and never
    /// reaches the graph or an automatic variable. What it separated is
    /// recorded for [`DepBuilder::apply_wait_barriers`].
    fn take_out_waits(&mut self, n: &Arc<Mutex<DepNode>>) {
        let mut node = n.lock();
        if !node.actual_inputs.contains(&self.wait_sym)
            && !node.actual_order_only_inputs.contains(&self.wait_sym)
        {
            return;
        }
        let (inputs, barriers) = self.without_waits(std::mem::take(&mut node.actual_inputs));
        node.actual_inputs = inputs;
        self.wait_barriers.extend(barriers);
        let (order_only, barriers) =
            self.without_waits(std::mem::take(&mut node.actual_order_only_inputs));
        node.actual_order_only_inputs = order_only;
        self.wait_barriers.extend(barriers);
    }

    fn without_waits(&self, inputs: Vec<Symbol>) -> (Vec<Symbol>, Vec<(Symbol, Vec<Symbol>)>) {
        let mut kept = Vec::with_capacity(inputs.len());
        let mut earlier: Vec<Symbol> = Vec::new();
        let mut barriers = Vec::new();
        for input in inputs {
            if input == self.wait_sym {
                // Everything to the left, not only the group just ended.
                earlier.clone_from(&kept);
                continue;
            }
            if !earlier.is_empty() {
                barriers.push((input, earlier.clone()));
            }
            kept.push(input);
        }
        (kept, barriers)
    }

    /// Make orders one rule's prerequisite list as it walks it, so a shared
    /// prerequisite is still free to run early for another rule's sake. An edge
    /// is added only where the later prerequisite has one consumer and the two
    /// readings agree; adding it otherwise deadlocks GNU Make's own test.
    fn apply_wait_barriers(&mut self) {
        if self.wait_barriers.is_empty() {
            return;
        }
        let mut consumers: HashMap<Symbol, usize> = HashMap::new();
        for node in self.done.values() {
            let node = node.lock();
            for input in node
                .actual_inputs
                .iter()
                .chain(node.actual_order_only_inputs.iter())
            {
                *consumers.entry(*input).or_default() += 1;
            }
        }
        for (later, earlier) in std::mem::take(&mut self.wait_barriers) {
            if consumers.get(&later).copied() != Some(1) {
                continue;
            }
            let Some(node) = self.done.get(&later).cloned() else {
                continue;
            };
            for before in earlier {
                let Some(dep) = self.done.get(&before).cloned() else {
                    continue;
                };
                let mut node = node.lock();
                if node.actual_order_only_inputs.contains(&before) {
                    continue;
                }
                node.actual_order_only_inputs.push(before);
                node.order_onlys.push((before, dep));
            }
        }
    }

    /// Each prerequisite, moved to where the search found it.
    ///
    /// Resolved first and interned after, because finding the file needs the
    /// session to read and naming the result needs it to write.
    fn at_vpaths(&mut self, inputs: Vec<Symbol>) -> Vec<Symbol> {
        inputs
            .into_iter()
            .map(|input| self.at_found_name(input))
            .collect()
    }

    /// One name, replaced by where the directory search found it.
    fn at_found_name(&mut self, name: Symbol) -> Symbol {
        match self.at_vpath(name) {
            Some((found, kept_by_gpath)) => self.take_found_name(name, found, kept_by_gpath),
            None => name,
        }
    }

    /// One name, replaced by where the search found it only when `GPATH` says
    /// that is where it belongs.
    ///
    /// For a caller that reaches a name directly rather than through the rule
    /// that wanted it, and so has no prerequisite of its own to rewrite.
    fn at_gpath(&mut self, name: Symbol) -> Symbol {
        match self.at_vpath(name) {
            Some((found, true)) => self.take_found_name(name, found, true),
            _ => name,
        }
    }

    /// Take the search's answer for `name`.
    ///
    /// A rename `GPATH` made is remembered, so that the rule declared for the
    /// name as written can be found again under the path it moved to.
    fn take_found_name(&mut self, name: Symbol, found: Bytes, kept_by_gpath: bool) -> Symbol {
        let found = self.ev.session.intern(found);
        if kept_by_gpath {
            self.gpath_origin.insert(found, name);
        }
        found
    }

    /// Where one name was found, if it had to be looked for, and whether
    /// `GPATH` is what kept the answer.
    ///
    /// A name with a rule of its own is normally left alone: it is going to be
    /// built here, so where an older copy of it might be lying is not a
    /// question worth asking. `GPATH` is the answer to that question anyway —
    /// it says the directory the search looked in is where the name belongs, so
    /// GNU Make renames the file to the found path before it asks anything else
    /// about it and remakes it there.
    fn at_vpath(&self, input: Symbol) -> Option<(Bytes, bool)> {
        if self.phony.contains(&input) {
            return None;
        }
        let name = input.as_bytes(&self.ev.session);
        if std::fs::exists(OsStr::from_bytes(&name)).is_ok_and(|found| found) {
            return None;
        }
        let found = self.vpath_of(input)?;
        if self.gpath_holds(&found, &name) {
            return Some((found, true));
        }
        if self.rules.contains_key(&input) {
            return None;
        }
        Some((found, false))
    }

    /// Where a prerequisite actually is, when it is not where it was named.
    ///
    /// GNU Make's directory search. A name with a rule, or one that names a
    /// file in the current directory, is already resolved and is left alone —
    /// the search is what happens when neither is true. The first `vpath`
    /// pattern that matches decides which directories are looked in; a name no
    /// pattern matches falls back to `VPATH`, which is a variable rather than a
    /// directive and so is read here rather than recorded.
    fn vpath_of(&self, target: Symbol) -> Option<Bytes> {
        let name = target.as_bytes(&self.ev.session);
        if name.is_empty() || self.ev.session.vpaths.is_empty() && self.vpath_variable().is_empty()
        {
            return None;
        }
        let matched = self
            .ev
            .session
            .vpaths
            .iter()
            .filter(|(pattern, _)| pattern.matches(&name))
            .flat_map(|(_, directories)| directories.iter().cloned())
            .collect::<Vec<_>>();
        let directories = if matched.is_empty() {
            self.vpath_variable()
        } else {
            matched
        };
        for directory in directories {
            let mut candidate = BytesMut::from(directory.as_ref());
            if !candidate.ends_with(b"/") {
                candidate.put_u8(b'/');
            }
            candidate.put_slice(&name);
            let candidate = candidate.freeze();
            if std::fs::exists(OsStr::from_bytes(&candidate)).is_ok_and(|found| found) {
                return Some(candidate);
            }
        }
        None
    }

    /// The directories `VPATH` names, separated by colons or by whitespace.
    fn vpath_variable(&self) -> Vec<Bytes> {
        let Some(var) = self.ev.session.peek_global_var(self.vpath_var_name) else {
            return Vec::new();
        };
        let read = var.read();
        let Ok(value) = read.string(&self.ev.session) else {
            return Vec::new();
        };
        let value = Bytes::copy_from_slice(value.as_ref());
        crate::strutil::word_scanner(&value)
            .flat_map(|word| word.split(|byte| *byte == b':'))
            .filter(|directory| !directory.is_empty())
            .map(|directory| value.slice_ref(directory))
            .collect()
    }

    fn get_rule_inputs(&mut self, s: Symbol) -> Result<Option<(Vec<Symbol>, Loc)>> {
        let Some(merger) = self.rules.get(&s).cloned() else {
            return Ok(None);
        };
        let rules = merger.lock().rules.clone();
        assert!(!rules.is_empty());
        let mut ret = Vec::new();
        for r in &rules {
            ret.extend(r.inputs.iter().copied());
            ret.extend(self.declared_by(s, r)?);
        }

        Ok(Some((ret, rules[0].loc.clone())))
    }

    /// GNU Make expands a special target's prerequisites once the makefiles are
    /// read and before it reads what they declare, so a `.PHONY` written under
    /// `.SECONDEXPANSION` still declares something.
    fn declared_by(&mut self, target: Symbol, rule: &Rule) -> Result<Vec<Symbol>> {
        let Some(text) = rule
            .deferred_prerequisites
            .as_ref()
            .filter(|_| prerequisites_reach(&self.ev.session, rule, target))
            .cloned()
        else {
            return Ok(Vec::new());
        };
        let (mut inputs, order_only) =
            self.expand_prerequisites_again(target, None, (&[], &[]), &text)?;
        inputs.extend(order_only);
        Ok(inputs)
    }

    fn populate_rules(&mut self) -> Result<()> {
        // TODO: Is this take necessary, or can we refactor how we pass around ev?
        for rule in std::mem::take(&mut self.ev.rules) {
            if rule.is_grouped
                && rule.cmds.is_empty()
                && (!rule.outputs.is_empty() || !rule.output_patterns.is_empty())
            {
                error_loc!(
                    self.ev,
                    Some(&rule.loc),
                    "*** grouped targets must provide a recipe."
                );
            }
            let rule = Arc::new(rule);
            if rule.outputs.is_empty() {
                self.populate_implicit_rule(rule)?;
            } else {
                self.populate_explicit_rule(rule)?;
            }
        }
        self.discard_suffix_rules_a_pattern_rule_holds();
        for rules in self.suffix_rules.values_mut() {
            rules.reverse();
        }
        // TODO: This clone likely isn't necessary with some refactoring
        for (symbol, merger) in self.rules.clone() {
            let Some(vars) = self.lookup_rule_vars(symbol) else {
                continue;
            };
            if let Some(var) = vars.lookup(
                &mut self.ev.session.used_env_vars,
                self.implicit_outputs_var_name,
            ) {
                let implicit_outputs = var.read().eval_to_buf(self.ev)?;

                for output in word_scanner(&implicit_outputs) {
                    let sym = self
                        .ev
                        .session
                        .intern(implicit_outputs.slice_ref(trim_leading_curdir(output)));
                    self.rules
                        .entry(sym)
                        .or_insert_with(RuleMerger::new)
                        .lock()
                        .set_implicit_output(&*self.ev, sym, symbol, merger.clone())?;
                    merger
                        .lock()
                        .add_implicit_output(sym, self.rules[&sym].clone());
                }
            }

            if let Some(var) = vars.lookup(
                &mut self.ev.session.used_env_vars,
                self.validations_var_name,
            ) {
                let validations = var.read().eval_to_buf(self.ev)?;

                for validation in word_scanner(&validations) {
                    let sym = self
                        .ev
                        .session
                        .intern(validations.slice_ref(trim_leading_curdir(validation)));
                    merger.lock().add_validation(sym);
                }
            }
        }
        Ok(())
    }

    fn populate_suffix_rule(&mut self, rule: &Rule, output: Symbol) -> Result<bool> {
        if !is_suffix_rule(&self.ev.session, &output) {
            return Ok(false);
        }

        if self.ev.session.flags.werror_suffix_rules {
            error_loc!(
                self.ev,
                Some(&rule.loc),
                "*** suffix rules are obsolete: {}",
                output.display(self.ev)
            );
        } else if self.ev.session.flags.warn_suffix_rules {
            warn_loc!(
                self.ev,
                Some(&rule.loc),
                "warning: suffix rules are deprecated: {}",
                output.display(self.ev)
            );
        }

        if rule.cmds.is_empty() {
            // `convert_to_pattern` looks the suffix pair's name up as a file and
            // passes over one with no recipe, so a recipe-less `.w.tex:` never
            // becomes a rule that could make anything. Writing one beside a
            // `.w.tex:` that does have a recipe therefore withdraws nothing:
            // the recipe is what was converted, and it is still there.
            return Ok(false);
        }

        let mut output = output.as_bytes(&self.ev.session);
        output.advance(1);
        let dot_index = memchr(b'.', &output).unwrap();

        let input_suffix = output.slice(..dot_index);
        let output_suffix = output.slice(dot_index + 1..);
        let mut r = rule.clone();
        let mut output_pattern = BytesMut::with_capacity(output_suffix.len() + 2);
        output_pattern.put_slice(b"%.");
        output_pattern.put_slice(&output_suffix);
        r.output_patterns.clear();
        r.output_patterns
            .push(self.ev.session.intern(output_pattern.freeze()));
        r.inputs.clear();
        r.prerequisite_names.clear();
        r.deferred_prerequisites = None;
        let input_sym = self.ev.session.intern(input_suffix);
        r.inputs.push(input_sym);
        r.prerequisite_names.push(input_sym);
        r.is_suffix_rule = true;
        self.suffix_rules
            .entry(output_suffix)
            .or_default()
            .push(Arc::new(r));
        Ok(true)
    }

    /// Throw away every suffix rule a written pattern rule already speaks for,
    /// once all of them are known. GNU Make converts suffix rules after the
    /// last makefile is read, so which side of a pattern rule one was written
    /// on never decides this.
    fn discard_suffix_rules_a_pattern_rule_holds(&mut self) {
        if self.implicit_rule_defs.is_empty() {
            return;
        }
        let names = &self.ev.session;
        let written = &self.implicit_rule_defs;
        self.suffix_rules.retain(|_, rules| {
            rules.retain(|rule| {
                !written
                    .iter()
                    .any(|existing| pattern_rule_holds_suffix_rule(names, existing, rule))
            });
            !rules.is_empty()
        });
    }

    fn populate_explicit_rule(&mut self, rule: Arc<Rule>) -> Result<()> {
        if rule.is_double_colon {
            let rule_id = Self::rule_id(&rule);
            if rule.is_grouped {
                self.double_action_indices.insert(
                    DoubleActionId {
                        rule: rule_id,
                        trigger: None,
                    },
                    self.next_double_action,
                );
                self.next_double_action += 1;
            } else {
                for output in &rule.outputs {
                    self.double_action_indices.insert(
                        DoubleActionId {
                            rule: rule_id,
                            trigger: Some(*output),
                        },
                        self.next_double_action,
                    );
                    self.next_double_action += 1;
                }
            }
            for output in &rule.outputs {
                self.double_memberships
                    .entry(*output)
                    .or_default()
                    .push(rule.clone());
            }
        }
        for input in rule.inputs.iter().chain(&rule.order_only_inputs) {
            if !input.as_bytes(&self.ev.session).contains(&b'%') {
                self.mentioned.insert(*input);
            }
        }
        for output in &rule.outputs {
            if self.first_rule.is_none() && !is_special_target(&self.ev.session, output) {
                self.first_rule = Some(*output);
            }
            self.rules
                .entry(*output)
                .or_insert_with(RuleMerger::new)
                .lock()
                .add_rule(&*self.ev, *output, rule.clone())?;
            self.populate_suffix_rule(&rule, *output)?;
        }
        Ok(())
    }

    fn is_ignorable_implicit_rule(names: &impl Interner, rule: &Rule) -> bool {
        // As kati doesn't have RCS/SCCS related default rules, we can
        // safely ignore suppression for them.
        if rule.inputs.len() != 1 {
            return false;
        }
        if !rule.order_only_inputs.is_empty() {
            return false;
        }
        if !rule.cmds.is_empty() {
            return false;
        }
        let i = rule.inputs[0].as_bytes(names);
        let i = i.as_ref();
        i == b"RCS/%,v" || i == b"RCS/%" || i == b"%,v" || i == b"s.%" || i == b"SCCS/s.%"
    }

    fn populate_implicit_rule(&mut self, rule: Arc<Rule>) -> Result<()> {
        if let Some(index) = self
            .implicit_rule_defs
            .iter()
            .position(|existing| replaces_pattern_rule(&rule, existing))
        {
            let existing = self.implicit_rule_defs.remove(index);
            self.implicit_rules.remove_rule(&existing);
        }
        self.implicit_rule_defs.push(rule.clone());

        for output_pattern in rule.output_patterns.clone() {
            let op = output_pattern.as_bytes(&self.ev.session);
            if op.as_ref() != b"%" || !Self::is_ignorable_implicit_rule(&self.ev.session, &rule) {
                if self.ev.session.flags.werror_implicit_rules {
                    error_loc!(
                        self.ev,
                        Some(&rule.loc),
                        "*** implicit rules are obsolete: {}",
                        output_pattern.display(self.ev)
                    );
                } else if self.ev.session.flags.warn_implicit_rules {
                    warn_loc!(
                        self.ev,
                        Some(&rule.loc),
                        "warning: implicit rules are deprecated: {}",
                        output_pattern.display(self.ev)
                    );
                }

                let order = self.implicit_rule_order;
                self.implicit_rule_order += 1;
                self.implicit_rules.add(
                    &op,
                    ImplicitCandidate {
                        rule: rule.clone(),
                        pattern: output_pattern,
                        order,
                    },
                )
            }
        }
        Ok(())
    }

    fn lookup_rule_merger(&self, o: Symbol) -> Option<Arc<Mutex<RuleMerger>>> {
        self.rules
            .get(&o)
            .or_else(|| self.rules.get(&self.written_as(o)))
            .cloned()
    }

    fn lookup_rule_vars(&self, o: Symbol) -> Option<Arc<Vars>> {
        self.rule_vars
            .get(&o)
            .or_else(|| self.rule_vars.get(&self.written_as(o)))
            .cloned()
    }

    /// The name a target's rule and its own variables were declared under.
    ///
    /// GNU Make renames one file object, so what was declared for the name as
    /// written arrives at the path `GPATH` kept it at rather than being looked
    /// up again. Every other name is declared under itself.
    fn written_as(&self, o: Symbol) -> Symbol {
        self.gpath_origin.get(&o).copied().unwrap_or(o)
    }

    /// Every pattern scope that applies to `output`, weakest first, in GNU
    /// Make's order. The target's own scope goes on top of these; see
    /// `scopes_for`.
    ///
    /// The scopes stay separate rather than being merged, because `+=` in one
    /// of them appends to what the ones before it left rather than to the
    /// makefile-level value, and a merged map cannot say what came first.
    fn matching_pattern_vars(&self, output: Symbol) -> Vec<Arc<Vars>> {
        let name = output.as_bytes(&self.ev.session);
        let mut scopes = Vec::new();
        for (sym, pattern) in &self.pattern_var_order {
            // A pattern variable needs a stem to have matched: GNU Make skips
            // any pattern at least as long as the name, so `%.z` reaches `a.z`
            // and not `.z`. Pattern *rules* match the empty stem, which is why
            // this is not `Pattern::matches` alone.
            if pattern.as_bytes().len() > name.len() || !pattern.matches(&name) {
                continue;
            }
            if let Some(vars) = self.rule_vars.get(sym) {
                scopes.push(vars.clone());
            }
        }
        scopes
    }

    /// Second expansion reads target-specific variables for the target whose
    /// prerequisite list is being expanded.  Dependency execution gets a
    /// different scope later: for grouped peers that is the member which
    /// triggered the shared action.
    fn push_expansion_scope(
        &mut self,
        vars: &[Arc<Vars>],
    ) -> (Option<Arc<Vars>>, Option<Arc<Vars>>) {
        let previous_rule_scope = self.cur_rule_vars.clone();
        let previous_eval_scope = self.ev.current_scope.clone();
        if vars.is_empty() {
            return (previous_rule_scope, previous_eval_scope);
        }
        let scope = Arc::new(Vars::new());
        if let Some(previous) = &previous_rule_scope {
            scope.merge_from(previous);
        }
        for vars in vars {
            scope.merge_from(vars);
        }
        self.cur_rule_vars = Some(scope.clone());
        self.ev.current_scope = Some(scope);
        (previous_rule_scope, previous_eval_scope)
    }

    fn pop_expansion_scope(&mut self, previous: (Option<Arc<Vars>>, Option<Arc<Vars>>)) {
        self.cur_rule_vars = previous.0;
        self.ev.current_scope = previous.1;
    }

    fn rule_id(rule: &Arc<Rule>) -> usize {
        Arc::as_ptr(rule) as usize
    }

    fn double_action_id(rule: &Arc<Rule>, trigger: Symbol) -> DoubleActionId {
        DoubleActionId {
            rule: Self::rule_id(rule),
            trigger: (!rule.is_grouped).then_some(trigger),
        }
    }

    /// Give one exact record a stable, compiler-owned graph output. Real group
    /// members remain public join nodes, so no action competes to produce a
    /// path another independent record also names.
    fn double_action_output(&mut self, action: DoubleActionId) -> Symbol {
        let index = self.double_action_indices[&action];
        let mut directory = self
            .ev
            .session
            .flags
            .ninja_dir
            .clone()
            .map(PathBuf::from)
            .unwrap_or_default();
        directory.push(".ronin_grouped_double");

        for suffix in 0usize.. {
            let filename = if suffix == 0 {
                index.to_string()
            } else {
                format!("{index}_{suffix}")
            };
            let output = {
                let mut path = directory.clone();
                path.push(filename);
                self.ev.session.intern(path.as_os_str().as_bytes().to_vec())
            };
            if self.rules.contains_key(&output)
                || self.done.contains_key(&output)
                || self.mentioned.contains(&output)
                || self.phony.contains(&output)
            {
                continue;
            }
            return output;
        }
        unreachable!("an unbounded numeric suffix has an available stamp name")
    }

    /// Build one independent double-colon action. The rule is the identity for
    /// `&::`; ordinary `::` also includes the triggering member so a
    /// multi-target record still runs once per target.
    fn build_double_action(
        &mut self,
        rule: Arc<Rule>,
        trigger: Symbol,
    ) -> Result<(Arc<Mutex<DepNode>>, bool)> {
        let id = Self::double_action_id(&rule, trigger);
        if let Some(action) = self.double_actions.get(&id) {
            return Ok((action.clone(), false));
        }

        let graph_output = self.double_action_output(id);
        let has_recipe = !rule.cmds.is_empty();
        let action = DepNode::new(
            graph_output,
            false,
            false,
            self.ignore_errors.contains(&trigger),
            false,
            false,
        );
        {
            let mut node = action.lock();
            node.recipe_output = trigger;
            if has_recipe {
                let members = if rule.is_grouped {
                    rule.outputs.clone()
                } else {
                    vec![trigger]
                };
                node.grouped_double_action = Some(GroupedDoubleAction {
                    has_phony_member: members.iter().any(|output| self.phony.contains(output)),
                    members,
                    phony_inputs: Vec::new(),
                });
            }
            node.cmds = rule.cmds.clone();
            node.actual_inputs =
                apply_output_pattern(&mut self.ev.session, &rule, trigger, &rule.inputs);
            node.actual_order_only_inputs = apply_output_pattern(
                &mut self.ev.session,
                &rule,
                trigger,
                &rule.order_only_inputs,
            );
            node.output_pattern = rule.output_patterns.first().copied();
            node.loc = rule.cmd_loc.clone().or_else(|| Some(rule.loc.clone()));
            node.has_rule = true;
            node.is_default_target = false;
        }

        // Cache before descending so a prerequisite cycle finds the same
        // action rather than recursively creating another producer.
        self.double_actions.insert(id, action.clone());
        self.double_action_creation_indices
            .insert(id, self.next_double_action_creation);
        self.next_double_action_creation += 1;
        self.done.insert(graph_output, action.clone());

        if let Some(text) = rule
            .deferred_prerequisites
            .as_ref()
            .filter(|_| prerequisites_reach(&self.ev.session, &rule, trigger))
        {
            let trigger_text = trigger.as_bytes(&self.ev.session);
            let stem = self.stem_of(&rule, &trigger_text);
            let recorded = {
                let node = action.lock();
                (
                    node.actual_inputs.clone(),
                    node.actual_order_only_inputs.clone(),
                )
            };
            let vars = self.applicable_rule_vars(trigger);
            let previous_scope = self.push_expansion_scope(&vars);
            let expanded =
                self.expand_prerequisites_again(trigger, stem, (&recorded.0, &recorded.1), text);
            self.pop_expansion_scope(previous_scope);
            let (inputs, order_only) = expanded?;
            let mut node = action.lock();
            node.actual_inputs.extend(inputs);
            node.actual_order_only_inputs.extend(order_only);
        }

        self.resolve_vpaths(&action);
        self.take_out_waits(&action);
        {
            let mut node = action.lock();
            let phony_inputs = node
                .actual_inputs
                .iter()
                .copied()
                .filter(|input| self.phony.contains(input))
                .collect::<Vec<_>>();
            if let Some(metadata) = &mut node.grouped_double_action {
                metadata.phony_inputs = phony_inputs;
            }
            // `update_file_1`: a double-colon entry with no prerequisites at
            // all is always out of date. Read after second expansion, because
            // that is when GNU Make reaches the same test, and counting both
            // kinds because it asks whether the entry declared any dependency
            // rather than any it would compare timestamps against.
            node.unconditional_double_colon = has_recipe
                && node.actual_inputs.is_empty()
                && node.actual_order_only_inputs.is_empty();
        }
        let vars = self.applicable_rule_vars(trigger);
        let mut scoped_vars = Vec::new();
        let mut private_scoped_vars = Vec::new();
        let trigger_text = trigger.as_bytes(&self.ev.session);
        let frame = self.ev.enter(
            FrameType::Dependency,
            trigger_text,
            action.lock().loc.clone().unwrap_or_default(),
        );
        self.apply_rule_vars(
            &vars,
            &action,
            &frame,
            &mut scoped_vars,
            &mut private_scoped_vars,
        )?;

        let scope = self.cur_rule_vars.as_ref().map(|vars| {
            let scope = Vars::new();
            scope.merge_from(vars);
            Arc::new(scope)
        });
        unbind(private_scoped_vars);
        action.lock().rule_vars = scope;

        let actual_inputs = action.lock().actual_inputs.clone();
        for input in actual_inputs {
            let dependency = self.build_plan(input, Some(trigger))?;
            action.lock().deps.push((input, dependency));
        }
        let actual_order_only_inputs = action.lock().actual_order_only_inputs.clone();
        for input in actual_order_only_inputs {
            let dependency = self.build_plan(input, Some(trigger))?;
            action.lock().order_onlys.push((input, dependency));
        }
        unbind(scoped_vars);

        Ok((action, true))
    }

    fn add_validations(
        &mut self,
        output: Symbol,
        n: &Arc<Mutex<DepNode>>,
        validations: Vec<Symbol>,
    ) -> Result<()> {
        for validation in validations {
            if n.lock().actual_validations.contains(&validation) {
                continue;
            }
            if !self.ev.session.flags.use_ninja_validations {
                error_loc!(
                    self.ev,
                    n.lock().loc.as_ref(),
                    ".KATI_VALIDATIONS not allowed without --use_ninja_validations"
                );
            }
            let dependency = self.build_plan(validation, Some(output))?;
            let mut node = n.lock();
            node.actual_validations.push(validation);
            node.validations.push((validation, dependency));
        }
        Ok(())
    }

    /// Every real member is a public completion join. It owns no recipe;
    /// consumers wait for every independent action that declared the member.
    fn build_grouped_double_member(
        &mut self,
        output: Symbol,
        join: Arc<Mutex<DepNode>>,
        rules: Vec<Arc<Rule>>,
        validations: Vec<Symbol>,
    ) -> Result<Arc<Mutex<DepNode>>> {
        let shared = self
            .double_memberships
            .get(&output)
            .is_some_and(|memberships| memberships.len() > 1);
        let mut actions = Vec::with_capacity(rules.len());
        let mut created_action = false;
        for rule in rules {
            let id = Self::double_action_id(&rule, output);
            let (action, newly_created) = self.build_double_action(rule, output)?;
            created_action |= newly_created;
            actions.push((id, action));
        }
        if shared && created_action {
            actions.sort_by_key(|(id, _)| self.double_action_creation_indices[id]);
            for pair in actions.windows(2) {
                let previous = &pair[0].1;
                let action = &pair[1].1;
                let previous_output = previous.lock().output;
                if !action
                    .lock()
                    .order_onlys
                    .iter()
                    .any(|(output, _)| *output == previous_output)
                {
                    action
                        .lock()
                        .order_onlys
                        .push((previous_output, previous.clone()));
                }
            }
        }

        {
            let mut node = join.lock();
            node.recipe_output = output;
            node.grouped_double_join = true;
            node.has_rule = true;
            node.is_default_target = self.first_rule == Some(output);
            node.loc = actions
                .first()
                .and_then(|(_, action)| action.lock().loc.clone());
            for (_, action) in &actions {
                let action_output = action.lock().output;
                node.actual_inputs.push(action_output);
                node.deps.push((action_output, action.clone()));
            }
            // `is_remakable` refuses a target that can never settle. The
            // actions carry that property now, but the Makefile is looked up
            // by its own name, so the join has to answer for the chain.
            node.unconditional_double_colon = actions
                .iter()
                .any(|(_, action)| action.lock().unconditional_double_colon);
        }
        self.done.insert(output, join.clone());
        self.add_validations(output, &join, validations)?;
        Ok(join)
    }

    /// An ordinary grouped rule keeps the outputs written on that rule even
    /// when a later grouped recipe changes which action one peer selects when
    /// reached directly. The rules attached to each peer still contribute
    /// scheduling prerequisites to this action, including another expansion
    /// of the shared rule in that peer's scope. They are returned separately so they do
    /// not enter `$<`, `$^`, `$+`, `$?`, or `$|`, and their target-specific
    /// variables never replace the triggering member's scope.
    fn grouped_single_peers(
        &self,
        output: Symbol,
        merger: &Arc<Mutex<RuleMerger>>,
    ) -> (Vec<Symbol>, Vec<(Symbol, Arc<Rule>)>) {
        let locked = merger.lock();
        let Some(primary_rule) = locked
            .primary_rule
            .as_ref()
            .filter(|rule| rule.is_grouped && !rule.is_double_colon)
            .cloned()
        else {
            return (Vec::new(), Vec::new());
        };

        let grouped_outputs = primary_rule.outputs.clone();
        let mut peer_rules = Vec::new();
        for grouped_output in &primary_rule.outputs {
            if *grouped_output == output {
                continue;
            }
            let Some(peer_merger) = self.rules.get(grouped_output) else {
                continue;
            };
            let peer_merger = peer_merger.lock();
            let mut seen = HashSet::new();
            for rule in &peer_merger.rules {
                if seen.insert(Self::rule_id(rule)) {
                    peer_rules.push((*grouped_output, rule.clone()));
                }
            }
        }
        (grouped_outputs, peer_rules)
    }

    /// Under `.SECONDEXPANSION` a pattern rule's prerequisites are not known
    /// until the stem is, so the expansion belongs here, once per candidate,
    /// rather than after the search has settled on one.
    fn expanded_pattern_inputs(
        &mut self,
        rule: &Rule,
        candidate_order: usize,
        output: Symbol,
        pat: &Pattern,
        output_str: &Bytes,
    ) -> Result<Option<(Vec<Symbol>, Vec<Symbol>)>> {
        let Some(text) = rule.deferred_prerequisites.clone() else {
            return Ok(None);
        };
        let key = (candidate_order, output);
        if let Some(found) = self.expanded.get(&key) {
            return Ok(Some(found.clone()));
        }
        let stem = Bytes::copy_from_slice(pat.stem(output_str));
        let (recorded, recorded_order_only) = self.recorded_prerequisites(output);
        let expanded = self.expand_pattern_prerequisites_again(
            output,
            stem,
            (&recorded, &recorded_order_only),
            &text,
        )?;
        self.expanded.insert(key, expanded.clone());
        Ok(Some(expanded))
    }

    /// The pattern rules that could make `output`, in the order GNU Make tries
    /// them.
    ///
    /// `pattern_search` collects one candidate per target pattern that matches,
    /// in the order the rules were written, and then stable-sorts them by stem
    /// length: the most specific rule is tried first and a tie is settled by
    /// which was written first. Population has already removed any rule that a
    /// later definition replaced.
    ///
    /// A rule with no recipe is never collected. That is the other half of how
    /// a redeclaration cancels: the replacement leaves the recipe-less rule
    /// holding the identity, and the search then refuses to consider it, so the
    /// target has no rule at all rather than one that makes it out of nothing.
    /// It also settles what such a rule does to targets reached some other way:
    /// nothing. Its prerequisites are not added to anything, because a rule the
    /// search never collects contributes neither recipe nor prerequisite.
    fn ordered_candidates(&self, output_str: &Bytes) -> Vec<ImplicitCandidate> {
        let mut candidates = self.implicit_rules.get(output_str);
        candidates.retain(|candidate| {
            !candidate.rule.cmds.is_empty()
                && Pattern::new(candidate.pattern.as_bytes(&self.ev.session)).matches(output_str)
        });
        candidates.sort_by_key(|candidate| {
            let pat = Pattern::new(candidate.pattern.as_bytes(&self.ev.session));
            (pat.stem(output_str).len(), candidate.order)
        });
        candidates
    }

    fn can_pick_implicit_rule(
        &mut self,
        rule: &Rule,
        matched: Symbol,
        candidate_order: usize,
        output: Symbol,
        n: Arc<Mutex<DepNode>>,
        chaining: bool,
    ) -> Result<Option<Arc<Rule>>> {
        let output_str = output.as_bytes(&self.ev.session);
        let pat = Pattern::new(matched.as_bytes(&self.ev.session));
        let deferred =
            self.expanded_pattern_inputs(rule, candidate_order, output, &pat, &output_str)?;
        let inputs: Vec<(Symbol, bool)> = match &deferred {
            // A deferred list is one string until it is expanded, so
            // which word the `%` was in is no longer knowable.
            Some((inputs, _)) => inputs.iter().map(|input| (*input, false)).collect(),
            None => rule
                .inputs
                .iter()
                .map(|input| {
                    let text = input.as_bytes(&self.ev.session);
                    let from_pattern = text.contains(&b'%');
                    let buf = pat.append_subst(&output_str, &text);
                    (self.ev.session.intern(buf), from_pattern)
                })
                .collect(),
        };
        let mut invented = Vec::new();
        for (sym, from_pattern) in inputs {
            if self.exists(sym) {
                continue;
            }
            if !(chaining && self.can_be_made_implicitly(sym, 0)?) {
                return Ok(None);
            }
            if from_pattern && !self.mentioned.contains(&sym) {
                invented.push(sym);
            }
        }
        self.intermediates.extend(invented);

        let mut rule = rule.clone();
        if let Some((inputs, order_only_inputs)) = deferred {
            rule.deferred_prerequisites = None;
            rule.inputs = inputs;
            rule.order_only_inputs = order_only_inputs;
        }
        if rule.output_patterns.len() > 1 {
            // A pattern rule with several target patterns is one recipe that
            // makes all of them, so the rest are this node's outputs — unless
            // the name already has a maker of its own. GNU Make's `also_make`
            // only marks such a name updated when this recipe runs; it does not
            // take it away from the rule its own search chose, and two rules
            // that can each make one name is not an error to it.
            let pat = Pattern::new(matched.as_bytes(&self.ev.session));
            for output_pattern in rule.output_patterns.clone() {
                if output_pattern == matched {
                    continue;
                }
                let buf = pat.append_subst(&output_str, &output_pattern.as_bytes(&self.ev.session));
                let sym = self.ev.session.intern(buf);
                if self.done.contains_key(&sym) {
                    continue;
                }
                // Each of these names is protected by the pattern that spelled
                // it, not by the one the search matched, and this is the last
                // point at which the two are still told apart.
                if self.precious_patterns.contains(&output_pattern) {
                    self.precious.insert(sym);
                }
                self.done.insert(sym, n.clone());
                let mut node = n.lock();
                node.implicit_outputs.push(sym);
                node.peer_outputs.push(sym);
            }
            rule.output_patterns.clear();
            rule.output_patterns.push(matched);
        }
        Ok(Some(Arc::new(rule)))
    }

    fn merge_implicit_rule_vars(
        &self,
        output: Symbol,
        vars: Option<Arc<Vars>>,
    ) -> Option<Arc<Vars>> {
        let Some(mut found) = self.rule_vars.get(&output).cloned() else {
            return vars;
        };
        let Some(vars) = vars else {
            return Some(found.clone());
        };
        let r = Arc::make_mut(&mut found);
        r.merge_from(&vars);
        Some(found)
    }

    /// Step 6 of GNU Make's implicit rule search: whether an implicit rule could
    /// make this. Nothing is built here — build_plan descends into the
    /// prerequisite anyway, and the search one level down succeeds normally.
    fn can_be_made_implicitly(&mut self, output: Symbol, depth: usize) -> Result<bool> {
        if depth >= MAX_IMPLICIT_CHAIN || !self.chaining.insert(output) {
            return Ok(false);
        }
        let answer = self.implicit_chain_exists(output, depth);
        self.chaining.remove(&output);
        answer
    }

    fn implicit_chain_exists(&mut self, output: Symbol, depth: usize) -> Result<bool> {
        let output_str = output.as_bytes(&self.ev.session);
        for candidate in self.ordered_candidates(&output_str) {
            let rule = candidate.rule;
            // Make's step 6a: a non-terminal match-anything rule is not allowed
            // to make an intermediate.
            if !rule.is_double_colon && self.matches_anything(&rule) {
                continue;
            }
            let pat = Pattern::new(candidate.pattern.as_bytes(&self.ev.session));
            let inputs = match self.expanded_pattern_inputs(
                &rule,
                candidate.order,
                output,
                &pat,
                &output_str,
            )? {
                Some((inputs, _)) => inputs,
                None => rule
                    .inputs
                    .iter()
                    .map(|input| {
                        let buf = pat.append_subst(&output_str, &input.as_bytes(&self.ev.session));
                        self.ev.session.intern(buf)
                    })
                    .collect(),
            };
            let mut ok = true;
            for i in inputs {
                if !self.exists(i) && !self.can_be_made_implicitly(i, depth + 1)? {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(true);
            }
        }

        let Some(suffix) = get_ext(&output_str) else {
            return Ok(false);
        };
        if !suffix.starts_with(b".") {
            return Ok(false);
        }
        let Some(found) = self.suffix_rules.get(&suffix[1..]).cloned() else {
            return Ok(false);
        };
        for irule in &found {
            let input = replace_suffix(&mut self.ev.session, output, &irule.inputs[0]);
            if self.exists(input) || self.can_be_made_implicitly(input, depth + 1)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn matches_anything(&self, rule: &Rule) -> bool {
        rule.output_patterns
            .iter()
            .any(|p| p.as_bytes(&self.ev.session).as_ref() == b"%")
    }

    fn pick_rule(
        &mut self,
        output: Symbol,
        n: &Arc<Mutex<DepNode>>,
    ) -> Result<Option<PickedRuleInfo>> {
        let rule_merger = self.lookup_rule_merger(output);
        // Applies however the recipe is found — GNU Make looks pattern
        // variables up from the target's name, not from the rule that makes it.
        let patterns = self.matching_pattern_vars(output);
        let vars = self.lookup_rule_vars(output);
        if let Some(rule_merger) = &rule_merger
            && rule_merger.lock().primary_rule.is_some()
        {
            let mut vars = vars;
            for (sym, _) in &rule_merger.lock().implicit_outputs {
                vars = self.merge_implicit_rule_vars(*sym, vars);
            }
            return Ok(Some(PickedRuleInfo {
                merger: Some(rule_merger.clone()),
                pattern_rule: None,
                vars: Self::scopes_for(&patterns, vars),
            }));
        }

        // Steps 5 then 6, over the same rules. The first pass must finish
        // first: a rule whose prerequisites exist beats one whose would have to
        // be invented, however far down the list it is.
        for chaining in [false, true] {
            if let Some(picked) =
                self.pick_pattern_rule(output, n, &rule_merger, &patterns, &vars, chaining)?
            {
                return Ok(Some(picked));
            }
        }

        if rule_merger.is_some() {
            return Ok(Some(PickedRuleInfo {
                merger: rule_merger,
                pattern_rule: None,
                vars: Self::scopes_for(&patterns, vars),
            }));
        }
        // Make's step 7, and the last thing it tries. Only for a target with no
        // rule at all that is not already there.
        let default_rule = self.default_rule.clone().filter(|_| !self.exists(output));
        Ok(default_rule.map(|rule| PickedRuleInfo {
            merger: None,
            pattern_rule: Some(rule),
            vars: Self::scopes_for(&patterns, vars),
        }))
    }

    /// The matching patterns, then the target's own scope on top of them.
    fn scopes_for(patterns: &[Arc<Vars>], own: Option<Arc<Vars>>) -> Vec<Arc<Vars>> {
        let mut scopes = patterns.to_vec();
        scopes.extend(own);
        scopes
    }

    /// Every scope that applies to `output`, weakest first, without consulting
    /// the rule that makes it. For callers that have a target's name and no
    /// picked rule to go with it.
    fn applicable_rule_vars(&self, output: Symbol) -> Vec<Arc<Vars>> {
        let patterns = self.matching_pattern_vars(output);
        Self::scopes_for(&patterns, self.lookup_rule_vars(output))
    }

    /// Install `scopes` into the rule scope, weakest first, and record the
    /// bindings so the caller can decide how long each one lives.
    ///
    /// One scope at a time rather than all at once: `+=` in a later scope
    /// appends to what the earlier ones left, which is what reading down GNU
    /// Make's chain of variable sets does, and merging them first would lose
    /// every value but the last.
    fn apply_rule_vars(
        &mut self,
        scopes: &[Arc<Vars>],
        node: &Arc<Mutex<DepNode>>,
        frame: &ScopedFrame,
        scoped: &mut Vec<ScopedVar>,
        private_scoped: &mut Vec<ScopedVar>,
    ) -> Result<()> {
        for vars in scopes {
            // Sorted because the order is observable and a HashMap's varies per
            // process. By name, not Make's order, which is as written — this
            // buys reproducibility only.
            let mut targeted = vars
                .0
                .lock()
                .iter()
                .map(|(name, var)| (*name, var.clone()))
                .collect::<Vec<_>>();
            targeted.sort_by_cached_key(|(name, _)| name.as_bytes(&self.ev.session));
            // `+=` last, and its right-hand side expanded once every other
            // target-specific variable is in scope. `all: A += $(Z)` beside
            // `all: Z = changed` appends `changed`, not whatever Z was outside
            // the rule, and expanding while the scope is half built reads the
            // outer one.
            targeted.sort_by_key(|(_, var)| var.read().assign_op == Some(AssignOp::PlusEq));
            for (name, var) in &targeted {
                // Off the declaration rather than the value: `+=` resolves to a
                // fresh simple variable and would leave the keyword behind.
                let is_private = var.read().is_private;
                let mut new_var = var.clone();
                match var.read().assign_op {
                    Some(AssignOp::PlusEq) => {
                        if let Some(old_var) = self.ev.lookup_var(*name)? {
                            let mut s = old_var.read().eval_to_buf_mut(self.ev)?;
                            if !s.is_empty() {
                                s.put_u8(b' ')
                            }
                            new_var.read().eval(self.ev, &mut s)?;
                            new_var = Variable::with_simple_string(
                                s.freeze(),
                                old_var.read().origin(),
                                frame.current(),
                                node.lock().loc.clone(),
                            );
                        }
                    }
                    Some(AssignOp::QuestionEq) if self.ev.lookup_var(*name)?.is_some() => {
                        continue;
                    }
                    _ => {}
                }

                if *name == self.depfile_var_name {
                    node.lock().depfile_var = Some(new_var);
                } else if *name == self.implicit_outputs_var_name
                    || *name == self.validations_var_name
                {
                } else if *name == self.ninja_pool_var_name {
                    node.lock().ninja_pool_var = Some(new_var);
                } else if *name == self.tags_var_name {
                    node.lock().tags_var = Some(new_var);
                } else {
                    let scoped_var =
                        ScopedVar::new(self.cur_rule_vars.clone().unwrap(), *name, new_var);
                    if is_private {
                        private_scoped.push(scoped_var);
                    } else {
                        scoped.push(scoped_var);
                    }
                }
            }
        }
        Ok(())
    }

    fn pick_pattern_rule(
        &mut self,
        output: Symbol,
        n: &Arc<Mutex<DepNode>>,
        rule_merger: &Option<Arc<Mutex<RuleMerger>>>,
        patterns: &[Arc<Vars>],
        vars: &Option<Arc<Vars>>,
        chaining: bool,
    ) -> Result<Option<PickedRuleInfo>> {
        let candidates = self.ordered_candidates(&output.as_bytes(&self.ev.session));
        for candidate in candidates {
            let Some(pattern_rule) = self.can_pick_implicit_rule(
                &candidate.rule,
                candidate.pattern,
                candidate.order,
                output,
                n.clone(),
                chaining,
            )?
            else {
                continue;
            };
            // The picked rule's own output pattern needs no special merge: it
            // matched this target, so `matching_pattern_vars` already found any
            // variables written against it.
            return Ok(Some(PickedRuleInfo {
                merger: rule_merger.clone(),
                pattern_rule: Some(pattern_rule),
                vars: Self::scopes_for(patterns, vars.clone()),
            }));
        }

        let output_str = output.as_bytes(&self.ev.session);
        let Some(output_suffix) = get_ext(&output_str) else {
            return Ok(None);
        };
        if !output_suffix.starts_with(b".") {
            return Ok(None);
        }
        let Some(found) = self.suffix_rules.get(&output_suffix[1..]).cloned() else {
            return Ok(None);
        };

        for irule in &found {
            assert!(irule.inputs.len() == 1);
            let input = replace_suffix(&mut self.ev.session, output, &irule.inputs[0]);
            if !self.exists(input) {
                if !(chaining && self.can_be_made_implicitly(input, 0)?) {
                    continue;
                }
                if !self.mentioned.contains(&input) {
                    self.intermediates.insert(input);
                }
            }

            let mut vars = vars.clone();
            // A suffix rule keeps `.c.o` as its written name, so variables set
            // against that name still belong to what it makes.
            if rule_merger.is_none() && vars.is_some() {
                assert!(irule.outputs.len() == 1);
                vars = self.merge_implicit_rule_vars(irule.outputs[0], vars);
            }
            return Ok(Some(PickedRuleInfo {
                merger: rule_merger.clone(),
                pattern_rule: Some(irule.clone()),
                vars: Self::scopes_for(patterns, vars),
            }));
        }
        Ok(None)
    }

    fn build_plan(
        &mut self,
        mut output: Symbol,
        needed_by: Option<Symbol>,
    ) -> Result<Arc<Mutex<DepNode>>> {
        log!(
            "BuildPlan: {} for {needed_by:?}",
            output.display(&self.ev.session)
        );

        if let Some(found) = self.done.get(&output) {
            // Reaching a name in its own right is what stops it being a peer:
            // GNU Make decides that name's freshness from that name, so its
            // absence has to be able to make the recipe run again.
            found.lock().peer_outputs.retain(|peer| *peer != output);
            return Ok(found.clone());
        }

        let is_intermediate = self.treat_as_intermediate(output);
        let n = DepNode::new(
            output,
            self.phony.contains(&output),
            self.restat.contains(&output),
            self.ignore_errors.contains(&output),
            is_intermediate,
            is_intermediate && !self.all_secondary && !self.secondary.contains(&output),
        );
        self.done.insert(output, n.clone());

        let Some(mut picked_rule_info) = self.pick_rule(output, &n)? else {
            return Ok(n);
        };
        if let Some(merger) = &picked_rule_info.merger
            && merger.lock().parent.is_some()
        {
            output = merger.lock().parent_sym.unwrap();
            self.done.insert(output, n.clone());
            n.lock().output = output;
            let Some(new_picked_rule_info) = self.pick_rule(output, &n)? else {
                return Ok(n);
            };
            // Update the picked_rule_info with the new values
            picked_rule_info = new_picked_rule_info;
        }
        if let Some(merger) = &picked_rule_info.merger {
            let grouped_double = {
                let merger = merger.lock();
                // Every `::` record is a rule of its own: GNU Make walks the
                // chain in `update_file` and weighs each entry against the
                // prerequisites that entry declared. Records only need to be
                // told apart once there is more than one of them, so a lone
                // `::` record keeps the single-node shape it already had.
                (merger.is_double_colon
                    && (merger.rules.len() > 1
                        || merger
                            .rules
                            .iter()
                            .any(|rule| rule.is_grouped && rule.is_double_colon)))
                .then(|| (merger.rules.clone(), merger.validations.clone()))
            };
            if let Some((rules, validations)) = grouped_double {
                return self.build_grouped_double_member(output, n, rules, validations);
            }
        }
        let mut grouped_outputs = Vec::new();
        let mut grouped_peer_rules = Vec::new();
        if let Some(merger) = picked_rule_info.merger.take() {
            let (outputs, peer_rules) = self.grouped_single_peers(output, &merger);
            picked_rule_info.merger = Some(merger);
            grouped_outputs = outputs;
            grouped_peer_rules = peer_rules;
        }
        let output_str = output.as_bytes(&self.ev.session);

        // A static pattern rule reaches this the same way an explicit one does,
        // so its stem is read off the rule rather than off the search.
        let (deferred, independent, unconditional_double_colon) = picked_rule_info
            .merger
            .as_ref()
            .map(|merger| {
                let merger = merger.lock();
                let deferred = merger
                    .rules
                    .iter()
                    .filter(|rule| {
                        rule.deferred_prerequisites.is_some()
                            && prerequisites_reach(&self.ev.session, rule, output)
                    })
                    .map(|rule| {
                        (
                            rule.deferred_prerequisites.clone().unwrap(),
                            self.stem_of(rule, &output_str),
                            merger.is_double_colon
                                && !rule.cmds.is_empty()
                                && rule.inputs.is_empty()
                                && rule.order_only_inputs.is_empty(),
                        )
                    })
                    .collect::<Vec<_>>();
                let unconditional = merger.is_double_colon
                    && merger.rules.iter().any(|rule| {
                        !rule.cmds.is_empty()
                            && rule.inputs.is_empty()
                            && rule.order_only_inputs.is_empty()
                            && rule.deferred_prerequisites.is_none()
                    });
                (deferred, merger.is_double_colon, unconditional)
            })
            .unwrap_or_default();
        n.lock().unconditional_double_colon = unconditional_double_colon;
        picked_rule_info
            .merger
            .clone()
            .unwrap_or_else(RuleMerger::new)
            .lock()
            .fill_dep_node(
                &mut self.ev.session,
                output,
                &picked_rule_info.pattern_rule,
                &grouped_outputs,
                &n,
            );
        let grouped_is_phony = picked_rule_info.merger.as_ref().is_some_and(|merger| {
            let merger = merger.lock();
            grouped_outputs
                .iter()
                .any(|grouped_output| self.phony.contains(grouped_output))
                || merger.rules.iter().any(|rule| {
                    rule.is_grouped
                        && rule
                            .outputs
                            .iter()
                            .any(|grouped_output| self.phony.contains(grouped_output))
                })
        });
        if grouped_is_phony {
            n.lock().is_phony = true;
        }

        let previous_scope = (!grouped_outputs.is_empty())
            .then(|| self.push_expansion_scope(&picked_rule_info.vars));
        let expanded = (|| -> Result<()> {
            for (text, stem, unconditional_candidate) in deferred {
                // Each `::` rule stands on its own, so nothing another one
                // declared is in scope for this one's automatic variables.
                let recorded = if independent {
                    (Vec::new(), Vec::new())
                } else {
                    let node = n.lock();
                    (
                        node.actual_inputs.clone(),
                        node.actual_order_only_inputs.clone(),
                    )
                };
                let (inputs, order_only) = self.expand_prerequisites_again(
                    output,
                    stem,
                    (&recorded.0, &recorded.1),
                    &text,
                )?;
                let unconditional =
                    unconditional_candidate && inputs.is_empty() && order_only.is_empty();
                let mut node = n.lock();
                node.unconditional_double_colon |= unconditional;
                node.actual_inputs.extend(inputs);
                node.actual_order_only_inputs.extend(order_only);
            }
            Ok(())
        })();
        if let Some(previous_scope) = previous_scope {
            self.pop_expansion_scope(previous_scope);
        }
        expanded?;

        // Ordinary `&:` includes every peer rule in the shared action's
        // scheduling and freshness test, but GNU Make hides those peer-only
        // prerequisites from the triggering member's automatic variables.
        let mut grouped_peer_inputs = Vec::new();
        let mut grouped_peer_order_only = Vec::new();
        for (peer_output, rule) in grouped_peer_rules {
            grouped_peer_inputs.extend(apply_output_pattern(
                &mut self.ev.session,
                &rule,
                peer_output,
                &rule.inputs,
            ));
            grouped_peer_order_only.extend(apply_output_pattern(
                &mut self.ev.session,
                &rule,
                peer_output,
                &rule.order_only_inputs,
            ));
            if let Some(text) = rule
                .deferred_prerequisites
                .as_ref()
                .filter(|_| prerequisites_reach(&self.ev.session, &rule, peer_output))
            {
                let peer_text = peer_output.as_bytes(&self.ev.session);
                let stem = self.stem_of(&rule, &peer_text);
                let recorded = self.recorded_prerequisites(peer_output);
                let peer_vars = self.applicable_rule_vars(peer_output);
                let previous_scope = self.push_expansion_scope(&peer_vars);
                let expanded = self.expand_prerequisites_again(
                    peer_output,
                    stem,
                    (&recorded.0, &recorded.1),
                    text,
                );
                self.pop_expansion_scope(previous_scope);
                let (inputs, order_only) = expanded?;
                grouped_peer_inputs.extend(inputs);
                grouped_peer_order_only.extend(order_only);
            }
        }

        // VPATH applies to hidden peer dependencies too. Append them for one
        // pass, then split them away before automatic variables see the node.
        let (visible_inputs, visible_order_only) = {
            let mut node = n.lock();
            let visible_inputs = node.actual_inputs.len();
            let visible_order_only = node.actual_order_only_inputs.len();
            node.actual_inputs.extend(grouped_peer_inputs);
            node.actual_order_only_inputs
                .extend(grouped_peer_order_only);
            (visible_inputs, visible_order_only)
        };
        self.resolve_vpaths(&n);
        let (grouped_peer_inputs, grouped_peer_order_only) = {
            let mut node = n.lock();
            let grouped_peer_inputs = node.actual_inputs.split_off(visible_inputs);
            let grouped_peer_order_only =
                node.actual_order_only_inputs.split_off(visible_order_only);
            (grouped_peer_inputs, grouped_peer_order_only)
        };
        self.take_out_waits(&n);
        let (grouped_peer_inputs, barriers) = self.without_waits(grouped_peer_inputs);
        self.wait_barriers.extend(barriers);
        let (grouped_peer_order_only, barriers) = self.without_waits(grouped_peer_order_only);
        self.wait_barriers.extend(barriers);

        let mut sv = Vec::new();
        let mut private_sv = Vec::new();
        let frame = self.ev.enter(
            FrameType::Dependency,
            output_str.clone(),
            n.lock().loc.clone().unwrap_or_default(),
        );

        self.apply_rule_vars(&picked_rule_info.vars, &n, &frame, &mut sv, &mut private_sv)?;

        // A `private` target-specific variable belongs to this target's own
        // recipe and to no prerequisite's, so the scope is read here, with it in
        // it, and it leaves before the prerequisites are planned.
        let scope = self.cur_rule_vars.as_ref().map(|vars| {
            let v = Vars::new();
            v.merge_from(vars);
            Arc::new(v)
        });
        unbind(private_sv);

        if self.ev.session.flags.warn_phony_looks_real
            && n.lock().is_phony
            && output_str.contains(&b'/')
        {
            if self.ev.session.flags.werror_phony_looks_real {
                error_loc!(
                    self.ev,
                    n.lock().loc.as_ref(),
                    "*** PHONY target \"{}\" looks like a real file (contains a \"/\")",
                    output.display(self.ev)
                );
            } else {
                warn_loc!(
                    self.ev,
                    n.lock().loc.as_ref(),
                    "warning: PHONY target \"{}\" looks like a real file (contains a \"/\")",
                    output.display(self.ev)
                );
            }
        }

        if !self.ev.session.flags.writable.is_empty() && !n.lock().is_phony {
            let mut found = false;
            for w in &self.ev.session.flags.writable {
                if output_str.starts_with(w.as_bytes()) {
                    found = true;
                    break;
                }
            }
            if !found {
                if self.ev.session.flags.werror_writable {
                    error_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "*** writing to readonly directory: \"{}\"",
                        output.display(self.ev)
                    );
                } else {
                    warn_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "warning: writing to readonly directory: \"{}\"",
                        output.display(self.ev)
                    );
                }
            }
        }

        // A grouped output may already have been reached through another
        // dependency path.  In that case its existing producer owns the name;
        // this action keeps only the peers that are still unclaimed.
        n.lock().implicit_outputs.retain(|implicit_output| {
            self.done
                .get(implicit_output)
                .is_none_or(|claimed| Arc::ptr_eq(claimed, &n))
        });
        let implicit_outputs = n.lock().implicit_outputs.clone();
        for output in implicit_outputs {
            self.done.insert(output, n.clone());

            let output_str = output.as_bytes(&self.ev.session);
            if self.ev.session.flags.warn_phony_looks_real
                && n.lock().is_phony
                && output_str.contains(&b'/')
            {
                if self.ev.session.flags.werror_phony_looks_real {
                    error_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "*** PHONY target \"{}\" looks like a real file (contains a \"/\")",
                        output.display(self.ev)
                    );
                } else {
                    warn_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "warning: PHONY target \"{}\" looks like a real file (contains a \"/\")",
                        output.display(self.ev)
                    );
                }
            }

            if !self.ev.session.flags.writable.is_empty() && !n.lock().is_phony {
                let mut found = false;
                for w in &self.ev.session.flags.writable {
                    if output_str.starts_with(w.as_bytes()) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    if self.ev.session.flags.werror_writable {
                        error_loc!(
                            self.ev,
                            n.lock().loc.as_ref(),
                            "*** writing to readonly directory: \"{}\"",
                            output.display(self.ev)
                        );
                    } else {
                        warn_loc!(
                            self.ev,
                            n.lock().loc.as_ref(),
                            "warning: writing to readonly directory: \"{}\"",
                            output.display(self.ev)
                        );
                    }
                }
            }
        }

        let actual_inputs = n.lock().actual_inputs.clone();
        for input in actual_inputs.into_iter().chain(grouped_peer_inputs) {
            let c = self.build_plan(input, Some(output))?;
            n.lock().deps.push((input, c.clone()));

            let mut is_phony = c.lock().is_phony;
            if !is_phony && !c.lock().has_rule && self.ev.session.flags.top_level_phony {
                is_phony = !input.as_bytes(&self.ev.session).contains(&b'/');
            }
            if !n.lock().is_phony && is_phony {
                if self.ev.session.flags.werror_real_to_phony {
                    error_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "*** real file \"{}\" depends on PHONY target \"{}\"",
                        output.display(self.ev),
                        input.display(self.ev)
                    );
                } else if self.ev.session.flags.warn_real_to_phony {
                    warn_loc!(
                        self.ev,
                        n.lock().loc.as_ref(),
                        "warning: real file \"{}\" depends on PHONY target \"{}\"",
                        output.display(self.ev),
                        input.display(self.ev)
                    );
                }
            }
        }

        let actual_order_only_inputs = n.lock().actual_order_only_inputs.clone();
        for input in actual_order_only_inputs
            .into_iter()
            .chain(grouped_peer_order_only)
        {
            let c = self.build_plan(input, Some(output))?;
            n.lock().order_onlys.push((input, c));
        }

        let actual_validations = n.lock().actual_validations.clone();
        for validation in actual_validations {
            if !self.ev.session.flags.use_ninja_validations {
                error_loc!(
                    self.ev,
                    n.lock().loc.as_ref(),
                    ".KATI_VALIDATIONS not allowed without --use_ninja_validations"
                );
            }
            let c = self.build_plan(validation, Some(output))?;
            n.lock().validations.push((validation, c));
        }

        // Block on werror_writable/werror_phony_looks_real, because otherwise we
        // can't rely on is_phony being valid for this check.
        if !n.lock().is_phony
            && n.lock().cmds.is_empty()
            && self.ev.session.flags.werror_writable
            && self.ev.session.flags.werror_phony_looks_real
        {
            let n = n.lock();
            if n.deps.is_empty() && n.order_onlys.is_empty() {
                if self.ev.session.flags.werror_real_no_cmds_or_deps {
                    error_loc!(
                        self.ev,
                        n.loc.as_ref(),
                        "*** target \"{}\" has no commands or deps that could create it",
                        output.display(self.ev)
                    );
                } else if self.ev.session.flags.warn_real_no_cmds_or_deps {
                    warn_loc!(
                        self.ev,
                        n.loc.as_ref(),
                        "warning: target \"{}\" has no commands or deps that could create it",
                        output.display(self.ev)
                    );
                }
            } else if n.actual_inputs.len() == 1 {
                if self.ev.session.flags.werror_real_no_cmds {
                    error_loc!(
                        self.ev,
                        n.loc.as_ref(),
                        "*** target \"{}\" has no commands. Should \"{}\" be using .KATI_IMPLICIT_OUTPUTS?",
                        output.display(self.ev),
                        n.actual_inputs[0].display(self.ev)
                    );
                } else if self.ev.session.flags.warn_real_no_cmds {
                    warn_loc!(
                        self.ev,
                        n.loc.as_ref(),
                        "warning: target \"{}\" has no commands. Should \"{}\" be using .KATI_IMPLICIT_OUTPUTS?",
                        output.display(self.ev),
                        n.actual_inputs[0].display(self.ev)
                    );
                }
            } else if self.ev.session.flags.werror_real_no_cmds {
                error_loc!(
                    self.ev,
                    n.loc.as_ref(),
                    "*** target \"{}\" has no commands that could create output file. Is a dependency missing .KATI_IMPLICIT_OUTPUTS?",
                    output.display(self.ev)
                );
            } else if self.ev.session.flags.warn_real_no_cmds {
                warn_loc!(
                    self.ev,
                    n.loc.as_ref(),
                    "warning: target \"{}\" has no commands that could create output file. Is a dependency missing .KATI_IMPLICIT_OUTPUTS?",
                    output.display(self.ev)
                );
            }
        }

        {
            let mut n = n.lock();
            n.has_rule = true;
            n.is_default_target = self.first_rule == Some(output);
            n.rule_vars = scope;
        }

        unbind(sv);
        Ok(n)
    }
}

/// Reduce the evaluated Makefile to the roots of a graph: the goals that were
/// asked for, and separately the generated Makefiles that have to exist before
/// those goals mean what they will mean.
pub fn make_dep(
    ev: &mut Evaluator,
    targets: Vec<Symbol>,
    read_makefiles: &[ReadMakefile],
    missing_includes: &[MissingInclude],
) -> Result<(Vec<NamedDepNode>, Vec<RegenerationRoot>)> {
    let mut db = DepBuilder::new(ev)?;
    let _tr = ScopedTimeReporter::new(&db.ev.session, "make dep (build)");
    db.build(targets, read_makefiles, missing_includes)
}

/// Whether the name has the shape Make reserves: a leading dot before any
/// directory separator.  A hidden-directory path such as `.deps/file.Po` is an
/// ordinary file target.
///
/// This is wider than the names that mean anything. To decide whether something
/// belongs in the graph, ask [`is_buildable_target`].
pub fn is_special_target(names: &impl Interner, output: &Symbol) -> bool {
    let s = output.as_bytes(names);
    s.starts_with(b".") && !s[1..].starts_with(b".") && !s.contains(&b'/')
}

const CONSUMED_BUILTIN_TARGETS: &[&str] = &[
    ".PHONY",
    ".SUFFIXES",
    ".KATI_RESTAT",
    ".WAIT",
    ".DEFAULT",
    ".SECONDEXPANSION",
    ".IGNORE",
    ".EXPORT_ALL_VARIABLES",
    ".ONESHELL",
    ".NOTPARALLEL",
    ".INTERMEDIATE",
    ".SECONDARY",
    ".NOTINTERMEDIATE",
    ".DELETE_ON_ERROR",
    ".PRECIOUS",
];

/// Special targets asking for what already happens: we never echo a recipe, and
/// 4.x ignores the last two.
const ACCEPTED_BUILTIN_TARGETS: &[&str] = &[".SILENT", ".LOW_RESOLUTION_TIME", ".POSIX"];

/// A closed list, because being a directive is not a property of the name's
/// shape: `.1` looks exactly like `.PHONY` and is an ordinary target.
pub fn is_directive_target(names: &impl Interner, output: &Symbol) -> bool {
    let s = output.as_bytes(names);
    CONSUMED_BUILTIN_TARGETS
        .iter()
        .chain(ACCEPTED_BUILTIN_TARGETS)
        .any(|name| name.as_bytes() == &s[..])
}

/// Suffix rules are excluded deliberately. Emitted as an ordinary node, `.c.o`
/// is claimed by the built-in `%.o: %.c` with an empty stem and runs
/// `cc -c -o .c.o` against no input, which is worse than refusing it.
pub fn is_buildable_target(names: &impl Interner, output: &Symbol) -> bool {
    !is_directive_target(names, output) && !is_suffix_rule(names, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_prerequisite_words_keep_reference_contents_whole() {
        let source = Bytes::from_static(b"\n %.a\\ $(subst |,x,$(S))| |tail");
        let words = implicit_prerequisite_words(&source).collect::<Vec<_>>();
        assert_eq!(
            words,
            vec![
                Bytes::from_static(br"%.a\"),
                Bytes::from_static(b"$(subst |,x,$(S))|"),
                Bytes::from_static(b"|"),
                Bytes::from_static(b"tail"),
            ]
        );
    }

    #[test]
    fn a_search_path_is_a_list_of_directories() {
        assert_eq!(
            search_path(&Bytes::from_static(b" build:. other/ ../up:: ")),
            vec![
                Bytes::from_static(b"build"),
                Bytes::from_static(b"other"),
                Bytes::from_static(b"../up"),
            ]
        );
        // A lone slash is a directory and keeps its only byte.
        assert_eq!(
            search_path(&Bytes::from_static(b"/")),
            vec![Bytes::from_static(b"/")]
        );
        assert!(search_path(&Bytes::from_static(b"  ")).is_empty());
    }

    #[test]
    fn a_search_directory_is_what_was_joined_to_the_name() {
        assert_eq!(
            search_directory(b"build/out.o", b"out.o"),
            Some(&b"build"[..])
        );
        // The name's own directory belongs to the name, not to the entry.
        assert_eq!(
            search_directory(b"build/sub/out.o", b"sub/out.o"),
            Some(&b"build"[..])
        );
        // A path shorter than the name it was made from cannot have one.
        assert_eq!(search_directory(b"out.o", b"out.o"), None);
    }

    #[test]
    fn test_is_suffix_rule() {
        let mut session = Session::new();
        let co = session.intern(".c.o");
        let foo = session.intern("foo");
        let dotco = session.intern(".co");
        let cob = session.intern(".c.o.b");
        let dep = session.intern(".deps/file.Po");
        assert!(is_suffix_rule(&session, &co));
        assert!(!is_suffix_rule(&session, &foo));
        assert!(!is_suffix_rule(&session, &dotco));
        assert!(!is_suffix_rule(&session, &cob));
        assert!(!is_suffix_rule(&session, &dep));
    }

    #[test]
    fn a_dot_named_target_is_something_to_build() {
        let mut session = Session::new();
        // An empty static-pattern stem leaves `.1`, which Make builds.
        for name in [".1", ".x", "foo", "..", ".deps/file.Po"] {
            let sym = session.intern(name);
            assert!(
                is_buildable_target(&session, &sym),
                "{name} should be built"
            );
        }
        for name in [
            ".PHONY",
            ".SUFFIXES",
            ".KATI_RESTAT",
            ".ONESHELL",
            ".WAIT",
            ".c.o",
        ] {
            let sym = session.intern(name);
            assert!(
                !is_buildable_target(&session, &sym),
                "{name} should not be built"
            );
        }
    }
}
