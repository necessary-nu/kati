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
    sync::Arc,
};

use crate::{
    error_loc,
    eval::{Evaluator, FrameType},
    expr::{Evaluable, Value},
    loc::Loc,
    log,
    rule::Rule,
    session::{Context, Session},
    stmt::AssignOp,
    strutil::{Pattern, get_ext, strip_ext, trim_leading_curdir, word_scanner},
    symtab::{Interner, Symbol},
    timeutil::ScopedTimeReporter,
    var::{ScopedVar, Var, Variable, Vars},
    warn_loc,
};

pub type NamedDepNode = (Symbol, Arc<Mutex<DepNode>>);

#[derive(Debug)]
pub struct DepNode {
    pub output: Symbol,
    pub cmds: Vec<Arc<Value>>,
    pub deps: Vec<NamedDepNode>,
    pub order_onlys: Vec<NamedDepNode>,
    pub validations: Vec<NamedDepNode>,
    pub has_rule: bool,
    pub is_default_target: bool,
    pub is_phony: bool,
    pub is_restat: bool,
    pub implicit_outputs: Vec<Symbol>,
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
    fn new(output: Symbol, is_phony: bool, is_restat: bool) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            output,
            cmds: Vec::new(),
            deps: Vec::new(),
            order_onlys: Vec::new(),
            validations: Vec::new(),
            has_rule: false,
            is_default_target: false,
            is_phony,
            is_restat,
            implicit_outputs: Vec::new(),
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

struct RuleTrieEntry {
    rule: Arc<Rule>,
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

    fn add(&mut self, name: &[u8], rule: Arc<Rule>) {
        if name.is_empty() || name.starts_with(b"%") {
            self.rules.push(RuleTrieEntry {
                rule,
                suffix: name.to_vec(),
            });
            return;
        }
        let c = name[0];
        self.children
            .entry(c)
            .or_insert_with(RuleTrie::new)
            .add(&name[1..], rule)
    }

    fn get(&self, name: &[u8]) -> Vec<Arc<Rule>> {
        let mut ret = Vec::new();
        for ent in &self.rules {
            if (ent.suffix.is_empty() && name.is_empty()) || name.ends_with(&ent.suffix[1..]) {
                ret.push(ent.rule.clone())
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
        n: &Arc<Mutex<DepNode>>,
    ) {
        let mut n = n.lock();
        if let Some(primary_rule) = &self.primary_rule {
            assert!(pattern_rule.is_none());
            self.fill_dep_node_from_rule(session, output, primary_rule, &mut n);
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
    cur_rule_vars: Option<Arc<Vars>>,

    implicit_rules: RuleTrie,
    suffix_rules: SuffixRuleMap,

    first_rule: Option<Symbol>,
    done: HashMap<Symbol, Arc<Mutex<DepNode>>>,
    phony: HashSet<Symbol>,
    restat: HashSet<Symbol>,
    depfile_var_name: Symbol,
    /// `VPATH`, the variable form of the directory search.
    vpath_var_name: Symbol,
    implicit_outputs_var_name: Symbol,
    ninja_pool_var_name: Symbol,
    validations_var_name: Symbol,
    tags_var_name: Symbol,
}

#[derive(Debug)]
struct PickedRuleInfo {
    merger: Option<Arc<Mutex<RuleMerger>>>,
    pattern_rule: Option<Arc<Rule>>,
    vars: Option<Arc<Vars>>,
}

impl<'a> DepBuilder<'a> {
    fn new(ev: &'a mut Evaluator) -> Result<Self> {
        let rule_vars = std::mem::take(&mut ev.rule_vars);
        let depfile_var_name = ev.session.intern(".KATI_DEPFILE");
        let vpath_var_name = ev.session.intern("VPATH");
        let implicit_outputs_var_name = ev.session.intern(".KATI_IMPLICIT_OUTPUTS");
        let ninja_pool_var_name = ev.session.intern(".KATI_NINJA_POOL");
        let validations_var_name = ev.session.intern(".KATI_VALIDATIONS");
        let tags_var_name = ev.session.intern(".KATI_TAGS");
        let mut ret = Self {
            ev,
            rules: HashMap::new(),
            rule_vars,
            cur_rule_vars: None,

            implicit_rules: RuleTrie::new(),
            suffix_rules: HashMap::new(),

            first_rule: None,
            done: HashMap::new(),
            phony: HashSet::new(),
            restat: HashSet::new(),
            depfile_var_name,
            vpath_var_name,
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

        ret.handle_special_targets();

        Ok(ret)
    }

    fn handle_special_targets(&mut self) {
        let phony = self.ev.session.intern(".PHONY");
        if let Some((targets, _)) = self.get_rule_inputs(phony) {
            for t in targets {
                self.phony.insert(t);
            }
        }
        let restat = self.ev.session.intern(".KATI_RESTAT");
        if let Some((targets, _)) = self.get_rule_inputs(restat) {
            for t in targets {
                self.restat.insert(t);
            }
        }
        let suffixes = self.ev.session.intern(".SUFFIXES");
        if let Some((targets, loc)) = self.get_rule_inputs(suffixes) {
            if targets.is_empty() {
                self.suffix_rules.clear();
            } else {
                warn_loc!(
                    self.ev,
                    Some(&loc),
                    "kati doesn't support .SUFFIXES with prerequisites"
                );
            }
        }

        for p in UNSUPPORTED_BUILTIN_TARGETS.iter().copied() {
            let sym = self.ev.session.intern(p);
            if let Some((_, loc)) = self.get_rule_inputs(sym) {
                warn_loc!(self.ev, Some(&loc), "kati doesn't support {p}");
            }
        }
    }

    fn build(&mut self, mut targets: Vec<Symbol>) -> Result<Vec<NamedDepNode>> {
        let Some(first_rule) = self.first_rule else {
            // GNU Make's own wording, because its test suite matches this
            // message exactly to learn what the program under test is called.
            // The name and the `Stop.` are added on the way out.
            error_loc!(self.ev, None, "*** No targets.");
        };

        if !self.ev.session.flags.gen_all_targets && targets.is_empty() {
            targets.push(first_rule);
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

        let mut nodes = Vec::new();
        for target in targets {
            let v = Arc::new(Vars::new());
            self.cur_rule_vars = Some(v.clone());
            self.ev.current_scope = Some(v.clone());
            let n = self.build_plan(target, None)?;
            nodes.push((target, n));
            self.ev.current_scope = None;
            self.cur_rule_vars = None;
        }
        Ok(nodes)
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

    /// Each prerequisite, moved to where the search found it.
    ///
    /// Resolved first and interned after, because finding the file needs the
    /// session to read and naming the result needs it to write.
    fn at_vpaths(&mut self, inputs: Vec<Symbol>) -> Vec<Symbol> {
        let mut resolved = Vec::with_capacity(inputs.len());
        for input in inputs {
            match self.at_vpath(input) {
                Some(found) => resolved.push(self.ev.session.intern(found)),
                None => resolved.push(input),
            }
        }
        resolved
    }

    /// Where one prerequisite was found, if it had to be looked for.
    ///
    /// A prerequisite with a rule of its own is left alone: it is going to be
    /// built here, so where an older copy of it might be lying is not a
    /// question worth asking.
    fn at_vpath(&self, input: Symbol) -> Option<Bytes> {
        if self.rules.contains_key(&input) || self.phony.contains(&input) {
            return None;
        }
        let name = input.as_bytes(&self.ev.session);
        if std::fs::exists(OsStr::from_bytes(&name)).is_ok_and(|found| found) {
            return None;
        }
        self.vpath_of(input)
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

    fn get_rule_inputs(&self, s: Symbol) -> Option<(Vec<Symbol>, Loc)> {
        let merger = self.rules.get(&s)?;
        let merger = merger.lock();
        let mut ret = Vec::new();
        assert!(!merger.rules.is_empty());
        for r in &merger.rules {
            for i in &r.inputs {
                ret.push(*i);
            }
        }

        Some((ret, merger.rules[0].loc.clone()))
    }

    fn populate_rules(&mut self) -> Result<()> {
        // TODO: Is this take necessary, or can we refactor how we pass around ev?
        for rule in std::mem::take(&mut self.ev.rules) {
            let rule = Arc::new(rule);
            if rule.outputs.is_empty() {
                self.populate_implicit_rule(rule)?;
            } else {
                self.populate_explicit_rule(rule)?;
            }
        }
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

        let mut output = output.as_bytes(&self.ev.session);
        output.advance(1);
        let dot_index = memchr(b'.', &output).unwrap();

        let input_suffix = output.slice(..dot_index);
        let output_suffix = output.slice(dot_index + 1..);
        let mut r = rule.clone();
        r.inputs.clear();
        let input_sym = self.ev.session.intern(input_suffix);
        r.inputs.push(input_sym);
        r.is_suffix_rule = true;
        self.suffix_rules
            .entry(output_suffix)
            .or_default()
            .push(Arc::new(r));
        Ok(true)
    }

    fn populate_explicit_rule(&mut self, rule: Arc<Rule>) -> Result<()> {
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

                self.implicit_rules.add(&op, rule.clone())
            }
        }
        Ok(())
    }

    fn lookup_rule_merger(&self, o: Symbol) -> Option<Arc<Mutex<RuleMerger>>> {
        self.rules.get(&o).cloned()
    }

    fn lookup_rule_vars(&self, o: Symbol) -> Option<Arc<Vars>> {
        self.rule_vars.get(&o).cloned()
    }

    fn can_pick_implicit_rule(
        &mut self,
        rule: &Rule,
        output: Symbol,
        n: Arc<Mutex<DepNode>>,
    ) -> Option<Arc<Rule>> {
        let output_str = output.as_bytes(&self.ev.session);
        let mut matched = None;
        for output_pattern in &rule.output_patterns {
            let pat = Pattern::new(output_pattern.as_bytes(&self.ev.session));
            if pat.matches(&output_str) {
                let mut ok = true;
                for input in &rule.inputs {
                    let buf = pat.append_subst(&output_str, &input.as_bytes(&self.ev.session));
                    let sym = self.ev.session.intern(buf);
                    if !self.exists(sym) {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    matched = Some(*output_pattern);
                    break;
                }
            }
        }
        let matched = matched?;

        let mut rule = rule.clone();
        if rule.output_patterns.len() > 1 {
            // We should mark all other output patterns as used.
            let pat = Pattern::new(matched.as_bytes(&self.ev.session));
            for output_pattern in rule.output_patterns.clone() {
                if output_pattern == matched {
                    continue;
                }
                let buf = pat.append_subst(&output_str, &output_pattern.as_bytes(&self.ev.session));
                let sym = self.ev.session.intern(buf);
                self.done.insert(sym, n.clone());
            }
            rule.output_patterns.clear();
            rule.output_patterns.push(matched);
        }
        Some(Arc::new(rule))
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

    fn pick_rule(&mut self, output: Symbol, n: &Arc<Mutex<DepNode>>) -> Option<PickedRuleInfo> {
        let rule_merger = self.lookup_rule_merger(output);
        let vars = self.lookup_rule_vars(output);
        if let Some(rule_merger) = &rule_merger
            && rule_merger.lock().primary_rule.is_some()
        {
            let mut vars = vars;
            for (sym, _) in &rule_merger.lock().implicit_outputs {
                vars = self.merge_implicit_rule_vars(*sym, vars);
            }
            return Some(PickedRuleInfo {
                merger: Some(rule_merger.clone()),
                pattern_rule: None,
                vars,
            });
        }

        let irules = self.implicit_rules.get(&output.as_bytes(&self.ev.session));
        for rule in irules.into_iter().rev() {
            let Some(pattern_rule) = self.can_pick_implicit_rule(&rule, output, n.clone()) else {
                continue;
            };
            if rule_merger.is_some() {
                return Some(PickedRuleInfo {
                    merger: rule_merger,
                    pattern_rule: Some(pattern_rule),
                    vars,
                });
            }
            assert!(pattern_rule.output_patterns.len() == 1);
            let vars = self.merge_implicit_rule_vars(pattern_rule.output_patterns[0], vars);
            return Some(PickedRuleInfo {
                merger: None,
                pattern_rule: Some(pattern_rule),
                vars,
            });
        }

        let output_str = output.as_bytes(&self.ev.session);
        let Some(output_suffix) = get_ext(&output_str) else {
            if rule_merger.is_some() {
                return Some(PickedRuleInfo {
                    merger: rule_merger,
                    pattern_rule: None,
                    vars,
                });
            } else {
                return None;
            }
        };
        if !output_suffix.starts_with(b".") {
            if rule_merger.is_some() {
                return Some(PickedRuleInfo {
                    merger: rule_merger,
                    pattern_rule: None,
                    vars,
                });
            } else {
                return None;
            }
        }
        let output_suffix = &output_suffix[1..];

        let Some(found) = self.suffix_rules.get(output_suffix) else {
            if rule_merger.is_some() {
                return Some(PickedRuleInfo {
                    merger: rule_merger,
                    pattern_rule: None,
                    vars,
                });
            } else {
                return None;
            }
        };

        let found = found.clone();
        for irule in &found {
            assert!(irule.inputs.len() == 1);
            let input = replace_suffix(&mut self.ev.session, output, &irule.inputs[0]);
            if !self.exists(input) {
                continue;
            }

            if rule_merger.is_some() {
                return Some(PickedRuleInfo {
                    merger: rule_merger,
                    pattern_rule: Some(irule.clone()),
                    vars,
                });
            }
            let mut vars = vars;
            if vars.is_some() {
                assert!(irule.outputs.len() == 1);
                vars = self.merge_implicit_rule_vars(irule.outputs[0], vars);
            }
            return Some(PickedRuleInfo {
                merger: rule_merger,
                pattern_rule: Some(irule.clone()),
                vars,
            });
        }

        if rule_merger.is_some() {
            Some(PickedRuleInfo {
                merger: rule_merger,
                pattern_rule: None,
                vars,
            })
        } else {
            None
        }
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
            return Ok(found.clone());
        }

        let n = DepNode::new(
            output,
            self.phony.contains(&output),
            self.restat.contains(&output),
        );
        self.done.insert(output, n.clone());

        let Some(mut picked_rule_info) = self.pick_rule(output, &n) else {
            return Ok(n);
        };
        if let Some(merger) = &picked_rule_info.merger
            && merger.lock().parent.is_some()
        {
            output = merger.lock().parent_sym.unwrap();
            self.done.insert(output, n.clone());
            n.lock().output = output;
            let Some(new_picked_rule_info) = self.pick_rule(output, &n) else {
                return Ok(n);
            };
            // Update the picked_rule_info with the new values
            picked_rule_info = new_picked_rule_info;
        }
        let output_str = output.as_bytes(&self.ev.session);

        picked_rule_info
            .merger
            .unwrap_or_else(RuleMerger::new)
            .lock()
            .fill_dep_node(
                &mut self.ev.session,
                output,
                &picked_rule_info.pattern_rule,
                &n,
            );
        self.resolve_vpaths(&n);

        let mut sv = Vec::new();
        let frame = self.ev.enter(
            FrameType::Dependency,
            output_str.clone(),
            n.lock().loc.clone().unwrap_or_default(),
        );

        if let Some(vars) = &picked_rule_info.vars {
            for (name, var) in vars.0.lock().iter() {
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
                                n.lock().loc.clone(),
                            );
                        }
                    }
                    Some(AssignOp::QuestionEq) if self.ev.lookup_var(*name)?.is_some() => {
                        continue;
                    }
                    _ => {}
                }

                if *name == self.depfile_var_name {
                    n.lock().depfile_var = Some(new_var);
                } else if *name == self.implicit_outputs_var_name
                    || *name == self.validations_var_name
                {
                } else if *name == self.ninja_pool_var_name {
                    n.lock().ninja_pool_var = Some(new_var);
                } else if *name == self.tags_var_name {
                    n.lock().tags_var = Some(new_var);
                } else {
                    sv.push(ScopedVar::new(
                        self.cur_rule_vars.clone().unwrap(),
                        *name,
                        new_var,
                    ));
                }
            }
        }

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
        for input in actual_inputs {
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
        for input in actual_order_only_inputs {
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
            if let Some(cur_rule_vars) = &self.cur_rule_vars {
                let v = Vars::new();
                v.merge_from(cur_rule_vars);
                n.rule_vars = Some(Arc::new(v));
            } else {
                n.rule_vars = None
            }
        }

        Ok(n)
    }
}

pub fn make_dep(ev: &mut Evaluator, targets: Vec<Symbol>) -> Result<Vec<NamedDepNode>> {
    let mut db = DepBuilder::new(ev)?;
    let _tr = ScopedTimeReporter::new(&db.ev.session, "make dep (build)");
    db.build(targets)
}

/// Whether the name has the shape Make reserves for itself.
///
/// This answers a question about spelling, not about meaning, and it is
/// deliberately wider than the set of names that actually mean something: it
/// decides which rule may become the default goal, and GNU Make's rule there is
/// the leading dot rather than a list. Nothing that puts nodes into a build
/// graph should ask this — a Makefile is free to write `.1:` and expect `.1`
/// built. Ask [`is_directive_target`] instead.
pub fn is_special_target(names: &impl Interner, output: &Symbol) -> bool {
    let s = output.as_bytes(names);
    s.starts_with(b".") && !s[1..].starts_with(b".")
}

/// The special targets whose whole handling is [`DepBuilder::handle_special_targets`]
/// reading them.
const CONSUMED_BUILTIN_TARGETS: &[&str] = &[".PHONY", ".SUFFIXES", ".KATI_RESTAT"];

/// The special targets kati declines. Declining consumes them just as
/// thoroughly, so for the purpose of building they belong with the others.
const UNSUPPORTED_BUILTIN_TARGETS: &[&str] = &[
    ".DEFAULT",
    ".PRECIOUS",
    ".INTERMEDIATE",
    ".SECONDARY",
    ".SECONDEXPANSION",
    ".IGNORE",
    ".LOW_RESOLUTION_TIME",
    ".SILENT",
    ".EXPORT_ALL_VARIABLES",
    ".NOTPARALLEL",
    ".ONESHELL",
];

/// Whether this target addresses Make itself rather than naming something to
/// build, and has therefore already been read for whatever it had to say.
///
/// A closed list, because being a directive is not a property of the name's
/// shape: `.1` and `.WAIT` look exactly like `.PHONY` and are not it.
pub fn is_directive_target(names: &impl Interner, output: &Symbol) -> bool {
    let s = output.as_bytes(names);
    CONSUMED_BUILTIN_TARGETS
        .iter()
        .chain(UNSUPPORTED_BUILTIN_TARGETS)
        .any(|name| name.as_bytes() == &s[..])
}

/// Whether this node names something to build, as opposed to something the
/// Makefile said to the evaluator and that the evaluator has already answered.
///
/// The suffix rules are here rather than under [`is_directive_target`] because
/// they are not directives: `.c.o` names a rule for making any `.o` from any
/// `.c`, and GNU Make will additionally build a file of that name if a Makefile
/// asks for one. We do not, and refusing is what we have always done — but the
/// refusal has to stay explicit, because the alternative is worse than being
/// wrong. Emitted as an ordinary node, `.c.o` is claimed by the built-in
/// `%.o: %.c` rule with an empty stem and runs `cc -c -o .c.o` against no input.
pub fn is_buildable_target(names: &impl Interner, output: &Symbol) -> bool {
    !is_directive_target(names, output) && !is_suffix_rule(names, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_suffix_rule() {
        let mut session = Session::new();
        let co = session.intern(".c.o");
        let foo = session.intern("foo");
        let dotco = session.intern(".co");
        let cob = session.intern(".c.o.b");
        assert!(is_suffix_rule(&session, &co));
        assert!(!is_suffix_rule(&session, &foo));
        assert!(!is_suffix_rule(&session, &dotco));
        assert!(!is_suffix_rule(&session, &cob));
    }

    #[test]
    fn a_dot_named_target_is_something_to_build() {
        let mut session = Session::new();
        // The names GNU Make's own test suite builds and we used to discard:
        // an empty static-pattern stem leaves `.1`, and `.WAIT` is written by
        // Makefiles that declare it themselves for the benefit of older makes.
        for name in [".1", ".WAIT", ".x", "foo", ".."] {
            let sym = session.intern(name);
            assert!(
                is_buildable_target(&session, &sym),
                "{name} should be built"
            );
        }
        // The directives are answered by the evaluator and never reach a sink,
        // and a suffix rule is a rule rather than a file.
        for name in [".PHONY", ".SUFFIXES", ".KATI_RESTAT", ".ONESHELL", ".c.o"] {
            let sym = session.intern(name);
            assert!(
                !is_buildable_target(&session, &sym),
                "{name} should not be built"
            );
        }
    }
}
