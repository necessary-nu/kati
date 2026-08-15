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
use memchr::{memchr, memrchr};
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
        substitute_stem, trim_leading_curdir, word_scanner,
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

/// Which of the two variable sets GNU Make gives a file a binding landed in.
///
/// `initialize_file_variables` (GNU `src/variable.c`) chains a file's own
/// target-specific set in front of one set holding every matching pattern's
/// variables. The distinction is not decoration: each is a hash table, so a
/// name bound twice inside one of them keeps only the last binding, and it is
/// that binding's `private` that decides whether a prerequisite may read the
/// name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum RuleScopeKind {
    /// GNU's `file->pat_variables`: every matching pattern, accumulated.
    Pattern,
    /// GNU's `file->variables`: what the target's own name bound.
    Own,
}

/// The scopes that apply to one target, kept in GNU Make's two sets rather
/// than flattened, so `private` can be answered per set.
#[derive(Debug, Default)]
struct RuleScopes {
    /// Every matching pattern's scope, weakest first.
    patterns: Vec<Arc<Vars>>,
    /// The target's own target-specific scope, strongest of all.
    own: Option<Arc<Vars>>,
}

impl RuleScopes {
    fn is_empty(&self) -> bool {
        self.patterns.is_empty() && self.own.is_none()
    }

    /// Weakest first, which is the order they must be installed in.
    fn iter(&self) -> impl Iterator<Item = (RuleScopeKind, &Arc<Vars>)> {
        self.patterns
            .iter()
            .map(|vars| (RuleScopeKind::Pattern, vars))
            .chain(self.own.iter().map(|vars| (RuleScopeKind::Own, vars)))
    }
}

/// One rule variable installed into the rule scope, kept with everything it
/// would take to install it again once the whole run has been unwound.
struct RuleBinding {
    guard: ScopedVar,
    kind: RuleScopeKind,
    sym: Symbol,
    var: Var,
    private: bool,
}

/// Unwind a target's rule bindings down to what its prerequisites may read.
///
/// GNU Make decides this in `lookup_variable` (`src/variable.c`): a
/// prerequisite reaches its parent's sets across a link marked
/// `next_is_parent`, and from there on any binding it finds carrying
/// `private_var` is stepped over instead of returned. The walk is still a walk
/// — it stops at the first set holding the name with the flag clear — so the
/// target's own set is asked first and the pattern set only if the own set
/// either lacks the name or hid it behind `private`.
///
/// Ronin installs both sets into one scope, so the run is unwound whole and
/// only the binding that survives GNU's walk is laid down again. Dropping the
/// private guards where they lie would be a different rule and a wrong one
/// twice over: it restores whatever each private binding happened to shadow
/// rather than what is outermost, and it lets an earlier public binding show
/// through a later private one that overwrote it inside the same set, where
/// GNU has kept only the last.
fn release_private(bindings: Vec<RuleBinding>, session: &Session) -> Vec<ScopedVar> {
    // What each set is left holding for a name: one entry, the last written,
    // exactly as one hash table slot is.
    let mut surviving: HashMap<(Symbol, RuleScopeKind), (Var, bool)> = HashMap::new();
    let mut scope = None;
    for binding in &bindings {
        surviving.insert(
            (binding.sym, binding.kind),
            (binding.var.clone(), binding.private),
        );
        scope.get_or_insert_with(|| binding.guard.scope());
    }
    let Some(scope) = scope else {
        return Vec::new();
    };

    // Every name the run bound, in GNU's walk order per name.
    let mut public = Vec::new();
    let mut names = surviving.keys().map(|(sym, _)| *sym).collect::<Vec<_>>();
    names.sort_by_cached_key(|sym| sym.as_bytes(session));
    names.dedup();
    for sym in names {
        for kind in [RuleScopeKind::Own, RuleScopeKind::Pattern] {
            let Some((var, private)) = surviving.get(&(sym, kind)) else {
                continue;
            };
            // A private binding is stepped over, not stopped at: GNU's loop
            // only returns where the flag is clear and otherwise carries on to
            // the next set, so a target's own `private` defers to a matching
            // pattern's public binding rather than hiding the name outright.
            if !private {
                public.push((sym, var.clone()));
                break;
            }
        }
    }

    unbind(bindings.into_iter().map(|binding| binding.guard).collect());
    public
        .into_iter()
        .map(|(sym, var)| ScopedVar::new(scope.clone(), sym, var))
        .collect()
}

/// Hold a group of scope bindings for as long as the guard lives.
///
/// A bare `Vec<ScopedVar>` would drop front to back, which [`unbind`] explains
/// is the wrong order; this routes the guard's own drop through it.
struct Unbind(Vec<ScopedVar>);

impl Drop for Unbind {
    fn drop(&mut self) {
        unbind(std::mem::take(&mut self.0));
    }
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

/// What dependency analysis made of one read.
pub struct Plan {
    /// The roots of the dependency graph, in the order the targets asked for
    /// them.
    pub nodes: Vec<NamedDepNode>,
    /// The Makefiles the read consulted that a rule says how to remake, in the
    /// order the read reached them.
    pub regenerations: Vec<RegenerationRoot>,
    /// A required Makefile nothing can make, which ends the run once the
    /// Makefiles ahead of it have been brought up to date.
    pub refusal: Option<anyhow::Error>,
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

/// The directories `library_search` looks in once the working directory and
/// the `vpath` search have both come up empty, in the order it looks.
///
/// GNU Make's `dirs[]` in `remake.c`: `/lib`, `/usr/lib`, and then whatever
/// `--libdir` the build was configured with, which for a default prefix is
/// `/usr/local/lib`. Compiled in there and compiled in here for the same
/// reason — it is where libraries are, not something a makefile chooses.
const SYSTEM_LIBRARY_DIRECTORIES: &[&str] = &["/lib", "/usr/lib", "/usr/local/lib"];

/// Where in the search order an answer came from, so answers found for
/// different `.LIBPATTERNS` elements can be weighed against each other.
///
/// The derived order is the semantics: every `vpath` answer outranks every
/// system-directory one, which is what GNU Make arranges by starting the
/// system directories' indices above every `vpath` index there could be.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum LibraryRank {
    Vpath(VpathRank),
    System(usize),
}

/// Which `vpath` entry answered, and which of its directories.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct VpathRank {
    entry: usize,
    directory: usize,
}

/// Which of GNU Make's passes over the pattern rules is running.
///
/// `pattern_search` walks the same candidates up to four times, relaxing one
/// question about a missing prerequisite each time. `chaining` is its
/// `intermed_ok`: whether a prerequisite may be invented by a search of its
/// own. `compat` is its `allow_compat_rules`: whether a prerequisite the
/// Makefile merely writes down somewhere may be taken on trust without being
/// found or made at all.
///
/// The passes are ordered rather than combined because the order is the
/// answer: a rule whose prerequisites are really there beats one whose would
/// have to be invented, and both beat one that is only reached by taking a
/// name on trust — however far up the candidate list the loser sits.
#[derive(Clone, Copy)]
struct SearchPass {
    chaining: bool,
    compat: bool,
}

impl SearchPass {
    /// Every pass, in the order GNU Make runs them.
    fn all() -> [Self; 4] {
        [
            Self {
                chaining: false,
                compat: false,
            },
            Self {
                chaining: true,
                compat: false,
            },
            Self {
                chaining: false,
                compat: true,
            },
            Self {
                chaining: true,
                compat: true,
            },
        ]
    }
}

/// How a candidate rule's prerequisites were had, for a rule that applies.
///
/// GNU Make keeps the same division in one array: a `patdeps` carrying a file
/// is a name the search made up to complete the chain, and one without is a
/// name it found already there or took on trust. Both halves are acted on
/// after the rule is chosen and neither means anything before it.
#[derive(Default)]
struct ReachedPrerequisites {
    /// Names the search made up, so the Makefile never said them.
    invented: Vec<Symbol>,
    /// Names the search was given rather than making them.
    found: Vec<Symbol>,
}

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
    /// Whether [`Self::implicit_outputs`] are members of a pattern rule's
    /// group rather than of an explicit one.
    ///
    /// The distinction is whose decision the run was. An explicit `&:` rule is
    /// one decision over all its targets, so reaching any of them runs the
    /// recipe for the one that was reached. A pattern rule spelling several
    /// target patterns is one recipe over targets that each decide for
    /// themselves, so the recipe runs for whichever of them turned out to need
    /// making — a name that can only be known once the build compares them.
    pub pattern_group: bool,
    pub actual_inputs: Vec<Symbol>,
    pub actual_order_only_inputs: Vec<Symbol>,
    pub actual_validations: Vec<Symbol>,
    pub rule_vars: Option<Arc<Vars>>,
    pub depfile_var: Option<Var>,
    pub ninja_pool_var: Option<Var>,
    pub tags_var: Option<Var>,
    pub output_pattern: Option<Symbol>,
    /// What `%` stood for, as the implicit search read it.
    ///
    /// GNU Make computes the stem inside `pattern_search` and hands it to
    /// `set_file_variables`, so `$*` reads back what the match found rather
    /// than a second reading of the pattern. The two part company whenever the
    /// search held a directory aside: `b%.x` matching `lib/bye.x` leaves
    /// `lib/ye`, which no substitution into the pattern can produce. Recorded
    /// only by that search; an explicit or static pattern rule leaves it None
    /// and `$*` is read off the pattern as before.
    pub stem: Option<Symbol>,
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
            pattern_group: false,
            actual_inputs: Vec::new(),
            actual_order_only_inputs: Vec::new(),
            actual_validations: Vec::new(),
            rule_vars: None,
            depfile_var: None,
            ninja_pool_var: None,
            tags_var: None,
            output_pattern: None,
            stem: None,
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

/// Rewrite a deferred prerequisite's `%` to a reference standing for the stem
/// ahead of the second expansion, the first one of each whitespace-separated
/// token as GNU Make does. Substituting the stem itself would expand it a third
/// time, which is wrong for a stem containing `$`.
///
/// Which reference depends on whether the search is holding a directory aside.
/// `$*` is the whole stem; `$(*F)` is what is left once the directory that is
/// about to go in front of this prerequisite is taken off. The returned flag
/// says whether any `%` was replaced, because a word that named no stem takes
/// no directory either — GNU Make's `add_dir` is set on the same branch that
/// writes the reference.
fn stem_references(text: &Bytes, hold_directory: bool) -> (Bytes, bool) {
    if memchr(b'%', text).is_none() {
        return (text.clone(), false);
    }
    let reference: &[u8] = if hold_directory { b"$(*F)" } else { b"$*" };
    let mut ret = BytesMut::with_capacity(text.len() + 8);
    let mut substituted = false;
    let mut any = false;
    for &c in text.iter() {
        match c {
            b'%' if !substituted => {
                ret.put_slice(reference);
                substituted = true;
                any = true;
            }
            _ => {
                if c.is_ascii_whitespace() {
                    substituted = false;
                }
                ret.put_u8(c);
            }
        }
    }
    (ret.freeze(), any)
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
    // The implicit search fills a pattern rule's `%` in for the name it matched
    // and hands back finished names. A `%` still standing in one of those is a
    // literal the rule asked for, not a second placeholder.
    if r.prerequisites_are_resolved {
        ret.extend(inputs);
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

/// How an implicit rule's target pattern matched the name being made.
///
/// GNU Make's `pattern_search` will not match a pattern against a name carrying
/// a directory when the pattern carries none of its own. It matches the file
/// part alone and holds the directory aside, under the flag it calls
/// `check_lastslash`, which is set from the target pattern and the target name
/// and takes no notice of the prerequisites. The directory comes back in
/// exactly two places — in front of every prerequisite the rule fills a `%`
/// into, and in front of the stem `$*` reads — so it is kept beside the stem
/// here rather than folded into it.
///
/// The split decides which rules apply and not only what they read: `l%.x` does
/// not match `lib/bye.x`, because `l` has to match the start of `bye.x`, while
/// `b%.x` does, leaving a stem of `lib/ye`.
///
/// Only the implicit search works this way. A static pattern rule substitutes
/// the whole stem where the `%` stands however many directories it names, which
/// is why this is not a property of [`Pattern`] itself.
#[derive(Clone)]
struct PatternMatch {
    /// The directory the matched name carried, empty when the pattern carried
    /// one of its own or the name had none.
    directory: Bytes,
    /// What `%` stood for, with `directory` already taken off the front.
    stem: Bytes,
}

impl PatternMatch {
    /// How `pattern` matches `output`, or `None` when it does not.
    fn of(pattern: &Pattern, output: &Bytes) -> Option<Self> {
        let path_len = directory_length(output);
        let hold_directory = path_len > 0 && !pattern.as_bytes().contains(&b'/');
        let matched = if hold_directory {
            output.slice(path_len..)
        } else {
            output.clone()
        };
        if !pattern.matches(&matched) {
            return None;
        }
        Some(Self {
            directory: if hold_directory {
                output.slice(..path_len)
            } else {
                Bytes::new()
            },
            stem: matched.slice_ref(pattern.stem(&matched)),
        })
    }

    /// The whole of what `%` stood for: what `$*` reads, and what the search
    /// measures a candidate's specificity by.
    fn whole_stem(&self) -> Bytes {
        if self.directory.is_empty() {
            return self.stem.clone();
        }
        let mut ret = BytesMut::with_capacity(self.directory.len() + self.stem.len());
        ret.put_slice(&self.directory);
        ret.put_slice(&self.stem);
        ret.freeze()
    }

    /// The name one prerequisite of the matched rule stands for.
    fn prerequisite(&self, prerequisite: &Bytes) -> Bytes {
        substitute_stem(prerequisite, &self.directory, &self.stem)
    }
}

/// How much of `name` is the directory it sits in, including the slash.
///
/// A trailing slash belongs to the last directory's own name rather than
/// separating it from anything, which is why GNU Make looks for the last slash
/// in all but the final byte: `foo/bar/` is `bar/` in `foo/`, not nothing in
/// `foo/bar/`.
fn directory_length(name: &[u8]) -> usize {
    memrchr(b'/', &name[..name.len().saturating_sub(1)]).map_or(0, |at| at + 1)
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

/// A suffix without the dot it is written with, which is how the map of suffix
/// rules is keyed.
fn undotted(suffix: &Bytes) -> Bytes {
    suffix.slice(usize::from(suffix.starts_with(b"."))..)
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
    /// The pattern rules a search further out is already working through.
    ///
    /// GNU Make marks a rule `in_use` while it decides whether the
    /// prerequisites that rule would need can be had, and `pattern_search`
    /// passes over a marked rule: no rule may be a link in the chain that
    /// supplies its own prerequisite. With a catalogue of rules that chain into
    /// one another this is what bounds the search rather than merely tidying
    /// it, and it is the "Avoiding implicit rule recursion" its `-d` reports.
    rules_in_use: HashSet<usize>,
    /// Whether the search just run passed over a rule for a prerequisite the
    /// Makefile writes down somewhere.
    ///
    /// GNU Make's `found_compat_rule`, and it is a local of `pattern_search`
    /// rather than a fact about the graph: each search saves the outer value,
    /// clears it, and puts it back. Only a search that set it is retried taking
    /// written-down names on trust — which matters for more than fidelity,
    /// since with a catalogue of chaining rules an unconditional retry doubles
    /// the work at every level of a search that is already six deep.
    found_compat_rule: bool,
    /// Names a chain search has already proven nothing can make.
    ///
    /// GNU Make's `file_impossible`, and it is what keeps the search from
    /// being exponential: every rule in the catalogue proposes a prerequisite,
    /// each of those proposes its own, and without a memory of the failures the
    /// same subtree is walked once per road into it. A name the Makefile writes
    /// down is never marked, because the compatibility pass has to be free to
    /// take it on trust later.
    ///
    /// Recorded against the shallowest depth the failure was reached at, which
    /// GNU Make has no need of: its search has no depth limit, so a failure is
    /// a failure. Here a name that ran out of budget at depth five says nothing
    /// about the same name at depth one, and the recorded depth is what tells
    /// the two apart — a later search may reuse the answer only from at least
    /// as deep, where it has no more budget than the search that proved it.
    impossible: HashMap<Symbol, usize>,
    /// Names a terminal rule was handed, which no implicit search may make.
    ///
    /// GNU Make's `tried_implicit`, set where a chosen terminal rule takes a
    /// prerequisite it did not invent, and read where `remake.c` decides
    /// whether a target with no recipe is worth searching for one. Terminal is
    /// the whole claim: the rule applies to what is there, so the name it was
    /// given has to be there rather than be arrived at.
    tried_implicit: HashSet<Symbol>,
    /// Whether the last chain search stopped because the cycle guard cut it
    /// short rather than because the rules ran out.
    ///
    /// A failure reached that way is about the road taken to the name, not
    /// about the name, so it must not be remembered at any depth.
    chain_truncated: bool,
    /// Whether each name asked about is there, asked once.
    ///
    /// The search asks the same question thousands of times over — every rule
    /// in the catalogue proposes a prerequisite, and the chain proposes each of
    /// theirs — and nothing between the first Makefile closing and the graph
    /// being finished can change the answer. GNU Make answers from `struct
    /// file` and a directory read it keeps; this is the same bargain.
    exists_cache: HashMap<Symbol, bool>,
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
    /// `.SUFFIXES` as the whole read left it, in the order it was written.
    ///
    /// GNU Make derives every suffix rule from this list once the last Makefile
    /// is closed, so it decides which rules exist rather than merely filtering
    /// them: a bare `.SUFFIXES:` withdraws the built-in catalogue's suffix half
    /// and a later line puts back whichever pairs it names. The entries keep
    /// their leading dot, because that is how the pair's name is spelled.
    suffixes: Vec<Bytes>,
    /// The name the invocation's own preamble is read under, which is how a
    /// `.SUFFIXES` Make wrote is told apart from one a Makefile wrote.
    bootstrap_filename: Symbol,
    extra_prereqs_var_name: Symbol,
    /// What a global `.EXTRA_PREREQS` adds to every target of the read, as the
    /// compared and order-only halves a `|` in the value separates.
    ///
    /// Expanded once the last Makefile is closed, because that is when GNU Make
    /// reads it — so a value written in terms of a variable defined further down
    /// the file still names what that variable finally held.
    global_extra_prereqs: (Vec<Symbol>, Vec<Symbol>),

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
    /// `.LIBPATTERNS`, which says how a `-lNAME` prerequisite is spelt on disk.
    libpatterns_var_name: Symbol,
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
    vars: RuleScopes,
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
        let libpatterns_var_name = ev.session.intern(".LIBPATTERNS");
        let gpath_var_name = ev.session.intern("GPATH");
        let implicit_outputs_var_name = ev.session.intern(".KATI_IMPLICIT_OUTPUTS");
        let ninja_pool_var_name = ev.session.intern(".KATI_NINJA_POOL");
        let validations_var_name = ev.session.intern(".KATI_VALIDATIONS");
        let tags_var_name = ev.session.intern(".KATI_TAGS");
        let wait_sym = ev.session.intern(".WAIT");
        let bootstrap_filename = ev.session.intern("*bootstrap*");
        let extra_prereqs_var_name = ev.session.intern(".EXTRA_PREREQS");
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
            rules_in_use: HashSet::new(),
            found_compat_rule: false,
            impossible: HashMap::new(),
            tried_implicit: HashSet::new(),
            chain_truncated: false,
            exists_cache: HashMap::new(),
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
            suffixes: Vec::new(),
            bootstrap_filename,
            extra_prereqs_var_name,
            global_extra_prereqs: (Vec::new(), Vec::new()),

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
            libpatterns_var_name,
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
        ret.install_builtin_rules()?;
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
        self.read_suffix_list()?;
        let global = self.ev.eval_var(self.extra_prereqs_var_name)?;
        self.global_extra_prereqs = self.prerequisite_names(&global);

        Ok(())
    }

    /// One expanded prerequisite list, read into the names it declares and
    /// split at the `|` that makes the rest order-only.
    ///
    /// GNU Make's `split_prereqs`, which is what reads this value: the same
    /// reading a rule's own prerequisites get, so a leading `./` comes off, a
    /// word holding a wildcard is matched where it was written, and a `|`
    /// separates what must merely exist first from what is compared.
    fn prerequisite_names(&mut self, text: &Bytes) -> (Vec<Symbol>, Vec<Symbol>) {
        let (compared, order_only) = split_order_only(text);
        let mut names = (Vec::new(), Vec::new());
        for (half, into) in [(compared, &mut names.0), (order_only, &mut names.1)] {
            for word in makefile_word_scanner(&half) {
                let word = word.slice_ref(trim_leading_curdir(&word));
                glob_word(&mut self.ev.session, word, into);
            }
        }
        names
    }

    /// What `.EXTRA_PREREQS` adds to `output`: prerequisites built and compared
    /// like any other, which appear in no automatic variable.
    ///
    /// GNU Make's `snap_file`. A target that has target-specific variables of
    /// its own reads the value out of that set alone — so a target-specific
    /// `.EXTRA_PREREQS` replaces the global one rather than adding to it, and,
    /// because the lookup does not fall back, a target carrying *any* other
    /// target-specific variable is left with no extra prerequisites at all.
    /// That is GNU 4.4.1's behaviour rather than its documentation, and it is
    /// what a Makefile written against it depends on.
    ///
    /// The global list reaches only names the read made targets: a file that is
    /// merely a prerequisite of something is not given prerequisites of its own.
    ///
    /// A list that names the target itself would be a cycle, and GNU drops the
    /// whole list rather than the one entry that closed it.
    ///
    /// The answer is the two halves a `|` in the value separates: compared, and
    /// order-only.
    fn extra_prerequisites(
        &mut self,
        output: Symbol,
        is_target: bool,
    ) -> Result<(Vec<Symbol>, Vec<Symbol>)> {
        let extras = if let Some(vars) = self.lookup_rule_vars(output) {
            let Some(var) = vars.lookup(
                &mut self.ev.session.used_env_vars,
                self.extra_prereqs_var_name,
            ) else {
                return Ok((Vec::new(), Vec::new()));
            };
            let text = var.read().eval_to_buf(self.ev)?;
            self.prerequisite_names(&text)
        } else if is_target {
            self.global_extra_prereqs.clone()
        } else {
            return Ok((Vec::new(), Vec::new()));
        };
        if extras.0.contains(&output) || extras.1.contains(&output) {
            return Ok((Vec::new(), Vec::new()));
        }
        Ok(extras)
    }

    /// Settle `.SUFFIXES` as the read left it.
    ///
    /// In order, because `.SUFFIXES:` clears the list and a later one adds to
    /// what is left. Merging them first loses the clear.
    fn read_suffix_list(&mut self) -> Result<()> {
        let suffixes = self.ev.session.intern(".SUFFIXES");
        let mut declared: Vec<Symbol> = Vec::new();
        let mut written_by_makefile = false;
        if let Some(merger) = self.rules.get(&suffixes).cloned() {
            let rules = merger.lock().rules.clone();
            for rule in &rules {
                written_by_makefile |= rule.loc.filename != self.bootstrap_filename;
                let mut inputs = rule.inputs.clone();
                inputs.extend(self.declared_by(suffixes, rule)?);
                if inputs.is_empty() {
                    declared.clear();
                } else {
                    declared.extend(inputs);
                }
            }
        }
        // A `-r` that only a Makefile's own `MAKEFLAGS` asked for arrives after
        // the list is already there, so GNU Make takes the list away at the end
        // of the read — but only while it is still the list Make itself set.
        // Naming `.SUFFIXES` at all, even to add to it, makes the list the
        // Makefile's, and a Makefile's list survives its own `-r`.
        if self.ev.session.flags.no_builtin_rules && !written_by_makefile {
            declared.clear();
        }
        self.suffixes = declared
            .iter()
            .map(|suffix| suffix.as_bytes(&self.ev.session))
            .collect();
        if declared.is_empty() {
            self.suffix_rules.clear();
        } else {
            self.keep_only_declared_suffix_rules(&declared);
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
            .map(|s| undotted(&s.as_bytes(&self.ev.session)))
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

    /// Turn `.SUFFIXES` into rules, and add the rules that were always
    /// patterns.
    ///
    /// GNU Make's `convert_to_pattern` followed by
    /// `install_default_implicit_rules`, in their place: after the last
    /// Makefile has closed. Doing it here rather than before the read is what
    /// makes a Makefile's own `%.out: %` outrank the built-in of that name
    /// instead of colliding with it, and it is why a `-r` that only a
    /// Makefile's `MAKEFLAGS` asked for still has the whole catalogue to
    /// withhold.
    fn install_builtin_rules(&mut self) -> Result<()> {
        let withheld = self.ev.session.flags.no_builtin_rules;
        for source in self.suffixes.clone() {
            self.install_suffix_disqualifier(&source)?;
            self.install_null_suffix_rule(&source, withheld)?;
        }
        if !withheld {
            self.install_builtin_suffix_pairs()?;
        }
        // A written pattern rule keeps any name a suffix rule would have taken,
        // and the built-in pairs have only just arrived, so the comparison runs
        // again now that they are all known.
        self.discard_suffix_rules_a_pattern_rule_holds();
        self.order_suffix_rules_by_suffix_list();
        if !withheld {
            self.install_default_pattern_rules()?;
        }
        Ok(())
    }

    /// The rule a suffix has by being a suffix: `%.c:`, with no prerequisites
    /// and no recipe.
    ///
    /// It can make nothing and is never used to. Its whole effect is that a
    /// name ending in a declared suffix has now been matched by a rule that is
    /// not a bare `%`, which withdraws every match-anything rule from that
    /// search — see [`DepBuilder::ordered_candidates`]. It is not withheld by
    /// `-r`, because it comes from the suffix list rather than from the
    /// catalogue: a Makefile that declares a suffix under `-r` gets it.
    fn install_suffix_disqualifier(&mut self, suffix: &Bytes) -> Result<()> {
        let loc = crate::builtin_rules::builtin_loc(&mut self.ev.session);
        let mut rule = Rule::new(loc, false, false);
        rule.output_patterns.push(self.suffix_pattern(suffix));
        self.populate_implicit_rule(Arc::new(rule), false)
    }

    /// The rule that makes a program out of one source file: `%: %.c`.
    ///
    /// GNU Make writes it as a suffix rule with one suffix, and it is the half
    /// of the catalogue `all: hello` beside `hello.c` rests on. A Makefile's
    /// own `.c:` recipe is converted the same way and is not withheld by `-r`,
    /// which withholds only the recipes the catalogue supplied.
    fn install_null_suffix_rule(&mut self, suffix: &Bytes, withheld: bool) -> Result<()> {
        let name = self.ev.session.intern(suffix.clone());
        let written = self
            .lookup_rule_merger(name)
            .and_then(|merger| merger.lock().primary_rule.clone());
        let (loc, cmd_loc, cmds) = match written {
            Some(rule) => (rule.loc.clone(), rule.cmd_loc.clone(), rule.cmds.clone()),
            None if withheld => return Ok(()),
            None => {
                let Some(recipe) = crate::builtin_rules::suffix_recipe(suffix) else {
                    return Ok(());
                };
                let loc = crate::builtin_rules::builtin_loc(&mut self.ev.session);
                let cmds = crate::builtin_rules::recipe_lines(&mut self.ev.session, recipe)?;
                (loc, None, cmds)
            }
        };
        let mut rule = Rule::new(loc, false, false);
        rule.cmd_loc = cmd_loc;
        rule.output_patterns.push(self.ev.session.intern("%"));
        let input = self.suffix_pattern(suffix);
        rule.inputs.push(input);
        rule.prerequisite_names.push(input);
        rule.cmds = cmds;
        self.populate_implicit_rule(Arc::new(rule), false)
    }

    /// The catalogue's compile rules, for whichever pairs the suffix list
    /// activates: `.c.o` becomes `%.o: %.c`.
    ///
    /// GNU Make writes every declared suffix against every other and looks each
    /// name up as a file. The same set is reached by reading the table and
    /// asking whether both halves of each name are declared, which is the same
    /// question with eleven hundred names that were never there left unasked.
    /// Which pair wins for a target is settled afterwards by
    /// [`DepBuilder::order_suffix_rules_by_suffix_list`], so the order this
    /// walks in is not the order they are tried in.
    ///
    /// A pair a Makefile gave a recipe to is already converted and is left
    /// alone: the catalogue's recipe is only ever written onto a name that has
    /// none.
    fn install_builtin_suffix_pairs(&mut self) -> Result<()> {
        for (name, recipe) in crate::builtin_rules::DEFAULT_SUFFIX_RULES {
            let Some((source, target)) = self.declared_suffix_pair(name) else {
                continue;
            };
            let written = self.ev.session.intern(name.as_bytes().to_vec());
            if self
                .lookup_rule_merger(written)
                .is_some_and(|merger| merger.lock().primary_rule.is_some())
            {
                continue;
            }
            let loc = crate::builtin_rules::builtin_loc(&mut self.ev.session);
            let mut rule = Rule::new(loc, false, false);
            rule.outputs.push(written);
            rule.output_patterns.push(self.suffix_pattern(&target));
            let input = self.ev.session.intern(undotted(&source));
            rule.inputs.push(input);
            rule.prerequisite_names.push(input);
            rule.is_suffix_rule = true;
            rule.cmds = crate::builtin_rules::recipe_lines(&mut self.ev.session, recipe)?;
            self.suffix_rules
                .entry(undotted(&target))
                .or_default()
                .push(Arc::new(rule));
        }
        Ok(())
    }

    /// A suffix-rule name read as the two declared suffixes it is written from,
    /// if both of them are on the list.
    fn declared_suffix_pair(&self, name: &str) -> Option<(Bytes, Bytes)> {
        for source in &self.suffixes {
            let Some(rest) = name.as_bytes().strip_prefix(source.as_ref()) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if let Some(target) = self.suffixes.iter().find(|target| target.as_ref() == rest) {
                return Some((source.clone(), target.clone()));
            }
        }
        None
    }

    /// Put the rules that make one suffix into the order GNU Make tries them.
    ///
    /// `convert_to_pattern` walks `.SUFFIXES` in its written order and installs
    /// each rule it derives at the end of one chain, so which source a target
    /// is made from when several are there is decided by where that source sits
    /// on the list — `.c` before `.cc` is why `foo.o` comes from `foo.c`. The
    /// sort is stable, so two rules with the same source suffix keep the order
    /// population left them in, which is the later definition first.
    fn order_suffix_rules_by_suffix_list(&mut self) {
        let order = self.suffixes.iter().map(undotted).collect::<Vec<_>>();
        let names = &self.ev.session;
        for rules in self.suffix_rules.values_mut() {
            rules.sort_by_key(|rule| {
                let source = rule.inputs[0].as_bytes(names);
                order
                    .iter()
                    .position(|suffix| *suffix == source)
                    .unwrap_or(usize::MAX)
            });
        }
    }

    /// The rules `default.c` writes as patterns rather than as suffixes, and
    /// then the terminal ones.
    ///
    /// Installed last, and each only if no rule of that identity is already
    /// there, which is `new_pattern_rule`'s override left off: a Makefile that
    /// wrote `%.out: %` — with a recipe or without one — keeps the name.
    fn install_default_pattern_rules(&mut self) -> Result<()> {
        for (terminal, table) in [
            (false, crate::builtin_rules::DEFAULT_PATTERN_RULES),
            (true, crate::builtin_rules::DEFAULT_TERMINAL_RULES),
        ] {
            for (target, prerequisites, recipe) in table {
                let rule = crate::builtin_rules::pattern_rule(
                    &mut self.ev.session,
                    target,
                    prerequisites,
                    recipe,
                    terminal,
                )?;
                self.populate_implicit_rule(Arc::new(rule), false)?;
            }
        }
        Ok(())
    }

    /// `.c` written as the pattern that matches a name ending in it.
    fn suffix_pattern(&mut self, suffix: &Bytes) -> Symbol {
        let mut pattern = BytesMut::with_capacity(suffix.len() + 1);
        pattern.put_u8(b'%');
        pattern.put_slice(suffix);
        self.ev.session.intern(pattern.freeze())
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
        targets: Vec<Symbol>,
        read_makefiles: &[ReadMakefile],
        missing_includes: &[MissingInclude],
    ) -> Result<Plan> {
        // Generated included Makefiles are compiler inputs rather than user
        // goals, and GNU Make remakes them before it picks a goal at all. Both
        // halves of that matter here: asking the graph for one must not change
        // what the Makefile builds once it is reread, and a required include
        // with no rule is the failure the run dies on, ahead of the complaint
        // about having nothing to aim at.
        let (regeneration_nodes, refusal) =
            self.plan_regeneration(read_makefiles, missing_includes)?;
        let planned = self.plan_goals(targets);
        // The refusal happens in GNU Make while the makefiles are being
        // remade, which is before it has looked at a goal at all — so a
        // makefile that also leaves the goals unanalysable is refused over,
        // not reported on.
        let nodes = match planned {
            Ok(nodes) => nodes,
            Err(error) if refusal.is_none() => return Err(error),
            Err(_) => Vec::new(),
        };
        self.drop_circular_dependencies(&regeneration_nodes, &nodes);
        Ok(Plan {
            nodes,
            regenerations: regeneration_nodes,
            refusal,
        })
    }

    /// Plan the goals this invocation was aimed at, choosing the default when
    /// it named none.
    fn plan_goals(&mut self, mut targets: Vec<Symbol>) -> Result<Vec<NamedDepNode>> {
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
        Ok(nodes)
    }

    /// Unlink the prerequisites that close a loop, the way GNU Make does,
    /// rather than refusing the build.
    ///
    /// `update_file_1` marks a target updating while it walks that target's
    /// prerequisites, and a prerequisite already marked is one the walk cannot
    /// follow. GNU Make says `Circular %s <- %s dependency dropped.` and takes
    /// that single entry out of the list it is walking (`remake.c`), so the
    /// update carries on and the build succeeds. The entry is gone for good:
    /// the list it left is the one `$^`, `$+` and `$?` are read from, so the
    /// target whose prerequisite was dropped never sees it again.
    ///
    /// Which entry goes is therefore a question about the order of the walk,
    /// and this is that walk: the Makefiles the read has to remake first, in
    /// the order it reached them, then the goals in the order they were asked
    /// for, and each target's prerequisites in the order they were written.
    /// Entering a target that is already finished is not a loop, so a diamond
    /// is left alone; only an ancestor of the edge being followed is.
    ///
    /// A frontend that reads the plan afterwards receives a graph with no
    /// cycles in it, which is what lets its own refusal go on meaning what it
    /// says for a graph nobody compiled from a Makefile.
    fn drop_circular_dependencies(
        &mut self,
        regenerations: &[RegenerationRoot],
        goals: &[NamedDepNode],
    ) {
        enum Step {
            /// Follow this edge, dropping it if it closes a loop. The edge is
            /// named by the node it leaves and which of that node's two lists
            /// it is on, because that is the list the drop takes it out of.
            Enter {
                from: Option<(Arc<Mutex<DepNode>>, Prerequisites)>,
                node: Arc<Mutex<DepNode>>,
            },
            /// This target's prerequisites have all been walked, so it is no
            /// longer one a deeper edge could close a loop through.
            Leave(Arc<Mutex<DepNode>>),
        }

        let mut updating = HashSet::new();
        let mut updated = HashSet::new();
        let roots = regenerations
            .iter()
            .map(|root| &root.node)
            .chain(goals)
            .map(|(_, node)| Step::Enter {
                from: None,
                node: node.clone(),
            });
        let mut work: Vec<Step> = roots.collect();
        work.reverse();
        while let Some(step) = work.pop() {
            let (from, node) = match step {
                Step::Leave(node) => {
                    let id = identity(&node);
                    updating.remove(&id);
                    updated.insert(id);
                    continue;
                }
                Step::Enter { from, node } => (from, node),
            };
            let id = identity(&node);
            if updating.contains(&id) {
                if let Some((from, list)) = from {
                    let target = recipe_name(&self.ev.session, &from);
                    let dropped = recipe_name(&self.ev.session, &node);
                    warn_loc!(
                        self.ev,
                        None,
                        "Circular {target} <- {dropped} dependency dropped."
                    );
                    drop_prerequisite(&from, list, &node);
                }
                continue;
            }
            if updated.contains(&id) {
                continue;
            }
            updating.insert(id);
            work.push(Step::Leave(node.clone()));
            let held = node.lock();
            let edges = held
                .deps
                .iter()
                .map(|(_, dep)| (dep, Prerequisites::Compared))
                .chain(
                    held.order_onlys
                        .iter()
                        .map(|(_, dep)| (dep, Prerequisites::OrderOnly)),
                )
                .map(|(dep, list)| Step::Enter {
                    from: Some((node.clone(), list)),
                    node: dep.clone(),
                })
                .collect::<Vec<_>>();
            drop(held);
            work.extend(edges.into_iter().rev());
        }
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
    ///
    /// It dies where it reaches it, not where it read it. GNU Make walks the
    /// makefiles in the order it read them and brings each one up to date in
    /// turn, so the ones ahead of the refusal are remade and the ones behind it
    /// never are — `complain()` ends the run from inside `update_goal_chain`,
    /// before `main.c` can even ask whether a remade makefile means the read
    /// should start over. So the refusal is returned rather than raised: the
    /// roots collected before it are what the frontend has to build, and the
    /// refusal is what it raises once that is done.
    fn plan_regeneration(
        &mut self,
        read_makefiles: &[ReadMakefile],
        missing_includes: &[MissingInclude],
    ) -> Result<(Vec<RegenerationRoot>, Option<anyhow::Error>)> {
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
            let refusal = crate::color_error_log(
                &self.ev.session,
                None,
                format!("*** No rule to make target '{name}'."),
            );
            return Ok((nodes, Some(refusal)));
        }
        Ok((nodes, None))
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

    fn exists(&mut self, target: Symbol) -> bool {
        if let Some(answer) = self.exists_cache.get(&target) {
            return *answer;
        }
        let answer = self.rules.contains_key(&target)
            || self.phony.contains(&target)
            || std::fs::exists(OsStr::from_bytes(&target.as_bytes(&self.ev.session)))
                .is_ok_and(|v| v)
            || self.vpath_of(target).is_some();
        self.exists_cache.insert(target, answer);
        answer
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
        let (inputs, order_only) = {
            let n = n.lock();
            (n.actual_inputs.clone(), n.actual_order_only_inputs.clone())
        };
        // The `-lNAME` search is the last resort of the same search and runs
        // whether or not anything wrote a `vpath`, so the way out has to ask
        // about both before it takes it.
        if self.ev.session.vpaths.is_empty()
            && self.vpath_variable().is_empty()
            && !inputs
                .iter()
                .chain(&order_only)
                .any(|input| input.as_bytes(&self.ev.session).starts_with(b"-l"))
        {
            return;
        }
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
        // A static pattern rule's match holds no directory aside: `%` stands
        // for the whole of what it matched, directories and all.
        let matched = stem.map(|stem| PatternMatch {
            directory: Bytes::new(),
            stem,
        });
        self.expand_deferred_prerequisites(output, matched, prerequisites, vec![text.clone()])
    }

    /// An implicit pattern rule expands each raw prerequisite word
    /// independently. This keeps a backslash at the end of one raw word from
    /// quoting the blank before the next one, while still letting an expansion
    /// introduce an escaped blank inside its own result.
    fn expand_pattern_prerequisites_again(
        &mut self,
        output: Symbol,
        matched_at: PatternMatch,
        prerequisites: (&[Symbol], &[Symbol]),
        text: &Bytes,
    ) -> Result<(Vec<Symbol>, Vec<Symbol>)> {
        self.expand_deferred_prerequisites(
            output,
            Some(matched_at),
            prerequisites,
            implicit_prerequisite_words(text).collect(),
        )
    }

    /// Bind the `D` and `F` forms of one automatic variable, alongside a base
    /// form bound in the same scope.
    ///
    /// GNU Make never computes these. `define_automatic_variables` writes them
    /// once at startup as recursive variables holding a `dir` or `notdir`
    /// expression over the base form, so they answer for whatever that form
    /// holds at the moment they are read. Binding the same definitions here
    /// keeps that property: they follow the base binding rather than freezing a
    /// value taken beside it.
    fn bind_path_forms(&mut self, scope: &Arc<Vars>, name: char) -> Result<Vec<ScopedVar>> {
        let mut bound = Vec::with_capacity(2);
        for (form, text) in [
            ('D', format!("$(patsubst %/,%,$(dir ${name}))")),
            ('F', format!("$(notdir ${name})")),
        ] {
            let sym = self.ev.session.intern(format!("{name}{form}"));
            let text = Bytes::from(text);
            let mut loc = self.ev.loc.clone().unwrap_or_default();
            let value = crate::expr::parse_expr(
                &mut self.ev.session,
                &mut loc,
                text.clone(),
                crate::expr::ParseExprOpt::Normal,
            )?;
            bound.push(ScopedVar::new(
                scope.clone(),
                sym,
                Variable::new_recursive(value, crate::var::VarOrigin::Automatic, None, None, text),
            ));
        }
        Ok(bound)
    }

    fn expand_deferred_prerequisites(
        &mut self,
        output: Symbol,
        matched: Option<PatternMatch>,
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
        let directory = matched
            .as_ref()
            .map(|matched| matched.directory.clone())
            .unwrap_or_default();
        // Paired with each text: whether it named the stem, and so whether the
        // held-aside directory goes in front of what it expands to.
        let texts: Vec<(Bytes, bool)> = match &matched {
            Some(_) => texts
                .into_iter()
                .map(|text| stem_references(&text, !directory.is_empty()))
                .collect(),
            None => texts.into_iter().map(|text| (text, false)).collect(),
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
            let mut bound = Vec::new();
            let at_value = output.as_bytes(&self.ev.session);
            bound.push(ScopedVar::new(scope.clone(), at, automatic(at_value)));
            bound.extend(self.bind_path_forms(&scope, '@')?);
            if let Some(matched) = &matched {
                let stem = matched.whole_stem();
                bound.push(ScopedVar::new(scope.clone(), star, automatic(stem)));
                bound.extend(self.bind_path_forms(&scope, '*')?);
            }
            bound.push(ScopedVar::new(scope.clone(), less, automatic(first)));
            bound.extend(self.bind_path_forms(&scope, '<')?);
            bound.push(ScopedVar::new(scope.clone(), hat, automatic(hat_value)));
            bound.extend(self.bind_path_forms(&scope, '^')?);
            bound.push(ScopedVar::new(scope.clone(), plus, automatic(plus_value)));
            bound.extend(self.bind_path_forms(&scope, '+')?);
            // `$|` keeps no D or F form: GNU Make reads `$(|D)` as an ordinary
            // variable nobody defined, which is what the register site says too.
            bound.push(ScopedVar::new(scope, bar, automatic(bar_value)));
            let _bound = Unbind(bound);
            let mut expanded = Vec::with_capacity(texts.len());
            for (text, add_directory) in texts {
                let mut loc = self.ev.loc.clone().unwrap_or_default();
                let expr = crate::expr::parse_expr(
                    &mut self.ev.session,
                    &mut loc,
                    text,
                    crate::expr::ParseExprOpt::Normal,
                )?;
                expanded.push((expr.eval_to_buf(self.ev)?, add_directory));
            }
            expanded
        };

        let mut inputs = Vec::new();
        let mut order_only_inputs = Vec::new();
        let mut order_only = false;
        for (expanded_word, add_directory) in expanded {
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
                    if !add_directory {
                        glob_word(&mut self.ev.session, word, into);
                        continue;
                    }
                    // GNU Make hands the directory to `parse_file_seq` as a
                    // prefix, which puts it on each name the sequence yields —
                    // after any globbing rather than before it, so the pattern
                    // is matched where the rule was written and the answer is
                    // then read one directory down.
                    let mut named = Vec::new();
                    glob_word(&mut self.ev.session, word, &mut named);
                    for name in named {
                        let name = name.as_bytes(&self.ev.session);
                        let mut buf = BytesMut::with_capacity(directory.len() + name.len());
                        buf.put_slice(&directory);
                        buf.put_slice(&name);
                        into.push(self.ev.session.intern(buf.freeze()));
                    }
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
            None => self.at_library(name),
        }
    }

    /// One `-lNAME` prerequisite, replaced by the library it refers to.
    ///
    /// GNU Make reaches `library_search` from `f_mtime`, as the last resort
    /// after the ordinary directory search has failed — so `-lfoo` is a file
    /// name whenever a file of that name is there, and a linker-style library
    /// reference only when it is not.
    ///
    /// Deliberately not conditioned on the name having a rule, which is where
    /// this parts company with the `vpath` search above it: `f_mtime` asks
    /// about the file before anything asks about the rule, so a `-lfoo:` rule
    /// written beside a `libfoo.a` on disk does not stop the search. GNU Make
    /// renames the file to what the search found and finds it current there,
    /// and moving the prerequisite to the found name is the same answer — the
    /// recipe does not run and `$^` reads the library. A search that finds
    /// nothing leaves the name as written, and then the rule does make it.
    fn at_library(&mut self, name: Symbol) -> Symbol {
        let reference = name.as_bytes(&self.ev.session);
        if !reference.starts_with(b"-l") || self.phony.contains(&name) {
            return name;
        }
        if std::fs::exists(OsStr::from_bytes(&reference)).is_ok_and(|found| found) {
            return name;
        }
        match self.library_search(&reference) {
            Some(found) => self.ev.session.intern(found),
            None => name,
        }
    }

    /// Where the library `-lNAME` refers to actually is, if anywhere.
    ///
    /// GNU Make's `library_search`. Each whitespace-separated element of
    /// `.LIBPATTERNS` says how a library of that name might be spelt; the
    /// wildcard takes NAME, and an element with no wildcard is warned about and
    /// passed over rather than taken literally.
    ///
    /// Every element is tried rather than the first that hits, because the
    /// answer is the *earliest* one any of them reaches — the linker-compatible
    /// behaviour the comment in `remake.c` asks for. Earliest means: a hit in
    /// the working directory beats everything and ends the search where it
    /// stands; otherwise the earliest `vpath` entry wins, whichever element
    /// reached it; and the compiled-in system directories rank behind every
    /// `vpath` entry, in their own order. Ties go to the earlier element.
    ///
    /// Only files that already exist are found. A pattern naming a target the
    /// makefile could make is not a match, so `-lfoo` under
    /// `.LIBPATTERNS = made_%.a` beside a `made_foo.a:` rule is refused rather
    /// than built.
    fn library_search(&mut self, reference: &[u8]) -> Option<Bytes> {
        let name = &reference[b"-l".len()..];
        let mut best: Option<(LibraryRank, Bytes)> = None;
        for element in self.libpatterns() {
            let Some(candidate) = Pattern::new(element.clone()).substitute(name) else {
                warn_loc!(
                    self.ev,
                    None,
                    ".LIBPATTERNS element `{}' is not a pattern",
                    String::from_utf8_lossy(&element)
                );
                continue;
            };
            if std::fs::exists(OsStr::from_bytes(&candidate)).is_ok_and(|found| found) {
                return Some(candidate);
            }
            if let Some((found, rank)) = self.vpath_search(&candidate) {
                Self::keep_earlier(&mut best, LibraryRank::Vpath(rank), found);
            }
            for (index, directory) in SYSTEM_LIBRARY_DIRECTORIES.iter().enumerate() {
                let mut path = BytesMut::from(directory.as_bytes());
                path.put_u8(b'/');
                path.put_slice(&candidate);
                let path = path.freeze();
                if std::fs::exists(OsStr::from_bytes(&path)).is_ok_and(|found| found) {
                    Self::keep_earlier(&mut best, LibraryRank::System(index), path);
                }
            }
        }
        best.map(|(_, path)| path)
    }

    /// Record a candidate only when it beats the one already held, so an
    /// equally ranked answer from a later `.LIBPATTERNS` element loses.
    fn keep_earlier(best: &mut Option<(LibraryRank, Bytes)>, rank: LibraryRank, path: Bytes) {
        if best.as_ref().is_none_or(|(held, _)| rank < *held) {
            *best = Some((rank, path));
        }
    }

    /// `.LIBPATTERNS`, expanded and split into elements.
    ///
    /// Expanded here rather than read as text, because GNU Make expands it at
    /// search time: a recursive value follows an assignment made anywhere in
    /// the read, including after the rule that names the library.
    fn libpatterns(&mut self) -> Vec<Bytes> {
        let Some(var) = self.ev.session.peek_global_var(self.libpatterns_var_name) else {
            return Vec::new();
        };
        let Ok(value) = var.read().eval_to_buf(self.ev) else {
            return Vec::new();
        };
        word_scanner(&value)
            .map(|element| value.slice_ref(element))
            .collect()
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
        Some(self.vpath_search(&name)?.0)
    }

    /// The same search, over a name rather than a symbol, reporting how early
    /// in the search order the answer came from.
    ///
    /// The rank is GNU Make's `vpath_index` and `path_index`, and it exists for
    /// `library_search`: that caller runs the search once per `.LIBPATTERNS`
    /// element and has to weigh the answers against each other, where the
    /// earliest `vpath` entry wins whichever element reached it. A name is a
    /// symbol only once it is a target, and a library candidate is a name the
    /// search invented, so this half takes bytes.
    fn vpath_search(&self, name: &[u8]) -> Option<(Bytes, VpathRank)> {
        if name.is_empty() || self.ev.session.vpaths.is_empty() && self.vpath_variable().is_empty()
        {
            return None;
        }
        let mut matched_any = false;
        for (entry, (pattern, directories)) in self.ev.session.vpaths.iter().enumerate() {
            if !pattern.matches(name) {
                continue;
            }
            matched_any = true;
            if let Some((found, directory)) = Self::first_directory_holding(directories, name) {
                return Some((found, VpathRank { entry, directory }));
            }
        }
        if matched_any {
            return None;
        }
        // `VPATH` is a variable rather than a directive and so is read here
        // rather than recorded. It is searched after every `vpath` entry, which
        // is where its rank puts it.
        let (found, directory) = Self::first_directory_holding(&self.vpath_variable(), name)?;
        let entry = self.ev.session.vpaths.len();
        Some((found, VpathRank { entry, directory }))
    }

    /// The first of `directories` that holds `name`, and which one it was.
    fn first_directory_holding(directories: &[Bytes], name: &[u8]) -> Option<(Bytes, usize)> {
        for (index, directory) in directories.iter().enumerate() {
            let mut candidate = BytesMut::from(directory.as_ref());
            if !candidate.ends_with(b"/") {
                candidate.put_u8(b'/');
            }
            candidate.put_slice(name);
            let candidate = candidate.freeze();
            if std::fs::exists(OsStr::from_bytes(&candidate)).is_ok_and(|found| found) {
                return Some((candidate, index));
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
                self.populate_implicit_rule(rule, true)?;
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

        // POSIX says a suffix rule has no prerequisites, and GNU Make 4.4.1
        // reads a suffix-named rule that has them two ways. Outside `.POSIX:`
        // it keeps the old behaviour — the pair is converted and the written
        // prerequisites are dropped on the way, with a warning saying so —
        // and under `.POSIX:` it passes the pair over entirely, so the rule
        // makes nothing but the file it literally names. Either way the file
        // target survives, which is the half `is_buildable_target` answers.
        if !rule.inputs.is_empty() || !rule.order_only_inputs.is_empty() {
            if self.ev.is_posix() {
                return Ok(false);
            }
            warn_loc!(
                self.ev,
                rule.cmd_loc.as_ref().or(Some(&rule.loc)),
                "warning: ignoring prerequisites on suffix rule definition"
            );
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
        r.order_only_inputs.clear();
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

    /// Record a pattern rule, as GNU Make's `new_pattern_rule` does.
    ///
    /// `override_existing` is that function's second argument. A rule read from
    /// a Makefile displaces one already holding its identity, so a later
    /// definition wins; a rule from the built-in catalogue does not, and is
    /// thrown away instead — which is how a Makefile keeps a name the catalogue
    /// would otherwise have taken, whether or not it gave that name a recipe.
    fn populate_implicit_rule(&mut self, rule: Arc<Rule>, override_existing: bool) -> Result<()> {
        if let Some(index) = self
            .implicit_rule_defs
            .iter()
            .position(|existing| replaces_pattern_rule(&rule, existing))
        {
            if !override_existing {
                return Ok(());
            }
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
        scopes: &RuleScopes,
    ) -> (Option<Arc<Vars>>, Option<Arc<Vars>>) {
        let previous_rule_scope = self.cur_rule_vars.clone();
        let previous_eval_scope = self.ev.current_scope.clone();
        if scopes.is_empty() {
            return (previous_rule_scope, previous_eval_scope);
        }
        let scope = Arc::new(Vars::new());
        if let Some(previous) = &previous_rule_scope {
            scope.merge_from(previous);
        }
        // Second expansion is the target's own, so `private` is no barrier and
        // the two sets flatten in order.
        for (_, vars) in scopes.iter() {
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
        let mut bound = Vec::new();
        let trigger_text = trigger.as_bytes(&self.ev.session);
        let frame = self.ev.enter(
            FrameType::Dependency,
            trigger_text,
            action.lock().loc.clone().unwrap_or_default(),
        );
        self.apply_rule_vars(&vars, &action, &frame, &mut bound)?;

        let scope = self.cur_rule_vars.as_ref().map(|vars| {
            let scope = Vars::new();
            scope.merge_from(vars);
            Arc::new(scope)
        });
        let scoped_vars = release_private(bound, &self.ev.session);
        action.lock().rule_vars = scope;

        // Each `::` record stands on its own, and what `.EXTRA_PREREQS` adds is
        // required of every one of them — out of the automatic variables here
        // as everywhere else, so it joins the graph rather than the inputs.
        let (extra_compared, extra_order_only) = self.extra_prerequisites(trigger, true)?;
        let actual_inputs = action.lock().actual_inputs.clone();
        for input in actual_inputs.into_iter().chain(extra_compared) {
            let dependency = self.build_plan(input, Some(trigger))?;
            action.lock().deps.push((input, dependency));
        }
        let actual_order_only_inputs = action.lock().actual_order_only_inputs.clone();
        for input in actual_order_only_inputs.into_iter().chain(extra_order_only) {
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
        matched_at: &PatternMatch,
    ) -> Result<Option<(Vec<Symbol>, Vec<Symbol>)>> {
        let Some(text) = rule.deferred_prerequisites.clone() else {
            return Ok(None);
        };
        let key = (candidate_order, output);
        if let Some(found) = self.expanded.get(&key) {
            return Ok(Some(found.clone()));
        }
        let (recorded, recorded_order_only) = self.recorded_prerequisites(output);
        let expanded = self.expand_pattern_prerequisites_again(
            output,
            matched_at.clone(),
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
    ///
    /// One recipe-less rule is still read for something. A rule with neither a
    /// recipe nor prerequisites — which is what every declared suffix has, and
    /// what a Makefile writes as a bare `%.tex:` — records that this name was
    /// matched by a pattern that does not match every name. GNU Make's
    /// `specific_rule_matched` then strikes out every non-terminal rule whose
    /// target is a bare `%`, wherever it sits in the order. That is why the
    /// built-in link rules do not claim a `foo.o` the way they claim a `foo`.
    fn ordered_candidates(&self, output_str: &Bytes) -> Vec<ImplicitCandidate> {
        let mut specific_rule_matched = false;
        // The stem length each candidate leaves, measured while its pattern is
        // in hand. The catalogue puts every match-anything rule in front of
        // every search, so measuring inside the comparison would take the
        // pattern apart again for each of them at every comparison.
        let mut matched: Vec<(usize, usize, ImplicitCandidate)> = Vec::new();
        for candidate in self.candidate_pool(output_str) {
            // A cancelled rule — prerequisites, no recipe — and one a search
            // further out is already working through are both passed over
            // before they can be read as a match at all.
            if candidate.rule.cmds.is_empty() && Self::has_prerequisites(&candidate.rule) {
                continue;
            }
            if self.rules_in_use.contains(&Self::rule_id(&candidate.rule)) {
                continue;
            }
            let pattern = candidate.pattern.as_bytes(&self.ev.session);
            let pat = Pattern::new(pattern.clone());
            let Some(matched_at) = PatternMatch::of(&pat, output_str) else {
                continue;
            };
            specific_rule_matched |= pattern.as_ref() != b"%";
            if candidate.rule.cmds.is_empty() {
                continue;
            }
            // The directory the match held aside counts towards specificity:
            // `tryrules` records `stemlen + pathlen` and sorts on that, so a
            // rule is measured by the whole of what its `%` stood for.
            let stem = matched_at.directory.len() + matched_at.stem.len();
            let order = candidate.order;
            matched.push((stem, order, candidate));
        }
        if specific_rule_matched {
            matched.retain(|(_, _, candidate)| {
                candidate.rule.is_double_colon || !self.matches_anything(&candidate.rule)
            });
        }
        matched.sort_by_key(|(stem, order, _)| (*stem, *order));
        matched
            .into_iter()
            .map(|(_, _, candidate)| candidate)
            .collect()
    }

    /// The rules the index offers for a name, including those whose pattern is
    /// written for a bare name and can therefore only match the file part.
    ///
    /// The index is keyed by the literal text a pattern starts with, so `b%.x`
    /// is filed under `b` and a walk for `lib/bye.x` never reaches it. GNU Make
    /// keeps no index and compares every pattern rule to the name both ways
    /// round, so the file part has to be asked about separately here. A pattern
    /// starting with `%` answers to both names and is offered once.
    fn candidate_pool(&self, output_str: &Bytes) -> Vec<ImplicitCandidate> {
        let mut pool = self.implicit_rules.get(output_str);
        let path_len = directory_length(output_str);
        if path_len == 0 {
            return pool;
        }
        let mut seen: HashSet<(usize, Symbol)> = pool
            .iter()
            .map(|candidate| (Self::rule_id(&candidate.rule), candidate.pattern))
            .collect();
        for candidate in self.implicit_rules.get(&output_str[path_len..]) {
            if seen.insert((Self::rule_id(&candidate.rule), candidate.pattern)) {
                pool.push(candidate);
            }
        }
        pool
    }

    /// Run `body` with `rule` marked as one a search further out is working
    /// through, which is GNU Make's `rule->in_use` around the prerequisite loop
    /// of `pattern_search`.
    ///
    /// The mark is taken off however `body` leaves — a rule that failed is
    /// available to the next search, and only the searches nested inside this
    /// one are meant to pass over it.
    fn while_rule_in_use<T>(
        &mut self,
        rule: &Arc<Rule>,
        body: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let id = Self::rule_id(rule);
        let marked = self.rules_in_use.insert(id);
        let outcome = body(self);
        if marked {
            self.rules_in_use.remove(&id);
        }
        outcome
    }

    /// Whether every prerequisite this rule would need can be had, and how each
    /// of them was had.
    ///
    /// `None` is the rule failing. An all-empty answer is it applying with
    /// nothing invented and nothing to remember.
    fn implicit_prerequisites_reachable(
        &mut self,
        inputs: Vec<(Symbol, bool)>,
        pass: SearchPass,
    ) -> Result<Option<ReachedPrerequisites>> {
        let mut reached = ReachedPrerequisites::default();
        for (sym, from_pattern) in inputs {
            // A name an earlier search proved nothing can make fails this rule
            // outright, on either pass: the answer cannot have changed.
            if self.proven_impossible(sym, 0) {
                return Ok(None);
            }
            if self.exists(sym) {
                reached.found.push(sym);
                continue;
            }
            // The compatibility pass takes a prerequisite the Makefile merely
            // names. It runs after everything else has failed, so the name is
            // not invented and never becomes an intermediate: the search did
            // not make it up, it read it.
            if self.is_written_down(sym) {
                if pass.compat {
                    reached.found.push(sym);
                    continue;
                }
                // A name that is known but not promised is what makes a
                // compatibility pass worth running at all. Noted here and not
                // acted on: this rule may still apply by making the name.
                self.found_compat_rule = true;
            }
            if !(pass.chaining && self.intermediate_reachable(sym, 0, pass.compat)?) {
                return Ok(None);
            }
            if from_pattern && !self.mentioned.contains(&sym) {
                reached.invented.push(sym);
            }
        }
        Ok(Some(reached))
    }

    /// Whether an implicit chain could make this name, remembering a failure
    /// that was the rules' answer rather than the search's own limits.
    ///
    /// GNU Make marks the name where it gives up on making it an intermediate,
    /// and every later search rejects a rule that asks for it without walking
    /// the subtree again. A name the Makefile writes down is left unmarked, so
    /// that a compatibility pass can still take it on trust.
    fn intermediate_reachable(&mut self, name: Symbol, depth: usize, compat: bool) -> Result<bool> {
        if self.proven_impossible(name, depth) {
            return Ok(false);
        }
        let outer_truncated = std::mem::replace(&mut self.chain_truncated, false);
        let reachable = self.can_be_made_implicitly(name, depth, compat)?;
        let conclusive = !self.chain_truncated;
        self.chain_truncated |= outer_truncated;
        if !reachable && conclusive && !self.is_written_down(name) {
            let shallowest = self.impossible.entry(name).or_insert(depth);
            *shallowest = (*shallowest).min(depth);
        }
        Ok(reachable)
    }

    /// Whether a search with no more budget than this one already failed on
    /// this name.
    fn proven_impossible(&self, name: Symbol, depth: usize) -> bool {
        self.impossible
            .get(&name)
            .is_some_and(|shallowest| *shallowest <= depth)
    }

    /// Whether the rule was written with prerequisites, however they are held.
    fn has_prerequisites(rule: &Rule) -> bool {
        !rule.inputs.is_empty()
            || !rule.order_only_inputs.is_empty()
            || rule.deferred_prerequisites.is_some()
    }

    /// Whether the Makefile has written this name down anywhere at all.
    ///
    /// GNU Make's `pattern_search` asks `lookup_file`, which answers for any
    /// name that reached the file database: a target of a rule, a prerequisite
    /// of one, a goal, or a name carrying target-specific variables. It is a
    /// far weaker claim than "ought to exist" — the name is not promised, it is
    /// merely known — and it is what the compatibility pass runs on.
    fn is_written_down(&self, name: Symbol) -> bool {
        self.rules.contains_key(&name)
            || self.mentioned.contains(&name)
            || self.phony.contains(&name)
            || self.rule_vars.contains_key(&name)
            || self.ev.goals.contains(&name)
    }

    /// The names a matched pattern rule's prerequisites stand for, with each
    /// paired with whether it came from the pattern rather than being written
    /// out in full.
    ///
    /// A prerequisite may hold shell wildcards, and GNU Make expands them here
    /// rather than where the rule was read: `parse_file_seq` globs the
    /// substituted name, so `%.t*` is looked up as `a.t*` and can stand for
    /// several files at once. A pattern that matches nothing is kept as it was
    /// written, which is what `GLOB_NOMATCH` falls through to when the search
    /// has not asked for existing names only.
    fn resolved_prerequisites(
        &mut self,
        prerequisites: &[Symbol],
        matched_at: &PatternMatch,
    ) -> Vec<(Symbol, bool)> {
        let mut resolved = Vec::with_capacity(prerequisites.len());
        for prerequisite in prerequisites {
            let text = prerequisite.as_bytes(&self.ev.session);
            let from_pattern = text.contains(&b'%');
            let name = matched_at.prerequisite(&text);
            let mut named = Vec::new();
            glob_word(&mut self.ev.session, name, &mut named);
            resolved.extend(named.into_iter().map(|name| (name, from_pattern)));
        }
        resolved
    }

    fn can_pick_implicit_rule(
        &mut self,
        rule: &Arc<Rule>,
        matched: Symbol,
        candidate_order: usize,
        output: Symbol,
        n: Arc<Mutex<DepNode>>,
        pass: SearchPass,
    ) -> Result<Option<Arc<Rule>>> {
        let output_str = output.as_bytes(&self.ev.session);
        let pat = Pattern::new(matched.as_bytes(&self.ev.session));
        let Some(matched_at) = PatternMatch::of(&pat, &output_str) else {
            return Ok(None);
        };
        let deferred = self.expanded_pattern_inputs(rule, candidate_order, output, &matched_at)?;
        let inputs: Vec<(Symbol, bool)> = match &deferred {
            // A deferred list is one string until it is expanded, so
            // which word the `%` was in is no longer knowable.
            Some((inputs, _)) => inputs.iter().map(|input| (*input, false)).collect(),
            None => self.resolved_prerequisites(&rule.inputs, &matched_at),
        };
        let resolved_inputs: Vec<Symbol> = inputs.iter().map(|(input, _)| *input).collect();
        let Some(reached) = self.while_rule_in_use(rule, |builder| {
            builder.implicit_prerequisites_reachable(inputs, pass)
        })?
        else {
            return Ok(None);
        };
        self.intermediates.extend(reached.invented);
        // A terminal rule reads what is already there, so a prerequisite it was
        // given rather than made is one no implicit search may go on to make.
        // GNU Make stamps it `tried_implicit` (`implicit.c`), which is what
        // keeps `%.z:: %.x` from being satisfied by the `%.x:` rule below it,
        // and the stamp lands as the rule is chosen — so whether a name was
        // already reached in its own right decides the answer, exactly as the
        // order of GNU Make's update walk does.
        if rule.is_double_colon {
            self.tried_implicit.extend(reached.found);
        }

        // What the match read, kept for `$*`: with a directory held aside the
        // stem is not recoverable from the pattern and the name alone.
        let stem = matched_at.whole_stem();
        n.lock().stem = Some(self.ev.session.intern(stem));

        // Either way the names are final now: the search has filled the `%` in
        // and put back the directory it held aside, so nothing downstream gets
        // to substitute into them a second time.
        let mut rule = rule.as_ref().clone();
        match deferred {
            Some((inputs, order_only_inputs)) => {
                rule.deferred_prerequisites = None;
                rule.inputs = inputs;
                rule.order_only_inputs = order_only_inputs;
            }
            None => {
                let order_only = self.resolved_prerequisites(&rule.order_only_inputs, &matched_at);
                rule.inputs = resolved_inputs;
                rule.order_only_inputs = order_only.into_iter().map(|(input, _)| input).collect();
            }
        }
        rule.prerequisites_are_resolved = true;
        if rule.output_patterns.len() > 1 {
            // A pattern rule with several target patterns is one recipe that
            // makes all of them, so the rest are this node's outputs — unless
            // the name already has a maker of its own. GNU Make's `also_make`
            // only marks such a name updated when this recipe runs; it does not
            // take it away from the rule its own search chose, and two rules
            // that can each make one name is not an error to it.
            n.lock().pattern_group = true;
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
    fn can_be_made_implicitly(
        &mut self,
        output: Symbol,
        depth: usize,
        compat: bool,
    ) -> Result<bool> {
        if depth >= MAX_IMPLICIT_CHAIN {
            return Ok(false);
        }
        if !self.chaining.insert(output) {
            self.chain_truncated = true;
            return Ok(false);
        }
        // One recursion is one whole search, so it runs a compatibility pass of
        // its own once its strict pass has failed and passed over a rule for a
        // written-down name — unless the search that reached it is already the
        // compatibility pass, which it inherits.
        let outer_compat = std::mem::replace(&mut self.found_compat_rule, false);
        let mut answer = self.implicit_chain_exists(output, depth, compat);
        if matches!(answer, Ok(false)) && !compat && self.found_compat_rule {
            answer = self.implicit_chain_exists(output, depth, true);
        }
        self.found_compat_rule = outer_compat;
        self.chaining.remove(&output);
        answer
    }

    fn implicit_chain_exists(
        &mut self,
        output: Symbol,
        depth: usize,
        compat: bool,
    ) -> Result<bool> {
        let output_str = output.as_bytes(&self.ev.session);
        for candidate in self.ordered_candidates(&output_str) {
            let rule = candidate.rule;
            // Make's step 6a: a non-terminal match-anything rule is not allowed
            // to make an intermediate.
            if !rule.is_double_colon && self.matches_anything(&rule) {
                continue;
            }
            let pat = Pattern::new(candidate.pattern.as_bytes(&self.ev.session));
            let Some(matched_at) = PatternMatch::of(&pat, &output_str) else {
                continue;
            };
            let inputs =
                match self.expanded_pattern_inputs(&rule, candidate.order, output, &matched_at)? {
                    Some((inputs, _)) => inputs,
                    None => rule
                        .inputs
                        .iter()
                        .map(|input| {
                            let buf = matched_at.prerequisite(&input.as_bytes(&self.ev.session));
                            self.ev.session.intern(buf)
                        })
                        .collect(),
                };
            // The same terminal restriction, one level in. A terminal rule is
            // never offered the pass that invents its prerequisites, so it can
            // serve as a link in a chain only when what it reads is there.
            let terminal = rule.is_double_colon;
            let reachable = self.while_rule_in_use(&rule, |builder| {
                for i in inputs {
                    if builder.proven_impossible(i, depth + 1) {
                        return Ok(false);
                    }
                    if builder.exists(i) {
                        continue;
                    }
                    if builder.is_written_down(i) {
                        if compat {
                            continue;
                        }
                        builder.found_compat_rule = true;
                    }
                    if terminal || !builder.intermediate_reachable(i, depth + 1, compat)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            })?;
            if reachable {
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
            if self.rules_in_use.contains(&Self::rule_id(irule)) {
                continue;
            }
            let input = replace_suffix(&mut self.ev.session, output, &irule.inputs[0]);
            let reachable = self.while_rule_in_use(irule, |builder| {
                if builder.proven_impossible(input, depth + 1) {
                    return Ok(false);
                }
                if builder.exists(input) {
                    return Ok(true);
                }
                if builder.is_written_down(input) {
                    if compat {
                        return Ok(true);
                    }
                    builder.found_compat_rule = true;
                }
                builder.intermediate_reachable(input, depth + 1, compat)
            })?;
            if reachable {
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

        // Steps 5 then 6, over the same rules, and then both again taking a
        // written-down prerequisite on trust. Each pass runs out over every
        // candidate before the next one begins; `SearchPass` says why.
        //
        // Not for a phony name. GNU Make's `remake.c` asks for an implicit rule
        // only where `!file->phony`, and it matters as soon as there are
        // built-in rules to find: `%: %.c` matches every name there is, so an
        // `all` declared `.PHONY` beside an `all.c` would otherwise acquire a
        // recipe that links a program nobody asked for.
        //
        // Nor for a name a terminal rule has already been given, which is the
        // other half of the same condition: `!file->tried_implicit`.
        if !self.phony.contains(&output) && !self.tried_implicit.contains(&output) {
            let outer_compat = std::mem::replace(&mut self.found_compat_rule, false);
            let mut picked = None;
            for pass in SearchPass::all() {
                // The passes that take a name on trust are a retry of the whole
                // search, and GNU Make retries only a search that passed over a
                // rule for such a name.
                if pass.compat && !self.found_compat_rule {
                    break;
                }
                picked = self.pick_pattern_rule(output, n, &rule_merger, &patterns, &vars, pass)?;
                if picked.is_some() {
                    break;
                }
            }
            self.found_compat_rule = outer_compat;
            if picked.is_some() {
                return Ok(picked);
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
        let already_there = self.exists(output);
        let default_rule = self.default_rule.clone().filter(|_| !already_there);
        Ok(default_rule.map(|rule| PickedRuleInfo {
            merger: None,
            pattern_rule: Some(rule),
            vars: Self::scopes_for(&patterns, vars),
        }))
    }

    /// The matching patterns, then the target's own scope on top of them.
    fn scopes_for(patterns: &[Arc<Vars>], own: Option<Arc<Vars>>) -> RuleScopes {
        RuleScopes {
            patterns: patterns.to_vec(),
            own,
        }
    }

    /// Every scope that applies to `output`, weakest first, without consulting
    /// the rule that makes it. For callers that have a target's name and no
    /// picked rule to go with it.
    fn applicable_rule_vars(&self, output: Symbol) -> RuleScopes {
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
        scopes: &RuleScopes,
        node: &Arc<Mutex<DepNode>>,
        frame: &ScopedFrame,
        bound: &mut Vec<RuleBinding>,
    ) -> Result<()> {
        for (kind, vars) in scopes.iter() {
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
                    bound.push(RuleBinding {
                        guard: ScopedVar::new(
                            self.cur_rule_vars.clone().unwrap(),
                            *name,
                            new_var.clone(),
                        ),
                        kind,
                        sym: *name,
                        var: new_var,
                        private: is_private,
                    });
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
        pass: SearchPass,
    ) -> Result<Option<PickedRuleInfo>> {
        let candidates = self.ordered_candidates(&output.as_bytes(&self.ev.session));
        for candidate in candidates {
            // A terminal rule is not offered the pass that invents what it
            // needs. GNU Make rejects one outright while it is looking to make
            // intermediate files, which is what "terminal" means: the rule
            // applies to what is already there. The catalogue's RCS and SCCS
            // rules are the reason it has to hold — each of them matches every
            // name, so a `foo` allowed to invent `foo,v` would then be allowed
            // to invent `foo,v,v` for it, without end.
            if pass.chaining && candidate.rule.is_double_colon {
                continue;
            }
            let Some(pattern_rule) = self.can_pick_implicit_rule(
                &candidate.rule,
                candidate.pattern,
                candidate.order,
                output,
                n.clone(),
                pass,
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
            if self.rules_in_use.contains(&Self::rule_id(irule)) {
                continue;
            }
            let input = replace_suffix(&mut self.ev.session, output, &irule.inputs[0]);
            if self.proven_impossible(input, 0) {
                continue;
            }
            let mut taken_on_trust = false;
            if !self.exists(input) && self.is_written_down(input) {
                taken_on_trust = pass.compat;
                self.found_compat_rule |= !pass.compat;
            }
            if !self.exists(input) && !taken_on_trust {
                let reachable = self.while_rule_in_use(irule, |builder| {
                    Ok(pass.chaining && builder.intermediate_reachable(input, 0, pass.compat)?)
                })?;
                if !reachable {
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

        // What `.EXTRA_PREREQS` adds is the same shape as a hidden peer: in the
        // graph and in the freshness test, out of every automatic variable. It
        // rides the same pass so VPATH and `.WAIT` reach it too, and it goes on
        // the end because GNU Make appends it to what the rule already declared.
        //
        // Reaching here at all means a rule was picked, which is what makes the
        // name a target and so eligible for the global list.
        let (extra_compared, extra_order_only) = self.extra_prerequisites(output, true)?;
        grouped_peer_inputs.extend(extra_compared);
        grouped_peer_order_only.extend(extra_order_only);

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

        let mut bound = Vec::new();
        let frame = self.ev.enter(
            FrameType::Dependency,
            output_str.clone(),
            n.lock().loc.clone().unwrap_or_default(),
        );

        self.apply_rule_vars(&picked_rule_info.vars, &n, &frame, &mut bound)?;

        // A `private` target-specific variable belongs to this target's own
        // recipe and to no prerequisite's, so the scope is read here, with it in
        // it, and it leaves before the prerequisites are planned.
        let scope = self.cur_rule_vars.as_ref().map(|vars| {
            let v = Vars::new();
            v.merge_from(vars);
            Arc::new(v)
        });
        let sv = release_private(bound, &self.ev.session);

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
) -> Result<Plan> {
    let mut db = DepBuilder::new(ev)?;
    let _tr = ScopedTimeReporter::new(&db.ev.session, "make dep (build)");
    let built = db.build(targets, read_makefiles, missing_includes)?;
    // Hand the planned scopes back, so a target-specific assignment made from a
    // recipe can still reach the target it names. GNU Make never has to do this
    // — its target variables live on the file for the whole run — but Ronin's
    // live on the node, and this is the only route from a name to one.
    let planned = db
        .done
        .iter()
        .filter_map(|(target, node)| node.lock().rule_vars.clone().map(|vars| (*target, vars)))
        .collect();
    db.ev.planned_scopes = planned;
    Ok(built)
}

/// Which of a target's two prerequisite lists an edge was written on.
///
/// GNU Make keeps one list and marks the order-only half `ignore_mtime`; the
/// plan keeps two. A dropped edge has to come off the one it was on, and off
/// the names beside it that the automatic variables are read from.
#[derive(Clone, Copy)]
enum Prerequisites {
    Compared,
    OrderOnly,
}

/// What tells two planned targets apart while the plan is walked.
///
/// The plan hands the same record back for every mention of a name, so the
/// record itself is the target — which is what GNU Make's `updating` flag is
/// set on, and what makes a `::` target's separate actions one target between
/// them.
fn identity(node: &Arc<Mutex<DepNode>>) -> usize {
    Arc::as_ptr(node) as usize
}

/// The Make target this planned record stands for, as a diagnostic names it.
fn recipe_name(names: &impl Interner, node: &Arc<Mutex<DepNode>>) -> String {
    let output = node.lock().recipe_output;
    String::from_utf8_lossy(&output.as_bytes(names)).into_owned()
}

/// Take one prerequisite off the list it was written on.
///
/// Both halves go: the edge, which is what the graph is built from, and the
/// name beside it, which is what `$^` and `$?` are read from. The edge list is
/// the name list with the grouped record's own members appended, so an entry
/// found before the names run out is that name — and a `&:` member, which was
/// never a written prerequisite, leaves the names alone.
fn drop_prerequisite(
    from: &Arc<Mutex<DepNode>>,
    list: Prerequisites,
    dropped: &Arc<Mutex<DepNode>>,
) {
    let mut held = from.lock();
    let from = &mut *held;
    let (edges, names) = match list {
        Prerequisites::Compared => (&mut from.deps, &mut from.actual_inputs),
        Prerequisites::OrderOnly => (&mut from.order_onlys, &mut from.actual_order_only_inputs),
    };
    let Some(at) = edges
        .iter()
        .position(|(_, node)| Arc::ptr_eq(node, dropped))
    else {
        return;
    };
    let (name, _) = edges.remove(at);
    if names.get(at) == Some(&name) {
        names.remove(at);
    }
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

/// Whether this node belongs in the manifest, given whether anything gave it a
/// recipe.
///
/// A suffix-shaped name is an ordinary file target, and being converted into a
/// pattern rule as well does not stop it being one: GNU Make's
/// `convert_to_pattern` reads the `.c.o` entry out of the file database and
/// leaves it there, so `make .c.o` runs that rule and `make .baz.biz` runs
/// that one. `has_rule` is what tells the two apart, because the exclusion is
/// only ever needed for a suffix-shaped name nothing declared: emitted then, it
/// is claimed by the built-in `%.o: %.c` with an empty stem and runs
/// `cc -c -o .c.o` against no input, which is worse than refusing it.
pub fn is_buildable_target(names: &impl Interner, output: &Symbol, has_rule: bool) -> bool {
    !is_directive_target(names, output) && (has_rule || !is_suffix_rule(names, output))
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

    /// Only the first `%` of each whitespace-separated segment stands for the
    /// stem. GNU Make replaces one, skips to the next blank, and looks again, so
    /// a percent that follows another inside the same segment is a literal.
    #[test]
    fn only_the_first_percent_of_a_segment_stands_for_the_stem() {
        assert_eq!(
            stem_references(&Bytes::from_static(b"$(wordlist 1, 99, %1%2%)"), false),
            (Bytes::from_static(b"$(wordlist 1, 99, $*1%2%)"), true)
        );
        assert_eq!(
            stem_references(&Bytes::from_static(b"$(wordlist 1, 99, %a %b)"), false),
            (Bytes::from_static(b"$(wordlist 1, 99, $*a $*b)"), true)
        );
        // Holding a directory aside means the stem reference is the file part
        // of it, because the directory is added back once, in front.
        assert_eq!(
            stem_references(&Bytes::from_static(b"6%"), true),
            (Bytes::from_static(b"6$(*F)"), true)
        );
        // A word naming no stem takes no directory either.
        assert_eq!(
            stem_references(&Bytes::from_static(b"nopercent.c"), true),
            (Bytes::from_static(b"nopercent.c"), false)
        );
    }

    /// A pattern carrying no directory of its own is matched against the file
    /// part of the name, and what it matched is read back with the directory in
    /// front of it.
    #[test]
    fn a_pattern_without_a_directory_matches_the_file_part() {
        let matched = |pattern: &'static [u8], output: &'static [u8]| {
            PatternMatch::of(
                &Pattern::new(Bytes::from_static(pattern)),
                &Bytes::from_static(output),
            )
        };

        let found = matched(b"%.x", b"lib/bye.x").expect("a match");
        assert_eq!(found.directory, Bytes::from_static(b"lib/"));
        assert_eq!(found.stem, Bytes::from_static(b"bye"));
        assert_eq!(found.whole_stem(), Bytes::from_static(b"lib/bye"));
        assert_eq!(
            found.prerequisite(&Bytes::from_static(b"6%")),
            Bytes::from_static(b"lib/6bye")
        );
        // The prerequisite's own directory is not special, and one naming no
        // stem is left exactly as it was written.
        assert_eq!(
            found.prerequisite(&Bytes::from_static(b"sub/%.c")),
            Bytes::from_static(b"lib/sub/bye.c")
        );
        assert_eq!(
            found.prerequisite(&Bytes::from_static(b"nopercent.c")),
            Bytes::from_static(b"nopercent.c")
        );

        // The literal before the `%` has to match the file part, not the whole
        // name: `l` is the start of `lib/` but not of `bye.x`.
        assert!(matched(b"l%.x", b"lib/bye.x").is_none());
        let found = matched(b"b%.x", b"lib/bye.x").expect("a match");
        assert_eq!(found.whole_stem(), Bytes::from_static(b"lib/ye"));

        // A pattern carrying a slash is matched whole and holds nothing aside.
        let found = matched(b"lib/%.x", b"lib/bye.x").expect("a match");
        assert!(found.directory.is_empty());
        assert_eq!(found.whole_stem(), Bytes::from_static(b"bye"));
        assert_eq!(
            found.prerequisite(&Bytes::from_static(b"6%")),
            Bytes::from_static(b"6bye")
        );

        // A name with no directory leaves the pattern reading it whole.
        let found = matched(b"%.x", b"bye.x").expect("a match");
        assert!(found.directory.is_empty());
        assert_eq!(found.whole_stem(), Bytes::from_static(b"bye"));
    }

    /// A trailing slash names the directory rather than separating it from
    /// anything, so it is not where the split happens.
    #[test]
    fn a_trailing_slash_belongs_to_the_directory_it_names() {
        assert_eq!(directory_length(b"foo/bar/"), 4);
        assert_eq!(directory_length(b"foo/bar"), 4);
        assert_eq!(directory_length(b"bar"), 0);
        assert_eq!(directory_length(b""), 0);
        assert_eq!(directory_length(b"/"), 0);
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

    /// GNU Make ranks the system directories by starting their indices above
    /// every `vpath` index there could be, so every vpath answer outranks every
    /// system-directory one. Here that is the enum's declaration order, which
    /// nothing else in the file would notice going wrong.
    #[test]
    fn every_vpath_library_answer_outranks_every_system_one() {
        let latest_vpath = LibraryRank::Vpath(VpathRank {
            entry: usize::MAX,
            directory: usize::MAX,
        });
        let earliest_system = LibraryRank::System(0);
        assert!(latest_vpath < earliest_system);
        // Within a vpath answer the entry decides first and the directory
        // inside it second, which is what `-l2` in GNU Make's own suite turns
        // on: two answers from one entry, settled by which directory held one.
        let first_entry_last_directory = LibraryRank::Vpath(VpathRank {
            entry: 0,
            directory: usize::MAX,
        });
        let second_entry_first_directory = LibraryRank::Vpath(VpathRank {
            entry: 1,
            directory: 0,
        });
        assert!(first_entry_last_directory < second_entry_first_directory);
        assert!(
            LibraryRank::Vpath(VpathRank {
                entry: 3,
                directory: 0
            }) < LibraryRank::Vpath(VpathRank {
                entry: 3,
                directory: 1
            })
        );
        assert!(LibraryRank::System(0) < LibraryRank::System(1));
    }

    /// The earlier `.LIBPATTERNS` element keeps an equally ranked answer, which
    /// is the strict `<` in GNU Make's two comparisons rather than a `<=`.
    #[test]
    fn an_equally_ranked_library_answer_does_not_displace_the_one_held() {
        let mut best = None;
        DepBuilder::keep_earlier(
            &mut best,
            LibraryRank::System(1),
            Bytes::from_static(b"first"),
        );
        DepBuilder::keep_earlier(
            &mut best,
            LibraryRank::System(1),
            Bytes::from_static(b"second"),
        );
        assert_eq!(
            best.as_ref().map(|(_, path)| path.clone()).unwrap(),
            "first"
        );
        DepBuilder::keep_earlier(
            &mut best,
            LibraryRank::System(0),
            Bytes::from_static(b"earlier"),
        );
        assert_eq!(
            best.as_ref().map(|(_, path)| path.clone()).unwrap(),
            "earlier"
        );
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
                is_buildable_target(&session, &sym, false),
                "{name} should be built"
            );
        }
        for name in [".PHONY", ".SUFFIXES", ".KATI_RESTAT", ".ONESHELL", ".WAIT"] {
            let sym = session.intern(name);
            assert!(
                !is_buildable_target(&session, &sym, true),
                "{name} should not be built"
            );
        }
    }

    /// A `.c.o:` rule names a file called `.c.o` as well as converting into a
    /// pattern rule, and GNU Make builds that file when it is asked for. The
    /// exclusion is only for the name nothing declared, which the built-in
    /// `%.o: %.c` would otherwise claim with an empty stem.
    #[test]
    fn a_suffix_shaped_target_is_built_when_a_rule_reached_it() {
        let mut session = Session::new();
        let sym = session.intern(".c.o");
        assert!(is_buildable_target(&session, &sym, true));
        assert!(!is_buildable_target(&session, &sym, false));
    }
}
