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

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::io::BufWriter;
use std::ops::Range;
use std::os::unix::ffi::OsStringExt;
use std::sync::{Arc, Weak};

use anyhow::Result;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use memchr::memchr;
use parking_lot::Mutex;

use crate::build_sink::{NewInputsTiming, ShellEvaluation};
use crate::expr::{Evaluable, ParseExprOpt, Value, parse_expr};
use crate::file::Source;
use crate::flags::Flags;
use crate::loc::Loc;
use crate::parser::{parse_assign_statement, parse_buf_no_stats};
use crate::rule::{Rule, glob_word, is_pattern_rule};
use crate::session::{Context, Session};
use crate::stats::StatsRegistry;
use crate::stmt::{
    AssignModifiers, AssignOp, AssignStmt, CommandStmt, CondOp, ExportStmt, IfStmt, IncludeStmt,
    RuleStmt, Statement, UndefineStmt, VpathStmt,
};
use crate::strutil::{
    Pattern, is_space_byte, makefile_word_scanner, strip_recipe_prefix_continuations,
    trim_leading_curdir, trim_left_space, trim_right_space, word_scanner,
};
use crate::symtab::{Interner, Symbol, Symtab};
use crate::var::{Var, VarExport, VarOrigin, Variable, Vars};
use crate::{collect_stats_with_slow_report, error_loc, log, warn_loc};

pub enum RulesAllowed {
    Allowed,
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuleState {
    None,
    Active,
    Ignored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuleColonOrigin {
    Literal,
    Expansion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuleDelimiter {
    colon: usize,
    origin: RuleColonOrigin,
}

#[derive(Clone, Copy)]
struct RuleWordByte {
    byte: u8,
    origin: RuleColonOrigin,
}

/// The expansion of one source word, retaining where each byte came from so
/// the effective colon keeps its provenance after GNU Make's unquoting pass.
#[derive(Default)]
struct RuleWordExpansion {
    output: Vec<RuleWordByte>,
}

impl RuleWordExpansion {
    fn push_literal(&mut self, literal: &[u8]) {
        self.output
            .extend(literal.iter().copied().map(|byte| RuleWordByte {
                byte,
                origin: RuleColonOrigin::Literal,
            }));
    }

    /// Append the result of expanding one source expression. Whitespace here
    /// remains inside that expression's source word.
    fn push_expansion(&mut self, expansion: &[u8]) {
        self.output
            .extend(expansion.iter().copied().map(|byte| RuleWordByte {
                byte,
                origin: RuleColonOrigin::Expansion,
            }));
    }

    fn find_char_unquote(&mut self, needle: u8) -> Option<(usize, RuleColonOrigin)> {
        let index = find_char_unquote_by(&mut self.output, needle, |byte| byte.byte)?;
        Some((index, self.output[index].origin))
    }

    fn take_after(&mut self, separator: usize) -> Bytes {
        let command = self
            .output
            .drain(separator + 1..)
            .map(|byte| byte.byte)
            .collect::<Vec<_>>();
        self.output.truncate(separator);
        Bytes::from(command)
    }

    fn finish(self) -> Bytes {
        Bytes::from(
            self.output
                .into_iter()
                .map(|byte| byte.byte)
                .collect::<Vec<_>>(),
        )
    }
}

struct ExpandedRuleHead {
    text: Bytes,
    rest: Bytes,
    delimiter: Option<RuleDelimiter>,
    expanded_command: Option<Bytes>,
    had_source_word: bool,
}

/// A canonical switch prefix, optionally retaining GNU Make's recursive
/// command-line override suffix.
fn makeflags_value(
    makeflags: Bytes,
    has_overrides: bool,
    overrides: Symbol,
) -> (Arc<Value>, Bytes) {
    if !has_overrides {
        return (Arc::new(Value::Literal(None, makeflags.clone())), makeflags);
    }

    let mut prefix = BytesMut::from(makeflags.as_ref());
    prefix.put_slice(b" -- ");
    let mut original = prefix.clone();
    original.put_slice(b"$(MAKEOVERRIDES)");
    (
        Arc::new(Value::List(
            None,
            vec![
                Arc::new(Value::Literal(None, prefix.freeze())),
                Arc::new(Value::SymRef(Loc::default(), overrides)),
            ],
        )),
        original.freeze(),
    )
}

struct HybridRuleText {
    text: Bytes,
}

#[derive(Debug, PartialEq, Eq)]
struct VariableDefinition {
    name: Range<usize>,
    value_start: usize,
    op: AssignOp,
}

#[derive(Debug, PartialEq, Eq)]
struct ScannedRuleAssignment {
    definition: VariableDefinition,
    modifiers: AssignModifiers,
}

struct RuleAssignment {
    name: Arc<Value>,
    rhs: Arc<Value>,
    orig_rhs: Bytes,
    op: AssignOp,
    modifiers: AssignModifiers,
    is_final: bool,
}

/// Find GNU Make's first effective separator while compacting each immediately
/// preceding backslash run. Half the run remains; an odd run quotes this
/// occurrence and scanning continues after it.
fn find_char_unquote_by<T>(
    text: &mut Vec<T>,
    needle: u8,
    mut byte_of: impl FnMut(&T) -> u8,
) -> Option<usize> {
    let mut search = 0usize;
    loop {
        let found = text[search..]
            .iter()
            .position(|byte| byte_of(byte) == needle)?
            + search;
        let mut slashes = found;
        while slashes > 0 && byte_of(&text[slashes - 1]) == b'\\' {
            slashes -= 1;
        }
        let slash_count = found - slashes;
        if slash_count == 0 {
            return Some(found);
        }

        let retained = slash_count / 2;
        let removed = slash_count - retained;
        text.drain(slashes..slashes + removed);
        let found = found - removed;
        if slash_count.is_multiple_of(2) {
            return Some(found);
        }
        search = found + 1;
    }
}

fn find_char_unquote(text: &mut Vec<u8>, needle: u8) -> Option<usize> {
    find_char_unquote_by(text, needle, |byte| *byte)
}

/// The next token GNU Make's `get_next_mword` would expand while looking for a
/// rule colon. Literal operators are their own tokens; a variable reference is
/// kept whole even when its spelling contains them.
fn next_rule_word(text: &[u8], mut at: usize) -> Option<Range<usize>> {
    while text.get(at).is_some_and(u8::is_ascii_whitespace) {
        at += 1;
    }
    if at == text.len() {
        return None;
    }

    let start = at;
    match text[at] {
        b':' => {
            at += 1;
            if matches!(text.get(at), Some(b':' | b'=')) {
                at += 1;
            }
            return Some(start..at);
        }
        b'&' if text.get(at + 1) == Some(&b':') => {
            at += 2;
            if text.get(at) == Some(&b':') {
                at += 1;
            }
            return Some(start..at);
        }
        b'=' | b';' => return Some(start..start + 1),
        b'?' | b'+' | b'!' if text.get(at + 1) == Some(&b'=') => {
            return Some(start..start + 2);
        }
        _ => {}
    }

    while let Some(byte) = text.get(at) {
        if byte.is_ascii_whitespace() || matches!(byte, b':' | b'=') {
            break;
        }
        if *byte == b'\\'
            && text
                .get(at + 1)
                .is_some_and(|next| matches!(next, b':' | b';' | b'=' | b'\\'))
        {
            at += 2;
            continue;
        }
        if *byte == b'$' {
            at += 1;
            let Some(next) = text.get(at) else {
                break;
            };
            if *next == b'$' {
                at += 1;
                continue;
            }
            let close = match next {
                b'(' => Some(b')'),
                b'{' => Some(b'}'),
                _ => None,
            };
            at += 1;
            if let Some(close) = close {
                let open = *next;
                let mut depth = 0usize;
                while let Some(inner) = text.get(at) {
                    if *inner == open {
                        depth += 1;
                    } else if *inner == close {
                        if depth == 0 {
                            at += 1;
                            break;
                        }
                        depth -= 1;
                    }
                    at += 1;
                }
            }
            continue;
        }
        if matches!(byte, b'?' | b'+') && text.get(at + 1) == Some(&b'=') {
            break;
        }
        if *byte == b'&' && text.get(at + 1) == Some(&b':') {
            break;
        }
        at += 1;
    }
    Some(start..at)
}

fn skip_make_space(text: &[u8], mut at: usize) -> usize {
    while text.get(at).is_some_and(is_space_byte) {
        at += 1;
    }
    at
}

/// Skip the variable reference beginning immediately after a `$`.
///
/// This is GNU Make's `skip_reference`: a one-character reference consumes one
/// byte, while a parenthesized or braced reference consumes through its
/// matching delimiter, including nested delimiters of the same kind.
fn skip_make_reference(text: &[u8], mut at: usize) -> usize {
    let Some(open) = text.get(at).copied() else {
        return at;
    };
    let close = match open {
        b'(' => b')',
        b'{' => b'}',
        _ => return at + 1,
    };
    let mut depth = 1usize;
    while at + 1 < text.len() {
        at += 1;
        if text[at] == open {
            depth += 1;
        } else if text[at] == close {
            depth -= 1;
            if depth == 0 {
                return at + 1;
            }
        }
    }
    text.len()
}

/// Port of GNU Make's `parse_variable_definition` scanner.
///
/// In particular, `#` and a non-operator `:` reject a definition, a reference
/// is skipped as one unit, and backslash has no quoting role here.  Those are
/// classification rules, not expression parsing rules.
fn scan_variable_definition(text: &[u8], at: usize) -> Option<VariableDefinition> {
    let mut at = skip_make_space(text, at);
    let name_start = at;
    let mut name_end = None;

    loop {
        let operator_start = at;
        let byte = *text.get(at)?;
        at += 1;

        if byte == b'#' {
            return None;
        }

        if matches!(byte, b' ' | b'\t') {
            if name_end.is_some() {
                return None;
            }
            name_end = Some(operator_start);
            at = skip_make_space(text, at);
            continue;
        }

        let op = if byte == b'=' {
            Some(AssignOp::Eq)
        } else if byte == b':' {
            match text.get(at..) {
                Some([b'=', ..]) => {
                    at += 1;
                    Some(AssignOp::ColonEq)
                }
                Some([b':', b'=', ..]) => {
                    at += 2;
                    Some(AssignOp::ColonEq)
                }
                Some([b':', b':', b'=', ..]) => {
                    at += 3;
                    Some(AssignOp::ImmediateRecursive)
                }
                _ => return None,
            }
        } else if text.get(at) == Some(&b'=') {
            match byte {
                b'?' => {
                    at += 1;
                    Some(AssignOp::QuestionEq)
                }
                b'+' => {
                    at += 1;
                    Some(AssignOp::PlusEq)
                }
                b'!' => {
                    at += 1;
                    Some(AssignOp::ShellEq)
                }
                _ => None,
            }
        } else {
            None
        };

        if let Some(op) = op {
            let name_end = name_end.unwrap_or(operator_start);
            return Some(VariableDefinition {
                name: name_start..name_end,
                value_start: skip_make_space(text, at),
                op,
            });
        }

        if name_end.is_some() {
            return None;
        }
        if byte == b'$' {
            at = skip_make_reference(text, at);
        }
    }
}

/// GNU Make tests for a definition before treating the leading word as a
/// target-variable modifier.  Thus `private = value` defines `private`, while
/// `private NAME = value` consumes the modifier and defines `NAME`.
fn scan_rule_assignment(text: &[u8]) -> Option<ScannedRuleAssignment> {
    let mut at = skip_make_space(text, 0);
    let mut modifiers = AssignModifiers::default();

    loop {
        if let Some(definition) = scan_variable_definition(text, at) {
            return Some(ScannedRuleAssignment {
                definition,
                modifiers,
            });
        }

        let end = text[at..]
            .iter()
            .position(is_space_byte)
            .map_or(text.len(), |end| at + end);
        match &text[at..end] {
            b"private" => modifiers.directive.is_private = true,
            b"export" => modifiers.directive.export = VarExport::Export,
            b"override" => modifiers.directive.is_override = true,
            b"unexport" => modifiers.directive.export = VarExport::NoExport,
            _ => return None,
        }
        modifiers.words += 1;
        at = skip_make_space(text, end);
        if at == text.len() {
            return None;
        }
    }
}

/// Collapse continuations in the rule text GNU Make examines before deciding
/// whether an expanded semicolon begins an inline recipe. A command following
/// a literal semicolon is cut off before this step and therefore stays intact.
fn collapse_rule_continuations(source: Bytes, posix: bool) -> Bytes {
    let Some(mut newline) = memchr(b'\n', &source) else {
        return source;
    };

    let mut output = BytesMut::with_capacity(source.len());
    let mut copied = 0usize;
    loop {
        let line_end = if newline > copied && source[newline - 1] == b'\r' {
            newline - 1
        } else {
            newline
        };
        let mut slashes = line_end;
        while slashes > copied && source[slashes - 1] == b'\\' {
            slashes -= 1;
        }
        let slash_count = line_end - slashes;
        output.put_slice(&source[copied..slashes]);
        for _ in 0..slash_count / 2 {
            output.put_u8(b'\\');
        }

        copied = newline + 1;
        if slash_count % 2 == 0 {
            if line_end != newline {
                output.put_u8(b'\r');
            }
            output.put_u8(b'\n');
        } else {
            while source
                .get(copied)
                .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
            {
                copied += 1;
            }
            if !posix {
                while output
                    .last()
                    .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
                {
                    output.truncate(output.len() - 1);
                }
            }
            output.put_u8(b' ');
        }

        let Some(next) = memchr(b'\n', &source[copied..]) else {
            break;
        };
        newline = copied + next;
    }
    output.put_slice(&source[copied..]);
    output.freeze()
}

/// Whether `export` directives are allowed.
pub enum ExportAllowed {
    /// Export directives are allowed, the default.
    Allowed,
    /// Export directives result in warnings with the specified message.
    Warning(String),
    /// Export directives result in errors with the specified message.
    Error(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum FrameType {
    Root,       // Root node. Exactly one of this exists.
    Phase,      // Markers for various phases of the execution.
    Parse,      // Initial evaluation pass: include, := variables, etc.
    Call,       // Evaluating the result of a function call
    FunCall,    // Evaluating a function call (not its result)
    Statement,  // Denotes individual statements for better location reporting
    Dependency, // Dependency analysis. += requires variable expansion here.
    Exec,       // Execution phase. Expansion of = and rule-specific variables.
    Ninja,      // Ninja file generation
}

#[derive(Debug)]
pub struct Frame {
    frame_type: FrameType,
    parent: Option<Weak<Frame>>,
    name: Bytes,
    location: Option<Loc>,
    children: Mutex<Vec<Arc<Frame>>>,
}

impl Frame {
    fn new(
        frame_type: FrameType,
        parent: Option<Arc<Frame>>,
        loc: Option<Loc>,
        name: Bytes,
    ) -> Self {
        assert!(parent.is_none() == (frame_type == FrameType::Root));
        Self {
            frame_type,
            parent: parent.map(|p| Arc::downgrade(&p)),
            name,
            location: loc,
            children: Mutex::new(Vec::new()),
        }
    }

    fn add(&self, child: Arc<Frame>) {
        self.children.lock().push(child);
    }

    fn print_json_trace(
        &self,
        names: &impl Interner,
        tf: &mut dyn std::io::Write,
        indent: usize,
    ) -> Result<()> {
        if self.frame_type == FrameType::Root {
            return Ok(());
        }

        let indent_string = " ".repeat(indent);
        let mut desc = String::from_utf8_lossy(&self.name);
        if let Some(loc) = &self.location {
            desc = Cow::Owned(format!("{desc} @ {}", loc.display(names)));
        }

        let parent = self.parent.clone().unwrap().upgrade();
        let comma = if parent
            .clone()
            .is_some_and(|p| p.frame_type == FrameType::Root)
        {
            ""
        } else {
            ","
        };
        writeln!(tf, "{indent_string}\"{desc}\"{comma}")?;
        if let Some(parent) = parent {
            parent.print_json_trace(names, tf, indent)?;
        }
        Ok(())
    }
}

pub struct ScopedFrame {
    stack: Arc<Mutex<Vec<Arc<Frame>>>>,
    frame: Option<Arc<Frame>>,
}

impl ScopedFrame {
    fn new(stack: Arc<Mutex<Vec<Arc<Frame>>>>, frame: Option<Arc<Frame>>) -> Self {
        if let Some(frame) = frame.clone() {
            let mut stack = stack.lock();
            stack.last().unwrap().add(frame.clone());
            stack.push(frame);
        }
        Self { stack, frame }
    }
    pub fn current(&self) -> Option<Arc<Frame>> {
        self.frame.clone()
    }
}

impl Drop for ScopedFrame {
    fn drop(&mut self) {
        if let Some(frame) = &self.frame {
            let mut stack = self.stack.lock();
            let last = stack.pop().unwrap();
            assert!(last.name == frame.name);
            assert!(last.location == frame.location);
        }
    }
}

#[derive(Default)]
struct IncludeGraphNode {
    includes: BTreeSet<Bytes>,
}

struct IncludeGraph {
    nodes: HashMap<Bytes, IncludeGraphNode>,
    include_stack: Vec<Arc<Frame>>,
}

/// One Makefile the read reached, and whether it had to be there.
///
/// `include` says the file must exist; `-include` and `sinclude` say it need
/// not, and GNU Make carries that indifference into the rule that would have
/// made it — read.c records the goaldep with `RM_DONTCARE`, and main.c reports
/// `Failed to remake makefile` only for one it cares about. A file reached both
/// ways is cared about, which is why a later required read raises this and no
/// later optional one lowers it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadMakefile {
    pub filename: Symbol,
    pub required: bool,
}

/// An included Makefile that was absent while this source unit was evaluated.
///
/// Evaluation continues so rules later in the Makefile can describe how to
/// generate it. Dependency analysis then turns buildable entries into
/// regeneration roots and forgets the rest, which is what GNU Make does with an
/// included file it cannot remake; the embedding frontend decides whether to
/// build those roots and evaluate the source again.
#[derive(Clone)]
pub struct MissingInclude {
    /// The name as the `include` line spelled it, after expansion. A glob that
    /// matched nothing is kept verbatim, because GNU Make goes on to look for a
    /// rule making a file of that name.
    pub filename: Symbol,
    /// Whether the directive was `include` rather than `-include` or
    /// `sinclude`, and therefore whether failing to remake it is fatal.
    pub required: bool,
    /// Where the directive was read, for the diagnostic that names it. Absent
    /// for a Makefile the command line named, which no line of any Makefile
    /// asked for and which GNU Make therefore reports under its own name.
    pub loc: Option<Loc>,
}

impl IncludeGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            include_stack: Vec::new(),
        }
    }

    fn dump_json(&self, tf: &mut dyn std::io::Write) -> Result<()> {
        writeln!(tf, "{{")?;
        write!(tf, "  \"include_graph\": [")?;
        let mut first_node = true;

        for (file, node) in &self.nodes {
            if first_node {
                first_node = false;
                writeln!(tf)?;
            } else {
                writeln!(tf, ",")?;
            }

            writeln!(tf, "    {{")?;
            // TODO(lberki): Quote all these strings properly
            writeln!(tf, "      \"file\": \"{}\",", String::from_utf8_lossy(file))?;
            write!(tf, "      \"includes\": [")?;
            let mut first_include = true;
            for include in &node.includes {
                if first_include {
                    first_include = false;
                    writeln!(tf)?;
                } else {
                    writeln!(tf, ",")?;
                }

                write!(tf, "        \"{}\"", String::from_utf8_lossy(include))?;
            }
            writeln!(tf, "\n      ]")?;
            write!(tf, "    }}")?;
        }
        writeln!(tf, "\n  ]")?;
        writeln!(tf, "}}")?;

        Ok(())
    }

    fn merge_tree_node(&mut self, frame: &Arc<Frame>) {
        if frame.frame_type == FrameType::Parse {
            self.nodes.entry(frame.name.clone()).or_default();

            if let Some(parent_frame) = self.include_stack.last()
                && let Some(parent_node) = self.nodes.get_mut(&parent_frame.name)
            {
                parent_node.includes.insert(frame.name.clone());
            }

            self.include_stack.push(frame.clone());
        }

        for child in &*frame.children.lock() {
            self.merge_tree_node(child);
        }

        if frame.frame_type == FrameType::Parse {
            self.include_stack.pop();
        }
    }
}

/// Make evaluation, over a session it owns.
///
/// Every diagnostic raised during evaluation renders a symbol or a location, so
/// the session has to be reachable from here; that is why it lives on the
/// evaluator rather than beside it.
// [spec:ronin:req:make.no-ambient-state]
pub struct Evaluator {
    /// Everything that used to be a process global.
    pub session: Session,

    pub rule_vars: HashMap<Symbol, Arc<Vars>>,
    /// The pattern keys of `rule_vars`, in the order their first assignment was
    /// read. GNU Make keeps its pattern variables in one list and applies every
    /// entry that matches a target, so which of two equally specific patterns
    /// wins is decided by which was written first; a `HashMap` alone cannot say.
    pub pattern_rule_var_order: Vec<Symbol>,
    pub rules: Vec<Rule>,
    symbols_for_eval: HashSet<Symbol>,

    rule_state: RuleState,
    /// Whether `.SECONDEXPANSION` has been read yet. It applies only to rules
    /// below the declaration, so this is a position in the file rather than a
    /// property of the Makefile.
    second_expansion: bool,
    pub current_scope: Option<Arc<Vars>>,
    /// How many `$(shell)` environments are being built right now.
    ///
    /// GNU Make's `env_recursion`. An exported variable whose value contains a
    /// `$(shell)` asks for its own value to answer the call that produces it,
    /// because answering the call means building an environment and the
    /// variable is in it. While this is nonzero that circle is broken rather
    /// than reported.
    environment_recursion: usize,
    /// Names whose expansion was answered from the invocation's environment
    /// instead of being entered, innermost last, so finishing one does not
    /// clear the guard belonging to the expansion it was nested inside.
    environment_substituted: Vec<Symbol>,

    pub loc: Option<Loc>,
    is_bootstrap: bool,
    is_commandline: bool,

    trace: bool,
    stack: Arc<Mutex<Vec<Arc<Frame>>>>,
    assignment_tracefile: Option<Box<dyn std::io::Write>>,
    assignment_sep: String,

    pub avoid_io: bool,
    /// Where the selected destination resolves `$?`.
    pub(crate) new_inputs_timing: NewInputsTiming,
    /// Who the selected destination lets answer a `$(shell)` in a recipe.
    pub(crate) shell_evaluation: ShellEvaluation,
    /// `filter-out` patterns applied to the deferred `$?` marker while the
    /// current recipe is expanded.
    pub(crate) deferred_new_inputs_filter_out: Vec<Bytes>,
    // This value tracks the nest level of make expressions. For
    // example, $(YYY) in $(XXX $(YYY)) is evaluated with depth==2.
    // This will be used to disallow $(shell) in other make constructs.
    pub eval_depth: i32,
    // Commands which should run at ninja-time (i.e., info, warning, and
    // error).
    pub delayed_output_commands: Vec<Bytes>,

    is_posix: bool,

    /// Whether `export`/`unexport` directives are allowed.
    pub export_allowed: ExportAllowed,

    pub profiled_files: Vec<OsString>,

    /// Missing `include` and `-include` inputs, in source order.
    pub(crate) missing_includes: Vec<MissingInclude>,

    /// Every Makefile the read reached, in the order it reached them, whether
    /// or not it was there.
    ///
    /// GNU Make treats each one as a target it tries to bring up to date before
    /// it chooses a goal, so this is the list dependency analysis consults to
    /// plan Makefile remaking. It is not `MAKEFILE_LIST`: that variable is the
    /// Makefile's to read and to overwrite, and it never names a file the read
    /// could not open.
    pub(crate) read_makefiles: Vec<ReadMakefile>,

    /// What the build this evaluation describes is aimed at: the goals the
    /// invocation named, or the single goal `.DEFAULT_GOAL` held once the read
    /// was over when it named none.
    ///
    /// Dependency analysis resolves it and leaves it here because the answer is
    /// not recoverable afterwards from the graph's roots: a front end adds the
    /// Makefiles it has to remake to those, and those are not goals.
    pub goals: Vec<Symbol>,

    /// Whether the rules have been taken out of this evaluator and turned into
    /// a graph, so that recording another one could no longer change anything.
    ///
    /// GNU Make's `snapped_deps`, and it means the same thing there: once
    /// `snap_deps` has resolved the rules it read into files, an `$(eval)`
    /// reached afterwards — from a recipe, or from a second expansion — may
    /// still assign variables but may not record a rule, and one that tries is
    /// fatal rather than quietly ineffective (Savannah bug #12124).
    pub(crate) rules_snapped: bool,

    pub is_evaluating_command: bool,
    /// Whether expanding the current recipe referenced `MAKE`.
    ///
    /// GNU Make treats that reference as recursive intent even when it arrived
    /// through another recursively-expanded variable. A graph sink needs the
    /// same semantic fact before the expanded text becomes an opaque shell
    /// script.
    pub expanded_make_in_command: Vec<Bytes>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(Session::new())
    }
}

impl Interner for Evaluator {
    fn symtab(&self) -> &Symtab {
        &self.session.symtab
    }
}

impl Context for Evaluator {
    fn flags(&self) -> &Flags {
        &self.session.flags
    }
    fn stats(&self) -> &StatsRegistry {
        &self.session.stats
    }
}

impl Evaluator {
    /// Apply GNU Make's special post-assignment treatment of `MAKEFLAGS`.
    ///
    /// A normal recursive variable keeps the bytes assigned to it. GNU Make
    /// instead reads these bytes as switches, mutates a persistent option
    /// table, and immediately writes the canonical table back. The embedding
    /// frontend supplies the grammar; kati supplies the exact assignment
    /// boundary and preserves the variable's provenance.
    fn normalize_makeflags_assignment(
        &mut self,
        lhs: Symbol,
        variable: &Var,
        assigned: bool,
    ) -> Result<()> {
        if !assigned || lhs.as_bytes(&self.session).as_ref() != b"MAKEFLAGS" {
            return Ok(());
        }
        let Some((decoder, previous, protected, has_overrides)) = self
            .session
            .flags
            .makeflags_assignment
            .as_ref()
            .map(|state| {
                (
                    state.decoder,
                    state.effective.clone(),
                    state.protected.clone(),
                    state.has_overrides,
                )
            })
        else {
            return Ok(());
        };

        let value = self.eval_var(lhs)?;
        let decoded = decoder(&previous, &value, &protected).map_err(anyhow::Error::msg)?;
        let overrides = self.session.intern("MAKEOVERRIDES");
        let (value, original) =
            makeflags_value(decoded.makeflags.clone(), has_overrides, overrides);
        variable.write().replace_recursive_value(value, original);

        let mflags = self.session.intern("MFLAGS");
        let mflags_value = Arc::new(Value::Literal(None, decoded.mflags.clone()));
        self.session.globals.define(
            mflags,
            Variable::new_recursive(
                mflags_value,
                VarOrigin::Environment,
                None,
                None,
                decoded.mflags.clone(),
            ),
        );

        self.session.flags.is_dry_run = decoded.is_dry_run;
        self.session.flags.is_silent_mode = decoded.is_silent_mode;
        self.session.flags.ignore_errors = decoded.ignore_errors;
        self.session.flags.environment_overrides = decoded.environment_overrides;
        self.session.flags.no_builtin_rules = decoded.no_builtin_rules;
        self.session.flags.no_builtin_variables = decoded.no_builtin_variables;
        if let Some(state) = &mut self.session.flags.makeflags_assignment {
            state.effective = decoded.makeflags;
        }
        Ok(())
    }

    pub fn new(session: Session) -> Self {
        let trace = session.flags.dump_variable_assignment_trace.is_some()
            || session.flags.dump_include_graph.is_some();
        Self {
            session,
            rule_vars: HashMap::new(),
            pattern_rule_var_order: Vec::new(),
            rules: Vec::new(),
            symbols_for_eval: HashSet::new(),

            rule_state: RuleState::None,
            second_expansion: false,
            current_scope: None,

            loc: None,
            is_bootstrap: false,
            is_commandline: false,

            trace,
            stack: Arc::new(Mutex::new(vec![Arc::new(Frame::new(
                FrameType::Root,
                None,
                None,
                Bytes::from_static(b"*root*"),
            ))])),
            assignment_tracefile: None,
            assignment_sep: "\n".to_string(),

            avoid_io: false,
            new_inputs_timing: NewInputsTiming::RecipeShell,
            shell_evaluation: ShellEvaluation::RecipeShell,
            deferred_new_inputs_filter_out: Vec::new(),
            eval_depth: 0,
            delayed_output_commands: Vec::new(),

            is_posix: false,

            environment_recursion: 0,
            environment_substituted: Vec::new(),
            export_allowed: ExportAllowed::Allowed,

            profiled_files: Vec::new(),

            missing_includes: Vec::new(),
            read_makefiles: Vec::new(),
            goals: Vec::new(),

            rules_snapped: false,
            is_evaluating_command: false,
            expanded_make_in_command: Vec::new(),
        }
    }

    pub fn start(&mut self) -> Result<()> {
        let Some(filename) = self.session.flags.dump_variable_assignment_trace.clone() else {
            return Ok(());
        };
        let filename = filename.as_os_str();

        if filename == "-" {
            self.assignment_tracefile = Some(Box::new(std::io::stderr()));
        } else {
            let f = std::fs::File::create(filename)
                .map_err(|err| crate::io_failure(std::path::Path::new(filename), &err))?;
            let w = BufWriter::new(f);
            self.assignment_tracefile = Some(Box::new(w));
        }

        let tf = self.assignment_tracefile.as_mut().unwrap();
        writeln!(tf, "{{")?;
        write!(tf, "  \"assignments\": [")?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if let Some(tf) = self.assignment_tracefile.as_mut() {
            write!(tf, " \n ]\n")?;
            writeln!(tf, "}}")?;
        }
        Ok(())
    }

    pub fn in_bootstrap(&mut self) {
        self.is_bootstrap = true;
        self.is_commandline = false;
    }

    pub fn in_command_line(&mut self) {
        self.is_bootstrap = false;
        self.is_commandline = true;
    }

    pub fn in_toplevel_makefile(&mut self) {
        self.is_bootstrap = false;
        self.is_commandline = false;
    }

    /// Snapshot command-line bindings in the recursive environment form GNU
    /// Make makes visible again while expanding a recipe after `override
    /// undefine` removed the makefile-scope value.
    pub fn capture_command_line_environment(&mut self) {
        let variables = self
            .session
            .globals
            .matching(|var| var.read().origin() == VarOrigin::CommandLine);
        for (name, variable) in variables {
            let recursive = variable.read().clone_for_recipe_environment();
            self.session.recipe_command_line.define(name, recursive);
        }
    }

    pub fn current_frame(&self) -> Arc<Frame> {
        self.stack.lock().last().unwrap().clone()
    }

    /// Run `f` with the rule scope lifted off the lookup chain, so an expansion
    /// reads what the makefile reads at this point and nothing the scope being
    /// built holds. Restores the scope even though the caller still needs it to
    /// assign into: for a pattern-specific assignment GNU Make separates the
    /// two roles, expanding in the ambient scope while defining into the
    /// pattern's own set.
    fn in_ambient_scope<T>(
        &mut self,
        ambient: bool,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if !ambient {
            return f(self);
        }
        let saved = self.current_scope.take();
        let result = f(self);
        self.current_scope = saved;
        result
    }

    /// `ambient_value` expands the right-hand side with the current rule scope
    /// lifted off. It says nothing about where the result is defined, or about
    /// where `+=` and `?=` read the value they build on: those stay in the
    /// scope, because a second assignment to the same pattern accumulates onto
    /// the first.
    pub fn eval_rhs(
        &mut self,
        lhs: Symbol,
        rhs_v: Arc<Value>,
        orig_rhs: Bytes,
        op: AssignOp,
        is_override: bool,
        ambient_value: bool,
    ) -> Result<(Var, bool)> {
        let (origin, current_frame) = if self.is_bootstrap {
            (VarOrigin::Default, None)
        } else if self.is_commandline {
            (VarOrigin::CommandLine, None)
        } else if is_override {
            (VarOrigin::Override, self.stack.lock().last().cloned())
        } else {
            (VarOrigin::File, self.stack.lock().last().cloned())
        };

        let result: Var;
        let prev: Option<Var>;
        let mut needs_assign = true;

        match op {
            AssignOp::ColonEq => {
                prev = self.peek_var_in_current_scope(lhs);
                let loc = self.loc.clone();
                result = self.in_ambient_scope(ambient_value, |ev| {
                    Variable::with_simple_value(origin, current_frame, loc, ev, &rhs_v)
                })?;
            }
            AssignOp::ImmediateRecursive => {
                prev = self.peek_var_in_current_scope(lhs);
                let expanded = self.in_ambient_scope(ambient_value, |ev| rhs_v.eval_to_buf(ev))?;
                let mut escaped = Vec::with_capacity(expanded.len());
                for byte in expanded {
                    if byte == b'$' {
                        escaped.push(byte);
                    }
                    escaped.push(byte);
                }
                let escaped = Bytes::from(escaped);
                let mut loc = self.loc.clone().unwrap_or_default();
                let value = parse_expr(
                    &mut self.session,
                    &mut loc,
                    escaped.clone(),
                    ParseExprOpt::Normal,
                )?;
                result = Variable::new_recursive(
                    value,
                    origin,
                    current_frame,
                    self.loc.clone(),
                    escaped,
                );
            }
            // `V != cmd` runs the command the way `$(shell)` does, down to
            // `.SHELLSTATUS`, then reads its output as a recursive value.
            AssignOp::ShellEq => {
                prev = self.peek_var_in_current_scope(lhs);
                let ran = Value::Func {
                    loc: self.loc.clone().unwrap_or_default(),
                    fi: &crate::func::SHELL_ASSIGNMENT,
                    args: vec![rhs_v],
                };
                let output = self.in_ambient_scope(ambient_value, |ev| ran.eval_to_buf(ev))?;
                let mut loc = self.loc.clone().unwrap_or_default();
                let value = parse_expr(
                    &mut self.session,
                    &mut loc,
                    output.clone(),
                    ParseExprOpt::Normal,
                )?;
                result =
                    Variable::new_recursive(value, origin, current_frame, self.loc.clone(), output);
            }
            AssignOp::Eq => {
                prev = self.peek_var_in_current_scope(lhs);
                result = Variable::new_recursive(
                    rhs_v,
                    origin,
                    current_frame,
                    self.loc.clone(),
                    orig_rhs,
                );
            }
            AssignOp::PlusEq => {
                prev = self.lookup_var_in_current_scope(lhs)?;
                if let Some(prev) = prev.clone() {
                    if prev.read().readonly {
                        error_loc!(
                            self,
                            self.loc.as_ref(),
                            "*** cannot assign to readonly variable: {}",
                            lhs.display(self)
                        );
                    }
                    // What `+=` has to paste on. A simple variable's value was
                    // expanded when it was set, so the right-hand side is
                    // expanded now to match it; a recursive one is appended to
                    // unexpanded, so what counts is the text as written.
                    // `V := x` with `V += $(EMPTY)` therefore has nothing to
                    // append, while `V = x` with that same line appends the
                    // reference and expands it later.
                    let appended = if prev.read().immediate_eval() {
                        Some(self.in_ambient_scope(ambient_value, |ev| rhs_v.eval_to_buf(ev))?)
                    } else {
                        None
                    };
                    let nothing_to_append = match &appended {
                        Some(expanded) => expanded.is_empty(),
                        None => orig_rhs.is_empty(),
                    };
                    if nothing_to_append {
                        // Appending nothing is not an assignment at all. GNU
                        // Make hands back the variable it looked up without
                        // redefining it, so no separator arrives, and the
                        // origin and location it already had are the ones it
                        // keeps.
                        needs_assign = false;
                        result = prev;
                    } else {
                        // `+=` reads what is in scope and defines the result
                        // where an ordinary assignment would, which is why
                        // precedence is decided on the copy rather than on what
                        // it was read from. A `foreach` or `call` binding is
                        // read through exactly like anything else: GNU Make
                        // pastes onto the loop word and defines the answer in
                        // the global set the binding is standing in front of,
                        // which is where the copy goes too.
                        result = prev.read().clone_for_assignment(
                            origin,
                            current_frame,
                            self.loc.clone(),
                        );
                        let frame = self.current_frame();
                        if let Some(expanded) = appended {
                            result.write().append_str(&self.session, &expanded, frame)?;
                        } else {
                            result.write().append_var(
                                &self.session,
                                rhs_v,
                                &orig_rhs,
                                frame,
                                self.loc.as_ref(),
                            )?;
                        }
                    }
                } else {
                    result = Variable::new_recursive(
                        rhs_v,
                        origin,
                        current_frame,
                        self.loc.clone(),
                        orig_rhs,
                    );
                }
            }
            AssignOp::QuestionEq => {
                prev = self.lookup_var_in_current_scope(lhs)?;
                if let Some(prev) = prev.clone() {
                    result = prev;
                    needs_assign = false;
                } else {
                    result = Variable::new_recursive(
                        rhs_v,
                        origin,
                        current_frame,
                        self.loc.clone(),
                        orig_rhs,
                    );
                }
            }
        }

        if let Some(prev) = prev {
            let prev = prev.read();
            prev.used(self, &lhs)?;
            if needs_assign && let Some(deprecated) = &prev.deprecated {
                result.write().deprecated = Some(deprecated.clone());
            }
        }

        Ok((result, needs_assign))
    }

    pub fn eval_assign(&mut self, stmt: &AssignStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;
        let lhs = stmt.get_lhs_symbol(self)?;

        if lhs == Symbol::KATI_READONLY {
            let rhs = stmt.rhs.eval_to_buf(self)?;
            for name in word_scanner(&rhs) {
                let name = self.session.intern(rhs.slice_ref(name));
                let Some(var) = self.session.get_global_var(name) else {
                    error_loc!(
                        self,
                        self.loc.as_ref(),
                        "*** unknown variable: {}",
                        name.display(self)
                    );
                };
                var.write().readonly = true;
            }
            return Ok(());
        }

        let is_override = stmt.directive.map(|v| v.is_override).unwrap_or(false);
        // GNU Make redefines a global rather than replacing it, and nothing
        // clears `private`: once a name is private here, every later assignment
        // to it is too. A target-specific one is settled per assignment.
        let is_private = stmt.directive.is_some_and(|v| v.is_private)
            || self
                .session
                .peek_global_var(lhs)
                .is_some_and(|var| var.read().is_private);
        let (var, needs_assign) = self.eval_rhs(
            lhs,
            stmt.rhs.clone(),
            stmt.orig_rhs.clone(),
            stmt.op,
            is_override,
            false,
        )?;
        if needs_assign {
            var.write().assign_op = Some(stmt.op);
            let mut readonly = false;
            self.session
                .set_global_var(lhs, var.clone(), is_override, Some(&mut readonly))?;
            if readonly {
                error_loc!(
                    self,
                    self.loc.as_ref(),
                    "*** cannot assign to readonly variable: {}",
                    lhs.display(self)
                );
            }
        }

        if is_private {
            var.write().is_private = true;
        }
        // `export NAME = value` and `unexport NAME = value` are one definition
        // carrying one answer, which is why the answer lands on the variable
        // the definition produced rather than on the name.
        if let Some(directive) = stmt.directive
            && directive.export != VarExport::Default
        {
            var.write().export = directive.export;
            self.check_export_allowed(lhs, directive.export == VarExport::Export)?;
        }
        if stmt.is_final {
            var.write().readonly = true
        }
        self.normalize_makeflags_assignment(lhs, &var, needs_assign)?;
        self.trace_variable_assign(&lhs, &var)?;
        Ok(())
    }

    // With rule broken into
    //   <before_term> <term> <after_term>
    // parses <before_term> into Symbol instances until encountering ':'
    // Returns the remainder of <before_term>.
    fn parse_rule_targets(
        session: &mut Session,
        loc: &Loc,
        before_term: &Bytes,
        delimiter: Option<RuleDelimiter>,
    ) -> Result<(Bytes, Vec<Symbol>, bool)> {
        let Some(delimiter) = delimiter else {
            error_loc!(session, Some(loc), "*** missing separator.");
        };
        let targets_end = delimiter.colon
            - usize::from(before_term.get(delimiter.colon.wrapping_sub(1)) == Some(&b'&'));
        let targets_string = before_term.slice(0..targets_end);
        let after = before_term.slice(delimiter.colon + 1..);
        let mut pattern_rule_count = 0;
        let mut targets: Vec<Symbol> = Vec::new();
        for word in makefile_word_scanner(&targets_string) {
            let target = word.slice_ref(trim_leading_curdir(&word));
            glob_word(session, target, &mut targets);
        }
        // The `%` is read off what the glob left, as GNU Make does.
        for target in &targets {
            if is_pattern_rule(&target.as_bytes(&*session)) {
                pattern_rule_count += 1;
            }
        }
        // Check consistency: either all outputs are patterns or none.
        if pattern_rule_count > 0 && pattern_rule_count != targets.len() {
            error_loc!(
                session,
                Some(loc),
                "*** mixed implicit and normal rules: deprecated syntax"
            );
        }
        Ok((after, targets, pattern_rule_count > 0))
    }

    fn eval_rule_word(&mut self, source: Bytes) -> Result<RuleWordExpansion> {
        let mut loc = self.loc.clone().unwrap_or_default();
        let value = parse_expr(&mut self.session, &mut loc, source, ParseExprOpt::Define)?;
        let mut expansion = RuleWordExpansion::default();
        match value.as_ref() {
            Value::List(_, values) => {
                for value in values {
                    match value.as_ref() {
                        Value::Literal(_, literal) => expansion.push_literal(literal),
                        _ => {
                            let value = value.eval_to_buf(self)?;
                            expansion.push_expansion(&value);
                        }
                    }
                }
            }
            Value::Literal(_, literal) => expansion.push_literal(literal),
            _ => {
                let value = value.eval_to_buf(self)?;
                expansion.push_expansion(&value);
            }
        }
        Ok(expansion)
    }

    /// Expand source words only through the one that produces the first rule
    /// colon. GNU Make leaves `rest` literal until it knows whether the line is
    /// a target-specific assignment.
    fn eval_rule_head(
        &mut self,
        source: &Bytes,
        detect_expanded_command: bool,
    ) -> Result<ExpandedRuleHead> {
        let mut output = BytesMut::new();
        let mut at = 0usize;
        let mut wrote_word = false;
        while let Some(word) = next_rule_word(source, at) {
            at = word.end;
            let mut expansion = self.eval_rule_word(source.slice(word.clone()))?;
            if wrote_word {
                output.put_u8(b' ');
            }
            let output_start = output.len();
            wrote_word = true;
            if detect_expanded_command
                && let Some((semicolon, _)) = expansion.find_char_unquote(b';')
            {
                let mut command = BytesMut::from(expansion.take_after(semicolon));
                let colon = expansion.find_char_unquote(b':');
                let expanded = expansion.finish();
                output.put_slice(&expanded);
                command.put_slice(&self.eval_rule_suffix(source.slice(at..))?);
                let delimiter = colon.map(|(colon, origin)| {
                    let colon = output_start + colon;
                    RuleDelimiter { colon, origin }
                });
                return Ok(ExpandedRuleHead {
                    text: output.freeze(),
                    rest: Bytes::new(),
                    delimiter,
                    expanded_command: Some(command.freeze()),
                    had_source_word: true,
                });
            }
            let colon = expansion.find_char_unquote(b':');
            let expanded = expansion.finish();
            output.put_slice(&expanded);
            if let Some((colon, origin)) = colon {
                let colon = output_start + colon;
                return Ok(ExpandedRuleHead {
                    text: output.freeze(),
                    rest: source.slice(at..),
                    delimiter: Some(RuleDelimiter { colon, origin }),
                    expanded_command: None,
                    had_source_word: true,
                });
            }
        }
        Ok(ExpandedRuleHead {
            text: output.freeze(),
            rest: Bytes::new(),
            delimiter: None,
            expanded_command: None,
            had_source_word: wrote_word,
        })
    }

    fn split_rule_source(source: &Bytes) -> (Bytes, Option<Bytes>) {
        let mut output = BytesMut::with_capacity(source.len());
        let mut at = 0usize;

        while at < source.len() {
            if source[at] == b'$' {
                let end = skip_make_reference(source, at + 1);
                output.put_slice(&source[at..end]);
                at = end;
                continue;
            }

            if source[at] == b'\\' {
                let run_start = at;
                while source.get(at) == Some(&b'\\') {
                    at += 1;
                }
                let slash_count = at - run_start;
                if source
                    .get(at)
                    .is_some_and(|byte| matches!(byte, b';' | b'#'))
                {
                    for _ in 0..slash_count / 2 {
                        output.put_u8(b'\\');
                    }
                    if slash_count % 2 == 1 {
                        output.put_u8(source[at]);
                        at += 1;
                        continue;
                    }
                } else {
                    output.put_slice(&source[run_start..at]);
                    continue;
                }
            }

            match source[at] {
                b'#' => return (output.freeze(), None),
                b';' => return (output.freeze(), Some(source.slice(at + 1..))),
                byte => output.put_u8(byte),
            }
            at += 1;
        }

        (output.freeze(), None)
    }

    fn hybrid_value(
        &mut self,
        hybrid: &HybridRuleText,
        range: Range<usize>,
        literal_suffix: Option<usize>,
    ) -> Result<Arc<Value>> {
        let mut values = Vec::new();
        let parsed_end = literal_suffix.unwrap_or(range.end).min(range.end);
        if range.start < parsed_end {
            let mut loc = self.loc.clone().unwrap_or_default();
            values.push(parse_expr(
                &mut self.session,
                &mut loc,
                hybrid.text.slice(range.start..parsed_end),
                ParseExprOpt::Define,
            )?);
        }
        if let Some(suffix_start) = literal_suffix {
            let suffix_start = range.start.max(suffix_start);
            if suffix_start < range.end {
                let mut loc = self.loc.clone().unwrap_or_default();
                // A literal semicolon ends comment recognition before GNU
                // Make appends it and its suffix to a target-variable value.
                // `Define` retains `#` as value text while still parsing `$`
                // references according to the assignment's flavor.
                values.push(parse_expr(
                    &mut self.session,
                    &mut loc,
                    hybrid.text.slice(suffix_start..range.end),
                    ParseExprOpt::Define,
                )?);
            }
        }
        Ok(match values.len() {
            0 => Arc::new(Value::Literal(None, Bytes::new())),
            1 => values.pop().unwrap(),
            _ => Arc::new(Value::List(self.loc.clone(), values)),
        })
    }

    /// Parse the hybrid text GNU Make presents to `parse_var_assignment`: the
    /// expanded remainder of the colon-bearing source word followed by the
    /// untouched source suffix.
    fn parse_rule_assignment(
        &mut self,
        candidate: HybridRuleText,
        literal_command: Option<&Bytes>,
    ) -> Result<Option<RuleAssignment>> {
        let Some(scanned) = scan_rule_assignment(&candidate.text) else {
            return Ok(None);
        };
        if scanned.definition.name.is_empty() {
            error_loc!(self, self.loc.as_ref(), "*** empty variable name.");
        }

        let mut definition = BytesMut::from(candidate.text.as_ref());
        let literal_suffix = literal_command.map(|_| definition.len());
        if let Some(command) = literal_command {
            definition.put_u8(b';');
            definition.put_slice(&collapse_rule_continuations(command.clone(), self.is_posix));
        }
        let definition = HybridRuleText {
            text: definition.freeze(),
        };
        let name_range = scanned.definition.name;
        let mut rhs_start = scanned.definition.value_start;
        let is_final = definition.text[rhs_start..].starts_with(b"$=");
        if is_final {
            rhs_start = skip_make_space(&definition.text, rhs_start + 2);
        }
        let rhs_range = rhs_start..definition.text.len();

        Ok(Some(RuleAssignment {
            name: self.hybrid_value(&definition, name_range, None)?,
            rhs: self.hybrid_value(&definition, rhs_range.clone(), literal_suffix)?,
            orig_rhs: definition.text.slice(rhs_range),
            op: scanned.definition.op,
            modifiers: scanned.modifiers,
            is_final,
        }))
    }

    fn eval_rule_suffix(&mut self, suffix: Bytes) -> Result<Bytes> {
        if suffix.is_empty() {
            return Ok(suffix);
        }
        let mut loc = self.loc.clone().unwrap_or_default();
        parse_expr(&mut self.session, &mut loc, suffix, ParseExprOpt::Define)?.eval_to_buf(self)
    }

    /// A recipe written on the rule's own line, after a `;`.
    ///
    /// It is continued the same way a recipe line is, onto lines that carry the
    /// recipe prefix, so the prefix comes off them here for the same reason it
    /// comes off a recipe line's continuations.
    fn parse_rule_command(&mut self, source: Bytes, recipe_prefix: u8) -> Result<Arc<Value>> {
        let source = source.slice_ref(trim_left_space(&source));
        let source = strip_recipe_prefix_continuations(source, recipe_prefix);
        let mut loc = self.loc.clone().unwrap_or_default();
        parse_expr(&mut self.session, &mut loc, source, ParseExprOpt::Command)
    }

    // Strip leading spaces and trailing spaces and colons.
    pub fn format_rule_error(before_term: &[u8]) -> String {
        let before_term = String::from_utf8_lossy(before_term).into_owned();
        if before_term.is_empty() {
            return before_term;
        }
        before_term
            .trim_ascii_start()
            .trim_end_matches(|c: char| c.is_ascii_whitespace() || c == ':')
            .to_string()
    }

    pub fn mark_vars_readonly(&mut self, vars_list: &Value) -> Result<()> {
        let vars_list_string = vars_list.eval_to_buf(self)?;
        let scope = self.current_scope.clone().unwrap();
        for name in word_scanner(&vars_list_string) {
            let name = self.session.intern(vars_list_string.slice_ref(name));
            let Some(var) = scope.lookup(&mut self.session.used_env_vars, name) else {
                error_loc!(
                    self,
                    self.loc.as_ref(),
                    "*** unknown variable: {}",
                    name.display(self)
                );
            };
            var.write().readonly = true;
        }
        Ok(())
    }

    /// Read `tgt: VAR = value`, expanding both halves in `tgt`'s own scope and
    /// putting back whatever scope that interrupted.
    ///
    /// Ordinarily it interrupts nothing, because a rule is read between
    /// statements rather than inside one. A recipe's `$(eval tgt: VAR = value)`
    /// is the exception: it arrives while the recipe's own target scope is
    /// installed, and dropping that scope instead of restoring it would leave
    /// every later line of the recipe reading globals where GNU Make still
    /// reads the target's variables.
    fn eval_rule_specific_assign(
        &mut self,
        targets: &[Symbol],
        assignment: RuleAssignment,
        is_pattern_rule: bool,
    ) -> Result<()> {
        let interrupted = self.current_scope.take();
        let result = self.record_rule_specific_assign(targets, assignment, is_pattern_rule);
        self.current_scope = interrupted;
        result
    }

    fn record_rule_specific_assign(
        &mut self,
        targets: &[Symbol],
        assignment: RuleAssignment,
        is_pattern_rule: bool,
    ) -> Result<()> {
        let modifiers = assignment.modifiers;
        for target in targets {
            let fresh = !self.rule_vars.contains_key(target);
            let scope = self
                .rule_vars
                .entry(*target)
                .or_insert_with(|| Arc::new(Vars::new()))
                .clone();
            if fresh && is_pattern_rule {
                self.pattern_rule_var_order.push(*target);
            }

            let name = if is_pattern_rule {
                assignment.name.eval_to_buf(self)?
            } else {
                self.current_scope = Some(scope.clone());
                assignment.name.eval_to_buf(self)?
            };
            if name.is_empty() {
                error_loc!(self, self.loc.as_ref(), "*** empty variable name.");
            }
            let var_sym = self.session.intern(name);
            // Whether this `+=` has finished, read before the assignment lands.
            // It has if it appended to a value this scope already settled: that
            // value replaced whatever the target inherits, so appending again at
            // build time would put the inherited one back. A run of `+=` with
            // nothing but each other to build on is still only the tail of a
            // value — it stays pending, and the build appends the whole run to
            // what the target inherits.
            let settled_in_scope = assignment.op == AssignOp::PlusEq
                && scope
                    .peek(var_sym)
                    .is_some_and(|prev| prev.read().assign_op != Some(AssignOp::PlusEq));
            self.current_scope = Some(scope);
            if var_sym == Symbol::KATI_READONLY {
                self.mark_vars_readonly(&assignment.rhs)?;
            } else {
                let (rhs_var, needs_assign) = self.eval_rhs(
                    var_sym,
                    assignment.rhs.clone(),
                    assignment.orig_rhs.clone(),
                    assignment.op,
                    modifiers.directive.is_override,
                    is_pattern_rule,
                )?;
                if needs_assign {
                    let mut readonly = false;
                    rhs_var.write().assign_op = Some(if settled_in_scope {
                        AssignOp::Eq
                    } else {
                        assignment.op
                    });
                    self.current_scope.as_ref().unwrap().assign(
                        var_sym,
                        rhs_var.clone(),
                        &mut readonly,
                    )?;
                    if readonly {
                        error_loc!(
                            self,
                            self.loc.as_ref(),
                            "*** cannot assign to readonly variable: {}",
                            var_sym.display(self)
                        );
                    }
                }
                if modifiers.directive.is_private {
                    rhs_var.write().is_private = true;
                }
                // A target-specific `export` reaches this target's recipe and
                // every recipe that reaches it as a prerequisite, because the
                // binding it marks is the one those scopes inherit.
                if modifiers.directive.export != VarExport::Default {
                    rhs_var.write().export = modifiers.directive.export;
                }
                if assignment.is_final {
                    rhs_var.write().readonly = true;
                }
            }
            self.current_scope = None
        }
        Ok(())
    }

    pub fn eval_rule(&mut self, stmt: &RuleStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;

        let (source, literal_command) = Self::split_rule_source(&stmt.orig());
        let source = collapse_rule_continuations(source, self.is_posix);
        let ExpandedRuleHead {
            text: before_term,
            rest,
            delimiter,
            expanded_command,
            had_source_word,
        } = self.eval_rule_head(&source, literal_command.is_none())?;
        // See semicolon.mk.
        if before_term.iter().all(|c| b" \t\n".contains(c)) {
            if literal_command.is_some() && !had_source_word {
                error_loc!(self, self.loc.as_ref(), "*** missing rule before commands.");
            }
            return Ok(());
        }

        let Some(delimiter) = delimiter else {
            let loc = self.loc.clone().unwrap();
            Evaluator::parse_rule_targets(&mut self.session, &loc, &before_term, None)?;
            unreachable!();
        };
        debug_assert_eq!(before_term[delimiter.colon], b':');
        log!("Rule colon: {:?}", delimiter.origin);
        let is_grouped = before_term.get(delimiter.colon.wrapping_sub(1)) == Some(&b'&');

        let loc = self.loc.clone().unwrap();
        let (mut after_targets, targets, is_pattern_rule) =
            Evaluator::parse_rule_targets(&mut self.session, &loc, &before_term, Some(delimiter))?;
        if targets.is_empty() {
            self.rule_state = RuleState::Ignored;
            return Ok(());
        }
        let is_double_colon = after_targets.starts_with(b":");
        if is_double_colon {
            after_targets.advance(1);
        }

        let mut candidate = BytesMut::with_capacity(after_targets.len() + rest.len());
        candidate.put_slice(&after_targets);
        candidate.put_slice(&rest);
        let candidate = HybridRuleText {
            text: candidate.freeze(),
        };
        if let Some(assignment) = self.parse_rule_assignment(candidate, literal_command.as_ref())? {
            return self.eval_rule_specific_assign(&targets, assignment, is_pattern_rule);
        }

        // Past this point the line is a rule, and a rule read after the graph
        // was compiled has nowhere to go. GNU Make refuses it here for the same
        // reason and in the same place — this is `record_files`' first act, and
        // it comes after the target-specific assignment above, which is why
        // `$(eval all: LOCAL = x)` in a recipe is accepted while
        // `$(eval all: extra)` is not.
        if self.rules_snapped {
            error_loc!(
                self,
                self.loc.as_ref(),
                "*** prerequisites cannot be defined in recipes."
            );
        }

        let scan_expanded_command = !rest.is_empty();
        let mut rest = rest.to_vec();
        find_char_unquote(&mut rest, b'=');
        let expanded_rest = self.eval_rule_suffix(Bytes::from(rest))?;
        let mut prerequisites = Vec::with_capacity(after_targets.len() + expanded_rest.len());
        prerequisites.extend_from_slice(&after_targets);
        prerequisites.extend_from_slice(&expanded_rest);
        let recipe_prefix = stmt.recipe_prefix;
        let command = if let Some(command) = literal_command {
            Some(self.parse_rule_command(command, recipe_prefix)?)
        } else if let Some(command) = expanded_command {
            Some(self.parse_rule_command(command, recipe_prefix)?)
        } else if scan_expanded_command {
            if let Some(semicolon) = find_char_unquote(&mut prerequisites, b';') {
                let command = Bytes::from(prerequisites.split_off(semicolon + 1));
                prerequisites.truncate(semicolon);
                Some(self.parse_rule_command(command, recipe_prefix)?)
            } else {
                None
            }
        } else {
            None
        };
        let prerequisites = Bytes::from(prerequisites);

        if !is_pattern_rule
            && targets
                .iter()
                .any(|t| t.as_bytes(&self.session).as_ref() == b".SECONDEXPANSION")
        {
            self.second_expansion = true;
        }

        let mut rule = Rule::new(self.loc.clone().unwrap(), is_double_colon, is_grouped);
        rule.expand_again = self.second_expansion;
        if is_pattern_rule {
            rule.output_patterns = targets;
        } else {
            rule.outputs = targets;
        }
        rule.parse_prerequisites(&mut self.session, &prerequisites)?;
        if let Some(command) = command {
            rule.cmds.push(command);
        }

        for o in &rule.outputs {
            // `.POSIX:` is read where it stands rather than collected with the
            // other special targets: it changes what the rest of the Makefile
            // reads, so a later `$(CC)` has to see `c99` and an earlier one
            // must not.
            if o == &Symbol::POSIX && !self.is_posix {
                self.is_posix = true;
                crate::builtins::install_posix_variables(&mut self.session);
            }
        }
        self.record_default_goal(&rule.outputs)?;

        log!("Rule: {:?}", rule);
        match self.get_allow_rules()? {
            RulesAllowed::Warning => {
                warn_loc!(
                    self,
                    self.loc.as_ref(),
                    "warning: Rule not allowed here for target: {}",
                    Evaluator::format_rule_error(&before_term)
                );
            }
            RulesAllowed::Error => {
                error_loc!(
                    self,
                    self.loc.as_ref(),
                    "*** Rule not allowed here for target: {}",
                    Evaluator::format_rule_error(&before_term),
                );
            }
            RulesAllowed::Allowed => {}
        }
        self.rules.push(rule);
        self.rule_state = RuleState::Active;
        Ok(())
    }

    /// Whether `.DEFAULT_GOAL` still holds the empty text a read starts from.
    ///
    /// The global binding and no other: a target-specific `.DEFAULT_GOAL` is a
    /// value that recipe sees, not a statement about what the build aims at.
    fn default_goal_unset(&self) -> bool {
        self.session
            .peek_global_var(Symbol::DEFAULT_GOAL)
            .is_none_or(|var| var.read().text_is_empty())
    }

    /// Offer a rule's targets to `.DEFAULT_GOAL`, as GNU Make offers every
    /// target it records.
    ///
    /// Selection is armed only while the variable's text is empty, and that
    /// single condition is the whole of three behaviours: the first eligible
    /// target of the read wins; a Makefile that assigns the variable takes the
    /// choice away from every target after it; and one that assigns it empty
    /// again hands the choice to the next target recorded.
    ///
    /// Not every target is eligible. A pattern ends the offer rather than
    /// skipping it — a rule that says how to make things names none. A name
    /// Make reserved is passed over while the targets beside it are still
    /// considered, and reserved means a leading dot with no directory
    /// separator after it: `.PHONY` is a declaration, `.deps/x.Po` is a file.
    fn record_default_goal(&mut self, outputs: &[Symbol]) -> Result<()> {
        if !self.default_goal_unset() {
            return Ok(());
        }
        for output in outputs {
            let name = output.as_bytes(&self.session);
            if name.contains(&b'%') {
                return Ok(());
            }
            if name.starts_with(b".") && !name.contains(&b'/') {
                continue;
            }
            if self.suffixes_reject_default_goal(&name) {
                continue;
            }
            let frame = self.current_frame();
            let loc = self.loc.clone();
            return self.session.set_global_var(
                Symbol::DEFAULT_GOAL,
                Variable::with_simple_string(name, VarOrigin::File, Some(frame), loc),
                false,
                None,
            );
        }
        Ok(())
    }

    /// Whether `.SUFFIXES` as the read stands disqualifies this name from
    /// becoming the default goal.
    fn suffixes_reject_default_goal(&self, name: &[u8]) -> bool {
        suffixes_reject_default_goal(name, &self.declared_suffixes())
    }

    /// The suffixes `.SUFFIXES` names as the read stands right now.
    ///
    /// GNU Make reads `suffix_file->deps` at the moment each target is
    /// recorded, so a `.SUFFIXES:` line below the rule has not happened yet and
    /// one above it has. Replayed from the rules read so far rather than
    /// tracked, because the shape of the answer is the shape of the replay: a
    /// bare `.SUFFIXES:` clears the list and every other one adds to whatever
    /// survived, so only the order they were read in can say what is on it.
    ///
    /// The same replay `DepBuilder::read_suffix_list` does once the read is
    /// over. It cannot serve here — the default goal is chosen during the read,
    /// and by the time dependency analysis settles the list the answer is long
    /// since given.
    fn declared_suffixes(&self) -> Vec<Bytes> {
        let mut declared: Vec<Symbol> = Vec::new();
        for rule in &self.rules {
            if !rule.outputs.contains(&Symbol::SUFFIXES) {
                continue;
            }
            if rule.inputs.is_empty() {
                declared.clear();
            } else {
                declared.extend(&rule.inputs);
            }
        }
        declared
            .iter()
            .map(|suffix| suffix.as_bytes(&self.session))
            .collect()
    }

    pub fn eval_command(&mut self, stmt: &CommandStmt) -> Result<()> {
        self.loc = Some(stmt.loc());

        if self.rule_state == RuleState::Ignored {
            return Ok(());
        }
        if self.rule_state == RuleState::None {
            let stmts = parse_buf_no_stats(&mut self.session, &stmt.orig(), stmt.loc())?;
            let stmts = stmts.lock();
            for a in &*stmts {
                a.eval(self)?;
            }
            return Ok(());
        }

        let last_rule = self.rules.last_mut().unwrap();
        last_rule.cmds.push(stmt.expr.clone());
        if last_rule.cmd_loc.is_none() {
            last_rule.cmd_loc = Some(stmt.loc());
        }
        log!("Command: {:?}", stmt.expr);

        Ok(())
    }

    pub fn eval_if(&mut self, stmt: &IfStmt) -> Result<()> {
        self.loc = Some(stmt.loc());

        let is_true = match stmt.op {
            CondOp::Ifdef | CondOp::Ifndef => {
                let var_name = stmt.lhs.eval_to_buf(self)?;
                let lhs = trim_right_space(&var_name);
                if lhs.iter().any(is_space_byte) {
                    error_loc!(
                        self,
                        self.loc.as_ref(),
                        "*** invalid syntax in conditional."
                    );
                }
                let lhs = self.session.intern(var_name.slice_ref(lhs));
                if let Some(v) = self.lookup_var_in_current_scope(lhs)? {
                    let v = v.read();
                    v.used(self, &lhs)?;
                    v.string(&self.session)?.is_empty() == (stmt.op == CondOp::Ifndef)
                } else {
                    stmt.op == CondOp::Ifndef
                }
            }
            CondOp::Ifeq | CondOp::Ifneq => {
                let lhs = stmt.lhs.eval_to_buf(self)?;
                let rhs = stmt
                    .rhs
                    .as_ref()
                    .map(|v| v.eval_to_buf(self))
                    .unwrap_or_else(|| Ok(Bytes::new()))?;
                (lhs == rhs) == (stmt.op == CondOp::Ifeq)
            }
        };

        let stmts = match is_true {
            true => &stmt.true_stmts,
            false => &stmt.false_stmts,
        };
        let stmts = stmts.lock();
        for a in stmts.iter() {
            log!("{:?}", a);
            a.eval(self)?;
        }
        Ok(())
    }

    /// Evaluate `fname` as part of the file including it.
    ///
    /// `required` is the difference between `include` and `-include`: a
    /// Makefile that wrote the second has said it does not care whether the
    /// file is there or readable, and GNU Make reports nothing when it is
    /// neither.
    pub fn do_include(&mut self, fname: &Bytes, required: bool) -> Result<()> {
        let filename = OsString::from_vec(fname.to_vec());
        collect_stats_with_slow_report!(self, "included makefiles", &filename);

        let mk = match self.session.get_makefile(&filename)? {
            Source::Read(mk) => mk,
            Source::Absent => error_loc!(
                self,
                self.loc.as_ref(),
                "{} does not exist",
                filename.to_string_lossy()
            ),
            Source::Unreadable(_) if !required => return Ok(()),
            // The system's own reason, at the `include` that asked for the
            // file: an `io::Error` reaching the user on its own names neither.
            Source::Unreadable(err) => error_loc!(
                self,
                self.loc.as_ref(),
                "{}: {}",
                filename.to_string_lossy(),
                crate::strerror(&err)
            ),
        };

        let v = fname.slice_ref(trim_leading_curdir(fname));
        self.note_read_makefile(v.clone(), required);
        self.note_makefile_list(v)?;
        let stmts = mk.stmts.lock().clone();
        for stmt in stmts {
            log!("{stmt:?}");
            stmt.eval(self)?;
        }

        if !self.profiled_files.is_empty() {
            for mk in std::mem::take(&mut self.profiled_files) {
                crate::stats::mark_interesting(self, "included makefiles", mk);
            }
        }
        Ok(())
    }

    /// Where `-I` found the file, or the name as written.
    ///
    /// Only when the working directory does not have it: GNU Make searches the
    /// path after failing, not instead of looking.
    fn at_include_dirs(&self, pat: Bytes) -> Bytes {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        use std::path::Path;

        if self.session.flags.include_dirs.is_empty()
            || pat.starts_with(b"/")
            || Path::new(OsStr::from_bytes(&pat)).exists()
        {
            return pat;
        }
        for dir in &self.session.flags.include_dirs {
            let candidate = dir.join(OsStr::from_bytes(&pat));
            if candidate.exists() {
                return Bytes::from(candidate.into_os_string().into_vec());
            }
        }
        pat
    }

    /// Add a Makefile that opened to `MAKEFILE_LIST`, as GNU Make does.
    ///
    /// The name goes on immediately before the file's statements run, so a
    /// Makefile reading the variable sees itself as the last entry. A file
    /// named twice is listed twice, because the variable records the reads and
    /// not the set of files: the same name reached the evaluator twice.
    pub(crate) fn note_makefile_list(&mut self, name: Bytes) -> Result<()> {
        if let Some(var_list) = self.lookup_var(Symbol::MAKEFILE_LIST)? {
            let frame = self.current_frame();
            var_list.write().append_str(&self.session, &name, frame)?;
            return Ok(());
        }
        let frame = self.current_frame();
        let loc = self.loc.clone();
        self.session.set_global_var(
            Symbol::MAKEFILE_LIST,
            Variable::with_simple_string(name, VarOrigin::File, Some(frame), loc),
            false,
            None,
        )
    }

    /// Record that the read reached this Makefile, so remaking can consider it.
    ///
    /// The order is the order GNU Make attempts them in, and a file included
    /// twice is one target, so a repeat is dropped rather than appended.
    pub(crate) fn note_read_makefile(&mut self, filename: Bytes, required: bool) {
        let filename = self.session.intern(filename);
        if let Some(existing) = self
            .read_makefiles
            .iter_mut()
            .find(|makefile| makefile.filename == filename)
        {
            existing.required |= required;
            return;
        }
        self.read_makefiles
            .push(ReadMakefile { filename, required });
    }

    pub(crate) fn note_missing_include(
        &mut self,
        filename: Bytes,
        required: bool,
        loc: Option<Loc>,
    ) {
        self.note_read_makefile(filename.clone(), required);
        let filename = self.session.intern(filename);
        if let Some(existing) = self
            .missing_includes
            .iter_mut()
            .find(|include| include.filename == filename)
        {
            existing.required |= required;
            return;
        }
        self.missing_includes.push(MissingInclude {
            filename,
            required,
            loc,
        });
    }

    pub fn eval_include(&mut self, stmt: &IncludeStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;

        let pats = stmt.expr.eval_to_buf(self)?;
        for pat in word_scanner(&pats) {
            let pat = pats.slice_ref(pat);
            let pat = self.at_include_dirs(pat);
            let files = self.session.glob(pat.clone());

            let missing = match files.as_ref() {
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
                Err(err) => {
                    if stmt.should_exist {
                        error_loc!(
                            self,
                            self.loc.as_ref(),
                            "{}: {}",
                            String::from_utf8_lossy(&pat),
                            crate::strerror(err)
                        );
                    }
                    continue;
                }
                Ok(files) => files.is_empty(),
            };
            if missing {
                self.note_missing_include(pat, stmt.should_exist, Some(stmt.loc()));
                continue;
            }
            let Ok(files) = files.as_ref() else {
                continue;
            };
            let files = files.clone();

            for fname in &files {
                if !stmt.should_exist
                    && self
                        .session
                        .flags
                        .ignore_optional_include_pattern
                        .as_ref()
                        .map(|p| p.matches(fname))
                        .unwrap_or(false)
                {
                    continue;
                }

                {
                    let _frame = self.enter(FrameType::Parse, fname.clone(), stmt.loc());
                    let included_from = format!(
                        "In file included from {}:",
                        stmt.loc().display(&self.session)
                    );
                    anyhow::Context::with_context(
                        self.do_include(fname, stmt.should_exist),
                        || included_from,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Record where to look for prerequisites a `vpath` directive covers.
    ///
    /// Three forms, told apart by how many words the line expands to. No words
    /// clears every search path; one word clears the paths for that pattern
    /// alone; two or more give a pattern and the directories to search for it.
    ///
    /// A repeated pattern extends rather than replaces, which is GNU Make's
    /// rule and the reason the paths are a list keyed by pattern rather than a
    /// map: `vpath %.c foo` then `vpath %.c bar` searches foo before bar.
    pub fn eval_vpath(&mut self, stmt: &VpathStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;

        let line = stmt.expr.eval_to_buf(self)?;
        let mut words = word_scanner(&line);
        let Some(pattern) = words.next() else {
            self.session.vpaths.clear();
            return Ok(());
        };
        let pattern = line.slice_ref(pattern);
        // Directories are separated by whitespace or by colons, and GNU Make
        // accepts both in one directive.
        let directories = words
            .flat_map(|word| word.split(|byte| *byte == b':'))
            .filter(|directory| !directory.is_empty())
            .map(|directory| line.slice_ref(directory))
            .collect::<Vec<_>>();
        if directories.is_empty() {
            self.session
                .vpaths
                .retain(|(existing, _)| existing.as_bytes() != &pattern);
            return Ok(());
        }
        let pattern = Pattern::new(pattern);
        if let Some((_, existing)) = self
            .session
            .vpaths
            .iter_mut()
            .find(|(candidate, _)| candidate.as_bytes() == pattern.as_bytes())
        {
            existing.extend(directories);
        } else {
            self.session.vpaths.push((pattern, directories));
        }
        Ok(())
    }

    /// `undefine name`, which removes the binding rather than emptying it, so
    /// `$(origin)` and `$(flavor)` answer `undefined` afterwards.
    ///
    /// The whole expanded line is one name — GNU Make does not split it into
    /// words — and it outranks what defined the variable only when the
    /// directive carried `override`.
    pub fn eval_undefine(&mut self, stmt: &UndefineStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;

        let name = stmt.expr.eval_to_buf(self)?;
        if name.is_empty() {
            error_loc!(self, self.loc.as_ref(), "*** empty variable name.");
        }
        let sym = self.session.intern(name);
        self.session.undefine_global_var(sym, stmt.is_override)
    }

    pub fn eval_export(&mut self, stmt: &ExportStmt) -> Result<()> {
        self.loc = Some(stmt.loc());
        self.rule_state = RuleState::None;

        // A directive that named nothing speaks for every variable. GNU Make
        // writes the same flag `.EXPORT_ALL_VARIABLES` writes, and writes it
        // here while the makefile is being read, so the last bare directive
        // wins over the ones before it — and the target, read after the whole
        // makefile is, wins over all of them.
        if stmt.is_bare {
            self.session.flags.export_all_variables = stmt.is_export;
            return Ok(());
        }
        let exports = stmt.expr.eval_to_buf(self)?;
        for tok in word_scanner(&exports) {
            let equal_index = memchr(b'=', tok);
            let lhs;
            if equal_index == Some(0)
                || (equal_index == Some(1)
                    && (tok.starts_with(b":") || tok.starts_with(b"?") || tok.starts_with(b"+")))
            {
                // Do not export tokens after an assignment.
                break;
            } else if let Some(equal_index) = equal_index {
                let assign = parse_assign_statement(tok, equal_index);
                lhs = assign.lhs;
            } else {
                lhs = tok;
            }
            let sym = self.session.intern(exports.slice_ref(lhs));
            self.mark_exported(sym, stmt.is_export)?;
            self.check_export_allowed(sym, stmt.is_export)?;
        }
        Ok(())
    }

    /// Report an `export` a `$(KATI_deprecate_export)` or
    /// `$(KATI_obsolete_export)` earlier in the read asked to hear about.
    ///
    /// Every way of saying it is one of these, so this is asked wherever the
    /// attribute is set rather than only where the directive spells out a list
    /// of names — `export NAME := value` is an export like any other.
    fn check_export_allowed(&mut self, name: Symbol, is_export: bool) -> Result<()> {
        let prefix = if is_export { "" } else { "un" };
        match &self.export_allowed {
            ExportAllowed::Allowed => {}
            ExportAllowed::Error(msg) => error_loc!(
                self,
                self.loc.as_ref(),
                "*** {}: {prefix}export is obsolete{msg}.",
                name.display(self)
            ),
            ExportAllowed::Warning(msg) => warn_loc!(
                self,
                self.loc.as_ref(),
                "{}: {prefix}export has been deprecated{msg}.",
                name.display(self)
            ),
        }
        Ok(())
    }

    /// Record what an `export NAME` or `unexport NAME` directive said.
    ///
    /// GNU Make looks the name up and, finding nothing, defines it as an empty
    /// file-origin variable so there is something to hang the answer on. That
    /// is observable through `$(origin)`, so it happens here too.
    fn mark_exported(&mut self, name: Symbol, is_export: bool) -> Result<()> {
        let attribute = if is_export {
            VarExport::Export
        } else {
            VarExport::NoExport
        };
        if let Some(var) = self.session.peek_global_var(name) {
            var.write().export = attribute;
            return Ok(());
        }
        let frame = Some(self.current_frame());
        let var =
            Variable::with_simple_string(Bytes::new(), VarOrigin::File, frame, self.loc.clone());
        var.write().export = attribute;
        self.session.set_global_var(name, var, false, None)
    }

    pub fn lookup_var_global(&mut self, name: Symbol) -> Option<Var> {
        let v = self.session.get_global_var(name);
        if v.is_none() {
            self.session.note_undefined_var(name);
        }
        v
    }

    pub fn is_traced(&self, name: &Symbol) -> bool {
        if self.assignment_tracefile.is_none() {
            return false;
        }

        // trace every variable unless filtered
        if self.session.flags.traced_variables_pattern.is_empty() {
            return true;
        }

        let name = name.as_bytes(&self.session);
        for pat in self.session.flags.traced_variables_pattern.iter() {
            if pat.matches(&name) {
                return true;
            }
        }
        false
    }

    pub fn trace_variable_lookup(
        &mut self,
        operation: &'static str,
        name: &Symbol,
        var: &Option<Var>,
    ) -> Result<()> {
        if !self.is_traced(name) {
            return Ok(());
        }
        let current_frame = self.current_frame();
        let name_str = name.display(&self.session).to_string();
        let session = &self.session;
        let sep = std::mem::replace(&mut self.assignment_sep, ",\n".to_string());
        let Some(tf) = self.assignment_tracefile.as_mut() else {
            return Ok(());
        };
        write!(tf, "{sep}")?;
        writeln!(tf, "    {{")?;
        writeln!(tf, "      \"name\": \"{name_str}\",")?;
        writeln!(tf, "      \"operation\": \"{operation}\",")?;
        writeln!(tf, "      \"defined\": {},", var.is_some())?;
        writeln!(tf, "      \"reference_stack\": [")?;
        current_frame.print_json_trace(session, tf, 8)?;
        writeln!(tf, "      ]")?;
        write!(tf, "    }}")?;
        Ok(())
    }

    pub fn trace_variable_assign(&mut self, name: &Symbol, var: &Var) -> Result<()> {
        if !self.is_traced(name) {
            return Ok(());
        }
        let name_str = name.display(&self.session).to_string();
        let session = &self.session;
        let sep = std::mem::replace(&mut self.assignment_sep, ",\n".to_string());
        let Some(tf) = self.assignment_tracefile.as_mut() else {
            return Ok(());
        };
        write!(tf, "{sep}")?;
        writeln!(tf, "    {{")?;
        writeln!(tf, "      \"name\": \"{name_str}\",")?;
        writeln!(tf, "      \"operation\": \"assign\",")?;
        write!(tf, "      \"value\": \"{var:?}\"")?;
        if let Some(definition) = var.read().definition().clone() {
            writeln!(tf, ",\n")?;
            writeln!(tf, "      \"value_stack\": [")?;
            definition.print_json_trace(session, tf, 8)?;
            writeln!(tf, "      ]")?;
        }
        write!(tf, "    }}")?;
        Ok(())
    }

    pub fn lookup_var_for_eval(&mut self, name: Symbol) -> Result<Option<Var>> {
        if let Some(var) = self.lookup_var(name)? {
            if self.symbols_for_eval.contains(&name) {
                // A variable waiting on itself is an error, except while a
                // `$(shell)`'s environment is being built: there it is what an
                // exported variable holding a `$(shell)` unavoidably does, and
                // GNU Make answers it with the bytes the invocation carried
                // rather than refusing the makefile.
                if self.environment_recursion > 0 {
                    self.environment_substituted.push(name);
                    return Ok(Some(self.inherited_binding(name)));
                }
                let loc = var.read().loc().clone();
                error_loc!(
                    self,
                    loc.as_ref(),
                    "*** Recursive variable \"{}\" references itself (eventually).",
                    name.display(self)
                );
            }
            self.symbols_for_eval.insert(name);
            return Ok(Some(var));
        }
        Ok(None)
    }

    pub fn var_eval_complete(&mut self, name: Symbol) {
        if let Some(position) = self
            .environment_substituted
            .iter()
            .rposition(|substituted| *substituted == name)
        {
            self.environment_substituted.remove(position);
            return;
        }
        self.symbols_for_eval.remove(&name);
    }

    /// What this name held in the environment the invocation was started with,
    /// as a binding an expansion can read.
    fn inherited_binding(&mut self, name: Symbol) -> Var {
        let inherited = crate::export::invocation_value(self, &name.as_bytes(&self.session))
            .map(Bytes::from)
            .unwrap_or_default();
        Variable::with_simple_string(inherited, VarOrigin::Environment, None, None)
    }

    /// Expand one exported variable's value for a child's environment, under
    /// GNU Make's `env_recursion` guard when the child is a `$(shell)`.
    ///
    /// # Errors
    ///
    /// Whatever expanding the value rejects.
    pub(crate) fn expand_for_environment(
        &mut self,
        name: Symbol,
        var: &Var,
        guarded: bool,
    ) -> Result<Bytes> {
        if guarded && self.symbols_for_eval.contains(&name) {
            return Ok(self
                .inherited_binding(name)
                .read()
                .string(&self.session)?
                .into_owned()
                .into());
        }
        let entered = self.symbols_for_eval.insert(name);
        if guarded {
            self.environment_recursion += 1;
        }
        let value = var
            .read()
            .eval_to_buf_mut(self)
            .map(bytes::BytesMut::freeze);
        if guarded {
            self.environment_recursion -= 1;
        }
        if entered {
            self.symbols_for_eval.remove(&name);
        }
        value
    }

    pub fn lookup_var(&mut self, name: Symbol) -> Result<Option<Var>> {
        let mut result = None;

        if let Some(current_scope) = self.current_scope.clone() {
            result = current_scope.lookup(&mut self.session.used_env_vars, name);
        }

        if result.is_none() {
            let global = self.lookup_var_global(name);
            result = global.clone();
            // A rule's scope reaches the global one through a parent, which is
            // the boundary `private` refuses to cross: from inside a rule the
            // variable is not there at all, so `$(origin)` says undefined.
            if self.current_scope.is_some() {
                result = result.filter(|var| !var.read().is_private);
            }
            if global.is_none() && self.is_evaluating_command {
                result = self.session.recipe_command_line.peek(name);
            }
        }

        self.trace_variable_lookup("lookup", &name, &result)?;
        Ok(result)
    }

    pub fn peek_var(&self, name: Symbol) -> Option<Var> {
        let mut result = None;

        if let Some(current_scope) = &self.current_scope {
            result = current_scope.peek(name);
        }

        if result.is_none() {
            result = self.session.peek_global_var(name);
        }

        result
    }

    pub fn lookup_var_in_current_scope(&mut self, name: Symbol) -> Result<Option<Var>> {
        let result = if let Some(current_scope) = self.current_scope.clone() {
            current_scope.lookup(&mut self.session.used_env_vars, name)
        } else {
            self.lookup_var_global(name)
        };

        self.trace_variable_lookup("scope lookup", &name, &result)?;
        Ok(result)
    }

    pub fn peek_var_in_current_scope(&self, name: Symbol) -> Option<Var> {
        if let Some(current_scope) = &self.current_scope {
            current_scope.peek(name)
        } else {
            self.session.peek_global_var(name)
        }
    }

    pub fn eval_var(&mut self, name: Symbol) -> Result<Bytes> {
        if let Some(var) = self.lookup_var(name)? {
            var.read().eval_to_buf(self)
        } else {
            Ok(Bytes::new())
        }
    }

    pub fn enter(&mut self, frame_type: FrameType, name: Bytes, loc: Loc) -> ScopedFrame {
        if !self.trace {
            return ScopedFrame::new(self.stack.clone(), None);
        }

        let parent = self.stack.lock().last().cloned();
        let frame = Frame::new(frame_type, parent, Some(loc), name);
        ScopedFrame::new(self.stack.clone(), Some(Arc::new(frame)))
    }

    /// Whether a `$(shell)` reached from here is written into the recipe for
    /// the recipe's own shell to answer, rather than answered now.
    ///
    /// Only a recipe being compiled for a manifest defers: outside one there is
    /// no recipe to write into, and a destination that runs the build itself
    /// asked for GNU Make's answer.
    pub fn defers_shell_to_the_recipe(&self) -> bool {
        self.avoid_io && self.shell_evaluation == ShellEvaluation::RecipeShell
    }

    pub fn get_shell(&mut self) -> Result<Bytes> {
        self.eval_var(Symbol::SHELL)
    }

    /// Whether the read saw `.POSIX:` as a target, which is GNU Make's
    /// `posix_pedantic`.
    ///
    /// Read after the read has finished as well as during it: the switch is
    /// latched by the target wherever it stands, and what it then governs
    /// includes decisions — the suffix-rule conversion among them — that GNU
    /// Make does not make until the last makefile is closed.
    pub fn is_posix(&self) -> bool {
        self.is_posix
    }

    /// The flags one recipe line's shell is invoked with.
    ///
    /// `dash_prefixed` is whether the line was written with a leading `-`, and
    /// nothing else: `-i` and `.IGNORE` also ignore a failure, but GNU Make
    /// reads only `lines_flags` here (`job.c` hands
    /// `cmds->lines_flags[command_line - 1]` to `construct_command_argv`), so
    /// they leave `-e` in place. The difference shows: under `.POSIX:` a
    /// recipe ignored by `.IGNORE` still stops at its first failing command,
    /// while the same line prefixed `-` runs to the end.
    ///
    /// An explicit `.SHELLFLAGS` outranks all of it. `.POSIX:` installs `-ec`
    /// as a *default*, so a Makefile that assigns the variable — before or
    /// after `.POSIX:`, globally or for one target — keeps its own value and
    /// gets no `-e` back.
    pub fn get_shell_flag(&mut self, dash_prefixed: bool) -> Result<Bytes> {
        let Some(var) = self.lookup_var(Symbol::SHELLFLAGS)? else {
            return Ok(Bytes::new());
        };
        let is_default = var.read().origin() == VarOrigin::Default;
        if self.is_posix && is_default {
            return Ok(Bytes::from_static(if dash_prefixed {
                b"-c"
            } else {
                b"-ec"
            }));
        }
        var.read().eval_to_buf(self)
    }

    fn get_allow_rules(&mut self) -> Result<RulesAllowed> {
        Ok(match self.eval_var(Symbol::KATI_ALLOW_RULES)?.as_ref() {
            b"warning" => RulesAllowed::Warning,
            b"error" => RulesAllowed::Error,
            _ => RulesAllowed::Allowed,
        })
    }

    pub fn dump_include_json(&self, filename: &OsStr) -> Result<()> {
        let mut graph = IncludeGraph::new();
        graph.merge_tree_node(self.stack.lock().first().unwrap());
        let mut w: Box<dyn std::io::Write> = if filename == OsStr::new("-") {
            Box::new(std::io::stdout())
        } else {
            let f = std::fs::File::create(filename)
                .map_err(|err| crate::io_failure(std::path::Path::new(filename), &err))?;
            Box::new(BufWriter::new(f))
        };

        graph.dump_json(&mut w)?;
        Ok(())
    }

    /// Bind `sym` to `var` for the duration of `f`, then put back whatever the
    /// symbol was bound to before — including nothing at all.
    ///
    /// This is the session-owned form of the explicit save and restore that
    /// replaced the scope-restoring `Drop` guard. The scope is reached at the
    /// save and at the restore and never handed to `f`: `f` re-enters
    /// evaluation and reaches the scope for itself, so a borrow of it held
    /// across the body would be the evaluator's own. Reborrowing `self` into
    /// the closure is legal where handing out a `&mut` to a field is not.
    ///
    /// The restore runs on the error path as well as the value path, which is
    /// the one thing `Drop` was genuinely buying: Make evaluation propagates
    /// errors straight out of `foreach` and `call`, so a body that fails
    /// partway must not leave an automatic variable bound behind it. A panic
    /// unwinding out of `f` does not restore, unlike `Drop`; evaluation reports
    /// every failure it is meant to survive as an `Err`, and the session a
    /// panic would corrupt does not outlive it.
    ///
    /// "Whatever the symbol was bound to before" is a slot rather than a value:
    /// an assignment written inside the body, which can only reach the session
    /// through `$(eval)`, is a write to the global binding and lands in that
    /// slot without disturbing what the body reads. See
    /// [`crate::var::GlobalVars::bind`].
    // [spec:ronin:req:make.no-ambient-state]
    pub fn with_bound<T>(
        &mut self,
        sym: Symbol,
        var: Var,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.session.globals.bind(sym, var);
        let result = f(self);
        self.session.globals.unbind();
        result
    }

    /// [`Self::with_bound`] for several symbols at once: all bound before `f`,
    /// all restored after it, in reverse order so nesting still holds if a
    /// symbol were ever to repeat. `$(call)` needs this — it binds every
    /// positional argument around a single body.
    pub fn with_bounds<T>(
        &mut self,
        bindings: Vec<(Symbol, Var)>,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let count = bindings.len();
        for (sym, var) in bindings {
            self.session.globals.bind(sym, var);
        }
        let result = f(self);
        for _ in 0..count {
            self.session.globals.unbind();
        }
        result
    }

    pub fn used_undefined_vars(&self) -> HashSet<Symbol> {
        self.session.used_undefined_vars.clone()
    }
}

/// Whether a declared suffix disqualifies `name` from becoming the default
/// goal.
///
/// GNU Make's `check_specials` passes over a target whose name is exactly a
/// declared suffix, and one whose name is two of them joined — and then
/// considers the next target rather than stopping.
///
/// The first test is guarded by the suffix not beginning with a dot and the
/// second is not, which is what makes the built-in list matter. Every built-in
/// suffix is dotted, so on its own the list can only reject a name that begins
/// with a dot and was rejected a line earlier anyway; it earns its keep in
/// company. `foo.c` under `.SUFFIXES: foo` is `foo` followed by the built-in
/// `.c` and is passed over, and under `-r` — where there is no built-in list to
/// join to — the same target is chosen.
///
/// Both loops range over the whole list, so a suffix joined to itself counts.
fn suffixes_reject_default_goal(name: &[u8], suffixes: &[Bytes]) -> bool {
    for suffix in suffixes {
        if !suffix.starts_with(b".") && name == suffix.as_ref() {
            return true;
        }
        for first in suffixes {
            if name
                .strip_prefix(first.as_ref())
                .is_some_and(|rest| rest == suffix.as_ref())
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::var::VarOrigin;

    /// The asymmetry between the two tests is the whole of this rule, and it is
    /// invisible in any makefile that does not declare an undotted suffix.
    #[test]
    fn a_declared_suffix_rejects_a_default_goal_of_its_own_name_only_undotted() {
        let undotted = [Bytes::from_static(b"foo")];
        assert!(suffixes_reject_default_goal(b"foo", &undotted));
        let dotted = [Bytes::from_static(b".foo")];
        assert!(!suffixes_reject_default_goal(b".foo", &dotted));
        // The two-suffix test has no such guard, which is what lets a dotted
        // built-in reject in company with an undotted declaration.
        let mixed = [Bytes::from_static(b"foo"), Bytes::from_static(b".c")];
        assert!(suffixes_reject_default_goal(b"foo.c", &mixed));
        assert!(!suffixes_reject_default_goal(b"foo.c", &undotted));
        // Both loops range over the whole list, so a suffix joins to itself.
        assert!(suffixes_reject_default_goal(b"foofoo", &undotted));
        // And a name that is neither is left alone.
        assert!(!suffixes_reject_default_goal(b"zed", &mixed));
        assert!(!suffixes_reject_default_goal(b"foobar", &undotted));
    }

    #[test]
    fn target_variable_definition_uses_gnu_make_scanning_rules() {
        for text in [
            b"FOO#BAR=good".as_slice(),
            b"FOO:BAR=good",
            b"FOO BAR=good",
            b"private FOO:BAR=good",
            b"?#X=x",
        ] {
            assert!(
                scan_rule_assignment(text).is_none(),
                "{}",
                String::from_utf8_lossy(text)
            );
        }

        for (text, name, value, op) in [
            (
                b"FOO\\=BAR=x".as_slice(),
                b"FOO\\".as_slice(),
                b"BAR=x".as_slice(),
                AssignOp::Eq,
            ),
            (b"FOO:=x", b"FOO", b"x", AssignOp::ColonEq),
            (b"FOO::=x", b"FOO", b"x", AssignOp::ColonEq),
            (b"FOO:::=x", b"FOO", b"x", AssignOp::ImmediateRecursive),
            (b"FOO?=x", b"FOO", b"x", AssignOp::QuestionEq),
            (b"??=x", b"?", b"x", AssignOp::QuestionEq),
            (b"?+=x", b"?", b"x", AssignOp::PlusEq),
            (b"?!=x", b"?", b"x", AssignOp::ShellEq),
            (b"?:=x", b"?", b"x", AssignOp::ColonEq),
            (b"FOO+=x", b"FOO", b"x", AssignOp::PlusEq),
            (b"FOO!=x", b"FOO", b"x", AssignOp::ShellEq),
            (
                b"$(FOO:BAR=hidden)=visible",
                b"$(FOO:BAR=hidden)",
                b"visible",
                AssignOp::Eq,
            ),
        ] {
            let assignment = scan_rule_assignment(text)
                .unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(text)));
            assert_eq!(&text[assignment.definition.name], name);
            assert_eq!(&text[assignment.definition.value_start..], value);
            assert_eq!(assignment.definition.op, op);
        }

        let assignment = scan_rule_assignment(b"private export override NAME = value").unwrap();
        assert_eq!(
            &b"private export override NAME = value"[assignment.definition.name],
            b"NAME"
        );
        assert_eq!(assignment.modifiers.words, 3);
        assert!(assignment.modifiers.directive.is_private);
        assert_eq!(assignment.modifiers.directive.export, VarExport::Export);
        assert!(assignment.modifiers.directive.is_override);

        let assignment = scan_rule_assignment(b"private = value").unwrap();
        assert_eq!(&b"private = value"[assignment.definition.name], b"private");
        assert_eq!(assignment.modifiers, AssignModifiers::default());
    }

    #[test]
    fn rule_source_unquotes_escaped_semicolon_before_command() {
        let (source, command) = Evaluator::split_rule_source(&Bytes::from_static(
            br"all: E:=left\;middle ; tail # kept",
        ));
        assert_eq!(source, Bytes::from_static(b"all: E:=left;middle "));
        assert_eq!(command, Some(Bytes::from_static(b" tail # kept")));
    }

    #[test]
    fn rule_source_settles_comments_before_expression_parsing() {
        let (source, command) =
            Evaluator::split_rule_source(&Bytes::from_static(br"all: out\#name ; tail"));
        assert_eq!(source, Bytes::from_static(b"all: out#name "));
        assert_eq!(command, Some(Bytes::from_static(b" tail")));

        let (source, command) =
            Evaluator::split_rule_source(&Bytes::from_static(br"all: H:=a\\\#b ; tail # kept"));
        assert_eq!(source, Bytes::from_static(br"all: H:=a\#b "));
        assert_eq!(command, Some(Bytes::from_static(b" tail # kept")));
    }

    #[test]
    fn rule_expansion_keeps_the_first_colons_provenance() {
        let mut written_colon = RuleWordExpansion::default();
        written_colon.push_literal(b"all:");
        assert_eq!(
            written_colon.find_char_unquote(b':'),
            Some((3, RuleColonOrigin::Literal))
        );
        assert_eq!(written_colon.finish(), Bytes::from_static(b"all:"));

        let mut expanded_colon = RuleWordExpansion::default();
        expanded_colon.push_expansion(b"all:X=good");
        expanded_colon.push_literal(b":");
        assert_eq!(
            expanded_colon.find_char_unquote(b':'),
            Some((3, RuleColonOrigin::Expansion))
        );
        assert_eq!(expanded_colon.finish(), Bytes::from_static(b"all:X=good:"));

        let line = b"all$(COLON) $(NAME)=value";
        assert_eq!(next_rule_word(line, 0), Some(0..b"all$(COLON)".len()));
        assert_eq!(
            next_rule_word(line, b"all$(COLON)".len()),
            Some(b"all$(COLON) ".len()..b"all$(COLON) $(NAME)".len())
        );
        assert_eq!(next_rule_word(b"literal:$(TAIL)", 0), Some(0..7));
        assert_eq!(next_rule_word(b"$(RULE):", 0), Some(0..7));
        assert_eq!(next_rule_word(br"escaped\#comment: tail", 0), Some(0..16));
        assert_eq!(next_rule_word(b"shell!=value", 0), Some(0..6));

        let mut escaped_expanded_colon = RuleWordExpansion::default();
        escaped_expanded_colon.push_literal(br"target\");
        escaped_expanded_colon.push_expansion(b":");
        assert_eq!(escaped_expanded_colon.find_char_unquote(b':'), None);
        assert_eq!(
            escaped_expanded_colon.finish(),
            Bytes::from_static(b"target:")
        );

        let mut paired_expanded_colon = RuleWordExpansion::default();
        paired_expanded_colon.push_expansion(br"foo\\:");
        assert_eq!(
            paired_expanded_colon.find_char_unquote(b':'),
            Some((4, RuleColonOrigin::Expansion))
        );
        assert_eq!(
            paired_expanded_colon.finish(),
            Bytes::from_static(br"foo\:")
        );

        let mut expanded_semicolon = RuleWordExpansion::default();
        expanded_semicolon.push_expansion(br"all: one\\; command");
        assert_eq!(
            expanded_semicolon.find_char_unquote(b';'),
            Some((9, RuleColonOrigin::Expansion))
        );
        assert_eq!(
            expanded_semicolon.finish(),
            Bytes::from_static(br"all: one\; command")
        );

        assert_eq!(
            collapse_rule_continuations(Bytes::from_static(b"target:  one  \\\n  two"), false),
            Bytes::from_static(b"target:  one two")
        );
        assert_eq!(
            collapse_rule_continuations(Bytes::from_static(b"target:  one  \\\n  two"), true),
            Bytes::from_static(b"target:  one   two")
        );
        assert_eq!(
            collapse_rule_continuations(Bytes::from_static(b"target: one\\\r\n  two"), false),
            Bytes::from_static(b"target: one two")
        );
        assert_eq!(
            collapse_rule_continuations(Bytes::from_static(b" after  \\\n  continued"), true),
            Bytes::from_static(b" after   continued")
        );
        assert_eq!(
            collapse_rule_continuations(Bytes::from_static(b" after  \\\n  continued"), false),
            Bytes::from_static(b" after continued")
        );
    }

    fn automatic(value: &'static [u8]) -> Var {
        Variable::with_simple_string(Bytes::from_static(value), VarOrigin::Automatic, None, None)
    }

    fn string_of(session: &Session, var: Var) -> String {
        String::from_utf8(var.read().string(session).unwrap().into_owned()).unwrap()
    }

    /// Restoring the *absence* of a binding is a different case from restoring
    /// a value, and it is the one a body that fails partway has to get right.
    #[test]
    fn test_with_bound_restores_absence_on_error() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_BOUND_ABSENT");
        assert!(ev.session.peek_global_var(sym).is_none());

        let result: Result<()> = ev.with_bound(sym, automatic(b"inner"), |ev| {
            let bound = ev.session.peek_global_var(sym).unwrap();
            assert_eq!(string_of(&ev.session, bound), "inner");
            crate::error!("body failed")
        });

        assert!(result.is_err());
        assert!(ev.session.peek_global_var(sym).is_none());
    }

    /// The same on the error path when there was a binding to go back to.
    #[test]
    fn test_with_bound_restores_previous_on_error() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_BOUND_PREVIOUS");
        ev.session.globals.replace(sym, Some(automatic(b"outer")));

        let result: Result<()> = ev.with_bound(sym, automatic(b"inner"), |ev| {
            let bound = ev.session.peek_global_var(sym).unwrap();
            assert_eq!(string_of(&ev.session, bound), "inner");
            crate::error!("body failed")
        });

        assert!(result.is_err());
        let bound = ev.session.peek_global_var(sym).unwrap();
        assert_eq!(string_of(&ev.session, bound), "outer");
    }

    /// Every binding of a group is restored when the body fails, whether it had
    /// a previous value or none.
    #[test]
    fn test_with_bounds_restores_every_binding_on_error() {
        let mut ev = Evaluator::new(Session::new());
        let kept = ev.session.intern("KATI_TEST_BOUND_MANY_KEPT");
        let absent = ev.session.intern("KATI_TEST_BOUND_MANY_ABSENT");
        ev.session.globals.replace(kept, Some(automatic(b"outer")));
        assert!(ev.session.peek_global_var(absent).is_none());

        let result: Result<()> = ev.with_bounds(
            vec![(kept, automatic(b"inner")), (absent, automatic(b"inner"))],
            |ev| {
                let a = ev.session.peek_global_var(kept).unwrap();
                let b = ev.session.peek_global_var(absent).unwrap();
                assert_eq!(string_of(&ev.session, a), "inner");
                assert_eq!(string_of(&ev.session, b), "inner");
                crate::error!("body failed")
            },
        );

        assert!(result.is_err());
        let bound = ev.session.peek_global_var(kept).unwrap();
        assert_eq!(string_of(&ev.session, bound), "outer");
        assert!(ev.session.peek_global_var(absent).is_none());
    }

    fn from_file(value: &'static [u8]) -> Var {
        Variable::with_simple_string(Bytes::from_static(value), VarOrigin::File, None, None)
    }

    /// An assignment written inside a binding's body is a write to the global
    /// binding: the body goes on reading the bound value, and the assignment is
    /// what the name means once the binding unwinds.
    #[test]
    fn test_assignment_under_a_binding_lands_outside_it() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_BINDING_LANDING");
        ev.session.globals.replace(sym, Some(from_file(b"outer")));

        ev.with_bound(sym, automatic(b"loop-word"), |ev| {
            ev.session
                .set_global_var(sym, from_file(b"assigned"), false, None)?;
            let read = ev.session.peek_global_var(sym).unwrap();
            assert_eq!(string_of(&ev.session, read.clone()), "loop-word");
            assert_eq!(read.read().origin(), VarOrigin::Automatic);
            Ok(())
        })
        .unwrap();

        let after = ev.session.peek_global_var(sym).unwrap();
        assert_eq!(string_of(&ev.session, after.clone()), "assigned");
        assert_eq!(after.read().origin(), VarOrigin::File);
    }

    /// Two bindings of one name, and the assignment still reaches past both:
    /// the slot the *outermost* binding saved is the global one.
    #[test]
    fn test_assignment_under_nested_bindings_reaches_the_global() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_BINDING_NESTED");
        ev.session.globals.replace(sym, Some(from_file(b"outer")));

        ev.with_bound(sym, automatic(b"first"), |ev| {
            ev.with_bound(sym, automatic(b"second"), |ev| {
                ev.session
                    .set_global_var(sym, from_file(b"assigned"), false, None)
            })?;
            // The inner binding unwinds to the outer binding, not to the write.
            let read = ev.session.peek_global_var(sym).unwrap();
            assert_eq!(string_of(&ev.session, read), "first");
            Ok(())
        })
        .unwrap();

        let after = ev.session.peek_global_var(sym).unwrap();
        assert_eq!(string_of(&ev.session, after), "assigned");
    }

    /// `undefine` reaches the same slot an assignment does, so it withdraws
    /// what the name meant outside the binding without disturbing the binding.
    #[test]
    fn test_undefine_under_a_binding_withdraws_the_global() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_BINDING_UNDEFINE");
        ev.session.globals.replace(sym, Some(from_file(b"outer")));

        ev.with_bound(sym, automatic(b"loop-word"), |ev| {
            ev.session.undefine_global_var(sym, false)?;
            let read = ev.session.peek_global_var(sym).unwrap();
            assert_eq!(string_of(&ev.session, read), "loop-word");
            Ok(())
        })
        .unwrap();

        assert!(ev.session.peek_global_var(sym).is_none());
    }
}
