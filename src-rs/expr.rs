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
use memchr::{memchr, memchr2, memchr3};

use crate::eval::{Evaluator, FrameType};
use crate::func::{FuncInfo, get_func_info};
use crate::loc::Loc;
use crate::session::Session;
use crate::strutil::{Pattern, WordWriter, trim_right_space, trim_suffix, word_scanner};
use crate::symtab::Symbol;
use crate::{error_loc, kati_warn_loc};

/// One expression node, named by where it sits in its session's arena.
///
/// A handle rather than a pointer, so it is `Copy`, costs four bytes wherever
/// one is stored, and carries no reference count. Reading the node it names
/// needs the arena, which is the session's; see [`ValueArena`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ValueId(u32);

impl ValueId {
    /// Expand this node into `out`.
    ///
    /// An inherent method rather than an [`Evaluable`] implementation: the
    /// trait's methods take `&self`, and a handle is passed by value.
    pub fn eval(self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
        eval_value(self, ev, out)
    }

    pub fn eval_to_buf(self, ev: &mut Evaluator) -> Result<Bytes> {
        eval_value_to_buf(self, ev)
    }

    pub fn eval_to_buf_mut(self, ev: &mut Evaluator) -> Result<BytesMut> {
        eval_value_to_buf_mut(self, ev)
    }

    /// See [`Value::resolve_folds`].
    #[must_use]
    pub fn resolve_folds(self, arena: &mut ValueArena, posix: bool) -> Self {
        Value::resolve_folds(arena, self, posix)
    }
}

/// A run of child nodes, stored contiguously in the arena's child table.
///
/// A list's items and a call's arguments are both this. They were a `Vec` each,
/// which is one allocation per expression that has more than one fragment in it
/// -- and, before the single-fragment case was answered by popping the vector,
/// one for every expression that has any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Children {
    start: u32,
    len: u32,
}

impl Children {
    pub const EMPTY: Self = Self { start: 0, len: 0 };

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The run with its first `n` entries dropped, which is what
    /// `$(call name,a,b)` hands the function it names.
    #[must_use]
    pub const fn skip(self, n: usize) -> Self {
        let n = if n as u32 > self.len {
            self.len
        } else {
            n as u32
        };
        Self {
            start: self.start + n,
            len: self.len - n,
        }
    }
}

/// Every expression node one session read, in two vectors.
///
/// The nodes an evaluation builds all die with the session that built them:
/// they are read out of makefile text, held by the variables, rules and
/// statements that text describes, and dropped together when the compilation
/// that owns them is over. That is an arena's lifetime exactly, and it is the
/// same argument `[dec:ronin:typed-graph-arenas]` makes for the graph side.
///
/// Dense indices rather than pointers, for the reason that decision gives and
/// for one more this side has: a [`crate::session::Session`] is moved -- into a
/// worker thread that composes a recursive unit, and out of it again -- and an
/// index survives a move where an interior pointer would not.
#[derive(Debug, Default)]
pub struct ValueArena {
    nodes: Vec<Value>,
    kids: Vec<ValueId>,
    /// Where a read under way accumulates the fragments of the expression it is
    /// on, before it knows how many there will be.
    ///
    /// One stack for the whole session rather than a vector per expression.
    /// Reads nest strictly -- a `$(` inside a list is read to its close before
    /// the list goes on -- so a caller marks the height on the way in and takes
    /// everything above the mark on the way out. The vector-per-expression this
    /// replaces was an allocation for every expression read, and four out of
    /// five of those held exactly one fragment: a line with no `$` in it, which
    /// in a real makefile is most lines.
    scratch: Vec<ValueId>,
}

impl ValueArena {
    /// A fresh arena with room for a small makefile's worth of nodes.
    ///
    /// Reserved up front rather than grown from nothing: a session that reads
    /// anything at all reaches four figures of nodes, and the doubling from
    /// zero is a dozen copies of everything already in it.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(512),
            kids: Vec::with_capacity(512),
            scratch: Vec::with_capacity(32),
        }
    }

    #[must_use]
    pub fn alloc(&mut self, value: Value) -> ValueId {
        let id = ValueId(u32::try_from(self.nodes.len()).expect("an expression arena under 4G"));
        self.nodes.push(value);
        id
    }

    #[must_use]
    pub fn get(&self, id: ValueId) -> &Value {
        &self.nodes[id.0 as usize]
    }

    /// Take a run of children into the child table.
    #[must_use]
    pub fn alloc_children(&mut self, items: &[ValueId]) -> Children {
        let start = u32::try_from(self.kids.len()).expect("an expression arena under 4G");
        self.kids.extend_from_slice(items);
        Children {
            start,
            len: u32::try_from(items.len()).expect("an argument list under 4G"),
        }
    }

    #[must_use]
    pub fn children(&self, run: Children) -> &[ValueId] {
        let start = run.start as usize;
        &self.kids[start..start + run.len as usize]
    }

    /// One child of a run, by position.
    ///
    /// Separate from [`Self::children`] because a caller that holds the arena
    /// through the evaluator cannot keep the slice across the `&mut` the next
    /// step wants; one `Copy` handle at a time is what it can keep.
    #[must_use]
    pub fn child(&self, run: Children, index: usize) -> ValueId {
        self.kids[run.start as usize + index]
    }

    /// The height of the fragment stack, to be handed back to
    /// [`Self::take_scratch`] or [`Self::scratch_above`].
    #[must_use]
    pub fn mark(&self) -> usize {
        self.scratch.len()
    }

    pub fn push_scratch(&mut self, id: ValueId) {
        self.scratch.push(id);
    }

    /// Allocate a node and put it on the fragment stack, which is what a read
    /// in the middle of an expression always wants.
    pub fn push_fragment(&mut self, value: Value) {
        let id = self.alloc(value);
        self.scratch.push(id);
    }

    /// How many fragments this read has accumulated since `mark`.
    #[must_use]
    pub fn scratch_above(&self, mark: usize) -> usize {
        self.scratch.len() - mark
    }

    /// The single fragment above `mark`, taken off the stack.
    ///
    /// The common case by a distance: an expression that is one literal, which
    /// is what a line with no `$` in it reads as. It becomes the value itself
    /// rather than a list of one, and nothing is taken into the child table.
    pub fn pop_scratch(&mut self) -> ValueId {
        self.scratch.pop().expect("a fragment above the mark")
    }

    /// Take everything above `mark` into the child table as one run.
    pub fn take_scratch(&mut self, mark: usize) -> Children {
        let start = u32::try_from(self.kids.len()).expect("an expression arena under 4G");
        let len = u32::try_from(self.scratch.len() - mark).expect("an expression under 4G");
        let Self { kids, scratch, .. } = self;
        kids.extend_from_slice(&scratch[mark..]);
        scratch.truncate(mark);
        Children { start, len }
    }

    /// Drop everything above `mark` without taking any of it.
    pub fn drop_scratch(&mut self, mark: usize) {
        self.scratch.truncate(mark);
    }

    /// How many nodes this arena holds, for `--kati_stats`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

pub trait Evaluable {
    fn eval(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()>;

    fn eval_to_buf_mut(&self, ev: &mut Evaluator) -> Result<BytesMut> {
        let mut out = BytesMut::new();
        self.eval(ev, &mut out)?;
        Ok(out)
    }

    fn eval_to_buf(&self, ev: &mut Evaluator) -> Result<Bytes> {
        Ok(self.eval_to_buf_mut(ev)?.freeze())
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ParseExprOpt {
    Normal,
    Define,
    Command,
    Func,
    /// Text a command wrote, which a `!=` assignment stores as a recursive
    /// value. It is not makefile source and was never read as a line, so
    /// nothing in it opens a comment — GNU Make hands the output straight to
    /// `define_variable_in_set` and only `variable_expand_string` ever looks at
    /// it again, and that reads `$` and nothing else.
    Captured,
}

/// Text a read could not make a call or a reference out of.
///
/// GNU Make discovers all three of these while it EXPANDS a value, not while it
/// reads one: `variable_expand_string` (expand.c) counts a reference's parens
/// as it walks the text it is expanding, and `handle_function` (function.c)
/// counts a call's parens and hands the count to `expand_builtin_function`,
/// which is where the argument count is judged. So text that no expansion ever
/// reaches is never judged at all, and a makefile is free to hold a call nobody
/// calls.
///
/// A read here builds a [`Value`] eagerly, so the complaint has to be held in
/// one until whatever holds it is expanded. That is what this is: a value whose
/// only behaviour is to raise.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Unreadable {
    /// A call whose closing `)` or `}` never arrived, carrying the function's
    /// name and the close that was wanted.
    UnterminatedCall(&'static [u8], u8),
    /// A `$(` or `${` whose close never arrived and which named no function.
    UnterminatedReference,
}

impl Unreadable {
    fn raise<C: crate::session::Context>(self, ctx: &C, loc: &Loc) -> anyhow::Error {
        match self {
            Unreadable::UnterminatedCall(name, close) => crate::color_error_log(
                ctx,
                Some(loc),
                format!(
                    "*** unterminated call to function '{}': missing '{}'.",
                    String::from_utf8_lossy(name),
                    char::from(close)
                ),
            ),
            Unreadable::UnterminatedReference => crate::color_error_log(
                ctx,
                Some(loc),
                "*** unterminated variable reference.".to_string(),
            ),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Value {
    Literal(Option<Loc>, Bytes),
    /// A complaint the read held rather than made, raised if and when the text
    /// holding it is expanded. See [`Unreadable`].
    Unreadable(Loc, Unreadable),
    List(Option<Loc>, Children),
    SymRef(Loc, Symbol),
    VarRef(Loc, ValueId),
    VarSubst {
        loc: Loc,
        name: ValueId,
        pat: ValueId,
        subst: ValueId,
    },
    Func {
        loc: Loc,
        fi: &'static FuncInfo,
        args: Children,
    },
    /// The bytes a run of line continuations collapses to, in both of the two
    /// readings `.POSIX:` chooses between: `plain` is what a fold ordinarily
    /// leaves — the blanks in front of the backslash discarded and the whole run
    /// a single space — and `posix` is what it leaves under `.POSIX:`, where the
    /// blanks stay and each folded newline is a space of its own.
    ///
    /// Held rather than settled at the read because a `.POSIX:` the evaluation
    /// reaches — through an `include`, a taken conditional, an expansion-named
    /// target — turns the flag on for a line kati had already read. The choice
    /// is made where the value is read: [`Value::resolve_folds`] settles it into
    /// the recursive value a definition stores, and [`Evaluable::eval`] answers
    /// it for text expanded on the spot. Both consult the evaluator's `is_posix`,
    /// which is the flag as the *evaluation* has reached it. A fold whose two
    /// readings are identical is never held as one -- it is emitted as literal
    /// text, so this stands only where `.POSIX:` would be observable.
    Folded {
        plain: Bytes,
        posix: Bytes,
    },
}

/// What one node asks of the evaluation, taken out of the node so the arena's
/// borrow can end before the evaluator is touched mutably.
///
/// Every field is a handle, a symbol or a location -- `Copy`, or cheap to copy
/// -- because that is the whole point: a `&Value` read out of the arena borrows
/// the evaluator that owns the arena, and every step below wants the evaluator
/// mutably. Text is not here: the two arms that only write bytes finish inside
/// the borrow and never reach this.
enum Step {
    Written,
    List(Children),
    SymRef(Symbol),
    VarRef(ValueId),
    VarSubst {
        name: ValueId,
        pat: ValueId,
        subst: ValueId,
    },
    Func {
        loc: Loc,
        fi: &'static FuncInfo,
        args: Children,
    },
    Unreadable(Loc, Unreadable),
}

/// Expand the node `id` names into `out`.
///
/// A free function rather than a method on [`Value`], because a node lives in
/// the arena the session owns: holding a `&Value` holds a borrow of the
/// evaluator, and expansion is the evaluator's most mutable operation. Each arm
/// takes what it needs out of the node and lets the borrow go before it
/// recurses, and the compiler is what checks that it did.
pub fn eval_value(id: ValueId, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let step = {
        let node = ev.session.values.get(id);
        match node {
            // The two arms that are only bytes finish here, inside the borrow.
            // Between them they are most of every makefile.
            Value::Literal(_, lit) => {
                out.put_slice(lit);
                Step::Written
            }
            // Text expanded on the spot answers to the flag the evaluation has
            // reached, which is exactly what GNU Make's read did: the line is
            // folded where it is read, and by then every `.POSIX:` above it has
            // been seen. A value a definition stored has had this settled
            // already by [`resolve_folds`], so what reaches here is the
            // immediate kind -- a rule word, a function argument, `$(info ...)`.
            Value::Folded { plain, posix } => {
                out.put_slice(if ev.is_posix() { posix } else { plain });
                Step::Written
            }
            Value::List(_, items) => Step::List(*items),
            Value::SymRef(_, sym) => Step::SymRef(*sym),
            Value::VarRef(_, name) => Step::VarRef(*name),
            Value::VarSubst {
                loc: _,
                name,
                pat,
                subst,
            } => Step::VarSubst {
                name: *name,
                pat: *pat,
                subst: *subst,
            },
            Value::Func { loc, fi, args } => Step::Func {
                loc: *loc,
                fi,
                args: *args,
            },
            Value::Unreadable(loc, unreadable) => Step::Unreadable(*loc, *unreadable),
        }
    };
    eval_step(step, ev, out)
}

fn eval_step(step: Step, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    match step {
        Step::Written => {}
        Step::List(items) => {
            for index in 0..items.len() {
                let item = ev.session.values.child(items, index);
                eval_value(item, ev, out)?;
            }
        }
        Step::SymRef(sym) => {
            {
                let is_make =
                    ev.is_evaluating_command && sym.as_bytes(&ev.session).as_ref() == b"MAKE";
                if let Some(var) = ev.lookup_var_for_eval(sym)? {
                    let v = var.read();
                    // The reference is where GNU Make installs the location a
                    // diagnostic raised inside the value will carry --
                    // `recursively_expand_for_file` in expand.c.
                    ev.enter_expanding_var(v.expansion_loc());
                    if is_make {
                        let expanded = v.eval_to_buf(ev)?;
                        out.put_slice(&expanded);
                        ev.expanded_make_in_command.push(expanded);
                    } else {
                        v.eval(ev, out)?;
                    }
                    drop(v);
                    ev.leave_expanding_var();
                    ev.var_eval_complete(&var);
                }
            }
        }
        Step::VarRef(name) => {
            ev.eval_depth += 1;
            let name = eval_value_to_buf(name, ev)?;
            ev.eval_depth -= 1;
            let sym = ev.session.intern(name);
            let is_make = ev.is_evaluating_command && sym.as_bytes(&ev.session).as_ref() == b"MAKE";
            if let Some(var) = ev.lookup_var_for_eval(sym)? {
                let v = var.read();
                // The reference is where GNU Make installs the location a
                // diagnostic raised inside the value will carry --
                // `recursively_expand_for_file` in expand.c.
                ev.enter_expanding_var(v.expansion_loc());
                if is_make {
                    let expanded = v.eval_to_buf(ev)?;
                    out.put_slice(&expanded);
                    ev.expanded_make_in_command.push(expanded);
                } else {
                    v.eval(ev, out)?;
                }
                drop(v);
                ev.leave_expanding_var();
                ev.var_eval_complete(&var);
            }
        }
        Step::VarSubst { name, pat, subst } => {
            ev.eval_depth += 1;
            let name = eval_value_to_buf(name, ev)?;
            let sym = ev.session.intern(name);
            let v = ev.lookup_var(sym)?;
            if v.is_none() {
                ev.warn_undefined(sym);
            }
            let pat_str = eval_value_to_buf(pat, ev)?;
            let subst = eval_value_to_buf(subst, ev)?;
            ev.eval_depth -= 1;
            if let Some(var) = v {
                let v = var.read();
                // `$(V:a=b)` reaches V's value through `recursively_expand`
                // as well, so it installs the location too.
                ev.enter_expanding_var(v.expansion_loc());
                let value = v.eval_to_buf(ev)?;
                ev.leave_expanding_var();
                let mut ww = WordWriter::new(out);
                let pat = Pattern::new(pat_str);
                for tok in word_scanner(&value) {
                    ww.maybe_add_space();
                    let tok = value.slice_ref(tok);
                    ww.out.put_slice(&pat.append_subst_ref(&tok, &subst));
                }
            }
        }
        // Raised where GNU Make raises it: `variable_expand_string` and
        // `handle_function` both die at `*expanding_var`, which names the
        // binding being expanded rather than the text inside it.
        Step::Unreadable(loc, unreadable) => {
            let at = ev.expanding_var_loc().unwrap_or(loc);
            return Err(unreadable.raise(ev, &at));
        }
        Step::Func { loc, fi, args } => {
            {
                let _frame = ev.enter(FrameType::FunCall, Bytes::from_static(fi.name), loc);
                ev.eval_depth += 1;
                // GNU Make counts the arguments in `expand_builtin_function`,
                // which is to say inside the expansion and after
                // `handle_function` has already expanded them -- so a call with
                // too few of them has already run whatever its arguments do by
                // the time it is refused. Which functions get that treatment is
                // GNU Make's `expand_args`, and this flag is that column: it is
                // exact rather than approximate here, because the only functions
                // this complaint can reach are the ones wanting two arguments or
                // more -- a call parses with at least one -- and every one of
                // those carries the same value for it that GNU Make's table
                // does.
                if (args.len() as i16) < fi.min_arity {
                    if fi.pre_expanded_args {
                        for index in 0..args.len() {
                            let arg = ev.session.values.child(args, index);
                            eval_value_to_buf(arg, ev)?;
                        }
                    }
                    let at = ev.expanding_var_loc().unwrap_or(loc);
                    error_loc!(
                        ev,
                        Some(&at),
                        "*** insufficient number of arguments ({}) to function '{}'.",
                        args.len(),
                        String::from_utf8_lossy(fi.name)
                    );
                }
                ev.function_depth += 1;
                let called = (fi.func)(args, ev, out);
                ev.function_depth -= 1;
                called?;
                ev.eval_depth -= 1;
            }
        }
    }
    Ok(())
}

/// [`eval_value`] into a fresh buffer, which is what most callers want.
pub fn eval_value_to_buf(id: ValueId, ev: &mut Evaluator) -> Result<Bytes> {
    Ok(eval_value_to_buf_mut(id, ev)?.freeze())
}

pub fn eval_value_to_buf_mut(id: ValueId, ev: &mut Evaluator) -> Result<BytesMut> {
    let mut out = BytesMut::new();
    eval_value(id, ev, &mut out)?;
    Ok(out)
}

impl Value {
    pub fn loc(&self) -> Option<Loc> {
        match self {
            Value::Literal(loc, _) => *loc,
            Value::Unreadable(loc, _) => Some(*loc),
            Value::List(loc, _) => *loc,
            Value::SymRef(loc, _) => Some(*loc),
            Value::VarRef(loc, _) => Some(*loc),
            Value::VarSubst { loc, .. } => Some(*loc),
            Value::Func { loc, .. } => Some(*loc),
            Value::Folded { .. } => None,
        }
    }

    /// Settle every `Folded` in this value against the `.POSIX:` state the
    /// evaluation has reached, replacing it with the reading that state selects.
    ///
    /// Called where a definition stores a recursive value, so the fold is read
    /// once — at the definition, as GNU Make reads it — rather than re-read
    /// against whatever `.POSIX:` is in force wherever the value is later
    /// expanded. A value holding no `Folded` is returned as itself, allocating
    /// nothing; only the nodes on the path to one are rebuilt.
    pub fn resolve_folds(arena: &mut ValueArena, id: ValueId, posix: bool) -> ValueId {
        fn resolved_run(arena: &mut ValueArena, items: Children, posix: bool) -> Option<Children> {
            let mut changed = false;
            // Settled onto the stack first, because taking them into the child
            // table as they are settled would interleave a nested list's run
            // with this one's.
            let mut out = Vec::with_capacity(items.len());
            for index in 0..items.len() {
                let item = arena.child(items, index);
                let settled = Value::resolve_folds(arena, item, posix);
                changed |= settled != item;
                out.push(settled);
            }
            changed.then(|| arena.alloc_children(&out))
        }
        // A copy of what the node needs settling, so the arena is free to grow
        // underneath the recursion.
        enum Settle {
            Same,
            Literal(Option<Loc>, Bytes),
            List(Option<Loc>, Children),
            VarRef(Loc, ValueId),
            VarSubst(Loc, ValueId, ValueId, ValueId),
            Func(Loc, &'static FuncInfo, Children),
        }
        let settle = match arena.get(id) {
            Value::Folded {
                plain,
                posix: under_posix,
            } => Settle::Literal(
                None,
                if posix {
                    under_posix.clone()
                } else {
                    plain.clone()
                },
            ),
            Value::List(loc, items) => Settle::List(*loc, *items),
            Value::VarRef(loc, name) => Settle::VarRef(*loc, *name),
            Value::VarSubst {
                loc,
                name,
                pat,
                subst,
            } => Settle::VarSubst(*loc, *name, *pat, *subst),
            Value::Func { loc, fi, args } => Settle::Func(*loc, fi, *args),
            Value::Literal(..) | Value::SymRef(..) | Value::Unreadable(..) => Settle::Same,
        };
        match settle {
            Settle::Same => id,
            Settle::Literal(loc, text) => arena.alloc(Value::Literal(loc, text)),
            Settle::List(loc, items) => match resolved_run(arena, items, posix) {
                Some(items) => arena.alloc(Value::List(loc, items)),
                None => id,
            },
            Settle::VarRef(loc, name) => {
                let settled = Value::resolve_folds(arena, name, posix);
                if settled == name {
                    id
                } else {
                    arena.alloc(Value::VarRef(loc, settled))
                }
            }
            Settle::VarSubst(loc, name, pat, subst) => {
                let name_s = Value::resolve_folds(arena, name, posix);
                let pat_s = Value::resolve_folds(arena, pat, posix);
                let subst_s = Value::resolve_folds(arena, subst, posix);
                if name_s == name && pat_s == pat && subst_s == subst {
                    id
                } else {
                    arena.alloc(Value::VarSubst {
                        loc,
                        name: name_s,
                        pat: pat_s,
                        subst: subst_s,
                    })
                }
            }
            Settle::Func(loc, fi, args) => match resolved_run(arena, args, posix) {
                Some(args) => arena.alloc(Value::Func { loc, fi, args }),
                None => id,
            },
        }
    }
}

fn close_paren(c: u8) -> Option<u8> {
    match c {
        b'(' => Some(b')'),
        b'{' => Some(b'}'),
        _ => None,
    }
}

/// Whether a `\#` written here is an escape the read takes off.
///
/// It is one only where the `#` it hides would otherwise open a comment. GNU
/// Make removes comments with `find_map_unquote (line, MAP_COMMENT|MAP_VARIABLE)`
/// (read.c), and `MAP_VARIABLE` is what makes the scan step over a `$(` or `${`
/// to its close without looking inside: a `#` in there was never going to start
/// a comment, so nothing unquotes the backslash in front of it and the value
/// keeps both characters. A `define` body, a recipe line and text a command
/// wrote are the other three places the scan does not reach.
fn should_handle_comments(opt: ParseExprOpt) -> bool {
    !matches!(
        opt,
        ParseExprOpt::Define | ParseExprOpt::Command | ParseExprOpt::Func | ParseExprOpt::Captured
    )
}

/// How GNU Make's `collapse_continuations` (misc.c) reads the run of
/// backslashes at the start of `text`.
///
/// The run escapes the newline that ends it only when it is odd, and each pair
/// of backslashes in it quotes itself down to one: `a\\\` before a newline is a
/// literal backslash and a continuation, while `a\\` before one is two literal
/// backslashes and the end of the line.
///
/// `Some((kept, consumed))` says how many of the run's backslashes survive into
/// the value, and how much of `text` the fold takes — the whole run and the
/// newline after it. `None` says this run does not continue the line, so it is
/// value text like any other.
fn continuation_fold(text: &[u8]) -> Option<(usize, usize)> {
    let run = text.iter().take_while(|byte| **byte == b'\\').count();
    if run % 2 == 0 {
        return None;
    }
    let newline = match text[run..] {
        [b'\r', b'\n', ..] => 2,
        [b'\r' | b'\n', ..] => 1,
        _ => return None,
    };
    Some((run / 2, run + newline))
}

/// Advance past everything a fold absorbs into its single space: the blanks on
/// the far side of the newline, and — outside `.POSIX:` — any further
/// continuation those blanks lead to.
///
/// The blanks always go: `collapse_continuations` (misc.c) skips them before it
/// asks anything else. What follows is where the two readings part. Ordinarily
/// GNU Make then discards the blanks it has already WRITTEN, so a run of
/// continuations separated by nothing but blanks comes out as one space however
/// long it is; under `.POSIX:` it discards nothing and each folded newline is a
/// space of its own, so the run is left for the caller to read one fold at a
/// time. A continuation whose run leaves a backslash behind ends the absorbing
/// either way, because that backslash is value text that has to be written
/// before the next space.
fn skip_folded(loc: &mut Loc, s: &[u8], mut at: usize, posix: bool) -> usize {
    loop {
        while matches!(s.get(at), Some(b' ' | b'\t')) {
            at += 1;
        }
        if posix {
            return at;
        }
        let Some((0, consumed)) = continuation_fold(&s[at..]) else {
            return at;
        };
        loc.line += 1;
        at += consumed;
    }
}

/// [`skip_folded`] with `posix` false, counting the folds it takes.
///
/// A `Folded` holds both readings of a run at once, so it must consume the run
/// maximally — the way the plain reading does — and know how many further folds
/// the first one absorbed, since under `.POSIX:` each of those is a space of its
/// own. Returns where the run ends and that count (not including the fold that
/// triggered it).
fn absorb_fold_run(loc: &mut Loc, s: &[u8], mut at: usize) -> (usize, usize) {
    let mut extra = 0;
    loop {
        while matches!(s.get(at), Some(b' ' | b'\t')) {
            at += 1;
        }
        let Some((0, consumed)) = continuation_fold(&s[at..]) else {
            return (at, extra);
        };
        loc.line += 1;
        at += consumed;
        extra += 1;
    }
}

/// The `.POSIX:` reading of a fold: the blanks written before the backslash,
/// kept rather than discarded, then one space for each folded newline.
fn posix_fold_bytes(trailing: &[u8], spaces: usize) -> Bytes {
    let mut bytes = Vec::with_capacity(trailing.len() + spaces);
    bytes.extend_from_slice(trailing);
    bytes.resize(trailing.len() + spaces, b' ');
    Bytes::from(bytes)
}

/// Whether this byte ends a function name where GNU Make ends one.
///
/// `lookup_function` (function.c) walks the name over `MAP_USERFUNC` and then
/// insists the byte it stopped on is in `MAP_NUL|MAP_SPACE` -- a NUL, or any
/// character `isspace` accepts. So a space is not special: a tab, a newline, a
/// carriage return, a vertical tab and a form feed all name a call too, and so
/// does running out of text. The backslash is here because a continuation
/// arrives at this decision unfolded, and the newline it stands for is one of
/// the six.
fn ends_a_function_name(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b'\\')
}

fn skip_spaces(loc: &mut Loc, s: &[u8], terms: &[u8]) -> usize {
    let mut i = 0;
    while i < s.len() {
        let remaining = &s[i..];
        let c = remaining[0];
        if terms.contains(&c) {
            return i;
        }

        if !c.is_ascii_whitespace() {
            if !remaining.starts_with(b"\\\r") && !remaining.starts_with(b"\\\n") {
                return i;
            }

            loc.line += 1; // This is a backspace continuation
        }
        i += 1;
    }
    s.len()
}

/// What reading a call's argument list found: the arguments, or that the text
/// ran out before the close did.
///
/// The second is not raised here. GNU Make counts a call's parens in
/// `handle_function`, which runs while the text is being expanded, so text no
/// expansion reaches keeps its missing close to itself -- see [`Unreadable`].
enum ParsedCall {
    Args(Children),
    Unterminated,
}

fn parse_func(
    session: &mut Session,
    loc: &mut Loc,
    fi: &FuncInfo,
    s: Bytes,
    mut i: usize,
    mut terms: Vec<u8>,
) -> Result<(usize, ParsedCall)> {
    terms.truncate(2);
    terms[1] = b',';
    i += skip_spaces(loc, &s[i..], &terms);
    if i == s.len() {
        // The text ended between the name and any argument, so the close never
        // arrived either.
        return Ok((i, ParsedCall::Unterminated));
    }

    let mut nargs = 1;
    let mark = session.values.mark();
    loop {
        if fi.arity > 0 && nargs >= fi.arity {
            terms.truncate(1); // Drop ','.
        }

        if fi.trim_space {
            while i < s.len() {
                let c = s[i];
                if c.is_ascii_whitespace() {
                    i += 1;
                    continue;
                }

                let t = &s[i..];
                if t.starts_with(b"\\\r") || t.starts_with(b"\\\n") {
                    loc.line += 1;
                    i += 1;
                    continue;
                }

                break;
            }
        }

        let trim_right_space = fi.trim_space || (nargs == 1 && fi.trim_right_space_1st);
        let (n, val) = parse_expr_impl(
            session,
            loc,
            s.slice(i..),
            Some(&terms),
            ParseExprOpt::Func,
            trim_right_space,
        )?;
        // TODO: concatLine???
        session.values.push_scratch(val);
        i += n;
        if i == s.len() {
            session.values.drop_scratch(mark);
            return Ok((i, ParsedCall::Unterminated));
        }
        nargs += 1;
        if s[i] == terms[0] {
            i += 1;
            break;
        }
        i += 1; // Should be ','.
        if i == s.len() {
            // A comma was the last thing in the text, so the close is missing
            // rather than an argument being.
            session.values.drop_scratch(mark);
            return Ok((i, ParsedCall::Unterminated));
        }
    }

    let args = session.values.take_scratch(mark);
    Ok((i, ParsedCall::Args(args)))
}

fn parse_dollar(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    end_paren: bool,
) -> Result<(usize, ValueId)> {
    assert!(s.len() >= 2);
    assert!(s.starts_with(b"$"));
    assert!(!s.starts_with(b"$$"));

    let start_loc = *loc;

    let Some(cp) = close_paren(s[1]) else {
        let sym = session.intern(s.slice(1..2));
        return Ok((2, session.values.alloc(Value::SymRef(start_loc, sym))));
    };

    // Every byte that ends a name for `lookup_function`, so the scan stops
    // where GNU Make's does. `terms.truncate(2)` below drops the whole tail of
    // them at once, exactly as it dropped the single space before.
    let mut terms = vec![cp, b':', b' ', b'\t', b'\n', 0x0b, 0x0c, b'\r'];
    let mut i = 2;
    loop {
        let (n, vname) = parse_expr_impl(
            session,
            loc,
            s.slice(i..),
            Some(&terms),
            ParseExprOpt::Normal,
            false,
        )?;
        i += n;

        let t: &[u8] = &s[i..];
        // A name the scan ended is a call if it spells one. GNU Make asks the
        // same question first: `variable_expand_string` (expand.c) tries
        // `handle_function` before it goes looking for the close, which is why
        // `$(subst)` -- a name ended by the close itself -- is a reference and
        // `$(subst` at the end of the text is a call.
        let name_ended = t.first().is_none_or(|c| ends_a_function_name(*c));
        let named_func = match session.values.get(vname) {
            Value::Literal(_, lit) => get_func_info(lit),
            _ => None,
        };
        if name_ended && let Some(fi) = named_func {
            // Step over the byte that ended the name. Where the text ran out
            // there is no byte to step over.
            let args_at = i + usize::from(!t.is_empty());
            let (idx, parsed) = parse_func(session, loc, fi, s, args_at, terms)?;
            let value = match parsed {
                ParsedCall::Args(args) => Value::Func {
                    loc: start_loc,
                    fi,
                    args,
                },
                ParsedCall::Unterminated => {
                    Value::Unreadable(start_loc, Unreadable::UnterminatedCall(fi.name, cp))
                }
            };
            return Ok((idx, session.values.alloc(value)));
        }

        if t.first() == Some(&cp) || (end_paren && t.is_empty() && cp == b')') {
            let literal = match session.values.get(vname) {
                Value::Literal(_, lit) => Some(lit.clone()),
                _ => None,
            };
            if let Some(lit) = literal {
                let sym = session.intern(lit);
                if session.flags.enable_kati_warnings {
                    let name = sym.display(&*session).to_string();
                    if let Some(found) = name.find([' ', '(', '{']) {
                        kati_warn_loc!(
                            session,
                            Some(&start_loc),
                            "*warning*: variable lookup with '{}': {}",
                            &name[found..found + 1],
                            String::from_utf8_lossy(&s)
                        )
                    }
                }
                return Ok((i + 1, session.values.alloc(Value::SymRef(start_loc, sym))));
            }
            return Ok((i + 1, session.values.alloc(Value::VarRef(start_loc, vname))));
        }

        if name_ended && !t.is_empty() {
            let literal = match session.values.get(vname) {
                Value::Literal(_, lit) => Some(lit.clone()),
                _ => None,
            };
            if let Some(lit) = literal {
                kati_warn_loc!(
                    session,
                    Some(&start_loc),
                    "*warning*: unknown make function {lit:?}: {}",
                    String::from_utf8_lossy(&s)
                );
            }

            // Not a function. Drop the name terminators from |terms| and parse
            // it again. This is inefficient, but this code path should be
            // rarely used.
            terms.truncate(2);
            i = 2;
            continue;
        }

        if t.first() == Some(&b':') {
            terms.truncate(2);
            terms[1] = b'=';
            let (n, pat) = parse_expr_impl(
                session,
                loc,
                s.slice(i + 1..),
                Some(&terms),
                ParseExprOpt::Normal,
                false,
            )?;
            i += 1 + n;
            if s.get(i) == Some(&cp) {
                let colon = session
                    .values
                    .alloc(Value::Literal(None, Bytes::from_static(b":")));
                let items = session.values.alloc_children(&[vname, colon, pat]);
                let spelling = session.values.alloc(Value::List(Some(start_loc), items));
                return Ok((
                    i + 1,
                    session.values.alloc(Value::VarRef(start_loc, spelling)),
                ));
            }

            terms.truncate(1);
            let (n, subst) = parse_expr_impl(
                session,
                loc,
                s.slice(i + 1..),
                Some(&terms),
                ParseExprOpt::Normal,
                false,
            )?;
            i += 1 + n;
            return Ok((
                i + 1,
                session.values.alloc(Value::VarSubst {
                    loc: start_loc,
                    name: vname,
                    pat,
                    subst,
                }),
            ));
        }

        // GNU make accepts expressions like $((). See unmatched_paren*.mk
        // for detail.
        if let Some(found) = memchr(cp, &s) {
            kati_warn_loc!(
                session,
                Some(&start_loc),
                "*warning*: unmatched parentheses: {}",
                String::from_utf8_lossy(&s)
            );
            let sym = session.intern(s.slice(2..found));
            return Ok((s.len(), session.values.alloc(Value::SymRef(start_loc, sym))));
        }

        // Held rather than raised: GNU Make finds an unterminated reference in
        // `variable_expand_string`, which only ever looks at text it is
        // expanding. See [`Unreadable`].
        return Ok((
            s.len(),
            session.values.alloc(Value::Unreadable(
                start_loc,
                Unreadable::UnterminatedReference,
            )),
        ));
    }
}

/// The bytes at which the scan in [`parse_expr_impl_ext`] has something to do.
///
/// That loop asks as many as six questions of every byte it reads, and a
/// makefile is mostly text where the answer to all six is no: a path, a
/// compiler flag, the prerequisite list of a rule. Which bytes can be answered
/// yes is fixed by the parse options, the caller's terminators, and whether a
/// parenthesis is open, so the set is computed once and the run between two of
/// its members is stepped over whole rather than a byte at a time.
///
/// The set must be exactly the bytes the loop acts on. Too few and a reference
/// is read as literal text; too many only costs a stop that decides to do
/// nothing, which the loop already handles.
#[derive(Clone, Copy)]
struct Stops {
    /// One bit per byte value, for a set too large to hand to `memchr`.
    bits: [u64; 4],
    /// The set itself while it still fits, because `memchr` over a run beats a
    /// bit test per byte of it by more than the branch here costs.
    few: [u8; 3],
    /// How many bytes are in the set, which may exceed what `few` holds.
    len: usize,
}

impl Stops {
    fn new(
        opt: ParseExprOpt,
        terms: Option<&[u8]>,
        terms_ignored: usize,
        save_paren: Option<u8>,
    ) -> Self {
        let mut stops = Self {
            bits: [0; 4],
            few: [0; 3],
            len: 0,
        };
        stops.add(b'$');
        if terms.is_none() && should_handle_comments(opt) {
            stops.add(b'#');
        }
        if opt != ParseExprOpt::Command {
            stops.add(b'\\');
        }
        if opt == ParseExprOpt::Func {
            stops.add(b'(');
            stops.add(b'{');
        }
        if let Some(close) = save_paren {
            stops.add(close);
        } else if let Some(terms) = terms {
            for term in &terms[terms_ignored..] {
                stops.add(*term);
            }
        }
        stops
    }

    fn add(&mut self, byte: u8) {
        let word = usize::from(byte) >> 6;
        let bit = 1u64 << (u32::from(byte) & 63);
        if self.bits[word] & bit != 0 {
            return;
        }
        self.bits[word] |= bit;
        if let Some(slot) = self.few.get_mut(self.len) {
            *slot = byte;
        }
        self.len += 1;
    }

    /// The first index at or after `from` holding a byte of the set, or the
    /// length of `s` when there is none.
    #[inline]
    fn next(&self, s: &[u8], from: usize) -> usize {
        let rest = &s[from..];
        let found = match (self.len, self.few) {
            (1, [a, _, _]) => memchr(a, rest),
            (2, [a, b, _]) => memchr2(a, b, rest),
            (3, [a, b, c]) => memchr3(a, b, c, rest),
            _ => return self.scan(s, from),
        };
        found.map_or(s.len(), |at| from + at)
    }

    fn scan(&self, s: &[u8], from: usize) -> usize {
        for (offset, byte) in s[from..].iter().enumerate() {
            let word = usize::from(*byte) >> 6;
            let bit = 1u64 << (u32::from(*byte) & 63);
            if self.bits[word] & bit != 0 {
                return from + offset;
            }
        }
        s.len()
    }
}

pub fn parse_expr_impl(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    terms: Option<&[u8]>,
    opt: ParseExprOpt,
    trim_right_sp: bool,
) -> Result<(usize, ValueId)> {
    parse_expr_impl_ext(session, loc, s, terms, opt, trim_right_sp, false)
}

/// Read one expression, leaving the fragment stack as it found it.
///
/// The unwind is the whole of this wrapper's job. A read that raises has
/// already pushed some of its fragments, and a caller that goes on evaluating
/// afterwards -- `$(eval)` inside a construct whose complaint is caught -- would
/// have the next read take those fragments as its own. Truncating to the mark
/// on the way out is what keeps the stack a stack.
pub fn parse_expr_impl_ext(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    terms: Option<&[u8]>,
    opt: ParseExprOpt,
    trim_right_sp: bool,
    // This is for compatibility with a read-past-end in ckati
    end_paren: bool,
) -> Result<(usize, ValueId)> {
    let mark = session.values.mark();
    let read = parse_expr_read(session, loc, s, terms, opt, trim_right_sp, end_paren, mark);
    if read.is_err() {
        session.values.drop_scratch(mark);
    }
    read
}

#[allow(clippy::too_many_arguments)]
fn parse_expr_read(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    terms: Option<&[u8]>,
    opt: ParseExprOpt,
    trim_right_sp: bool,
    end_paren: bool,
    mark: usize,
) -> Result<(usize, ValueId)> {
    let list_loc = *loc;
    // What `collapse_continuations` asks about every line it folds. Read once:
    // the answer belongs to the moment the line is read, and nothing this parse
    // does can change it.
    let posix = session.posix_pedantic;

    let s = s.slice_ref(trim_suffix(&s, b"\r"));

    let mut b = 0usize;
    let mut save_paren: Option<u8> = None;
    let mut paren_depth: i32 = 0;
    let mut i = 0usize;
    let mut terms_ignored = 0;
    let mut stops = Stops::new(opt, terms, terms_ignored, save_paren);

    while i < s.len() {
        // Step over the run of bytes no arm below can act on. Recomputed
        // wherever `save_paren` or `terms_ignored` moves, which is the only
        // thing that changes which bytes those are.
        i = stops.next(&s, i);
        if i >= s.len() {
            break;
        }
        let item_loc = *loc;

        let remaining = &s[i..];
        let c = remaining[0];
        if let Some(terms) = terms
            && save_paren.is_none()
            && terms[terms_ignored..].contains(&c)
        {
            break;
        }

        // Handle a comment
        if terms.is_none() && c == b'#' && should_handle_comments(opt) {
            if i > b {
                session
                    .values
                    .push_fragment(Value::Literal(None, s.slice(b..i)));
            }
            let mut was_backslash = false;
            while i < s.len() && s[i] != b'\n' || was_backslash {
                was_backslash = !was_backslash && s[i] == b'\\';
                i += 1;
            }
            if session.values.scratch_above(mark) == 1 {
                return Ok((i, session.values.pop_scratch()));
            }
            let items = session.values.take_scratch(mark);
            return Ok((i, session.values.alloc(Value::List(Some(item_loc), items))));
        }

        if c == b'$' {
            if i > b {
                session
                    .values
                    .push_fragment(Value::Literal(None, s.slice(b..i)));
            }

            // A `$` with nothing after it is one literal dollar, exactly as
            // `$$` is: GNU Make's expander gives the two the same arm
            // (expand.c variable_expand_string, `case '$': case '\0':`). The
            // blanks in front of it are not trailing any more, so a caller
            // asking for a right trim does not reach them.
            if i + 1 >= s.len() {
                session
                    .values
                    .push_fragment(Value::Literal(None, Bytes::from_static(b"$")));
                i += 1;
                b = i;
                continue;
            }

            if remaining.starts_with(b"$$") {
                session
                    .values
                    .push_fragment(Value::Literal(None, Bytes::from_static(b"$")));
                i += 2;
                b = i;
                continue;
            }

            // GNU Make folds the continuation before it reads the reference, so
            // the name this `$` takes is whatever the fold left beside it: the
            // space the newline became, or the first of the backslashes the run
            // kept. A recipe is the exception, because there the continuation is
            // the shell's and Make hands it over unfolded.
            let folded = (opt != ParseExprOpt::Command)
                .then(|| continuation_fold(&remaining[1..]))
                .flatten();
            let named = match folded {
                Some((0, _)) => b' ',
                Some(_) => b'\\',
                None => remaining[1],
            };

            if let Some(terms) = terms
                && terms[terms_ignored..].contains(&named)
            {
                let val = session
                    .values
                    .alloc(Value::Literal(None, Bytes::from_static(b"$")));
                if session.values.scratch_above(mark) == 0 {
                    return Ok((i + 1, val));
                }
                session.values.push_scratch(val);
                let items = session.values.take_scratch(mark);
                return Ok((
                    i + 1,
                    session.values.alloc(Value::List(Some(item_loc), items)),
                ));
            }

            if let Some((kept, consumed)) = folded {
                loc.line += 1;
                let name = if kept == 0 { &b" "[..] } else { &b"\\"[..] };
                let sym = session.intern(Bytes::from_static(name));
                session.values.push_fragment(Value::SymRef(item_loc, sym));
                // The reference took the first backslash the run kept. The rest
                // of them, and the space the newline became, are value text.
                if kept > 0 {
                    session
                        .values
                        .push_fragment(Value::Literal(None, s.slice(i + 2..i + 1 + kept)));
                    session
                        .values
                        .push_fragment(Value::Literal(None, Bytes::from_static(b" ")));
                }
                if posix {
                    i = skip_folded(loc, &s, i + 1 + consumed, true);
                } else {
                    // The reference took the fold's space as its name. A further
                    // run behind it is one space per fold under `.POSIX:` and
                    // nothing without it — the plain reading swallows the run —
                    // so hold that difference for the evaluation to settle.
                    let (end, extra) = absorb_fold_run(loc, &s, i + 1 + consumed);
                    if extra > 0 {
                        session.values.push_fragment(Value::Folded {
                            plain: Bytes::new(),
                            posix: posix_fold_bytes(b"", extra),
                        });
                    }
                    i = end;
                }
                b = i;
                continue;
            }

            let (n, v) = parse_dollar(session, loc, s.slice(i..), end_paren)?;
            session.values.push_scratch(v);
            i += n;
            b = i;
            continue;
        }

        if (c == b'(' || c == b'{') && opt == ParseExprOpt::Func {
            let cp = close_paren(c);
            if terms
                .map(|v| v[terms_ignored..].first() == cp.as_ref())
                .unwrap_or(false)
            {
                paren_depth += 1;
                save_paren = cp;
                terms_ignored += 1;
                stops = Stops::new(opt, terms, terms_ignored, save_paren);
            } else if cp == save_paren {
                paren_depth += 1;
            }
            i += 1;
            continue;
        }

        if Some(c) == save_paren {
            paren_depth -= 1;
            if paren_depth == 0 {
                terms_ignored -= 1;
                save_paren = None;
                stops = Stops::new(opt, terms, terms_ignored, save_paren);
            }
        }

        if c == b'\\' && i + 1 < s.len() && opt != ParseExprOpt::Command {
            if let Some((kept, consumed)) = continuation_fold(remaining) {
                loc.line += 1;
                if let Some(terms) = terms
                    && terms.contains(&b' ')
                {
                    break;
                }
                // Half the run stays, so the literal reaches into it. The
                // blanks written before it go only when none of it does: GNU
                // Make discards what it has already written back to the last
                // byte that is not a blank, and a backslash is not one.
                let literal_end = i + kept;
                let text = &s[b..literal_end];
                if posix {
                    // `.POSIX:` was written above and the read has seen it: the
                    // blanks in front of the backslash stay and each folded
                    // newline is a space of its own, so the run is left for the
                    // outer loop to read one fold at a time.
                    if literal_end > b {
                        session
                            .values
                            .push_fragment(Value::Literal(None, s.slice_ref(text)));
                    }
                    session
                        .values
                        .push_fragment(Value::Literal(None, Bytes::from_static(b" ")));
                    i = skip_folded(loc, &s, i + consumed, true);
                } else {
                    // The read has seen no `.POSIX:`, but the evaluation may
                    // still reach one before this value is read. Emit the text
                    // the two readings share — the run's leading non-blank bytes
                    // — and hold the rest as a `Folded` when they differ: `plain`
                    // discards the blanks and the whole run is one space, `posix`
                    // keeps the blanks and adds a space per fold.
                    let (end, extra) = absorb_fold_run(loc, &s, i + consumed);
                    let head = trim_right_space(text);
                    if !head.is_empty() {
                        session
                            .values
                            .push_fragment(Value::Literal(None, s.slice_ref(head)));
                    }
                    let trailing = &text[head.len()..];
                    if trailing.is_empty() && extra == 0 {
                        session
                            .values
                            .push_fragment(Value::Literal(None, Bytes::from_static(b" ")));
                    } else {
                        session.values.push_fragment(Value::Folded {
                            plain: Bytes::from_static(b" "),
                            posix: posix_fold_bytes(trailing, 1 + extra),
                        });
                    }
                    i = end;
                }
                b = i;
                continue;
            }
            let n = remaining[1];
            if n == b'\\' {
                i += 2;
                continue;
            }
            if n == b'#' && should_handle_comments(opt) {
                session
                    .values
                    .push_fragment(Value::Literal(None, s.slice(b..i)));
                i += 1;
                b = i;
                i += 1;
                continue;
            }
        }

        i += 1;
    }

    if i > b {
        let mut rest = &s[b..i];
        if trim_right_sp {
            rest = trim_right_space(rest);
        }
        if !rest.is_empty() {
            session
                .values
                .push_fragment(Value::Literal(None, s.slice_ref(rest)))
        }
    }
    if session.values.scratch_above(mark) == 1 {
        Ok((i, session.values.pop_scratch()))
    } else {
        let items = session.values.take_scratch(mark);
        Ok((i, session.values.alloc(Value::List(Some(list_loc), items))))
    }
}

pub fn parse_expr(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    opt: ParseExprOpt,
) -> Result<ValueId> {
    let (_i, val) = parse_expr_impl(session, loc, s, None, opt, false)?;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parsed value as a shape, for a test that is about what the read built
    /// rather than about what expanding it produces.
    ///
    /// A rendering rather than a comparison against a literal tree, because a
    /// tree is now a session's arena and handles into it: the expected shape
    /// has no way to name a handle, and the rendering says the same thing about
    /// the same structure while staying readable in a failure.
    fn shape(session: &Session, id: ValueId) -> String {
        fn text(bytes: &Bytes) -> String {
            format!("{:?}", String::from_utf8_lossy(bytes))
        }
        fn run(session: &Session, items: Children) -> String {
            session
                .values
                .children(items)
                .iter()
                .map(|item| shape(session, *item))
                .collect::<Vec<_>>()
                .join(" ")
        }
        match session.values.get(id) {
            Value::Literal(_, lit) => format!("(lit {})", text(lit)),
            Value::Unreadable(_, why) => format!("(unreadable {why:?})"),
            Value::List(_, items) => format!("(list {})", run(session, *items)),
            Value::SymRef(_, sym) => format!("(sym {})", sym.display(session)),
            Value::VarRef(_, name) => format!("(varref {})", shape(session, *name)),
            Value::VarSubst {
                name, pat, subst, ..
            } => format!(
                "(varsubst {} {} {})",
                shape(session, *name),
                shape(session, *pat),
                shape(session, *subst)
            ),
            Value::Func { fi, args, .. } => format!(
                "(func {} {})",
                String::from_utf8_lossy(fi.name),
                run(session, *args)
            ),
            Value::Folded { plain, posix } => {
                format!("(folded {} {})", text(plain), text(posix))
            }
        }
    }

    /// What a value parses to, as the bytes a literal-only expansion produces.
    fn literal_text(source: &'static [u8], opt: ParseExprOpt) -> String {
        let mut session = Session::new();
        let value = parse_expr(
            &mut session,
            &mut Loc::default(),
            Bytes::from_static(source),
            opt,
        )
        .expect("a parsed value");
        let mut out = Vec::new();
        fn walk(session: &Session, id: ValueId, out: &mut Vec<u8>) {
            match session.values.get(id) {
                Value::Literal(_, text) => out.extend_from_slice(text),
                Value::List(_, list) => {
                    for item in session.values.children(*list) {
                        walk(session, *item, out);
                    }
                }
                // These cases assert the reading a fold has without `.POSIX:`,
                // which is what `Folded` holds as its `plain` half.
                Value::Folded { plain, .. } => out.extend_from_slice(plain),
                // A reference to an unset variable is empty, and every name
                // these cases produce is one nothing ever assigns.
                _ => {}
            }
        }
        walk(&session, value, &mut out);
        String::from_utf8_lossy(&out).into_owned()
    }

    /// GNU Make's expander gives `$$` and a `$` at the end of the text the same
    /// arm, so both are one written dollar.
    #[test]
    fn a_dollar_with_nothing_after_it_is_a_written_dollar() {
        assert_eq!(literal_text(b"x$", ParseExprOpt::Normal), "x$");
        assert_eq!(literal_text(b"$", ParseExprOpt::Normal), "$");
        assert_eq!(literal_text(b"x$$", ParseExprOpt::Normal), "x$");
        assert_eq!(literal_text(b"x$$$", ParseExprOpt::Normal), "x$$");
        assert_eq!(literal_text(b"x  $", ParseExprOpt::Normal), "x  $");
        assert_eq!(literal_text(b"A$", ParseExprOpt::Command), "A$");
    }

    #[test]
    fn an_odd_backslash_run_before_a_newline_keeps_half_of_itself() {
        assert_eq!(continuation_fold(b"\\\nx"), Some((0, 2)));
        assert_eq!(continuation_fold(b"\\\\\\\nx"), Some((1, 4)));
        assert_eq!(continuation_fold(b"\\\\\\\\\\\nx"), Some((2, 6)));
        assert_eq!(continuation_fold(b"\\\r\nx"), Some((0, 3)));
    }

    #[test]
    fn an_even_backslash_run_does_not_continue_the_line() {
        assert_eq!(continuation_fold(b"\\\\\nx"), None);
        assert_eq!(continuation_fold(b"\\\\\\\\\nx"), None);
        assert_eq!(continuation_fold(b"\\x"), None);
        assert_eq!(continuation_fold(b"\\"), None);
        assert_eq!(continuation_fold(b"x\\\n"), None);
    }

    #[test]
    fn a_continuation_becomes_one_space_however_much_it_spans() {
        assert_eq!(literal_text(b"a\\\n  b", ParseExprOpt::Normal), "a b");
        assert_eq!(literal_text(b"a   \\\n\t\tb", ParseExprOpt::Normal), "a b");
        assert_eq!(literal_text(b"a\\\n\\\n  b", ParseExprOpt::Normal), "a b");
        assert_eq!(literal_text(b"a\\\\\\\n  b", ParseExprOpt::Normal), "a\\ b");
        assert_eq!(
            literal_text(b"a\\\n  \\\\\\\nb", ParseExprOpt::Normal),
            "a \\ b"
        );
    }

    #[test]
    fn a_dollar_before_a_continuation_names_what_the_fold_leaves() {
        // The fold puts a space beside the `$`, so the reference is to the
        // variable whose name is a space, and nothing is left between the two
        // halves of the value.
        assert_eq!(literal_text(b"x$\\\n  y", ParseExprOpt::Normal), "xy");
        assert_eq!(literal_text(b"x$\\\n  y", ParseExprOpt::Define), "xy");
        // With backslashes surviving the fold it is the first of them that is
        // named, and the rest of them and the space are value text.
        assert_eq!(literal_text(b"x$\\\\\\\n  y", ParseExprOpt::Normal), "x y");
        assert_eq!(
            literal_text(b"x$\\\\\\\\\\\n  y", ParseExprOpt::Normal),
            "x\\ y"
        );
    }

    #[test]
    fn a_recipe_hands_its_continuation_to_the_shell() {
        assert_eq!(
            literal_text(b"echo a\\\nb", ParseExprOpt::Command),
            "echo a\\\nb"
        );
        // `$\` in a recipe is a reference to the variable named `\`, and the
        // newline behind it stays where it is rather than folding away.
        assert_eq!(
            literal_text(b"echo x$\\\ny", ParseExprOpt::Command),
            "echo x\ny"
        );
    }

    /// Read `source` and render what it built.
    fn parsed_shape(source: &'static [u8], opt: ParseExprOpt) -> String {
        let mut session = Session::new();
        let value = parse_expr(
            &mut session,
            &mut Loc::default(),
            Bytes::from_static(source),
            opt,
        )
        .expect("a parsed value");
        shape(&session, value)
    }

    #[test]
    fn test_parse_expr() {
        assert_eq!(parsed_shape(b"foo", ParseExprOpt::Normal), r#"(lit "foo")"#);
        assert_eq!(parsed_shape(b"$(foo)", ParseExprOpt::Normal), "(sym foo)");
    }

    #[test]
    fn test_eval_define_simplified() {
        assert_eq!(
            parsed_shape(b"$(eval dst := $$(notdir $$(src)))", ParseExprOpt::Define),
            concat!(
                r#"(func eval (list (lit "dst := ") (lit "$") "#,
                r#"(lit "(notdir ") (lit "$") (lit "(src))")))"#
            )
        )
    }

    #[test]
    fn test_parse_dollar() {
        fn read(source: &'static [u8]) -> (usize, String) {
            let mut session = Session::new();
            let (at, value) = parse_dollar(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(source),
                false,
            )
            .expect("a parsed reference");
            (at, shape(&session, value))
        }
        assert_eq!(read(b"${foo}bar"), (6, "(sym foo)".to_owned()));
        assert_eq!(
            read(b"$(info ***   - Re-execute)"),
            (26, r#"(func info (lit "***   - Re-execute"))"#.to_owned())
        );
        assert_eq!(
            read(b"$(info ***   - Re-execute envsetup (\". envsetup.sh\"))"),
            (
                53,
                r#"(func info (lit "***   - Re-execute envsetup (\". envsetup.sh\")"))"#.to_owned()
            )
        );
    }

    #[test]
    fn test_call_func() {
        assert_eq!(
            parsed_shape(b"$(call to-lower,$(upper))", ParseExprOpt::Normal),
            r#"(func call (lit "to-lower") (sym upper))"#
        )
    }

    #[test]
    fn test_subst2() {
        assert_eq!(
            parsed_shape(b"$(subst $(space),$,,$(foo))", ParseExprOpt::Normal),
            r#"(func subst (sym space) (lit "$") (list (lit ",") (sym foo)))"#
        )
    }

    /// Every shape a read cannot make a call out of is held rather than raised,
    /// and each one raises GNU Make's own words when the text is expanded.
    #[test]
    fn a_call_a_read_cannot_finish_is_held_until_it_is_expanded() {
        for (text, expected) in [
            (
                &b"$(subst a,b,c"[..],
                "<unknown>:0: *** unterminated call to function 'subst': missing ')'.  Stop.",
            ),
            (
                b"${subst a,b,c",
                "<unknown>:0: *** unterminated call to function 'subst': missing '}'.  Stop.",
            ),
            // The text ran out between the name and the first argument, which
            // is a missing close and not a missing argument. Read as a call
            // with no arguments at all it would reach the function's own body,
            // which indexes the list it is handed.
            (
                b"$(subst ",
                "<unknown>:0: *** unterminated call to function 'subst': missing ')'.  Stop.",
            ),
            // A comma was the last thing in the text, likewise.
            (
                b"$(subst a,",
                "<unknown>:0: *** unterminated call to function 'subst': missing ')'.  Stop.",
            ),
            (
                b"$(NAME",
                "<unknown>:0: *** unterminated variable reference.  Stop.",
            ),
            (
                b"$(subst a)",
                "<unknown>:0: *** insufficient number of arguments (1) to function 'subst'.  Stop.",
            ),
        ] {
            let mut session = Session::new();
            let mut loc = Loc::default();
            let value = parse_expr(
                &mut session,
                &mut loc,
                Bytes::from_static(text),
                ParseExprOpt::Normal,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{:?} was refused by the read: {e}",
                    String::from_utf8_lossy(text)
                )
            });
            let mut ev = Evaluator::new(session);
            assert_eq!(
                value.eval_to_buf(&mut ev).unwrap_err().to_string(),
                expected,
                "{:?}",
                String::from_utf8_lossy(text)
            );
        }
    }

    /// Which byte after a name makes the name a call.
    ///
    /// GNU Make's `lookup_function` ends the name at a NUL or at anything
    /// `isspace` accepts, and `variable_expand_string` asks it before it goes
    /// looking for the close -- so the six whitespace bytes and the end of the
    /// text all name a call, and the close paren and the comma, which end a
    /// name for every other purpose, do not.
    #[test]
    fn a_name_is_a_call_wherever_gnu_make_ends_one() {
        for (text, expected) in [
            (&b"$(subst a,b,aaa)"[..], Ok("bbb")),
            (b"$(subst\ta,b,aaa)", Ok("bbb")),
            (b"$(subst\na,b,aaa)", Ok("bbb")),
            (b"$(subst\x0ba,b,aaa)", Ok("bbb")),
            (b"$(subst\x0ca,b,aaa)", Ok("bbb")),
            (b"$(subst\ra,b,aaa)", Ok("bbb")),
            // The close ends the name without being whitespace, so this reads
            // as a reference to an unset variable called `subst` rather than
            // as a call with no arguments at all.
            (b"$(subst)", Ok("")),
            // A comma likewise -- the whole of `subst,a,b,aaa` is the name.
            (b"$(subst,a,b,aaa)", Ok("")),
            // The end of the text ends the name, and the close it wanted is
            // then plainly missing.
            (
                b"$(subst",
                Err("<unknown>:0: *** unterminated call to function 'subst': missing ')'.  Stop."),
            ),
            (
                b"${subst",
                Err("<unknown>:0: *** unterminated call to function 'subst': missing '}'.  Stop."),
            ),
            // A name that is not a function's, ended the same way, is the
            // reference it looks like.
            (
                b"$(substx",
                Err("<unknown>:0: *** unterminated variable reference.  Stop."),
            ),
            (
                b"$(sub",
                Err("<unknown>:0: *** unterminated variable reference.  Stop."),
            ),
            // The name ran into a comma rather than into the end of the text,
            // so `lookup_function` never accepted it and the missing close is
            // the reference's.
            (
                b"$(subst,a,b",
                Err("<unknown>:0: *** unterminated variable reference.  Stop."),
            ),
        ] {
            let mut session = Session::new();
            let mut loc = Loc::default();
            let value = parse_expr(
                &mut session,
                &mut loc,
                Bytes::from_static(text),
                ParseExprOpt::Normal,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{:?} was refused by the read: {e}",
                    String::from_utf8_lossy(text)
                )
            });
            let mut ev = Evaluator::new(session);
            let got = value
                .eval_to_buf(&mut ev)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .map_err(|e| e.to_string());
            let got = match &got {
                Ok(text) => Ok(text.as_str()),
                Err(text) => Err(text.as_str()),
            };
            assert_eq!(got, expected, "{:?}", String::from_utf8_lossy(text));
        }
    }

    #[test]
    fn test_ckati_end_paren() {
        // ckati does not error on lines like `ifeq (foo,$(BAR)` as parse_expr
        // gets `$(BAR`, but reads off the end of the string view to find the
        // ending `)`.
        let mut session = Session::new();
        let mut loc = Loc::default();
        let (consumed, unread) = parse_expr_impl_ext(
            &mut session,
            &mut loc,
            Bytes::from_static(b"$(BAR"),
            None,
            ParseExprOpt::Normal,
            false,
            false,
        )
        .unwrap();
        // The read does not raise: it hands back the complaint to be made if and
        // when something expands the text.
        assert_eq!(consumed, 5);
        let Value::Unreadable(held_loc, Unreadable::UnterminatedReference) =
            session.values.get(unread)
        else {
            panic!(
                "expected a held complaint, got {:?}",
                session.values.get(unread)
            )
        };
        assert_eq!(*held_loc, Loc::default());
        let mut ev = Evaluator::new(session);
        assert_eq!(
            unread.eval_to_buf(&mut ev).unwrap_err().to_string(),
            // GNU Make ends the diagnostic it dies on with `Stop.`, wherever it
            // was raised, and this is one it dies on.
            "<unknown>:0: *** unterminated variable reference.  Stop."
        );
        let mut session = ev.session;
        let (consumed, read) = parse_expr_impl_ext(
            &mut session,
            &mut loc,
            Bytes::from_static(b"$(BAR"),
            None,
            ParseExprOpt::Normal,
            false,
            true,
        )
        .unwrap();
        assert_eq!(consumed, 6);
        assert_eq!(shape(&session, read), "(sym BAR)");
    }
}
