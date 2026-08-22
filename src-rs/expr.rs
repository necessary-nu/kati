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

use std::sync::Arc;

use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use memchr::memchr;

use crate::eval::{Evaluator, FrameType};
use crate::func::{FuncInfo, get_func_info};
use crate::loc::Loc;
use crate::session::Session;
use crate::strutil::{Pattern, WordWriter, trim_right_space, trim_suffix, word_scanner};
use crate::symtab::{Symbol, Symtab};
use crate::{error_loc, kati_warn_loc, log};

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

    // Whether this Evaluable is either knowably a function (e.g. one of the
    // built-ins) or likely to be a function-type macro (i.e. one that has
    // positional $(1) arguments to be expanded inside it. However, this is
    // only a heuristic guess. In order to not actually evaluate the expression,
    // because doing so could have side effects like calling $(error ...) or
    // doing a nested eval that assigns variables, we don't handle the case where
    // the variable name is itself a variable expansion inside a deferred
    // expansion variable, and return true in that case. Implementations of this
    // function must also not mark variables as used, as that can trigger unwanted
    // warnings. They should use ev->PeekVar().
    fn is_func(&self, names: &Symtab) -> bool;
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ParseExprOpt {
    Normal,
    Define,
    Command,
    Func,
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
    List(Option<Loc>, Vec<Arc<Value>>),
    SymRef(Loc, Symbol),
    VarRef(Loc, Arc<Value>),
    VarSubst {
        loc: Loc,
        name: Arc<Value>,
        pat: Arc<Value>,
        subst: Arc<Value>,
    },
    Func {
        loc: Loc,
        fi: &'static FuncInfo,
        args: Vec<Arc<Value>>,
    },
}

impl Evaluable for Value {
    fn eval(&self, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
        match self {
            Value::Literal(_, lit) => out.put_slice(lit),
            Value::List(_, vec) => {
                for v in vec {
                    v.eval(ev, out)?;
                }
            }
            Value::SymRef(_, sym) => {
                let sym = *sym;
                let is_make =
                    ev.is_evaluating_command && sym.as_bytes(&ev.session).as_ref() == b"MAKE";
                if let Some(var) = ev.lookup_var_for_eval(sym)? {
                    let v = var.read();
                    // The reference is where GNU Make installs the location a
                    // diagnostic raised inside the value will carry --
                    // `recursively_expand_for_file` in expand.c.
                    ev.enter_expanding_var(v.expansion_loc());
                    v.used(ev, &sym)?;
                    if is_make {
                        let expanded = v.eval_to_buf(ev)?;
                        out.put_slice(&expanded);
                        ev.expanded_make_in_command.push(expanded);
                    } else {
                        v.eval(ev, out)?;
                    }
                    let loc = ev.loc.clone();
                    v.check_current_referencing_file(&ev.session, &loc, sym)?;
                    drop(v);
                    ev.leave_expanding_var();
                    ev.var_eval_complete(&var);
                }
            }
            Value::VarRef(_, var) => {
                ev.eval_depth += 1;
                let name = var.eval_to_buf(ev)?;
                ev.eval_depth -= 1;
                let sym = ev.session.intern(name);
                let is_make =
                    ev.is_evaluating_command && sym.as_bytes(&ev.session).as_ref() == b"MAKE";
                if let Some(var) = ev.lookup_var_for_eval(sym)? {
                    let v = var.read();
                    // The reference is where GNU Make installs the location a
                    // diagnostic raised inside the value will carry --
                    // `recursively_expand_for_file` in expand.c.
                    ev.enter_expanding_var(v.expansion_loc());
                    v.used(ev, &sym)?;
                    if is_make {
                        let expanded = v.eval_to_buf(ev)?;
                        out.put_slice(&expanded);
                        ev.expanded_make_in_command.push(expanded);
                    } else {
                        v.eval(ev, out)?;
                    }
                    let loc = ev.loc.clone();
                    v.check_current_referencing_file(&ev.session, &loc, sym)?;
                    drop(v);
                    ev.leave_expanding_var();
                    ev.var_eval_complete(&var);
                }
            }
            Value::VarSubst {
                loc: _,
                name,
                pat,
                subst,
            } => {
                ev.eval_depth += 1;
                let name = name.eval_to_buf(ev)?;
                let sym = ev.session.intern(name);
                let v = ev.lookup_var(sym)?;
                let pat_str = pat.eval_to_buf(ev)?;
                let subst = subst.eval_to_buf(ev)?;
                ev.eval_depth -= 1;
                if let Some(var) = v {
                    let v = var.read();
                    // `$(V:a=b)` reaches V's value through `recursively_expand`
                    // as well, so it installs the location too.
                    ev.enter_expanding_var(v.expansion_loc());
                    v.used(ev, &sym)?;
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
            Value::Unreadable(loc, unreadable) => {
                let at = ev.expanding_var_loc().unwrap_or_else(|| loc.clone());
                return Err(unreadable.raise(ev, &at));
            }
            Value::Func { loc, fi, args } => {
                let _frame = ev.enter(FrameType::FunCall, Bytes::from_static(fi.name), loc.clone());
                log!(
                    "Invoke func {}({:?})",
                    String::from_utf8_lossy(fi.name),
                    args
                );
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
                        for arg in args {
                            arg.eval_to_buf(ev)?;
                        }
                    }
                    let at = ev.expanding_var_loc().unwrap_or_else(|| loc.clone());
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
        Ok(())
    }

    fn is_func(&self, names: &Symtab) -> bool {
        match self {
            Value::Func { .. } => true,
            Value::List(_, list) => list.iter().any(|v| v.is_func(names)),
            Value::SymRef(_, sym) => {
                // This is a heuristic, where say that if a variable has positional
                // parameters, we think it is likely to be a function. Callers can use
                // .KATI_SYMBOLS to extract variables and their values, without evaluating
                // macros that are likely to have side effects.
                crate::strutil::is_integer(&sym.as_bytes(names))
            }
            Value::VarRef(_, _) => {
                // This is the unhandled edge case as described in the Evaluable::is_func
                true
            }
            Value::VarSubst {
                name, pat, subst, ..
            } => name.is_func(names) || pat.is_func(names) || subst.is_func(names),
            // Evaluating it raises, which is the one thing a caller asking this
            // question is trying not to provoke.
            Value::Unreadable(_, _) => true,
            Value::Literal(_, _) => false,
        }
    }
}

impl Value {
    pub fn loc(&self) -> Option<Loc> {
        match self {
            Value::Literal(loc, _) => loc.clone(),
            Value::Unreadable(loc, _) => Some(loc.clone()),
            Value::List(loc, _) => loc.clone(),
            Value::SymRef(loc, _) => Some(loc.clone()),
            Value::VarRef(loc, _) => Some(loc.clone()),
            Value::VarSubst { loc, .. } => Some(loc.clone()),
            Value::Func { loc, .. } => Some(loc.clone()),
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

fn should_handle_comments(opt: ParseExprOpt) -> bool {
    !matches!(opt, ParseExprOpt::Define | ParseExprOpt::Command)
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
/// the far side of the newline, and any further continuation those blanks lead
/// to.
///
/// GNU Make writes one space per folded newline but first discards the blanks
/// it has already written, so a run of continuations separated by nothing but
/// blanks comes out as one space however long it is. A continuation whose run
/// leaves a backslash behind ends the absorbing, because that backslash is
/// value text that has to be written before the next space.
fn skip_folded(loc: &mut Loc, s: &[u8], mut at: usize) -> usize {
    loop {
        while matches!(s.get(at), Some(b' ' | b'\t')) {
            at += 1;
        }
        let Some((0, consumed)) = continuation_fold(&s[at..]) else {
            return at;
        };
        loc.line += 1;
        at += consumed;
    }
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
    Args(Vec<Arc<Value>>),
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
    let mut args = Vec::new();
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
        args.push(val);
        i += n;
        if i == s.len() {
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
            return Ok((i, ParsedCall::Unterminated));
        }
    }

    Ok((i, ParsedCall::Args(args)))
}

fn parse_dollar(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    end_paren: bool,
) -> Result<(usize, Arc<Value>)> {
    assert!(s.len() >= 2);
    assert!(s.starts_with(b"$"));
    assert!(!s.starts_with(b"$$"));

    let start_loc = loc.clone();

    let Some(cp) = close_paren(s[1]) else {
        let sym = session.intern(s.slice(1..2));
        return Ok((2, Arc::new(Value::SymRef(start_loc.clone(), sym))));
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
        if name_ended
            && let Value::Literal(_, lit) = &*vname
            && let Some(fi) = get_func_info(lit)
        {
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
            return Ok((idx, Arc::new(value)));
        }

        if t.first() == Some(&cp) || (end_paren && t.is_empty() && cp == b')') {
            if let Value::Literal(_, lit) = &*vname {
                let sym = session.intern(lit.clone());
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
                return Ok((i + 1, Arc::new(Value::SymRef(start_loc, sym))));
            }
            return Ok((i + 1, Arc::new(Value::VarRef(start_loc, vname))));
        }

        if name_ended && !t.is_empty() {
            if let Value::Literal(_, lit) = &*vname {
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
                return Ok((
                    i + 1,
                    Arc::new(Value::VarRef(
                        start_loc.clone(),
                        Arc::new(Value::List(
                            Some(start_loc),
                            vec![
                                vname,
                                Arc::new(Value::Literal(None, Bytes::from_static(b":"))),
                                pat,
                            ],
                        )),
                    )),
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
                Arc::new(Value::VarSubst {
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
            return Ok((s.len(), Arc::new(Value::SymRef(start_loc.clone(), sym))));
        }

        // Held rather than raised: GNU Make finds an unterminated reference in
        // `variable_expand_string`, which only ever looks at text it is
        // expanding. See [`Unreadable`].
        return Ok((
            s.len(),
            Arc::new(Value::Unreadable(
                start_loc,
                Unreadable::UnterminatedReference,
            )),
        ));
    }
}

pub fn parse_expr_impl(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    terms: Option<&[u8]>,
    opt: ParseExprOpt,
    trim_right_sp: bool,
) -> Result<(usize, Arc<Value>)> {
    parse_expr_impl_ext(session, loc, s, terms, opt, trim_right_sp, false)
}

pub fn parse_expr_impl_ext(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    terms: Option<&[u8]>,
    opt: ParseExprOpt,
    trim_right_sp: bool,
    // This is for compatibility with a read-past-end in ckati
    end_paren: bool,
) -> Result<(usize, Arc<Value>)> {
    let list_loc = loc.clone();

    let s = s.slice_ref(trim_suffix(&s, b"\r"));

    let mut b = 0usize;
    let mut save_paren: Option<u8> = None;
    let mut paren_depth: i32 = 0;
    let mut i = 0usize;
    let mut list: Vec<Arc<Value>> = Vec::new();
    let mut terms_ignored = 0;

    while i < s.len() {
        let item_loc = loc.clone();

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
                list.push(Arc::new(Value::Literal(None, s.slice(b..i))));
            }
            let mut was_backslash = false;
            while i < s.len() && s[i] != b'\n' || was_backslash {
                was_backslash = !was_backslash && s[i] == b'\\';
                i += 1;
            }
            if list.len() == 1 {
                return Ok((i, list.pop().unwrap()));
            }
            return Ok((i, Arc::new(Value::List(Some(item_loc), list))));
        }

        if c == b'$' {
            if i > b {
                list.push(Arc::new(Value::Literal(None, s.slice(b..i))));
            }

            // A `$` with nothing after it is one literal dollar, exactly as
            // `$$` is: GNU Make's expander gives the two the same arm
            // (expand.c variable_expand_string, `case '$': case '\0':`). The
            // blanks in front of it are not trailing any more, so a caller
            // asking for a right trim does not reach them.
            if i + 1 >= s.len() {
                list.push(Arc::new(Value::Literal(None, Bytes::from_static(b"$"))));
                i += 1;
                b = i;
                continue;
            }

            if remaining.starts_with(b"$$") {
                list.push(Arc::new(Value::Literal(None, Bytes::from_static(b"$"))));
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
                let val = Arc::new(Value::Literal(None, Bytes::from_static(b"$")));
                if list.is_empty() {
                    return Ok((i + 1, val));
                }
                list.push(val);
                return Ok((i + 1, Arc::new(Value::List(Some(item_loc), list))));
            }

            if let Some((kept, consumed)) = folded {
                loc.line += 1;
                let name = if kept == 0 { &b" "[..] } else { &b"\\"[..] };
                let sym = session.intern(Bytes::from_static(name));
                list.push(Arc::new(Value::SymRef(item_loc, sym)));
                // The reference took the first backslash the run kept. The rest
                // of them, and the space the newline became, are value text.
                if kept > 0 {
                    list.push(Arc::new(Value::Literal(None, s.slice(i + 2..i + 1 + kept))));
                    list.push(Arc::new(Value::Literal(None, Bytes::from_static(b" "))));
                }
                i = skip_folded(loc, &s, i + 1 + consumed);
                b = i;
                continue;
            }

            let (n, v) = parse_dollar(session, loc, s.slice(i..), end_paren)?;
            list.push(v);
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
                if literal_end > b {
                    let text = &s[b..literal_end];
                    let text = if kept == 0 {
                        trim_right_space(text)
                    } else {
                        text
                    };
                    list.push(Arc::new(Value::Literal(None, s.slice_ref(text))));
                }
                list.push(Arc::new(Value::Literal(None, Bytes::from_static(b" "))));
                i = skip_folded(loc, &s, i + consumed);
                b = i;
                continue;
            }
            let n = remaining[1];
            if n == b'\\' {
                i += 2;
                continue;
            }
            if n == b'#' && should_handle_comments(opt) {
                list.push(Arc::new(Value::Literal(None, s.slice(b..i))));
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
            list.push(Arc::new(Value::Literal(None, s.slice_ref(rest))))
        }
    }
    if list.len() == 1 {
        Ok((i, list.pop().unwrap()))
    } else {
        Ok((i, Arc::new(Value::List(Some(list_loc), list))))
    }
}

pub fn parse_expr(
    session: &mut Session,
    loc: &mut Loc,
    s: Bytes,
    opt: ParseExprOpt,
) -> Result<Arc<Value>> {
    let (_i, val) = parse_expr_impl(session, loc, s, None, opt, false)?;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn walk(value: &Value, out: &mut Vec<u8>) {
            match value {
                Value::Literal(_, text) => out.extend_from_slice(text),
                Value::List(_, list) => list.iter().for_each(|v| walk(v, out)),
                // A reference to an unset variable is empty, and every name
                // these cases produce is one nothing ever assigns.
                _ => {}
            }
        }
        walk(&value, &mut out);
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

    #[test]
    fn test_parse_expr() {
        let mut session = Session::new();
        assert_eq!(
            parse_expr(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"foo"),
                ParseExprOpt::Normal
            )
            .unwrap(),
            Arc::new(Value::Literal(None, Bytes::from_static(b"foo")))
        );
        let foo = session.intern("foo");
        assert_eq!(
            parse_expr(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"$(foo)"),
                ParseExprOpt::Normal
            )
            .unwrap(),
            Arc::new(Value::SymRef(Loc::default(), foo))
        );
    }

    #[test]
    fn test_eval_define_simplified() {
        let mut session = Session::new();
        let s = Bytes::from_static(b"$(eval dst := $$(notdir $$(src)))");
        assert_eq!(
            parse_expr(&mut session, &mut Loc::default(), s, ParseExprOpt::Define).unwrap(),
            Arc::new(Value::Func {
                loc: Loc::default(),
                fi: get_func_info(b"eval").unwrap(),
                args: vec![Arc::new(Value::List(
                    Some(Loc::default()),
                    vec![
                        Arc::new(Value::Literal(None, Bytes::from_static(b"dst := "))),
                        Arc::new(Value::Literal(None, Bytes::from_static(b"$"))),
                        Arc::new(Value::Literal(None, Bytes::from_static(b"(notdir "))),
                        Arc::new(Value::Literal(None, Bytes::from_static(b"$"))),
                        Arc::new(Value::Literal(None, Bytes::from_static(b"(src))"))),
                    ]
                ))],
            })
        )
    }

    #[test]
    fn test_parse_dollar() {
        let mut session = Session::new();
        let foo = session.intern("foo");
        assert_eq!(
            parse_dollar(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"${foo}bar"),
                false
            )
            .unwrap(),
            (6, Arc::new(Value::SymRef(Loc::default(), foo)))
        );
        assert_eq!(
            parse_dollar(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"$(info ***   - Re-execute)"),
                false,
            )
            .unwrap(),
            (
                26,
                Arc::new(Value::Func {
                    loc: Loc::default(),
                    fi: get_func_info(b"info").unwrap(),
                    args: vec![Arc::new(Value::Literal(
                        None,
                        Bytes::from_static(b"***   - Re-execute")
                    ))],
                })
            )
        );
        assert_eq!(
            parse_dollar(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"$(info ***   - Re-execute envsetup (\". envsetup.sh\"))"),
                false,
            )
            .unwrap(),
            (
                53,
                Arc::new(Value::Func {
                    loc: Loc::default(),
                    fi: get_func_info(b"info").unwrap(),
                    args: vec![Arc::new(Value::Literal(
                        None,
                        Bytes::from_static(b"***   - Re-execute envsetup (\". envsetup.sh\")")
                    ))],
                })
            )
        );
    }

    #[test]
    fn test_call_func() {
        let mut session = Session::new();
        let upper = session.intern("upper");
        assert_eq!(
            parse_expr(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"$(call to-lower,$(upper))"),
                ParseExprOpt::Normal
            )
            .unwrap(),
            Arc::new(Value::Func {
                loc: Loc::default(),
                fi: get_func_info(b"call").unwrap(),
                args: vec![
                    Arc::new(Value::Literal(None, Bytes::from_static(b"to-lower"))),
                    Arc::new(Value::SymRef(Loc::default(), upper)),
                ],
            })
        )
    }

    #[test]
    fn test_subst2() {
        let mut session = Session::new();
        let space = session.intern("space");
        let foo = session.intern("foo");
        assert_eq!(
            parse_expr(
                &mut session,
                &mut Loc::default(),
                Bytes::from_static(b"$(subst $(space),$,,$(foo))"),
                ParseExprOpt::Normal
            )
            .unwrap(),
            Arc::new(Value::Func {
                loc: Loc::default(),
                fi: get_func_info(b"subst").unwrap(),
                args: vec![
                    Arc::new(Value::SymRef(Loc::default(), space)),
                    Arc::new(Value::Literal(None, Bytes::from_static(b"$"))),
                    Arc::new(Value::List(
                        Some(Loc::default()),
                        vec![
                            Arc::new(Value::Literal(None, Bytes::from_static(b","))),
                            Arc::new(Value::SymRef(Loc::default(), foo)),
                        ]
                    )),
                ],
            })
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
        let Value::Unreadable(held_loc, Unreadable::UnterminatedReference) = &*unread else {
            panic!("expected a held complaint, got {unread:?}")
        };
        assert_eq!(held_loc, &Loc::default());
        let mut ev = Evaluator::new(session);
        assert_eq!(
            unread.eval_to_buf(&mut ev).unwrap_err().to_string(),
            // GNU Make ends the diagnostic it dies on with `Stop.`, wherever it
            // was raised, and this is one it dies on.
            "<unknown>:0: *** unterminated variable reference.  Stop."
        );
        let mut session = ev.session;
        let bar = session.intern("BAR");
        assert_eq!(
            parse_expr_impl_ext(
                &mut session,
                &mut loc,
                Bytes::from_static(b"$(BAR"),
                None,
                ParseExprOpt::Normal,
                false,
                true
            )
            .unwrap(),
            (6, Arc::new(Value::SymRef(loc, bar)))
        );
    }
}
