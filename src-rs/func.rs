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

use std::{
    ffi::OsStr,
    fmt::Debug,
    fs::File,
    io::{Read, Write},
    os::unix::{ffi::OsStrExt, process::ExitStatusExt},
    sync::LazyLock,
};

use anyhow::Result;
use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::{
    build_sink::NewInputsTiming,
    collect_stats, collect_stats_with_slow_report,
    command::DEFERRED_NEW_INPUTS_REFERENCE,
    error_loc,
    eval::{Evaluator, FrameType},
    expr::{Children, Evaluable, Value},
    fasthash::FastMap,
    fileutil::{RedirectStderr, run_command},
    find::FindCommand,
    kati_warn_loc,
    loc::Loc,
    log,
    parser::parse_buf,
    session::{Asked, GroundQuestion, Session},
    strutil::{
        Pattern, WordWriter, escape_printf_b, format_for_command_substitution,
        format_for_shell_assignment, is_space_byte, trim_left_space, trim_space, word_scanner,
    },
    var::{VarOrigin, Variable},
    warn_loc,
};

type MakeFuncImpl = fn(Children, &mut Evaluator, &mut dyn BufMut) -> Result<()>;

pub struct FuncInfo {
    pub name: &'static [u8],
    pub func: MakeFuncImpl,
    pub arity: i16,
    pub min_arity: i16,
    // For all parameters.
    pub trim_space: bool,
    // Only for the first parameter.
    pub trim_right_space_1st: bool,
    /// Whether GNU Make expands this function's arguments before the function
    /// sees them.
    ///
    /// Every function here expands its own arguments, so this changes nothing
    /// about a direct call. It decides what `$(call)` hands over: `$(call)`
    /// expands its arguments first, and a function that would have expanded
    /// them again then does — `$(call foreach,v,1 2,$$(v))` iterates, where
    /// `$(call notdir,$$(V))` gets the literal text `$(V)`.
    pub pre_expanded_args: bool,
}

// Function pointers are not comparable, so just compare by name
impl PartialEq for FuncInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Debug for FuncInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Func({})", String::from_utf8_lossy(self.name))
    }
}

// TODO: This code is very similar to
// NinjaGenerator::TranslateCommand. Factor them out.
fn strip_shell_comment(cmd: Bytes) -> Bytes {
    if !cmd.contains(&b'#') {
        return cmd;
    }

    let mut res = BytesMut::new();
    let mut prev_backslash = false;
    // Set space as an initial value so the leading comment will be
    // stripped out.
    let mut prev_char = b' ';
    let mut quote = None;
    let mut inp = cmd;
    while !inp.is_empty() {
        let c = inp[0];
        match c {
            b'#' => {
                if quote.is_none() && prev_char.is_ascii_whitespace() {
                    while inp.len() > 1 && !inp.starts_with(b"\n") {
                        inp.advance(1);
                    }
                } else {
                    if let Some(q) = quote {
                        if q == c {
                            quote = None;
                        }
                    } else if !prev_backslash {
                        quote = Some(c);
                    }
                    res.put_u8(c);
                }
            }
            b'\'' | b'"' | b'`' => {
                if let Some(q) = quote {
                    if q == c {
                        quote = None;
                    }
                } else if !prev_backslash {
                    quote = Some(c);
                }
                res.put_u8(c);
            }
            _ => res.put_u8(c),
        }

        if inp.starts_with(b"\\") {
            prev_backslash = !prev_backslash;
        } else {
            prev_backslash = false;
        }

        prev_char = c;
        inp.advance(1);
    }
    res.into()
}

fn patsubst_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pat_str = ev.eval_arg(args, 0)?;
    let repl = ev.eval_arg(args, 1)?;
    let s = ev.eval_arg(args, 2)?;
    let mut ww = WordWriter::new(out);
    let pat = Pattern::new(pat_str);
    for tok in word_scanner(&s) {
        let tok = s.slice_ref(tok);
        ww.write(&pat.append_subst(&tok, &repl));
    }
    Ok(())
}

fn strip_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let s = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&s) {
        ww.write(tok);
    }
    Ok(())
}

fn subst_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pat = ev.eval_arg(args, 0)?;
    let repl = ev.eval_arg(args, 1)?;
    let s = ev.eval_arg(args, 2)?;
    if pat.is_empty() {
        out.put_slice(&s);
        out.put_slice(&repl);
        return Ok(());
    }
    let f = memchr::memmem::Finder::new(&pat);
    let mut remainder = s.as_ref();
    while !remainder.is_empty() {
        let Some(found) = f.find(remainder) else {
            out.put_slice(remainder);
            break;
        };
        out.put_slice(&remainder[..found]);
        out.put_slice(&repl);
        remainder = &remainder[found + pat.len()..];
    }
    Ok(())
}

fn findstring_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let find = ev.eval_arg(args, 0)?;
    let f = memchr::memmem::Finder::new(&find);
    let haystack = ev.eval_arg(args, 1)?;
    if f.find(&haystack).is_some() {
        out.put_slice(&find);
    }
    Ok(())
}

fn filter_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pat_buf = ev.eval_arg(args, 0)?;
    let text = ev.eval_arg(args, 1)?;
    let pats: Vec<Pattern> = word_scanner(&pat_buf)
        .map(|p| Pattern::new(pat_buf.slice_ref(p)))
        .collect();
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        for pat in &pats {
            if pat.matches(tok) {
                ww.write(tok);
                break;
            }
        }
    }
    Ok(())
}

fn filter_out_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pat_buf = ev.eval_arg(args, 0)?;
    let text = ev.eval_arg(args, 1)?;
    let pats: Vec<Pattern> = word_scanner(&pat_buf)
        .map(|p| Pattern::new(pat_buf.slice_ref(p)))
        .collect();
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        if ev.new_inputs_timing == NewInputsTiming::SchedulerBoundary
            && tok == DEFERRED_NEW_INPUTS_REFERENCE
        {
            ev.deferred_new_inputs_filter_out
                .extend(pats.iter().map(|pattern| pattern.as_bytes().clone()));
            ww.write(tok);
            continue;
        }
        let mut matched = false;
        for pat in &pats {
            if pat.matches(tok) {
                matched = true;
                break;
            }
        }
        if !matched {
            ww.write(tok);
        }
    }
    Ok(())
}

fn sort_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let list = ev.eval_arg(args, 0)?;
    collect_stats!(ev, "func sort time");
    let mut toks: Vec<&[u8]> = word_scanner(&list).collect();
    toks.sort();
    let mut ww = WordWriter::new(out);
    let mut prev = [].as_slice();
    for tok in toks {
        if tok != prev {
            ww.write(tok);
            prev = tok;
        }
    }
    Ok(())
}

/// GNU Make's `parse_numeric`: whitespace either side, an optional sign, and
/// digits, read as the `long long` the index functions go on to compare.
///
/// The three ways it can fail are three different diagnostics, and telling them
/// apart is the point. A value that is all digits but too large to be an index
/// is *out of range* rather than non-numeric, and GNU Make refuses the makefile
/// rather than reading it as an index no list could have — which would answer
/// with the empty string and let a build run that should have stopped.
///
/// `what` names the argument and the function, and is the whole of the message
/// up to the colon, because GNU builds these diagnostics the same way.
fn parse_numeric(text: &[u8], what: &str, ev: &mut Evaluator) -> Result<i64> {
    // GNU Make raises every one of these at `*expanding_var`, so a bad number
    // inside a `define` names the `define` and not the body line.
    let at = ev.expanding_var_loc();
    let trimmed = trim_space(text);
    if trimmed.is_empty() {
        error_loc!(ev, at.as_ref(), "*** {what}: empty value.");
    }
    let negative = trimmed[0] == b'-';
    let digits = &trimmed[usize::from(negative || trimmed[0] == b'+')..];
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        error_loc!(
            ev,
            at.as_ref(),
            "*** {what}: '{}'.",
            String::from_utf8_lossy(text)
        );
    }
    // `strtoll` reports an overflow in either direction as ERANGE, so the
    // accumulation is signed and checked rather than taken as a magnitude:
    // that way the most negative value is not itself an overflow.
    let mut value: i64 = 0;
    for byte in digits {
        let digit = i64::from(byte - b'0');
        let stepped = value.checked_mul(10).and_then(|scaled| {
            if negative {
                scaled.checked_sub(digit)
            } else {
                scaled.checked_add(digit)
            }
        });
        let Some(stepped) = stepped else {
            error_loc!(
                ev,
                at.as_ref(),
                "*** {what}: '{}' out of range.",
                String::from_utf8_lossy(text)
            );
        };
        value = stepped;
    }
    Ok(value)
}

fn word_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let n_str = ev.eval_arg(args, 0)?;
    let mut n = parse_numeric(&n_str, "invalid first argument to 'word' function", ev)?;
    if n < 1 {
        let at = ev.expanding_var_loc();
        error_loc!(
            ev,
            at.as_ref(),
            "*** first argument to 'word' function must be greater than 0."
        );
    }

    let text = ev.eval_arg(args, 1)?;
    for tok in word_scanner(&text) {
        n -= 1;
        if n == 0 {
            out.put_slice(tok);
            break;
        }
    }
    Ok(())
}

fn wordlist_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let bad_first = "invalid first argument to 'wordlist' function";
    let bad_second = "invalid second argument to 'wordlist' function";
    let s_str = ev.eval_arg(args, 0)?;
    let si = parse_numeric(&s_str, bad_first, ev)?;
    if si < 1 {
        // The value as read, not as written: GNU prints the number it parsed,
        // so `$(wordlist 000,…)` is refused as '0'.
        let at = ev.expanding_var_loc();
        error_loc!(ev, at.as_ref(), "*** {bad_first}: '{si}'.");
    }

    let e_str = ev.eval_arg(args, 1)?;
    let ei = parse_numeric(&e_str, bad_second, ev)?;
    if ei < 0 {
        let at = ev.expanding_var_loc();
        error_loc!(ev, at.as_ref(), "*** {bad_second}: '{ei}'.");
    }

    let text = ev.eval_arg(args, 2)?;
    let mut ww = WordWriter::new(out);
    let mut i: i64 = 0;
    for tok in word_scanner(&text) {
        i += 1;
        if si <= i {
            if i <= ei {
                ww.write(tok);
            } else {
                break;
            }
        }
    }
    Ok(())
}

fn words_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let n = word_scanner(&text).count();
    out.put_slice(format!("{n}").as_bytes());
    Ok(())
}

fn firstword_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    if let Some(tok) = word_scanner(&text).next() {
        out.put_slice(tok);
    }
    Ok(())
}

fn lastword_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    if let Some(tok) = word_scanner(&text).last() {
        out.put_slice(tok);
    }
    Ok(())
}

fn join_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let list1 = ev.eval_arg(args, 0)?;
    let list2 = ev.eval_arg(args, 1)?;
    let mut ws1 = word_scanner(&list1);
    let mut ws2 = word_scanner(&list2);
    let mut ww = WordWriter::new(out);
    loop {
        match (ws1.next(), ws2.next()) {
            (Some(tok1), Some(tok2)) => {
                ww.write(tok1);
                ww.out.put_slice(tok2);
            }
            (Some(tok), None) => ww.write(tok),
            (None, Some(tok)) => ww.write(tok),
            (None, None) => break,
        }
    }
    Ok(())
}

fn wildcard_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pat = ev.eval_arg(args, 0)?;
    collect_stats!(ev, "func wildcard time");
    // Note GNU make does not delay the execution of $(wildcard) so we
    // do not need to check avoid_io here.
    let slot = match ev
        .session
        .ground_journal
        .answered(GroundQuestion::Wildcard, &pat)
    {
        Asked::Answered(answered) => {
            out.put_slice(&answered.answer);
            return Ok(());
        }
        Asked::Ask(slot) => slot,
    };
    // Written into a buffer of its own rather than straight out, because the
    // answer is what a later read of this same text is handed. A fresh
    // `WordWriter` produces the same bytes either way: the separating space
    // goes before every word but the writer's first.
    let mut answer = BytesMut::new();
    let mut ww = WordWriter::new(&mut answer);
    for tok in word_scanner(&pat) {
        let tok = pat.slice_ref(tok);
        let files = ev.session.glob(tok);
        if let Ok(files) = files.as_ref() {
            for f in files {
                ww.write(f);
            }
        }
    }
    let answer = answer.freeze();
    out.put_slice(&answer);
    ev.session
        .ground_journal
        .record(slot, GroundQuestion::Wildcard, pat, answer, None);
    Ok(())
}

fn dir_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        let tok = text.slice_ref(tok);
        ww.write(&crate::strutil::dirname(&tok));
        ww.out.put_u8(b'/');
    }
    Ok(())
}

fn notdir_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        ww.write(crate::strutil::basename(tok));
    }
    Ok(())
}

fn suffix_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        if let Some(suf) = crate::strutil::get_ext(tok) {
            ww.write(suf);
        }
    }
    Ok(())
}

fn basename_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        ww.write(crate::strutil::strip_ext(tok));
    }
    Ok(())
}

fn addsuffix_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let suf = ev.eval_arg(args, 0)?;
    let text = ev.eval_arg(args, 1)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        ww.write(tok);
        ww.out.put_slice(&suf);
    }
    Ok(())
}

fn addprefix_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let pre = ev.eval_arg(args, 0)?;
    let text = ev.eval_arg(args, 1)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        ww.write(&pre);
        ww.out.put_slice(tok);
    }
    Ok(())
}

/// GNU Make resolves `$(realpath)` where it reads it, so a word that names
/// nothing on disk — including a symbolic link with no target — leaves the
/// output altogether rather than appearing as an empty word.
///
/// Like `$(wildcard)`, this answers during evaluation even when the value is
/// bound for a recipe: upstream kati deferred it into a helper the recipe's
/// shell would run, which is a different value at a different time, and puts
/// shell syntax where the Makefile asked for a pathname.
fn realpath_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let slot = match ev
        .session
        .ground_journal
        .answered(GroundQuestion::RealPath, &text)
    {
        Asked::Answered(answered) => {
            out.put_slice(&answered.answer);
            return Ok(());
        }
        Asked::Ask(slot) => slot,
    };
    let mut answer = BytesMut::new();
    let mut ww = WordWriter::new(&mut answer);
    for tok in word_scanner(&text) {
        let tok = <OsStr as OsStrExt>::from_bytes(tok);
        if let Ok(path) = std::fs::canonicalize(tok) {
            ww.write(path.as_os_str().as_bytes());
        }
    }
    let answer = answer.freeze();
    out.put_slice(&answer);
    ev.session
        .ground_journal
        .record(slot, GroundQuestion::RealPath, text, answer, None);
    Ok(())
}

fn abspath_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&text) {
        ww.write(&crate::strutil::abs_path(tok)?);
    }
    Ok(())
}

fn if_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let cond = ev.eval_arg(args, 0)?;
    if cond.is_empty() {
        if args.len() > 2 {
            {
                let arg = ev.arg(args, 2);
                crate::expr::eval_value(arg, ev, out)
            }?;
        }
    } else {
        {
            let arg = ev.arg(args, 1);
            crate::expr::eval_value(arg, ev, out)
        }?;
    }
    Ok(())
}

fn and_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let mut cond = Bytes::new();
    for index in 0..args.len() {
        let a = ev.arg(args, index);
        cond = a.eval_to_buf(ev)?;
        if cond.is_empty() {
            return Ok(());
        }
    }
    if !cond.is_empty() {
        out.put_slice(&cond);
    }
    Ok(())
}

fn or_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    for index in 0..args.len() {
        let a = ev.arg(args, index);
        let cond = a.eval_to_buf(ev)?;
        if !cond.is_empty() {
            out.put_slice(&cond);
            break;
        }
    }
    Ok(())
}

fn value_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let var_name = ev.eval_arg(args, 0)?;
    let sym = ev.session.intern(var_name);
    let Some(var) = ev.lookup_var(sym)? else {
        return Ok(());
    };
    // An automatic variable keeps no text for the reader to have stored, so
    // what `value` reads back has to come from how GNU Make defined it. The `D`
    // and `F` forms were defined from an expression and read back as that
    // expression; a base form was defined as a simple variable holding the
    // computed name, which is what evaluating it here produces.
    let automatic = var.read().autocommand();
    if let Some(automatic) = automatic {
        match automatic.definition() {
            Some(text) => out.put_slice(&text),
            None => automatic.eval(ev, out)?,
        }
        return Ok(());
    }
    out.put_slice(&var.read().string(&ev.session)?);
    Ok(())
}

fn eval_func(args: Children, ev: &mut Evaluator, _out: &mut dyn BufMut) -> Result<()> {
    let text = ev.eval_arg(args, 0)?;
    if ev.avoid_io {
        kati_warn_loc!(
            ev,
            ev.loc.as_ref(),
            "*warning*: $(eval) in a recipe is not recommended: {}",
            String::from_utf8_lossy(&text)
        );
    }
    let loc = ev.loc.unwrap_or_default();
    let stmts = parse_buf(&mut ev.session, &text, loc)?;
    let stmts = stmts.lock().clone();
    for stmt in stmts.iter() {
        log!("{:?}", stmt);
        stmt.eval(ev)?;
    }
    Ok(())
}

// A hack for Android build. We need to evaluate things like $((3+4))
// when we emit ninja file, because the result of such expressions
// will be passed to other make functions.
// TODO: Maybe we should introduce a helper binary which evaluate
// make expressions at ninja-time.
fn has_no_io_in_shell_script(cmd: &[u8]) -> bool {
    if cmd.is_empty() {
        return true;
    }
    if cmd.starts_with(b"echo $((") && cmd.ends_with(b")") {
        return true;
    }
    false
}

fn shell_func_impl(
    session: &Session,
    shell: crate::fileutil::ShellToReadWith<'_>,
    cmd: &Bytes,
    environment: &[(Bytes, Option<Bytes>)],
    loc: &Loc,
    trailing: Trailing,
) -> Result<(i32, Bytes, Option<FindCommand>)> {
    log!("ShellFunc: {:?}", cmd);

    // GNU Make bumps `command_count` when it reaps the child, which is before
    // anything else can be expanded, so noting it here rather than after the
    // command is the same moment as far as a makefile can tell. The find
    // emulator returns without starting a child, and still counts: GNU ran a
    // command there, and a `find` that wrote something through a `-exec` is
    // not what the emulator claims to answer.
    session.note_command_ran();

    if session.flags.use_find_emulator
        && let Some(fc) = crate::find::parse(session, cmd)?
        && let Some(out) = crate::find::find(session, cmd, &fc, loc)?
    {
        return Ok((0, out, Some(fc)));
    }

    collect_stats_with_slow_report!(session, "func shell time", OsStr::from_bytes(cmd));
    let (status, output) = run_command(
        shell,
        cmd,
        environment,
        RedirectStderr::None,
        &crate::diagnostic_prefix(session),
        crate::session::Context::diagnostics(session),
        session.interrupts.as_deref(),
    )?;
    let output = Bytes::from(match trailing {
        Trailing::Drop => format_for_command_substitution(output),
        Trailing::Fold => format_for_shell_assignment(output),
    });

    // A command killed by a signal exited with nothing, and GNU Make's
    // `shell_completed` reports it the way a shell reports one of its own
    // children: 128 plus the signal number.
    let exit_code = status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or_default());
    Ok((exit_code, output, None))
}

fn should_store_command_result(session: &Session, cmd: &[u8]) -> bool {
    // We really just want to ignore this one, or remove BUILD_DATETIME from
    // Android completely
    if cmd == b"date +%s" {
        return false;
    }

    if let Some(pat) = &session.flags.ignore_dirty_pattern {
        let nopat = &session.flags.no_ignore_dirty_pattern;
        for tok in word_scanner(cmd) {
            if pat.matches(tok) && !nopat.as_ref().map(|p| p.matches(tok)).unwrap_or(false) {
                return false;
            }
        }
    }

    true
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CommandOp {
    Shell,
    Find,
    Read,
    ReadMissing,
    Write,
    Append,
}

impl CommandOp {
    pub fn as_int(&self) -> i32 {
        match self {
            CommandOp::Shell => 0,
            CommandOp::Find => 1,
            CommandOp::Read => 2,
            CommandOp::ReadMissing => 3,
            CommandOp::Write => 4,
            CommandOp::Append => 5,
        }
    }

    pub fn from_int(i: i32) -> Option<CommandOp> {
        match i {
            0 => Some(CommandOp::Shell),
            1 => Some(CommandOp::Find),
            2 => Some(CommandOp::Read),
            3 => Some(CommandOp::ReadMissing),
            4 => Some(CommandOp::Write),
            5 => Some(CommandOp::Append),
            _ => None,
        }
    }

    /// The op as the regeneration stamp records it, carrying [`Self`] and the
    /// one-script bit in one int.
    ///
    /// The stamp is a fixed sequence of length-prefixed fields with no version
    /// word to branch on and no room for a new one — the format is this tool's
    /// own now that the C++ writer is retired, but its field ORDER is still
    /// what both readers ([`crate::regen`] and [`crate::regen_dump`]) walk in
    /// lockstep. So the one thing a replay needs and cannot otherwise have —
    /// whether the recorded `$(shell)` was read by the one-shell branch —
    /// rides in a high bit of the op int rather than as a field of its own. A
    /// stamp written before this bit existed holds an op in `0..=5` with the
    /// bit clear, which reads back as `one_script = false`: exactly the launch
    /// such a stamp was replayed with before, so an old stamp is read safely.
    pub fn as_stamp_int(self, one_script: bool) -> i32 {
        self.as_int()
            | if one_script {
                Self::STAMP_ONE_SCRIPT
            } else {
                0
            }
    }

    /// The op and the one-script bit back out of what [`Self::as_stamp_int`]
    /// packed. `None` only where the op itself is unknown, which is a corrupt
    /// stamp either way.
    pub fn from_stamp_int(i: i32) -> Option<(CommandOp, bool)> {
        let one_script = i & Self::STAMP_ONE_SCRIPT != 0;
        Self::from_int(i & !Self::STAMP_ONE_SCRIPT).map(|op| (op, one_script))
    }

    /// The bit [`Self::as_stamp_int`] sets, above the op values it shares the
    /// int with.
    const STAMP_ONE_SCRIPT: i32 = 0x100;
}

pub struct CommandResult {
    pub op: CommandOp,
    pub shell: Bytes,
    pub shellflag: Bytes,
    pub cmd: Bytes,
    pub find: Option<FindCommand>,
    pub result: Bytes,
    pub loc: Loc,
    /// Whether the launch that produced [`Self::result`] was read by GNU Make's
    /// one-shell branch — `.ONESHELL:` was in force. A replay that does not
    /// know this reads a script as a command line and can answer differently
    /// for a `$(shell)` that did not change, regenerating when it need not.
    pub one_script: bool,
}

/// What a command's run of trailing newlines becomes.
#[derive(Clone, Copy)]
enum Trailing {
    /// `$(shell)` drops the run.
    Drop,
    /// `!=` folds it into spaces bar one, which is the only place GNU Make has
    /// the two differ.
    Fold,
}

/// `V != cmd`. Named `shell` for its diagnostics and absent from `FUNC_INFO`,
/// so no makefile can call it.
pub const SHELL_ASSIGNMENT: FuncInfo = func(b"shell", shell_assignment_func, 1);

fn shell_assignment_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    shell_func_with(args, ev, out, Trailing::Fold)
}

fn shell_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    shell_func_with(args, ev, out, Trailing::Drop)
}

fn shell_func_with(
    args: Children,
    ev: &mut Evaluator,
    out: &mut dyn BufMut,
    trailing: Trailing,
) -> Result<()> {
    let cmd = ev.eval_arg(args, 0)?;
    if ev.defers_shell_to_the_recipe() && !has_no_io_in_shell_script(&cmd) {
        if ev.eval_depth > 1 {
            let program = ev.session.flags.program_name.clone();
            error_loc!(
                ev,
                ev.loc.as_ref(),
                "{program} doesn't support passing results of $(shell) to other make constructs: {}",
                String::from_utf8_lossy(&cmd)
            );
        }
        let cmd = strip_shell_comment(cmd);
        out.put_slice(b"$(");
        out.put_slice(&cmd);
        out.put_u8(b')');
        return Ok(());
    }

    // The command this read would run was run by the read before it, over this
    // same text, and its output is what that expansion was handed. Running it
    // again would perform the effect twice — GNU Make performs it once, on the
    // ground the build started with.
    let slot = match ev
        .session
        .ground_journal
        .answered(GroundQuestion::Shell, &cmd)
    {
        Asked::Answered(answered) => {
            out.put_slice(&answered.answer);
            ev.session.record_shell_status(answered.status)?;
            return Ok(());
        }
        Asked::Ask(slot) => slot,
    };

    let loc = ev.loc.unwrap_or_default();
    let shell = ev.get_shell()?;
    // GNU Make passes no command flags here (`func_shell` hands
    // `construct_command_argv` a zero), so `.POSIX:` keeps its `-e`.
    let shellflag = ev.get_shell_flag(false)?;
    // `one_shell` is a GLOBAL in GNU Make, not a parameter, and
    // `func_shell_base` reaches `construct_command_argv` like every other
    // launch — so under `.ONESHELL:` a `$(shell)` is read by the one-shell
    // branch too, newline separator and all, and the question is only whether
    // the line has been read YET. The bytes are what that branch's own
    // recursion defaults its flags to while it re-reads `.SHELLFLAGS` for this
    // launch; the zero line flags above are why `false` is the whole of what it
    // asks.
    let one_script = ev
        .session
        .flags
        .one_shell
        .then(|| ev.default_shell_flag(false));
    let current_scope = ev.current_scope.clone();

    // GNU Make's `func_shell` builds the child's environment with
    // `target_environment (NULL, 0)`, whose NULL means the variable sets that
    // are current here rather than none at all — so a recipe-time `$(shell)`
    // sees that target's exports and a read-time one sees the globals.
    let environment = crate::export::exported_environment(
        ev,
        current_scope.as_deref(),
        crate::export::ChildKind::Expansion,
    )?;
    let (exit_code, output, fc) = shell_func_impl(
        &ev.session,
        crate::fileutil::ShellToReadWith {
            program: &shell,
            flag: &shellflag,
            stand_in: ev.session.flags.default_shell_program.as_deref(),
            one_script,
        },
        &cmd,
        &environment,
        &loc,
        trailing,
    )?;
    out.put_slice(&output);
    if should_store_command_result(&ev.session, &cmd) {
        ev.session.command_results.push(CommandResult {
            op: if fc.is_some() {
                CommandOp::Find
            } else {
                CommandOp::Shell
            },
            shell,
            shellflag,
            cmd: cmd.clone(),
            find: fc,
            result: output.clone(),
            loc,
            // The launch that just ran read its flags the way this run's
            // `.ONESHELL:` state says, and a replay has no other way to learn
            // it — see [`CommandResult::one_script`].
            one_script: one_script.is_some(),
        })
    }
    ev.session
        .ground_journal
        .record(slot, GroundQuestion::Shell, cmd, output, Some(exit_code));
    ev.session.record_shell_status(Some(exit_code))?;
    Ok(())
}

/// `$(call name,...)` where the name is a built-in function's.
///
/// GNU Make's `func_call` looks the name up in the function table before it
/// looks for a variable, so a built-in always wins — a Makefile that defines
/// `notdir` cannot reach its own through `$(call)`. Calling one with no
/// arguments at all is not an error but an empty answer; too few for the
/// function is the same diagnostic a direct call would give.
///
/// The arguments arrive expanded, because `$(call)` expands its own. A
/// function GNU would have handed unexpanded text therefore expands it a second
/// time, which is why `$(call foreach,v,1 2,$$(v))` iterates while a single
/// `$` there would not have survived to be bound.
fn call_builtin(
    fi: &'static FuncInfo,
    args: Children,
    ev: &mut Evaluator,
    out: &mut dyn BufMut,
) -> Result<()> {
    if args.is_empty() {
        return Ok(());
    }
    if (args.len() as i16) < fi.min_arity {
        let at = ev.expanding_var_loc();
        error_loc!(
            ev,
            at.as_ref(),
            "*** insufficient number of arguments ({}) to function '{}'.",
            args.len(),
            String::from_utf8_lossy(fi.name)
        );
    }
    let mut loc = ev.loc.unwrap_or_default();
    let mut expanded = Vec::with_capacity(args.len());
    for index in 0..args.len() {
        let arg = ev.arg(args, index);
        let text = arg.eval_to_buf(ev)?;
        expanded.push(if fi.pre_expanded_args {
            ev.session.values.alloc(Value::Literal(None, text))
        } else {
            crate::expr::parse_expr(
                &mut ev.session,
                &mut loc,
                text,
                crate::expr::ParseExprOpt::Normal,
            )?
        });
    }
    let expanded = ev.session.values.alloc_children(&expanded);
    let _frame = ev.enter(FrameType::FunCall, Bytes::from_static(fi.name), loc);
    ev.function_depth += 1;
    let called = (fi.func)(expanded, ev, out);
    ev.function_depth -= 1;
    called
}

fn call_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let func_name_buf = ev.eval_arg(args, 0)?;
    let func_name_buf = func_name_buf.slice_ref(trim_space(&func_name_buf));
    if func_name_buf.is_empty() {
        return Ok(());
    }
    if let Some(fi) = get_func_info(&func_name_buf) {
        return call_builtin(fi, args.skip(1), ev, out);
    }
    let func_sym = ev.session.intern(func_name_buf.clone());
    let func = ev.lookup_var(func_sym)?;
    if func.is_none() {
        // `func_call` (function.c) asks `warn_undefined` about the name it was
        // handed, exactly as an expansion of `$(name)` would.
        ev.warn_undefined(func_sym);
        if ev.session.flags.enable_kati_warnings {
            kati_warn_loc!(
                ev,
                ev.loc.as_ref(),
                "*warning*: undefined user function: {}",
                func_sym.display(ev)
            );
        }
    }
    let mut av = Vec::with_capacity(args.len() - 1);
    for index in 1..args.len() {
        let arg = ev.arg(args, index);
        av.push(Variable::with_simple_string(
            arg.eval_to_buf(ev)?,
            VarOrigin::Automatic,
            None,
            None,
        ));
    }
    let mut bindings = Vec::new();
    let mut i = 1;
    loop {
        let tmpvar_name_sym = ev.session.intern(format!("{i}"));
        if let Some(a) = av.get(i - 1) {
            bindings.push((tmpvar_name_sym, a.clone()));
        } else {
            // We need to blank further automatic vars
            let Some(v) = ev.lookup_var(tmpvar_name_sym)? else {
                break;
            };
            if v.read().origin() != VarOrigin::Automatic {
                break;
            }

            let v = Variable::new_simple(VarOrigin::Automatic, None, None);
            bindings.push((tmpvar_name_sym, v));
        }
        i += 1;
    }

    ev.eval_depth -= 1;

    // The positional arguments stay bound for the whole body and are put back
    // afterwards, including when the body fails.
    ev.with_bounds(bindings, |ev| {
        let loc = ev.loc.unwrap_or_default();
        let _frame = ev.enter(FrameType::Call, func_name_buf, loc);
        if let Some(func) = func {
            // GNU Make's `func_call` builds the text `$(NAME)` and expands
            // THAT, so the body is reached through an ordinary reference and
            // installs the same diagnostic location an ordinary reference
            // would -- the variable's own, not the call site's. Only the
            // location: `func_call` deliberately does not mark the variable as
            // expanding, which is what lets `$(call)` recurse.
            let body = func.read();
            ev.enter_expanding_var(body.expansion_loc());
            let result = body.eval(ev, out);
            ev.leave_expanding_var();
            result?;
        }
        Ok(())
    })?;

    ev.eval_depth += 1;

    Ok(())
}

fn foreach_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let name = ev.eval_arg(args, 0)?;
    // The name is the first token of what the first argument expanded to, and
    // nothing else: GNU Make's `func_foreach` runs `next_token` over it to skip
    // leading whitespace and then writes a NUL at `end_of_token`, so
    // `$(foreach  a , ...)` binds `a` and so does `$(foreach a b, ...)`. A
    // continuation is whitespace like any other, which is how a `foreach`
    // written across several lines binds anything at all.
    let name = word_scanner(&name)
        .next()
        .map(|token| name.slice_ref(token))
        .unwrap_or_default();
    let varname = ev.session.intern(name);
    let list = ev.eval_arg(args, 1)?;
    ev.eval_depth -= 1;
    let mut ww = WordWriter::new(out);
    for tok in word_scanner(&list) {
        let tok = list.slice_ref(tok);
        let v = Variable::with_simple_string(tok, VarOrigin::Automatic, None, None);
        ww.maybe_add_space();
        ev.with_bound(varname, v, |ev| {
            let arg = ev.arg(args, 2);
            crate::expr::eval_value(arg, ev, ww.out)
        })?;
    }
    ev.eval_depth += 1;
    Ok(())
}

/// `$(let names,list,body)`: bind each name to one word of the list for the
/// duration of the body, and the last name to everything the others left.
///
/// The remainder really is the remainder, not a re-joined word list: leading
/// whitespace comes off, and what is inside and after it stays. Names past the
/// end of the list bind to the empty string rather than going unbound, and the
/// bindings are automatic ones that unwind when the body is done, so a name
/// that already meant something means it again afterwards.
fn let_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let names = ev.eval_arg(args, 0)?;
    let list = ev.eval_arg(args, 1)?;
    let mut rest = list.slice_ref(trim_left_space(&list));
    let mut bindings = Vec::new();
    let mut remaining_names = word_scanner(&names).count();
    for name in word_scanner(&names) {
        let sym = ev.session.intern(names.slice_ref(name));
        remaining_names -= 1;
        let value = if remaining_names == 0 {
            std::mem::take(&mut rest)
        } else {
            let word_len = rest.iter().position(is_space_byte).unwrap_or(rest.len());
            let word = rest.slice(..word_len);
            rest = rest.slice_ref(trim_left_space(&rest[word_len..]));
            word
        };
        let var = Variable::with_simple_string(value, VarOrigin::Automatic, None, None);
        bindings.push((sym, var));
    }
    ev.eval_depth -= 1;
    ev.with_bounds(bindings, |ev| {
        let arg = ev.arg(args, 2);
        crate::expr::eval_value(arg, ev, out)
    })?;
    ev.eval_depth += 1;
    Ok(())
}

/// One side of an `$(intcmp)` comparison, as GNU Make reads it: optional
/// whitespace, an optional sign, and digits.
///
/// The digits are kept from the first nonzero one, and `sign` is zero for a
/// zero however it was written — so `-0`, `0` and `000` are one value, and the
/// two-argument form answers with `0`.
struct MakeInt {
    sign: i32,
    digits: Bytes,
}

impl MakeInt {
    /// GNU Make's `parse_textint`. `ordinal` names which argument this is, for
    /// the diagnostic a non-numeric one dies with.
    fn parse(text: &Bytes, ordinal: &str, ev: &mut Evaluator) -> Result<MakeInt> {
        let at = ev.expanding_var_loc();
        let trimmed = trim_space(text);
        if trimmed.is_empty() {
            error_loc!(
                ev,
                at.as_ref(),
                "*** non-numeric {ordinal} argument to 'intcmp' function: empty value."
            );
        }
        let negative = trimmed[0] == b'-';
        let after_sign = usize::from(negative || trimmed[0] == b'+');
        let digits = &trimmed[after_sign..];
        let end = digits
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .unwrap_or(digits.len());
        if end == 0 || end != digits.len() {
            error_loc!(
                ev,
                at.as_ref(),
                "*** non-numeric {ordinal} argument to 'intcmp' function: '{}'.",
                String::from_utf8_lossy(text)
            );
        }
        let first_significant = digits.iter().position(|byte| *byte != b'0').unwrap_or(end);
        let digits = text.slice_ref(&digits[first_significant..]);
        let sign = i32::from(!digits.is_empty()) * if negative { -1 } else { 1 };
        Ok(MakeInt { sign, digits })
    }

    /// GNU Make compares the sign, then the number of significant digits, then
    /// the digits themselves — which orders two negatives by magnitude rather
    /// than by value, so it reads `-5` as below `-6`. Replicated because this
    /// is what `$(intcmp)` means in the Make being ported.
    fn cmp(&self, other: &MakeInt) -> std::cmp::Ordering {
        self.sign
            .cmp(&other.sign)
            .then_with(|| self.digits.len().cmp(&other.digits.len()))
            .then_with(|| self.digits.cmp(&other.digits))
    }

    fn write(&self, out: &mut dyn BufMut) {
        if self.sign == 0 {
            out.put_u8(b'0');
            return;
        }
        if self.sign < 0 {
            out.put_u8(b'-');
        }
        out.put_slice(&self.digits);
    }
}

/// `$(intcmp lhs,rhs[,lt[,eq[,gt]]])`: compare two integers and expand the arm
/// the comparison chose.
///
/// With no arms at all it answers with the value when the two are equal and
/// with nothing when they are not. A missing greater-than arm falls back to the
/// equal one rather than to nothing, and a missing equal arm means an equal or
/// greater comparison expands to nothing.
fn intcmp_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let lhs = ev.eval_arg(args, 0)?;
    let rhs = ev.eval_arg(args, 1)?;
    let lhs = MakeInt::parse(&lhs, "first", ev)?;
    let rhs = MakeInt::parse(&rhs, "second", ev)?;
    let ordering = lhs.cmp(&rhs);
    if args.len() == 2 {
        if ordering.is_eq() {
            lhs.write(out);
        }
        return Ok(());
    }
    let chosen = match ordering {
        std::cmp::Ordering::Less => 2,
        std::cmp::Ordering::Equal => 3,
        std::cmp::Ordering::Greater => {
            if args.len() > 4 {
                4
            } else {
                3
            }
        }
    };
    if let Some(arm) = ev.arg_opt(args, chosen) {
        arm.eval(ev, out)?;
    }
    Ok(())
}

fn origin_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let var_name = ev.eval_arg(args, 0)?;
    let sym = ev.session.intern(var_name);
    if let Some(var) = ev.lookup_var(sym)? {
        let orig = var.read().origin();
        out.put_slice(crate::var::get_origin_str(orig).as_bytes());
    } else {
        out.put_slice(b"undefined");
    }
    Ok(())
}

fn flavor_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    let var_name = ev.eval_arg(args, 0)?;
    let sym = ev.session.intern(var_name);
    if let Some(var) = ev.lookup_var(sym)? {
        out.put_slice(var.read().flavor().as_bytes());
    } else {
        out.put_slice(b"undefined");
    }
    Ok(())
}

/// How output deferred to a command spells itself.
///
/// `$(info)` and its siblings print immediately when evaluation can do IO. When
/// it cannot — the manifest writer and the graph sink both evaluate with IO
/// withheld — the text becomes a command instead, and that command has to
/// survive being written on one line, dequoted by the shell, and then unescaped
/// back to what the Makefile said. [`escape_printf_b`] encodes exactly that
/// round trip without leaving Makefile text active as shell syntax.
///
/// Upstream spells it `echo -e`, in both the C++ original and this port. That
/// is a bashism: `/bin/sh` is dash on a Debian system, its `echo` has no `-e`,
/// and the flag is printed as part of the output. `printf` with `%b` interprets
/// the same escapes, is specified by POSIX, and needs no flag.
const DEFERRED_OUTPUT: &[u8] = b"printf '%b\\n' \"";

fn deferred_output(message: &[u8], suffix: &[u8]) -> Bytes {
    let mut command = BytesMut::new();
    command.put_slice(DEFERRED_OUTPUT);
    command.put_slice(&escape_printf_b(message));
    command.put_u8(b'"');
    command.put_slice(suffix);
    command.freeze()
}

fn info_func(args: Children, ev: &mut Evaluator, _out: &mut dyn BufMut) -> Result<()> {
    let a = ev.eval_arg(args, 0)?;
    if ev.repeats_a_finished_read() {
        return Ok(());
    }
    if ev.defers_output_to_the_recipe() {
        ev.delayed_output_commands.push(deferred_output(&a, b""));
    } else {
        println!("{}", String::from_utf8_lossy(&a));
    }
    Ok(())
}

fn warning_func(args: Children, ev: &mut Evaluator, _out: &mut dyn BufMut) -> Result<()> {
    let a = ev.eval_arg(args, 0)?;
    if ev.repeats_a_finished_read() {
        return Ok(());
    }
    if ev.defers_output_to_the_recipe() {
        let mut message = BytesMut::new();
        let loc = ev.loc.unwrap_or_default();
        message.put_slice(loc.display(&ev.session).to_string().as_bytes());
        message.put_slice(b": ");
        message.put_slice(&a);
        ev.delayed_output_commands
            .push(deferred_output(&message, b" 2>&1"));
        return Ok(());
    }
    warn_loc!(ev, ev.loc.as_ref(), "{}", String::from_utf8_lossy(&a));
    Ok(())
}

fn error_func(args: Children, ev: &mut Evaluator, _out: &mut dyn BufMut) -> Result<()> {
    let a = ev.eval_arg(args, 0)?;
    if ev.defers_output_to_the_recipe() {
        let mut message = BytesMut::new();
        let loc = ev.loc.unwrap_or_default();
        message.put_slice(loc.display(&ev.session).to_string().as_bytes());
        message.put_slice(b": *** ");
        message.put_slice(&a);
        message.put_u8(b'.');
        ev.delayed_output_commands
            .push(deferred_output(&message, b" 2>&1 && false"));
        return Ok(());
    }
    error_loc!(ev, ev.loc.as_ref(), "*** {}.", String::from_utf8_lossy(&a));
}

fn file_read_func(
    ev: &mut Evaluator,
    filename: &OsStr,
    out: &mut dyn BufMut,
    rerun: bool,
) -> Result<()> {
    // A file that is not there reads as nothing, which is `$(file <)`'s own
    // rule and not a failure. Anything else the system refuses is one, and it
    // is reported at the line that asked for the file.
    //
    // Opening and reading are kept apart because GNU Make reports them apart,
    // and a directory is where that shows: opening one succeeds and reading it
    // does not, so `$(file < adir)` fails as a `read:` rather than an `open:`.
    let asked = Bytes::from(filename.as_bytes().to_vec());
    let slot = match ev
        .session
        .ground_journal
        .answered(GroundQuestion::FileRead, &asked)
    {
        Asked::Answered(answered) => {
            out.put_slice(&answered.answer);
            return Ok(());
        }
        Asked::Ask(slot) => slot,
    };
    let mut file = match File::open(filename) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            if should_store_command_result(&ev.session, filename.as_bytes()) {
                let loc = ev.loc.unwrap_or_default();
                ev.session.command_results.push(CommandResult {
                    op: CommandOp::ReadMissing,
                    shell: Bytes::new(),
                    shellflag: Bytes::new(),
                    cmd: asked.clone(),
                    find: None,
                    result: Bytes::new(),
                    loc,
                    // A `$(file)` is not a shell launch; the one-shell branch
                    // never touches it.
                    one_script: false,
                })
            }
            // A file that was not there is an answer too, and the read after
            // this one must be told the same thing rather than find the file
            // the staged work has since written.
            ev.session.ground_journal.record(
                slot,
                GroundQuestion::FileRead,
                asked,
                Bytes::new(),
                None,
            );
            return Ok(());
        }
        Err(err) => error_loc!(
            ev,
            ev.loc.as_ref(),
            "*** open: {}: {}.",
            filename.to_string_lossy(),
            crate::strerror(&err)
        ),
    };
    let mut buf = Vec::new();
    if let Err(err) = file.read_to_end(&mut buf) {
        error_loc!(
            ev,
            ev.loc.as_ref(),
            "*** read: {}: {}.",
            filename.to_string_lossy(),
            crate::strerror(&err)
        );
    }
    // One trailing newline goes, and no more: what is removed is the line
    // terminator the file's last line carries, not the blank lines before it.
    // A carriage return goes with it, so a file written with CRLF endings
    // reads back as its last line rather than as that line plus a stray `\r`.
    // GNU Make's `func_file`:
    // `if (n && o[-1] == '\n') o -= 1 + (n > 1 && o[-2] == '\r');`
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
    }
    let buf = Bytes::from(buf);

    if rerun && should_store_command_result(&ev.session, filename.as_bytes()) {
        let loc = ev.loc.unwrap_or_default();
        ev.session.command_results.push(CommandResult {
            op: CommandOp::Read,
            shell: Bytes::new(),
            shellflag: Bytes::new(),
            cmd: asked.clone(),
            find: None,
            result: buf.clone(),
            loc,
            one_script: false,
        })
    }
    out.put_slice(&buf);
    ev.session
        .ground_journal
        .record(slot, GroundQuestion::FileRead, asked, buf, None);
    Ok(())
}

fn file_write_func(
    ev: &mut Evaluator,
    filename: &OsStr,
    append: bool,
    text: Bytes,
    rerun: bool,
) -> Result<()> {
    // The write this read would perform was performed by the read before it,
    // over this same text. `$(file >> log,pass)` appends one line to `log`
    // however many passes the compilation takes, which is the one line GNU
    // Make's single read appends.
    if ev.repeats_a_finished_read() {
        return Ok(());
    }
    {
        let opened = File::options()
            .write(true)
            .append(append)
            .truncate(!append)
            .create(true)
            .open(filename);
        let mut f = match opened {
            Ok(f) => f,
            Err(err) => error_loc!(
                ev,
                ev.loc.as_ref(),
                "*** open: {}: {}.",
                filename.to_string_lossy(),
                crate::strerror(&err)
            ),
        };
        if let Err(err) = f.write_all(&text) {
            error_loc!(
                ev,
                ev.loc.as_ref(),
                "*** write: {}: {}.",
                filename.to_string_lossy(),
                crate::strerror(&err)
            );
        }
    }

    // A directory just gained a file, or one of its files changed. GNU Make
    // counts the write as a command for this reason and nothing else — see
    // `func_file`, which bumps `command_count` with a comment saying so.
    ev.session.note_command_ran();

    if rerun && should_store_command_result(&ev.session, filename.as_bytes()) {
        let loc = ev.loc.unwrap_or_default();
        ev.session.command_results.push(CommandResult {
            op: CommandOp::Write,
            shell: Bytes::new(),
            shellflag: Bytes::new(),
            cmd: Bytes::from(filename.as_bytes().to_vec()),
            find: None,
            result: text,
            loc,
            one_script: false,
        })
    }

    Ok(())
}

fn file_func_impl(
    args: Children,
    ev: &mut Evaluator,
    out: &mut dyn BufMut,
    rerun: bool,
) -> Result<()> {
    // GNU Make performs this wherever it is written, a recipe included. Only a
    // destination that cannot — one compiling the recipe into a manifest some
    // later run will execute — refuses, and the refusal is that destination's
    // rather than this function's.
    if ev.refuses_file_operations() {
        error_loc!(
            ev,
            ev.loc.as_ref(),
            "*** $(file ...) is not supported in rules."
        );
    }

    let arg = ev.eval_arg(args, 0)?;
    let filename = trim_space(&arg);

    if filename.is_empty() {
        let at = ev.expanding_var_loc();
        error_loc!(ev, at.as_ref(), "*** Missing filename");
    }

    if filename[0] == b'<' {
        let filename = trim_left_space(&filename[1..]);
        if filename.is_empty() {
            let at = ev.expanding_var_loc();
            error_loc!(ev, at.as_ref(), "*** Missing filename");
        }
        if args.len() > 1 {
            let at = ev.expanding_var_loc();
            error_loc!(ev, at.as_ref(), "*** invalid argument");
        }

        let filename = <OsStr as OsStrExt>::from_bytes(filename);
        file_read_func(ev, filename, out, rerun)?;
    } else if filename[0] == b'>' {
        let append = filename.starts_with(b">>");
        let filename = trim_left_space(&filename[if append { 2 } else { 1 }..]);
        if filename.is_empty() {
            let at = ev.expanding_var_loc();
            error_loc!(ev, at.as_ref(), "*** Missing filename");
        }

        let mut text = BytesMut::new();
        if let Some(contents) = ev.arg_opt(args, 1) {
            contents.eval(ev, &mut text)?;
            if text.is_empty() || !text.ends_with(b"\n") {
                text.put_u8(b'\n');
            }
        }

        let filename = <OsStr as OsStrExt>::from_bytes(filename);
        file_write_func(ev, filename, append, text.freeze(), rerun)?;
    } else {
        let at = ev.expanding_var_loc();
        error_loc!(
            ev,
            at.as_ref(),
            "*** Invalid file operation: {}.  Stop.",
            String::from_utf8_lossy(filename)
        );
    }
    Ok(())
}

fn file_func(args: Children, ev: &mut Evaluator, out: &mut dyn BufMut) -> Result<()> {
    file_func_impl(args, ev, out, true)
}

const fn func(name: &'static [u8], f: MakeFuncImpl, arity: i16) -> FuncInfo {
    FuncInfo {
        name,
        func: f,
        arity,
        min_arity: arity,
        trim_space: false,
        trim_right_space_1st: false,
        pre_expanded_args: true,
    }
}

/// A function whose arguments GNU Make leaves unexpanded for it, because what
/// the function does with them is not "read their value once".
const fn lazy_func(name: &'static [u8], f: MakeFuncImpl, arity: i16) -> FuncInfo {
    FuncInfo {
        pre_expanded_args: false,
        ..func(name, f, arity)
    }
}
const FUNC_INFO: &[FuncInfo] = &[
    func(b"patsubst", patsubst_func, 3),
    func(b"strip", strip_func, 1),
    func(b"subst", subst_func, 3),
    func(b"findstring", findstring_func, 2),
    func(b"filter", filter_func, 2),
    func(b"filter-out", filter_out_func, 2),
    func(b"sort", sort_func, 1),
    func(b"word", word_func, 2),
    func(b"wordlist", wordlist_func, 3),
    func(b"words", words_func, 1),
    func(b"firstword", firstword_func, 1),
    func(b"lastword", lastword_func, 1),
    func(b"join", join_func, 2),
    func(b"wildcard", wildcard_func, 1),
    func(b"dir", dir_func, 1),
    func(b"notdir", notdir_func, 1),
    func(b"suffix", suffix_func, 1),
    func(b"basename", basename_func, 1),
    func(b"addsuffix", addsuffix_func, 2),
    func(b"addprefix", addprefix_func, 2),
    func(b"realpath", realpath_func, 1),
    func(b"abspath", abspath_func, 1),
    FuncInfo {
        name: b"if",
        func: if_func,
        arity: 3,
        min_arity: 2,
        trim_space: false,
        pre_expanded_args: false,
        trim_right_space_1st: true,
    },
    FuncInfo {
        name: b"and",
        func: and_func,
        arity: 0,
        min_arity: 0,
        trim_space: true,
        pre_expanded_args: false,
        trim_right_space_1st: false,
    },
    FuncInfo {
        name: b"or",
        func: or_func,
        arity: 0,
        min_arity: 0,
        trim_space: true,
        pre_expanded_args: false,
        trim_right_space_1st: false,
    },
    func(b"value", value_func, 1),
    func(b"eval", eval_func, 1),
    func(b"shell", shell_func, 1),
    func(b"call", call_func, 0),
    lazy_func(b"foreach", foreach_func, 3),
    lazy_func(b"let", let_func, 3),
    FuncInfo {
        name: b"intcmp",
        func: intcmp_func,
        arity: 5,
        min_arity: 2,
        trim_space: false,
        pre_expanded_args: false,
        trim_right_space_1st: false,
    },
    func(b"origin", origin_func, 1),
    func(b"flavor", flavor_func, 1),
    func(b"info", info_func, 1),
    func(b"warning", warning_func, 1),
    func(b"error", error_func, 1),
    FuncInfo {
        name: b"file",
        func: file_func,
        arity: 2,
        min_arity: 1,
        trim_space: false,
        pre_expanded_args: true,
        trim_right_space_1st: false,
    },
    // The twelve `KATI_*` functions kati added to this table were removed on
    // 2026-08-24 by the operator's ruling: Ronin implements GNU Make, and a
    // function GNU Make has no name for is surface nobody asked for. This
    // table is GNU Make 4.4.1's, and the one entry it is missing —
    // `$(guile ...)` — is a gap rather than an extension. See
    // docs/make-kati-extensions.md.
];

// no-globals-gate: read-only dispatch table built once from the const array
// above, permitted by plan/decisions/session-owned-evaluation.md.
static FUNC_INFO_MAP: LazyLock<FastMap<&'static [u8], &'static FuncInfo>> =
    LazyLock::new(|| FUNC_INFO.iter().map(|f| (f.name, f)).collect());

pub fn get_func_info(name: &[u8]) -> Option<&'static FuncInfo> {
    FUNC_INFO_MAP.get(name).map(|v| &**v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_sink::{FileEvaluation, ShellEvaluation};
    use crate::expr::{ParseExprOpt, parse_expr};
    use crate::symtab::Symbol;

    /// The stamp op int carries the one-script bit and gives it back, for every
    /// op and both bit values.
    #[test]
    fn the_stamp_op_int_round_trips_the_one_script_bit() {
        for op in [
            CommandOp::Shell,
            CommandOp::Find,
            CommandOp::Read,
            CommandOp::ReadMissing,
            CommandOp::Write,
            CommandOp::Append,
        ] {
            for one_script in [false, true] {
                let packed = op.as_stamp_int(one_script);
                assert_eq!(CommandOp::from_stamp_int(packed), Some((op, one_script)));
            }
        }
    }

    /// A stamp written before the one-script bit existed holds a bare op in
    /// `0..=5`, and it must read back as that op with `one_script = false` —
    /// the launch it was replayed with before — so an old stamp is read safely.
    #[test]
    fn a_bare_op_int_reads_as_not_one_script() {
        for (bare, op) in [
            (0, CommandOp::Shell),
            (1, CommandOp::Find),
            (2, CommandOp::Read),
            (3, CommandOp::ReadMissing),
            (4, CommandOp::Write),
            (5, CommandOp::Append),
        ] {
            assert_eq!(CommandOp::from_stamp_int(bare), Some((op, false)));
            // And a bare op is exactly what a not-one-script launch still writes.
            assert_eq!(op.as_stamp_int(false), bare);
        }
    }

    /// Evaluate a Make expression with a fresh evaluator, returning both the
    /// result and whatever the expression managed to write before failing.
    fn eval_with(ev: &mut Evaluator, src: &'static str) -> (Result<()>, Bytes) {
        eval_source(ev, Bytes::from_static(src.as_bytes()))
    }

    /// The same, for an expression assembled at run time — a path under this
    /// test's own directory, which cannot be a literal.
    fn eval_source(ev: &mut Evaluator, src: Bytes) -> (Result<()>, Bytes) {
        let expr = parse_expr(
            &mut ev.session,
            &mut Loc::default(),
            src,
            ParseExprOpt::Normal,
        )
        .unwrap();
        let mut out = BytesMut::new();
        let result = expr.eval(ev, &mut out);
        (result, out.freeze())
    }

    /// A directory of this test's own, so one test's files are never another's.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("kati-file-func-{}-{test}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("a scratch directory");
            Self(path)
        }

        /// Write `contents` to `name` and answer the path to it.
        fn holding(&self, name: &str, contents: &[u8]) -> String {
            let path = self.0.join(name);
            std::fs::write(&path, contents).expect("a file to read back");
            path.to_str().expect("a UTF-8 scratch path").to_owned()
        }

        fn path(&self, name: &str) -> String {
            self.0
                .join(name)
                .to_str()
                .expect("a UTF-8 scratch path")
                .to_owned()
        }

        fn read(&self, name: &str) -> Vec<u8> {
            std::fs::read(self.0.join(name)).expect("a written file")
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An evaluator expanding a recipe for a destination that runs the build
    /// itself: the position GNU Make is in when it expands a recipe.
    fn expanding_a_recipe_that_runs_here() -> Evaluator {
        let mut ev = Evaluator::new(Session::new());
        ev.avoid_io = true;
        ev.shell_evaluation = ShellEvaluation::Expansion;
        ev.file_evaluation = FileEvaluation::Expansion;
        ev
    }

    fn simple(value: &'static [u8]) -> crate::var::Var {
        Variable::with_simple_string(Bytes::from_static(value), VarOrigin::File, None, None)
    }

    fn string_of(session: &Session, var: crate::var::Var) -> String {
        String::from_utf8(var.read().string(session).unwrap().into_owned()).unwrap()
    }

    /// A destination that runs the build itself gets GNU Make's answer: the
    /// value arrives during the recipe's own expansion, so it composes with the
    /// functions around it instead of being written out as shell syntax that
    /// only an unquoted position would ever substitute.
    #[test]
    fn a_recipe_shell_composes_when_expansion_answers() {
        let mut ev = Evaluator::new(Session::new());
        ev.avoid_io = true;
        ev.shell_evaluation = ShellEvaluation::Expansion;
        ev.session
            .set_global_var(Symbol::SHELL, simple(b"/bin/sh"), false)
            .unwrap();
        ev.session
            .set_global_var(Symbol::SHELLFLAGS, simple(b"-c"), false)
            .unwrap();

        let (result, out) = eval_with(&mut ev, "$(subst b,B,$(shell echo abc))");
        result.unwrap();
        assert_eq!(out, "aBc");
        assert_eq!(ev.session.shell_status, Some(0));
    }

    /// The other destination is a manifest, where the recipe's own shell is
    /// what answers. A composed one has nowhere to put the answer and says so
    /// rather than writing a command substitution into the middle of a value.
    #[test]
    fn a_recipe_shell_defers_when_a_manifest_runs() {
        let mut ev = Evaluator::new(Session::new());
        ev.avoid_io = true;

        let (result, out) = eval_with(&mut ev, "$(shell echo abc)");
        result.unwrap();
        assert_eq!(out, "$(echo abc)");
        assert!(ev.session.shell_status.is_none());

        let (composed, _) = eval_with(&mut ev, "$(subst b,B,$(shell echo abc))");
        assert!(composed.is_err());
    }

    /// Deferred output is Makefile data, not another opportunity to evaluate
    /// shell syntax. Execute the generated command with `/bin/sh` so the test
    /// covers both halves of the encoding: the shell's double quotes and
    /// `printf %b`'s backslash processing.
    #[test]
    fn deferred_info_prints_shell_metacharacters_literally() {
        let mut ev = Evaluator::new(Session::new());
        ev.avoid_io = true;
        let (result, out) = eval_with(
            &mut ev,
            r#"$(info dollars=$$HOME tick=`printf SUBSTITUTED` slashes=one\ttwo quote=")"#,
        );

        result.unwrap();
        assert!(out.is_empty());
        let command = ev.delayed_output_commands.pop().unwrap();
        assert!(ev.delayed_output_commands.is_empty());
        let output = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(std::str::from_utf8(&command).unwrap())
            .env("HOME", "EXPANDED")
            .output()
            .unwrap();

        assert!(output.status.success());
        assert_eq!(
            output.stdout,
            b"dollars=$HOME tick=`printf SUBSTITUTED` slashes=one\\ttwo quote=\"\n"
        );
        assert!(output.stderr.is_empty());
    }

    /// A `foreach` body that fails partway must leave the loop variable
    /// unbound again — not bound to the last token, and not bound to an empty
    /// value. Restoring an absence is a distinct case from restoring a value,
    /// and it is the one the error path used to reach only through `Drop`.
    #[test]
    fn test_foreach_restores_unbound_variable_when_body_fails() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_FOREACH_UNBOUND");
        assert!(ev.session.peek_global_var(sym).is_none());

        let (result, out) = eval_with(
            &mut ev,
            "$(foreach KATI_TEST_FOREACH_UNBOUND,a b c,\
             $(if $(filter b,$(KATI_TEST_FOREACH_UNBOUND)),\
             $(error stop),$(KATI_TEST_FOREACH_UNBOUND)))",
        );

        assert!(result.is_err());
        // The first token was written and the third was not: the loop failed
        // partway rather than before it started.
        assert_eq!(out.as_ref(), b"a ");
        assert!(ev.session.peek_global_var(sym).is_none());
    }

    /// The same failure with a binding to go back to must go back to it.
    #[test]
    fn test_foreach_restores_previous_binding_when_body_fails() {
        let mut ev = Evaluator::new(Session::new());
        let sym = ev.session.intern("KATI_TEST_FOREACH_BOUND");
        ev.session
            .set_global_var(sym, simple(b"outer"), false)
            .unwrap();

        let (result, out) = eval_with(
            &mut ev,
            "$(foreach KATI_TEST_FOREACH_BOUND,a b c,\
             $(if $(filter b,$(KATI_TEST_FOREACH_BOUND)),\
             $(error stop),$(KATI_TEST_FOREACH_BOUND)))",
        );

        assert!(result.is_err());
        assert_eq!(out.as_ref(), b"a ");
        let var = ev.session.peek_global_var(sym).unwrap();
        assert_eq!(string_of(&ev.session, var), "outer");
    }

    /// `call` binds every positional argument at once. A body that fails must
    /// leave all of them as it found them.
    #[test]
    fn test_call_restores_positional_arguments_when_body_fails() {
        let mut ev = Evaluator::new(Session::new());
        let body = b"$(1)$(error stop)";
        let expr = parse_expr(
            &mut ev.session,
            &mut Loc::default(),
            Bytes::from_static(body),
            ParseExprOpt::Normal,
        )
        .unwrap();
        let func = ev.session.intern("KATI_TEST_CALL_FUNC");
        ev.session
            .set_global_var(
                func,
                Variable::new_recursive(
                    expr,
                    VarOrigin::File,
                    None,
                    None,
                    Bytes::from_static(body),
                ),
                false,
            )
            .unwrap();

        let one = ev.session.intern("1");
        let two = ev.session.intern("2");
        assert!(ev.session.peek_global_var(one).is_none());
        assert!(ev.session.peek_global_var(two).is_none());

        let (result, out) = eval_with(&mut ev, "$(call KATI_TEST_CALL_FUNC,x,y)");

        assert!(result.is_err());
        // $1 was bound while the body ran, and the body failed after using it.
        assert_eq!(out.as_ref(), b"x");
        assert!(ev.session.peek_global_var(one).is_none());
        assert!(ev.session.peek_global_var(two).is_none());
    }

    /// `let` binds every name at once, the way `call` binds its positional
    /// arguments, so a body that fails must leave all of them as it found them.
    #[test]
    fn let_restores_bindings_when_body_fails() {
        let mut ev = Evaluator::new(Session::new());
        let bound = ev.session.intern("KATI_TEST_LET_BOUND");
        let unbound = ev.session.intern("KATI_TEST_LET_UNBOUND");
        ev.session
            .set_global_var(bound, simple(b"outer"), false)
            .unwrap();

        let (result, out) = eval_with(
            &mut ev,
            "$(let KATI_TEST_LET_BOUND KATI_TEST_LET_UNBOUND,a b,\
             $(KATI_TEST_LET_BOUND)$(error stop))",
        );

        assert!(result.is_err());
        assert_eq!(out.as_ref(), b"a");
        assert!(ev.session.peek_global_var(unbound).is_none());
        let var = ev.session.peek_global_var(bound).unwrap();
        assert_eq!(string_of(&ev.session, var), "outer");
    }

    /// An argument that is not an integer stops the build, and an empty one
    /// says so in its own words rather than quoting nothing.
    #[test]
    fn intcmp_refuses_a_non_numeric_argument() {
        let mut ev = Evaluator::new(Session::new());
        let (result, _) = eval_with(&mut ev, "$(intcmp 12a,1,foo)");
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("non-numeric first argument to 'intcmp' function: '12a'."),
            "{message}"
        );

        let (result, _) = eval_with(&mut ev, "$(intcmp 0, ,foo)");
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("non-numeric second argument to 'intcmp' function: empty value."),
            "{message}"
        );
    }

    /// An index that is all digits but too large to be one is refused as out of
    /// range, which is a different rejection from a non-numeric one and the only
    /// one of the group that used to be let through: reading it as an index no
    /// list can have answers with the empty string and runs a build GNU Make
    /// stopped.
    #[test]
    fn word_refuses_an_index_out_of_range() {
        let mut ev = Evaluator::new(Session::new());
        let (result, _) = eval_with(&mut ev, "$(word 9999999999999999999,a b c)");
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains(
                "invalid first argument to 'word' function: '9999999999999999999' out of range."
            ),
            "{message}"
        );

        // An index that merely runs off the end of the list is not an error.
        let (result, out) = eval_with(&mut ev, "$(word 4294967296,a b c)");
        result.unwrap();
        assert!(out.is_empty());
    }

    /// The three ways a numeric argument can be unreadable are three
    /// diagnostics, and a sign or surrounding space is not one of them.
    #[test]
    fn word_tells_its_numeric_refusals_apart() {
        let mut ev = Evaluator::new(Session::new());
        for (expression, expected) in [
            (
                "$(word abc,a b c)",
                "invalid first argument to 'word' function: 'abc'.",
            ),
            (
                "$(word ,a b c)",
                "invalid first argument to 'word' function: empty value.",
            ),
            (
                "$(word 0,a b c)",
                "first argument to 'word' function must be greater than 0.",
            ),
            (
                "$(word -1,a b c)",
                "first argument to 'word' function must be greater than 0.",
            ),
        ] {
            let (result, _) = eval_with(&mut ev, expression);
            let message = result.unwrap_err().to_string();
            assert!(message.contains(expected), "{expression}: {message}");
        }

        // GNU strips whitespace either side and takes a leading sign.
        for expression in ["$(word  3 ,a b c)", "$(word +3,a b c)", "$(word 03,a b c)"] {
            let (result, out) = eval_with(&mut ev, expression);
            result.unwrap();
            assert_eq!(out.as_ref(), b"c", "{expression}");
        }
    }

    /// `$(wordlist)` refuses a start below one and a stop below zero, and names
    /// the number it read rather than the text it was written as.
    #[test]
    fn wordlist_refuses_indices_outside_its_range() {
        let mut ev = Evaluator::new(Session::new());
        for (expression, expected) in [
            (
                "$(wordlist 000,3,a b c)",
                "invalid first argument to 'wordlist' function: '0'.",
            ),
            (
                "$(wordlist 2,-1,a b c)",
                "invalid second argument to 'wordlist' function: '-1'.",
            ),
            (
                "$(wordlist 2,9999999999999999999,a b c)",
                "invalid second argument to 'wordlist' function: '9999999999999999999' out of range.",
            ),
        ] {
            let (result, _) = eval_with(&mut ev, expression);
            let message = result.unwrap_err().to_string();
            assert!(message.contains(expected), "{expression}: {message}");
        }

        // A stop of zero, and a stop below the start, are empty rather than errors.
        for expression in ["$(wordlist 2,0,a b c)", "$(wordlist 3,2,a b c)"] {
            let (result, out) = eval_with(&mut ev, expression);
            result.unwrap();
            assert!(out.is_empty(), "{expression}");
        }
    }

    /// A builtin reached through `$(call)` answers to the same argument count
    /// it would have refused directly — except for none at all, which GNU Make
    /// answers with nothing rather than a diagnostic.
    #[test]
    fn call_refuses_too_few_builtin_arguments() {
        let mut ev = Evaluator::new(Session::new());
        let (result, _) = eval_with(&mut ev, "$(call filter,%.c)");
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("insufficient number of arguments (1) to function 'filter'."),
            "{message}"
        );

        let (result, out) = eval_with(&mut ev, "$(call filter)");
        result.unwrap();
        assert!(out.is_empty());
    }

    /// A destination that runs the build itself performs a recipe's
    /// `$(file <)` where GNU Make performs it, and hands the contents back to
    /// the expansion that asked for them.
    ///
    /// This is the Linux kernel's shape, reduced: `read-file` in
    /// `scripts/Kbuild.include` is `$(subst $(newline),$(space),$(file < $1))`,
    /// and it reaches a recipe through a recursively expanded `KERNELRELEASE`.
    /// A read cannot be written into the recipe for its shell to answer, the
    /// way `$(shell)` can, because its result has to compose with the Make
    /// functions around it — which is why the refusal below has to be the
    /// destination's rather than this function's.
    #[test]
    fn a_recipe_file_read_is_performed_when_expansion_answers() {
        let scratch = Scratch::new("recipe-read");
        let path = scratch.holding("kernel.release", b"6.18.2-necessary\n");
        let mut ev = expanding_a_recipe_that_runs_here();

        let (result, out) = eval_source(&mut ev, Bytes::from(format!("[$(file < {path})]")));

        result.unwrap();
        assert_eq!(out, "[6.18.2-necessary]");
    }

    /// The other destination is a manifest, which will be executed by another
    /// program on another day. Reading then would answer from a tree that is
    /// not the one the build will run against, and writing then would put the
    /// file on disk while the manifest is being written, so the whole function
    /// is refused where a rule can reach it.
    #[test]
    fn a_recipe_file_operation_is_refused_when_a_manifest_runs() {
        let scratch = Scratch::new("manifest-refusal");
        let path = scratch.holding("kernel.release", b"6.18.2-necessary\n");
        let mut ev = Evaluator::new(Session::new());
        ev.avoid_io = true;

        let (result, _) = eval_source(&mut ev, Bytes::from(format!("$(file < {path})")));
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("$(file ...) is not supported in rules."),
            "{message}"
        );

        let written = scratch.path("written");
        let (result, _) = eval_source(&mut ev, Bytes::from(format!("$(file > {written},text)")));
        assert!(result.is_err());
        assert!(!std::path::Path::new(&written).exists());
    }

    /// Outside a recipe there is no destination to ask: `$(file ...)` is
    /// GNU Make's own, and kati has always performed it there.
    #[test]
    fn a_file_operation_outside_a_recipe_is_performed_whatever_the_destination() {
        let scratch = Scratch::new("outside-a-recipe");
        let path = scratch.holding("contents", b"answered\n");
        let mut ev = Evaluator::new(Session::new());

        let (result, out) = eval_source(&mut ev, Bytes::from(format!("$(file < {path})")));

        result.unwrap();
        assert_eq!(out, "answered");
    }

    /// GNU Make's `func_file` removes one trailing newline and no more, and
    /// takes a carriage return with it so a CRLF file reads back as its last
    /// line. A file that is not there reads as nothing and is not an error.
    #[test]
    fn a_file_read_removes_one_trailing_line_terminator() {
        let scratch = Scratch::new("trailing-newline");
        let mut ev = Evaluator::new(Session::new());
        for (contents, expected) in [
            (b"a\n".as_slice(), "a"),
            (b"a\n\n".as_slice(), "a\n"),
            (b"a\n\n\n".as_slice(), "a\n\n"),
            (b"a".as_slice(), "a"),
            (b"a\r\n".as_slice(), "a"),
            (b"a\r\n\r\n".as_slice(), "a\r\n"),
            (b"\n".as_slice(), ""),
            (b"".as_slice(), ""),
        ] {
            let path = scratch.holding("contents", contents);
            let (result, out) = eval_source(&mut ev, Bytes::from(format!("$(file < {path})")));
            result.unwrap();
            assert_eq!(out, expected, "reading {contents:?}");
        }

        let absent = scratch.path("absent");
        let (result, out) = eval_source(&mut ev, Bytes::from(format!("$(file < {absent})")));
        result.unwrap();
        assert!(out.is_empty());
    }

    /// A directory opens and does not read, so GNU Make reports it against the
    /// read rather than the open. Reporting both the same way would name the
    /// wrong operation for the one case where they differ.
    #[test]
    fn a_file_read_of_a_directory_fails_at_the_read() {
        let scratch = Scratch::new("directory");
        let mut ev = Evaluator::new(Session::new());

        let path = scratch.path("");
        let (result, _) = eval_source(&mut ev, Bytes::from(format!("$(file < {path})")));

        let message = result.unwrap_err().to_string();
        assert!(message.contains("*** read: "), "{message}");
        assert!(message.contains("Is a directory."), "{message}");
    }

    /// A recipe's `$(file >)` writes while the recipe is expanded, and the
    /// directory it changed is one a `$(wildcard)` later in the same expansion
    /// has to look at again — which is what GNU Make's `++command_count` in
    /// `func_file` is for.
    #[test]
    fn a_recipe_file_write_is_seen_by_a_later_wildcard() {
        let scratch = Scratch::new("write-then-wildcard");
        let pattern = scratch.path("*.written");
        let written = scratch.path("made.written");
        let mut ev = expanding_a_recipe_that_runs_here();

        let (result, before) = eval_source(&mut ev, Bytes::from(format!("$(wildcard {pattern})")));
        result.unwrap();
        assert!(before.is_empty());

        let (result, out) = eval_source(
            &mut ev,
            Bytes::from(format!("$(file > {written},contents)")),
        );
        result.unwrap();
        assert!(out.is_empty());
        assert_eq!(scratch.read("made.written"), b"contents\n");

        let (result, after) = eval_source(&mut ev, Bytes::from(format!("$(wildcard {pattern})")));
        result.unwrap();
        assert_eq!(after, written.as_str());
    }
}
