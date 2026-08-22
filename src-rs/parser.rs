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

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use memchr::{memchr, memchr2, memchr3};
use parking_lot::Mutex;

use crate::{
    collect_stats, error_loc,
    expr::{ParseExprOpt, parse_expr},
    loc::Loc,
    session::Session,
    stmt::{
        AssignDirective, AssignModifiers, AssignOp, AssignStmt, CommandStmt, CondComplaint, CondOp,
        ExportStmt, IfStmt, IncludeStmt, ParseErrorStmt, RuleStmt, Stmt, UndefineStmt, VpathStmt,
    },
    strutil::{
        find_end_of_line, find_outside_paren, find_outside_reference,
        strip_recipe_prefix_continuations, trim_left_space, trim_right_space, trim_space,
        word_scanner,
    },
    symtab::Symbol,
    var::VarExport,
    warn_loc,
};

/// What introduces a recipe line before `.RECIPEPREFIX` says otherwise.
const RECIPE_PREFIX_DEFAULT: u8 = b'\t';

/// Where an `ifeq`/`ifneq` condition's two compared strings sit in the text
/// after the directive, and where whatever follows the condition begins.
struct IfeqCondition {
    lhs: Range<usize>,
    rhs: Range<usize>,
    /// First byte after the condition's close that is not a blank. Equal to the
    /// line's length when the condition ends the line.
    rest: usize,
}

/// A blank in GNU Make's reader: `ISBLANK`, which is a space or a tab and
/// nothing else.
fn is_blank(c: u8) -> bool {
    c == b' ' || c == b'\t'
}

/// GNU Make's `NEXT_TOKEN`: the offset of the first byte at or after `i` that
/// is not a blank.
fn skip_blanks(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && is_blank(s[i]) {
        i += 1;
    }
    i
}

/// What splitting an `ifeq`/`ifneq` condition found.
enum SplitCondition {
    /// Both strings, and where whatever follows the close begins.
    Read(IfeqCondition),
    /// The invalid syntax GNU returns -1 for, carrying the first string when
    /// the split had already read it.
    ///
    /// Which of the two it is decides what has happened by the time the
    /// complaint is made. `conditional_line` expands the first string the
    /// moment it has found its end, and goes looking for the second string's
    /// close afterwards — so every way of failing from there on has already run
    /// whatever the first string does, and every way of failing before it has
    /// run nothing.
    Invalid { lhs: Option<Range<usize>> },
}

/// Splits an `ifeq`/`ifneq` condition the way `conditional_line` does
/// (reference/gnumake/src/read.c).
///
/// The first byte picks the form — `(` the parenthesised one, a quote the
/// quoted one — and the condition's close is then found by scanning FORWARD,
/// counting nested parens, rather than by reading the line's last byte. So a
/// line whose last byte is not the close is not a different form: it is this
/// form with trailing text after it, which the caller warns about rather than
/// refusing over.
fn split_ifeq_condition(s: &[u8]) -> SplitCondition {
    // Nothing is read yet, so nothing GNU Make would have expanded has been.
    let unread = SplitCondition::Invalid { lhs: None };
    let Some(first) = s.first() else {
        return unread;
    };
    let termin = match *first {
        b'(' => b',',
        quote @ (b'"' | b'\'') => quote,
        _ => return unread,
    };

    // The first string runs to the terminator. Inside the parenthesised form a
    // comma only ends it at paren depth zero, so a comma belonging to a nested
    // call stays part of the string.
    let lhs_start = 1;
    let mut i = lhs_start;
    if termin == b',' {
        let mut depth = 0i32;
        while i < s.len() {
            match s[i] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                b',' if depth <= 0 => break,
                _ => {}
            }
            i += 1;
        }
    } else {
        while i < s.len() && s[i] != termin {
            i += 1;
        }
    }
    if i >= s.len() {
        // The first string's own terminator never arrived, so GNU Make has not
        // reached the expansion below it.
        return unread;
    }

    let mut lhs_end = i;
    if termin == b',' {
        // Blanks between the first string and the comma belong to neither.
        while lhs_end > lhs_start && is_blank(s[lhs_end - 1]) {
            lhs_end -= 1;
        }
    }
    i += 1;
    // From here down the first string is read, and GNU Make expands it before
    // it looks for anything else — so every refusal below carries it.
    let lhs = lhs_start..lhs_end;

    // What closes the second string: the matching paren for the parenthesised
    // form, and for the quoted one the next non-blank byte, which has to be a
    // quote of its own.
    if termin != b',' {
        i = skip_blanks(s, i);
    }
    let close = if termin == b',' {
        b')'
    } else {
        match s.get(i) {
            Some(close) => *close,
            None => return SplitCondition::Invalid { lhs: Some(lhs) },
        }
    };
    if close != b')' && close != b'"' && close != b'\'' {
        return SplitCondition::Invalid { lhs: Some(lhs) };
    }

    let rhs_start;
    if close == b')' {
        // Blanks before the second string are skipped; blanks after it are not.
        rhs_start = skip_blanks(s, i);
        i = rhs_start;
        let mut depth = 0i32;
        while i < s.len() {
            match s[i] {
                b'(' => depth += 1,
                b')' => {
                    if depth <= 0 {
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            i += 1;
        }
    } else {
        i += 1;
        rhs_start = i;
        while i < s.len() && s[i] != close {
            i += 1;
        }
    }
    if i >= s.len() {
        return SplitCondition::Invalid { lhs: Some(lhs) };
    }

    SplitCondition::Read(IfeqCondition {
        lhs,
        rhs: rhs_start..i,
        rest: skip_blanks(s, i + 1),
    })
}

struct IfState {
    stmt: Arc<IfStmt>,
    is_in_else: bool,
    num_nest: i32,
}

struct Parser<'a> {
    /// Parsing interns and raises located diagnostics, so it needs the session
    /// as much as evaluation does.
    // [spec:ronin:req:make.no-ambient-state]
    session: &'a mut Session,
    buf: Bytes,
    l: usize,
    // Represents if we just parsed a rule or an expression.
    // Expressions are included because they can expand into
    // a rule, see testcase/rule_in_var.mk.
    after_rule: bool,

    stmts: Arc<Mutex<Vec<Stmt>>>,
    out_stmts: Arc<Mutex<Vec<Stmt>>>,

    define_name: Option<Bytes>,
    define_op: AssignOp,
    num_define_nest: i32,
    define_start: usize,
    define_start_line: i32,

    orig_line_with_directives: Option<Bytes>,
    current_directive: Option<AssignDirective>,

    num_if_nest: i32,
    if_stack: Vec<IfState>,

    /// What introduces a recipe line, from `.RECIPEPREFIX`.
    cmd_prefix: u8,

    loc: Loc,
    fixed_lineno: bool,
}

impl<'a> Parser<'a> {
    fn with_buf(
        session: &'a mut Session,
        buf: &Bytes,
        loc: Loc,
        stmts: Arc<Mutex<Vec<Stmt>>>,
        fixed_lineno: bool,
    ) -> Self {
        Self {
            session,
            buf: buf.clone(),
            l: 0,
            after_rule: false,

            stmts: stmts.clone(),
            out_stmts: stmts,

            define_name: None,
            define_op: AssignOp::Eq,
            num_define_nest: 0,
            define_start: 0,
            define_start_line: 0,

            orig_line_with_directives: None,
            current_directive: None,

            num_if_nest: 0,
            if_stack: Vec::new(),

            cmd_prefix: RECIPE_PREFIX_DEFAULT,

            loc,
            fixed_lineno,
        }
    }

    fn parse(&mut self) -> Result<()> {
        self.l = 0;
        let buf = self.buf.clone();

        while self.l < buf.len() {
            let eol = find_end_of_line(&buf.slice(self.l..));
            let new_l = self.l + eol.line.len();
            if !self.fixed_lineno {
                self.loc.line += 1;
            }
            let mut line = eol.line;
            if line.ends_with(b"\r") {
                line.truncate(line.len() - 1);
            }
            self.orig_line_with_directives = Some(line.clone());
            self.parse_line(line)?;
            if !self.fixed_lineno {
                self.loc.line += eol.lf_cnt - 1;
            }
            if new_l == buf.len() {
                break;
            }
            self.l = new_l + 1
        }

        if !self.if_stack.is_empty() {
            let mut loc = self.loc.clone();
            loc.line += 1;
            // Said where GNU Make says it: at the end of the read, once every
            // line above has been read and everything those lines do has
            // happened. It stands at the FILE's level rather than in whichever
            // branch was still open, because GNU checks `conditionals->if_cmds`
            // whether or not it was ignoring — and standing there is also what
            // lets a conditional whose own condition could not be read speak
            // first, which is the order a one-pass read produces.
            self.stmts
                .lock()
                .push(ParseErrorStmt::new(loc, "*** missing 'endif'.".to_string()));
        }
        if self.define_name.is_some() {
            let mut loc = self.loc.clone();
            loc.line = self.define_start_line;
            error_loc!(
                &*self.session,
                Some(&loc),
                "*** missing 'endef', unterminated 'define'.",
            );
        }

        Ok(())
    }

    fn parse_line(&mut self, line: Bytes) -> Result<()> {
        if self.define_name.is_some() {
            return self.parse_inside_define(line);
        }

        if line.is_empty() || &*line == b"\r" {
            return Ok(());
        }

        self.current_directive = None;

        if line.first() == Some(&self.cmd_prefix) && self.after_rule {
            let loc = self.loc.clone();
            let mut mutable_loc = self.loc.clone();
            // The line the prefix opened may have been continued, and every
            // line it was continued onto carries the prefix too. Only the first
            // one is this slice; the rest are still inside it.
            let body = strip_recipe_prefix_continuations(line.slice(1..), self.cmd_prefix);
            let expr = parse_expr(self.session, &mut mutable_loc, body, ParseExprOpt::Command)?;
            self.out_stmts
                .lock()
                .push(CommandStmt::new(loc, line, expr));
            return Ok(());
        }

        let line = line.slice_ref(trim_left_space(&line));

        if line.starts_with(b"#") {
            return Ok(());
        }

        if self.handle_make_directive(&line)? {
            return Ok(());
        }

        self.parse_rule_or_assign(line)
    }

    /// Decide whether this line defines a variable or describes a rule, in
    /// GNU Make's order: the assignment question first, over the whole line,
    /// and a rule only when the answer is no.
    ///
    /// A `;` takes no part in the first question. `parse_variable_definition`
    /// (reference/gnumake/src/variable.c) has no case for one — it is not in
    /// `STOP_SET (c, MAP_COMMENT|MAP_NUL)`, it is not blank, it is not `=` or
    /// `:`, and `*p` is not `=` — so it falls through `other:` and the walk
    /// carries on to whatever operator is written after it. `a;b=c` therefore
    /// defines a variable whose name holds a semicolon.
    ///
    /// It takes part in the second question, where it is the separator between
    /// a rule and the recipe written on the same line: GNU Make cuts the line
    /// at the first unquoted `;` and then looks for the colon in what is left,
    /// which is why `a;b: c` is `missing separator` rather than a rule — the
    /// colon went with the recipe.
    ///
    /// So the two questions are separated rather than the character set tuned.
    /// Asking them at once, on whichever of `:`, `=` and `;` came first, is a
    /// scan that stops on a character GNU Make had no case for.
    fn parse_rule_or_assign(&mut self, line: Bytes) -> Result<()> {
        let Some(sep) = find_outside_reference(line.as_ref(), b":=") else {
            return self.parse_rule(line, None);
        };
        let s = &line[sep..];
        if s.starts_with(b"=") {
            if sep != 0 && !is_variable_name(parse_assign_statement(&line, sep).lhs) {
                return self.parse_rule(line, None);
            }
            return self.parse_assign(line, sep);
        } else if s.starts_with(b":") {
            let colons = s.iter().take_while(|byte| **byte == b':').count();
            if (1..=3).contains(&colons) && s.get(colons) == Some(&b'=') {
                let assign_sep = sep + colons;
                if sep != 0 && !is_variable_name(parse_assign_statement(&line, assign_sep).lhs) {
                    return self.parse_rule(line, None);
                }
                return self.parse_assign(line, assign_sep);
            }
            return self.parse_rule(line, Some(sep));
        }
        unreachable!()
    }

    fn parse_rule(&mut self, line: Bytes, _sep: Option<usize>) -> Result<()> {
        let orig_line = self.orig_line_with_directives.clone().unwrap();
        let mut line = line;
        if self.current_directive.is_some() {
            if self.is_in_export() {
                return Ok(());
            }
            line = orig_line.clone();
        }

        line = line.slice_ref(trim_left_space(&line));
        if line.is_empty() {
            return Ok(());
        }

        if orig_line.first() == Some(&self.cmd_prefix) {
            error_loc!(
                &*self.session,
                Some(&self.loc),
                "*** commands commence before first target."
            );
        }

        self.after_rule = true;
        self.note_posix_target(&line);
        self.out_stmts
            .lock()
            .push(RuleStmt::new(self.loc.clone(), line, self.cmd_prefix));
        Ok(())
    }

    /// Read `.POSIX` off a rule's targets, the way GNU Make's `check_specials`
    /// (read.c) does, so that every line read after it folds its continuations
    /// the POSIX way.
    ///
    /// Only a name this parse can read for itself counts, and there are two
    /// ways it cannot.
    ///
    /// A target list holding a `$` is text rather than names: GNU Make asks the
    /// question of the EXPANDED list, so `$(P):` with `P = .POSIX` names the
    /// target and `$(info .POSIX:)` names nothing at all — and reading either as
    /// written gets one of them wrong. Reading the written text would take the
    /// second for a declaration no makefile made, which is the worse of the two
    /// mistakes: it turns a run's continuations POSIX in a makefile that never
    /// asked.
    ///
    /// A line inside a conditional is one GNU Make may never read. The branch is
    /// chosen by the evaluation, and this parse buffers both, so a `.POSIX`
    /// written in the branch that loses would otherwise be read anyway.
    ///
    /// What is missed either way falls to the evaluator, which sets the same
    /// flag when it reaches such a rule — in time for whatever is read after
    /// that, and too late for what was read before it. That gap is what
    /// make-a-posix-target-is-read-where-it-is-written owns.
    fn note_posix_target(&mut self, line: &[u8]) {
        if self.session.posix_pedantic || !self.if_stack.is_empty() {
            return;
        }
        let targets = match line.iter().position(|byte| *byte == b':') {
            Some(colon) => &line[..colon],
            None => line,
        };
        if targets.contains(&b'$') {
            return;
        }
        if word_scanner(targets).any(|word| word == b".POSIX") {
            self.session.posix_pedantic = true;
        }
    }

    fn parse_assign(&mut self, line: Bytes, separator_pos: usize) -> Result<()> {
        if separator_pos == 0 {
            error_loc!(
                &*self.session,
                Some(&self.loc),
                "*** empty variable name ***"
            );
        }
        let mut assign = parse_assign_statement(&line, separator_pos);
        self.note_recipe_prefix(&assign);

        // If rhs starts with '$=', this is 'final assignment',
        // e.g., a combination of the assignment and
        //  .KATI_READONLY := <lhs>
        // statement. Note that we assume that ParseAssignStatement
        // trimmed the left
        let is_final = assign.rhs.starts_with(b"$=");
        if is_final {
            assign.rhs = trim_left_space(&assign.rhs[2..]);
        }

        let assign_loc = self.loc.clone();
        let mut mutable_loc = self.loc.clone();
        let lhs = parse_expr(
            self.session,
            &mut mutable_loc,
            line.slice_ref(assign.lhs),
            ParseExprOpt::Normal,
        )?;
        let orig_rhs = line.slice_ref(assign.rhs);
        let rhs = parse_expr(
            self.session,
            &mut mutable_loc,
            orig_rhs.clone(),
            ParseExprOpt::Normal,
        )?;

        self.after_rule = false;
        self.out_stmts.lock().push(AssignStmt::new(
            assign_loc,
            lhs,
            rhs,
            orig_rhs,
            assign.op,
            self.current_directive,
            is_final,
        ));
        Ok(())
    }

    /// Read `.RECIPEPREFIX` where it is written.
    ///
    /// GNU Make applies it as it reads, so a rule below the assignment is
    /// introduced by the new character and one above it is not. Parsing here
    /// runs ahead of evaluation, which costs three narrowings, each of them a
    /// refusal rather than a wrong build: a conditional's branches are both
    /// parsed and only one runs, so an assignment inside one is not read; a
    /// simply expanded value holding a `$` cannot be expanded yet; and the
    /// prefix does not reach an included file, which is parsed when the
    /// `include` runs. A recursive value is taken verbatim, which is what GNU
    /// Make stores and therefore what it reads the first character of.
    fn note_recipe_prefix(&mut self, assign: &ParsedAssign) {
        if !self.if_stack.is_empty() || assign.lhs != b".RECIPEPREFIX" {
            return;
        }
        let value = trim_left_space(Parser::remove_comment(assign.rhs));
        match assign.op {
            AssignOp::Eq => {}
            AssignOp::ColonEq if memchr(b'$', value).is_none() => {}
            _ => return,
        }
        self.cmd_prefix = value.first().copied().unwrap_or(RECIPE_PREFIX_DEFAULT);
    }

    fn parse_include(&mut self, line: Bytes, directive: &[u8]) -> Result<()> {
        let loc = self.loc.clone();
        let mut mutable_loc = loc.clone();
        let expr = parse_expr(self.session, &mut mutable_loc, line, ParseExprOpt::Normal)?;
        self.out_stmts
            .lock()
            .push(IncludeStmt::new(loc, expr, directive.starts_with(b"i")));
        self.after_rule = false;
        Ok(())
    }

    fn parse_define(&mut self, line: Bytes) -> Result<()> {
        if line.is_empty() {
            error_loc!(&*self.session, Some(&self.loc), "*** empty variable name.");
        }
        if let Some(separator) = find_outside_reference(&line, b"=") {
            let assign = parse_assign_statement(&line, separator);
            self.define_name = Some(line.slice_ref(assign.lhs));
            self.define_op = assign.op;
        } else {
            self.define_name = Some(line);
            self.define_op = AssignOp::Eq;
        }
        self.num_define_nest = 1;
        self.define_start = 0;
        self.define_start_line = self.loc.line;
        self.after_rule = false;
        Ok(())
    }

    fn parse_inside_define(&mut self, line: Bytes) -> Result<()> {
        let line = line.slice_ref(trim_left_space(&line));
        let directive = Parser::get_directive(&line);
        if directive == b"define" {
            self.num_define_nest += 1;
        } else if directive == b"endef" {
            self.num_define_nest -= 1;
        }
        if self.num_define_nest > 0 {
            if self.define_start == 0 {
                self.define_start = self.l;
            }
            return Ok(());
        }

        let rest = trim_right_space(Parser::remove_comment(trim_left_space(
            &line["endef".len()..],
        )));
        if !rest.is_empty() {
            warn_loc!(
                &*self.session,
                Some(&self.loc),
                "extraneous text after 'endef' directive"
            );
        }

        let assign_loc = Loc {
            filename: self.loc.filename,
            line: self.define_start_line,
        };
        let mut mutable_loc = assign_loc.clone();
        let lhs = parse_expr(
            self.session,
            &mut mutable_loc,
            self.define_name.clone().unwrap(),
            ParseExprOpt::Normal,
        )?;
        mutable_loc.line += 1;
        let orig_rhs = if self.define_start > 0 {
            self.buf.slice(self.define_start..(self.l - 1))
        } else {
            Bytes::new()
        };
        let rhs = parse_expr(
            self.session,
            &mut mutable_loc,
            orig_rhs.clone(),
            ParseExprOpt::Define,
        )?;

        self.out_stmts.lock().push(AssignStmt::new(
            assign_loc,
            lhs,
            rhs,
            orig_rhs,
            self.define_op,
            self.current_directive,
            false,
        ));
        self.define_name = None;
        self.define_op = AssignOp::Eq;
        Ok(())
    }

    fn enter_if(&mut self, stmt: Arc<IfStmt>) {
        self.if_stack.push(IfState {
            stmt: stmt.clone(),
            is_in_else: false,
            num_nest: self.num_if_nest,
        });
        self.out_stmts = stmt.true_stmts.clone();
    }

    fn parse_ifdef(&mut self, line: Bytes, directive: &[u8]) -> Result<()> {
        let loc = self.loc.clone();
        let op = if directive[2] == b'n' {
            CondOp::Ifndef
        } else {
            CondOp::Ifdef
        };
        let mut mutable_loc = loc.clone();
        let lhs = parse_expr(self.session, &mut mutable_loc, line, ParseExprOpt::Normal)?;
        let stmt = IfStmt::new(loc, op, lhs, None, None);
        self.out_stmts.lock().push(stmt.clone());
        self.enter_if(stmt);
        Ok(())
    }

    fn parse_ifeq(&mut self, line: Bytes, directive: &[u8]) -> Result<()> {
        let loc = self.loc.clone();
        let op = if directive[2] == b'n' {
            CondOp::Ifneq
        } else {
            CondOp::Ifeq
        };

        // Neither what the split found nor what followed the condition is said
        // here. GNU Make would not have looked at this line's condition at all
        // unless the branch around it is being taken, so both are carried on
        // the statement and reach the evaluator, which is where GNU decides.
        // The statement stands either way, because an unreadable condition is
        // still a conditional as far as the `endif` closing it is concerned.
        let (lhs_text, rhs_text, complaint) = match split_ifeq_condition(&line) {
            SplitCondition::Read(condition) => (
                line.slice(condition.lhs),
                line.slice(condition.rhs),
                (condition.rest < line.len()).then_some(CondComplaint::ExtraneousText),
            ),
            // The first string still reaches the statement when the split read
            // it, because the evaluator has to expand it before it says the
            // condition cannot be read.
            SplitCondition::Invalid { lhs } => (
                lhs.map_or_else(Bytes::new, |lhs| line.slice(lhs)),
                Bytes::new(),
                Some(CondComplaint::Unreadable),
            ),
        };

        let mut mutable_loc = loc.clone();
        let lhs = parse_expr(
            self.session,
            &mut mutable_loc,
            lhs_text,
            ParseExprOpt::Normal,
        )?;
        let rhs = parse_expr(
            self.session,
            &mut mutable_loc,
            rhs_text,
            ParseExprOpt::Normal,
        )?;

        let stmt = IfStmt::new(loc, op, lhs, Some(rhs), complaint);
        self.out_stmts.lock().push(stmt.clone());
        self.enter_if(stmt);
        Ok(())
    }

    fn parse_else(&mut self, line: Bytes) -> Result<()> {
        self.check_if_stack("else")?;
        let st = self.if_stack.last_mut().unwrap();
        if st.is_in_else {
            error_loc!(
                &*self.session,
                Some(&self.loc),
                "*** only one 'else' per conditional."
            );
        }
        st.is_in_else = true;
        self.out_stmts = st.stmt.false_stmts.clone();

        let next_if = trim_left_space(&line);
        if next_if.is_empty() {
            return Ok(());
        }

        self.num_if_nest = st.num_nest + 1;
        if !self.handle_else_if_directive(&line.slice_ref(next_if))? {
            warn_loc!(
                &*self.session,
                Some(&self.loc),
                "extraneous text after 'else' directive"
            );
        }
        self.num_if_nest = 0;
        Ok(())
    }

    fn parse_endif(&mut self, line: Bytes) -> Result<()> {
        self.check_if_stack("endif")?;
        // Complained about and then read anyway. GNU Make's `conditional_line`
        // says this through `EXTRATEXT`, which is `error` — the call that
        // prints and returns — where the `endif` with no conditional open
        // beside it is `EXTRACMD`, which is `fatal`. Two spellings, one line
        // apart in read.c, and only the second ends the read.
        if !line.is_empty() {
            warn_loc!(
                &*self.session,
                Some(&self.loc),
                "extraneous text after 'endif' directive"
            );
        }
        let num_nest = self.if_stack.last().unwrap().num_nest;
        for _ in 0..=num_nest {
            self.if_stack.pop();
        }
        if let Some(st) = self.if_stack.last() {
            if st.is_in_else {
                self.out_stmts = st.stmt.false_stmts.clone();
            } else {
                self.out_stmts = st.stmt.true_stmts.clone();
            }
        } else {
            self.out_stmts = self.stmts.clone();
        }
        Ok(())
    }

    fn is_in_export(&self) -> bool {
        self.current_directive
            .is_some_and(|d| d.export != VarExport::Default)
    }

    /// Whether the `export`/`unexport` word this line is nested inside was the
    /// exporting one, so a `private` or `override` in between reaches the same
    /// answer the outer word gave.
    fn nested_export_polarity(&self) -> bool {
        self.current_directive
            .is_some_and(|d| d.export == VarExport::Export)
    }

    fn create_export(&mut self, line: &Bytes, is_export: bool) -> Result<()> {
        let loc = self.loc.clone();
        let mut mutable_loc = loc.clone();
        let is_bare = trim_space(line).is_empty();
        let expr = parse_expr(
            self.session,
            &mut mutable_loc,
            line.clone(),
            ParseExprOpt::Normal,
        )?;
        self.out_stmts
            .lock()
            .push(ExportStmt::new(loc, expr, is_export, is_bare));
        Ok(())
    }

    fn parse_override(&mut self, line: Bytes) -> Result<()> {
        let mut current_directive = self.current_directive.unwrap_or_default();
        current_directive.is_override = true;
        self.current_directive = Some(current_directive);
        if self.handle_assign_directive(&line)? {
            return Ok(());
        }
        if self.is_in_export() {
            let polarity = self.nested_export_polarity();
            self.create_export(&line, polarity)?;
        }
        self.parse_rule_or_assign(line)
    }

    /// `private`, which defines the variable and withholds it from every scope
    /// that reaches this one through a parent.
    fn parse_private(&mut self, line: Bytes) -> Result<()> {
        let mut current_directive = self.current_directive.unwrap_or_default();
        current_directive.is_private = true;
        self.current_directive = Some(current_directive);
        if self.handle_assign_directive(&line)? {
            return Ok(());
        }
        if self.is_in_export() {
            let polarity = self.nested_export_polarity();
            self.create_export(&line, polarity)?;
        }
        self.parse_rule_or_assign(line)
    }

    fn parse_export(&mut self, line: Bytes) -> Result<()> {
        self.parse_export_directive(line, true)
    }

    /// `export` and `unexport`, which differ only in the answer they record.
    ///
    /// GNU Make looks for a variable definition on the line before it looks
    /// for a directive, so `unexport NAME = value` defines `NAME` and marks it
    /// withheld exactly as `export NAME = value` defines and marks it. The
    /// list-of-names directive is what is left when no definition is there.
    fn parse_export_directive(&mut self, line: Bytes, is_export: bool) -> Result<()> {
        let mut current_directive = self.current_directive.unwrap_or_default();
        current_directive.export = if is_export {
            VarExport::Export
        } else {
            VarExport::NoExport
        };
        self.current_directive = Some(current_directive);
        if self.handle_assign_directive(&line)? {
            return Ok(());
        }
        // A definition carries the answer itself. Recording a directive for it
        // too would name the variable before the definition runs, which GNU
        // Make never does — and `export V ?= x` would then find `V` already
        // defined and assign nothing.
        if !is_variable_definition(&line) {
            self.create_export(&line, is_export)?;
        }
        self.parse_rule_or_assign(line)
    }

    fn parse_unexport(&mut self, line: Bytes) -> Result<()> {
        self.parse_export_directive(line, false)
    }

    /// `vpath pattern dirs`, `vpath pattern`, or bare `vpath`.
    ///
    /// Which of the three it is cannot be decided here: the line is expanded
    /// when the statement runs, and a single variable can supply the pattern,
    /// the directories, or both. So the whole of it is carried across and the
    /// evaluator counts words.
    fn parse_vpath(&mut self, line: Bytes) -> Result<()> {
        let loc = self.loc.clone();
        let mut mutable_loc = loc.clone();
        let expr = parse_expr(self.session, &mut mutable_loc, line, ParseExprOpt::Normal)?;
        self.out_stmts.lock().push(VpathStmt::new(loc, expr));
        Ok(())
    }

    /// `undefine name`, whose name is expanded when the statement runs.
    fn parse_undefine(&mut self, line: Bytes) -> Result<()> {
        let loc = self.loc.clone();
        let mut mutable_loc = loc.clone();
        let expr = parse_expr(self.session, &mut mutable_loc, line, ParseExprOpt::Normal)?;
        let is_override = self.current_directive.is_some_and(|d| d.is_override);
        self.out_stmts
            .lock()
            .push(UndefineStmt::new(loc, expr, is_override));
        self.after_rule = false;
        Ok(())
    }

    fn check_if_stack(&self, keyword: &'static str) -> Result<()> {
        if self.if_stack.is_empty() {
            error_loc!(
                &*self.session,
                Some(&self.loc),
                "*** extraneous '{keyword}'."
            );
        }
        Ok(())
    }

    fn remove_comment(line: &[u8]) -> &[u8] {
        if let Some(i) = find_outside_paren(line, b"#") {
            return &line[..i];
        }
        line
    }

    fn get_directive(line: &[u8]) -> &[u8] {
        if line.len() < 4 {
            return &[];
        }
        let l = &line[0..9.min(line.len())];
        if let Some(i) = memchr3(b' ', b'\t', b'#', l) {
            return &l[..i];
        }
        l
    }

    fn handle_make_directive(&mut self, line: &Bytes) -> Result<bool> {
        let directive = Parser::get_directive(line);
        let rest = line.slice_ref(trim_right_space(Parser::remove_comment(trim_left_space(
            &line[directive.len()..],
        ))));
        match directive {
            b"include" | b"-include" | b"sinclude" => self.parse_include(rest, directive)?,
            b"define" => self.parse_define(rest)?,
            b"ifdef" | b"ifndef" => self.parse_ifdef(rest, directive)?,
            b"ifeq" | b"ifneq" => self.parse_ifeq(rest, directive)?,
            b"else" => self.parse_else(rest)?,
            b"endif" => self.parse_endif(rest)?,
            b"override" if !starts_assignment(&rest) => self.parse_override(rest)?,
            b"export" if !starts_assignment(&rest) => self.parse_export(rest)?,
            b"private" if !starts_assignment(&rest) => self.parse_private(rest)?,
            b"unexport" => self.parse_unexport(rest)?,
            b"undefine" if !starts_assignment(&rest) => self.parse_undefine(rest)?,
            b"vpath" => self.parse_vpath(rest)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn handle_else_if_directive(&mut self, line: &Bytes) -> Result<bool> {
        let directive = Parser::get_directive(line);
        let rest = line.slice_ref(trim_right_space(Parser::remove_comment(trim_left_space(
            &line[directive.len()..],
        ))));
        match directive {
            b"ifdef" | b"ifndef" => self.parse_ifdef(rest, directive)?,
            b"ifeq" | b"ifneq" => self.parse_ifeq(rest, directive)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn handle_assign_directive(&mut self, line: &Bytes) -> Result<bool> {
        let directive = Parser::get_directive(line);
        let rest = line.slice_ref(trim_right_space(Parser::remove_comment(trim_left_space(
            &line[directive.len()..],
        ))));
        match directive {
            b"define" => self.parse_define(rest)?,
            b"override" if !starts_assignment(&rest) => self.parse_override(rest)?,
            b"export" if !starts_assignment(&rest) => self.parse_export(rest)?,
            b"private" if !starts_assignment(&rest) => self.parse_private(rest)?,
            b"undefine" if !starts_assignment(&rest) => self.parse_undefine(rest)?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

pub fn parse_file(
    session: &mut Session,
    buf: &Bytes,
    filename: Symbol,
) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    collect_stats!(&*session, "parse file time");
    let loc = Loc { filename, line: 0 };
    parse_buf_no_stats_impl(session, buf, loc, false)
}

pub fn parse_buf(session: &mut Session, buf: &Bytes, loc: Loc) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    collect_stats!(&*session, "parse eval time");
    parse_buf_no_stats_impl(session, buf, loc, true)
}

pub fn parse_buf_no_stats(
    session: &mut Session,
    buf: &Bytes,
    loc: Loc,
) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    parse_buf_no_stats_impl(session, buf, loc, true)
}

fn parse_buf_no_stats_impl(
    session: &mut Session,
    buf: &Bytes,
    loc: Loc,
    fixed_lineno: bool,
) -> Result<Arc<Mutex<Vec<Stmt>>>> {
    let stmts = Arc::new(Mutex::new(Vec::new()));
    let mut p = Parser::with_buf(session, buf, loc, stmts.clone(), fixed_lineno);
    p.parse()?;
    Ok(stmts)
}

/// Whether an assignment operator comes first, which makes the word before it a
/// variable name rather than a directive: `undefine = x` defines `undefine`.
/// Whether this line defines a variable, which is what GNU Make asks before it
/// asks whether the line is a directive.
///
/// The same question [`Parser::parse_rule_or_assign`] answers by dispatching,
/// asked without dispatching, so a caller can tell which of the two a line
/// carrying an `export` word in front of it turned out to be.
fn is_variable_definition(line: &[u8]) -> bool {
    let Some(sep) = find_outside_reference(line, b":=") else {
        return false;
    };
    let rest = &line[sep..];
    let name_end = if rest.starts_with(b"=") {
        sep
    } else if rest.starts_with(b":") {
        let colons = rest.iter().take_while(|byte| **byte == b':').count();
        if !(1..=3).contains(&colons) || rest.get(colons) != Some(&b'=') {
            return false;
        }
        sep + colons
    } else {
        return false;
    };
    name_end == 0 || is_variable_name(parse_assign_statement(line, name_end).lhs)
}

fn starts_assignment(rest: &[u8]) -> bool {
    [b"=".as_slice(), b":=", b"::=", b":::=", b"+=", b"?=", b"!="]
        .iter()
        .any(|op| rest.starts_with(op))
}

/// Whether `name` names a variable.
///
/// GNU Make requires one word: the assignment operator may be separated from
/// the name by blanks, but a blank anywhere else means the line is not an
/// assignment at all and is read as a rule, which then has no separator. Blanks
/// inside a `$(...)` reference do not divide the name, since the reference is
/// one token however it expands.
pub(crate) fn is_variable_name(name: &[u8]) -> bool {
    let mut depth = 0usize;
    let mut i = 0;
    while i < name.len() {
        match name[i] {
            // A `$` reads the character after it as the name it references, so
            // a blank there is inside the reference rather than the break
            // between two words that would stop this being a name at all. A
            // `$(` opens a reference the parenthesis depth already follows.
            b'$' if !matches!(name.get(i + 1), Some(b'(' | b'{')) => i += 1,
            b'(' | b'{' => depth += 1,
            b')' | b'}' => depth = depth.saturating_sub(1),
            b' ' | b'\t' if depth == 0 => return false,
            _ => {}
        }
        i += 1;
    }
    true
}

/// Take the modifier keywords a target-specific assignment may carry in front
/// of its variable name, in any order and any number, and answer whether
/// `private` was among them.
///
/// The last word is the name however it is spelled, so `a: private = 1` assigns
/// to `private` rather than declaring one. `export`, `unexport` and `override`
/// are keywords in this position too and are taken off the name.
pub fn take_assign_modifiers(name: &[u8]) -> (&[u8], AssignModifiers) {
    let mut name = name;
    let mut modifiers = AssignModifiers::default();
    while let Some(end) = memchr2(b' ', b'\t', name) {
        match &name[..end] {
            b"private" => modifiers.directive.is_private = true,
            b"export" => modifiers.directive.export = VarExport::Export,
            b"override" => modifiers.directive.is_override = true,
            b"unexport" => modifiers.directive.export = VarExport::NoExport,
            _ => break,
        }
        modifiers.words += 1;
        name = trim_left_space(&name[end..]);
    }
    (name, modifiers)
}

/// Remove the modifier words the parser saw literally. Expanded text that
/// happens to spell a modifier is part of the variable name, not a keyword.
pub fn strip_assign_modifiers(mut name: &[u8], words: usize) -> &[u8] {
    for _ in 0..words {
        let Some(end) = memchr2(b' ', b'\t', name) else {
            return name;
        };
        name = trim_left_space(&name[end..]);
    }
    name
}

pub struct ParsedAssign<'a> {
    pub lhs: &'a [u8],
    pub rhs: &'a [u8],
    pub op: AssignOp,
}
pub fn parse_assign_statement(line: &[u8], sep: usize) -> ParsedAssign<'_> {
    assert!(sep != 0);
    let mut op = AssignOp::Eq;
    let mut lhs = &line[..sep];
    if written_operator(lhs, b":::") {
        lhs = &lhs[..lhs.len() - 3];
        op = AssignOp::ImmediateRecursive;
    } else if written_operator(lhs, b"::") {
        lhs = &lhs[..lhs.len() - 2];
        op = AssignOp::ColonEq;
    } else if written_operator(lhs, b":") {
        lhs = &lhs[..lhs.len() - 1];
        op = AssignOp::ColonEq;
    } else if written_operator(lhs, b"+") {
        lhs = &lhs[..lhs.len() - 1];
        op = AssignOp::PlusEq;
    } else if written_operator(lhs, b"?") {
        lhs = &lhs[..lhs.len() - 1];
        op = AssignOp::QuestionEq;
    } else if written_operator(lhs, b"!") {
        lhs = &lhs[..lhs.len() - 1];
        op = AssignOp::ShellEq;
    }
    let name_end = trim_right_space(lhs).len();
    // GNU Make scans for the assignment operator by consuming the character
    // after a `$` as the name that reference reads, so that character is inside
    // the variable name however the split fell — `U$ := b` names `U$ `, which
    // is `U` and the unset variable named " " (variable.c
    // parse_variable_definition, the `default: continue` arm). Splitting the
    // line by position and then trimming would leave the `$` at the end of a
    // text, where it would read as the literal dollar a `$` at the end of a
    // value is. An even run of dollars is written text rather than a reference,
    // and takes nothing after it.
    let takes_the_next = ends_with_reference_dollar(&line[..name_end]);
    let name_end = if takes_the_next && name_end < line.len() {
        name_end + 1
    } else {
        name_end
    };
    let lhs = trim_left_space(&line[..name_end]);
    let rhs = trim_left_space(&line[line.len().min(sep + 1)..]);
    ParsedAssign { lhs, rhs, op }
}

/// Whether `text` ends with `operator` written as an operator, rather than with
/// its first character read by a `$` standing in front of it.
///
/// GNU Make's scan spends the character after a `$` on the reference, so an
/// operator that begins there was never one: `T$:= a` assigns with `=` and
/// `T$:::= f` with `::=`, the leading colon of each having been read as the
/// name `$:` references.
fn written_operator(text: &[u8], operator: &[u8]) -> bool {
    text.ends_with(operator) && !ends_with_reference_dollar(&text[..text.len() - operator.len()])
}

/// Whether the text ends with a `$` that begins a reference, rather than with
/// written dollars.
///
/// GNU Make pairs the dollars off as it reads, so an odd run leaves one that
/// takes whatever follows it as a name and an even run leaves none.
fn ends_with_reference_dollar(text: &[u8]) -> bool {
    text.iter().rev().take_while(|byte| **byte == b'$').count() % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A condition the split cannot read is still a conditional, because the
    /// `endif` that closes it has to find it. GNU Make counts the nesting of a
    /// branch it is ignoring without reading a byte of any condition in it, so
    /// the statements after the inner `endif` belong to the outer branch and
    /// the ones after the outer `endif` belong to the file.
    #[test]
    fn an_unreadable_condition_still_holds_its_nesting_open() {
        let mut session = Session::new();
        let stmts = parse_buf(
            &mut session,
            &Bytes::from_static(b"ifeq (x,y)\nifeq (a,a junk\nendif\nX := 1\nendif\nY := 2\n"),
            Loc::default(),
        )
        .expect("the read carries on past a condition it cannot split");

        let top = stmts.lock();
        assert_eq!(top.len(), 2, "the outer conditional and `Y := 2` after it");
        let outer = format!("{:?}", top[0]);
        assert!(
            outer.contains("t=2") && outer.contains("f=0"),
            "the inner conditional and `X := 1` sit inside the outer branch: {outer}"
        );
    }

    /// How many `endif`s an `else if...` whose condition cannot be read takes.
    ///
    /// GNU Make has two answers and picks between them while it reads.
    /// `conditional_line` (reference/gnumake/src/read.c) recurses into the text
    /// after an `else`, and the recursion does `o = conditionals->if_cmds++`
    /// before it looks at the condition. On the success path the `else` gives
    /// that level straight back with `--conditionals->if_cmds`; on the `< 0`
    /// path it reaches `EXTRATEXT()` instead, which prints and returns, so the
    /// level stays and the rest of the file is one `endif` short. But a
    /// condition inside a branch that is already being ignored is never read at
    /// all -- the recursion's own `for (i = 0; i < o; ++i) if (ignoring[i])`
    /// returns 1 before the condition is touched -- so the same text takes one
    /// `endif` when the `else` arm is dead and two when it is live.
    ///
    /// Which one is decided by the value of the enclosing condition, and this
    /// reader compiles the whole file before it evaluates any of it. It has to
    /// answer once, and it answers with the dead-arm reading: one `endif`
    /// closes the `else` and the conditional the `else` carried.
    ///
    /// The reason is the direction of the disagreement rather than a count of
    /// cells. This reading refuses a file GNU Make builds -- the one written
    /// with the second `endif` that GNU Make's refusal path leaves room for --
    /// and never builds a file GNU Make refuses. The other reading trades those
    /// round, and contradicts `an-else-if-under-a-taken-branch-is-not-read` and
    /// `a-second-endif-after-an-unread-else-if-is-extraneous` in the port
    /// corpus while it is at it.
    #[test]
    fn an_unreadable_else_if_is_closed_by_the_endif_that_closes_the_else() {
        let mut session = Session::new();
        let stmts = parse_buf(
            &mut session,
            &Bytes::from_static(b"ifeq (a,a)\nelse ifeq (x,y junk\nendif\nX := 1\n"),
            Loc::default(),
        )
        .expect("the read carries on past a condition it cannot split");
        {
            let top = stmts.lock();
            assert_eq!(
                top.len(),
                2,
                "the one conditional and `X := 1` after it: {top:?}"
            );
        }

        // And so the `endif` GNU Make would have wanted for the level its
        // refusal left standing has nothing to close here.
        let mut session = Session::new();
        let refused = parse_buf(
            &mut session,
            &Bytes::from_static(b"ifeq (a,a)\nelse ifeq (x,y junk\nendif\nendif\nX := 1\n"),
            Loc::default(),
        )
        .expect_err("a second endif closes nothing");
        assert!(
            refused.to_string().contains("extraneous 'endif'"),
            "{refused}"
        );
    }

    /// Which refusals carry a first string and which do not, which is the
    /// whole of what decides whether the expansion has happened when the
    /// condition is refused.
    ///
    /// `conditional_line` expands the first string as soon as it has found its
    /// end, so the two ways of giving up above that line hand back nothing and
    /// the three below it hand back what was read.
    #[test]
    fn a_refusal_carries_the_first_string_it_had_already_read() {
        let read = |text: &'static [u8]| match split_ifeq_condition(text) {
            SplitCondition::Read(condition) => {
                Some(String::from_utf8_lossy(&text[condition.lhs]).into_owned())
            }
            SplitCondition::Invalid { lhs } => {
                lhs.map(|lhs| String::from_utf8_lossy(&text[lhs]).into_owned())
            }
        };

        // Refused above the expansion: nothing has been read.
        assert_eq!(read(b""), None, "no opener at all");
        assert_eq!(read(b"junk"), None, "an opener that is neither");
        assert_eq!(
            read(b"(a b"),
            None,
            "the first string's comma never arrives"
        );
        assert_eq!(
            read(b"\"a b"),
            None,
            "the first string's quote never arrives"
        );

        // Refused below it: the first string is read and must be expanded.
        assert_eq!(
            read(b"\"a\""),
            Some("a".to_owned()),
            "text ends after the close quote"
        );
        assert_eq!(
            read(b"\"a\" b"),
            Some("a".to_owned()),
            "a second terminator that is neither"
        );
        assert_eq!(
            read(b"(a,b"),
            Some("a".to_owned()),
            "the second string's close never arrives"
        );

        // Read: the same first string, by the ordinary road.
        assert_eq!(
            read(b"(a ,b)"),
            Some("a".to_owned()),
            "blanks before the comma belong to neither"
        );
        assert_eq!(
            read(b"\"a\" \"b\""),
            Some("a".to_owned()),
            "the quoted form"
        );
    }

    #[test]
    fn test_get_directive() {
        assert_eq!(
            Parser::get_directive(&Bytes::from_static(b"ifdef VAR")),
            Bytes::from_static(b"ifdef")
        );
        assert_eq!(
            Parser::get_directive(&Bytes::from_static(b"endif")),
            Bytes::from_static(b"endif")
        );
    }

    /// The character after a `$` is the name that reference reads, so the split
    /// between name and operator keeps it on the name's side rather than
    /// leaving a `$` at the end of a text, where it would be a written dollar.
    #[test]
    fn a_name_ending_in_a_dollar_keeps_what_the_dollar_reads() {
        let name = |line: &'static [u8], sep: usize| {
            String::from_utf8_lossy(parse_assign_statement(line, sep).lhs).into_owned()
        };
        assert_eq!(name(b"U$ := b", 4), "U$ ");
        assert_eq!(name(b"T$:= a", 3), "T$:");
        assert_eq!(name(b"A$ = b", 3), "A$ ");
        assert_eq!(name(b"A$$ := b", 5), "A$$");
        assert_eq!(name(b"A := b", 3), "A");
        assert_eq!(name(b"A = b", 2), "A");
    }

    /// A blank a `$` reads is inside the reference, so it does not stop the
    /// text being a name; a blank between two words still does.
    #[test]
    fn a_blank_a_dollar_reads_leaves_a_name() {
        assert!(is_variable_name(b"U$ "));
        assert!(is_variable_name(b"T$:"));
        assert!(is_variable_name(b"$(FOO BAR)"));
        assert!(is_variable_name(b"A"));
        assert!(!is_variable_name(b"A $$"));
        assert!(!is_variable_name(b"A B"));
    }

    /// An operator character a `$` reads is part of the name, so the operator
    /// is whatever is written after it — the shorter one, or none at all.
    #[test]
    fn an_operator_a_dollar_reads_is_not_the_operator() {
        let split = |line: &'static [u8], sep: usize| {
            let assign = parse_assign_statement(line, sep);
            (
                String::from_utf8_lossy(assign.lhs).into_owned(),
                assign.op,
                String::from_utf8_lossy(assign.rhs).into_owned(),
            )
        };
        assert_eq!(
            split(b"T$:= a", 3),
            ("T$:".into(), AssignOp::Eq, "a".into())
        );
        assert_eq!(
            split(b"P$+= more", 3),
            ("P$+".into(), AssignOp::Eq, "more".into())
        );
        assert_eq!(
            split(b"C$?= c", 3),
            ("C$?".into(), AssignOp::Eq, "c".into())
        );
        assert_eq!(
            split(b"D$!= d", 3),
            ("D$!".into(), AssignOp::Eq, "d".into())
        );
        // The first colon is the name's; `::=` after it is the operator.
        assert_eq!(
            split(b"E$::= e", 4),
            ("E$:".into(), AssignOp::ColonEq, "e".into())
        );
        assert_eq!(
            split(b"F$:::= f", 5),
            ("F$:".into(), AssignOp::ColonEq, "f".into())
        );
        // An even run of dollars reads nothing after it, so `:=` is written.
        assert_eq!(
            split(b"G$$:= g", 4),
            ("G$$".into(), AssignOp::ColonEq, "g".into())
        );
        // A parenthesised reference closes itself, leaving `+=` written.
        assert_eq!(
            split(b"H$(X)+= h", 6),
            ("H$(X)".into(), AssignOp::PlusEq, "h".into())
        );
    }

    /// A `;` takes no part in the question "does this line define a variable".
    ///
    /// `parse_variable_definition` has no case for one, so the scan walks past
    /// it to whatever operator is written after — which makes the semicolon
    /// part of the variable's name. It is the rule-and-recipe separator only in
    /// the second question, which is asked of a line that defines nothing.
    #[test]
    fn a_semicolon_does_not_stop_an_assignment_scan() {
        for line in [
            b"a;b=c".as_slice(),
            b"x;y := z",
            b"a;b += c",
            b"a;b ?= c",
            b"a;b != echo c",
            b"a;b:::= c",
        ] {
            assert!(is_variable_definition(line), "{}", line.escape_ascii());
        }
        // The name really is the whole of it, semicolon included.
        assert_eq!(parse_assign_statement(b"a;b=c", 3).lhs, b"a;b");
        assert_eq!(parse_assign_statement(b"x;y := z", 5).lhs, b"x;y");
        let appended = parse_assign_statement(b"a;b += c", 5);
        assert_eq!(appended.lhs, b"a;b");
        assert_eq!(appended.op, AssignOp::PlusEq);
        // A blank still ends the name, so a second word before the operator
        // means the line defines nothing however the semicolon fell.
        for line in [b"a; b = c".as_slice(), b"foo ; bar=baz"] {
            assert!(!is_variable_definition(line), "{}", line.escape_ascii());
        }
        // A colon that is not an assignment operator leaves this a rule, and
        // the rule parse is where the semicolon becomes the recipe separator.
        for line in [b"a;b: c".as_slice(), b"all: a;b=c", b"all: dep ; recipe"] {
            assert!(!is_variable_definition(line), "{}", line.escape_ascii());
        }
    }

    /// A target-specific assignment carries its modifiers in any order, and the
    /// word left over is the name however it is spelled."""

    #[test]
    fn modifiers_come_off_a_target_specific_name() {
        for (line, name, words, is_private) in [
            (b"F".as_slice(), b"F".as_slice(), 0, false),
            (b"private F", b"F", 1, true),
            (b"export override private _X", b"_X", 3, true),
            (b"private override B", b"B", 2, true),
            (b"override X", b"X", 1, false),
            (b"private", b"private", 0, false),
            (b"X Y", b"X Y", 0, false),
        ] {
            let (actual, modifiers) = take_assign_modifiers(line);
            assert_eq!(actual, name);
            assert_eq!(modifiers.words, words);
            assert_eq!(modifiers.directive.is_private, is_private);
            assert_eq!(strip_assign_modifiers(line, words), name);
        }
    }

    #[test]
    fn posix_assignment_operators_are_distinct() {
        for (line, separator, name, op) in [
            (b"x:=y".as_slice(), 2, b"x".as_slice(), AssignOp::ColonEq),
            (b"x::=y", 3, b"x", AssignOp::ColonEq),
            (b"x:::=y", 4, b"x", AssignOp::ImmediateRecursive),
        ] {
            let assign = parse_assign_statement(line, separator);
            assert_eq!(assign.lhs, name);
            assert_eq!(assign.op, op);
            assert_eq!(assign.rhs, b"y");
        }
    }

    /// A name is one word once the operator is off it, and a `$(...)` reference
    /// is one word however many spaces it holds.
    #[test]
    fn a_variable_name_is_one_word() {
        for name in [b"x".as_slice(), b"$(a b)", b"a$(b c)d", b"$X"] {
            assert!(is_variable_name(name), "{}", String::from_utf8_lossy(name));
        }
        for name in [b"x y".as_slice(), b"x $X", b"x $(a b)", b"a\\ b"] {
            assert!(!is_variable_name(name), "{}", String::from_utf8_lossy(name));
        }
    }
}
