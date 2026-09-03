//! Lifting the static child invocations out of a recipe line, by reading the
//! line as the shell would.
//!
//! A recipe line that starts a Make is compiled into the parent's graph
//! instead of being run, when the compiler can say exactly which Make it
//! starts, in which directory, with which goals, and under what condition.
//! This module is where that is said. It reads the line through nsh — the
//! shell Ronin runs the line with, so the reading is the one that would have
//! run — and walks the shell's own grammar with one question at every node:
//! is what this construct does exactly what a list of composed children does?
//!
//! A list of composed children runs in order, each only after the last one
//! finished and never after it failed, and the recipe fails when one fails.
//! That is what `&&` means, and it is what
//! `for X in …; do (cd $X && $(MAKE) …) || exit 1; done` means once `X`'s
//! values are known: iteration two starts after iteration one finished and
//! not after it failed, because `exit 1` ended the line. It is not what `;`
//! means, nor a loop whose body carries on past a failed iteration, nor
//! `|| true`, nor a word list that holds a command substitution. Every one of
//! those refuses, and refuses with the reason — because a lift is an
//! optimisation whose fallback, running the line as written, is always
//! right, so nothing may be lifted whose semantics are not provably kept.
//!
//! The walk carries the shell state a line can settle for itself — the
//! variables it assigned literal values, the directory it changed into, and
//! whether `set -e` is armed — and a context saying what a failure where it
//! stands would do to the line. An invocation is admitted only where its
//! failure ends the line and fails it: because nothing follows it, because
//! an `exit` catches it, or because errexit is live there.

use bytes::Bytes;
use nsh::script::{Assignment, Command, Piece, Reader, Script, SimpleCommand, Word};

use crate::census::NestingReason;
use crate::command::LiftedInvocation;

thread_local! {
    /// One reader per thread, built on first use and kept: building a shell
    /// costs a locale and a variable table, and a compilation reads many
    /// lines.
    static READER: std::cell::RefCell<Option<Reader>> = const { std::cell::RefCell::new(None) };
}

/// Read `line` as the shell would.
fn read(line: &[u8]) -> Result<Script, NestingReason> {
    READER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let reader = match slot.as_mut() {
            Some(reader) => reader,
            None => slot.insert(Reader::new().map_err(|_| NestingReason::Unreadable)?),
        };
        reader
            .read(line.into())
            .map_err(|_| NestingReason::Unreadable)
    })
}

/// The static child invocations one recipe line names, in the order the
/// shell would run them, or why the line has to run as written.
///
/// `Err` is a line the walk could not prove equal to a list of composed
/// children, with the first construct that stopped the proof — whether or
/// not the line starts a Make anywhere, which is the caller's question to
/// ask of the line before it reads the reason as a nesting. `Ok` with
/// nothing in it is a line made of nothing but what the walk settles: an
/// assignment, a `set`, a guard that is false. `errexit` is whether the
/// shell flags the line runs under arm `-e` before the line's own `set`.
// [spec:ronin:req:make.recursive-invocation+3]
pub fn lift(
    line: &Bytes,
    make_values: &[Bytes],
    errexit: bool,
) -> Result<Vec<LiftedInvocation>, NestingReason> {
    if make_values.is_empty() {
        return Ok(Vec::new());
    }
    let script = read(line)?;
    let mut walk = Walk {
        make_values,
        state: State {
            bindings: Vec::new(),
            cwd: None,
            errexit,
        },
        found: Vec::new(),
    };
    let top = Context {
        aborts: true,
        exempt: false,
        in_subshell: false,
        carrier: Carrier::Line,
    };
    let commands = script.commands;
    let last = commands.len().saturating_sub(1);
    for (index, command) in commands.iter().enumerate() {
        // A `.ONESHELL` recipe or a line holding a literal newline is several
        // lines to the shell, and every line but the last is followed by
        // another whatever it did.
        let context = if index == last {
            top
        } else {
            Context {
                aborts: false,
                carrier: Carrier::Sequence,
                ..top
            }
        };
        walk.command(command, context)?;
    }
    Ok(walk.found)
}

/// What a failure where a command stands does to the line.
#[derive(Clone, Copy)]
struct Context {
    /// Whether a command failing here ends the line and fails it.
    aborts: bool,
    /// Whether this position is one POSIX exempts from `-e`: any command of
    /// an AND-OR list but the last, and everything inside one.
    exempt: bool,
    /// Whether an `exit` here ends a subshell rather than the line.
    in_subshell: bool,
    /// Which construct decided `aborts`, for the reason a refusal gives.
    carrier: Carrier,
}

#[derive(Clone, Copy)]
enum Carrier {
    Line,
    Sequence,
    Loop,
}

/// The shell state the line has settled for itself so far.
#[derive(Clone)]
struct State {
    /// Variables assigned a literal value on this line, latest last.
    bindings: Vec<(Vec<u8>, Vec<u8>)>,
    /// Where a `cd` outside any subshell has put the shell, relative to
    /// where the line started, or absolute.
    cwd: Option<Vec<u8>>,
    /// Whether `-e` is armed here.
    errexit: bool,
}

impl State {
    fn bound(&self, name: &[u8]) -> Option<&[u8]> {
        self.bindings
            .iter()
            .rev()
            .find(|(bound, _)| bound == name)
            .map(|(_, value)| value.as_slice())
    }

    fn bind(&mut self, name: &[u8], value: &[u8]) {
        self.bindings.push((name.to_vec(), value.to_vec()));
    }
}

struct Walk<'a> {
    make_values: &'a [Bytes],
    state: State,
    found: Vec<LiftedInvocation>,
}

/// One word as the invocation will be spelled from it.
enum Spelled {
    /// A field the walk resolved, to be quoted as needed.
    Field(Vec<u8>),
    /// A word passed through in the shell's own spelling of it.
    Verbatim(Vec<u8>),
}

impl Walk<'_> {
    fn command(&mut self, command: &Command, context: Context) -> Result<(), NestingReason> {
        match command {
            Command::Simple(simple) => self.simple(simple, context),
            Command::And(left, right) => {
                if let Some((directory, invocation)) = self.entered_invocation(left, right)? {
                    return self.invoke(invocation, Some(directory), context);
                }
                self.command(
                    left,
                    Context {
                        exempt: true,
                        ..context
                    },
                )?;
                self.command(right, context)
            }
            Command::Or(left, right) => {
                if !is_failing_exit(right) {
                    return Err(NestingReason::Alternation);
                }
                // `exit` ends the shell it runs in. In the line's own shell
                // that is the line, failed; in a subshell it is the subshell,
                // failed, and what that does is the subshell's position.
                let left_context = if context.in_subshell {
                    Context {
                        exempt: true,
                        ..context
                    }
                } else {
                    Context {
                        aborts: true,
                        exempt: true,
                        ..context
                    }
                };
                self.command(left, left_context)
            }
            Command::Sequence(left, right) => {
                self.command(
                    left,
                    Context {
                        aborts: false,
                        carrier: Carrier::Sequence,
                        ..context
                    },
                )?;
                self.command(right, context)
            }
            Command::Subshell(inner) => {
                // Nothing the subshell settles reaches the line, so the
                // state it walks with is thrown away with it.
                let saved = self.state.clone();
                let outcome = self.command(
                    inner,
                    Context {
                        in_subshell: true,
                        ..context
                    },
                );
                self.state = saved;
                outcome
            }
            Command::If {
                condition,
                then,
                otherwise,
            } => match self.decide(condition)? {
                true => self.command(then, context),
                false => match otherwise {
                    Some(branch) => self.command(branch, context),
                    None => Ok(()),
                },
            },
            Command::For {
                variable,
                words,
                body,
            } => {
                let mut values = Vec::new();
                for word in words {
                    values.extend(self.fields(word)?);
                }
                for value in values {
                    self.state.bind(variable, &value);
                    self.command(
                        body,
                        Context {
                            aborts: false,
                            carrier: Carrier::Loop,
                            ..context
                        },
                    )?;
                }
                Ok(())
            }
            Command::Pipeline { .. } => Err(NestingReason::Pipeline),
            Command::Redirected { .. } => Err(NestingReason::Redirection),
            Command::Background(_) => Err(NestingReason::Construct {
                construct: "a command started in the background",
            }),
            Command::While { .. } => Err(NestingReason::Construct {
                construct: "a `while` loop",
            }),
            Command::Until { .. } => Err(NestingReason::Construct {
                construct: "an `until` loop",
            }),
            Command::Case { .. } => Err(NestingReason::Construct {
                construct: "a `case`",
            }),
            Command::Function { .. } => Err(NestingReason::Construct {
                construct: "a function definition",
            }),
            _ => Err(NestingReason::Construct {
                construct: "a shell form the compiler does not read",
            }),
        }
    }

    /// `cd DIR && $(MAKE) …`: the one `&&` whose left is not a command of
    /// its own but where the right is run. The directory travels with the
    /// invocation, joined onto whatever the line had already entered.
    fn entered_invocation<'c>(
        &mut self,
        left: &Command,
        right: &'c Command,
    ) -> Result<Option<(Vec<u8>, &'c SimpleCommand)>, NestingReason> {
        let Command::Simple(entered) = left else {
            return Ok(None);
        };
        let Command::Simple(invocation) = right else {
            return Ok(None);
        };
        if !entered.assignments.is_empty()
            || !entered.redirections.is_empty()
            || entered.words.len() != 2
            || entered.words[0].source != "cd"
        {
            return Ok(None);
        }
        let mut fields = self.fields(&entered.words[1])?;
        if fields.len() != 1 || fields[0].starts_with(b"-") {
            return Err(NestingReason::DirectoryChange);
        }
        let directory = self.joined(&fields.remove(0));
        // The `cd` is the shell's own and outlives the invocation it was
        // written for: `cd a && make x && make y` runs both in `a`. Inside a
        // subshell the walk's state is restored when the subshell closes.
        self.state.cwd = Some(directory.clone());
        Ok(Some((directory, invocation)))
    }

    /// Where `directory` is from where the line started.
    fn joined(&self, directory: &[u8]) -> Vec<u8> {
        match &self.state.cwd {
            Some(cwd) if !directory.starts_with(b"/") => {
                let mut joined = cwd.clone();
                if !joined.ends_with(b"/") {
                    joined.push(b'/');
                }
                joined.extend_from_slice(directory);
                joined
            }
            _ => directory.to_vec(),
        }
    }

    fn simple(&mut self, simple: &SimpleCommand, context: Context) -> Result<(), NestingReason> {
        if !simple.redirections.is_empty() {
            return Err(NestingReason::Redirection);
        }
        if simple.words.is_empty() {
            // Assignments alone: the line is settling a value for itself.
            for assignment in &simple.assignments {
                self.assign(assignment)?;
            }
            return Ok(());
        }
        if !simple.assignments.is_empty() {
            return Err(NestingReason::Prefix);
        }
        let words = simple.words.iter().collect::<Vec<_>>();
        let Some(first) = words.first().copied() else {
            return Ok(());
        };
        if let Some((make, used)) = self.make_prefix(&words)? {
            return self.invoke_words(&make, words[used..].iter().copied(), None, context);
        }
        let name = self.fields(first)?;
        let name = match name.as_slice() {
            [name] => name.clone(),
            _ => {
                return Err(NestingReason::BesideAnotherCommand {
                    command: String::from_utf8_lossy(&first.source).into_owned(),
                });
            }
        };
        let rest = &words[1..];
        match name.as_slice() {
            b":" | b"true" => Ok(()),
            b"set" => self.set(rest.iter().copied()),
            // `exec` in front of the invocation runs the same Make and nothing
            // after it; a line that IS the invocation has nothing after it
            // either.
            b"exec" => match self.make_prefix(rest)? {
                Some((make, used)) => {
                    self.invoke_words(&make, rest[used..].iter().copied(), None, context)
                }
                None => Err(NestingReason::BesideAnotherCommand {
                    command: "exec".to_owned(),
                }),
            },
            b"cd" => self.change_directory(rest.iter().copied(), context),
            b"env" => Err(NestingReason::Prefix),
            b"exit" => Err(NestingReason::Construct {
                construct: "an `exit` that ends the line",
            }),
            _ => Err(NestingReason::BesideAnotherCommand {
                command: String::from_utf8_lossy(&name).into_owned(),
            }),
        }
    }

    /// Whether a command failing at `context` ends the line and fails it,
    /// which is the one thing a composed child's failure does.
    fn admitted(&self, context: Context) -> Result<(), NestingReason> {
        if context.aborts || (self.state.errexit && !context.exempt) {
            return Ok(());
        }
        Err(match context.carrier {
            Carrier::Loop => NestingReason::LoopCarriesOn,
            Carrier::Line | Carrier::Sequence => NestingReason::Sequence,
        })
    }

    /// The `MAKE` value the line's leading words spell, and how many words
    /// they take.
    ///
    /// `MAKE` is Make's text and not one shell word: `make -j8` is a value
    /// kati hands a child, and a Makefile may say `MAKE = $(MAKE_COMMAND)
    /// --no-print-directory`. The invocation is the words after the value's
    /// words, so the value is matched word by word and passed on as the text
    /// it is.
    fn make_prefix(&self, words: &[&Word]) -> Result<Option<(Bytes, usize)>, NestingReason> {
        for make in self.make_values {
            let tokens = make
                .split(|byte| byte.is_ascii_whitespace())
                .filter(|token| !token.is_empty())
                .collect::<Vec<_>>();
            if tokens.is_empty() || words.is_empty() {
                continue;
            }
            let mut fields = Vec::new();
            let mut used = 0;
            while fields.len() < tokens.len() && used < words.len() {
                fields.extend(self.fields(words[used])?);
                used += 1;
            }
            if fields.len() == tokens.len()
                && fields
                    .iter()
                    .zip(&tokens)
                    .all(|(field, token)| field.as_slice() == *token)
            {
                return Ok(Some((make.clone(), used)));
            }
        }
        Ok(None)
    }

    /// `set -e` and `set +e` are the line arming or disarming errexit for
    /// what follows; `-x` and `-u` change nothing the walk decides on.
    fn set<'w>(&mut self, words: impl Iterator<Item = &'w Word>) -> Result<(), NestingReason> {
        for word in words {
            let fields = self.fields(word)?;
            for field in fields {
                let (on, flags) = match field.split_first() {
                    Some((b'-', flags)) => (true, flags),
                    Some((b'+', flags)) => (false, flags),
                    _ => {
                        return Err(NestingReason::Construct {
                            construct: "a `set` the compiler does not read",
                        });
                    }
                };
                for flag in flags {
                    match flag {
                        b'e' => self.state.errexit = on,
                        b'x' | b'u' | b'v' => {}
                        _ => {
                            return Err(NestingReason::Construct {
                                construct: "a `set` option the compiler does not read",
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A `cd` of its own, outside the `cd DIR && $(MAKE)` idiom, moves the
    /// line for everything after it.
    fn change_directory<'w>(
        &mut self,
        mut words: impl Iterator<Item = &'w Word>,
        context: Context,
    ) -> Result<(), NestingReason> {
        // A `cd` that fails and is not the end of the line leaves the next
        // command running where the line started, which is not where the
        // composition would put it.
        self.admitted(context)?;
        let Some(target) = words.next() else {
            return Err(NestingReason::DirectoryChange);
        };
        if words.next().is_some() {
            return Err(NestingReason::DirectoryChange);
        }
        let mut fields = self.fields(target)?;
        if fields.len() != 1 || fields[0].starts_with(b"-") {
            return Err(NestingReason::DirectoryChange);
        }
        let joined = self.joined(&fields.remove(0));
        self.state.cwd = Some(joined);
        Ok(())
    }

    fn assign(&mut self, assignment: &Assignment) -> Result<(), NestingReason> {
        if assignment.name == "IFS" {
            return Err(NestingReason::Construct {
                construct: "an assignment to IFS",
            });
        }
        // An assignment's value is expanded without field splitting, which
        // is what a quoted word is.
        let value = self.value(&assignment.value)?;
        self.state.bind(&assignment.name, &value);
        Ok(())
    }

    /// One invocation whose command word is `right`, entered through
    /// `directory` if the line wrote a `cd` in front of it.
    fn invoke(
        &mut self,
        invocation: &SimpleCommand,
        directory: Option<Vec<u8>>,
        context: Context,
    ) -> Result<(), NestingReason> {
        if !invocation.redirections.is_empty() {
            return Err(NestingReason::Redirection);
        }
        if !invocation.assignments.is_empty() {
            return Err(NestingReason::Prefix);
        }
        let mut words = invocation.words.iter().collect::<Vec<_>>();
        let Some(first) = words.first().copied() else {
            return Err(NestingReason::DirectoryChange);
        };
        if is_literal(first, b"exec") {
            words.remove(0);
        }
        match self.make_prefix(&words)? {
            Some((make, used)) => {
                self.invoke_words(&make, words[used..].iter().copied(), directory, context)
            }
            None => Err(NestingReason::BesideAnotherCommand {
                command: String::from_utf8_lossy(&first.source).into_owned(),
            }),
        }
    }

    fn invoke_words<'w>(
        &mut self,
        make: &[u8],
        words: impl Iterator<Item = &'w Word>,
        directory: Option<Vec<u8>>,
        context: Context,
    ) -> Result<(), NestingReason> {
        self.admitted(context)?;
        // The value as Make expanded it, which is what the resolver, the census
        // and every reader of the line before this one were handed.
        let mut spelled = vec![Spelled::Verbatim(make.to_vec())];
        for word in words {
            spelled.extend(self.spell(word)?);
        }
        let directory = directory.or_else(|| self.state.cwd.clone());
        let mut command = Vec::new();
        if let Some(directory) = directory {
            command.extend_from_slice(b"cd ");
            command.extend(quoted_field(&directory));
            command.extend_from_slice(b" && ");
        }
        for (index, word) in spelled.iter().enumerate() {
            if index > 0 {
                command.push(b' ');
            }
            match word {
                Spelled::Field(field) => command.extend(quoted_field(field)),
                Spelled::Verbatim(source) => command.extend_from_slice(source),
            }
        }
        self.found.push(LiftedInvocation {
            command: Bytes::from(command),
            make: Bytes::copy_from_slice(make),
        });
        Ok(())
    }

    /// How one argument of an invocation is written into it.
    ///
    /// A word the shell would have expanded from what the line settled is
    /// written as the fields it settles to. A word holding a command
    /// substitution is handed on in the shell's own spelling for the
    /// resolver to run — which it does today for `$(MAKE) V=$(cmd)` — but
    /// only while the line has settled nothing, because the resolver's
    /// shell would not know a variable this line bound.
    fn spell(&self, word: &Word) -> Result<Vec<Spelled>, NestingReason> {
        let substitutes = word
            .pieces
            .iter()
            .any(|piece| matches!(piece, Piece::CommandSubstitution));
        if substitutes {
            if !self.state.bindings.is_empty() {
                return Err(NestingReason::CommandSubstitution);
            }
            if word.pieces.iter().any(|piece| {
                matches!(
                    piece,
                    Piece::Parameter { .. } | Piece::Expansion { .. } | Piece::Arithmetic
                )
            }) {
                return Err(NestingReason::CommandSubstitution);
            }
            return Ok(vec![Spelled::Verbatim(word.source.to_vec())]);
        }
        let resolved = word
            .pieces
            .iter()
            .any(|piece| matches!(piece, Piece::Parameter { .. }));
        if resolved {
            return Ok(self.fields(word)?.into_iter().map(Spelled::Field).collect());
        }
        // Nothing to settle: the word is text, and its own spelling is the
        // one the resolver, the census and the tests already read.
        self.fields(word)?;
        Ok(vec![Spelled::Verbatim(word.source.to_vec())])
    }

    /// The fields `word` expands to, when the line has settled everything
    /// that decides them.
    fn fields(&self, word: &Word) -> Result<Vec<Vec<u8>>, NestingReason> {
        let mut fields: Vec<Vec<u8>> = Vec::new();
        let mut current: Option<Vec<u8>> = None;
        let mut first = true;
        for piece in &word.pieces {
            match piece {
                Piece::Literal { bytes, quoted } => {
                    if !quoted && (globs(bytes) || (first && bytes.starts_with(b"~"))) {
                        return Err(NestingReason::Glob);
                    }
                    current
                        .get_or_insert_with(Vec::new)
                        .extend_from_slice(bytes);
                }
                Piece::Parameter { name, quoted } => {
                    let value = self.state.bound(name).ok_or_else(|| {
                        NestingReason::UnresolvedParameter {
                            name: String::from_utf8_lossy(name).into_owned(),
                        }
                    })?;
                    if *quoted {
                        current
                            .get_or_insert_with(Vec::new)
                            .extend_from_slice(value);
                    } else {
                        if globs(value) {
                            return Err(NestingReason::Glob);
                        }
                        // Field splitting on the default IFS: whitespace
                        // ends the field before it and starts none of its
                        // own.
                        let ends_open = value.last().is_some_and(|byte| is_blank(*byte));
                        let starts_open = value.first().is_some_and(|byte| is_blank(*byte));
                        let mut parts = value
                            .split(|byte| is_blank(*byte))
                            .filter(|p| !p.is_empty());
                        if starts_open && let Some(open) = current.take() {
                            fields.push(open);
                        }
                        if let Some(head) = parts.next() {
                            current.get_or_insert_with(Vec::new).extend_from_slice(head);
                            for part in parts {
                                if let Some(open) = current.take() {
                                    fields.push(open);
                                }
                                current = Some(part.to_vec());
                            }
                        }
                        if ends_open && let Some(open) = current.take() {
                            fields.push(open);
                        }
                    }
                }
                Piece::Expansion { .. } => {
                    return Err(NestingReason::UnresolvedParameter {
                        name: String::from_utf8_lossy(&word.source).into_owned(),
                    });
                }
                Piece::CommandSubstitution => return Err(NestingReason::CommandSubstitution),
                Piece::Arithmetic => {
                    return Err(NestingReason::Construct {
                        construct: "an arithmetic expansion",
                    });
                }
                _ => {
                    return Err(NestingReason::Construct {
                        construct: "a word form the compiler does not read",
                    });
                }
            }
            first = false;
        }
        if word.pieces.is_empty() {
            // `''` and `""`: one empty field.
            fields.push(Vec::new());
        }
        if let Some(open) = current {
            fields.push(open);
        }
        Ok(fields)
    }

    /// `word` expanded as an assignment's value is: settled, and never split.
    fn value(&self, word: &Word) -> Result<Vec<u8>, NestingReason> {
        let mut value = Vec::new();
        for piece in &word.pieces {
            match piece {
                Piece::Literal { bytes, .. } => value.extend_from_slice(bytes),
                Piece::Parameter { name, .. } => {
                    let bound = self.state.bound(name).ok_or_else(|| {
                        NestingReason::UnresolvedParameter {
                            name: String::from_utf8_lossy(name).into_owned(),
                        }
                    })?;
                    value.extend_from_slice(bound);
                }
                Piece::Expansion { .. } => {
                    return Err(NestingReason::UnresolvedParameter {
                        name: String::from_utf8_lossy(&word.source).into_owned(),
                    });
                }
                Piece::CommandSubstitution => return Err(NestingReason::CommandSubstitution),
                _ => {
                    return Err(NestingReason::Construct {
                        construct: "a word form the compiler does not read",
                    });
                }
            }
        }
        Ok(value)
    }

    /// Whether an `if`'s condition holds, when the line has settled
    /// everything it reads: `test` and `[` over strings the line knows,
    /// `true`, `false`, `:`, and those joined by `&&`, `||` and `!`.
    ///
    /// A file test reads the disk at run time, after the lines before this
    /// one have written to it, and refuses.
    fn decide(&self, condition: &Command) -> Result<bool, NestingReason> {
        match condition {
            Command::Simple(simple) => {
                if !simple.assignments.is_empty() || !simple.redirections.is_empty() {
                    return Err(NestingReason::UndecidableCondition);
                }
                let mut fields = Vec::new();
                for word in &simple.words {
                    fields.extend(self.fields(word).map_err(|reason| match reason {
                        NestingReason::UnresolvedParameter { .. } | NestingReason::Glob => {
                            NestingReason::UndecidableCondition
                        }
                        other => other,
                    })?);
                }
                match fields
                    .split_first()
                    .map(|(name, args)| (name.as_slice(), args))
                {
                    Some((b"true" | b":", _)) => Ok(true),
                    Some((b"false", _)) => Ok(false),
                    Some((b"test", args)) => test::evaluate(args),
                    Some((b"[", args)) => match args.split_last() {
                        Some((close, args)) if close == b"]" => test::evaluate(args),
                        _ => Err(NestingReason::UndecidableCondition),
                    },
                    _ => Err(NestingReason::UndecidableCondition),
                }
            }
            Command::And(left, right) => Ok(self.decide(left)? && self.decide(right)?),
            Command::Or(left, right) => Ok(self.decide(left)? || self.decide(right)?),
            Command::Pipeline { negated, commands } => match commands.as_slice() {
                [single] => Ok(self.decide(single)? != *negated),
                _ => Err(NestingReason::UndecidableCondition),
            },
            Command::Subshell(inner) => self.decide(inner),
            _ => Err(NestingReason::UndecidableCondition),
        }
    }
}

/// Whether `command` is an `exit` that fails the shell it ends: `exit` alone,
/// which leaves the failed status it follows, or `exit N` with a literal
/// non-zero N.
fn is_failing_exit(command: &Command) -> bool {
    let Command::Simple(simple) = command else {
        return false;
    };
    if !simple.assignments.is_empty() || !simple.redirections.is_empty() {
        return false;
    }
    let mut words = simple.words.iter();
    let Some(exit) = words.next() else {
        return false;
    };
    if !is_literal(exit, b"exit") {
        return false;
    }
    match (words.next(), words.next()) {
        (None, _) => true,
        (Some(status), None) => literal_text(status).is_some_and(|text| {
            !text.is_empty()
                && text.iter().all(u8::is_ascii_digit)
                && text.iter().any(|b| *b != b'0')
        }),
        _ => false,
    }
}

/// The word's text when it holds nothing but text.
fn literal_text(word: &Word) -> Option<Vec<u8>> {
    let mut text = Vec::new();
    for piece in &word.pieces {
        match piece {
            Piece::Literal { bytes, .. } => text.extend_from_slice(bytes),
            _ => return None,
        }
    }
    Some(text)
}

fn is_literal(word: &Word, text: &[u8]) -> bool {
    literal_text(word).is_some_and(|literal| literal == text)
}

/// Whether unquoted text would be matched against the disk.
///
/// A `[` opens a bracket expression only when a `]` closes it; on its own it
/// is the byte it is, which is how `[ -n "$V" ]` names the command it names.
fn globs(bytes: &[u8]) -> bool {
    bytes.iter().enumerate().any(|(at, byte)| match byte {
        b'*' | b'?' => true,
        b'[' => bytes[at + 1..].contains(&b']'),
        _ => false,
    })
}

const fn is_blank(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n')
}

/// A field as the resolver reads it back: bare where every byte is one the
/// resolver reads as itself, and single-quoted otherwise.
fn quoted_field(field: &[u8]) -> Vec<u8> {
    let bare = !field.is_empty()
        && field.iter().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'_' | b'-' | b'.' | b'/' | b'=' | b':' | b',' | b'+' | b'@' | b'%'
                )
        });
    if bare {
        return field.to_vec();
    }
    let mut quoted = vec![b'\''];
    for &byte in field {
        if byte == b'\'' {
            quoted.extend_from_slice(b"'\\''");
        } else {
            quoted.push(byte);
        }
    }
    quoted.push(b'\'');
    quoted
}

/// `test` over strings the line has settled.
mod test {
    use crate::census::NestingReason;

    /// The POSIX `test` grammar over literal operands: `-n`, `-z`, `=`,
    /// `!=`, the six integer comparisons, `!`, `-a`, `-o` and parentheses.
    /// Every other primary reads something outside the line — a file, a
    /// descriptor — and refuses.
    pub(super) fn evaluate(args: &[Vec<u8>]) -> Result<bool, NestingReason> {
        let mut parser = Parser { args, at: 0 };
        let value = parser.or()?;
        if parser.at != args.len() {
            return Err(NestingReason::UndecidableCondition);
        }
        Ok(value)
    }

    struct Parser<'a> {
        args: &'a [Vec<u8>],
        at: usize,
    }

    impl Parser<'_> {
        fn peek(&self) -> Option<&[u8]> {
            self.args.get(self.at).map(Vec::as_slice)
        }

        fn next(&mut self) -> Option<&[u8]> {
            let arg = self.args.get(self.at).map(Vec::as_slice);
            self.at += 1;
            arg
        }

        fn or(&mut self) -> Result<bool, NestingReason> {
            let mut value = self.and()?;
            while self.peek() == Some(b"-o") {
                self.at += 1;
                value |= self.and()?;
            }
            Ok(value)
        }

        fn and(&mut self) -> Result<bool, NestingReason> {
            let mut value = self.primary()?;
            while self.peek() == Some(b"-a") {
                self.at += 1;
                value &= self.primary()?;
            }
            Ok(value)
        }

        fn primary(&mut self) -> Result<bool, NestingReason> {
            let remaining = self.args.len() - self.at;
            match self.peek() {
                None => Ok(false),
                Some(b"!") => {
                    self.at += 1;
                    Ok(!self.primary()?)
                }
                Some(b"(") => {
                    self.at += 1;
                    let value = self.or()?;
                    if self.next() != Some(b")") {
                        return Err(NestingReason::UndecidableCondition);
                    }
                    Ok(value)
                }
                Some(b"-n") if remaining >= 2 => {
                    self.at += 1;
                    Ok(!self.next().unwrap_or_default().is_empty())
                }
                Some(b"-z") if remaining >= 2 => {
                    self.at += 1;
                    Ok(self.next().unwrap_or_default().is_empty())
                }
                Some(unary) if unary.len() == 2 && unary[0] == b'-' && remaining >= 2 => {
                    Err(NestingReason::UndecidableCondition)
                }
                Some(_) => {
                    let left = self.next().unwrap_or_default().to_vec();
                    let binary = self
                        .peek()
                        .filter(|operator| is_binary_operator(operator))
                        .map(<[u8]>::to_vec);
                    let Some(operator) = binary else {
                        return Ok(!left.is_empty());
                    };
                    self.at += 1;
                    let Some(right) = self.next() else {
                        return Err(NestingReason::UndecidableCondition);
                    };
                    match operator.as_slice() {
                        b"=" => Ok(left == right),
                        b"!=" => Ok(left != right),
                        _ => {
                            let (Some(left), Some(right)) = (integer(&left), integer(right)) else {
                                return Err(NestingReason::UndecidableCondition);
                            };
                            Ok(match operator.as_slice() {
                                b"-eq" => left == right,
                                b"-ne" => left != right,
                                b"-lt" => left < right,
                                b"-le" => left <= right,
                                b"-gt" => left > right,
                                _ => left >= right,
                            })
                        }
                    }
                }
            }
        }
    }

    fn is_binary_operator(operator: &[u8]) -> bool {
        matches!(
            operator,
            b"=" | b"!=" | b"-eq" | b"-ne" | b"-lt" | b"-le" | b"-gt" | b"-ge"
        )
    }

    fn integer(text: &[u8]) -> Option<i64> {
        std::str::from_utf8(text).ok()?.trim().parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifted(line: &'static [u8]) -> Result<Vec<String>, NestingReason> {
        lift(
            &Bytes::from_static(line),
            &[Bytes::from_static(b"make")],
            false,
        )
        .map(|invocations| {
            invocations
                .into_iter()
                .map(|invocation| String::from_utf8_lossy(&invocation.command).into_owned())
                .collect()
        })
    }

    fn lifted_strict(line: &'static [u8]) -> Result<Vec<String>, NestingReason> {
        lift(
            &Bytes::from_static(line),
            &[Bytes::from_static(b"make")],
            true,
        )
        .map(|invocations| {
            invocations
                .into_iter()
                .map(|invocation| String::from_utf8_lossy(&invocation.command).into_owned())
                .collect()
        })
    }

    /// The shapes lifted before the line was read as shell, spelled as they
    /// were: a line that is an invocation, one entered through `cd`, one
    /// inside a wrapper, and `&&`-chains of those.
    #[test]
    fn what_lifted_before_still_lifts_and_spells_the_same() {
        assert_eq!(lifted(b"make -C sub all").unwrap(), ["make -C sub all"]);
        assert_eq!(
            lifted(b"cd sub && make child").unwrap(),
            ["cd sub && make child"]
        );
        assert_eq!(
            lifted(b"(cd sub && make child)").unwrap(),
            ["cd sub && make child"]
        );
        assert_eq!(
            lifted(b"{ cd sub && make child; }").unwrap(),
            ["cd sub && make child"]
        );
        assert_eq!(
            lifted(b"make -C a && make -C b").unwrap(),
            ["make -C a", "make -C b"]
        );
        assert_eq!(
            lifted(b"(cd a && make) && { cd b && make; }").unwrap(),
            ["cd a && make", "cd b && make"]
        );
        assert_eq!(lifted(b"exec make all").unwrap(), ["make all"]);
        assert_eq!(
            lifted(b"make CFLAGS='-O2 -g' V=$(cat v)").unwrap(),
            ["make CFLAGS='-O2 -g' V=$(cat v)"]
        );
        // A line that mentions Make without starting one is refused like
        // any other line that is not a list of invocations; the caller asks
        // separately whether it starts a Make before reading that as
        // nesting. A line made of nothing the walk has to run lifts nothing.
        assert_eq!(
            lifted(b"echo \"run make install\""),
            Err(NestingReason::BesideAnotherCommand {
                command: "echo".to_owned()
            })
        );
        assert_eq!(lifted(b"x=1").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn what_refused_before_still_refuses_and_says_why() {
        assert_eq!(
            lifted(b"make -C a ; make -C b"),
            Err(NestingReason::Sequence)
        );
        assert_eq!(
            lifted(b"cd a && make && echo done"),
            Err(NestingReason::BesideAnotherCommand {
                command: "echo".to_owned()
            })
        );
        assert_eq!(lifted(b"make -C a > log"), Err(NestingReason::Redirection));
        assert_eq!(
            lifted(b"echo make && make -C b"),
            Err(NestingReason::BesideAnotherCommand {
                command: "echo".to_owned()
            })
        );
        assert_eq!(lifted(b"V=1 make child"), Err(NestingReason::Prefix));
        assert_eq!(lifted(b"env V=1 make child"), Err(NestingReason::Prefix));
        assert_eq!(lifted(b"make child *.o"), Err(NestingReason::Glob));
        assert_eq!(lifted(b"make | tee log"), Err(NestingReason::Pipeline));
        assert_eq!(
            lifted(b"true || make fallback"),
            Err(NestingReason::Alternation)
        );
        assert_eq!(
            lifted(b"test -d sub && make -C sub"),
            Err(NestingReason::BesideAnotherCommand {
                command: "test".to_owned()
            })
        );
        assert_eq!(
            lifted(b"if test -d sub; then make -C sub; fi"),
            Err(NestingReason::UndecidableCondition)
        );
        assert_eq!(
            lifted(b"make -C $dir"),
            Err(NestingReason::UnresolvedParameter {
                name: "dir".to_owned()
            })
        );
        // A substitution in the invocation's own arguments passes through
        // for the resolver to run, as `$(pwd)` always did; a backquoted one
        // is the same substitution and is now spelled the same way.
        assert_eq!(lifted(b"make -C `pwd`").unwrap(), ["make -C $(pwd)"]);
    }

    /// `MAKE` is Make's text and may be several shell words: kati hands a
    /// child `make -j8`, and a Makefile may set `MAKE = $(MAKE_COMMAND)
    /// --no-print-directory`. The invocation begins after the value's words,
    /// and the value goes on as the text it is.
    #[test]
    fn a_make_value_of_several_words_is_matched_word_by_word() {
        let lifted = |line: &'static [u8]| {
            lift(
                &Bytes::from_static(line),
                &[Bytes::from_static(b"make -j8")],
                false,
            )
            .map(|invocations| {
                invocations
                    .into_iter()
                    .map(|invocation| {
                        (
                            String::from_utf8_lossy(&invocation.command).into_owned(),
                            String::from_utf8_lossy(&invocation.make).into_owned(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(
            lifted(b"make -j8 --no-print-directory -C sub0 all").unwrap(),
            [(
                "make -j8 --no-print-directory -C sub0 all".to_owned(),
                "make -j8".to_owned()
            )]
        );
        assert_eq!(
            lifted(b"cd a && exec make -j8 x").unwrap(),
            [("cd a && make -j8 x".to_owned(), "make -j8".to_owned())]
        );
        assert_eq!(
            lifted(b"for d in a b; do (cd $d && make -j8) || exit 1; done").unwrap(),
            [
                ("cd a && make -j8".to_owned(), "make -j8".to_owned()),
                ("cd b && make -j8".to_owned(), "make -j8".to_owned())
            ]
        );
        // `make` alone is not the value `make -j8`.
        assert_eq!(
            lifted(b"make -C sub"),
            Err(NestingReason::BesideAnotherCommand {
                command: "make".to_owned()
            })
        );
    }

    /// zsh's shape, from all four of its Makefiles.
    #[test]
    fn a_loop_over_literal_words_that_exits_on_failure_unrolls() {
        assert_eq!(
            lifted(
                b"for subdir in Src Doc; do (cd $subdir && make prefix=/usr all) || exit 1; done"
            )
            .unwrap(),
            [
                "cd Src && make prefix=/usr all",
                "cd Doc && make prefix=/usr all"
            ]
        );
        // The same loop with its word list settled by an assignment earlier
        // on the line, which Src/Makefile writes as `subdirs='$(SUBDIRS)'`.
        assert_eq!(
            lifted(b"subdirs='Builtins Modules Zle'; for subdir in $subdirs; do (cd $subdir && make all) || exit 1; done").unwrap(),
            ["cd Builtins && make all", "cd Modules && make all", "cd Zle && make all"]
        );
        // A continued line, as a Makefile writes it.
        assert_eq!(
            lifted(b"for subdir in Src Doc; do \\\n  (cd $subdir && make $@) || exit 1; \\\ndone")
                .map(|_| ()),
            Err(NestingReason::UnresolvedParameter {
                name: "@".to_owned()
            })
        );
    }

    /// The kernel's `__build_one_by_one`: `set -e` on the line arms errexit
    /// for the loop's body, so a failed iteration ends the line.
    #[test]
    fn errexit_lets_a_bare_loop_unroll() {
        assert_eq!(
            lifted(b"set -e; for i in a b; do make -f /src/Makefile $i; done").unwrap(),
            ["make -f /src/Makefile a", "make -f /src/Makefile b"]
        );
        assert_eq!(
            lifted_strict(b"for i in a b; do make $i; done").unwrap(),
            ["make a", "make b"]
        );
        assert_eq!(
            lifted_strict(b"make a; make b").unwrap(),
            ["make a", "make b"]
        );
        // And without it the loop carries on past a failed iteration, which
        // a composed child cannot.
        assert_eq!(
            lifted(b"for i in a b; do make $i; done"),
            Err(NestingReason::LoopCarriesOn)
        );
        // `set +e` takes it away again.
        assert_eq!(
            lifted_strict(b"set +e; for i in a b; do make $i; done"),
            Err(NestingReason::LoopCarriesOn)
        );
    }

    /// POSIX ignores `-e` for every command of an AND-OR list but the last,
    /// so `make a && make b` in a loop body still carries on past a failed
    /// `make a`, and so does a subshell standing in that position.
    #[test]
    fn errexit_is_not_live_where_posix_exempts_it() {
        assert_eq!(
            lifted_strict(b"for i in a b; do make $i && make $i-post; done"),
            Err(NestingReason::LoopCarriesOn)
        );
        // But `|| exit 1` after the whole AND-list catches either failure.
        assert_eq!(
            lifted(b"for i in a b; do (make $i && make $i-post) || exit 1; done").unwrap(),
            ["make a", "make a-post", "make b", "make b-post"]
        );
    }

    /// `exit` ends the shell it runs in. Inside a subshell it ends the
    /// subshell, and the loop around it carries on.
    #[test]
    fn an_exit_inside_a_subshell_does_not_end_the_line() {
        assert_eq!(
            lifted(b"for d in a b; do (cd $d && make || exit 1); done"),
            Err(NestingReason::LoopCarriesOn)
        );
        assert_eq!(lifted(b"(make a || exit 1)").unwrap(), ["make a"]);
        // `exit 0` after a failure is a line that succeeds.
        assert_eq!(lifted(b"make a || exit 0"), Err(NestingReason::Alternation));
        assert_eq!(lifted(b"make a || exit").unwrap(), ["make a"]);
        assert_eq!(lifted(b"make a || true"), Err(NestingReason::Alternation));
    }

    /// The subshell keeps a `cd` and an assignment to itself; a `cd` of the
    /// line's own moves everything after it.
    #[test]
    fn a_subshell_keeps_its_state_and_a_bare_cd_moves_the_line() {
        assert_eq!(
            lifted(b"(cd a && make) && make b").unwrap(),
            ["cd a && make", "make b"]
        );
        assert_eq!(
            lifted(b"cd a && make x && make y").unwrap(),
            ["cd a && make x", "cd a && make y"]
        );
        assert_eq!(
            lifted(b"set -e; cd a; make x; cd ../b; make y").unwrap(),
            ["cd a && make x", "cd a/../b && make y"]
        );
        assert_eq!(lifted(b"cd /abs && make").unwrap(), ["cd /abs && make"]);
        assert_eq!(lifted(b"cd sub; make child"), Err(NestingReason::Sequence));
        assert_eq!(lifted(b"cd && make"), Err(NestingReason::DirectoryChange));
        assert_eq!(
            lifted(b"cd -P a && make"),
            Err(NestingReason::DirectoryChange)
        );
    }

    /// vim's guard, and the shapes a guard can take.
    #[test]
    fn a_condition_over_settled_strings_is_decided() {
        assert_eq!(
            lifted(
                b"if test \"all\" = \"test\" -o \"all\" = \"testtiny\"; then make indenttest; fi"
            )
            .unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            lifted(b"if test \"test\" = \"test\"; then make indenttest; fi").unwrap(),
            ["make indenttest"]
        );
        assert_eq!(
            lifted(b"V=x; if test -n \"$V\"; then make with; else make without; fi").unwrap(),
            ["make with"]
        );
        assert_eq!(
            lifted(b"V=; if [ -n \"$V\" ]; then make with; else make without; fi").unwrap(),
            ["make without"]
        );
        assert_eq!(
            lifted(b"if [ 3 -gt 2 ] && ! false; then make; fi").unwrap(),
            ["make"]
        );
        assert_eq!(lifted(b"if true; then make; fi").unwrap(), ["make"]);
        assert_eq!(
            lifted(b"if false; then make; fi").unwrap(),
            Vec::<String>::new()
        );
        // What the disk answers is not decided here.
        assert_eq!(
            lifted(b"if [ -f Makefile ]; then make; fi"),
            Err(NestingReason::UndecidableCondition)
        );
        assert_eq!(
            lifted(b"if test -n \"$UNSET\"; then make; fi"),
            Err(NestingReason::UndecidableCondition)
        );
        assert_eq!(
            lifted(b"if grep -q x y; then make; fi"),
            Err(NestingReason::UndecidableCondition)
        );
    }

    /// The hard boundary: a value computed by a command is known only by
    /// running it.
    #[test]
    fn a_command_substitution_refuses_wherever_it_decides_anything() {
        assert_eq!(
            lifted(b"target=`echo all | sed s/-recursive//`; for d in a; do (cd $d && make $target) || exit 1; done"),
            Err(NestingReason::CommandSubstitution)
        );
        assert_eq!(
            lifted(b"for d in $(ls); do (cd $d && make) || exit 1; done"),
            Err(NestingReason::CommandSubstitution)
        );
        // In an argument it passes through as it always did — but not
        // beside a variable the resolver's shell would not know.
        assert_eq!(
            lifted(b"x=1; make V=$(cat v)"),
            Err(NestingReason::CommandSubstitution)
        );
    }

    #[test]
    fn words_split_and_quote_as_the_shell_would() {
        assert_eq!(
            lifted(b"goals=' a  b '; make $goals").unwrap(),
            ["make a b"]
        );
        assert_eq!(
            lifted(b"goals='a b'; make \"$goals\"").unwrap(),
            ["make 'a b'"]
        );
        assert_eq!(lifted(b"g='a b'; make x$g").unwrap(), ["make xa b"]);
        assert_eq!(lifted(b"g=; make $g all").unwrap(), ["make all"]);
        assert_eq!(lifted(b"g=; make \"$g\" all").unwrap(), ["make '' all"]);
        assert_eq!(lifted(b"g='*.c'; make $g"), Err(NestingReason::Glob));
        assert_eq!(lifted(b"g='*.c'; make \"$g\"").unwrap(), ["make '*.c'"]);
        assert_eq!(lifted(b"make ~/x"), Err(NestingReason::Glob));
        assert_eq!(
            lifted(b"make ${g:-x}"),
            Err(NestingReason::UnresolvedParameter {
                name: "${g:-x}".to_owned()
            })
        );
        assert_eq!(
            lifted(b"IFS=,; make a,b"),
            Err(NestingReason::Construct {
                construct: "an assignment to IFS"
            })
        );
    }

    #[test]
    fn the_rest_of_the_grammar_refuses_by_name() {
        assert_eq!(
            lifted(b"make &"),
            Err(NestingReason::Construct {
                construct: "a command started in the background"
            })
        );
        assert_eq!(
            lifted(b"while true; do make; done"),
            Err(NestingReason::Construct {
                construct: "a `while` loop"
            })
        );
        assert_eq!(
            lifted(b"case x in x) make;; esac"),
            Err(NestingReason::Construct {
                construct: "a `case`"
            })
        );
        assert_eq!(lifted(b"{ make; } 2>&1"), Err(NestingReason::Redirection));
        assert_eq!(lifted(b"! make"), Err(NestingReason::Pipeline));
        assert_eq!(
            lifted(b"for d in a b; do make"),
            Err(NestingReason::Unreadable)
        );
        assert_eq!(lifted(b"make; exit 1"), Err(NestingReason::Sequence));
        assert_eq!(
            lifted_strict(b"make; exit 1"),
            Err(NestingReason::Construct {
                construct: "an `exit` that ends the line"
            })
        );
    }

    #[test]
    fn a_line_of_several_lines_admits_only_what_each_line_ends() {
        // `.ONESHELL` hands the shell the whole recipe; a newline is `;`.
        assert_eq!(lifted(b"make a\nmake b"), Err(NestingReason::Sequence));
        assert_eq!(
            lifted_strict(b"make a\nmake b").unwrap(),
            ["make a", "make b"]
        );
        assert_eq!(lifted(b"x=a\nmake $x").unwrap(), ["make a"]);
    }
}
