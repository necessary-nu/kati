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

//! A command line with no shell syntax in it, taken apart the way GNU Make
//! takes one apart.
//!
//! `construct_command_argv_internal` (reference/gnumake/src/job.c) tokenizes a
//! command itself and execs it directly whenever nothing in the line needs a
//! shell. Everything else reaches `goto slow` and is handed to `$(SHELL)`. The
//! difference is visible from a Makefile: a program that is not there is
//! reported by whoever went looking for it, so
//!
//! ```text
//! all: ; ./nosuchprog arg
//! ```
//!
//! gets `make: ./nosuchprog: No such file or directory` and status 127, while
//!
//! ```text
//! all: ; ./nosuchprog > out
//! ```
//!
//! gets `/bin/sh: 1: ./nosuchprog: not found` and the same status — the `>`
//! made it the shell's errand. Both agree on the status either way; what the
//! fast path decides is who reports, how many processes there are, and which
//! quoting the arguments have already lost.
//!
//! The four gates before any tokenizing, in GNU Make's order, are all here:
//! the shell must be the built-in default, `.SHELLFLAGS` must be exactly `-c`
//! or `-ec`, `IFS` must be nothing but whitespace, and `.ONESHELL` must not be
//! turning newlines into separators. A `SHELL` a Makefile set — even to
//! another POSIX shell — takes the slow path, because GNU Make compares it
//! against `default_shell` and nothing else.

use bytes::{BufMut, Bytes, BytesMut};

/// The shell GNU Make was compiled with, which is the only one the fast path
/// is willing to stand in for.
pub const DEFAULT_SHELL: &[u8] = b"/bin/sh";

/// The process that runs the shell spelled `named`.
///
/// Where `named` is the default shell and `stand_in` names a program, the
/// stand-in is what runs and `named` is its `argv[0]`. dash prefixes its
/// diagnostics with `argv[0]` exactly as written, so carrying the spelling
/// through is what makes a substituted shell say what the shell it replaced
/// would have said. Anything else — a `SHELL` the Makefile set, the `/bin/sh`
/// wrapping a complicated one — is spawned as named.
pub fn shell_process(
    named: &std::ffi::OsStr,
    stand_in: Option<&std::path::Path>,
) -> std::process::Command {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::process::CommandExt;

    if named.as_bytes() == DEFAULT_SHELL
        && let Some(stand_in) = stand_in
    {
        let mut command = std::process::Command::new(stand_in);
        command.arg0(named);
        return command;
    }
    std::process::Command::new(named)
}

/// Characters that mean something to a shell, so a line holding one of them is
/// the shell's errand. GNU Make's `sh_chars` for a Unix-ish host.
///
/// Note what is *not* here: a single quote, which the tokenizer below handles
/// itself, and a backslash, which it also handles. A double quote is here, so
/// `prog "a b"` goes to the shell while `prog 'a b'` does not.
const SHELL_CHARACTERS: &[u8] = b"#;\"*?[]&|<>(){}$`^~!";

/// Words that are a shell builtin when they lead a line, so the shell has to
/// be the one to run them. GNU Make's `sh_cmds`, in its order.
const SHELL_BUILTINS: &[&[u8]] = &[
    b".",
    b":",
    b"alias",
    b"bg",
    b"break",
    b"case",
    b"cd",
    b"command",
    b"continue",
    b"eval",
    b"exec",
    b"exit",
    b"export",
    b"fc",
    b"fg",
    b"for",
    b"getopts",
    b"hash",
    b"if",
    b"jobs",
    b"login",
    b"logout",
    b"read",
    b"readonly",
    b"return",
    b"set",
    b"shift",
    b"test",
    b"times",
    b"trap",
    b"type",
    b"ulimit",
    b"umask",
    b"unalias",
    b"unset",
    b"wait",
    b"while",
];

/// What the shell would have been asked to do, when it need not be asked.
///
/// `None` is GNU Make's `goto slow`: hand the whole line to `$(SHELL)`.
/// `Some` is the argument list to exec, already unquoted — the caller runs it
/// with no shell in between and reports a failure to start it itself.
pub fn direct_argv(
    line: &[u8],
    shell: &[u8],
    shell_flags: &[u8],
    one_shell: bool,
) -> Option<Vec<Bytes>> {
    if shell != DEFAULT_SHELL {
        return None;
    }
    if shell_flags != b"-c" && shell_flags != b"-ec" {
        return None;
    }
    tokenize(line, one_shell)
}

/// `.SHELLFLAGS` as the words a `.ONESHELL` launch passes.
///
/// GNU Make's one-shell branch parses them with `construct_command_argv_internal`
/// itself — "Parse shellflags using construct_command_argv_internal to handle
/// quotes" (job.c) — so `-E 'use warnings FATAL => "all";'` is two words and
/// not five, and the quotes come off. It passes no shell and no flags of its
/// own, so none of the fast path's gates apply, and it guards the result with
/// `if (argv)`, which is only ever false for flags that are blank.
///
/// Only the one-shell branch does this. Every other recipe builds `$(SHELL)
/// $(.SHELLFLAGS) LINE` as text and re-tokenizes the whole of it — see
/// [`command_line_flag_argv`].
///
/// WHERE THE TOKENIZER REFUSES, the flags do not go to a command line: they
/// come back as THREE WORDS. `one_shell` is a global, so the recursion this
/// branch makes lands in this very branch again, one level down, with the
/// flags text as its recipe — and there the shell is the default `/bin/sh`,
/// which is Bourne-compatible, so the text loses the blanks and `[@+-]` at the
/// front of each of its lines and the launch built out of it is `[/bin/sh,
/// <the flags GNU defaults to>, <that text>]`. The `if (argv)` guard sees
/// those three and copies them in ahead of the script. Measured on 4.4.1:
/// `.SHELLFLAGS := "-e" -c` reaches an `argv`-printing `SHELL` as `/bin/sh`,
/// `-c`, `"-e" -c`; `-e $X -c` reaches it as `/bin/sh`, `-c`, `e $X -c`, the
/// leading `-` gone; `-e\n-c` as `/bin/sh`, `-c`, `e\nc`, both of them gone.
///
/// `default_flags` is what that recursion defaults its own `shellflags` to,
/// having been handed none — [`crate::eval::Evaluator::default_shell_flag`].
///
/// 4.4.1 REACHES THIS ANSWER THROUGH A HEAP OVERFLOW, which is why the band it
/// can be asked about has a floor. The one-shell branch sizes its buffer for
/// the flags it expects to split — `shell_len + sflags_len + line_len + 3`,
/// which is exact for words that came out of the flags text — and the three
/// words above are `/bin/sh` and `-c` longer than that. Valgrind names the
/// `stpcpy` at job.c:3452 writing past the block for every input on this path;
/// short ones die of it (`.SHELLFLAGS := set` and `:= exec` both segfault
/// 4.4.1), longer ones survive with the argv the source says they should have.
/// This implements what the source says, which is the only thing there is to
/// implement: the overflow is 4.4.1's bug and not a behaviour to reproduce.
pub fn shell_flag_argv(shell_flags: &[u8], default_flags: &[u8]) -> Vec<Bytes> {
    // GNU Make's own first test, and the whole of what its `if (argv)` guard
    // is ever false for: `while (ISBLANK (*line)) ++line; if (*line == '\0')
    // return 0;`. A newline is not a blank, so a flags value of one is a
    // script of one empty line rather than nothing to run.
    if shell_flags.iter().all(|byte| matches!(byte, b' ' | b'\t')) {
        return Vec::new();
    }
    match tokenize(shell_flags, true) {
        Some(words) => words,
        None => vec![
            Bytes::from_static(DEFAULT_SHELL),
            Bytes::copy_from_slice(default_flags),
            one_shell_prefixes_stripped(shell_flags),
        ],
    }
}

/// A one-shell script with the blanks and `[@+-]` taken off the front of each
/// of its lines.
///
/// What GNU Make's one-shell branch does to the text it is about to hand a
/// Bourne-compatible shell (job.c), and its comment says why only that shell:
/// a `SHELL` Make does not recognise may be reading those characters as its
/// own script. A line ends at a newline the text did not escape, which is why
/// this counts backslashes rather than splitting on `\n`.
///
/// The caller answers the shell question. Two reach here: the refused-flags
/// fallback above, where the shell of the recursion is the default `/bin/sh`
/// and therefore always compatible, and [`crate::fileutil`]'s assembly of a
/// one-script launch, which asks
/// [`is_bourne_compatible_shell`](crate::command::is_bourne_compatible_shell)
/// about the `SHELL` this launch actually names.
pub(crate) fn one_shell_prefixes_stripped(text: &[u8]) -> Bytes {
    let mut stripped = BytesMut::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let start = rest
            .iter()
            .position(|byte| !matches!(byte, b' ' | b'\t' | b'-' | b'@' | b'+'))
            .unwrap_or(rest.len());
        rest = &rest[start..];
        let mut escaped = false;
        let mut index = 0;
        while index < rest.len() {
            let byte = rest[index];
            stripped.put_u8(byte);
            index += 1;
            if byte == b'\\' {
                escaped = !escaped;
            } else {
                if byte == b'\n' && !escaped {
                    break;
                }
                escaped = false;
            }
        }
        rest = &rest[index..];
    }
    stripped.freeze()
}

/// `.SHELLFLAGS` as the words every other launch passes ahead of the command.
///
/// GNU Make builds the text `$(SHELL) $(.SHELLFLAGS) LINE` — the shell and the
/// line escaped, the flags copied in as they stand — and re-tokenizes all of
/// it (job.c), so the flags are split and quoted exactly like words on a
/// command line while the line comes back as the single argument it was
/// escaped into. Tokenizing the flags alone reaches the same words: the
/// tokenizer's leading-word rules are about the first word of a command line,
/// and there the shell is already ahead of them.
///
/// `None` is that tokenizer's `goto slow`: the flags have shell syntax in them
/// — or trip a leading-word rule that the shell ahead of them would have
/// answered — so the caller hands the whole command line to a shell and lets
/// that shell split them, which is what GNU Make's own slow path does with it.
///
/// Blank flags are `Some` of no words at all rather than one empty word: the
/// text has a run of blanks where they were, and the tokenizer skips a run of
/// blanks. `.SHELLFLAGS :=` therefore starts the shell with the script as its
/// first argument — a file to read, which is what GNU Make 4.4.1 does with it.
pub fn command_line_flag_argv(shell_flags: &[u8]) -> Option<Vec<Bytes>> {
    if shell_flags
        .iter()
        .all(|byte| matches!(byte, b' ' | b'\t' | b'\n'))
    {
        return Some(Vec::new());
    }
    tokenize(shell_flags, false)
}

/// The shell's own word splitting, with no question asked about what the words
/// turn out to be. GNU Make's `construct_command_argv_internal` past its gates.
fn tokenize(line: &[u8], one_shell: bool) -> Option<Vec<Bytes>> {
    let line = Bytes::copy_from_slice(line);
    let mut argv: Vec<Bytes> = Vec::new();
    let mut word = Vec::new();
    let mut instring: Option<u8> = None;
    // "Equals is a special character in leading words before the first word
    // with no equals sign in it" — a `VAR=value prog` prefix is the shell's.
    let mut word_has_equals = false;
    let mut seen_nonequals = false;
    let mut last_argument_was_empty = false;

    let bytes = line.as_ref();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(quote) = instring {
            if byte == quote {
                instring = None;
                if word.is_empty() {
                    last_argument_was_empty = true;
                }
            } else if byte == b'\\' && bytes.get(index + 1) == Some(&b'\n') {
                // Kept inside single quotes, dropped inside double ones — and
                // a double-quoted string never gets here, because `"` is a
                // shell character.
                word.push(byte);
                index += 1;
                word.push(b'\n');
            } else {
                word.push(byte);
            }
            index += 1;
            continue;
        }
        if SHELL_CHARACTERS.contains(&byte) {
            return None;
        }
        if one_shell && byte == b'\n' {
            // `.ONESHELL` makes a newline a separator like `;`.
            return None;
        }
        match byte {
            b'=' => {
                if !seen_nonequals {
                    return None;
                }
                word_has_equals = true;
                word.push(b'=');
                index += 1;
            }
            b'\\' => {
                match bytes.get(index + 1) {
                    Some(b'\n') => {
                        index += 2;
                        // At the start of an argument, skip the blanks before
                        // the next word.
                        if word.is_empty() {
                            while matches!(bytes.get(index), Some(b' ' | b'\t')) {
                                index += 1;
                            }
                        }
                    }
                    Some(next) => {
                        word.push(*next);
                        index += 2;
                    }
                    None => index += 1,
                }
            }
            b'\'' => {
                instring = Some(byte);
                index += 1;
            }
            b'\n' | b' ' | b'\t' => {
                seen_nonequals |= !word_has_equals;
                if word_has_equals && !seen_nonequals {
                    return None;
                }
                word_has_equals = false;
                argv.push(Bytes::from(std::mem::take(&mut word)));
                last_argument_was_empty = false;
                if argv.len() == 1 && is_shell_builtin(&argv[0]) {
                    return None;
                }
                index += 1;
                while matches!(bytes.get(index), Some(b' ' | b'\t')) {
                    index += 1;
                }
            }
            _ => {
                word.push(byte);
                index += 1;
            }
        }
    }
    if instring.is_some() {
        // "Let the shell deal with an unterminated quote."
        return None;
    }
    if !word.is_empty() || last_argument_was_empty {
        argv.push(Bytes::from(word));
    }
    let first = argv.first()?;
    // The builtin test again, for a line that is one word with no trailing
    // whitespace to have triggered the check above.
    if argv.len() == 1 && is_shell_builtin(first) {
        return None;
    }
    (!first.is_empty()).then_some(argv)
}

fn is_shell_builtin(word: &[u8]) -> bool {
    SHELL_BUILTINS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(line: &str) -> Option<Vec<String>> {
        direct_argv(line.as_bytes(), DEFAULT_SHELL, b"-c", false).map(|words| {
            words
                .into_iter()
                .map(|word| String::from_utf8(word.to_vec()).unwrap())
                .collect()
        })
    }

    fn flag_words(flags: &str) -> Vec<String> {
        words_of(shell_flag_argv(flags.as_bytes(), b"-c"))
    }

    /// The same, for a recipe read under `.POSIX:` whose first line carries no
    /// `-`, which is the whole of what changes the flags GNU Make defaults to.
    fn posix_flag_words(flags: &str) -> Vec<String> {
        words_of(shell_flag_argv(flags.as_bytes(), b"-ec"))
    }

    fn words_of(argv: Vec<Bytes>) -> Vec<String> {
        argv.into_iter()
            .map(|word| String::from_utf8(word.to_vec()).unwrap())
            .collect()
    }

    #[test]
    fn one_shell_flags_are_read_with_the_shells_own_quoting() {
        assert_eq!(flag_words("-e"), vec!["-e".to_owned()]);
        assert_eq!(flag_words("-e -c"), vec!["-e".to_owned(), "-c".to_owned()]);
        // The upstream case: a quoted flag with blanks, a `;`, a `>` and a
        // pair of double quotes in it is ONE word, and the quotes come off.
        assert_eq!(
            flag_words("-w -E 'use warnings FATAL => \"all\";' -E"),
            vec![
                "-w".to_owned(),
                "-E".to_owned(),
                "use warnings FATAL => \"all\";".to_owned(),
                "-E".to_owned(),
            ]
        );
        // No flags at all contribute nothing: GNU Make's `if (argv)` guard,
        // which is false for blanks and for nothing else.
        assert!(flag_words("").is_empty());
        assert!(flag_words("  \t ").is_empty());
    }

    /// What a `.SHELLFLAGS` the tokenizer refuses reaches the launch as.
    ///
    /// Three words, because GNU Make's `one_shell` is a global and the
    /// recursion that parses the flags therefore lands in the one-shell branch
    /// itself: `/bin/sh`, the flags that branch defaults to, and the flags text
    /// with the prefixes stripped off the front of each of its lines.
    #[test]
    fn one_shell_flags_a_shell_would_split_are_three_words() {
        // Measured on 4.4.1 with an argv-printing `SHELL`. A double quote is a
        // shell character, and nothing about this text is a prefix, so it comes
        // back whole.
        assert_eq!(
            flag_words("\"-e\" -c"),
            vec![
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "\"-e\" -c".to_owned()
            ]
        );
        // The prefix stripping, which is the part that is not a round trip.
        // `-e $X -c` reaches 4.4.1's launch with its leading `-` gone.
        assert_eq!(
            flag_words("-e $X -c"),
            vec!["/bin/sh".to_owned(), "-c".to_owned(), "e $X -c".to_owned()]
        );
        // A run of them, blanks between, all gone; and only at the front.
        assert_eq!(
            flag_words(" +- e \"x\" -c"),
            vec![
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "e \"x\" -c".to_owned()
            ]
        );
        assert_eq!(
            flag_words("@e \"x\" -c"),
            vec![
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "e \"x\" -c".to_owned()
            ]
        );
        // A newline is what sends this text slow in the first place — under
        // `.ONESHELL` it separates commands — and it is also what starts
        // another line for the stripping to reach.
        assert_eq!(
            flag_words("-e\n-c"),
            vec!["/bin/sh".to_owned(), "-c".to_owned(), "e\nc".to_owned()]
        );
        // An escaped newline does not end a line, so what follows it is script
        // rather than the front of anything.
        assert_eq!(
            flag_words("-e\\\n-c\n-x"),
            vec![
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "e\\\n-c\nx".to_owned()
            ]
        );
        // Under `.POSIX:` the recursion defaults its own flags to `-ec`, so
        // the second word is not always `-c`. Measured: `.IGNORE:` does not
        // change it, because only the line's own `-` clears COMMANDS_NOERROR.
        assert_eq!(
            posix_flag_words("\"-e\" -c"),
            vec![
                "/bin/sh".to_owned(),
                "-ec".to_owned(),
                "\"-e\" -c".to_owned()
            ]
        );
        // The gates that are part of the tokenizer itself send the flags the
        // same way, because GNU Make calls the whole of the function: a lone
        // word that is a shell builtin is the shell's to run. Measured on
        // `export`, `unset`, `readonly` and `ulimit -c`; `set` and `exec` are
        // in the band where the oracle's own heap overflow kills it before it
        // can answer.
        assert_eq!(
            flag_words("export"),
            vec!["/bin/sh".to_owned(), "-c".to_owned(), "export".to_owned()]
        );
        assert_eq!(
            flag_words("set"),
            vec!["/bin/sh".to_owned(), "-c".to_owned(), "set".to_owned()]
        );
    }

    fn command_line_flags(flags: &str) -> Option<Vec<String>> {
        command_line_flag_argv(flags.as_bytes()).map(|words| {
            words
                .into_iter()
                .map(|word| String::from_utf8(word.to_vec()).unwrap())
                .collect()
        })
    }

    #[test]
    fn every_other_launchs_flags_are_the_words_of_a_command_line() {
        assert_eq!(command_line_flags("-c"), Some(vec!["-c".to_owned()]));
        assert_eq!(
            command_line_flags("-e -c"),
            Some(vec!["-e".to_owned(), "-c".to_owned()])
        );
        // The quoting is a command line's, so a quoted flag with a blank in it
        // is still one word: GNU Make copies the flags into the assembled line
        // as they stand and lets the tokenizer read them there.
        assert_eq!(
            command_line_flags("-E 'a b' -c"),
            Some(vec!["-E".to_owned(), "a b".to_owned(), "-c".to_owned()])
        );
        // Blank flags are no word rather than an empty one — the assembled
        // line has a run of blanks where they were, and a run of blanks is not
        // a word. This is what makes `.SHELLFLAGS :=` hand the script to the
        // shell as a file operand, which is 4.4.1's answer for it.
        assert_eq!(command_line_flags(""), Some(Vec::new()));
        assert_eq!(command_line_flags("  \t "), Some(Vec::new()));
        // `None` is the whole command line going to a shell, which is where
        // the flags get split instead. Shell syntax in the flags is one way in
        // and a leading-word rule is the other — the shell would have been
        // ahead of them on the real line, so answering `None` and letting a
        // shell split them reaches the words GNU Make reaches.
        assert_eq!(command_line_flags("-c $(unterminated"), None);
        assert_eq!(command_line_flags("set -c"), None);
    }

    #[test]
    fn a_plain_command_is_taken_apart() {
        assert_eq!(argv("./prog"), Some(vec!["./prog".to_owned()]));
        assert_eq!(
            argv("./prog arg1 arg2"),
            Some(vec![
                "./prog".to_owned(),
                "arg1".to_owned(),
                "arg2".to_owned()
            ])
        );
        assert_eq!(
            argv("/bin/echo   a\tb"),
            Some(vec!["/bin/echo".to_owned(), "a".to_owned(), "b".to_owned()])
        );
    }

    #[test]
    fn every_shell_character_sends_the_line_to_the_shell() {
        for line in [
            "./prog > out",
            "./prog *",
            "./prog ; true",
            "./prog $HOME",
            "./prog \"a b\"",
            "./prog a!b",
            "./prog ~/x",
            "./prog | wc",
            "./prog && true",
            "./prog # comment",
            "./prog `date`",
            "./prog (x)",
            "./prog {x}",
            "./prog [x]",
            "./prog a?b",
            "./prog a^b",
        ] {
            assert_eq!(argv(line), None, "{line}");
        }
    }

    #[test]
    fn a_leading_shell_builtin_sends_the_line_to_the_shell() {
        assert_eq!(argv("test -f Makefile"), None);
        assert_eq!(argv("cd /nosuchdir"), None);
        assert_eq!(argv("exec ./prog"), None);
        assert_eq!(argv(":"), None);
        // Not a builtin, so it is exec'd through PATH like anything else.
        assert_eq!(
            argv("echo hi"),
            Some(vec!["echo".to_owned(), "hi".to_owned()])
        );
    }

    #[test]
    fn a_leading_assignment_sends_the_line_to_the_shell() {
        assert_eq!(argv("FOO=bar ./prog"), None);
        // Past the first word without one, an equals is an ordinary byte.
        assert_eq!(
            argv("./prog a=b"),
            Some(vec!["./prog".to_owned(), "a=b".to_owned()])
        );
    }

    #[test]
    fn quoting_the_tokenizer_handles_itself_stays_on_the_fast_path() {
        assert_eq!(
            argv("./prog 'a b'"),
            Some(vec!["./prog".to_owned(), "a b".to_owned()])
        );
        assert_eq!(
            argv("./prog a\\ b"),
            Some(vec!["./prog".to_owned(), "a b".to_owned()])
        );
        assert_eq!(argv("./prog 'unterminated"), None);
    }

    #[test]
    fn the_shell_and_its_flags_are_gates_of_their_own() {
        assert_eq!(direct_argv(b"./prog", b"/bin/bash", b"-c", false), None);
        assert_eq!(direct_argv(b"./prog", DEFAULT_SHELL, b"-xc", false), None);
        assert!(direct_argv(b"./prog", DEFAULT_SHELL, b"-ec", false).is_some());
        assert!(direct_argv(b"./prog", DEFAULT_SHELL, b"-c", false).is_some());
    }

    #[test]
    fn a_line_with_nothing_in_it_is_nobodys_errand() {
        assert_eq!(argv(""), None);
        assert_eq!(argv("   "), None);
    }

    #[test]
    fn oneshell_makes_a_newline_a_separator() {
        assert!(direct_argv(b"./prog\ntrue", DEFAULT_SHELL, b"-c", true).is_none());
        assert_eq!(
            direct_argv(b"./prog\ntrue", DEFAULT_SHELL, b"-c", false),
            Some(vec![
                Bytes::from_static(b"./prog"),
                Bytes::from_static(b"true")
            ])
        );
    }
}
