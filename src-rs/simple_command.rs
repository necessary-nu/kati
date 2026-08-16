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

use bytes::Bytes;

/// The shell GNU Make was compiled with, which is the only one the fast path
/// is willing to stand in for.
pub const DEFAULT_SHELL: &[u8] = b"/bin/sh";

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
