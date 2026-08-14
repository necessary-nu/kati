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

// TODO: Add docs
#![allow(missing_docs)]
// These are the lints enabled by default in Android
// #![deny(missing_docs)]
// Upstream also denies `warnings` here. Vendored, that turns every rustc
// release into a build break for a lint we did not ask for and a downstream
// `cargo install` cannot suppress, which is a worse failure than the one it
// prevents. The denial moved to the embedding workspace's release gate, which
// runs `cargo clippy -p kati --all-targets -- -D warnings`: the same check, at
// a point where a compiler upgrade fails something a person is running rather
// than everybody's build. `tests/no_globals.rs` covers the case that made this
// worth keeping — an unused `static` — and covers it whether or not it is
// used.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

use strutil::trim_prefix_str;

/// The error type every fallible entry point here returns.
///
/// Re-exported because an out-of-crate [`build_sink::BuildSink`] has to name it
/// to implement the trait, and an embedder that depends on its own copy of the
/// crate gets a different type with the same name.
pub use anyhow;
/// The byte string this crate hands out for every name and command.
///
/// Re-exported for the same reason as [`anyhow`]: it is in the signature of
/// [`symtab::Symbol::as_bytes`] and [`strutil::escape_shell`], so an embedder
/// cannot avoid naming it.
pub use bytes;

pub mod build_sink;
pub mod builtin_rules;
pub mod builtins;
pub mod command;
pub mod dep;
pub mod eval;
pub mod evaluate;
pub mod exec;
pub mod export;
pub mod expr;
pub mod file;
pub mod file_cache;
pub mod fileutil;
pub mod find;
pub mod flags;
pub mod func;
pub mod io;
pub mod loc;
pub mod logging;
pub mod ninja;
pub mod parser;
pub mod regen;
pub mod regen_dump;
pub mod rule;
pub mod session;
pub mod stats;
pub mod stmt;
pub mod strutil;
pub mod symtab;
pub mod timeutil;
pub mod var;

#[macro_export]
macro_rules! log {
    ($fmt:expr $(, $($arg:tt)*)?) => {
        log::trace!($fmt, $($($arg)*)?)
    };
}

#[macro_export]
macro_rules! log_stat {
    ($ctx:expr, $fmt:expr $(, $($arg:tt)*)?) => {
        if $crate::session::Context::flags($ctx).enable_stat_logs {
            eprintln!("*kati*: {}", format!($fmt, $($($arg)*)?))
        }
    };
}

#[macro_export]
macro_rules! warn {
    ($fmt:expr $(, $($arg:tt)*)?) => {
        eprintln!($fmt, $($($arg)*)?)
    };
}

#[macro_export]
macro_rules! kati_warn {
    ($ctx:expr, $fmt:expr $(, $($arg:tt)*)?) => {
        if $crate::session::Context::flags($ctx).enable_kati_warnings {
            eprintln!($fmt, $($($arg)*)?)
        }
    };
}

#[macro_export]
macro_rules! error {
    ($fmt:expr $(, $($arg:tt)*)?) => {
        anyhow::bail!($fmt, $($($arg)*)?)
    };
}

/// Warn about something at a location. The first argument is whatever carries
/// the session — an `Evaluator` or a `Session` — because rendering the
/// location needs the interner its filename was interned into.
// [spec:ronin:req:make.no-ambient-state]
#[macro_export]
macro_rules! warn_loc {
    ($ctx:expr, $loc:expr, $fmt:expr $(, $($arg:tt)*)?) => {
        $crate::color_warn_log($ctx, $loc, format!($fmt, $($($arg)*)?))
    };
}

#[macro_export]
macro_rules! kati_warn_loc {
    ($ctx:expr, $loc:expr, $fmt:expr $(, $($arg:tt)*)?) => {
        if $crate::session::Context::flags($ctx).enable_kati_warnings {
            $crate::color_warn_log($ctx, $loc, format!($fmt, $($($arg)*)?))
        }
    };
}

#[macro_export]
macro_rules! error_loc {
    ($ctx:expr, $loc:expr, $fmt:expr $(, $($arg:tt)*)?) => {
        return Err($crate::color_error_log($ctx, $loc, format!($fmt, $($($arg)*)?)))
    };
}

/// How GNU Make ends the diagnostic it dies on: two spaces, then `Stop.`.
const STOP: &str = "  Stop.";

/// The system's own words for an I/O failure, as `strerror` gives them.
///
/// Rust renders an `io::Error` as `strerror(errno)` with ` (os error N)`
/// appended. GNU Make quotes `strerror` and nothing else, and so does the
/// manifest front end this crate compiles for, so the suffix comes off: it is
/// Rust's spelling rather than either tool's. It is removed by the errno the
/// error itself reports rather than by scanning the text for a shape, so a
/// message that legitimately contains those words keeps them.
///
/// An error with no errno behind it — one built from a `Kind`, or from another
/// error — renders as it is.
pub fn strerror(error: &std::io::Error) -> String {
    let rendered = error.to_string();
    let Some(code) = error.raw_os_error() else {
        return rendered;
    };
    match rendered.strip_suffix(&format!(" (os error {code})")) {
        Some(message) => message.to_owned(),
        None => rendered,
    }
}

/// An I/O failure that names the path it happened to.
///
/// For the sites that read or write a file with no evaluation location to hand
/// — the executor's own stats, a file the caller named on the command line.
/// Where there is a location, raise through [`error_loc!`] instead so the
/// diagnostic points at the directive that asked for the file.
pub fn io_failure(path: &std::path::Path, error: &std::io::Error) -> anyhow::Error {
    anyhow::format_err!("{}: {}", path.display(), strerror(error))
}

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";

/// What a diagnostic with no makefile location opens with, colon and space
/// included, or nothing when no name was set.
///
/// Most callers get this from [`color_error_log`] without asking. This is for
/// the one that cannot: GNU Make's recipe-failure line takes the name like
/// every other diagnostic, but it is not a fatal it dies on, so it must not
/// pick up the `Stop.` that goes with one.
pub fn diagnostic_prefix(ctx: &impl crate::session::Context) -> String {
    let program = &ctx.flags().program_name;
    if program.is_empty() {
        return String::new();
    }
    format!("{program}: ")
}

fn color_error_log(
    ctx: &impl crate::session::Context,
    loc: Option<&crate::loc::Loc>,
    msg: String,
) -> anyhow::Error {
    // Everything raised through here is a fatal one, and GNU Make ends one the
    // same way whether or not it has a place to point at: the message, two
    // spaces, `Stop.`. The exception is its recipe-failure line, which is not a
    // fatal it dies on and so is raised without coming through here.
    let msg = format!("{msg}{STOP}");
    let Some(loc) = loc else {
        // With no file and line to lead with, GNU Make leads with its own name.
        // Empty is kati's own binary, which has never led with anything.
        let program = &ctx.flags().program_name;
        if program.is_empty() {
            return anyhow::format_err!("{msg}");
        }
        return anyhow::format_err!("{program}: {msg}");
    };
    let loc = loc.display(ctx);

    if ctx.flags().color_warnings {
        let filtered = trim_prefix_str(&msg, "*** ");

        anyhow::format_err!("{BOLD}{loc}: {RED}error: {RESET}{BOLD}{filtered}{RESET}")
    } else {
        anyhow::format_err!("{loc}: {msg}")
    }
}

fn color_warn_log(ctx: &impl crate::session::Context, loc: Option<&crate::loc::Loc>, msg: String) {
    let Some(loc) = loc else {
        // With no file and line to lead with, GNU Make leads with its own name,
        // exactly as it does for a fatal raised from nowhere in particular.
        // Empty is kati's own binary, which has never led with anything.
        let program = &ctx.flags().program_name;
        if program.is_empty() {
            eprintln!("{msg}");
        } else {
            eprintln!("{program}: {msg}");
        }
        return;
    };
    let loc = loc.display(ctx);

    if ctx.flags().color_warnings {
        let mut filtered = trim_prefix_str(&msg, "*warning*: ");
        filtered = trim_prefix_str(filtered, "warning: ");

        eprintln!("{BOLD}{loc}: {MAGENTA}warning: {RESET}{BOLD}{filtered}{RESET}")
    } else {
        eprintln!("{loc}: {msg}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_system_error_reads_as_strerror_alone() {
        let denied = std::io::Error::from_raw_os_error(13);
        assert!(denied.to_string().contains("(os error 13)"));
        assert_eq!(strerror(&denied), "Permission denied");
    }

    #[test]
    fn an_error_with_no_errno_reads_as_it_is() {
        let ours = std::io::Error::other("something we said ourselves");
        assert_eq!(strerror(&ours), "something we said ourselves");
    }

    #[test]
    fn a_message_that_says_os_error_itself_keeps_it() {
        // Keyed on the errno the error reports rather than on the shape of the
        // text, so a message that happens to carry those words survives.
        let quoting = std::io::Error::other("the words (os error 2) appeared in a file");
        assert_eq!(
            strerror(&quoting),
            "the words (os error 2) appeared in a file"
        );
    }

    #[test]
    fn an_io_failure_names_the_path_it_happened_to() {
        let denied = std::io::Error::from_raw_os_error(13);
        assert_eq!(
            io_failure(std::path::Path::new("sub/inc.mk"), &denied).to_string(),
            "sub/inc.mk: Permission denied"
        );
    }
}
