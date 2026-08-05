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
#![deny(warnings)]
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
pub mod command;
pub mod dep;
pub mod eval;
pub mod evaluate;
pub mod exec;
pub mod expr;
pub mod file;
pub mod file_cache;
pub mod fileutil;
pub mod find;
pub mod flags;
pub mod func;
pub mod io;
pub mod loc;
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

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";

fn color_error_log(
    ctx: &impl crate::session::Context,
    loc: Option<&crate::loc::Loc>,
    msg: String,
) -> anyhow::Error {
    let Some(loc) = loc else {
        return anyhow::format_err!("{msg}");
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
        eprintln!("{msg}");
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
