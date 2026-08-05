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

//! The `KATI_LOG` tracing that [`crate::log!`] writes into.
//!
//! This is the only tracing a Make evaluation has, so it stays; what went is
//! the machinery underneath it. `env_logger` reached this crate for two
//! things — turning `KATI_LOG` into a level filter, and writing a line to
//! stderr — and brought `env_filter`, `regex`, `aho-corasick`, `jiff` and the
//! `anstyle` colour stack with it, none of which any call site here uses:
//! every logging site in the crate is a `trace!`, with one target, no colour
//! and no timestamp.
//!
//! What it supports is the subset of `RUST_LOG` syntax that was reachable: a
//! bare level (`KATI_LOG=trace`), or comma-separated `target=level`
//! directives with the longest matching prefix winning
//! (`KATI_LOG=warn,kati::find=trace`). What it drops is regex message
//! filtering, timestamps and colour.

use std::io::Write;
use std::str::FromStr;

use log::{LevelFilter, Log, Metadata, Record};

/// Writes kati's log records to stderr, filtered by target.
struct StderrLogger {
    /// The level for a target no directive matches.
    default: LevelFilter,
    /// Target prefixes and their levels; the longest match wins.
    targets: Vec<(String, LevelFilter)>,
}

impl StderrLogger {
    /// The logger a directive string asks for.
    ///
    /// A directive that names no level, or names one that is not a level, is
    /// reported and skipped rather than silently changing what is traced: the
    /// point of setting the variable is to see something, and a typo that
    /// quietly logs nothing looks exactly like the code not running.
    fn from_directives(spec: &str, default: LevelFilter) -> Self {
        let mut logger = Self {
            default,
            targets: Vec::new(),
        };
        for directive in spec.split(',').map(str::trim).filter(|d| !d.is_empty()) {
            let (target, level) = match directive.split_once('=') {
                // `kati::find=` is `kati::find` at its most verbose, as in
                // `RUST_LOG`.
                Some((target, "")) => (target.trim(), LevelFilter::Trace),
                Some((target, level)) => {
                    let Ok(level) = LevelFilter::from_str(level.trim()) else {
                        eprintln!("*kati*: ignoring log directive {directive:?}: not a level");
                        continue;
                    };
                    (target.trim(), level)
                }
                // A lone word is either the level for everything or a target
                // to trace.
                None => match LevelFilter::from_str(directive) {
                    Ok(level) => {
                        logger.default = level;
                        continue;
                    }
                    Err(_) => (directive, LevelFilter::Trace),
                },
            };
            if target.is_empty() {
                eprintln!("*kati*: ignoring log directive {directive:?}: no target");
                continue;
            }
            logger.targets.push((target.to_string(), level));
        }
        logger
    }

    /// The level in force for `target`.
    fn level_for(&self, target: &str) -> LevelFilter {
        self.targets
            .iter()
            .filter(|(prefix, _)| target.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.default, |(_, level)| *level)
    }

    /// The most verbose level any target can reach, so that a disabled
    /// `trace!` costs one atomic load rather than a walk of the directives.
    fn max_level(&self) -> LevelFilter {
        self.targets
            .iter()
            .map(|(_, level)| *level)
            .chain(std::iter::once(self.default))
            .max()
            .unwrap_or(LevelFilter::Off)
    }
}

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level_for(metadata.target())
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // One `writeln!` under one lock, because these interleave with a
        // build's own output on the same stream.
        let mut stderr = std::io::stderr().lock();
        let _ = match (record.file(), record.line()) {
            (Some(file), Some(line)) => {
                writeln!(stderr, "*kati*: {file}:{line}: {}", record.args())
            }
            _ => writeln!(stderr, "*kati*: {}", record.args()),
        };
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

/// Send this process's log records to stderr, filtered by `var`.
///
/// `default` is the level for anything the variable does not mention, and
/// applies on its own when the variable is unset. Installing a logger is a
/// process-wide act, so this is for a binary's `main` to call: a second call,
/// or a call after something else has installed one, does nothing.
pub fn init(var: &str, default: LevelFilter) {
    let logger = match std::env::var(var) {
        Ok(spec) => StderrLogger::from_directives(&spec, default),
        Err(_) => StderrLogger {
            default,
            targets: Vec::new(),
        },
    };
    let max_level = logger.max_level();
    // Leaked rather than held in a `static`: the logger has to outlive the
    // call, `log::set_boxed_logger` would need the `log/std` feature that
    // nothing else here wants, and a `static` is the shape this crate has a
    // test forbidding.
    if log::set_logger(Box::leak(Box::new(logger))).is_ok() {
        log::set_max_level(max_level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_level_applies_to_everything() {
        let logger = StderrLogger::from_directives("trace", LevelFilter::Warn);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Trace);
        assert_eq!(logger.level_for("anything"), LevelFilter::Trace);
        assert_eq!(logger.max_level(), LevelFilter::Trace);
    }

    #[test]
    fn unset_variable_leaves_the_default() {
        let logger = StderrLogger::from_directives("", LevelFilter::Warn);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Warn);
        assert_eq!(logger.max_level(), LevelFilter::Warn);
    }

    #[test]
    fn a_target_directive_narrows_to_that_target() {
        let logger = StderrLogger::from_directives("kati::find=trace", LevelFilter::Warn);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Trace);
        assert_eq!(logger.level_for("kati::eval"), LevelFilter::Warn);
        // A target is a prefix, as in `RUST_LOG`.
        assert_eq!(logger.level_for("kati::find::inner"), LevelFilter::Trace);
    }

    #[test]
    fn the_longest_matching_prefix_wins_whatever_the_order() {
        let spec = "kati=trace,kati::find=off";
        for spec in [spec, "kati::find=off,kati=trace"] {
            let logger = StderrLogger::from_directives(spec, LevelFilter::Warn);
            assert_eq!(logger.level_for("kati::find"), LevelFilter::Off, "{spec}");
            assert_eq!(logger.level_for("kati::eval"), LevelFilter::Trace, "{spec}");
        }
    }

    #[test]
    fn a_bare_target_is_that_target_at_its_most_verbose() {
        let logger = StderrLogger::from_directives("kati::find", LevelFilter::Warn);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Trace);
        assert_eq!(logger.level_for("kati::eval"), LevelFilter::Warn);
    }

    #[test]
    fn a_directive_naming_no_level_is_skipped_not_guessed() {
        let logger = StderrLogger::from_directives("kati::find=louder,warn", LevelFilter::Off);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Warn);
        assert!(logger.targets.is_empty());
    }

    #[test]
    fn max_level_covers_the_most_verbose_directive() {
        let logger = StderrLogger::from_directives("off,kati::find=debug", LevelFilter::Warn);
        assert_eq!(logger.max_level(), LevelFilter::Debug);
        assert_eq!(logger.level_for("kati::eval"), LevelFilter::Off);
    }

    #[test]
    fn levels_are_case_insensitive_and_spacing_is_tolerated() {
        let logger = StderrLogger::from_directives(" WARN , kati::find = TRACE ", LevelFilter::Off);
        assert_eq!(logger.level_for("kati::eval"), LevelFilter::Warn);
        assert_eq!(logger.level_for("kati::find"), LevelFilter::Trace);
    }
}
