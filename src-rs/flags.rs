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
    env,
    ffi::{OsStr, OsString},
    os::unix::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
    vec::IntoIter,
};

use crate::{
    strutil::{Pattern, word_scanner},
    symtab::Symtab,
};
use bytes::Bytes;

/// The canonical switch state produced after a Makefile writes `MAKEFLAGS`.
///
/// The evaluator owns when a global assignment takes effect, while an
/// embedding Make frontend owns the option grammar. This value is the narrow
/// boundary between them: kati never grows a second GNU Make option parser.
pub struct DecodedMakeflags {
    /// `MAKEFLAGS` before the optional `MAKEOVERRIDES` reference.
    pub makeflags: Bytes,
    /// The switch table the next assignment is decoded over.
    ///
    /// Not the same text as [`Self::makeflags`], because GNU Make's table
    /// remembers switches it deliberately does not publish — `--jobserver-style`
    /// is held in a variable of its own and never written into `MAKEFLAGS`, yet
    /// a later assignment is still decoded with it in force. Publishing what the
    /// table holds and carrying what it remembers are two questions, and only
    /// the frontend knows which switches belong to which.
    pub carried: Bytes,
    /// The same switches in command-line spelling.
    pub mflags: Bytes,
    pub is_dry_run: bool,
    pub is_silent_mode: bool,
    pub ignore_errors: bool,
    pub environment_overrides: bool,
    pub no_builtin_rules: bool,
    pub no_builtin_variables: bool,
}

/// Decode one evaluated Makefile assignment into GNU Make's switch table.
pub type MakeflagsDecoder =
    fn(previous: &[u8], assigned: &[u8], protected: &[u8]) -> Result<DecodedMakeflags, String>;

/// Session-owned state for GNU Make's special `MAKEFLAGS` assignment rules.
pub struct MakeflagsAssignment {
    pub decoder: MakeflagsDecoder,
    /// Switches inherited from the environment and argv, which outrank writes
    /// in the Makefile, in the switch table's own spelling rather than the
    /// published one.
    pub protected: Bytes,
    /// The accumulated switch table after the last assignment, which is
    /// [`DecodedMakeflags::carried`] and not what `MAKEFLAGS` shows.
    pub effective: Bytes,
    /// Command-line overrides make `$(MAKEOVERRIDES)` a permanent recursive
    /// suffix of `MAKEFLAGS`, even if the Makefile later empties it.
    pub has_overrides: bool,
}

/// Everything the command line says, as a value.
///
/// This used to be a `LazyLock` that read `std::env::args_os()` the first time
/// anything touched it, which made the process command line an input to every
/// evaluation in it. It is now constructed by [`Flags::from_args`] and owned by
/// the session.
// [spec:ronin:req:make.no-ambient-state]
#[derive(Default)]
pub struct Flags {
    pub detect_android_echo: bool,
    pub detect_depfiles: bool,
    pub dump_kati_stamp: bool,
    pub dump_include_graph: Option<OsString>,
    pub dump_variable_assignment_trace: Option<OsString>,
    pub enable_debug: bool,
    pub enable_kati_warnings: bool,
    pub enable_stat_logs: bool,
    pub gen_all_targets: bool,
    pub generate_ninja: bool,
    pub generate_empty_ninja: bool,
    pub is_dry_run: bool,
    pub is_silent_mode: bool,
    pub is_syntax_check_only: bool,
    pub regen: bool,
    pub regen_debug: bool,
    pub regen_ignoring_kati_binary: bool,
    pub use_find_emulator: bool,
    pub color_warnings: bool,
    /// GNU Make's `-i`: every recipe line is run for its effect and not for its
    /// status, which is the `-` prefix applied to all of them at once.
    pub ignore_errors: bool,
    /// GNU Make's `-e`: a variable that came from the environment outranks the
    /// makefile's own assignment to it, which is the difference between
    /// [`VarOrigin::Environment`](crate::var::VarOrigin::Environment) and
    /// [`VarOrigin::EnvironmentOverride`](crate::var::VarOrigin::EnvironmentOverride).
    pub environment_overrides: bool,
    pub no_builtin_rules: bool,
    pub no_ninja_prelude: bool,
    pub use_ninja_phony_output: bool,
    pub use_ninja_validations: bool,
    pub emit_sandbox_disabled: bool,
    pub werror_find_emulator: bool,
    pub werror_overriding_commands: bool,
    pub warn_implicit_rules: bool,
    pub werror_implicit_rules: bool,
    pub warn_suffix_rules: bool,
    pub werror_suffix_rules: bool,
    pub top_level_phony: bool,
    pub warn_real_to_phony: bool,
    pub werror_real_to_phony: bool,
    pub warn_phony_looks_real: bool,
    pub werror_phony_looks_real: bool,
    pub werror_writable: bool,
    pub warn_real_no_cmds_or_deps: bool,
    pub werror_real_no_cmds_or_deps: bool,
    pub warn_real_no_cmds: bool,
    pub werror_real_no_cmds: bool,
    pub default_pool: OsString,
    pub ignore_dirty_pattern: Option<crate::strutil::Pattern>,
    pub no_ignore_dirty_pattern: Option<crate::strutil::Pattern>,
    pub ignore_optional_include_pattern: Option<crate::strutil::Pattern>,
    /// The Makefiles the invocation named, in the order it named them.
    ///
    /// GNU Make reads every `-f` argument, one after another, as though the
    /// files had been concatenated — so this is a list, and the order carries
    /// meaning beyond the reading: variables a file assigns are in scope for
    /// the ones after it, a later rule for the same target overrides an
    /// earlier one, and the default goal comes from the first file that
    /// declares an eligible target rather than the last.
    pub makefiles: Vec<OsString>,
    pub ninja_dir: Option<OsString>,
    pub ninja_suffix: OsString,
    pub working_dir: Option<OsString>, // -C <dir>
    pub num_cpus: usize,
    pub num_jobs: usize,
    pub remote_num_jobs: usize,
    pub subkati_args: Vec<OsString>,
    pub targets: Vec<crate::symtab::Symbol>,
    pub cl_vars: Vec<Bytes>,
    /// The option portion of `MAKEFLAGS`, supplied by an embedding Make
    /// frontend. Assignments stay separate so evaluation can expose GNU Make's
    /// recursive `$(MAKEOVERRIDES)` relationship rather than a flattened
    /// environment string. `None` preserves standalone kati's inherited
    /// environment behavior.
    pub makeflags: Option<Bytes>,
    pub make_overrides: Option<Bytes>,
    /// How an embedding Make frontend handles writes to `MAKEFLAGS`.
    /// Standalone kati leaves this absent and retains its historical behavior.
    pub makeflags_assignment: Option<MakeflagsAssignment>,
    pub writable: Vec<OsString>,
    pub traced_variables_pattern: Vec<crate::strutil::Pattern>,

    pub cpu_profile_path: Option<OsString>,
    pub memory_profile_path: Option<OsString>,

    /// What the program calls itself in a diagnostic that carries no location.
    ///
    /// GNU Make leads those with its own name — `make: *** No targets.  Stop.`
    /// — and at depth with the level too, as `make[1]:`. Its test suite reads
    /// the name back out of exactly that message and then writes it into every
    /// expected diagnostic, so a front end that answers this correctly is
    /// measured under its own name rather than having to claim Make's.
    ///
    /// Empty leaves the message unprefixed, which is what kati's own binary has
    /// always done.
    pub program_name: String,

    /// Features the front end running the recipes provides, added to
    /// [`EVALUATOR_FEATURES`](crate::evaluate::EVALUATOR_FEATURES) in
    /// `.FEATURES`.
    ///
    /// The jobserver is the case that needs this: whether a build shares one
    /// token budget is decided by whoever spawns the recipes, and the evaluator
    /// never does.
    pub extra_features: Vec<String>,

    /// Where `-I` says to look for an `include` the working directory does not
    /// have, in the order given.
    pub include_dirs: Vec<PathBuf>,

    /// `.EXPORT_ALL_VARIABLES`: every variable the Makefile defined reaches the
    /// recipe's environment without being named.
    pub export_all_variables: bool,

    /// `-R`: no `CC`, `CXX` or `AR` unless the Makefile says so itself. What
    /// Make defines about itself stays, which is why this is not `-r`.
    pub no_builtin_variables: bool,

    /// `.ONESHELL`: the whole recipe is one script rather than a line at a
    /// time, so a `cd` carries and a failing line does not stop the rest.
    pub one_shell: bool,

    /// `.NOTPARALLEL` with no prerequisites: this makefile's own targets run
    /// one at a time. What it hands a sub-make is untouched.
    pub not_parallel: bool,

    /// `--shuffle`: the order the goals and each target's prerequisites are
    /// considered in, when it is not the order the Makefile wrote.
    pub shuffle: crate::shuffle::Shuffle,
}

fn parse_command_line_option_with_arg(
    option: &str,
    arg: &OsStr,
    args: &mut IntoIter<OsString>,
) -> Option<OsString> {
    let arg = arg.as_bytes();
    let arg = arg.strip_prefix(option.as_bytes())?;
    if arg.is_empty() {
        return args.next();
    }
    if let Some(arg) = arg.strip_prefix(b"=") {
        return Some(OsString::from_vec(arg.to_vec()));
    }
    // E.g, -j999
    if option.len() == 2 {
        return Some(OsString::from_vec(arg.to_vec()));
    }
    None
}

impl Flags {
    /// The flags implied by `args`, which is a whole `argv` including the
    /// program name.
    ///
    /// `symtab` is here only because command-line targets are interned; the
    /// flags keep no reference to it.
    pub fn from_args(args: Vec<OsString>, symtab: &mut Symtab) -> Flags {
        let mut iter = args.into_iter();
        let mut flags = Flags::default();
        let program = iter.next().unwrap();
        // What GNU Make leads a location-less diagnostic with: the name it was
        // invoked under. kati printed nothing there, which reads as output from
        // whatever ran it rather than from the tool that failed.
        flags.program_name = Path::new(&program)
            .file_name()
            .unwrap_or(program.as_os_str())
            .to_string_lossy()
            .into_owned();
        flags.subkati_args.push(program);
        flags.num_jobs = std::thread::available_parallelism().map_or(1, |p| p.get());
        flags.num_cpus = flags.num_jobs;

        if let Some(makeflags) = env::var_os("MAKEFLAGS") {
            for tok in crate::strutil::word_scanner(makeflags.as_bytes()) {
                if !tok.starts_with(b"-") && tok.contains(&b'=') {
                    flags.cl_vars.push(Bytes::from(tok.to_vec()));
                }
            }
        }

        while let Some(arg) = iter.next() {
            let mut should_propagate = true;
            match arg.as_bytes() {
                b"-f" => {
                    flags.makefiles.extend(iter.next());
                    should_propagate = false;
                }
                b"-c" => flags.is_syntax_check_only = true,
                b"-i" => flags.is_dry_run = true,
                b"-s" => flags.is_silent_mode = true,
                b"-d" => flags.enable_debug = true,
                b"--kati_stats" => flags.enable_stat_logs = true,
                b"--warn" => flags.enable_kati_warnings = true,
                b"--ninja" => flags.generate_ninja = true,
                b"--empty_ninja_file" => flags.generate_empty_ninja = true,
                b"--gen_all_targets" => flags.gen_all_targets = true,
                b"--regen" => {
                    // TODO: Make this default.
                    flags.regen = true
                }
                b"--regen_debug" => flags.regen_debug = true,
                b"--regen_ignoring_kati_binary" => flags.regen_ignoring_kati_binary = true,
                b"--dump_kati_stamp" => {
                    flags.dump_kati_stamp = true;
                    flags.regen_debug = true;
                }
                b"--detect_android_echo" => flags.detect_android_echo = true,
                b"--detect_depfiles" => flags.detect_depfiles = true,
                b"--color_warnings" => flags.color_warnings = true,
                b"--no_builtin_rules" => flags.no_builtin_rules = true,
                b"--no_ninja_prelude" => flags.no_ninja_prelude = true,
                b"--use_ninja_phony_output" => flags.use_ninja_phony_output = true,
                b"--use_ninja_validations" => flags.use_ninja_validations = true,
                b"--emit_sandbox_disabled" => flags.emit_sandbox_disabled = true,
                b"--werror_find_emulator" => flags.werror_find_emulator = true,
                b"--werror_overriding_commands" => flags.werror_overriding_commands = true,
                b"--warn_implicit_rules" => flags.warn_implicit_rules = true,
                b"--werror_implicit_rules" => flags.werror_implicit_rules = true,
                b"--warn_suffix_rules" => flags.warn_suffix_rules = true,
                b"--werror_suffix_rules" => flags.werror_suffix_rules = true,
                b"--top_level_phony" => flags.top_level_phony = true,
                b"--warn_real_to_phony" => flags.warn_real_to_phony = true,
                b"--werror_real_to_phony" => {
                    flags.warn_real_to_phony = true;
                    flags.werror_real_to_phony = true;
                }
                b"--warn_phony_looks_real" => flags.warn_phony_looks_real = true,
                b"--werror_phony_looks_real" => {
                    flags.warn_phony_looks_real = true;
                    flags.werror_phony_looks_real = true;
                }
                b"--werror_writable" => flags.werror_writable = true,
                b"--warn_real_no_cmds_or_deps" => flags.warn_real_no_cmds_or_deps = true,
                b"--werror_real_no_cmds_or_deps" => {
                    flags.warn_real_no_cmds_or_deps = true;
                    flags.werror_real_no_cmds_or_deps = true;
                }
                b"--warn_real_no_cmds" => flags.warn_real_no_cmds = true,
                b"--werror_real_no_cmds" => {
                    flags.warn_real_no_cmds = true;
                    flags.werror_real_no_cmds = true;
                }
                b"--use_find_emulator" => flags.use_find_emulator = true,
                _ => {
                    if let Some(arg) = parse_command_line_option_with_arg("-C", &arg, &mut iter) {
                        flags.working_dir = Some(arg);
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--dump_include_graph", &arg, &mut iter)
                    {
                        flags.dump_include_graph = Some(arg);
                    } else if let Some(arg) = parse_command_line_option_with_arg(
                        "--dump_variable_assignment_trace",
                        &arg,
                        &mut iter,
                    ) {
                        flags.dump_variable_assignment_trace = Some(arg);
                    } else if let Some(arg) = parse_command_line_option_with_arg(
                        "--variable_assignment_trace_filter",
                        &arg,
                        &mut iter,
                    ) {
                        for pat in word_scanner(arg.as_bytes()) {
                            flags
                                .traced_variables_pattern
                                .push(Pattern::new(Bytes::from(pat.to_vec())));
                        }
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("-j", &arg, &mut iter)
                    {
                        let Some(num_jobs) = arg.to_string_lossy().parse::<usize>().ok() else {
                            panic!("Invalid -j flag: {}", arg.to_string_lossy());
                        };
                        flags.num_jobs = num_jobs;
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--remote_num_jobs", &arg, &mut iter)
                    {
                        let Some(num_jobs) = arg.to_string_lossy().parse::<usize>().ok() else {
                            panic!("Invalid --remote_num_jobs flag: {}", arg.to_string_lossy());
                        };
                        flags.remote_num_jobs = num_jobs;
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--ninja_suffix", &arg, &mut iter)
                    {
                        flags.ninja_suffix = arg;
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--ninja_dir", &arg, &mut iter)
                    {
                        flags.ninja_dir = Some(arg);
                    } else if let Some(arg) = parse_command_line_option_with_arg(
                        "--ignore_optional_include",
                        &arg,
                        &mut iter,
                    ) {
                        flags.ignore_optional_include_pattern =
                            Some(Pattern::new(Bytes::from(arg.as_bytes().to_vec())));
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--ignore_dirty", &arg, &mut iter)
                    {
                        flags.ignore_dirty_pattern =
                            Some(Pattern::new(Bytes::from(arg.as_bytes().to_vec())));
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--no_ignore_dirty", &arg, &mut iter)
                    {
                        flags.no_ignore_dirty_pattern =
                            Some(Pattern::new(Bytes::from(arg.as_bytes().to_vec())));
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--writable", &arg, &mut iter)
                    {
                        flags.writable.push(arg);
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--default_pool", &arg, &mut iter)
                    {
                        flags.default_pool = arg;
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--cpu_profile", &arg, &mut iter)
                    {
                        flags.cpu_profile_path = Some(arg)
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--mem_profile", &arg, &mut iter)
                    {
                        flags.memory_profile_path = Some(arg)
                    } else if arg.as_bytes().starts_with(b"-") {
                        panic!("Unknown flag: {}", arg.to_string_lossy());
                    } else if arg.as_bytes().contains(&b'=') {
                        flags.cl_vars.push(Bytes::from(arg.as_bytes().to_vec()));
                    } else {
                        should_propagate = false;
                        let arg = Bytes::from(arg.as_bytes().to_vec());
                        flags.targets.push(symtab.intern(arg));
                    }
                }
            }
            if should_propagate {
                flags.subkati_args.push(arg);
            }
        }

        if !flags.traced_variables_pattern.is_empty()
            && flags.dump_variable_assignment_trace.is_none()
        {
            panic!(
                "--variable_assignment_trace_filter is valid only together with --dump_variable_assignment_trace"
            );
        }

        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flags() {
        let mut symtab = Symtab::new();
        let flags = Flags::from_args(
            vec!["test", "-f", "main.mk"]
                .into_iter()
                .map(|s| s.into())
                .collect(),
            &mut symtab,
        );
        assert_eq!(flags.makefiles, vec![OsString::from("main.mk")]);
    }

    /// Every `-f` names a Makefile to read, and the order they were written in
    /// is the order they are read in.
    #[test]
    fn every_file_argument_is_kept_in_order() {
        let mut symtab = Symtab::new();
        let flags = Flags::from_args(
            vec!["test", "-f", "one.mk", "-f", "two.mk", "-f", "one.mk"]
                .into_iter()
                .map(|s| s.into())
                .collect(),
            &mut symtab,
        );
        assert_eq!(
            flags.makefiles,
            vec![
                OsString::from("one.mk"),
                OsString::from("two.mk"),
                OsString::from("one.mk"),
            ]
        );
    }

    #[test]
    fn test_parse_command_line_option_with_arg() {
        assert_eq!(
            parse_command_line_option_with_arg(
                "--ignore_optional_include",
                &OsString::from("--ignore_optional_include=out/%.P"),
                &mut vec![].into_iter()
            ),
            Some(OsString::from("out/%.P"))
        );
    }
}
