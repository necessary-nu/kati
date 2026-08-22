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
    /// The words of the write that bound a name rather than setting a switch.
    ///
    /// GNU Make reads these through the same `handle_non_switch_argument` as a
    /// command line's, and `try_variable_definition` then defines each at the
    /// origin the write carried — `o_file` for a Makefile's own. So they are
    /// ordinary Makefile assignments made where the write stands, and the
    /// evaluator applies them: the frontend owns the splitting, kati owns what
    /// an assignment means.
    pub assignments: Vec<Bytes>,
    /// `MAKEFLAGS` before the `-*-eval-flags-*-` and `MAKEOVERRIDES`
    /// references, which is the switch table alone.
    pub makeflags: Bytes,
    /// The `--eval` fragments this write's own text carried, quoted and joined.
    ///
    /// Read only for whether it is empty. GNU Make's `decode_switches` appends
    /// a `--eval` from any origin to `eval_strings`, which is what makes
    /// `define_makeflags` write the `$(-*-eval-flags-*-)` reference — but the
    /// variable that reference names was defined once at startup and is not
    /// touched again, so a fragment a makefile writes is named and not held.
    pub eval_flags: Bytes,
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
    /// The switch table's `-I` list as the write left it.
    ///
    /// A list rather than a bit, because `reset_makeflags` (main.c) calls
    /// `construct_include_path (include_dirs ? include_dirs->list : NULL)` on
    /// every write: a makefile's `MAKEFLAGS += -I dir` has to reach the search
    /// before the next `include`, and a `-I -` it writes has to turn the
    /// built-in directories off there and then. The frontend replays the whole
    /// table, so this is the accumulated list and not the addition.
    pub include_dirs: Vec<PathBuf>,
    pub is_dry_run: bool,
    pub is_silent_mode: bool,
    pub ignore_errors: bool,
    pub environment_overrides: bool,
    pub no_builtin_rules: bool,
    pub no_builtin_variables: bool,
    /// What the write had to say about a switch it dropped rather than died of.
    ///
    /// GNU Make's `decode_switches` complains about an empty string argument,
    /// or a job count that is not a positive integer, whatever origin the word
    /// came from — only the dying afterwards is the command line's alone. A
    /// makefile's own write to `MAKEFLAGS` therefore says something and carries
    /// on, and these are the lines it says. Rendered by the frontend, which
    /// owns the option grammar; raised by the evaluator, which owns the moment.
    pub complaints: Vec<Bytes>,
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
    /// Whether a `--eval` has been seen at all, which is what decides that
    /// `MAKEFLAGS` names `$(-*-eval-flags-*-)`.
    ///
    /// Sticky, and deliberately not the same question as whether the variable
    /// exists. GNU Make's `eval_strings` is a list that is only ever appended
    /// to — by the command line, by an inherited `MAKEFLAGS`, and by a
    /// makefile's own write — and `define_makeflags` writes the reference
    /// whenever it is non-empty. The variable behind it is defined once, from
    /// the fragments the INVOCATION carried, and never again. So a makefile
    /// that writes a `--eval` of its own makes `MAKEFLAGS` name a variable
    /// that does not hold it, and a makefile that writes `MAKEFLAGS` away
    /// cannot take the invocation's fragments off it.
    pub has_evals: bool,
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
    /// The program to run in place of `/bin/sh`, when the tool this was
    /// compiled into carries a shell of its own.
    ///
    /// A `$(shell)` call and a recipe line are the same language, so a front
    /// end whose build will run recipes under its own shell reads them with
    /// that shell too. Left `None` — which is what `rkati` leaves it — the
    /// machine's `/bin/sh` runs them, as it always did.
    pub default_shell_program: Option<std::path::PathBuf>,
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
    /// Whether this unit's makefile was already read, over this same text, on
    /// an earlier pass of the same compilation.
    ///
    /// A front end that compiles a recursive child into its parent's graph
    /// cannot read the child's makefile until the parent's staged work is on
    /// the ground, so it reads everything again once that work exists. GNU
    /// Make has no such pass — it reads once, before any recipe runs — and a
    /// Makefile must not be able to tell that this one happened. What a read
    /// does on the way through therefore happens on the first read of a
    /// makefile and not on the repeats: `$(info)` says its piece once,
    /// `$(warning)` warns once, and `$(file >>)` appends one line, all with
    /// the values the first read had, which are the values GNU Make's one read
    /// would have had.
    ///
    /// This is not what `-s` is. `$(info)` is a Makefile speaking rather than
    /// a recipe being echoed, and `-s` leaves it alone.
    pub is_repeated_read: bool,
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
    /// The `--eval` fragments this invocation was given, quoted as
    /// `MAKEFLAGS` carries them and joined by one space, or empty for an
    /// invocation given none.
    ///
    /// Held apart from [`Self::makeflags`] because GNU Make holds them apart:
    /// its `MAKEFLAGS` carries the text `$(-*-eval-flags-*-)` where these
    /// would go and the fragments live in a variable of that name. The switch
    /// table proper never contains one, which is also why `MFLAGS` — written
    /// in `define_makeflags` before the reference is appended — does not.
    pub eval_flags: Bytes,
    pub make_overrides: Option<Bytes>,
    /// How an embedding Make frontend handles writes to `MAKEFLAGS`.
    /// Standalone kati leaves this absent and retains its historical behavior.
    pub makeflags_assignment: Option<MakeflagsAssignment>,
    pub writable: Vec<OsString>,
    /// The names `-o` / `--old-file` / `--assume-old` asserted a date for, as
    /// the switch canonicalised them.
    ///
    /// GNU Make's `old_files`, and the read is the one place the list is
    /// legible to. `main` stamps `last_mtime = OLD_MTIME` on each of them
    /// (main.c:2312), and `file_mtime` (filedef.h) calls `f_mtime` only for a
    /// name whose date is still unknown — so the date of a name on this list is
    /// asserted and never stated, and the code at the end of `f_mtime` never
    /// runs for it. Turning the intermediate bit off is that code, which is why
    /// the list has to reach a read that decides the bit ahead of the build.
    pub old_files: Vec<Bytes>,
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

    /// What every `-I` named, in the order given and in the spelling the switch
    /// table stores.
    ///
    /// The switch table's own list rather than a search path: a bare `-` is an
    /// entry in it, not a state, because `construct_include_path` (read.c)
    /// reads the list from the start on every call and reaches the `-` where it
    /// was written. That is what makes `-I inc -I - -I inc` search nothing at
    /// all — the second `inc` is de-duplicated away before it is ever stored,
    /// so the reset has nothing after it to restore.
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

    /// `-k`: a failure is reported and passed over rather than ending the run.
    ///
    /// It reaches the evaluator because the makefile update is where GNU Make
    /// reads it first. `complain()` chooses between `fatal` and `error` on
    /// `keep_going_flag` (reference/gnumake/src/remake.c:422), so the switch
    /// decides whether the first makefile nothing can make is the last one
    /// considered or merely the first of several reported.
    pub keep_going: bool,
}

/// Why a command line was refused, in this front end's own words.
///
/// [`Flags::from_args`] is a library entry point — Ronin's equivalence gate
/// builds a session straight from an argv — so it can neither print nor exit.
/// It hands the complaint back and whoever owns the process renders it. The
/// standalone binary writes it to stderr and exits 2, which is the status GNU
/// Make leaves for a command line it would not take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// How much of the word a short option takes with it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShortArgument {
    /// None at all, so the letter beside it in a cluster is another option.
    None,
    /// Whatever follows it in the same word, or the next word when nothing
    /// does — GNU Make's `filename`, `strlist` and `positive_int` switches.
    Word,
    /// Only what is attached: GNU Make's `string` switch with a `noarg_value`,
    /// which getopt spells `O::`. A following word is a goal, not an argument.
    Attached,
    /// Attached, or the following word when it opens with a digit or a `.` —
    /// the extra peek `main.c` does for a `floating` switch, which is why
    /// `-l 2` consumes the `2` and `-l foo` leaves `foo` to be a goal.
    AttachedOrNumber,
}

/// What a GNU Make short option costs this front end, by letter.
///
/// `None` for a letter neither tool knows. The three answers otherwise are the
/// three things a compiler front end can do with a runner's switch: read it
/// into a field, drop it, or refuse it.
fn short_option(letter: u8) -> Option<(ShortArgument, ShortOption)> {
    use ShortArgument::{Attached, AttachedOrNumber, None, Word};
    use ShortOption::{Dropped, Read, Refused};
    Some(match letter {
        // Read into a field this front end already has, and already uses.
        b'c' | b'd' | b'e' | b'i' | b'k' | b'n' | b'R' | b'r' | b'S' | b's' => (None, Read),
        b'C' | b'f' | b'j' => (Word, Read),

        // Dropped. `-b` and `-m` GNU Make ignores itself; the rest name
        // something no compiled graph can carry — whether the database is
        // printed, whether symlink times are checked, where a directory
        // announcement goes, how a runner interleaves its output and how
        // heavily it loads the machine. `MAKE_OPTION_SURFACE` in
        // src/make/cli.rs classifies the same seven as no-ops, and is the
        // reference this follows.
        b'L' | b'b' | b'm' | b'p' | b'w' => (None, Dropped),
        b'O' => (Attached, Dropped),
        b'l' => (AttachedOrNumber, Dropped),

        // Refused, because accepting one silently would answer a question this
        // front end never asked. `-B`, `-t` and `-q` each say what to do
        // INSTEAD of building — remake everything, touch instead of make,
        // report instead of make — and a front end that compiles a graph and
        // reports success would be lying about all three. `-v` and `-h` build
        // nothing at all. `-E` evaluates a statement before the makefiles and
        // there is no field behind it.
        b'B' | b'h' | b'q' | b't' | b'v' => (None, Refused),
        b'E' => (Word, Refused),
        // `-I` and `-o` have fields — `include_dirs` and `old_files` — and are
        // still refused, because both take an argument whose canonicalisation
        // belongs to the front end rather than to the field: `-I -` resets the
        // search path where it stands rather than adding a directory named
        // `-`, and an old file is stored as the switch canonicalised it.
        // Reading the argument in raw would put the wrong bytes in a real
        // field, which is worse than not reading it. `-W` is the third of that
        // shape and has no field at all.
        b'I' | b'W' | b'o' => (Word, Refused),
        _ => return Option::None,
    })
}

/// What reading a short option costs, beside how much of the word it takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShortOption {
    Read,
    Dropped,
    Refused,
}

/// The words `arg` stands for once GNU Make's getopt has split it.
///
/// `-rR` is two options and GNU Make reads it as two; this table read it as
/// one word it had never heard of. Splitting happens only where it cannot
/// change an existing reading: the word must open with a single `-` followed
/// by a letter that takes no argument, which leaves `-j8`, `-Csub` and every
/// `--long` spelling exactly where they were. The first letter that does take
/// an argument ends the split and keeps the rest of the word attached to
/// itself, which is what getopt does with `-Cdir`.
///
/// A letter neither tool knows is emitted on its own rather than complained
/// about here, so the complaint below names the letter the way GNU Make's does
/// rather than the cluster it was written in.
fn split_short_cluster(arg: &OsStr) -> Option<Vec<OsString>> {
    let bytes = arg.as_bytes();
    let letters = bytes.strip_prefix(b"-")?;
    if letters.len() < 2 || letters[0] == b'-' {
        return None;
    }
    if short_option(letters[0]).is_none_or(|(shape, _)| shape != ShortArgument::None) {
        return None;
    }
    // `-` is the one byte getopt keeps for itself, so a word carrying one
    // among its letters is not a cluster at all — GNU Make answers `-rR-q`
    // with `invalid option -- '-'`. Left whole, it is complained about below
    // as the word it was written as.
    if letters.contains(&b'-') {
        return None;
    }
    let mut words = Vec::with_capacity(letters.len());
    for (index, &letter) in letters.iter().enumerate() {
        let takes_a_word =
            short_option(letter).is_some_and(|(shape, _)| shape != ShortArgument::None);
        if takes_a_word {
            let mut word = vec![b'-', letter];
            word.extend_from_slice(&letters[index + 1..]);
            words.push(OsString::from_vec(word));
            return Some(words);
        }
        words.push(OsString::from_vec(vec![b'-', letter]));
    }
    Some(words)
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
    pub fn from_args(args: Vec<OsString>, symtab: &mut Symtab) -> Result<Flags, Refusal> {
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

        // What a cluster split off the word in hand, waiting to be read as
        // words of their own. Never holds anything while an option is reaching
        // for its argument: the split attaches the rest of the word to the
        // first letter that takes one, so no letter after that is separate.
        let mut split: std::collections::VecDeque<OsString> = std::collections::VecDeque::new();
        let complain = |program: &str, text: String| Err(Refusal(format!("{program}: {text}")));
        while let Some(arg) = split.pop_front().or_else(|| iter.next()) {
            if split.is_empty()
                && let Some(words) = split_short_cluster(&arg)
                && words.len() > 1
            {
                split.extend(words);
                continue;
            }
            let mut should_propagate = true;
            if let Some(letter) = arg
                .as_bytes()
                .strip_prefix(b"-")
                .filter(|rest| rest.len() == 1)
                .map(|rest| rest[0])
            {
                match short_option(letter) {
                    Some((_, ShortOption::Refused)) => {
                        return complain(
                            &flags.program_name,
                            format!("unsupported option -- '{}'", char::from(letter)),
                        );
                    }
                    Some((shape, ShortOption::Dropped)) => {
                        if shape == ShortArgument::AttachedOrNumber
                            && iter
                                .as_slice()
                                .first()
                                .and_then(|next| next.as_bytes().first())
                                .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
                        {
                            iter.next();
                        }
                        flags.subkati_args.push(arg);
                        continue;
                    }
                    _ => {}
                }
            }
            match arg.as_bytes() {
                // A dropped switch written with its argument attached. Nothing
                // reads either half; the word is passed on as it was written.
                [b'-', b'O' | b'l', ..] => {}
                b"-f" => {
                    flags.makefiles.extend(iter.next());
                    should_propagate = false;
                }
                b"-c" => flags.is_syntax_check_only = true,
                // GNU Make's own letters. `-i` is `--ignore-errors` and `-n`
                // is `--dry-run`; this table read `-i` as the dry run and had
                // no `-n` at all, so `-i` ran nothing and reported success
                // where Make ignores a status, and `-n` reached the unknown
                // arm. Every field behind these was already written by the
                // embedding frontend and already read by the evaluator — only
                // the spelling on this side was wrong.
                b"-i" => flags.ignore_errors = true,
                b"-n" => flags.is_dry_run = true,
                b"-k" => flags.keep_going = true,
                // `-S` is Make's way of taking `-k` back off, which is what a
                // `MAKEFLAGS` inherited from a parent that was given both
                // relies on.
                b"-S" => flags.keep_going = false,
                b"-e" => flags.environment_overrides = true,
                b"-r" => flags.no_builtin_rules = true,
                b"-R" => flags.no_builtin_variables = true,
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
                            return complain(
                                &flags.program_name,
                                format!("Invalid -j flag: {}", arg.to_string_lossy()),
                            );
                        };
                        flags.num_jobs = num_jobs;
                    } else if let Some(arg) =
                        parse_command_line_option_with_arg("--remote_num_jobs", &arg, &mut iter)
                    {
                        let Some(num_jobs) = arg.to_string_lossy().parse::<usize>().ok() else {
                            return complain(
                                &flags.program_name,
                                format!(
                                    "Invalid --remote_num_jobs flag: {}",
                                    arg.to_string_lossy()
                                ),
                            );
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
                        return complain(
                            &flags.program_name,
                            format!("Unknown flag: {}", arg.to_string_lossy()),
                        );
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
            return complain(
                &flags.program_name,
                "--variable_assignment_trace_filter is valid only together with \
                 --dump_variable_assignment_trace"
                    .to_owned(),
            );
        }

        Ok(flags)
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert_eq!(
            flags.makefiles,
            vec![
                OsString::from("one.mk"),
                OsString::from("two.mk"),
                OsString::from("one.mk"),
            ]
        );
    }

    /// The letters, read the way GNU Make spells them.
    fn switches(words: &[&str]) -> Flags {
        let mut symtab = Symtab::new();
        let mut argv = vec!["rkati".to_owned()];
        argv.extend(words.iter().map(|word| (*word).to_owned()));
        Flags::from_args(argv.into_iter().map(Into::into).collect(), &mut symtab).unwrap()
    }

    /// The complaint a command line this front end will not take produces,
    /// which is a value rather than a crash or an exit.
    fn refusal(words: &[&str]) -> String {
        let mut symtab = Symtab::new();
        let mut argv = vec!["rkati".to_owned()];
        argv.extend(words.iter().map(|word| (*word).to_owned()));
        Flags::from_args(argv.into_iter().map(Into::into).collect(), &mut symtab)
            .err()
            .map_or_else(|| "accepted".to_owned(), |refusal| refusal.to_string())
    }

    /// `-i` is `--ignore-errors` and nothing else.
    ///
    /// This table read it as the dry run, which is a different switch and very
    /// nearly the opposite one: GNU Make 4.4.1 given `-i` over a recipe of
    /// `@touch m1; false` then `@touch m2` makes both markers and exits 0
    /// having run them, while a dry run makes neither and exits 0 having run
    /// nothing. Both spellings answer "it worked", so the difference is only
    /// ever visible in what is on the disk afterwards.
    #[test]
    fn a_dash_i_ignores_errors_rather_than_running_nothing() {
        let flags = switches(&["-i"]);
        assert!(flags.ignore_errors);
        assert!(!flags.is_dry_run);
    }

    /// `-n` is the dry run, and this table had no `-n` at all — it reached the
    /// unknown-flag arm and took the process with it.
    #[test]
    fn a_dash_n_is_the_dry_run() {
        let flags = switches(&["-n"]);
        assert!(flags.is_dry_run);
        assert!(!flags.ignore_errors);
    }

    /// `-k` carries on past a failure, and `-S` is how Make takes it back off
    /// — which is what a `MAKEFLAGS` inherited from a parent given both needs.
    #[test]
    fn keep_going_is_read_and_can_be_taken_back_off() {
        assert!(!switches(&[]).keep_going);
        assert!(switches(&["-k"]).keep_going);
        assert!(!switches(&["-k", "-S"]).keep_going);
        assert!(switches(&["-S", "-k"]).keep_going);
    }

    /// The three letters that change what a Makefile evaluates to rather than
    /// how its recipes run. Each names a field the evaluator already read and
    /// that only an embedding frontend could set.
    #[test]
    fn the_letters_that_change_the_evaluation_are_read() {
        assert!(switches(&["-e"]).environment_overrides);
        assert!(switches(&["-r"]).no_builtin_rules);
        assert!(switches(&["-R"]).no_builtin_variables);
        let none = switches(&[]);
        assert!(!none.environment_overrides);
        assert!(!none.no_builtin_rules);
        assert!(!none.no_builtin_variables);
    }

    /// A switch a sub-make has to be given again is passed on, and the letters
    /// above are all of that kind: GNU Make writes every one of them into
    /// `MAKEFLAGS` for the child.
    #[test]
    fn the_switches_reach_a_sub_make() {
        let flags = switches(&["-i", "-n", "-k", "-e", "-r", "-R"]);
        let carried: Vec<_> = flags
            .subkati_args
            .iter()
            .skip(1)
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(carried, ["-i", "-n", "-k", "-e", "-r", "-R"]);
    }

    /// The letters GNU Make has that this front end accepts and drops.
    ///
    /// Each reached `panic!("Unknown flag: ...")` before, which is a crash on
    /// a command line GNU Make 4.4.1 takes without comment — `-b` and `-m` it
    /// ignores itself, and the other five it honours in a runner this front
    /// end is not. Accepting them leaves every field where it was, so the
    /// assertion is that nothing moved and the argv was taken at all.
    #[test]
    fn the_letters_a_compiler_can_drop_are_taken() {
        for letter in ["-b", "-m", "-L", "-p", "-w", "-O", "-l"] {
            assert_eq!(refusal(&[letter]), "accepted", "for {letter}");
        }
        let dropped = switches(&["-b", "-m", "-L", "-p", "-w", "-O", "-l"]);
        let none = switches(&[]);
        assert_eq!(dropped.is_dry_run, none.is_dry_run);
        assert_eq!(dropped.is_silent_mode, none.is_silent_mode);
        assert_eq!(dropped.keep_going, none.keep_going);
        assert_eq!(dropped.no_builtin_rules, none.no_builtin_rules);
        assert_eq!(dropped.num_jobs, none.num_jobs);
        assert!(dropped.targets.is_empty());
    }

    /// How much of the command line a dropped switch takes with it, which is
    /// the only thing about it still observable: swallow a word too many and a
    /// goal disappears, one too few and an argument becomes a goal.
    ///
    /// GNU Make's shapes, and its own peculiarity. `-O` is `string` with a
    /// `noarg_value`, so getopt spells it `O::` and only an ATTACHED argument
    /// counts — measured on 4.4.1, `-O line zz` reports `No rule to make
    /// target 'line'`. `-l` is `floating`, and `main.c` additionally takes the
    /// following word when it opens with a digit or a `.`, so `-l 2 zz` builds
    /// `zz` alone and `-l foo` reports `No rule to make target 'foo'`.
    #[test]
    fn a_dropped_switch_takes_the_words_gnu_takes() {
        let goals = |words: &[&str]| {
            let mut symtab = Symtab::new();
            let mut argv = vec!["rkati".to_owned()];
            argv.extend(words.iter().map(|word| (*word).to_owned()));
            let flags =
                Flags::from_args(argv.into_iter().map(Into::into).collect(), &mut symtab).unwrap();
            flags
                .targets
                .iter()
                .map(|goal| String::from_utf8_lossy(&symtab.name(*goal)).into_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(goals(&["-l", "2", "zz"]), ["zz"]);
        assert_eq!(goals(&["-l", "2.5", "zz"]), ["zz"]);
        assert_eq!(goals(&["-l", "foo"]), ["foo"]);
        assert_eq!(goals(&["-O", "line", "zz"]), ["line", "zz"]);
        assert_eq!(goals(&["-Oline", "zz"]), ["zz"]);
    }

    /// A switch this front end will not honour is refused rather than
    /// accepted, and refused rather than crashed on.
    ///
    /// Silence would be the worst answer of the three: `rkati -q` that
    /// compiled the graph and reported success would be answering a question
    /// nobody asked it, and `-B` and `-t` each say what to do INSTEAD of
    /// building. The complaint is this binary's own — it is not GNU Make and
    /// does not speak for one — and the status is 2, which is what GNU Make
    /// leaves for a command line it would not take.
    #[test]
    fn a_switch_this_front_end_cannot_honour_is_refused_and_not_crashed_on() {
        for letter in ["B", "t", "q", "v", "h", "E", "I", "o", "W"] {
            assert_eq!(
                refusal(&[&format!("-{letter}")]),
                format!("rkati: unsupported option -- '{letter}'")
            );
        }
        // The letter nothing knows keeps the words it always had.
        assert_eq!(refusal(&["-Z"]), "rkati: Unknown flag: -Z");
        // And so do the three malformed-argument complaints beside it, which
        // were the other panics on this path.
        assert_eq!(refusal(&["-j", "many"]), "rkati: Invalid -j flag: many");
        assert_eq!(
            refusal(&["--remote_num_jobs", "many"]),
            "rkati: Invalid --remote_num_jobs flag: many"
        );
        assert!(
            refusal(&["--variable_assignment_trace_filter", "V"])
                .starts_with("rkati: --variable_assignment_trace_filter is valid only together")
        );
    }

    /// `-rR` is two switches, because GNU Make's getopt reads a cluster.
    ///
    /// It read as one word nothing knew, so the standalone binary crashed on a
    /// spelling the corpus under tests/make/ writes seven times. The split
    /// reaches only words that opened with a letter taking no argument, so
    /// `-j8` and `-Csub` are read exactly as before; the first letter that
    /// does take one keeps the rest of the word, which is what getopt does.
    #[test]
    fn a_cluster_of_short_switches_is_read_as_the_switches_it_is() {
        let clustered = switches(&["-rR"]);
        assert!(clustered.no_builtin_rules);
        assert!(clustered.no_builtin_variables);
        let three = switches(&["-rRs"]);
        assert!(three.no_builtin_rules && three.no_builtin_variables && three.is_silent_mode);
        // A letter that takes an argument ends the split and keeps the rest.
        assert_eq!(switches(&["-rj8"]).num_jobs, 8);
        assert!(switches(&["-rj8"]).no_builtin_rules);
        // Untouched: these never opened with a no-argument letter.
        assert_eq!(switches(&["-j8"]).num_jobs, 8);
        assert_eq!(
            switches(&["-Csub"]).working_dir,
            Some(OsString::from("sub"))
        );
        // A cluster is refused by the letter that earns it, not by the word.
        assert_eq!(refusal(&["-rZ"]), "rkati: Unknown flag: -Z");
        assert_eq!(refusal(&["-rRq"]), "rkati: unsupported option -- 'q'");
        // `-` is the byte getopt keeps, so this is not a cluster at all. GNU
        // Make 4.4.1 answers `invalid option -- '-'` and exits 2.
        assert_eq!(refusal(&["-rR-q"]), "rkati: Unknown flag: -rR-q");
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
