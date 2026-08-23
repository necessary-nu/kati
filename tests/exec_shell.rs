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

//! Which shell rkati's own executor starts a recipe with.
//!
//! GNU Make expands `$(SHELL)` once per recipe it is about to run, with that
//! recipe's target in front of it — `construct_command_argv` (job.c) calls
//! `allocated_variable_expand_for_file ("$(SHELL)", file)` — so a `SHELL` one
//! target sets governs that target's launches and descends to the
//! prerequisites built for it.
//!
//! The gate lives here rather than in the ported corpus because the corpus
//! runs Ronin, and this executor is the other tool: the standalone binary is
//! what the conformance corpus measures, so a cell that diverges here is one
//! the corpus cannot see the truth of. The recipes are all `echo`, and what is
//! asserted is which program was asked to run them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A shell that says its own name and arguments before handing the script to a
/// real one, so a recipe's launch can be read off the output.
const ANNOUNCING_SHELL: &str =
    "#!/bin/sh\necho \"OWN[$(basename \"$0\")] $*\" >&2\nexec /bin/sh \"$@\"\n";

/// A shell that says every argument it was handed, one to a line, and runs
/// nothing at all.
///
/// Which words a launch was split into cannot be read off [`ANNOUNCING_SHELL`]:
/// it prints `$*`, and one argument holding a blank and two arguments come out
/// as the same line. This one brackets each argument on its own line, so
/// `.SHELLFLAGS := -e -c` arriving as one word and as two are different
/// outputs.
const COUNTING_SHELL: &str =
    "#!/bin/sh\nfor word in \"$@\"; do\n  echo \"WORD[$word]\" >&2\ndone\n";

/// A directory of this test's own, emptied first so a rerun starts where the
/// first run did.
fn scratch(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("exec-shell")
        .join(name);
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn shell_at(directory: &Path, name: &str, text: &str) {
    let path = directory.join(name);
    fs::write(&path, text).unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
    fs::set_permissions(&path, permissions).unwrap();
}

/// Run one makefile through the executor and return everything it said.
fn ran(name: &str, makefile: &str) -> String {
    ran_beside(name, makefile, &[])
}

/// The same, with the files a rule needs to find already there.
///
/// `{here}` in the makefile is the run's own directory, spelled absolutely.
/// Which branch a launch takes turns on whether the shell is a plain path, so
/// a shell named `./own` and the same shell named `/tmp/.../own` are two
/// different measurements and the tests need both spellings.
fn ran_beside(name: &str, makefile: &str, present: &[&str]) -> String {
    let directory = scratch(name);
    shell_at(&directory, "own", ANNOUNCING_SHELL);
    shell_at(&directory, "own2", ANNOUNCING_SHELL);
    shell_at(&directory, "count", COUNTING_SHELL);
    for file in present {
        fs::write(directory.join(file), "").unwrap();
    }
    let makefile = makefile.replace("{here}", &directory.display().to_string());
    fs::write(directory.join("Makefile"), makefile).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rkati"))
        .current_dir(&directory)
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .env_remove("CARGO_MAKEFLAGS")
        .env_remove("MAKELEVEL")
        .output()
        .unwrap();
    String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr)
}

/// Every launch the run announced, as the shell's name and its arguments.
fn launches(said: &str) -> Vec<&str> {
    said.lines()
        .filter_map(|line| line.strip_prefix("OWN["))
        .collect()
}

/// Every argument [`COUNTING_SHELL`] was handed, in order and one to an entry.
fn words(said: &str) -> Vec<&str> {
    said.lines()
        .filter_map(|line| line.strip_prefix("WORD[")?.strip_suffix(']'))
        .collect()
}

/// A `SHELL` a target set is the shell that target's own recipe runs under,
/// and the global one is what every other target keeps.
///
/// The executor held a single shell for the whole build and read it with no
/// scope in front of it, so the value the evaluation had just worked out for
/// this node — sitting on the evaluator it had called one line above — was
/// dropped and the global started instead.
#[test]
fn a_targets_own_shell_starts_its_recipe() {
    let said = ran("own-shell", "all: SHELL := ./own\nall:\n\t@echo one\n");
    assert_eq!(
        launches(&said),
        vec!["own] -c echo one"],
        "the target's own shell did not run its recipe: {said}"
    );
}

/// The same descent every target-specific variable makes: a prerequisite built
/// for that target inherits it, and `private` is the word that stops it.
#[test]
fn a_targets_own_shell_descends_to_its_prerequisites() {
    let descends = ran(
        "own-shell-descends",
        "all: SHELL := ./own\nall: dep\n\t@echo one\ndep:\n\t@echo two\n",
    );
    assert_eq!(
        launches(&descends),
        vec!["own] -c echo two", "own] -c echo one"],
        "the prerequisite did not inherit the shell: {descends}"
    );

    let private = ran(
        "own-shell-private",
        "all: private SHELL := ./own\nall: dep\n\t@echo one\ndep:\n\t@echo two\n",
    );
    assert_eq!(
        launches(&private),
        vec!["own] -c echo one"],
        "a private shell reached the prerequisite: {private}"
    );
}

/// A pattern rule's own `SHELL`, and the target the pattern did not match as
/// the control beside it.
#[test]
fn a_pattern_rules_own_shell_starts_what_it_makes() {
    let said = ran_beside(
        "own-shell-pattern",
        "%.o: SHELL := ./own\n%.o: %.c\n\t@echo make-$@\nall: x.o\n\t@echo top\n",
        &["x.c"],
    );
    assert_eq!(
        launches(&said),
        vec!["own] -c echo make-x.o"],
        "the pattern rule's shell did not run its recipe, or reached the goal: {said}"
    );
}

/// Two assignments for one target: the last one written is the value, exactly
/// as it is for any other target-specific variable.
#[test]
fn the_last_shell_written_for_a_target_wins() {
    let said = ran(
        "own-shell-last",
        "all: SHELL := ./own\nall: SHELL := ./own2\nall:\n\t@echo one\n",
    );
    assert_eq!(
        launches(&said),
        vec!["own2] -c echo one"],
        "the second assignment did not win: {said}"
    );
}

/// `.SHELLFLAGS` is the other half of what starting a shell needs, and it has
/// been read with the target's scope in front of it all along. The gate is
/// that the two now arrive together.
#[test]
fn a_targets_own_shell_flags_arrive_with_it() {
    let said = ran(
        "own-shell-flags",
        "all: SHELL := ./own\nall: .SHELLFLAGS := -xc\nall:\n\t@echo one\n",
    );
    assert_eq!(
        launches(&said),
        vec!["own] -xc echo one"],
        "the target's own shell flags did not reach the launch: {said}"
    );
}

/// A `SHELL` whose own value has to start a shell to expand is refused when a
/// recipe needs one and never asked for when none does — the laziness the read
/// carries, kept while the read moved.
#[test]
fn a_shell_no_recipe_needs_is_unasked() {
    let refused = ran(
        "own-shell-recursive",
        "SHELL = $(shell echo /bin/sh)\nall:\n\techo one\n",
    );
    assert!(
        refused.contains("references itself"),
        "a recursive shell a recipe needed was not refused: {refused}"
    );

    let unasked = ran(
        "own-shell-unneeded",
        "SHELL = $(shell echo /bin/sh)\n.PHONY: all\nall:\n",
    );
    assert!(
        !unasked.contains("references itself"),
        "a recursive shell no recipe needed was asked for: {unasked}"
    );
}

/// `.ONESHELL` says a recipe's lines are ONE script handed to one shell, and
/// the executor has to say it too.
///
/// Every reader of the flag used to be on the compiling side: the separator
/// that joins the lines, whether the script arms `errexit`, whether a line
/// needs a subshell, and what the shell is called. The executor beside it ran
/// a shell per line, so a variable set on one line was gone by the next, a
/// `cd` did not persist, and the first line's status became the recipe's.
///
/// Each case below is GNU Make 4.4.1's answer, measured on the same makefile.
/// A variable is the plainest of the three: the second line reads what the
/// first set, which one shell can do and two cannot.
#[test]
fn one_shell_keeps_a_variable_set() {
    let said = ran(
        "one-shell-variable",
        ".ONESHELL:\nall:\n\tx=1\n\t@echo \"held $$x\"\n",
    );
    // Read off a whole line rather than by substring: the recipe echo prints
    // the script before it runs, so every word of the makefile is in this
    // output already and only what the shell answered is new.
    assert!(
        said.lines().any(|line| line == "held 1"),
        "the variable did not survive the line that set it: {said}"
    );
}

/// The same about the working directory, which is the case a makefile is most
/// likely to be relying on: `cd` and then work, with no `&&` between them.
#[test]
fn one_shell_keeps_a_directory_change() {
    let said = ran(
        "one-shell-directory",
        ".ONESHELL:\nall:\n\tmkdir -p sub\n\tcd sub\n\t@pwd\n",
    );
    assert!(
        said.lines().any(|line| line.ends_with("/sub")),
        "the directory change did not survive the line that made it: {said}"
    );
}

/// One script has one status, and it is the last line's. A failing line in the
/// middle of a script the shell was not told to stop for is not the recipe's
/// answer — where a shell per line makes it the first failure's.
#[test]
fn one_shell_reports_its_last_status() {
    let said = ran(
        "one-shell-status",
        ".ONESHELL:\nall:\n\tfalse\n\t@echo went-on\n",
    );
    assert!(
        said.lines().any(|line| line == "went-on"),
        "the recipe stopped at a line the script would have run past: {said}"
    );
    assert!(
        !said.contains("Error 1"),
        "a line's status was read as the recipe's: {said}"
    );
}

/// The count itself, read off the shell rather than off what it did: three
/// lines, one launch. This is the cell the other three are consequences of.
#[test]
fn one_shell_starts_one_shell() {
    let said = ran(
        "one-shell-count",
        "SHELL := ./own\n.ONESHELL:\nall:\n\t@echo one\n\techo two\n\techo three\n",
    );
    // `launches` reads one line each, and this launch's argument holds two
    // newlines — which is the whole claim, so the count is what is asserted
    // and the first line names the script it starts with.
    assert_eq!(
        launches(&said),
        vec!["own] -c echo one"],
        "the recipe was not one launch: {said}"
    );
    assert!(
        said.contains("OWN[own] -c echo one\necho two\necho three\n"),
        "the one launch was not handed the whole recipe: {said}"
    );
}

/// A `-` below the first line is script text and not a flag: `chop_commands`
/// never chopped the recipe, so the scan that fills `lines_flags` saw only its
/// first line. The recipe therefore runs to its end and reports zero, because
/// the failing line is neither forgiven nor fatal — it is just a command in the
/// middle of a script.
#[test]
fn a_dash_below_the_first_line() {
    let said = ran(
        "one-shell-dash-below",
        ".ONESHELL:\nall:\n\ttrue\n\t-false\n\t@echo went-on\n",
    );
    assert!(
        said.lines().any(|line| line == "went-on"),
        "the recipe stopped at an interior line: {said}"
    );
    assert!(
        !said.contains("Error"),
        "an interior line's status ended the recipe: {said}"
    );
}

/// `.POSIX:` is the other direction and the control for the case above: the
/// one shell is started with `-e`, so the script really does stop at the
/// failing line, and the recipe reports it. One script, and the shell's own
/// setting decides — not a status the executor read between two launches.
#[test]
fn a_posix_one_shell_recipe_stops() {
    let said = ran(
        "one-shell-posix",
        ".POSIX:\n.ONESHELL:\nall:\n\tfalse\n\t@echo went-on\n",
    );
    assert!(
        !said.lines().any(|line| line == "went-on"),
        "the strict script ran past the line that failed: {said}"
    );
    assert!(
        said.contains("Error 1"),
        "the strict script's failure was not the recipe's: {said}"
    );
}

/// `.SHELLFLAGS` holding more than one word is more than one argument.
///
/// GNU Make 4.4.1 hands `/bin/sh` the flags `-e` and `-c` for this makefile and
/// the recipe runs; the executor handed over the single word `-e -c`, which
/// `dash` answers with `Illegal option -` before running anything at all. It is
/// not `.ONESHELL`'s — measured the same with and without one.
#[test]
fn multi_word_shell_flags_are_words() {
    let said = ran(
        "flag-words",
        "SHELL := {here}/count\n.SHELLFLAGS := -e -c\nall:\n\t@echo one\n",
    );
    assert_eq!(
        words(&said),
        vec!["-e", "-c", "echo one"],
        "the flags did not reach the launch as words: {said}"
    );
}

/// The same for the launch a `.ONESHELL` recipe makes, which GNU Make splits
/// with the shell's own tokenizer rather than on blanks: a quoted flag with a
/// space in it is one word and the quotes come off.
#[test]
fn a_one_shell_launchs_flags_are_read_with_the_shells_own_quoting() {
    let said = ran(
        "flag-words-one-shell",
        ".ONESHELL:\nSHELL := {here}/count\n.SHELLFLAGS := -e 'a b' -c\nall:\n\t@echo one\n",
    );
    assert_eq!(
        words(&said),
        vec!["-e", "a b", "-c", "echo one"],
        "the one-shell launch's flags were not read with the shell's quoting: {said}"
    );
}

/// An empty `.SHELLFLAGS` is no argument rather than an empty one.
///
/// GNU Make assembles the command line with a run of blanks where the flags
/// would have been and then splits it, and a run of blanks is not a word — so
/// the script arrives as the shell's first argument, which a Bourne shell reads
/// as a FILE to run. The executor passed an empty word ahead of it, so the
/// shell was asked to open `` rather than the script.
#[test]
fn blank_shell_flags_are_no_word_at_all() {
    let said = ran(
        "flag-words-blank",
        "SHELL := {here}/count\n.SHELLFLAGS :=\nall:\n\t@echo one\n",
    );
    assert_eq!(
        words(&said),
        vec!["echo one"],
        "an empty flags value became an argument: {said}"
    );
}

/// `$(shell)` asks the same question, and it is the reason this reaches Ronin
/// too: the compiled path splits a recipe's flags itself, but a `$(shell)` is
/// run by this executor whoever the front end is.
#[test]
fn multi_word_shell_flags_reach_a_shell_function() {
    let said = ran(
        "flag-words-shell-function",
        ".SHELLFLAGS := -e -c\nX := $(shell echo hi)\nall:\n\t@echo [$(X)]\n",
    );
    assert!(
        said.lines().any(|line| line == "[hi]"),
        "the flags did not reach the shell function's launch as words: {said}"
    );
}

/// A `.ONESHELL` recipe's `SHELL` is a program and not a command line.
///
/// GNU Make's one-shell branch execs the shell exactly as spelled, with the
/// flags and the script after it, and never assembles a command line for
/// another shell to read — so a `SHELL` of more than one word names a program
/// that is not there, and 4.4.1 says so against the whole of it and stops with
/// `Error 127`. The executor handed the spelling to `/bin/sh`, which split it
/// and ran it.
#[test]
fn a_one_shell_recipes_shell_is_a_program() {
    let said = ran(
        "one-shell-shell-is-a-program",
        ".ONESHELL:\nSHELL := {here}/count -q\nall:\n\t@echo one\n\t@echo two\n",
    );
    assert!(
        words(&said).is_empty(),
        "the multi-word shell was started anyway: {said}"
    );
    assert!(
        said.contains("count -q: No such file or directory"),
        "the failure was not reported against the whole spelling: {said}"
    );
    assert!(
        said.contains("Error 127"),
        "the recipe did not stop with the status of a command that could not run: {said}"
    );
}
