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

fn announcing_shell_at(directory: &Path, name: &str) {
    let path = directory.join(name);
    fs::write(&path, ANNOUNCING_SHELL).unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o755);
    fs::set_permissions(&path, permissions).unwrap();
}

/// Run one makefile through the executor and return everything it said.
fn ran(name: &str, makefile: &str) -> String {
    ran_beside(name, makefile, &[])
}

/// The same, with the files a rule needs to find already there.
fn ran_beside(name: &str, makefile: &str, present: &[&str]) -> String {
    let directory = scratch(name);
    announcing_shell_at(&directory, "own");
    announcing_shell_at(&directory, "own2");
    for file in present {
        fs::write(directory.join(file), "").unwrap();
    }
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
