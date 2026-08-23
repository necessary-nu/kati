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

//! Whether a regeneration replay reads a recorded `$(shell)` the way it was
//! recorded.
//!
//! kati records the `$(shell)` calls a `build.ninja` was generated from into a
//! `.kati_stamp`, and on the next run replays each one to ask whether it still
//! answers the same thing — a call whose answer moved means the ninja file is
//! out of date. The replay runs before any makefile has been read, so it
//! cannot ask the session whether `.ONESHELL:` was in force; a `$(shell)`
//! expanded below one was read by GNU Make's one-shell branch, and reading it
//! back as a command line can answer differently for a call that did not
//! change. The stamp carries the bit so the replay reads it the same way.
//!
//! The gate lives here rather than in the ported corpus because the corpus
//! measures build intent, and regeneration is the standalone generator's own
//! mechanism — there is no ninja file to be stale in a single build-intent run.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A directory of this test's own, emptied first so a rerun starts clean.
fn scratch(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("regen-stamp")
        .join(name);
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

/// Run kati once in `directory` with the regeneration flags, byte-identical
/// every call so the recorded arguments match on the check that follows the
/// generate.
fn run(directory: &Path) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_rkati"))
        .current_dir(directory)
        .args(["--ninja", "--regen", "--regen_debug"])
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .env_remove("CARGO_MAKEFLAGS")
        .env_remove("MAKELEVEL")
        .output()
        .unwrap();
    String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr)
}

/// Generate the ninja file and its stamp, then run the regeneration check with
/// nothing changed, and return what the check said.
fn generate_then_check(name: &str, makefile: &str) -> String {
    let directory = scratch(name);
    fs::write(directory.join("Makefile"), makefile).unwrap();
    // First run: no stamp yet, so this generates and records one.
    run(&directory);
    // Second run: identical arguments, nothing changed — the check.
    run(&directory)
}

/// A `$(shell)` below `.ONESHELL:` whose one-shell reading differs from its
/// command-line reading: the one-shell branch strips the `@` off the front and
/// runs `echo hi; echo bye`, where a command line runs `@echo` and fails. The
/// replay must read it the one-shell way it was recorded, or it reads a
/// different answer and regenerates a ninja file that is not stale.
#[test]
fn a_one_shell_shell_replays_as_a_script_and_does_not_regenerate() {
    let said = generate_then_check(
        "oneshell",
        ".ONESHELL:\nV := $(shell @echo hi; echo bye)\nall:\n\t@echo x > out\n",
    );
    assert!(
        said.contains("No need to regenerate ninja file"),
        "a one-shell $(shell) that did not change forced a regeneration: {said}"
    );
}

/// The control the bit must not disturb: a `$(shell)` with no `.ONESHELL:` was
/// recorded as a command line and replays as one, and its stamp bytes are
/// exactly what a kati that never knew the bit wrote — an old stamp, read
/// safely as not-one-script.
#[test]
fn a_plain_shell_still_does_not_regenerate() {
    let said = generate_then_check(
        "plain",
        "V := $(shell @echo hi; echo bye)\nall:\n\t@echo x > out\n",
    );
    // A command line runs `@echo` and fails, so the recording and the replay
    // agree on the same (empty-ish) answer either way; what matters is that the
    // reading did not move between them.
    assert!(
        said.contains("No need to regenerate ninja file"),
        "a plain $(shell) that did not change forced a regeneration: {said}"
    );
}

/// The replay still catches a real change: a `$(shell)` reading a file the
/// makefile does not name regenerates when that file's contents move, and the
/// makefile itself is left untouched so the file-timestamp check cannot be
/// what fired.
#[test]
fn a_one_shell_shell_regenerates_when_its_answer_moves() {
    let directory = scratch("oneshell-moves");
    fs::write(
        directory.join("Makefile"),
        ".ONESHELL:\nV := $(shell @cat data.txt; true)\nall:\n\t@echo x > out\n",
    )
    .unwrap();
    fs::write(directory.join("data.txt"), "one\n").unwrap();
    run(&directory); // generate
    let clean = run(&directory); // nothing changed
    assert!(
        clean.contains("No need to regenerate ninja file"),
        "the recorded answer was not reproduced on replay: {clean}"
    );
    // Move only the file the shell reads, not the makefile.
    fs::write(directory.join("data.txt"), "two\n").unwrap();
    let moved = run(&directory);
    assert!(
        moved.contains("was changed, regenerating"),
        "a changed $(shell) answer did not regenerate: {moved}"
    );
}
