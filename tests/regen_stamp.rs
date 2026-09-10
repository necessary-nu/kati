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

/// Run kati as [`run`] does, with `MAKEFILES` naming a makefile as well.
///
/// `MAKEFILES` is the one entry point that reaches a read's makefile cache
/// with a name nothing globbed first, so it is how a test puts a genuinely
/// absent path into the set of files the read wanted and did not get.
fn run_with_makefiles(directory: &Path, makefiles: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_rkati"))
        .current_dir(directory)
        .args(["--ninja", "--regen", "--regen_debug"])
        .env("MAKEFILES", makefiles)
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

/// A file an `include` looked for and did not find is a file the read
/// depended on, and the stamp records it. Recording it among the files that
/// WERE read makes the check stat it, fail, and call the read dirty — so a
/// makefile with one optional include regenerates on every single run,
/// whatever the disk says. An absence is clean while it is still an absence.
#[test]
fn a_recorded_absence_alone_does_not_regenerate() {
    let said = generate_then_check(
        "absent-include",
        "-include missing.mk\nall:\n\t@echo x > out\n",
    );
    assert!(
        said.contains("No need to regenerate ninja file"),
        "an include that was absent both times regenerated: {said}"
    );
}

/// The other half of the same rule: the absence is what the read depended on,
/// so the file appearing is a change. GNU Make's `eval_makefile` opens the
/// name every time, and a run that would now read a file the recorded run did
/// not is compiling different text.
#[test]
fn an_absent_include_that_appears_regenerates() {
    let directory = scratch("absent-include-appears");
    fs::write(
        directory.join("Makefile"),
        "-include missing.mk\nall:\n\t@echo x > out\n",
    )
    .unwrap();
    run(&directory);
    let clean = run(&directory);
    assert!(
        clean.contains("No need to regenerate ninja file"),
        "an include that was absent both times regenerated: {clean}"
    );
    fs::write(directory.join("missing.mk"), "V := 1\n").unwrap();
    let appeared = run(&directory);
    assert!(
        appeared.contains("regenerating"),
        "an include that appeared did not regenerate: {appeared}"
    );
}

/// A `$(wildcard)` over a plain name that is not there answers `Err` rather
/// than an empty list, because the name has no metacharacter and the glob is
/// a `stat`. Dropping those from the stamp drops the whole dependency: the
/// file appearing changes what the makefile expanded to and nothing notices.
/// Isolated from the include case above, which records the same absence twice.
#[test]
fn an_absent_wildcard_that_appears_regenerates() {
    let directory = scratch("absent-wildcard-appears");
    fs::write(
        directory.join("Makefile"),
        "V := $(wildcard absent.txt)\nall:\n\t@echo $(V) > out\n",
    )
    .unwrap();
    run(&directory);
    let clean = run(&directory);
    assert!(
        clean.contains("No need to regenerate ninja file"),
        "a wildcard that was absent both times regenerated: {clean}"
    );
    fs::write(directory.join("absent.txt"), "here\n").unwrap();
    let appeared = run(&directory);
    assert!(
        appeared.contains("regenerating"),
        "a wildcard whose name appeared did not regenerate: {appeared}"
    );
}

/// `MAKEFILES` naming a file that is not there is forgiven by the read —
/// `read_makefiles_entry` notes it and goes on — but it is still something the
/// read depended on, so the stamp records the name. Recording it among the
/// files that WERE read makes the check stat it, get nothing, and call the
/// read dirty; and since the file is still not there, it does that on every
/// run for ever. An absence is clean while it is still an absence.
#[test]
fn an_absent_named_makefile_alone_does_not_regenerate() {
    let directory = scratch("absent-named");
    fs::write(directory.join("Makefile"), "all:\n\t@echo x > out\n").unwrap();
    run_with_makefiles(&directory, "absent.mk");
    let clean = run_with_makefiles(&directory, "absent.mk");
    assert!(
        clean.contains("No need to regenerate ninja file"),
        "a named makefile absent both times regenerated: {clean}"
    );
}

/// And the other half: the read depended on the absence, so the file arriving
/// is a change, because the next read would evaluate text this one never saw.
#[test]
fn an_absent_named_makefile_that_appears_regenerates() {
    let directory = scratch("absent-named-appears");
    fs::write(directory.join("Makefile"), "all:\n\t@echo x > out\n").unwrap();
    run_with_makefiles(&directory, "absent.mk");
    fs::write(directory.join("absent.mk"), "V := 1\n").unwrap();
    let appeared = run_with_makefiles(&directory, "absent.mk");
    assert!(
        appeared.contains("regenerating"),
        "a named makefile that appeared did not regenerate: {appeared}"
    );
}
