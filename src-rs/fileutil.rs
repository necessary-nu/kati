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

use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::ExitStatusExt as _;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, OsStr},
    path::Path,
    process::ExitStatus,
    slice,
    sync::Arc,
    time::SystemTime,
};

use anyhow::Result;
use bytes::{BufMut, Bytes, BytesMut};
use memchr::memchr2;
use parking_lot::Mutex;

use crate::log;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedirectStderr {
    None,
    Stdout,
    DevNull,
}

/// When `filename` was last written, or `None` when it is not there.
///
/// One `stat` rather than an `exists` and then a `metadata`, and a failure
/// that is not "no such file" names the path it happened to: a bare
/// `io::Error` reaching a caller from here says only what went wrong, never
/// which of a build's files it went wrong on.
pub fn get_timestamp(filename: &[u8]) -> Result<Option<SystemTime>> {
    let filename = <OsStr as OsStrExt>::from_bytes(filename);
    let metadata = match std::fs::metadata(filename) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(crate::io_failure(Path::new(filename), &err)),
    };
    metadata
        .modified()
        .map(Some)
        .map_err(|err| crate::io_failure(Path::new(filename), &err))
}

/// POSIX's exit status for a command that could not be run at all.
const COMMAND_NOT_FOUND: i32 = 127;

/// What the C library would have said about this error, which is the wording
/// GNU Make reports because it reports `strerror (errno)`.
fn system_message(error: &std::io::Error) -> String {
    match error.raw_os_error() {
        Some(code) => std::io::Error::from_raw_os_error(code).to_string(),
        None => error.to_string(),
    }
    .split(" (os error")
    .next()
    .unwrap_or_default()
    .to_owned()
}

/// Which shell reads a command, in the three parts that always travel
/// together: `$(SHELL)`, the `$(.SHELLFLAGS)` it takes ahead of the command,
/// and the executable standing in for the default shell where the build
/// declared one.
#[derive(Clone, Copy)]
pub struct ShellToReadWith<'a> {
    pub program: &'a [u8],
    pub flag: &'a [u8],
    pub stand_in: Option<&'a Path>,
    /// Whether the text handed over is one SCRIPT, whose newlines separate the
    /// commands in it, rather than one command line.
    ///
    /// `.ONESHELL:` is the whole of what makes one, and it makes one of a
    /// `$(shell)` expanded below it as readily as of a recipe: GNU Make's
    /// `one_shell` is a GLOBAL rather than a parameter, and `func_shell_base`
    /// reaches `construct_command_argv` like every other launch. It has to be
    /// said here because the direct-exec fast path below reads the text as one
    /// command's words — to which a newline is a blank like any other, so a
    /// script would be exec'd as a single argument list holding every line's
    /// words. GNU Make asks at the same point and for the same reason:
    /// `construct_command_argv_internal` (job.c:3034) has `else if (one_shell
    /// && *p == '\n') goto slow`, "in .ONESHELL mode \n is a separator like ;
    /// or &&".
    ///
    /// The bytes it carries are the flags GNU Make's own recursion defaults to
    /// while it parses `.SHELLFLAGS` for such a launch — read only where that
    /// parse has to hand the flags back to a shell, which is the one case a
    /// one-script launch passes words that did not come out of the flags text.
    /// See [`crate::simple_command::shell_flag_argv`].
    pub one_script: Option<&'a [u8]>,
}

/// The argument list a launch execs, or `None` where a shell has to read the
/// whole thing as a command line.
///
/// GNU Make asks this in `construct_command_argv_internal` (job.c) and gets
/// three different answers, and so does this:
///
///   * a line with no shell syntax in it is exec'd with no shell in between at
///     all — so a program that is not there is reported against its own name
///     and by whoever went looking, rather than in the words of a shell that
///     was never needed;
///   * a one-script launch is the shell, its flags, and the whole script as
///     one argument. GNU Make's one-shell branch builds exactly that and never
///     falls back to a command line, so a `SHELL` of more than one word is a
///     program of more than one word that nothing can start — which is the
///     answer 4.4.1 gives, `No such file or directory` against the whole of it.
///     The script also loses the blanks and `[@+-]` off the front of each of
///     its lines, for a Bourne-compatible shell and no other;
///   * every other launch is the shell, its flags, and the line as one
///     argument, whenever the flags can be split here. GNU Make reaches that
///     by assembling `$(SHELL) $(.SHELLFLAGS) LINE` and re-tokenizing all of
///     it; where the tokenizer refuses, the text really does go to a shell and
///     `None` says so.
///
/// The two flag splittings are different functions because GNU Make has two:
/// see [`crate::simple_command::shell_flag_argv`] and
/// [`crate::simple_command::command_line_flag_argv`].
fn argv_to_exec(
    shell: &[u8],
    shellflag: &[u8],
    cmd: &Bytes,
    one_script: Option<&[u8]>,
) -> Option<Vec<Bytes>> {
    if let Some(direct) =
        crate::simple_command::direct_argv(cmd, shell, shellflag, one_script.is_some())
    {
        return Some(direct);
    }
    let (flags, line) = if let Some(default_flags) = one_script {
        (
            crate::simple_command::shell_flag_argv(shellflag, default_flags),
            // Only the shell's errand loses the prefixes. GNU Make's one-shell
            // branch is on the far side of `goto slow`, so a script the
            // tokenizer took apart itself is exec'd with its `[@+-]` intact —
            // measured on 4.4.1, where `.ONESHELL:` and `V := $(shell @echo
            // hi)` report `@echo: No such file or directory` and the same text
            // with a `>` in it prints `hi`.
            if crate::command::is_bourne_compatible_shell(shell) {
                crate::simple_command::one_shell_prefixes_stripped(cmd)
            } else {
                cmd.clone()
            },
        )
    } else {
        // A shell that is a command line rather than a program cannot be
        // exec'd, and neither can flags with shell syntax in them.
        if !shell.starts_with(b"/") || memchr2(b' ', b'$', shell).is_some() {
            return None;
        }
        (
            crate::simple_command::command_line_flag_argv(shellflag)?,
            cmd.clone(),
        )
    };
    let mut argv = Vec::with_capacity(flags.len() + 2);
    argv.push(Bytes::copy_from_slice(shell));
    argv.extend(flags);
    argv.push(line);
    Some(argv)
}

/// Run one command and read back what it wrote.
///
/// `environment` is what Make's export set says about the child: a name bound
/// to `Some` bytes is set, and one bound to `None` is removed from what this
/// process would otherwise pass on. It is a delta rather than a whole
/// environment because that is what Make computes — a variable the makefile
/// never touched reaches the child as the bytes this process was started with,
/// unexpanded.
///
/// `interrupts` is what the host has observed about the user stopping the read.
/// Given one, the wait for this command ends when an interrupt arrives, and the
/// child is ABANDONED where it stands — not signalled, not reaped — which is
/// what GNU Make 4.4.1 leaves behind when its `fatal_error_signal` re-raises
/// past a `$(shell)` it knows nothing about. See [`crate::interrupt`].
// [spec:ronin:req:make.read-interrupt]
pub fn run_command(
    shell: ShellToReadWith<'_>,
    cmd: &Bytes,
    environment: &[(Bytes, Option<Bytes>)],
    redirect_stderr: RedirectStderr,
    diagnostic_prefix: &str,
    diagnostics: &crate::diagnostics::Diagnostics,
    interrupts: Option<&dyn crate::interrupt::Interruptible>,
) -> Result<(ExitStatus, Vec<u8>)> {
    let ShellToReadWith {
        program: shell,
        flag: shellflag,
        stand_in: default_shell_program,
        one_script,
    } = shell;
    let words = argv_to_exec(shell, shellflag, cmd, one_script);
    let mut cmd_with_shell;
    let owned;
    let args: &[&OsStr] = if let Some(words) = &words {
        owned = words
            .iter()
            .map(|word| <OsStr as OsStrExt>::from_bytes(word))
            .collect::<Vec<_>>();
        &owned
    } else {
        // GNU Make's `goto slow` for a command line nothing here can take
        // apart: the text is assembled and one shell reads all of it, which is
        // also what splits `$(.SHELLFLAGS)` into words for this launch.
        //
        // The shell is escaped into the line and the flags are not, which is
        // GNU Make's own asymmetry: it copies `$(.SHELLFLAGS)` in as it stands
        // and walks `$(SHELL)` character by character. See
        // [`crate::simple_command::escaped_shell_name`].
        let cmd_escaped = crate::strutil::escape_shell(cmd);
        cmd_with_shell = BytesMut::new();
        cmd_with_shell.put_slice(&crate::simple_command::escaped_shell_name(shell));
        cmd_with_shell.put_u8(b' ');
        cmd_with_shell.put_slice(shellflag);
        cmd_with_shell.put_slice(b" \"");
        cmd_with_shell.put_slice(&cmd_escaped);
        cmd_with_shell.put_u8(b'\"');
        &[
            <OsStr as OsStrExt>::from_bytes(b"/bin/sh"),
            <OsStr as OsStrExt>::from_bytes(b"-c"),
            <OsStr as OsStrExt>::from_bytes(&cmd_with_shell),
        ]
    };

    log!("run_command({args:?})");

    // The shell is in program position here, so the build's own shell can
    // stand in for the default one — for the `$(SHELL)` a plain call names and
    // for the `/bin/sh` wrapping a complicated one alike.
    let mut cmd = crate::simple_command::shell_process(args[0], default_shell_program);
    cmd.args(&args[1..]);
    for (name, value) in environment {
        let name = <OsStr as OsStrExt>::from_bytes(name);
        match value {
            Some(value) => cmd.env(name, <OsStr as OsStrExt>::from_bytes(value)),
            None => cmd.env_remove(name),
        };
    }

    // Nothing further is launched once the read has been stopped. The wait
    // below is what the interrupt was measured against, and this is the promise
    // beside it: an interrupt that arrived between two commands means the
    // second one does not run at all.
    if crate::interrupt::stopped(interrupts) {
        return Err(crate::interrupt::Interrupted.into());
    }

    let (mut reader, writer) = os_pipe::pipe()?;
    match redirect_stderr {
        RedirectStderr::None => {
            cmd.stderr(std::process::Stdio::inherit());
        }
        RedirectStderr::Stdout => {
            cmd.stderr(writer.try_clone()?);
        }
        RedirectStderr::DevNull => {
            cmd.stderr(std::process::Stdio::null());
        }
    }
    cmd.stdout(writer);

    let mut handle = match cmd.spawn() {
        Ok(handle) => handle,
        // Only a launch that execs answers for this: with a shell in the way
        // the shell is what failed to find the program, and it says so itself.
        Err(error) if words.is_some() => {
            let name = String::from_utf8_lossy(args[0].as_bytes()).into_owned();
            diagnostics.write_line(&format!(
                "{diagnostic_prefix}{name}: {}",
                system_message(&error)
            ));
            // POSIX's status for a command that could not be run, which is what
            // GNU Make's child reports for the same failure.
            return Ok((ExitStatus::from_raw(COMMAND_NOT_FOUND << 8), Vec::new()));
        }
        Err(error) => return Err(error.into()),
    };
    // Drop the cmd, otherwise the pipe will be retained.
    drop(cmd);

    let mut output = Vec::new();
    let reading = crate::interrupt::read_to_end_or_abandon(&mut reader, &mut output, interrupts)?;
    if matches!(reading, crate::interrupt::Reading::Interrupted) {
        // Dropping a `Child` neither waits nor kills, so this leaves the
        // command exactly where GNU Make leaves it: running, unsignalled, and
        // about to be the init process's to bury when this one exits.
        drop(handle);
        return Err(crate::interrupt::Interrupted.into());
    }

    let Some(res) = crate::interrupt::wait_or_abandon(&mut handle, interrupts)? else {
        // A command that closed its output early is waited for here rather than
        // above, and is abandoned in the same way and for the same reason.
        drop(handle);
        return Err(crate::interrupt::Interrupted.into());
    };

    Ok((res, output))
}

pub type GlobResults = Arc<Result<Vec<Bytes>, std::io::Error>>;

/// One pattern's answers, and the epoch the latest of them was read in.
struct GlobEntry {
    /// The epoch [`GlobCache::current`] was read in. An entry read in an
    /// earlier epoch has to be read again before it can be believed.
    epoch: u64,
    current: GlobResults,
    /// What the pattern answered the first time the session asked it, kept for
    /// the regeneration stamp: a check runs before any of the makefile's own
    /// commands do, so the answer it can compare against is the one the read
    /// started from rather than the one a command left behind.
    first: GlobResults,
}

/// Glob results memoised for one session. Owned by [`crate::session::Session`].
///
/// GNU Make caches directory contents the same way, in `dir.c`, and keeps that
/// cache honest with a counter: `find_directory` believes what it read only
/// while `command_count` still holds the value it read at, and every command
/// Make runs bumps that counter. A makefile can only change the filesystem by
/// running a command — `$(shell)`, `$(file >)`, or a recipe — so the cache is
/// invisible to a makefile even though it saves the reads.
///
/// [`invalidate`](Self::invalidate) is that counter, and the epoch on an entry
/// is the counter value it was read at.
// [spec:ronin:req:make.no-ambient-state]
#[derive(Default)]
pub struct GlobCache {
    inner: Mutex<GlobCacheInner>,
}

#[derive(Default)]
struct GlobCacheInner {
    epoch: u64,
    entries: HashMap<Bytes, GlobEntry>,
}

impl GlobCache {
    /// Glob `pat`, reading the filesystem only when nothing has run since the
    /// last time this pattern was read.
    pub fn glob(&self, pat: Bytes) -> GlobResults {
        let mut inner = self.inner.lock();
        let epoch = inner.epoch;
        if let Some(entry) = inner.entries.get(&pat)
            && entry.epoch == epoch
        {
            return entry.current.clone();
        }
        let glob = Arc::new(
            if pat.contains(&b'?')
                || pat.contains(&b'*')
                || pat.contains(&b'[')
                || pat.contains(&b'\\')
            {
                libc_glob(&pat)
            } else if let Err(err) = std::fs::metadata(<OsStr as OsStrExt>::from_bytes(&pat)) {
                Err(err)
            } else {
                Ok(vec![pat.clone()])
            },
        );
        match inner.entries.get_mut(&pat) {
            Some(entry) => {
                entry.epoch = epoch;
                entry.current = glob.clone();
            }
            None => {
                inner.entries.insert(
                    pat,
                    GlobEntry {
                        epoch,
                        current: glob.clone(),
                        first: glob.clone(),
                    },
                );
            }
        }
        glob
    }

    /// Note that a command ran, so anything read before it is now hearsay.
    ///
    /// This is GNU Make's `++command_count`: the whole cache ages at once
    /// rather than the one directory the command is guessed to have touched,
    /// because a command can touch anything.
    pub fn invalidate(&self) {
        self.inner.lock().epoch += 1;
    }

    /// The epoch every entry read now would be stamped with.
    ///
    /// Published so that a cache of the filesystem which is NOT this one can
    /// be kept honest by the same counter rather than by a second one of its
    /// own. [`crate::dircache::DirectoryCache`] is the other, and two counters
    /// would be two things to remember to bump.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.inner.lock().epoch
    }

    /// Forget everything, including what was recorded for the stamp.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.epoch = 0;
    }

    /// Whether the session has globbed anything yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Every pattern the session globbed, with the answer it first gave, for
    /// the regeneration stamp to check a later filesystem against.
    #[must_use]
    pub fn recorded(&self) -> Vec<(Bytes, GlobResults)> {
        self.inner
            .lock()
            .entries
            .iter()
            .map(|(pat, entry)| (pat.clone(), entry.first.clone()))
            .collect()
    }
}

// Use libc glob over the `glob` crate, to maintain compatibility.
// The glob crate ends up normalizing the paths too much:
//   ./src/*_test.cc -> src/find_test.cc
// This breaks makefiles that do further string manipulation.
fn libc_glob(pattern: &[u8]) -> Result<Vec<Bytes>, std::io::Error> {
    let pat = CString::new(pattern).unwrap();
    let mut ret = Vec::new();
    // SAFETY: All of the types in glob_t are safe to be zero'd.
    let mut gl: libc::glob_t = unsafe { std::mem::zeroed() };
    // SAFETY: gl has been zero'd above, and pat is used as an input.
    // We'll free any allocated memory with globfree below.
    let r = unsafe { libc::glob(pat.as_ptr(), 0, None, &mut gl) };
    if r == 0 && gl.gl_pathc > 0 && !gl.gl_pathv.is_null() {
        // SAFETY: We've verified that glob succeeded, and the
        // gl_pathv is not null.
        //
        // We assume that the pointers are properly aligned.
        //
        // We can't guarantee that these came from the same allocated
        // object, but this is also only temporary, and will not be
        // used past the globfree which will deallocate any memory.
        let paths = unsafe { slice::from_raw_parts(gl.gl_pathv, gl.gl_pathc) };
        ret.reserve_exact(gl.gl_pathc);
        for ptr in paths {
            if !ptr.is_null() {
                // SAFETY: This is a non-null pointer, and we assume
                // glob created valid C strings. We're immediately
                // copying out of this string, so mutability and
                // lifetimes aren't issues.
                let s = unsafe { CStr::from_ptr(*ptr) };
                ret.push(Bytes::from(s.to_bytes().to_owned()));
            }
        }
    }
    // SAFETY: we're no longer using anything from gl, and this will
    // only free things allocated by libc::glob.
    unsafe { libc::globfree(&mut gl) };
    Ok(ret)
}

pub fn fnmatch(pattern: &CString, string: &[u8], flags: i32) -> bool {
    let string = CString::new(string).unwrap();
    // SAFETY: This is a relatively simple C func, both CStrings are inputs
    // and only need to last through the function call.
    unsafe { libc::fnmatch(pattern.as_ptr(), string.as_ptr(), flags) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own, so one test's files are never another's
    /// glob results. Removed again when the test that made it ends. Named for
    /// the test rather than counted, because a counter would be exactly the
    /// process-wide mutable state this crate keeps on the session instead.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("kati-glob-cache-{}-{test}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("a scratch directory");
            Self(path)
        }

        fn pattern(&self, of: &str) -> Bytes {
            Bytes::from(self.0.join(of).into_os_string().into_encoded_bytes())
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn names(results: &GlobResults) -> Vec<Bytes> {
        results
            .as_ref()
            .as_ref()
            .expect("a readable directory")
            .clone()
    }

    /// A pattern is answered from what was already read until something runs,
    /// and read again once something has. GNU Make's `dir.c` is the same rule
    /// against `command_count`, and it is what makes a `$(shell)` between two
    /// wildcards enough for the second to find what the first could not.
    #[test]
    fn a_pattern_is_reread_only_after_a_command() {
        let scratch = Scratch::new("reread-after-command");
        let pattern = scratch.pattern("*.probe");
        let cache = GlobCache::default();
        assert!(names(&cache.glob(pattern.clone())).is_empty());

        let made = scratch.0.join("made.probe");
        std::fs::write(&made, "x").expect("writing the probe");
        assert!(
            names(&cache.glob(pattern.clone())).is_empty(),
            "nothing has run, so what was read still stands"
        );

        cache.invalidate();
        assert_eq!(
            names(&cache.glob(pattern.clone())),
            vec![scratch.pattern("made.probe")]
        );

        std::fs::remove_file(&made).expect("removing the probe");
        cache.invalidate();
        assert!(names(&cache.glob(pattern)).is_empty());
    }

    /// A name that is not a pattern ages the same way, so `$(wildcard f)` and
    /// `$(wildcard *)` answer the same filesystem as each other.
    #[test]
    fn a_plain_name_is_reread_after_a_command_too() {
        let scratch = Scratch::new("plain-name");
        let name = scratch.pattern("named");
        let cache = GlobCache::default();
        assert!(cache.glob(name.clone()).is_err());

        std::fs::write(scratch.0.join("named"), "x").expect("writing the file");
        cache.invalidate();
        assert_eq!(names(&cache.glob(name)), vec![scratch.pattern("named")]);
    }

    /// The regeneration stamp records the answer the read started from rather
    /// than the one a command left behind, because the check that reads the
    /// stamp back runs before any of the makefile's commands do.
    #[test]
    fn the_stamp_records_the_first_answer() {
        let scratch = Scratch::new("first-answer");
        let pattern = scratch.pattern("*.probe");
        let cache = GlobCache::default();
        assert!(cache.is_empty());
        cache.glob(pattern.clone());
        assert!(!cache.is_empty());

        std::fs::write(scratch.0.join("made.probe"), "x").expect("writing the probe");
        cache.invalidate();
        assert_eq!(names(&cache.glob(pattern.clone())).len(), 1);

        let recorded = cache.recorded();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, pattern);
        assert!(
            names(&recorded[0].1).is_empty(),
            "the stamp keeps what the pattern answered before anything ran"
        );

        cache.clear();
        assert!(cache.is_empty());
    }

    /// A one-script launch loses the blanks and `[@+-]` off the front of each
    /// of its lines, and only where GNU Make's own test says the shell would
    /// not be reading them as script. `construct_command_argv_internal` does
    /// this on the far side of `goto slow`, so a text the tokenizer took apart
    /// itself keeps them — which is why the two `@echo` cases below differ only
    /// in whether the line has a shell character in it.
    #[test]
    fn a_one_script_launch_keeps_the_prefixes_a_shell_may_be_reading() {
        let script = Bytes::from_static(b"@echo hi > out\n  -echo bye\n");
        let bourne = argv_to_exec(b"/bin/dash", b"-c", &script, Some(b"-c"))
            .expect("a one-script launch always has an argv");
        assert_eq!(
            bourne,
            vec![
                Bytes::from_static(b"/bin/dash"),
                Bytes::from_static(b"-c"),
                Bytes::from_static(b"echo hi > out\necho bye\n"),
            ]
        );

        let unknown = argv_to_exec(b"/opt/myshell", b"-c", &script, Some(b"-c"))
            .expect("a one-script launch always has an argv");
        assert_eq!(unknown[2], script, "the prefixes could be its own script");

        // The fast path is not the one-shell branch, so nothing comes off a
        // line it can take apart by itself.
        let plain = Bytes::from_static(b"@echo hi");
        assert_eq!(
            argv_to_exec(b"/bin/sh", b"-c", &plain, Some(b"-c")),
            Some(vec![
                Bytes::from_static(b"@echo"),
                Bytes::from_static(b"hi")
            ])
        );
    }
}
