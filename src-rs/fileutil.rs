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

use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, OsStr},
    path::Path,
    process::{Command, ExitStatus},
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

pub fn run_command(
    shell: &[u8],
    shellflag: &[u8],
    cmd: &Bytes,
    redirect_stderr: RedirectStderr,
) -> Result<(ExitStatus, Vec<u8>)> {
    let mut cmd_with_shell;
    let args = if !shell.starts_with(b"/") || memchr2(b' ', b'$', shell).is_some() {
        let cmd_escaped = crate::strutil::escape_shell(cmd);
        cmd_with_shell = BytesMut::new();
        cmd_with_shell.put_slice(shell);
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
    } else {
        // If the shell isn't complicated, we don't need to wrap in /bin/sh
        &[
            <OsStr as OsStrExt>::from_bytes(shell),
            <OsStr as OsStrExt>::from_bytes(shellflag),
            <OsStr as OsStrExt>::from_bytes(cmd),
        ]
    };

    log!("run_command({args:?})");

    let mut cmd = Command::new(args[0]);
    cmd.args(&args[1..]);

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

    let mut handle = cmd.spawn()?;
    // Drop the cmd, otherwise the pipe will be retained.
    drop(cmd);

    let mut output = Vec::new();
    reader.read_to_end(&mut output)?;

    let res = handle.wait()?;

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
}
