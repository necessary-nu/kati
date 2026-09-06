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

use std::{ffi::OsStr, os::unix::ffi::OsStrExt, sync::Arc};

use anyhow::Result;
use bytes::Bytes;
use parking_lot::Mutex;

use crate::{parser::parse_file, session::Session, stmt::Stmt, symtab::Symbol};

pub struct Makefile {
    pub filename: Symbol,
    pub stmts: Arc<Mutex<Vec<Stmt>>>,
}

/// What asking for a makefile found.
///
/// Four answers rather than two, and the line between them is GNU Make's own:
/// the open and the read fail in different places and it treats them
/// differently.
///
/// `eval_makefile` (reference/gnumake/src/read.c:347) stores the open's errno on
/// the goaldep and returns without a word, whatever the errno was — so a file
/// that is not there and a file that would not open are the same kind of answer,
/// an ordinary thing for a Makefile to describe how to generate, and both are
/// deferred to the update that refuses over them. The three exceptions it names
/// mean make itself is out of resources rather than that this file is a problem,
/// and they end the run where they happened.
///
/// A read that fails after the open succeeded is not deferred at all:
/// `readline` checks `ferror` and calls `pfatal_with_name` (read.c:2744), which
/// names the file under make's own name and stops. That is the path `include`ing
/// a directory takes on Linux, where the open succeeds and the read is what says
/// `Is a directory`.
///
/// The caller passed the path in, so what comes back carries the system's reason
/// alone; naming the path and the directive that asked for it is the caller's,
/// and an `io::Error` on its own carries neither.
pub enum Source {
    /// The file, parsed.
    Read(Arc<Makefile>),
    /// Nothing is at that path.
    Absent,
    /// Something is, and it would not open. Deferred like absence: the read says
    /// nothing, and whether this ends the run is the update's question.
    Unopened(std::io::Error),
    /// It opened, and then would not read. Not deferred — the run ends here.
    Unreadable(std::io::Error),
    /// The open failed for a reason that is about make rather than about this
    /// file: no descriptors left, or no memory. Also not deferred.
    Exhausted(std::io::Error),
}

impl Makefile {
    /// Parse `buf` as the makefile named by `filename`.
    ///
    /// The bytes are noted as they were read and parsed with any byte order
    /// mark taken off the front, because those are two different questions: a
    /// pass that re-reads this unit is handed back what the file holds, and the
    /// read that makes statements out of it is the one GNU Make skips the mark
    /// for.
    pub(crate) fn from_bytes(
        session: &mut Session,
        filename: &OsStr,
        buf: Bytes,
    ) -> Result<Arc<Self>> {
        session
            .makefiles
            .note_source(filename.to_os_string(), buf.clone());
        let posix_pedantic = session.posix_pedantic;
        if let Some(parsed) = session
            .makefiles
            .already_parsed(filename, &buf, posix_pedantic)
        {
            return Ok(parsed);
        }
        let name = session.intern(filename.as_bytes().to_vec());
        let (stmts, warned) = parse_file(session, &without_byte_order_mark(&buf), name)?;
        let parsed = Arc::new(Self {
            filename: name,
            stmts,
        });
        // A parse that said something, or that latched `.POSIX:` and so changed
        // what the lines after it in this same read mean, is not a function of
        // the bytes alone and is not offered to the read that repeats this one.
        if !warned && session.posix_pedantic == posix_pedantic {
            session.makefiles.note_parse(
                filename.to_os_string(),
                buf,
                posix_pedantic,
                Arc::clone(&parsed),
            );
        }
        Ok(parsed)
    }

    /// Read and parse `filename`.
    ///
    /// The open and the read are taken apart rather than done in one `fs::read`,
    /// because GNU Make answers them differently and this is the only place that
    /// still knows which of the two failed. No `exists` ahead of either: the two
    /// asked the same question, the pair could disagree when something else was
    /// writing the tree, and `exists` answered a failure it could not classify —
    /// a directory that cannot be searched — with the same `Err` as everything
    /// else.
    pub fn from_file(session: &mut Session, filename: &OsStr) -> Result<Source> {
        use std::io::Read as _;

        let mut file = match std::fs::File::open(filename) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Source::Absent),
            Err(err) if is_exhaustion(&err) => return Ok(Source::Exhausted(err)),
            Err(err) => return Ok(Source::Unopened(err)),
        };
        // Sized from the metadata the way `fs::read` does, and a failure to ask
        // is not a failure to read: the read below is what decides.
        let mut buf = Vec::with_capacity(
            file.metadata()
                .ok()
                .and_then(|metadata| usize::try_from(metadata.len()).ok())
                .unwrap_or_default(),
        );
        if let Err(err) = file.read_to_end(&mut buf) {
            return Ok(Source::Unreadable(err));
        }

        Ok(Source::Read(Self::from_bytes(
            session,
            filename,
            Bytes::from(buf),
        )?))
    }
}

/// A makefile's text with a leading UTF-8 byte order mark taken off.
///
/// GNU Make skips one at the head of a makefile and nowhere else: `eval`
/// (read.c) tests for `EF BB BF` only while `ebuf->floc.lineno == 1`, so an
/// editor that writes the mark does not give the first target a three-byte
/// prefix nobody can name. A mark further into the file is ordinary text on
/// both sides, and this is why: the test is on the line, not on the bytes.
fn without_byte_order_mark(buf: &Bytes) -> Bytes {
    const MARK: &[u8] = b"\xEF\xBB\xBF";
    match buf.starts_with(MARK) {
        true => buf.slice(MARK.len()..),
        false => buf.clone(),
    }
}

/// Whether the open failed because make has run out of something, rather than
/// because of anything about this file.
///
/// GNU Make's own three, named in `eval_makefile` (read.c:347) as the errnos it
/// will not defer: out of descriptors for this process, out of them for the
/// system, out of memory.
fn is_exhaustion(err: &std::io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(libc::EMFILE | libc::ENFILE | libc::ENOMEM)
    )
}
