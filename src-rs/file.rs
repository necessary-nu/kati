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
    pub(crate) fn from_bytes(
        session: &mut Session,
        filename: &OsStr,
        buf: Bytes,
    ) -> Result<Arc<Self>> {
        session
            .makefiles
            .note_source(filename.to_os_string(), buf.clone());
        let filename = session.intern(filename.as_bytes().to_vec());
        let stmts = parse_file(session, &buf, filename)?;
        Ok(Arc::new(Self { filename, stmts }))
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
