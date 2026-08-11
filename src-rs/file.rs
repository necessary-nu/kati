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
/// A file that is not there and a file that will not open are different
/// answers: the first is an ordinary thing for a Makefile to describe how to
/// generate, and the second is a failure the caller has to report. Telling
/// them apart here is what lets the caller name the path and the directive
/// that asked for it — an `io::Error` on its own carries neither.
pub enum Source {
    /// The file, parsed.
    Read(Arc<Makefile>),
    /// Nothing is at that path.
    Absent,
    /// Something is, and the system would not let us read it. The caller
    /// passed the path in, so what comes back is the system's reason alone.
    Unreadable(std::io::Error),
}

impl Makefile {
    /// Parse `buf` as the makefile named by `filename`.
    pub(crate) fn from_bytes(
        session: &mut Session,
        filename: &OsStr,
        buf: Bytes,
    ) -> Result<Arc<Self>> {
        let filename = session.intern(filename.as_bytes().to_vec());
        let stmts = parse_file(session, &buf, filename)?;
        Ok(Arc::new(Self { filename, stmts }))
    }

    /// Read and parse `filename`.
    ///
    /// One `read` rather than an `exists` and then a `read`. The two asked the
    /// same question, the pair could disagree when something else was writing
    /// the tree, and `exists` answered a failure it could not classify — a
    /// directory that cannot be searched — with the same `Err` as everything
    /// else, which is the context this reports instead.
    pub fn from_file(session: &mut Session, filename: &OsStr) -> Result<Source> {
        let buf = match std::fs::read(filename) {
            Ok(buf) => Bytes::from(buf),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Source::Absent),
            Err(err) => return Ok(Source::Unreadable(err)),
        };

        Ok(Source::Read(Self::from_bytes(session, filename, buf)?))
    }
}
