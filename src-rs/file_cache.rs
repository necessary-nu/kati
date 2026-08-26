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
    collections::{HashMap, HashSet},
    ffi::{OsStr, OsString},
    sync::Arc,
};

use anyhow::Result;
use bytes::Bytes;

use crate::{
    file::{Makefile, Source},
    session::Session,
};

/// Parsed makefiles and extra file dependencies, for one session.
// [spec:ronin:req:make.no-ambient-state]
pub struct MakefileCache {
    /// The makefiles this session has READ, by name. A name that is not
    /// here is a name to ask the system about again, not one known to be
    /// absent.
    cache: HashMap<OsString, Arc<Makefile>>,
    supplied: HashMap<OsString, Bytes>,
    /// The bytes this session got for every makefile it read, by name.
    ///
    /// A read repeated over the same text has to be given the same text. A
    /// makefile a staged child has rewritten since is not the makefile GNU
    /// Make's one read saw, so the front end takes these out when a read ends
    /// and supplies them to the read that repeats it.
    sources: HashMap<OsString, Bytes>,
    /// Files this session depended on and did not get — an `include` whose name
    /// is not there, or would not open. They are not in `cache`, because
    /// nothing is cached about a file the session has not read, and a later run
    /// still has to compare their timestamps.
    unread: HashSet<OsString>,
}

impl Default for MakefileCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MakefileCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            supplied: HashMap::new(),
            sources: HashMap::new(),
            unread: HashSet::new(),
        }
    }

    /// Supply a makefile's bytes without requiring a filesystem path.
    pub fn supply(&mut self, filename: OsString, contents: Bytes) {
        self.supplied.insert(filename, contents);
    }

    /// Record the bytes a makefile was read as, for the read that repeats this
    /// one.
    pub(crate) fn note_source(&mut self, filename: OsString, contents: Bytes) {
        self.sources.insert(filename, contents);
    }

    /// Every makefile this session read, with the bytes it read.
    pub fn sources(&self) -> impl Iterator<Item = (&OsString, &Bytes)> {
        self.sources.iter()
    }

    /// Every file the session read, which is what the regeneration stamp
    /// records.
    pub fn all_filenames(&self) -> HashSet<OsString> {
        let mut ret = HashSet::new();
        for p in self.cache.keys() {
            ret.insert(p.clone());
        }
        for f in &self.unread {
            ret.insert(f.clone());
        }
        ret
    }
}

/// The parsed form of `filename`, read and parsed on first use.
///
/// Parsing interns, so this takes the whole session rather than the cache.
///
/// A file the session did not get is still a file this evaluation depended on,
/// so it joins the set a later run compares timestamps against. What is NOT
/// remembered is the failure: a second `include` of the same name asks the
/// system again, because the answer can have changed since — a makefile is
/// allowed to create the file between two `include` lines, and GNU Make's
/// `eval_makefile` (read.c) opens the name every time.
pub fn get_makefile(session: &mut Session, filename: &OsStr) -> Result<Source> {
    if let Some(mk) = session.makefiles.cache.get(filename) {
        return Ok(Source::Read(mk.clone()));
    }
    let filename = filename.to_os_string();
    let supplied = session.makefiles.supplied.get(&filename).cloned();
    let source = if let Some(contents) = supplied {
        Source::Read(Makefile::from_bytes(session, &filename, contents)?)
    } else {
        Makefile::from_file(session, &filename)?
    };
    match &source {
        Source::Read(mk) => {
            session.makefiles.cache.insert(filename, mk.clone());
        }
        // Absence is not cached either, for the same reason and one more: a
        // makefile can create the file between two `include` lines. GNU Make's
        // `eval_makefile` (read.c) opens the name every time it is included, so
        // a `$(shell)` that writes `hello.mk` after an `-include` of it failed
        // is a file the `include` below finds. The name still joins the set a
        // later run compares timestamps against — an evaluation that asked for
        // a file and did not get one depends on its absence.
        Source::Absent | Source::Unopened(_) | Source::Unreadable(_) | Source::Exhausted(_) => {
            session.makefiles.unread.insert(filename);
        }
    }
    Ok(source)
}
