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
    cache: HashMap<OsString, Option<Arc<Makefile>>>,
    supplied: HashMap<OsString, Bytes>,
    extra_file_deps: HashSet<OsString>,
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
            extra_file_deps: HashSet::new(),
        }
    }

    pub fn add_extra_file_dep(&mut self, filename: OsString) {
        self.extra_file_deps.insert(filename);
    }

    /// Supply a makefile's bytes without requiring a filesystem path.
    pub fn supply(&mut self, filename: OsString, contents: Bytes) {
        self.supplied.insert(filename, contents);
    }

    /// Every file the session read, which is what the regeneration stamp
    /// records.
    pub fn all_filenames(&self) -> HashSet<OsString> {
        let mut ret = HashSet::new();
        for p in self.cache.keys() {
            ret.insert(p.clone());
        }
        for f in &self.extra_file_deps {
            ret.insert(f.clone());
        }
        ret
    }
}

/// The parsed form of `filename`, read and parsed on first use.
///
/// Parsing interns, so this takes the whole session rather than the cache.
///
/// A file that would not open is still a file this evaluation depended on and
/// did not get, so it joins the set a later run compares timestamps against.
/// The failure itself is handed straight back rather than cached, so a second
/// `include` of it asks the system again rather than being told it is absent —
/// which it is not.
pub fn get_makefile(session: &mut Session, filename: &OsStr) -> Result<Source> {
    if let Some(mk) = session.makefiles.cache.get(filename) {
        return Ok(match mk {
            Some(mk) => Source::Read(mk.clone()),
            None => Source::Absent,
        });
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
            session.makefiles.cache.insert(filename, Some(mk.clone()));
        }
        Source::Absent => {
            session.makefiles.cache.insert(filename, None);
        }
        Source::Unopened(_) | Source::Unreadable(_) | Source::Exhausted(_) => {
            session.makefiles.extra_file_deps.insert(filename);
        }
    }
    Ok(source)
}
