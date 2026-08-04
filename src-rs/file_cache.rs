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

use crate::{file::Makefile, session::Session};

/// Parsed makefiles and extra file dependencies, for one session.
// [spec:ronin:req:make.no-ambient-state]
pub struct MakefileCache {
    cache: HashMap<OsString, Option<Arc<Makefile>>>,
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
            extra_file_deps: HashSet::new(),
        }
    }

    pub fn add_extra_file_dep(&mut self, filename: OsString) {
        self.extra_file_deps.insert(filename);
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
pub fn get_makefile(session: &mut Session, filename: &OsStr) -> Result<Option<Arc<Makefile>>> {
    if let Some(mk) = session.makefiles.cache.get(filename) {
        return Ok(mk.clone());
    }
    let filename = filename.to_os_string();
    let mk = Makefile::from_file(session, &filename)?;
    session.makefiles.cache.insert(filename, mk.clone());
    Ok(mk)
}
