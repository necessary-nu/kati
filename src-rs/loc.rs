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

use std::fmt::Display;

use crate::symtab::{Interner, Symbol, Symtab};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loc {
    pub filename: Symbol,
    pub line: i32,
}

impl Loc {
    /// A borrowing wrapper that renders this location through `names`.
    ///
    /// The inherent `Display` this replaces reached for a process-global
    /// interner to render the filename. Every user-facing diagnostic goes
    /// through here, so the interner has to be reachable wherever one is
    /// raised.
    // [spec:ronin:req:make.no-ambient-state]
    pub fn display<'a, T: Interner + ?Sized>(&self, names: &'a T) -> LocDisplay<'a> {
        LocDisplay {
            symtab: names.symtab(),
            filename: self.filename,
            line: self.line,
        }
    }
}

pub struct LocDisplay<'a> {
    symtab: &'a Symtab,
    filename: Symbol,
    line: i32,
}

impl Display for LocDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.filename.display(self.symtab), self.line)
    }
}

/// The location of something with no location, whose filename is the
/// `<unknown>` every interner preloads.
impl Default for Loc {
    fn default() -> Self {
        Loc {
            filename: Symbol::UNKNOWN_FILENAME,
            line: 0,
        }
    }
}
