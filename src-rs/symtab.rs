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

//! Symbol interning.
//!
//! This module owns the mapping between byte strings and [`Symbol`] handles and
//! nothing else. Make's global variable scope, which is keyed by `Symbol`,
//! lives in [`crate::var::GlobalVars`]; interning a name neither creates nor
//! reads a variable binding. See `[spec:ronin:req:make.scope-separation]`.

use std::{
    fmt::{Debug, Display},
    num::NonZeroUsize,
    vec,
};

use bytes::{BufMut, Bytes, BytesMut};

use crate::fasthash::FastMap;

/// Anything that can hand out the interner a [`Symbol`] was minted from.
///
/// A `Symbol` is an index, so rendering one needs the interner that produced
/// it. This is the smallest thing a caller can hold to be able to do that: the
/// interner itself, or a value that owns one.
pub trait Interner {
    fn symtab(&self) -> &Symtab;
}

impl Interner for Symtab {
    fn symtab(&self) -> &Symtab {
        self
    }
}

impl<T: Interner + ?Sized> Interner for &T {
    fn symtab(&self) -> &Symtab {
        (**self).symtab()
    }
}

/// Names every interner preloads, at fixed indices immediately after the 255
/// single-byte names.
///
/// This is what makes [`Symbol::SHELL`] and friends `const` rather than a
/// lazily interned static. A static would hold an index into whichever
/// interner happened to touch it first, which is a wrong answer or a panic
/// against any other one. Preloading the same names in the same order in every
/// interner makes the index the same everywhere, so the handle is meaningful in
/// every session. [`Symtab::new`] asserts it.
const WELL_KNOWN: &[&[u8]] = &[
    b"<unknown>",
    b"SHELL",
    b".VARIABLES",
    b"MAKEFILE_LIST",
    b".POSIX",
    b".SHELLSTATUS",
    b".RECIPEPREFIX",
    b".SHELLFLAGS",
    b".DEFAULT_GOAL",
    b".SUFFIXES",
];

/// Slot 0 is reserved and slots 1..=255 are the single-byte names.
const WELL_KNOWN_BASE: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(NonZeroUsize);

/// `Symbol` renders only through [`Symbol::display`], which takes the interner.
/// `Debug` says what a bare handle can say for itself without one.
impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol({})", self.0.get())
    }
}

impl Symbol {
    const fn well_known(i: usize) -> Symbol {
        match NonZeroUsize::new(WELL_KNOWN_BASE + i) {
            Some(n) => Symbol(n),
            None => unreachable!(),
        }
    }

    pub const UNKNOWN_FILENAME: Symbol = Symbol::well_known(0);
    pub const SHELL: Symbol = Symbol::well_known(1);
    pub const VARIABLES: Symbol = Symbol::well_known(2);
    pub const MAKEFILE_LIST: Symbol = Symbol::well_known(3);
    pub const POSIX: Symbol = Symbol::well_known(4);
    pub const SHELLSTATUS: Symbol = Symbol::well_known(5);
    pub const RECIPEPREFIX: Symbol = Symbol::well_known(6);
    pub const SHELLFLAGS: Symbol = Symbol::well_known(7);
    pub const DEFAULT_GOAL: Symbol = Symbol::well_known(8);
    pub const SUFFIXES: Symbol = Symbol::well_known(9);

    /// The bytes this handle was interned from, as a SECOND OWNER of them.
    ///
    /// A `Bytes` is a shared handle, so taking one out and dropping it again is
    /// a pair of atomic read-modify-writes on a refcount that never went
    /// anywhere. A caller that only reads the name wants
    /// [`Symtab::name_bytes`], which borrows it from the interner instead.
    // [spec:ronin:req:make.no-ambient-state]
    pub fn as_bytes(&self, names: &impl Interner) -> Bytes {
        names.symtab().name(*self)
    }

    /// The bytes this handle was interned from, BORROWED from the interner.
    ///
    /// The one to reach for. [`Self::as_bytes`] hands back a second owner, and
    /// a second owner is an atomic increment now and an atomic decrement later
    /// on a refcount that never went anywhere: one vim `src` read asks the
    /// interner for a name 33,178 times, so that is 66,356 read-modify-writes
    /// on cache lines eight composing threads share. A caller that reads the
    /// name and is done with it before it touches the interner again — which is
    /// every predicate, every comparison and every write into a buffer — wants
    /// this one, and the borrow checker refuses it exactly where an owner was
    /// genuinely needed.
    // [spec:ronin:req:make.no-ambient-state]
    pub fn name_bytes<'a, T: Interner + ?Sized>(&self, names: &'a T) -> &'a [u8] {
        names.symtab().name_bytes(*self)
    }

    /// A borrowing wrapper that renders this handle through `names`.
    ///
    /// This replaces the inherent `Display` the symbol used to have, which
    /// could only work by reaching for a process-global interner.
    // [spec:ronin:req:make.no-ambient-state]
    pub fn display<'a, T: Interner + ?Sized>(&self, names: &'a T) -> SymbolDisplay<'a> {
        SymbolDisplay {
            symtab: names.symtab(),
            sym: *self,
        }
    }

    /// The interner slot this handle names. Only the interner and the variable
    /// scope keyed by it have any business looking at this.
    pub(crate) fn index(self) -> usize {
        self.0.get()
    }

    /// The handle for interner slot `index`, or `None` for the reserved slot 0.
    pub(crate) fn from_index(index: usize) -> Option<Self> {
        NonZeroUsize::new(index).map(Self)
    }
}

/// What [`Symbol::display`] returns: a symbol's name, borrowed from the
/// interner that was passed in.
pub struct SymbolDisplay<'a> {
    symtab: &'a Symtab,
    sym: Symbol,
}

impl Display for SymbolDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            String::from_utf8_lossy(&self.symtab.name(self.sym))
        )
    }
}

impl Debug for SymbolDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.symtab.name(self.sym))
    }
}

/// The interner: byte strings to [`Symbol`] handles and back.
///
/// It holds no variable bindings. A `Symtab` and a [`crate::var::GlobalVars`]
/// are constructed, replaced, and dropped independently of each other.
// [spec:ronin:req:make.scope-separation]
pub struct Symtab {
    symbols: Vec<Bytes>,
    index: FastMap<Bytes, Symbol>,
}

/// Every byte in order, so that the one-byte name for byte `b` can be a slice
/// of this rather than a `Vec` of its own. See [`Symtab::new`].
///
/// A `static` rather than a `const` because `Bytes::from_static` wants a
/// `&'static [u8]`, and a `const` array read at a runtime index yields a
/// temporary rather than a borrow of one place.
// no-globals-gate: 256 constant bytes, read-only, holding no evaluation state
static SINGLE_BYTE_NAMES: [u8; 256] = {
    let mut names = [0u8; 256];
    let mut byte = 0usize;
    while byte < 256 {
        // `as` rather than a fallible conversion: the loop bound is the width
        // of the type being written, and a `const` block has no `?` to take a
        // conversion's answer with.
        #[expect(clippy::cast_possible_truncation, reason = "byte < 256 by the loop")]
        {
            names[byte] = byte as u8;
        }
        byte += 1;
    }
    names
};

impl Default for Symtab {
    fn default() -> Self {
        Self::new()
    }
}

impl Symtab {
    /// An interner preloaded with the 255 single-byte names, which are interned
    /// at the slot equal to their byte so that `intern` can answer for them
    /// without consulting the map, followed by the [`WELL_KNOWN`] names at
    /// fixed slots.
    ///
    /// The single-byte names are not in the map, and that is what the sentence
    /// above buys rather than an oversight: both readers of the map —
    /// [`Self::intern`] and [`Self::peek_symbol`] — answer a one-byte name from
    /// its byte and return before they reach it, so an entry for one could
    /// never be looked up. Two hundred and fifty-five unreachable entries were
    /// being hashed into every interner a session built, and a session is built
    /// per compilation unit.
    ///
    /// They are also carved out of one static byte table rather than allocated.
    /// A `Bytes` over a static is not refcounted at all, so beyond the two
    /// allocations per name this saves at construction, holding one of these
    /// names afterwards costs no atomic either.
    pub fn new() -> Self {
        let mut symtab = Self {
            symbols: vec![Bytes::new()],
            index: FastMap::default(),
        };
        for i in 1u8..=255 {
            assert!(symtab.symbols.len() == i as usize);
            let at = usize::from(i);
            symtab
                .symbols
                .push(Bytes::from_static(&SINGLE_BYTE_NAMES[at..=at]));
        }
        for (i, name) in WELL_KNOWN.iter().enumerate() {
            let sym = symtab.intern(*name);
            assert!(
                sym == Symbol::well_known(i),
                "well-known symbol {} landed at slot {}, not {}",
                String::from_utf8_lossy(name),
                sym.index(),
                WELL_KNOWN_BASE + i
            );
        }
        symtab
    }

    /// The handle for `s`, minting one if this interner has not seen it.
    ///
    /// The lookup reads `s` borrowed and the conversion to [`Bytes`] happens
    /// only for a name that turns out to be new. A caller handing over a
    /// `&[u8]`, a `&str` or a `Vec` pays an allocation and a copy to become one,
    /// and most of what a read interns has been interned already: the same
    /// target named by four rules, the same variable read in twenty recipes.
    pub fn intern<T: Into<Bytes> + AsRef<[u8]>>(&mut self, s: T) -> Symbol {
        if let [c] = s.as_ref() {
            return Symbol(NonZeroUsize::new(*c as usize).unwrap());
        }
        if let Some(sym) = self.index.get(s.as_ref()) {
            return *sym;
        }
        let s = s.into();
        let sym = Symbol(NonZeroUsize::new(self.symbols.len()).unwrap());
        self.symbols.push(s.clone());
        self.index.insert(s, sym);
        sym
    }

    /// The symbol `name` already has, without minting one for a name nothing
    /// has used. A caller asking a question about a name — rather than
    /// establishing one — must not grow the table to ask it.
    pub fn peek_symbol(&self, name: &[u8]) -> Option<Symbol> {
        if let [c] = name {
            return Some(Symbol(NonZeroUsize::new(*c as usize).unwrap()));
        }
        self.index.get(name).copied()
    }

    /// The bytes `sym` was interned from. Panics if `sym` came from a different
    /// interner and names a slot this one does not have.
    pub fn name(&self, sym: Symbol) -> Bytes {
        self.symbols[sym.0.get()].clone()
    }

    /// The same bytes, borrowed from the interner rather than handed over as a
    /// second holder of them.
    ///
    /// [`Self::name`] hands back a `Bytes`, which is a shared owner: taking one
    /// out and dropping it again is a pair of atomic read-modify-writes on a
    /// refcount that never went anywhere. A caller that reads the name and is
    /// done with it before it touches the interner again wants this one.
    pub fn name_bytes(&self, sym: Symbol) -> &[u8] {
        &self.symbols[sym.0.get()]
    }

    /// The number of interned names, counting the reserved slot 0.
    pub fn count(&self) -> usize {
        self.symbols.len()
    }
}

pub fn join_symbols(names: &impl Interner, symbols: &[Symbol], sep: &[u8]) -> Bytes {
    let symtab = names.symtab();
    let mut r = BytesMut::new();
    let mut first = true;
    for s in symbols {
        if !first {
            r.put_slice(sep);
        } else {
            first = false;
        }
        r.put_slice(&symtab.name(*s));
    }
    r.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern() {
        let mut symtab = Symtab::new();
        let sym = symtab.intern("foo");
        let sym2 = symtab.intern("bar");
        let sym3 = symtab.intern("foo");
        assert_ne!(sym, sym2);
        assert_eq!(sym, sym3);
    }

    #[test]
    fn test_symbol_to_string() {
        let mut symtab = Symtab::new();
        let sym = symtab.intern("foo");
        assert_eq!(sym.display(&symtab).to_string(), "foo");
    }

    #[test]
    fn test_single_letter_symbol() {
        let mut symtab = Symtab::new();
        let sym = symtab.intern("a");
        assert_eq!(sym.0.get(), 'a' as usize);
    }

    #[test]
    fn test_symtab_is_independently_constructible() {
        let mut a = Symtab::new();
        let mut b = Symtab::new();
        let foo = a.intern("foo");
        assert_eq!(a.name(foo), Bytes::from_static(b"foo"));
        // A private interner is unaffected by every name interned elsewhere.
        assert_eq!(b.count(), Symtab::new().count());
        assert_eq!(b.intern("foo"), foo);
    }

    /// The well-known handles must name the same bytes in any interner, which
    /// is the whole reason they can be `const`.
    // [spec:ronin:req:make.no-ambient-state/test]
    #[test]
    fn test_well_known_symbols_agree_across_interners() {
        let a = Symtab::new();
        let mut b = Symtab::new();
        b.intern("something else entirely");
        for sym in [
            Symbol::UNKNOWN_FILENAME,
            Symbol::SHELL,
            Symbol::VARIABLES,
            Symbol::MAKEFILE_LIST,
            Symbol::POSIX,
            Symbol::SHELLSTATUS,
            Symbol::RECIPEPREFIX,
            Symbol::SHELLFLAGS,
            Symbol::DEFAULT_GOAL,
        ] {
            assert_eq!(a.name(sym), b.name(sym));
        }
        assert_eq!(a.name(Symbol::SHELL), Bytes::from_static(b"SHELL"));
        assert_eq!(b.intern("SHELL"), Symbol::SHELL);
        assert_eq!(b.intern(".RECIPEPREFIX"), Symbol::RECIPEPREFIX);
    }
}
