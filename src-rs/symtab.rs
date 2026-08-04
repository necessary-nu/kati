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
    collections::HashMap,
    fmt::{Debug, Display},
    num::NonZeroUsize,
    sync::LazyLock,
    vec,
};

use bytes::{BufMut, Bytes, BytesMut};
use parking_lot::Mutex;

static SYMTAB: LazyLock<Mutex<Symtab>> = LazyLock::new(|| Mutex::new(Symtab::new()));

/// Run `f` with exclusive access to the process-global interner.
///
/// Temporary: `kati-session-value` replaces this with an interner reached
/// through the session. Nothing called from `f` may intern or render a symbol,
/// because the interner lock is not reentrant.
pub fn with_symtab<R>(f: impl FnOnce(&mut Symtab) -> R) -> R {
    f(&mut SYMTAB.lock())
}

pub static SHELL_SYM: LazyLock<Symbol> = LazyLock::new(|| intern("SHELL"));
pub static ALLOW_RULES_SYM: LazyLock<Symbol> = LazyLock::new(|| intern(".KATI_ALLOW_RULES"));
pub static KATI_READONLY_SYM: LazyLock<Symbol> = LazyLock::new(|| intern(".KATI_READONLY"));
pub static VARIABLES_SYM: LazyLock<Symbol> = LazyLock::new(|| intern(".VARIABLES"));
pub static KATI_SYMBOLS_SYM: LazyLock<Symbol> = LazyLock::new(|| intern(".KATI_SYMBOLS"));
pub static MAKEFILE_LIST: LazyLock<Symbol> = LazyLock::new(|| intern("MAKEFILE_LIST"));

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(NonZeroUsize);

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.as_bytes()))
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}({})", self.as_bytes(), self.0.get())
    }
}

impl Symbol {
    pub fn as_bytes(&self) -> Bytes {
        with_symtab(|symtab| symtab.name(*self))
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

/// The interner: byte strings to [`Symbol`] handles and back.
///
/// It holds no variable bindings. A `Symtab` and a [`crate::var::GlobalVars`]
/// are constructed, replaced, and dropped independently of each other.
// [spec:ronin:req:make.scope-separation]
pub struct Symtab {
    symbols: Vec<Bytes>,
    index: HashMap<Bytes, Symbol>,
}

impl Default for Symtab {
    fn default() -> Self {
        Self::new()
    }
}

impl Symtab {
    /// An interner preloaded with the 255 single-byte names, which are interned
    /// at the slot equal to their byte so that `intern` can answer for them
    /// without consulting the map.
    pub fn new() -> Self {
        let mut symtab = Self {
            symbols: vec![Bytes::new()],
            index: HashMap::new(),
        };
        for i in 1u8..=255 {
            assert!(symtab.symbols.len() == i as usize);
            let name = Bytes::from(vec![i]);
            let sym = Symbol(NonZeroUsize::new(i.into()).unwrap());
            symtab.symbols.push(name.clone());
            symtab.index.insert(name, sym);
        }
        symtab
    }

    pub fn intern<T: Into<Bytes> + AsRef<[u8]>>(&mut self, s: T) -> Symbol {
        if let [c] = s.as_ref() {
            return Symbol(NonZeroUsize::new(*c as usize).unwrap());
        }
        let s = s.into();
        if let Some(sym) = self.index.get(&s) {
            return *sym;
        }
        let sym = Symbol(NonZeroUsize::new(self.symbols.len()).unwrap());
        self.symbols.push(s.clone());
        self.index.insert(s, sym);
        sym
    }

    /// The bytes `sym` was interned from. Panics if `sym` came from a different
    /// interner and names a slot this one does not have.
    pub fn name(&self, sym: Symbol) -> Bytes {
        self.symbols[sym.0.get()].clone()
    }

    /// The number of interned names, counting the reserved slot 0.
    pub fn count(&self) -> usize {
        self.symbols.len()
    }
}

pub fn intern<T: Into<Bytes> + AsRef<[u8]>>(s: T) -> Symbol {
    with_symtab(|symtab| symtab.intern(s))
}

pub fn join_symbols(symbols: &[Symbol], sep: &[u8]) -> Bytes {
    let mut r = BytesMut::new();
    let mut first = true;
    for s in symbols {
        if !first {
            r.put_slice(sep);
        } else {
            first = false;
        }
        r.put_slice(&s.as_bytes());
    }
    r.freeze()
}

pub fn symbol_count() -> usize {
    with_symtab(|symtab| symtab.count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern() {
        let sym = intern("foo");
        let sym2 = intern("bar");
        let sym3 = intern("foo");
        assert_ne!(sym, sym2);
        assert_eq!(sym, sym3);
    }

    #[test]
    fn test_symbol_to_string() {
        let sym = intern("foo");
        assert_eq!(sym.to_string(), "foo");
    }

    #[test]
    fn test_single_letter_symbol() {
        let sym = intern("a");
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
}
