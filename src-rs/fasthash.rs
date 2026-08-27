//! A fast, non-cryptographic hasher for the evaluator's internal lookup
//! tables.
//!
//! Rust's default `HashMap` hashes with SipHash-1-3 and a per-process random
//! key, which is the right default for a table whose keys a stranger chooses:
//! it makes collision flooding impossible to arrange. None of the tables this
//! module is for have that problem. Their keys are [`crate::symtab::Symbol`]
//! — an interner index, a `NonZeroUsize` — and interned byte strings out of a
//! Makefile on the machine's own disk, and the tables are private caches that
//! are never iterated for output.
//!
//! What that default costs was measured rather than assumed. Profiling a
//! Make-mode no-op against GNU Make 4.4.1 put a sixth of the whole run inside
//! SipHash and the probing around it: `DefaultHasher::write` at 5.4%,
//! `hash_one::<&Symbol>` at 4.8%, `hash_one::<&Bytes>` at 1.9% and the rest
//! spread across inserts and rehashes. Hashing a single `usize` through a
//! keyed cryptographic permutation to look up an array index is most of that,
//! and it buys nothing here — GNU Make hashes the same keys with a handful of
//! multiplies.
//!
//! The algorithm is rustc's own `FxHasher`: rotate, xor in the next word,
//! multiply by an odd constant. Weak in the ways that do not matter here
//! (nearby keys stay nearby, an adversary could collide it at will) and strong
//! in the one that does — it is a few instructions per word.
//!
//! DO NOT reach for this for a table whose keys come from outside the machine,
//! and do not reach for it where iteration order is observable: it changes the
//! order a `HashMap` yields, which is unspecified either way but stable within
//! a build, and code that had quietly come to depend on the old order would
//! change behaviour rather than fail. Every table converted here was checked
//! for that: each is a membership or lookup cache, and the one place the rule
//! table is iterated sorts its keys by name before using them.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// The odd multiplier from rustc's `FxHasher`, which is the 64-bit fractional
/// part of the golden ratio.
const MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

/// How far the accumulator turns before each word is mixed in, so that a key
/// differing only in its high bits still moves the low ones.
const ROTATE: u32 = 5;

/// rustc's `FxHasher`, which is a rotate, an xor and a multiply per word.
#[derive(Default, Clone, Copy)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_ne_bytes(chunk.try_into().unwrap_or_default());
            self.add(word);
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut last = [0u8; 8];
            last[..remainder.len()].copy_from_slice(remainder);
            self.add(u64::from_ne_bytes(last));
        }
        // Length last, so that a short key is not a prefix of a longer one
        // that happens to be zero-padded to the same words.
        self.add(bytes.len() as u64);
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// The `BuildHasher` behind [`FastMap`] and [`FastSet`].
pub type FastBuildHasher = BuildHasherDefault<FastHasher>;

/// A `HashMap` over keys this crate interns itself.
pub type FastMap<K, V> = HashMap<K, V, FastBuildHasher>;

/// A `HashSet` over keys this crate interns itself.
pub type FastSet<T> = HashSet<T, FastBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = FastHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn equal_keys_hash_alike() {
        assert_eq!(
            hash_of(&b"src/params.c".to_vec()),
            hash_of(&b"src/params.c".to_vec())
        );
        assert_eq!(hash_of(&41usize), hash_of(&41usize));
    }

    /// The point of mixing the length in. Without it a key and its
    /// zero-extension land in the same bucket, which is not wrong but wastes
    /// the table on exactly the shape a Makefile produces most of.
    #[test]
    fn a_zero_extended_key_hashes_apart() {
        assert_ne!(hash_of(&vec![b'o']), hash_of(&vec![b'o', 0, 0, 0]));
    }

    /// Adjacent interner indices must not share a bucket, because every table
    /// this hasher is for is keyed by one.
    #[test]
    fn adjacent_indices_spread() {
        let mut seen = HashSet::new();
        for index in 1usize..2_000 {
            assert!(seen.insert(hash_of(&index)), "index {index} collided");
        }
    }

    /// A table built with it has to behave like any other, which is the whole
    /// claim being made about swapping the default out.
    #[test]
    fn a_fast_map_stores_and_finds() {
        let mut map: FastMap<Vec<u8>, usize> = FastMap::default();
        for index in 0..1_000usize {
            map.insert(format!("obj/target_{index}.o").into_bytes(), index);
        }
        assert_eq!(map.len(), 1_000);
        for index in 0..1_000usize {
            let key = format!("obj/target_{index}.o").into_bytes();
            assert_eq!(map.get(&key).copied(), Some(index));
        }
        assert!(!map.contains_key(b"obj/absent.o".as_slice()));
    }
}
