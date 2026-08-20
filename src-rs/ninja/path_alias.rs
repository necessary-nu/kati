//! Physical Ninja names for logical phony targets no filesystem can address.

use crate::symtab::{Symbol, Symtab};
use std::collections::{HashMap, HashSet};

const PREFIX: &str = "_kati_unaddressable_phony_";

// [spec:ronin:req:make.compiler-boundary]
#[derive(Default)]
pub(super) struct PhonyAliases {
    aliases: HashMap<Symbol, Symbol>,
}

impl PhonyAliases {
    pub(super) fn prepare(
        &mut self,
        names: &mut Symtab,
        occupied: &HashSet<Symbol>,
        phonies: &[Symbol],
    ) {
        let mut used = occupied.clone();
        for logical in phonies {
            let name = names.name(*logical);
            if !unaddressable(&name) {
                continue;
            }
            let base = format!("{PREFIX}{:016x}", stable_hash(&name));
            for suffix in 0usize.. {
                let candidate = if suffix == 0 {
                    base.clone()
                } else {
                    format!("{base}_{suffix}")
                };
                let physical = names.intern(candidate.into_bytes());
                if used.insert(physical) {
                    self.aliases.insert(*logical, physical);
                    break;
                }
            }
        }
    }

    pub(super) fn resolve(&self, logical: Symbol) -> Symbol {
        self.aliases.get(&logical).copied().unwrap_or(logical)
    }
}

/// The longest a single path component may be.
///
/// Written out rather than read from `libc`, which offers `NAME_MAX` on Linux
/// alone: ext4, APFS, and HFS+ all stop a component at the same 255 bytes, so
/// the number is not the part that varies by host. `PATH_MAX` below stays
/// `libc`'s for the opposite reason — it really does vary (4096 on Linux, 1024
/// on Apple), and a name the running host cannot address is exactly what an
/// alias stands in for.
const NAME_MAX: usize = 255;

fn unaddressable(path: &[u8]) -> bool {
    let path_max = usize::try_from(libc::PATH_MAX).expect("PATH_MAX is positive");
    path.len() >= path_max
        || path
            .split(|byte| *byte == b'/')
            .any(|name| name.len() > NAME_MAX)
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    // [spec:ronin:req:make.compiler-boundary/test]
    #[test]
    fn detects_unaddressable_paths() {
        assert!(!unaddressable(&vec![b'x'; NAME_MAX]));
        assert!(unaddressable(&vec![b'x'; NAME_MAX + 1]));

        let path_max = usize::try_from(libc::PATH_MAX).unwrap();
        let path = [b"x/".repeat(path_max / 2), vec![b'x'; path_max % 2]].concat();
        assert!(unaddressable(&path));
    }

    // [spec:ronin:req:make.compiler-boundary/test]
    #[test]
    fn avoids_real_target_collisions() {
        let mut names = Symtab::new();
        let logical = names.intern(vec![b'x'; NAME_MAX + 1]);
        let base = format!("{PREFIX}{:016x}", stable_hash(&names.name(logical)));
        let collision = names.intern(base.clone().into_bytes());
        let occupied = HashSet::from([logical, collision]);
        let mut aliases = PhonyAliases::default();
        aliases.prepare(&mut names, &occupied, &[logical]);

        let physical = aliases.resolve(logical);
        assert_ne!(physical, logical);
        assert_ne!(physical, collision);
        let expected = format!("{base}_1");
        assert_eq!(names.name(physical).as_ref(), expected.as_bytes());
    }
}
