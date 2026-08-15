//! `--shuffle`: build in an order the Makefile did not write.
//!
//! GNU Make performs this inside make itself, before it walks the goals
//! (`main.c` calls `shuffle_goaldeps_recursive` immediately ahead of
//! `update_goal_chain (goals)`), which is why it lives here rather than in a
//! frontend that reads the finished plan: the walk that drops circular
//! prerequisites is the same walk, and it has to see the order the shuffle
//! chose.

use crate::dep::NamedDepNode;

/// What `--shuffle` reorders the goals and each target's prerequisites by.
///
/// The point of it is that the order a Makefile happens to write is not one it
/// may rely on: a build that only works in written order has a dependency it
/// never stated, and a run in some other order is what finds it. A seed settles
/// the permutation completely, so a run that found something can be repeated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Shuffle {
    /// Build in the order the Makefile wrote.
    #[default]
    None,
    /// Ask for a shuffle and get that same order. GNU Make keeps this apart
    /// from asking for none at all, and so does the value a sub-make inherits.
    Identity,
    /// Back to front.
    Reverse,
    /// The permutation this seed names.
    Seed(u32),
}

impl Shuffle {
    /// What `--shuffle`'s argument asks for, in GNU Make's spellings, which it
    /// compares without regard to case. `None` for a word that is neither a
    /// mode nor a seed.
    ///
    /// `random` is settled here rather than carried as a request, so that what
    /// travels onward names the permutation this run actually used.
    #[must_use]
    pub fn requested(spec: &[u8]) -> Option<Self> {
        Some(match spec.to_ascii_lowercase().as_slice() {
            b"none" => Self::None,
            b"identity" => Self::Identity,
            b"reverse" => Self::Reverse,
            b"random" => {
                use std::hash::{BuildHasher, Hasher};
                let entropy = std::collections::hash_map::RandomState::new()
                    .build_hasher()
                    .finish();
                #[expect(clippy::cast_possible_truncation, reason = "any 32 bits will do")]
                Self::Seed(entropy as u32)
            }
            digits => Self::Seed(
                std::str::from_utf8(digits)
                    .ok()
                    .and_then(|digits| digits.parse().ok())?,
            ),
        })
    }

    /// How a sub-make is told what this one did.
    ///
    /// The seed it settled on rather than the word that asked for one, which is
    /// what makes a tree of makes reproduce a run that failed. `None` for the
    /// mode that reorders nothing, which travels as nothing.
    #[must_use]
    pub fn spelling(self) -> Option<String> {
        match self {
            Self::None => None,
            Self::Identity => Some("identity".to_owned()),
            Self::Reverse => Some("reverse".to_owned()),
            Self::Seed(seed) => Some(seed.to_string()),
        }
    }
}

/// The draws one shuffle is made of.
enum Draw {
    Reverse,
    /// `SplitMix64`, whose whole state is the seed: the permutation follows
    /// from it and from the order the graph is walked in, and nothing else.
    Random(u64),
}

impl Draw {
    fn permute<T>(&mut self, items: &mut [T]) {
        match self {
            Self::Reverse => items.reverse(),
            Self::Random(state) => {
                for index in 0..items.len() {
                    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut draw = *state;
                    draw = (draw ^ (draw >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    draw = (draw ^ (draw >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    draw ^= draw >> 31;
                    let picked = usize::try_from(draw % items.len() as u64)
                        .expect("a remainder of a list length fits a length");
                    items.swap(index, picked);
                }
            }
        }
    }
}

/// Reorder the goals, and each target's prerequisites, before anything walks
/// them.
///
/// The order the graph is walked in afterwards is the order its edges are
/// minted in, and among edges that are equally ready the scheduler takes the one
/// minted first — so reordering the walk is what reorders the build. It is also
/// what decides which prerequisite closes a loop, because the walk that drops
/// circular edges reads the same lists: GNU Make's `update_file_1` takes
/// `d = du->shuf ? du->shuf : du` before it asks whether that file is already
/// being updated (`remake.c`).
///
/// Only the edge lists move. `actual_inputs` — what `$^`, `$+` and `$<` are
/// read from — is left in the order the Makefile wrote, which is GNU Make's
/// rule too: its shuffle populates a second `->shuf` chain over the same
/// prerequisites and leaves `->next`, which the automatic variables walk, where
/// it was.
///
/// `.NOTPARALLEL` takes it back. A Makefile saying its own recipes cannot
/// overlap is describing an order, and reordering it would read past what it
/// said.
///
/// `.WAIT` needs no exception, where GNU Make leaves a list holding one alone:
/// the evaluator has already turned each barrier into order-only prerequisites,
/// so the order it asked for is in the graph rather than in the list, and
/// survives being reordered.
pub fn reorder(shuffle: Shuffle, not_parallel: bool, nodes: &mut [NamedDepNode]) {
    let mut draw = match shuffle {
        Shuffle::None | Shuffle::Identity => return,
        Shuffle::Reverse => Draw::Reverse,
        Shuffle::Seed(seed) => Draw::Random(u64::from(seed)),
    };
    if not_parallel {
        return;
    }
    draw.permute(nodes);
    let mut seen = std::collections::HashSet::new();
    let mut work = nodes
        .iter()
        .rev()
        .map(|(_, node)| std::sync::Arc::clone(node))
        .collect::<Vec<_>>();
    while let Some(node) = work.pop() {
        let mut node = node.lock();
        if !seen.insert(node.output) {
            continue;
        }
        draw.permute(&mut node.deps);
        draw.permute(&mut node.order_onlys);
        for (_, dep) in node.deps.iter().chain(node.order_onlys.iter()).rev() {
            work.push(std::sync::Arc::clone(dep));
        }
    }
}
