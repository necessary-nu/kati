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

//! `lib.a(member.o)`: the one target name that is not a filename.
//!
//! GNU Make reads this shape wherever a target or a prerequisite is written,
//! in `ar_name` (reference/gnumake/src/ar.c), and it changes four separate
//! answers about the name:
//!
//! * `$@` is the archive and `$%` is the member, rather than `$@` being the
//!   whole name and `$%` being empty (`set_file_variables`, src/commands.c).
//! * A prerequisite written in the same shape contributes only its member name
//!   to `$^`, `$?`, `$+` and `$<`.
//! * The implicit search gets a second pass in which the name being matched is
//!   `(member)` — the archive name held aside entirely — which is how the
//!   built-in `(%): %` rule ever fires (`try_implicit_rule`, src/implicit.c).
//! * Its timestamp comes from the archive's index rather than from a file of
//!   that name, which is Ronin's half of the same feature.
//!
//! The test itself is deliberately narrow, because a filename may legitimately
//! contain parentheses: the name must contain a `(`, it must not begin with
//! one, it must end with `)`, and the parentheses must not be adjacent — so
//! `lib.a()` is an ordinary filename and `(x)` is too.

use bytes::Bytes;

/// Whether `name` is written in the archive-member shape.
///
/// GNU Make's `ar_name`. A doubled `((member))` is the one shape it refuses
/// outright rather than reading either way; that refusal is the caller's,
/// because it is a diagnostic rather than a question about the name.
pub fn is_archive_name(name: &[u8]) -> bool {
    let Some(open) = name.iter().position(|byte| *byte == b'(') else {
        return false;
    };
    // GNU Make: `p == name` and `*end != ')'` and `end == p + 1` each refuse,
    // where `end` is the last byte. The third is what keeps `lib.a()` an
    // ordinary filename.
    open != 0 && name.last() == Some(&b')') && name.len() - 1 > open + 1
}

/// Whether the name is the doubled `lib((member))` form GNU Make refuses
/// outright as an unsupported feature.
pub fn is_nested_archive_name(name: &[u8]) -> bool {
    let Some(open) = name.iter().position(|byte| *byte == b'(') else {
        return false;
    };
    is_archive_name(name)
        && name.get(open + 1) == Some(&b'(')
        && name.get(name.len() - 2) == Some(&b')')
}

/// Split `lib(member)` into its two halves, or `None` for an ordinary name.
///
/// GNU Make's `ar_parse_name`, which is only ever called where `ar_name` has
/// already answered yes.
pub fn split_archive_name(name: &[u8]) -> Option<(&[u8], &[u8])> {
    if !is_archive_name(name) {
        return None;
    }
    let open = name.iter().position(|byte| *byte == b'(')?;
    Some((&name[..open], &name[open + 1..name.len() - 1]))
}

/// The member half of an archive name, or the whole name.
///
/// This is what a prerequisite written as `lib(member)` contributes to `$^`,
/// `$?`, `$+` and `$<`: GNU Make walks past the `(` and drops the final `)`
/// for every prerequisite it puts in those (src/commands.c).
pub fn member_or_whole(name: &Bytes) -> Bytes {
    match split_archive_name(name) {
        Some((archive, _)) => name.slice(archive.len() + 1..name.len() - 1),
        None => name.clone(),
    }
}

/// The name the implicit search matches on its archive pass: `(member)`,
/// with the archive held aside.
///
/// GNU Make's `pattern_search` is handed `strchr (file->name, '(')`, so the
/// parentheses are part of what the pattern must match — which is why the
/// built-in rule is written `(%)` and why its stem is the member name.
pub fn archive_search_name(name: &Bytes) -> Option<Bytes> {
    let open = name.iter().position(|byte| *byte == b'(')?;
    is_archive_name(name).then(|| name.slice(open..))
}

/// Whether this is the `(member)` form the implicit search's archive pass
/// matches against, rather than an ordinary filename.
///
/// It is the shape [`archive_search_name`] produces: a leading `(` and a
/// trailing `)`. GNU Make does not test for it — the archive pass sets
/// `lastslash = 0` because it knows which pass it is in — but here the name is
/// all the match is given, so the shape is what says so.
pub fn is_archive_search_name(name: &[u8]) -> bool {
    name.len() > 2 && name.first() == Some(&b'(') && name.last() == Some(&b')')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_archive_shape_is_recognised_where_gnu_make_recognises_it() {
        assert!(is_archive_name(b"lib.a(foo.o)"));
        assert!(is_archive_name(b"sub/lib.a(foo.o)"));
        assert!(is_archive_name(b"lib.a(d/foo.o)"));
        // A name that only looks like one.
        assert!(!is_archive_name(b"lib.a()"));
        assert!(!is_archive_name(b"(foo.o)"));
        assert!(!is_archive_name(b"lib.a(foo.o"));
        assert!(!is_archive_name(b"plain.o"));
        assert!(!is_archive_name(b"lib.a(foo.o).bak"));
    }

    #[test]
    fn the_doubled_form_is_the_one_gnu_make_refuses() {
        assert!(is_nested_archive_name(b"lib.a((foo.o))"));
        assert!(!is_nested_archive_name(b"lib.a(foo.o)"));
        assert!(!is_nested_archive_name(b"lib.a((foo.o)"));
    }

    #[test]
    fn the_two_halves_come_apart_at_the_first_parenthesis() {
        assert_eq!(
            split_archive_name(b"lib.a(foo.o)"),
            Some((&b"lib.a"[..], &b"foo.o"[..]))
        );
        assert_eq!(
            split_archive_name(b"sub/lib.a(d/foo.o)"),
            Some((&b"sub/lib.a"[..], &b"d/foo.o"[..]))
        );
        assert_eq!(split_archive_name(b"plain.o"), None);
    }

    #[test]
    fn a_prerequisite_contributes_only_its_member() {
        assert_eq!(
            member_or_whole(&Bytes::from_static(b"lib.a(foo.o)")),
            Bytes::from_static(b"foo.o")
        );
        assert_eq!(
            member_or_whole(&Bytes::from_static(b"plain.o")),
            Bytes::from_static(b"plain.o")
        );
    }

    #[test]
    fn the_search_name_keeps_the_parentheses() {
        assert_eq!(
            archive_search_name(&Bytes::from_static(b"lib.a(foo.o)")),
            Some(Bytes::from_static(b"(foo.o)"))
        );
        assert_eq!(archive_search_name(&Bytes::from_static(b"plain.o")), None);
    }
}
