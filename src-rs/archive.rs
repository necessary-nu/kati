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

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use bytes::{BufMut, Bytes, BytesMut};

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

/// Reopen the archive groups in one rule's worth of file names.
///
/// `libf.a(x.o y.o z.o)` is one word to the tokenizer and three names to GNU
/// Make. `parse_file_seq` (reference/gnumake/src/read.c) keeps the text up to
/// and including the `(` as a prefix and puts each following word inside a
/// pair of its own, so the group above becomes `libf.a(x.o)`, `libf.a(y.o)`
/// and `libf.a(z.o)` — which is why that spelling and the three written out
/// longhand behave identically.
///
/// Three details are the whole of the shape, and each is GNU Make's:
///
/// * A group only opens when some **later** word ends in `)`. Without one the
///   text is not a group at all and every word is left exactly as written, so
///   `lib.a(a.o b.o` names a file called `lib.a(a.o` and another called `b.o`.
/// * A word that is only the opening `lib.a(`, or only the closing `)`, is a
///   separator rather than a member, and contributes no name. That is what
///   makes `lib.a( a.o b.o )` mean what `lib.a(a.o b.o)` means.
/// * A word already ending in `)` closes the group; any other word is given a
///   closing parenthesis of its own and the group stays open.
///
/// A word beginning with `(` never opens a group, because the member name
/// would be empty.
pub fn reopen_groups(words: Vec<Bytes>) -> Vec<Bytes> {
    // GNU Make reaches this only for a name holding a `(`; the caller keeps
    // that check, because the overwhelming majority of rules hold none.
    let mut names = Vec::with_capacity(words.len());
    // `Some(prefix)` is GNU Make's `tp > tmpbuf`: a group is open and every
    // word is a member of it.
    let mut prefix: Option<Bytes> = None;

    for (index, word) in words.iter().enumerate() {
        let mut member = word.clone();
        if prefix.is_none() {
            if word.is_empty() || word.first() == Some(&b'(') || word.last() == Some(&b')') {
                names.push(member);
                continue;
            }
            let Some(open) = word.iter().position(|byte| *byte == b'(') else {
                names.push(member);
                continue;
            };
            // A valid group MUST have a word ending in `)` still to come.
            if !words[index + 1..]
                .iter()
                .any(|later| later.last() == Some(&b')'))
            {
                names.push(member);
                continue;
            }
            prefix = Some(word.slice(..=open));
            member = word.slice(open + 1..);
            // The word was the bare `lib.a(`, so it names no member.
            if member.is_empty() {
                continue;
            }
        }

        let open = prefix.clone().expect("a group is open");
        if member.last() == Some(&b')') {
            prefix = None;
            // The word was the bare `)`, which closes the group and no more.
            if member.len() == 1 {
                continue;
            }
            names.push(join(&open, &member, b""));
        } else {
            names.push(join(&open, &member, b")"));
        }
    }
    names
}

fn join(prefix: &[u8], member: &[u8], suffix: &[u8]) -> Bytes {
    let mut name = BytesMut::with_capacity(prefix.len() + member.len() + suffix.len());
    name.put_slice(prefix);
    name.put_slice(member);
    name.put_slice(suffix);
    name.freeze()
}

/// Whether a member name is a pattern to match against an archive's index.
///
/// GNU Make's `ar_glob_pattern_p` (reference/gnumake/src/ar.c), which is
/// deliberately not the same test as the one for a filename: a `[` counts only
/// once a `]` follows it, so `lib.a(m[13.o)` names a member spelt that way
/// rather than globbing. A backslash quotes the character after it.
pub fn member_is_a_pattern(pattern: &[u8]) -> bool {
    let mut index = 0usize;
    let mut opened = false;
    while let Some(&byte) = pattern.get(index) {
        match byte {
            b'?' | b'*' => return true,
            b'\\' => index += 1,
            b'[' => opened = true,
            b']' if opened => return true,
            _ => {}
        }
        index += 1;
    }
    false
}

/// One member of an archive, as the scan met it.
pub struct Member<'a> {
    /// The member's name, with the long-name table and the 4.4BSD extended
    /// form already resolved.
    pub name: &'a [u8],
    /// The seconds-since-epoch the index records, which is zero for an archive
    /// `ar` wrote in its default deterministic mode.
    pub date: i64,
    /// Where this member's 60-byte header begins, for a caller that means to
    /// write into it.
    pub position: u64,
    /// The name came out of the fixed header field, which keeps only its first
    /// [`NAME_KEPT`] bytes — so that is all of it a comparison may use.
    pub truncated: bool,
}

/// Why a walk of an archive's index stopped before it reached the end.
///
/// The two are kept apart because a caller that writes says two different
/// things about them: an archive that is not there is a build that has not
/// reached it yet, and one that does not parse is a file that is not an
/// archive at all. A caller that only reads folds both into "no such member",
/// which is what GNU Make's `ar_scan` reports with -1 and -2.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanFailure {
    NoArchive,
    NotAnArchive,
}

/// Bytes of the magic that opens an archive.
const MAGIC: &[u8; 8] = b"!<arch>\n";
/// Bytes of one member header.
const HEADER: usize = 60;
/// How much of a member's name the fixed header field keeps, which is all a
/// short name is ever compared over.
const NAME_KEPT: usize = 15;
/// Where a member's date sits within its header.
const DATE_OFFSET: u64 = 16;
/// How wide the date field is. A date written into it is padded out to the
/// whole width, so nothing of a longer previous date survives behind it.
const DATE_FIELD: usize = 12;

/// Walk an archive's index, offering each member to `visit` and stopping at
/// the first answer it gives.
///
/// GNU Make's `ar_scan` (reference/gnumake/src/arscan.c), which is one walk
/// with a callback over it and not three walks that happen to agree: the
/// member names the front end globs against, the date the build compares, and
/// the header a `-t` writes into are all read out of the same headers, and the
/// format is delicate enough that a second reader of it is a second thing to
/// remember when the format is wrong.
///
/// Only the SysV/GNU format `ar` writes here is read — the `!<arch>\n` magic,
/// fixed 60-byte member headers, the `//` long-name table for names too long
/// for the header, and 4.4BSD's `#1/LEN` extended names. A short read where a
/// header would begin is the end of the archive; anything else that does not
/// parse is [`ScanFailure::NotAnArchive`], which is `ar_scan`'s `goto invalid`.
///
/// The file is opened for writing when `write` is set, because a caller that
/// means to write into a header must hold the same handle the walk found it
/// with: reopening would be a second walk.
fn ar_scan<T>(
    archive: &Path,
    write: bool,
    mut visit: impl FnMut(&Member<'_>, &mut std::fs::File) -> Option<T>,
) -> Result<Option<T>, ScanFailure> {
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(write)
        .open(archive)
        .map_err(|_| ScanFailure::NoArchive)?;
    let mut magic = [0u8; MAGIC.len()];
    if file.read_exact(&mut magic).is_err() || &magic != MAGIC {
        return Err(ScanFailure::NotAnArchive);
    }

    let mut long_names: Vec<u8> = Vec::new();
    let mut header = [0u8; HEADER];
    let mut position = MAGIC.len() as u64;
    loop {
        // A short read where a header would start is the end of the archive
        // rather than a failure: every member found before it is still one.
        if file.read_exact(&mut header).is_err() {
            return Ok(None);
        }
        if &header[58..60] != b"`\n" {
            return Err(ScanFailure::NotAnArchive);
        }
        let Some(size) = parse_field::<i64>(&header[48..58]).filter(|size| *size >= 0) else {
            return Err(ScanFailure::NotAnArchive);
        };
        // Every member's data is padded to an even offset, so what the next
        // header costs is its own width plus the data plus that pad byte.
        let Some(entry) = u64::try_from(size)
            .ok()
            .and_then(|size| size.checked_add(HEADER as u64 + size % 2))
        else {
            return Err(ScanFailure::NotAnArchive);
        };
        let date = parse_field::<i64>(&header[16..28]).unwrap_or(0);
        let raw = trim_trailing(&header[..16]);

        // The long-name table is a member like any other, and is read for its
        // data rather than named as one.
        if raw == b"//" || raw == b"ARFILENAMES/" {
            let Ok(length) = usize::try_from(size) else {
                return Err(ScanFailure::NotAnArchive);
            };
            long_names.resize(length, 0);
            if file.read_exact(&mut long_names).is_err() {
                return Err(ScanFailure::NotAnArchive);
            }
            position += entry;
            if file.seek(SeekFrom::Start(position)).is_err() {
                return Err(ScanFailure::NotAnArchive);
            }
            continue;
        }

        let mut extended = Vec::new();
        let name: &[u8] = if raw.first() == Some(&b'/') || raw.first() == Some(&b' ') {
            // GNU `ar`: an offset into the long-name table.
            let Some(table) = parse_field::<usize>(&raw[1..]).and_then(|at| long_names.get(at..))
            else {
                return Err(ScanFailure::NotAnArchive);
            };
            let end = table
                .iter()
                .position(|byte| *byte == b'\n' || *byte == b'\0')
                .unwrap_or(table.len());
            trim_trailing_slash(&table[..end])
        } else if raw.starts_with(b"#1/") {
            // 4.4BSD: the real name is the first bytes of the member's data.
            let Some(length) = parse_field::<usize>(&raw[3..]) else {
                return Err(ScanFailure::NotAnArchive);
            };
            extended.resize(length, 0);
            if file.read_exact(&mut extended).is_err() {
                return Err(ScanFailure::NotAnArchive);
            }
            let end = extended
                .iter()
                .position(|byte| *byte == b'\0')
                .unwrap_or(extended.len());
            extended.truncate(end);
            &extended
        } else {
            trim_trailing_slash(raw)
        };

        let member = Member {
            name,
            date,
            position,
            truncated: extended.is_empty() && !raw.starts_with(b"/"),
        };
        if let Some(answer) = visit(&member, &mut file) {
            return Ok(Some(answer));
        }

        // Sought absolutely rather than stepped over: the 4.4BSD form has
        // already taken its name out of the data, and a visitor that wrote into
        // a header has moved the handle again, so where the next header begins
        // is the only thing either of them still agrees about.
        position += entry;
        if file.seek(SeekFrom::Start(position)).is_err() {
            return Err(ScanFailure::NotAnArchive);
        }
    }
}

/// GNU Make's `ar_name_equal`. A name that came out of the fixed header field
/// is compared over that field's width only, because that is all of it the
/// archive kept, and the directory is dropped first — so `lib.a(d/foo.o)` asks
/// about the entry named `foo.o`.
fn name_matches(wanted: &[u8], member: &Member<'_>) -> bool {
    let wanted = wanted
        .iter()
        .rposition(|byte| *byte == b'/')
        .map_or(wanted, |slash| &wanted[slash + 1..]);
    if !member.truncated {
        return wanted == member.name;
    }
    wanted.get(..NAME_KEPT).unwrap_or(wanted) == member.name.get(..NAME_KEPT).unwrap_or(member.name)
}

/// Every member name the archive's index holds, in the order it holds them.
///
/// The one question the front end's globbing asks of the scan. An archive that
/// does not parse is one with no members, which is what `ar_scan` reports for
/// an invalid archive to a caller that only wanted to match names.
fn member_names(archive: &Path) -> Option<Vec<Bytes>> {
    let mut found = Vec::new();
    let walked = ar_scan(archive, false, |member, _| {
        if !member.name.is_empty() {
            found.push(Bytes::copy_from_slice(member.name));
        }
        None::<()>
    });
    match walked {
        Err(ScanFailure::NoArchive) => None,
        // A malformed archive keeps whatever the walk read before it, which is
        // every member `ar_scan` had reported by the time it gave up.
        Err(ScanFailure::NotAnArchive) | Ok(_) => Some(found),
    }
}

/// The seconds-since-epoch the index records for `member`, or `None` when the
/// archive has no such member — including a member whose recorded date is zero.
///
/// GNU Make's `ar_member_date`, which folds "not found" and "date zero"
/// together: `ar_scan` returns the first non-zero date a matching member has,
/// and a return of zero or less becomes -1, which `f_mtime` reads as
/// nonexistent. That is not a corner case on a modern Linux host — `ar`
/// defaults to deterministic mode, which writes every member's date as zero, so
/// every member of an archive built by plain `ar -rv` reads as out of date, and
/// GNU Make 4.4.1 re-runs `$(AR) $(ARFLAGS)` on every invocation for exactly
/// that reason.
pub fn member_date(archive: &Path, member: &[u8]) -> Option<i64> {
    ar_scan(archive, false, |found, _| {
        (name_matches(member, found) && found.date > 0).then_some(found.date)
    })
    .ok()
    .flatten()
}

/// Write the archive's own modification time into `member`'s index entry, which
/// is the only way a member's date can be set: it is not a file, so there is
/// nothing to `utime`.
///
/// GNU Make's `ar_member_touch` (reference/gnumake/src/arscan.c:923), which
/// finds the header and formats a date into the 12-byte `ar_date` field in
/// place. Three details of it are load-bearing and none of them is obvious:
///
/// * the date written is the ARCHIVE's, not the wall clock, and it is read
///   before the write — so the member ends a touch fractionally older than the
///   archive holding it, which is what GNU Make settles for and what a
///   subsequent read then finds current;
/// * the field is decimal seconds padded to its full width with spaces, because
///   a shorter number left over from a longer one would leave the tail of the
///   old date behind it;
/// * on an archive `ar` wrote in its default deterministic mode every date is
///   zero, so every member reads as absent — and a touch is then the only way a
///   real date ever gets into that index at all.
///
/// Unlike [`member_date`], a member whose recorded date is zero still counts as
/// found: a touch is precisely the thing that gives such a member a date.
pub fn touch_member(archive: &Path, member: &[u8]) -> Result<(), TouchFailure> {
    use std::io::Write as _;

    let modified = std::fs::metadata(archive)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_secs());
    let mut field = [b' '; DATE_FIELD];
    let text = modified.to_string();
    let written = text.as_bytes();
    if written.len() <= DATE_FIELD {
        field[..written.len()].copy_from_slice(written);
    }

    let touched = ar_scan(archive, true, |found, file| {
        if !name_matches(member, found) {
            return None;
        }
        Some(
            file.seek(SeekFrom::Start(found.position + DATE_OFFSET))
                .and_then(|_| file.write_all(&field)),
        )
    });
    match touched {
        Ok(Some(Ok(()))) => Ok(()),
        Ok(Some(Err(_))) => Err(TouchFailure::NotAnArchive),
        Ok(None) => Err(TouchFailure::NoMember),
        Err(ScanFailure::NoArchive) => Err(TouchFailure::NoArchive),
        Err(ScanFailure::NotAnArchive) => Err(TouchFailure::NotAnArchive),
    }
}

/// Why a member's date could not be written.
///
/// The three are separate because GNU Make says three different things and
/// distinguishing them is the whole information the diagnostic carries: a
/// missing archive is a build that has not reached this yet, a missing member
/// is a name that is wrong, and an unreadable one is a file that is not an
/// archive at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TouchFailure {
    NoArchive,
    NotAnArchive,
    NoMember,
}

/// An archive member header field, which is left-aligned ASCII padded with
/// spaces and may be entirely blank.
fn parse_field<T: TryFrom<i64>>(field: &[u8]) -> Option<T> {
    let text = std::str::from_utf8(trim_trailing(field)).ok()?;
    if text.is_empty() {
        return T::try_from(0).ok();
    }
    T::try_from(text.parse::<i64>().ok()?).ok()
}

fn trim_trailing(field: &[u8]) -> &[u8] {
    let end = field
        .iter()
        .rposition(|byte| *byte != b' ' && *byte != 0)
        .map_or(0, |last| last + 1);
    &field[..end]
}

const fn trim_trailing_slash(name: &[u8]) -> &[u8] {
    match name.split_last() {
        Some((b'/', rest)) => rest,
        _ => name,
    }
}

/// The members of `archive` matching `pattern`, sorted, or `None` when the
/// pattern is not one or nothing matched.
///
/// GNU Make's `ar_glob`. Two things about it are worth stating rather than
/// inferring, because both are visible in what a build does:
///
/// * The pattern is matched against the archive's **index**, not against the
///   filesystem. An object sitting beside the archive and not filed in it does
///   not match, and a member whose object has been deleted still does.
/// * `fnmatch` is called with `FNM_PATHNAME|FNM_PERIOD`, so `*.o` does not
///   match a member called `.hidden.o` — a leading period has to be written.
///
/// `None` rather than an empty list for no matches, because GNU Make then uses
/// the pattern as the member's literal name and lets the build refuse over it.
pub fn glob_members(archive: &[u8], pattern: &[u8]) -> Option<Vec<Bytes>> {
    use std::os::unix::ffi::OsStrExt as _;

    if !member_is_a_pattern(pattern) {
        return None;
    }
    let names = member_names(Path::new(std::ffi::OsStr::from_bytes(archive)))?;
    let pattern = std::ffi::CString::new(pattern).ok()?;
    let mut matched: Vec<Bytes> = names
        .into_iter()
        .filter(|name| {
            crate::fileutil::fnmatch(&pattern, name, libc::FNM_PATHNAME | libc::FNM_PERIOD)
        })
        .collect();
    if matched.is_empty() {
        return None;
    }
    // GNU Make sorts the whole `lib.a(member)` names it built with
    // `alpha_compare`, which is `strcmp` once the first byte ties. The archive
    // half is the same for all of them, so sorting the members is the same
    // order.
    matched.sort_unstable();
    Some(matched)
}

/// Rejoin an archive and a member into the one name the rest of the front end
/// reads.
pub fn member_name(archive: &[u8], member: &[u8]) -> Bytes {
    let mut name = BytesMut::with_capacity(archive.len() + member.len() + 2);
    name.put_slice(archive);
    name.put_u8(b'(');
    name.put_slice(member);
    name.put_u8(b')');
    name.freeze()
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

    fn reopened(text: &str) -> Vec<String> {
        let words = text
            .split_whitespace()
            .map(|word| Bytes::copy_from_slice(word.as_bytes()))
            .collect();
        reopen_groups(words)
            .into_iter()
            .map(|name| String::from_utf8(name.to_vec()).unwrap())
            .collect()
    }

    #[test]
    fn one_pair_of_parentheses_can_name_several_members() {
        assert_eq!(reopened("lib.a(a.o b.o)"), ["lib.a(a.o)", "lib.a(b.o)"]);
        assert_eq!(
            reopened("lib.a(a.o   b.o\tc.o)"),
            ["lib.a(a.o)", "lib.a(b.o)", "lib.a(c.o)"]
        );
        // The two spellings a makefile is entitled to use interchangeably.
        assert_eq!(
            reopened("lib.a(a.o) lib.a(b.o)"),
            reopened("lib.a(a.o b.o)")
        );
    }

    #[test]
    fn a_bare_parenthesis_separates_rather_than_names() {
        assert_eq!(reopened("lib.a( a.o b.o)"), ["lib.a(a.o)", "lib.a(b.o)"]);
        assert_eq!(reopened("lib.a(a.o b.o )"), ["lib.a(a.o)", "lib.a(b.o)"]);
        assert_eq!(reopened("lib.a( a.o b.o )"), ["lib.a(a.o)", "lib.a(b.o)"]);
    }

    #[test]
    fn a_group_that_never_closes_is_not_a_group() {
        // No later word ends in `)`, so nothing is reopened and both words are
        // the names the makefile wrote.
        assert_eq!(reopened("lib.a(a.o b.o"), ["lib.a(a.o", "b.o"]);
        assert_eq!(reopened("lib.a("), ["lib.a("]);
    }

    #[test]
    fn a_closed_group_lets_the_words_after_it_alone() {
        assert_eq!(
            reopened("lib.a(a.o b.o) plain.txt"),
            ["lib.a(a.o)", "lib.a(b.o)", "plain.txt"]
        );
        assert_eq!(
            reopened("lib.a(a.o b.o) other.a(c.o)"),
            ["lib.a(a.o)", "lib.a(b.o)", "other.a(c.o)"]
        );
    }

    #[test]
    fn a_name_beginning_with_a_parenthesis_opens_nothing() {
        assert_eq!(reopened("(a.o b.o)"), ["(a.o", "b.o)"]);
    }

    #[test]
    fn a_member_pattern_is_the_one_gnu_make_globs() {
        assert!(member_is_a_pattern(b"*.o"));
        assert!(member_is_a_pattern(b"m?.o"));
        assert!(member_is_a_pattern(b"m[13].o"));
        // `[` alone is not a pattern until a `]` closes it.
        assert!(!member_is_a_pattern(b"m[13.o"));
        assert!(!member_is_a_pattern(b"plain.o"));
        // A backslash quotes the character after it.
        assert!(!member_is_a_pattern(br"m\*.o"));
    }

    /// A directory of this test's own, named for the test rather than counted,
    /// so two tests never read each other's archives. Removed again when the
    /// test ends.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("kati-archive-{}-{test}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("a scratch directory");
            Self(path)
        }

        /// Built by `ar` on this host rather than by hand, so the reader is
        /// measured against the format that actually reaches it.
        ///
        /// `ar` defaults to deterministic mode here, which writes every
        /// member's date as zero; `-rcU` is what records a real one.
        fn written(&self, flags: &str, members: &[&str]) -> std::path::PathBuf {
            for member in members {
                std::fs::write(self.0.join(member), b"body\n").unwrap();
            }
            let path = self.0.join("lib.a");
            let ok = std::process::Command::new("ar")
                .arg(flags)
                .arg(&path)
                .args(members)
                .current_dir(&self.0)
                .status()
                .unwrap()
                .success();
            assert!(ok);
            path
        }

        fn archive(&self, members: &[&str]) -> Vec<u8> {
            self.written("-rc", members)
                .into_os_string()
                .into_encoded_bytes()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn the_index_is_what_a_member_pattern_is_matched_against() {
        let scratch = Scratch::new("index-not-filesystem");
        let archive = scratch.archive(&["m3.o", "m1.o", "m2.o"]);
        // Filed after the archive was written, so it is on the filesystem and
        // not in the index — and must not match.
        std::fs::write(scratch.0.join("q.o"), b"body\n").unwrap();
        let archive = archive.as_slice();

        assert_eq!(
            glob_members(archive, b"*.o"),
            Some(vec![
                Bytes::from_static(b"m1.o"),
                Bytes::from_static(b"m2.o"),
                Bytes::from_static(b"m3.o"),
            ]),
            "sorted, and the index's own order is not it"
        );
        assert_eq!(
            glob_members(archive, b"m[13].o"),
            Some(vec![
                Bytes::from_static(b"m1.o"),
                Bytes::from_static(b"m3.o")
            ])
        );
        assert_eq!(glob_members(archive, b"*.zzz"), None);
        assert_eq!(glob_members(archive, b"m1.o"), None, "not a pattern");
        assert_eq!(glob_members(b"/nonexistent/lib.a", b"*.o"), None);
    }

    #[test]
    fn a_long_member_name_comes_out_of_the_long_name_table() {
        let scratch = Scratch::new("long-names");
        let archive = scratch.archive(&["short.o", "a-very-long-member-name.o"]);
        assert_eq!(
            glob_members(&archive, b"*.o"),
            Some(vec![
                Bytes::from_static(b"a-very-long-member-name.o"),
                Bytes::from_static(b"short.o"),
            ])
        );
    }

    #[test]
    fn a_leading_period_has_to_be_written_out() {
        let scratch = Scratch::new("leading-period");
        let archive = scratch.archive(&[".hidden.o", "plain.o"]);
        // FNM_PERIOD: `*` does not match the leading period.
        assert_eq!(
            glob_members(&archive, b"*.o"),
            Some(vec![Bytes::from_static(b"plain.o")])
        );
        assert_eq!(
            glob_members(&archive, b".*.o"),
            Some(vec![Bytes::from_static(b".hidden.o")])
        );
    }

    #[test]
    fn a_file_that_is_not_an_archive_holds_no_members() {
        let scratch = Scratch::new("not-an-archive");
        let path = scratch.0.join("not-an-archive");
        std::fs::write(&path, b"just bytes\n").unwrap();
        assert_eq!(
            glob_members(
                path.into_os_string().into_encoded_bytes().as_slice(),
                b"*.o"
            ),
            None
        );
    }

    /// The cell the deterministic default makes interesting: `ar` wrote every
    /// date as zero, so every member reads as absent, and GNU Make refiles the
    /// archive on every invocation because of it.
    #[test]
    fn a_deterministic_archive_records_no_date() {
        let scratch = Scratch::new("deterministic-date");
        let path = scratch.written("-rc", &["member.o", "a-very-long-member-name.o"]);
        assert_eq!(member_date(&path, b"member.o"), None);
    }

    #[test]
    fn a_dated_archive_answers_both_name_lengths() {
        let scratch = Scratch::new("dated");
        let path = scratch.written("-rcU", &["member.o", "a-very-long-member-name.o"]);
        assert!(member_date(&path, b"member.o").is_some_and(|date| date > 0));
        assert!(
            member_date(&path, b"a-very-long-member-name.o").is_some_and(|date| date > 0),
            "a name too long for the header comes out of the long-name table"
        );
        assert_eq!(member_date(&path, b"absent.o"), None);
        assert!(
            member_date(&path, b"sub/member.o").is_some(),
            "the directory is dropped before the comparison, as ar_name_equal does"
        );
    }

    #[test]
    fn a_non_archive_answers_for_no_member() {
        let scratch = Scratch::new("dateless");
        let path = scratch.0.join("not-an-archive");
        std::fs::write(&path, b"just bytes\n").unwrap();
        assert_eq!(member_date(&path, b"member.o"), None);
        assert_eq!(member_date(&scratch.0.join("absent.a"), b"m.o"), None);
    }

    /// A touch is the only thing that will ever put a real date into the index
    /// of an archive `ar` wrote in its default mode, and the date it writes is
    /// the archive's own rather than the wall clock's.
    // [spec:ronin:req:make.semantics+1/test]
    #[test]
    fn a_touch_files_the_archives_own_date() {
        let scratch = Scratch::new("touch-date");
        let path = scratch.written("-rc", &["member.o", "a-very-long-member-name.o"]);
        assert_eq!(member_date(&path, b"member.o"), None);

        touch_member(&path, b"member.o").unwrap();

        let archive_date = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let dated = member_date(&path, b"member.o").expect("the touch filed a date");
        // GNU Make reads the archive's mtime before it writes, and writing then
        // moves that mtime on, so the member ends fractionally behind the file
        // holding it rather than level with it.
        assert!(
            dated > 0 && u64::try_from(dated).unwrap() <= archive_date,
            "the member was dated {dated} against an archive of {archive_date}"
        );
        assert_eq!(
            member_date(&path, b"a-very-long-member-name.o"),
            None,
            "a touch dated a member nobody asked about"
        );
    }

    /// A name too long for the fixed header is found through the long-name
    /// table, and the header the touch writes into is still that member's own —
    /// which is the half of the walk a scan that only reads never exercises.
    // [spec:ronin:req:make.semantics+1/test]
    #[test]
    fn a_touch_reaches_a_long_named_member() {
        let scratch = Scratch::new("touch-long-name");
        let path = scratch.written("-rc", &["member.o", "a-very-long-member-name.o"]);

        touch_member(&path, b"a-very-long-member-name.o").unwrap();

        assert!(member_date(&path, b"a-very-long-member-name.o").is_some_and(|date| date > 0));
        assert_eq!(member_date(&path, b"member.o"), None);
    }

    /// The three ways a touch has nothing to write into, which are three
    /// different things to say and not one.
    // [spec:ronin:req:make.semantics+1/test]
    #[test]
    fn a_touch_says_why_it_failed() {
        let scratch = Scratch::new("touch-failures");
        let path = scratch.written("-rc", &["member.o"]);
        assert_eq!(
            touch_member(&path, b"absent.o"),
            Err(TouchFailure::NoMember)
        );
        assert_eq!(
            touch_member(&scratch.0.join("nope.a"), b"member.o"),
            Err(TouchFailure::NoArchive)
        );
        let bad = scratch.0.join("bad.a");
        std::fs::write(&bad, b"just bytes\n").unwrap();
        assert_eq!(
            touch_member(&bad, b"member.o"),
            Err(TouchFailure::NotAnArchive)
        );
    }
}
