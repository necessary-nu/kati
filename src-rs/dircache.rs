//! Answering "is this name there?" from a directory read rather than a
//! `stat` per name.
//!
//! The implicit-rule search asks about a name for every pattern in the
//! catalogue, and almost every answer is no: a Makefile with `build/0: src/0`
//! makes the search ask after `build/0.c`, `build/0.cc`, `build/0.p`,
//! `build/0.web`, `RCS/build/0,v` and a hundred more that were never going to
//! be there. Measured on a 4,000-rule no-op, Ronin issued 480,137 `statx`
//! calls and 476,122 of them returned `ENOENT`. GNU Make 4.4.1 ran the same
//! Makefile with 8,010 stats and six `getdents64`: it reads each directory
//! once into a hash and answers every one of those questions out of it
//! (`dir.c`). 488,146 filesystem syscalls against 8,028.
//!
//! This is that hash, with one deliberate difference from GNU Make.
//!
//! IT ONLY EVER PROVES ABSENCE. A name the listing does not hold cannot be
//! opened, so the answer is no and no syscall is needed. A name the listing
//! DOES hold is handed back to the caller to `stat` as before, because a
//! directory entry is not the same claim as a file: a symlink whose target
//! was removed is listed and does not exist, and answering "yes" from the
//! listing would report it present where a `stat` reports it gone. The
//! confirming `stat` costs one call per name that is actually there, which on
//! the workload above is four thousand against the four hundred and seventy-six
//! thousand it removes.
//!
//! IT IS KEPT HONEST BY THE COUNTER THAT WAS ALREADY THERE. A `$(shell)` in a
//! second expansion can create a file the search has already looked for, so a
//! listing may only be trusted across a stretch in which nothing ran. GNU Make
//! keeps `dir.c` honest with `command_count`; this crate's
//! [`crate::session::Session::filesystem_epoch`] is the same counter, already
//! bumped by `note_command_ran` wherever a makefile can reach the disk, and
//! already keeping [`crate::fileutil::GlobCache`] honest. The caller hands it
//! in and the listings are given up whenever it moves. A second counter of
//! this cache's own would be a second thing to remember to bump.
//!
//! A directory that is NOT THERE is an answer rather than an obstacle, and
//! saying so is worth as much as the listings: the built-in catalogue proposes
//! an `SCCS/` and an `RCS/` beside every source it is asked about, almost no
//! tree has either, and treating a missing directory as unknown left half the
//! search's syscalls in place. A directory that exists and cannot be READ —
//! permission, an I/O error — is a different thing and proves nothing.
//!
//! Three shapes are handed straight back to the caller rather than answered:
//! a file part of `.` or `..`, which `read_dir` does not list; a name whose
//! file part is empty or ends in `/`, which is not an entry name; and a
//! directory that could not be read for a reason that is not its absence.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use memchr::memrchr;

use crate::fasthash::{FastMap, FastSet};

type Entries = FastSet<Box<[u8]>>;

/// What a directory turned out to be.
enum Listing {
    /// It was read, and held exactly these names.
    Read(Entries),
    /// It is not there, or is not a directory. Nothing can be inside it, so
    /// every name under it is absent without asking.
    ///
    /// This is not a nicety: the built-in catalogue proposes an `SCCS/` and an
    /// `RCS/` beside every source, and almost no tree has either, so a search
    /// over four thousand targets asks after a hundred and twenty thousand
    /// names in two directories that do not exist.
    Nothing,
    /// It could not be read for some other reason — permission, an I/O error
    /// — and a listing that was not taken proves nothing about what is in it.
    Unknown,
}

/// The directories this search has read, and what was in each.
#[derive(Default)]
pub struct DirectoryCache {
    directories: FastMap<Box<[u8]>, Listing>,
    /// The filesystem epoch the listings were taken at. Any change empties
    /// them; the counter is the session's, not this cache's.
    epoch: u64,
}

impl DirectoryCache {
    /// Whether this name certainly does not exist.
    ///
    /// `epoch` is [`crate::session::Session::filesystem_epoch`], and handing
    /// it in on every call is what makes the listings honest.
    ///
    /// `true` is a proof and the caller may stop. `false` is "ask the
    /// filesystem": it is returned both for a name the listing holds and for
    /// every shape this cache declines to answer, so a caller that treats it
    /// as "exists" is wrong.
    pub fn certainly_absent(&mut self, epoch: u64, name: &[u8]) -> bool {
        if epoch != self.epoch {
            self.directories.clear();
            self.epoch = epoch;
        }
        let (directory, file) = split(name);
        if file.is_empty() || file == b"." || file == b".." || file.last() == Some(&b'/') {
            return false;
        }
        // Asked before it is filled rather than through `entry`, which wants an
        // owned key that it throws away again whenever the directory has
        // already been read — and every call but the first has. Now that the
        // implicit search asks this rather than interning the name it is asking
        // about, it asks it 900,000 times on the workload above, and the six
        // directories those questions are in are read once each: `entry` there
        // is 900,000 allocations to store six keys.
        if !self.directories.contains_key(directory) {
            let listing = read(directory);
            self.directories.insert(directory.into(), listing);
        }
        match &self.directories[directory] {
            Listing::Read(entries) => !entries.contains(file),
            Listing::Nothing => true,
            Listing::Unknown => false,
        }
    }

    /// How many directories have been read, for the tests and for `--stats`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directories.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directories.is_empty()
    }
}

/// The directory part of a name, with its trailing slash, and the file part.
///
/// An empty directory part means the working directory, which `read` spells
/// as `.` rather than as the empty string `read_dir` would reject.
fn split(name: &[u8]) -> (&[u8], &[u8]) {
    // The last byte is held back so that `dir/` splits into `` and `dir/`
    // rather than into `dir/` and ``, which keeps a trailing slash out of the
    // file part's own hands and into the shapes declined above.
    let at = memrchr(b'/', &name[..name.len().saturating_sub(1)]).map_or(0, |at| at + 1);
    name.split_at(at)
}

fn read(directory: &[u8]) -> Listing {
    let path: &Path = Path::new(OsStr::from_bytes(if directory.is_empty() {
        b"."
    } else {
        directory
    }));
    let read = match fs::read_dir(path) {
        Ok(read) => read,
        // The two failures that are an answer rather than an obstacle, and
        // they are the same two `std::fs::exists` reports as "not there": a
        // missing component, and a component that is not a directory.
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            return Listing::Nothing;
        }
        Err(_) => return Listing::Unknown,
    };
    let mut entries = Entries::default();
    for entry in read {
        // A directory that vanishes mid-walk proves nothing about the names
        // it held, so the whole listing is given up rather than half kept.
        let Ok(entry) = entry else {
            return Listing::Unknown;
        };
        entries.insert(entry.file_name().as_bytes().into());
    }
    Listing::Read(entries)
}

#[cfg(test)]
mod tests {
    /// Any value; what matters is that it is the same one until a test moves it.
    const EPOCH: u64 = 7;

    use super::*;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("kati-dircache-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn at(&self, name: &str) -> Vec<u8> {
            self.0.join(name).into_os_string().into_vec()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    use std::os::unix::ffi::OsStringExt as _;

    /// The whole point: a candidate the search invented, which was never going
    /// to be there, answered without a syscall.
    #[test]
    fn an_absent_candidate_is_proven_absent() {
        let scratch = Scratch::new();
        fs::write(scratch.0.join("params.o"), b"x").unwrap();
        let mut cache = DirectoryCache::default();

        assert!(cache.certainly_absent(EPOCH, &scratch.at("params.c")));
        assert!(cache.certainly_absent(EPOCH, &scratch.at("params.web")));
        assert!(!cache.certainly_absent(EPOCH, &scratch.at("params.o")));
        // One directory read, however many names were asked about.
        assert_eq!(cache.len(), 1);
    }

    /// A directory entry is not the same claim as a file. Answering "present"
    /// from the listing would report a broken symlink as existing, where the
    /// `stat` the caller falls through to reports it gone.
    #[test]
    fn a_dangling_symlink_is_not_proven_absent() {
        let scratch = Scratch::new();
        symlink("nowhere", scratch.0.join("link.o")).unwrap();
        let mut cache = DirectoryCache::default();

        assert!(!cache.certainly_absent(EPOCH, &scratch.at("link.o")));
        assert!(!fs::exists(scratch.0.join("link.o")).unwrap());
    }

    /// A directory that is not there cannot hold anything, and the built-in
    /// catalogue proposes two of those — `SCCS/` and `RCS/` — beside every
    /// source it is asked about. Left as "prove nothing", half the search's
    /// syscalls stayed.
    #[test]
    fn a_missing_directory_holds_nothing() {
        let scratch = Scratch::new();
        let mut cache = DirectoryCache::default();
        assert!(cache.certainly_absent(EPOCH, &scratch.at("SCCS/s.params.c")));
        assert!(cache.certainly_absent(EPOCH, &scratch.at("SCCS/s.params.o")));
        assert_eq!(cache.len(), 1);
    }

    /// The same for a path whose directory component is a file, which is what
    /// `std::fs::exists` also reports as not there.
    #[test]
    fn a_file_used_as_a_directory_holds_nothing() {
        let scratch = Scratch::new();
        fs::write(scratch.0.join("notadir"), b"x").unwrap();
        let mut cache = DirectoryCache::default();
        assert!(cache.certainly_absent(EPOCH, &scratch.at("notadir/params.c")));
        assert!(!fs::exists(scratch.0.join("notadir/params.c")).unwrap_or(false));
    }

    /// A directory that exists and cannot be read proves nothing about what is
    /// in it, which is the case that must NOT be folded in with the two above.
    #[test]
    fn an_unreadable_directory_proves_nothing() {
        use std::os::unix::fs::PermissionsExt as _;
        let scratch = Scratch::new();
        let closed = scratch.0.join("closed");
        fs::create_dir(&closed).unwrap();
        fs::write(closed.join("params.c"), b"x").unwrap();
        fs::set_permissions(&closed, fs::Permissions::from_mode(0o000)).unwrap();
        let mut cache = DirectoryCache::default();
        let proven = cache.certainly_absent(EPOCH, &scratch.at("closed/params.c"));
        fs::set_permissions(&closed, fs::Permissions::from_mode(0o755)).unwrap();
        // Skipped rather than asserted when the test runs as root, for whom
        // the mode is not an obstacle and the directory reads fine.
        // SAFETY: `geteuid` reads the calling process's own effective user id.
        // It takes no arguments, touches no memory, cannot fail and is
        // async-signal-safe, so there is no precondition to uphold.
        if unsafe { libc::geteuid() } != 0 {
            assert!(!proven);
        }
    }

    /// `read_dir` does not list `.` or `..`, so a name ending in one must not
    /// be read as missing from the listing that does not mention it.
    #[test]
    fn dot_names_are_declined() {
        let scratch = Scratch::new();
        let mut cache = DirectoryCache::default();
        assert!(!cache.certainly_absent(EPOCH, &scratch.at(".")));
        assert!(!cache.certainly_absent(EPOCH, &scratch.at("..")));
    }

    /// A trailing slash is not part of an entry name.
    #[test]
    fn a_trailing_slash_is_declined() {
        let scratch = Scratch::new();
        fs::create_dir(scratch.0.join("sub")).unwrap();
        let mut cache = DirectoryCache::default();
        let mut name = scratch.at("sub");
        name.push(b'/');
        assert!(!cache.certainly_absent(EPOCH, &name));
    }

    /// The coherence rule. A listing may only be trusted across a stretch in
    /// which nothing ran, and the session's filesystem epoch is what says so.
    #[test]
    fn a_moved_epoch_empties_the_listings() {
        let scratch = Scratch::new();
        let mut cache = DirectoryCache::default();
        assert!(cache.certainly_absent(EPOCH, &scratch.at("made-later")));
        assert_eq!(cache.len(), 1);

        fs::write(scratch.0.join("made-later"), b"x").unwrap();
        // Still believed absent, because nothing has said the disk moved.
        assert!(cache.certainly_absent(EPOCH, &scratch.at("made-later")));

        assert!(!cache.certainly_absent(EPOCH + 1, &scratch.at("made-later")));
    }

    /// The session's counter is the one that moves, so the cache has to follow
    /// a real `note_command_ran` and not only a number the test made up.
    #[test]
    fn a_session_command_moves_the_epoch() {
        let session = crate::session::Session::default();
        let before = session.filesystem_epoch();
        session.note_command_ran();
        assert_ne!(session.filesystem_epoch(), before);
    }

    #[test]
    fn a_name_with_no_directory_reads_the_working_directory() {
        let mut cache = DirectoryCache::default();
        assert!(cache.certainly_absent(EPOCH, b"a-name-nothing-in-this-directory-has"));
        assert!(!cache.is_empty());
    }
}
