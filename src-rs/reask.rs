//! Putting a recorded ground question back to the ground.
//!
//! Deciding whether a record still describes the tree means asking its
//! questions and encoding the answers the way the read encoded them. So this
//! belongs beside the calls that produce those answers rather than in the
//! front end that stores them: each arm below is the producer's own body with
//! the journal taken out, and the encodings are stated in one crate.
//!
//! A `$(shell)` is a launch, not a lookup. Its answer is a function of the
//! command, the working directory, the shell the makefile named, and the
//! environment the makefile had exported by that point — and the last of those
//! is recorded nowhere. A launch under a different environment is a different
//! launch, so one that AGREED would be evidence of nothing. It answers
//! [`Reasked::Unanswerable`], and a caller meeting one reads the makefiles
//! again, which runs the command properly: in order, in the right place, under
//! the environment the read itself builds.
//!
//! A `$(file <)` whose file is now unreadable for any reason but absence
//! answers the same way. That is the read raising rather than answering, and
//! the read is what must raise it.

use crate::bytes::Bytes;
use crate::session::{GroundQuestion, Session};
use crate::strutil::{WordWriter, word_scanner};
use bytes::{BufMut, BytesMut};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;

/// The ground's answer, or the absence of one this can compare.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reasked {
    /// The answer, encoded as the read that asked it would have recorded it.
    Answered(Bytes),
    /// Nothing this can compare. The unit has to be read again.
    Unanswerable,
}

/// Ask one recorded question again.
///
/// `asked` is what the record holds, which is already expanded: a makefile
/// variable inside the call became text before the question was ever asked, so
/// nothing here needs a variable scope. The working directory does have to be
/// the one the unit was read in — `$(wildcard *.c)` is a question about a
/// directory — and that is the caller's to arrange.
#[must_use]
pub fn reask(session: &Session, question: GroundQuestion, asked: &Bytes) -> Reasked {
    match question {
        GroundQuestion::Shell => Reasked::Unanswerable,
        GroundQuestion::Wildcard => Reasked::Answered(wildcard(session, asked)),
        GroundQuestion::RealPath => Reasked::Answered(realpath(asked)),
        GroundQuestion::FileRead => file_read(asked),
        GroundQuestion::Glob => Reasked::Answered(glob(session, asked)),
        GroundQuestion::Include => Reasked::Answered(include(session, asked)),
    }
}

/// `$(wildcard)`: every word globbed, the matches written out space-separated.
fn wildcard(session: &Session, pat: &Bytes) -> Bytes {
    let mut answer = BytesMut::new();
    let mut words = WordWriter::new(&mut answer);
    for token in word_scanner(pat) {
        let token = pat.slice_ref(token);
        if let Ok(files) = session.glob(token).as_ref() {
            for file in files {
                words.write(file);
            }
        }
    }
    answer.freeze()
}

/// `$(realpath)`: every word canonicalised, the ones that resolve written out.
fn realpath(text: &Bytes) -> Bytes {
    let mut answer = BytesMut::new();
    let mut words = WordWriter::new(&mut answer);
    for token in word_scanner(text) {
        if let Ok(path) = std::fs::canonicalize(OsStr::from_bytes(token)) {
            words.write(path.as_os_str().as_bytes());
        }
    }
    answer.freeze()
}

/// `$(file <)`: the file's bytes with one line terminator taken off.
///
/// A file that is not there reads as nothing, which is the function's own rule
/// rather than a failure. Anything else the system refuses IS a failure, and
/// the read is where it is raised — so it is not answered here.
fn file_read(name: &Bytes) -> Reasked {
    let mut file = match std::fs::File::open(OsStr::from_bytes(name)) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Reasked::Answered(Bytes::new());
        }
        Err(_) => return Reasked::Unanswerable,
    };
    let mut buf = Vec::new();
    if std::io::Read::read_to_end(&mut file, &mut buf).is_err() {
        return Reasked::Unanswerable;
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
    }
    Reasked::Answered(Bytes::from(buf))
}

/// A target or prerequisite word holding a metacharacter: the names it
/// matched, NUL-separated, and nothing at all where it matched none.
fn glob(session: &Session, word: &Bytes) -> Bytes {
    let matched = match session.glob(word.clone()).as_ref() {
        Ok(paths) if !paths.is_empty() => paths.clone(),
        _ => Vec::new(),
    };
    let mut answer = BytesMut::new();
    for (position, name) in matched.iter().enumerate() {
        if position > 0 {
            answer.put_u8(0);
        }
        answer.put_slice(name);
    }
    answer.freeze()
}

/// Which files one word of an `include` line reaches, search and all.
///
/// The names NUL-separated, or a leading NUL and the system's own reason where
/// there are none — a name is never empty, so nothing else begins that way.
fn include(session: &Session, word: &Bytes) -> Bytes {
    let pat = crate::eval::at_include_dirs(session, word.clone());
    let globbed = session.glob(pat);
    let (files, unread) = match globbed.as_ref() {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            (Vec::new(), Some(crate::eval::absent()))
        }
        Err(err) => (Vec::new(), Some(crate::strerror(err))),
        Ok(files) if files.is_empty() => (Vec::new(), Some(crate::eval::absent())),
        Ok(files) => (files.clone(), None),
    };
    let mut answer = BytesMut::new();
    if let Some(reason) = &unread {
        answer.put_u8(0);
        answer.put_slice(reason.as_bytes());
    } else {
        for (position, name) in files.iter().enumerate() {
            if position > 0 {
                answer.put_u8(0);
            }
            answer.put_slice(name);
        }
    }
    answer.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Evaluator;

    /// A directory of this test's own, with something for each question to
    /// find.
    fn scratch(name: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "kati-reask-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(directory.join("sub")).unwrap();
        std::fs::write(directory.join("a.c"), "").unwrap();
        std::fs::write(directory.join("b.c"), "").unwrap();
        std::fs::write(directory.join("data.txt"), "hello\n").unwrap();
        std::fs::write(directory.join("sub").join("inc.mk"), "INCLUDED := yes\n").unwrap();
        directory
    }

    fn evaluate(session: Session, source: &str) -> Evaluator {
        let mut ev = Evaluator::new(session);
        let statements = crate::parser::parse_buf(
            &mut ev.session,
            &Bytes::from(source.as_bytes().to_vec()),
            crate::loc::Loc::default(),
        )
        .unwrap();
        let statements = statements.lock().clone();
        for statement in statements {
            statement.eval(&mut ev).unwrap();
        }
        ev
    }

    /// Every recorded answer, put back to a ground that has not moved, comes
    /// back byte for byte. This is the whole contract: an encoding that drifts
    /// from the producer's would report a settled tree as changed, or worse,
    /// compare two spellings of the same answer and call them equal.
    #[test]
    fn every_answerable_question_reproduces_its_answer() {
        let directory = scratch("roundtrip");
        let at = |name: &str| directory.join(name).display().to_string();
        // The last line is a rule whose TARGET holds a metacharacter, which is
        // where the fifth kind is asked: `glob_word` matches it against the
        // filesystem as the statement is evaluated.
        let source = format!(
            "W := $(wildcard {})\nR := $(realpath {})\nF := $(file <{})\ninclude {}\n{}:\n\t@echo x\n",
            at("*.c"),
            at("a.c"),
            at("data.txt"),
            at("sub/inc.mk"),
            at("*.c"),
        );
        let mut ev = evaluate(Session::new(), &source);
        let recorded = ev.session.ground_journal.close_read();
        for kind in [
            GroundQuestion::Wildcard,
            GroundQuestion::RealPath,
            GroundQuestion::FileRead,
            GroundQuestion::Include,
            GroundQuestion::Glob,
        ] {
            assert!(
                recorded.iter().any(|answer| answer.question == kind),
                "{kind:?} was not asked, so nothing below checks it"
            );
        }

        // Otherwise every answer below comes back out of the memo that
        // produced it, and the test proves nothing about the encoding.
        ev.session.clear_glob_cache();

        for answer in &recorded {
            assert_eq!(
                reask(&ev.session, answer.question, &answer.asked),
                Reasked::Answered(answer.answer.clone()),
                "{:?} of {:?}",
                answer.question,
                String::from_utf8_lossy(&answer.asked)
            );
        }
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// And it notices when the ground does move, which is the other half.
    #[test]
    fn a_wildcard_that_gained_a_file_answers_anew() {
        let directory = scratch("moved");
        let source = format!("W := $(wildcard {})\n", directory.join("*.c").display());
        let mut ev = evaluate(Session::new(), &source);
        let recorded = ev.session.ground_journal.close_read();
        let asked = &recorded[0];

        std::fs::write(directory.join("c.c"), "").unwrap();
        ev.session.clear_glob_cache();
        let now = reask(&ev.session, asked.question, &asked.asked);
        assert_ne!(now, Reasked::Answered(asked.answer.clone()));
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// A `$(shell)` is a launch whose environment nothing recorded, so it is
    /// never answered here — the caller reads the makefiles again instead.
    #[test]
    fn a_shell_is_never_answered_here() {
        let session = Session::new();
        assert_eq!(
            reask(
                &session,
                GroundQuestion::Shell,
                &Bytes::from_static(b"echo hi")
            ),
            Reasked::Unanswerable
        );
    }

    /// An `include` that found nothing records the reason, and a run over a
    /// tree where the file has arrived must not reproduce it.
    #[test]
    fn an_include_that_arrived_answers_anew() {
        let directory = scratch("arrived");
        let name = directory.join("late.mk");
        let source = format!("-include {}\n", name.display());
        let mut ev = evaluate(Session::new(), &source);
        let recorded = ev.session.ground_journal.close_read();
        let asked = recorded
            .iter()
            .find(|answer| answer.question == GroundQuestion::Include)
            .expect("the include asked");
        assert_eq!(asked.answer.first(), Some(&0), "recorded as not found");

        std::fs::write(&name, "V := 1\n").unwrap();
        ev.session.clear_glob_cache();
        assert_eq!(
            reask(&ev.session, asked.question, &asked.asked),
            Reasked::Answered(Bytes::from(name.display().to_string().into_bytes())),
            "the file arriving is a different answer"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }
}
