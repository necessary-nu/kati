//! The shell's reading of recipe lines, kept for the readings that repeat it.

use std::sync::Arc;

use bytes::Bytes;
use nsh::script::{Reader, Script};

use crate::fasthash::FastMap;

/// One shell reader and what it has already read.
///
/// `nsh::script::Reader::read` neither expands a word nor runs a substitution
/// nor touches a variable, so what it answers is a function of the bytes it was
/// given and of nothing else — the reader it holds is a parser's workspace
/// rather than state a reading carries to the next one. That is what makes the
/// answers keepable, and keeping them is what a staging pass needs: a pass
/// re-reads the whole composition over text that has not moved, so every
/// `$(MAKE)` line of every unit is read again once per pass and once per goal
/// that reaches the unit, always to the same answer.
///
/// Shared by every session of one invocation, the way the diagnostics
/// descriptor and the census are, rather than reached for through a global:
/// two units reading on two threads take the same lock and neither can see any
/// state the other's evaluation left, there being none.
#[derive(Default)]
pub struct Scripts {
    reader: parking_lot::Mutex<Option<Reader>>,
    /// `None` for a line the shell would not parse, which is an answer worth
    /// keeping for the same reason a parse is.
    read: parking_lot::Mutex<FastMap<Bytes, Option<Arc<Script>>>>,
}

impl Scripts {
    /// Read `line` as the shell would, answering from what has been read
    /// before where the same bytes have been read before.
    ///
    /// `None` where the shell reports a syntax error, or where its reader
    /// cannot be built at all — the caller has one thing to do with either,
    /// which is to leave the line to run as written.
    pub fn read(&self, line: &Bytes) -> Option<Arc<Script>> {
        if let Some(answer) = self.read.lock().get(line) {
            return answer.clone();
        }
        let answer = self.parse(line);
        self.read.lock().insert(line.clone(), answer.clone());
        answer
    }

    /// The reading itself, with the reader built the first time an invocation
    /// needs one and not before: building it costs a locale, a variable table
    /// and three descriptors, and most invocations never name a Make.
    fn parse(&self, line: &Bytes) -> Option<Arc<Script>> {
        let mut slot = self.reader.lock();
        let reader = match slot.as_mut() {
            Some(reader) => reader,
            None => slot.insert(Reader::new().ok()?),
        };
        reader.read(line[..].into()).ok().map(Arc::new)
    }
}
