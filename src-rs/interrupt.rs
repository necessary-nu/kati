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

//! What the host tells a read about the user having stopped it.
//!
//! A Makefile read runs processes — `$(shell)`, the regeneration check, the
//! executor's recipes — and a read that waits for one of them is a read that
//! ignores Ctrl-C for as long as the command takes. GNU Make 4.4.1 does not:
//! its `fatal_error_signal` (commands.c) waits for the children it is running
//! as JOBS and knows nothing about the child a `$(shell)` left behind, so it
//! re-raises the signal and is gone in single-digit milliseconds while that
//! child runs on. Measured against `make-4.4.1`: SIGINT sent to the tool alone
//! while `$(shell touch started; sleep 10)` ran left in 1-3 ms, and the
//! `/bin/sh` and its `sleep` were still there afterwards, reparented.
//!
//! So the effect to reproduce is ABANDONMENT, not termination: stop waiting,
//! do not signal the child, do not reap it. The process that was reading is
//! about to exit, and the child becomes the init process's to bury exactly as
//! GNU Make leaves it.
//!
//! The signal itself is the host's business — this crate installs no handler
//! and owns no flag. A session that was given an [`Interruptible`] asks it; a
//! session that was not, which is what `rkati` and every unit test leave, runs
//! the code that ran before this module existed. Session-owned rather than
//! process-global for the reason everything else here is: see
//! `[spec:ronin:req:make.no-ambient-state]`.

use std::fmt;

/// What the host has observed about interruption signals.
///
/// Implemented by the embedder, because the flag belongs to whoever installed
/// the handler that sets it. Two questions, because stopping promptly needs
/// both: whether a signal has arrived, and something to wait on that becomes
/// ready when one does.
pub trait Interruptible: Send + Sync {
    /// Whether an interruption signal has been observed.
    ///
    /// Sticky: once an interrupt has arrived this answers `true` for the rest
    /// of the process, which is what makes a check at the top of a loop enough.
    fn interrupted(&self) -> bool;

    /// A descriptor that becomes readable when an interrupt arrives.
    ///
    /// Optional. Without one a wait still ends, on the poll interval below,
    /// which bounds the delay rather than removing it. The descriptor is only
    /// ever polled here and never read from, so a host may hand out the same
    /// read end it waits on itself.
    fn wake(&self) -> Option<std::os::unix::io::BorrowedFd<'_>> {
        None
    }
}

/// A read abandoned because the user stopped it.
///
/// Carried through `anyhow` so an embedder can tell it from a Makefile that
/// would not evaluate — the two leave with different statuses, and the text of
/// an error is not something to decide a status on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// Whether an [`anyhow::Error`] is [`Interrupted`], anywhere in its chain.
///
/// A read that stops unwinds through callers that add context to what they
/// were doing, so the marker is looked for in the chain rather than at its
/// head.
#[must_use]
pub fn was_interrupted(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(<dyn std::error::Error>::is::<Interrupted>)
}

/// Whether the host says the read has been stopped.
///
/// Asked before a command is launched as well as while one is being waited for.
/// The two are different promises: the wait is what the measured defect was
/// about, and this is the one that says a read already stopped starts nothing
/// further — the same promise a build makes when it launches no further recipe
/// line into the gap between two of them.
pub(crate) fn stopped(interrupts: Option<&dyn Interruptible>) -> bool {
    interrupts.is_some_and(Interruptible::interrupted)
}

/// How long a wait goes without asking again, when the host offered no
/// descriptor to wake on.
///
/// A backstop rather than the mechanism: with a wake descriptor the wait ends
/// as soon as the signal handler writes, and this only bounds how long a host
/// without one can go on waiting.
const POLL_INTERVAL_MS: i32 = 20;

/// What ended a read.
pub(crate) enum Reading {
    /// The writer closed: everything it wrote is in the buffer.
    Complete,
    /// An interrupt arrived first. Whatever had been read is in the buffer and
    /// is not to be believed, and the child was left where it stood.
    Interrupted,
}

/// Read `reader` to its end, unless the host is interrupted first.
///
/// Without an [`Interruptible`] this is [`std::io::Read::read_to_end`] and
/// nothing else, so a session that was given no watch reads exactly as it read
/// before.
pub(crate) fn read_to_end_or_abandon<R>(
    reader: &mut R,
    output: &mut Vec<u8>,
    interrupts: Option<&dyn Interruptible>,
) -> std::io::Result<Reading>
where
    R: std::io::Read + std::os::unix::io::AsFd,
{
    let Some(watch) = interrupts else {
        reader.read_to_end(output)?;
        return Ok(Reading::Complete);
    };

    let mut chunk = [0u8; 8 * 1024];
    loop {
        // First, because the flag is sticky and an interrupt that arrived
        // before the child was even spawned must not buy a whole poll interval.
        if watch.interrupted() {
            return Ok(Reading::Interrupted);
        }
        let ready = wait_for_input(reader.as_fd(), watch.wake())?;
        if ready {
            let read = reader.read(&mut chunk)?;
            if read == 0 {
                return Ok(Reading::Complete);
            }
            output.extend_from_slice(&chunk[..read]);
        }
    }
}

/// Wait for `handle` to exit, unless the host is interrupted first.
///
/// The other half of [`read_to_end_or_abandon`], and reachable whenever a
/// command closes its standard output before it finishes — a shell that was
/// told `exec >/dev/null` is the deterministic form. The read ends at that
/// point and everything left is this wait, so a wait that could not be stopped
/// would leave the whole defect standing behind a fixed read. Measured: a
/// `$(shell)` under `SHELL := /bin/bash` doing exactly that held Ronin for
/// 3,996 ms of a 4-second child, against GNU Make 4.4.1's 2 ms.
///
/// `None` back is the abandonment: the caller drops the handle, which neither
/// waits nor kills.
pub(crate) fn wait_or_abandon(
    handle: &mut std::process::Child,
    interrupts: Option<&dyn Interruptible>,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let Some(watch) = interrupts else {
        return handle.wait().map(Some);
    };
    loop {
        if watch.interrupted() {
            return Ok(None);
        }
        if let Some(status) = handle.try_wait()? {
            return Ok(Some(status));
        }
        // Nothing to read here, so the wake descriptor is the only thing worth
        // waiting on and the interval is what bounds a host without one. The
        // child's own exit is caught by the poll above rather than by a
        // descriptor, which is why this loop asks again rather than blocking.
        wait_for_wake(watch.wake())?;
    }
}

/// Wait for `wake` to become readable, or for the poll interval to run out.
fn wait_for_wake(wake: Option<std::os::unix::io::BorrowedFd<'_>>) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd as _;

    let Some(wake) = wake else {
        std::thread::sleep(std::time::Duration::from_millis(
            POLL_INTERVAL_MS.unsigned_abs().into(),
        ));
        return Ok(());
    };
    let mut watched = [libc::pollfd {
        fd: wake.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    }];
    // SAFETY: `watched` is a live array of one initialised `pollfd`.
    let ready = unsafe { libc::poll(watched.as_mut_ptr(), 1, POLL_INTERVAL_MS) };
    if ready < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
    Ok(())
}

/// Wait until `fd` has something to say, `wake` becomes readable, or the poll
/// interval runs out. Answers whether `fd` is the one that is ready.
fn wait_for_input(
    fd: std::os::unix::io::BorrowedFd<'_>,
    wake: Option<std::os::unix::io::BorrowedFd<'_>>,
) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd as _;

    let mut watched = [
        libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: wake.map_or(-1, |wake| wake.as_raw_fd()),
            events: libc::POLLIN,
            revents: 0,
        },
    ];
    // A negative descriptor is ignored by `poll`, so the second entry is
    // present and inert when the host offered nothing to wake on.
    let count = if wake.is_some() { 2 } else { 1 };
    // SAFETY: `watched` is a live array of `count` initialised `pollfd`s, and
    // `count` is at most its length.
    let ready = unsafe { libc::poll(watched.as_mut_ptr(), count, POLL_INTERVAL_MS) };
    if ready < 0 {
        let error = std::io::Error::last_os_error();
        // A signal the host handles lands here whenever its handler was
        // installed without `SA_RESTART`. It is not a failure: the next turn
        // of the loop asks the flag, which is what the wake was for.
        if error.kind() == std::io::ErrorKind::Interrupted {
            return Ok(false);
        }
        return Err(error);
    }
    // POLLHUP and POLLERR mean the read is over too, and are reported in
    // `revents` whether or not they were asked for. Reading answers both.
    Ok(watched[0].revents != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Watch {
        flag: AtomicBool,
    }

    impl Interruptible for Watch {
        fn interrupted(&self) -> bool {
            self.flag.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn a_read_with_no_watch_reads_to_the_end() {
        let (mut reader, mut writer) = os_pipe::pipe().unwrap();
        writer.write_all(b"hello").unwrap();
        drop(writer);
        let mut output = Vec::new();
        let reading = read_to_end_or_abandon(&mut reader, &mut output, None).unwrap();
        assert!(matches!(reading, Reading::Complete));
        assert_eq!(output, b"hello");
    }

    #[test]
    fn a_watch_that_never_fires_reads_to_the_end() {
        let (mut reader, mut writer) = os_pipe::pipe().unwrap();
        writer.write_all(b"hello").unwrap();
        drop(writer);
        let watch = Watch {
            flag: AtomicBool::new(false),
        };
        let mut output = Vec::new();
        let reading = read_to_end_or_abandon(&mut reader, &mut output, Some(&watch)).unwrap();
        assert!(matches!(reading, Reading::Complete));
        assert_eq!(output, b"hello");
    }

    /// The defect this module exists for: the writer never closes, so a plain
    /// `read_to_end` would not return at all.
    #[test]
    fn an_interrupt_ends_a_read_the_writer_never_closes() {
        let (mut reader, _writer) = os_pipe::pipe().unwrap();
        let watch = Watch {
            flag: AtomicBool::new(true),
        };
        let mut output = Vec::new();
        let reading = read_to_end_or_abandon(&mut reader, &mut output, Some(&watch)).unwrap();
        assert!(matches!(reading, Reading::Interrupted));
    }

    /// And it is noticed while the read is already waiting, rather than only
    /// when it was set before the call.
    #[test]
    fn an_interrupt_that_arrives_during_the_wait_ends_it() {
        let (mut reader, _writer) = os_pipe::pipe().unwrap();
        let watch = std::sync::Arc::new(Watch {
            flag: AtomicBool::new(false),
        });
        let setter = std::sync::Arc::clone(&watch);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            setter.flag.store(true, Ordering::SeqCst);
        });
        let mut output = Vec::new();
        let reading =
            read_to_end_or_abandon(&mut reader, &mut output, Some(watch.as_ref())).unwrap();
        assert!(matches!(reading, Reading::Interrupted));
    }

    #[test]
    fn a_wait_with_no_watch_waits_for_the_child() {
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "exit 7"])
            .spawn()
            .unwrap();
        let status = wait_or_abandon(&mut child, None).unwrap().unwrap();
        assert_eq!(status.code(), Some(7));
    }

    /// The child outlives the wait, which is the whole point: a wait that could
    /// not be stopped would sit here for the child's ten seconds.
    #[test]
    fn an_interrupt_ends_a_wait_for_a_child_that_is_still_running() {
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 10"])
            .spawn()
            .unwrap();
        let watch = Watch {
            flag: AtomicBool::new(true),
        };
        assert!(wait_or_abandon(&mut child, Some(&watch)).unwrap().is_none());
        // Abandoned, not killed: it is still there, and this test is the one
        // that cleans it up rather than the code under test.
        assert!(child.try_wait().unwrap().is_none());
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn an_interrupt_that_arrives_during_a_wait_ends_it() {
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 10"])
            .spawn()
            .unwrap();
        let watch = std::sync::Arc::new(Watch {
            flag: AtomicBool::new(false),
        });
        let setter = std::sync::Arc::clone(&watch);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            setter.flag.store(true, Ordering::SeqCst);
        });
        assert!(
            wait_or_abandon(&mut child, Some(watch.as_ref()))
                .unwrap()
                .is_none()
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn nothing_is_stopped_without_a_watch_that_says_so() {
        assert!(!stopped(None));
        assert!(!stopped(Some(&Watch {
            flag: AtomicBool::new(false)
        })));
        assert!(stopped(Some(&Watch {
            flag: AtomicBool::new(true)
        })));
    }

    #[test]
    fn the_marker_is_found_through_a_chain_of_context() {
        let error = anyhow::Error::from(Interrupted).context("reading Makefile");
        assert!(was_interrupted(&error));
        assert!(!was_interrupted(&anyhow::anyhow!("something else")));
    }
}
