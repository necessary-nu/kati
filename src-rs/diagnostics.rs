//! Where a compiler diagnostic goes once it has been rendered.
//!
//! The fork is a library first: a caller hands it a Makefile and gets a graph,
//! a result value, and whatever the compilation had to say. Errors already
//! travel that way — [`crate::color_error_log`](crate) returns an
//! `anyhow::Error` and the caller decides where it is written. Warnings did
//! not: they were written to the process's standard error where they were
//! raised, so a caller collecting a compilation's output got the errors and
//! not the warnings, and nothing that reads a run's result as a value could
//! see them at all.
//!
//! This is the descriptor the fork writes them to instead. A session holds
//! one; the default writes straight through to standard error, which is what
//! the fork's own binary wants and what every caller had before. A front end
//! that means to collect them asks for [`Diagnostics::collected`] and drains
//! it wherever its own diagnostic plumbing runs.

use std::sync::Mutex;

/// The descriptor a compilation's non-fatal diagnostics are written to.
///
/// Shared rather than owned: one compilation is several sessions once a
/// recursive `$(MAKE)` is composed into its parent's graph, and what all of
/// them say belongs to the one invocation that asked. Interior mutability
/// because a diagnostic is raised from an evaluation holding `&Session`.
#[derive(Debug, Default)]
pub struct Diagnostics {
    /// What has been raised and not yet drained, or `None` when nothing is
    /// collecting and each diagnostic goes straight to standard error.
    held: Option<Mutex<Vec<u8>>>,
}

impl Diagnostics {
    /// A descriptor that writes each diagnostic to standard error as it is
    /// raised.
    #[must_use]
    pub fn to_stderr() -> Self {
        Self { held: None }
    }

    /// A descriptor that holds what is raised until it is drained.
    #[must_use]
    pub fn collected() -> Self {
        Self {
            held: Some(Mutex::new(Vec::new())),
        }
    }

    /// Write one rendered diagnostic, which is a whole line.
    pub fn write_line(&self, rendered: &str) {
        let Some(held) = &self.held else {
            eprintln!("{rendered}");
            return;
        };
        let mut held = held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        held.extend_from_slice(rendered.as_bytes());
        held.push(b'\n');
    }

    /// Everything raised since the last drain, and nothing if this descriptor
    /// writes through.
    #[must_use]
    pub fn take(&self) -> Vec<u8> {
        let Some(held) = &self.held else {
            return Vec::new();
        };
        let mut held = held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut held)
    }

    /// Whether anything has been raised and not drained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let Some(held) = &self.held else {
            return true;
        };
        held.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_collecting_descriptor_holds_a_line_until_it_is_drained() {
        let diagnostics = Diagnostics::collected();
        assert!(diagnostics.is_empty());
        diagnostics.write_line("Makefile:1: careful");
        assert!(!diagnostics.is_empty());
        assert_eq!(diagnostics.take(), b"Makefile:1: careful\n");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn a_writing_descriptor_has_nothing_to_drain() {
        let diagnostics = Diagnostics::to_stderr();
        diagnostics.write_line("Makefile:1: careful");
        assert!(diagnostics.take().is_empty());
    }
}
