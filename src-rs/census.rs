//! What a compilation decided about each recursive invocation it saw.
//!
//! The compiler settles this question for every `$(MAKE)` in every recipe it
//! reaches — lift the child out and compile it into the parent's graph, or
//! leave the line to start a Make of its own at run time — and then, having
//! acted on the answer, forgets it. Nothing in a build needs it afterwards.
//!
//! A report about a build does need it, and needs it to be the compiler's own
//! answer rather than a second reading of the same recipe that could differ.
//! This is the ledger the compiler writes it into on its way past. It is empty
//! and inert unless a caller asked for one, because a build that kept a census
//! nobody reads would be paying for a report nobody asked for.

use std::sync::Mutex;

/// What became of one recursive invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Disposition {
    /// Lifted out of the recipe and compiled into the parent's graph, as this
    /// invocation: the child's directory and goals as the compiler read them,
    /// with the `MAKE` reference written back in place of the path it
    /// expanded to, because the path is this process and says nothing.
    Composed { command: Vec<u8> },
    /// Left where it was written, to start a nested Make when the recipe runs.
    /// There is no one invocation to name — that is what the reason says —
    /// and the location names the line.
    Nested(NestingReason),
    /// Composed, and then there was no makefile where it pointed.
    ///
    /// Recorded by whoever went to read the child rather than by the classifier
    /// above, because the classifier settles what the recipe line IS and this is
    /// what happened when the compiler acted on that: the directory named here
    /// exists and holds none of the names a Make reads. It follows the
    /// [`Disposition::Composed`] entry for the same line rather than replacing
    /// it, because both are true and the first is what the compiler decided.
    MissingMakefile {
        /// Where the invocation pointed, as a reader would write it: relative
        /// to the build's root where it sits under one.
        directory: String,
    },
}

/// Why an invocation the compiler could see was not composed.
///
/// Recorded where the decision is made rather than worked out afterwards from
/// the recipe text, so what a report says is what the compile did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NestingReason {
    /// The invocation is not the recipe line's own command: a shell construct
    /// stands between them — a conditional, a sequence, an alternation, a
    /// pipeline — and the compiler lifts out a line that IS an invocation,
    /// not a line that contains one somewhere.
    ThroughAConstruct,
    /// The line's command is the invocation and it is written as more than
    /// the argument list the resolver reads: an assignment or `env` prefix, a
    /// redirection, a glob, an expansion it will not settle.
    NotAnArgumentList,
    /// A `.ONESHELL` recipe of more than one line, whose lines share one
    /// shell, so no reading of the recipe establishes what an earlier line
    /// left for this invocation to read.
    SharedShell,
}

/// One recursive invocation a compile classified, and what it decided.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Invocation {
    /// The Makefile and line the recipe line was written on, rendered here
    /// because the file name lives in an interner the reader does not hold.
    pub location: Option<String>,
    pub disposition: Disposition,
}

/// The ledger a compilation records its recursive invocations in.
///
/// Shared rather than owned, exactly as [`crate::diagnostics::Diagnostics`]
/// is: one compilation is several sessions once a recursive `$(MAKE)` is
/// composed into its parent's graph, and what all of them classified belongs
/// to the one invocation that asked. Interior mutability because the record is
/// made from an evaluation holding `&Session`.
#[derive(Debug, Default)]
pub struct Census {
    /// What has been recorded, or `None` when nobody asked for a census and
    /// each classification is acted on and forgotten as it always was.
    held: Option<Mutex<Vec<Invocation>>>,
}

impl Census {
    /// A ledger that records nothing, which is what a build wants.
    #[must_use]
    pub const fn ignored() -> Self {
        Self { held: None }
    }

    /// A ledger that keeps what the compile classified until it is taken.
    #[must_use]
    pub fn collected() -> Self {
        Self {
            held: Some(Mutex::new(Vec::new())),
        }
    }

    /// Whether anything is being recorded at all.
    ///
    /// Worth asking before rendering a location, which costs an interner
    /// lookup and a string that an ignoring ledger would drop.
    #[must_use]
    pub const fn is_recording(&self) -> bool {
        self.held.is_some()
    }

    /// Record one classified invocation.
    pub fn record(&self, invocation: Invocation) {
        let Some(held) = &self.held else {
            return;
        };
        held.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(invocation);
    }

    /// Take everything recorded so far, leaving the ledger empty.
    #[must_use]
    pub fn take(&self) -> Vec<Invocation> {
        let Some(held) = &self.held else {
            return Vec::new();
        };
        let mut held = held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut held)
    }
}

#[cfg(test)]
mod tests {
    use super::{Census, Disposition, Invocation, NestingReason};

    fn invocation(disposition: Disposition) -> Invocation {
        Invocation {
            location: Some("Makefile:1".to_owned()),
            disposition,
        }
    }

    fn composed(command: &str) -> Disposition {
        Disposition::Composed {
            command: command.as_bytes().to_vec(),
        }
    }

    /// A ledger nobody asked for keeps nothing, so a build pays for no report.
    #[test]
    fn an_ignoring_ledger_keeps_nothing() {
        let census = Census::ignored();
        assert!(!census.is_recording());
        census.record(invocation(composed("make -C sub")));
        assert!(census.take().is_empty());
    }

    /// A collecting one keeps what it was given, in the order it was given it,
    /// and hands it over once.
    #[test]
    fn a_collecting_ledger_keeps_the_order() {
        let census = Census::collected();
        assert!(census.is_recording());
        census.record(invocation(composed("make -C sub")));
        census.record(invocation(Disposition::Nested(
            NestingReason::ThroughAConstruct,
        )));
        let taken = census.take();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].disposition, composed("make -C sub"));
        assert_eq!(
            taken[1].disposition,
            Disposition::Nested(NestingReason::ThroughAConstruct)
        );
        assert!(
            census.take().is_empty(),
            "a ledger hands its record over once"
        );
    }
}
