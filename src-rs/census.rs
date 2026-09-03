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
    /// Never started: the invocation stands under a condition the compiler
    /// settled as false, or in a loop over no words, so nothing is composed
    /// for it and nothing runs it. vim's `if test "$@" = "test" ...` guards
    /// are this for every goal but `test`.
    Unreached,
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
/// the recipe text, so what a report says is what the compile did. Each is
/// the first construct that stopped the line's reading — see
/// [`crate::lift`] — from proving the line equal to a list of composed
/// children, named precisely enough that a reader knows what to change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NestingReason {
    /// A `.ONESHELL` recipe of more than one line, whose lines share one
    /// shell, so no reading of the recipe establishes what an earlier line
    /// left for this invocation to read.
    SharedShell,
    /// The shell could not read the line at all: it is not POSIX shell
    /// syntax, or not complete.
    Unreadable,
    /// A command substitution decides what the line does — the goals, the
    /// directory, a loop's words, a variable the invocation reads — and what
    /// it expands to is known only by running it.
    CommandSubstitution,
    /// A shell variable the invocation reads has no value the line settles:
    /// it was never assigned on the line, or assigned from something the
    /// line cannot settle, or it is a positional or special parameter.
    UnresolvedParameter {
        /// The parameter as the line wrote it.
        name: String,
    },
    /// A word the invocation reads would be matched against the disk — an
    /// unquoted pattern, or a `~` — and what it matches is known only there.
    Glob,
    /// The invocation stands under an `if` whose condition the line does not
    /// settle: a file test, a command's status, a value the line never gave.
    UndecidableCondition,
    /// `||` hands the invocation's failure to another command. Only `exit`
    /// composes there, because a failed child fails the recipe exactly as
    /// `exit` would; anything else runs after the failure and a composed
    /// child has no edge to run it on.
    Alternation,
    /// `;` runs what follows whether or not the invocation failed, where a
    /// composed child's failure stops everything after it.
    Sequence,
    /// A `for` loop runs its next iteration whether or not this one failed,
    /// where a composed child's failure stops everything after it.
    LoopCarriesOn,
    /// A pipeline or a `!` stands between the line and the invocation.
    Pipeline,
    /// A redirection changes what the invocation reads or writes, and the
    /// graph has nowhere to put it.
    Redirection,
    /// An assignment or `env` stands in front of the invocation's command,
    /// giving the child an environment the line composes at run time.
    Prefix,
    /// A `cd` the compiler cannot follow: one without a directory, or with
    /// an option.
    DirectoryChange,
    /// Another command runs on the line beside the invocation, and a
    /// composed child carries nothing but the invocation.
    BesideAnotherCommand {
        /// The command's name.
        command: String,
    },
    /// A shell form the compiler does not read: a `while`, a `case`, a
    /// function, a command in the background.
    Construct {
        /// The form, as a noun phrase.
        construct: &'static str,
    },
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
        census.record(invocation(Disposition::Nested(NestingReason::Sequence)));
        let taken = census.take();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].disposition, composed("make -C sub"));
        assert_eq!(
            taken[1].disposition,
            Disposition::Nested(NestingReason::Sequence)
        );
        assert!(
            census.take().is_empty(),
            "a ledger hands its record over once"
        );
    }
}
