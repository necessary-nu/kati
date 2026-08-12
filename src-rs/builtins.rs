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

//! The tool catalogue GNU Make defines before it reads anything.
//!
//! `AR`, `CC`, `RM` and the compound `COMPILE.c`/`LINK.o` command macros are
//! not conveniences a Makefile may assume: real Makefiles write recipes that
//! consist of nothing else, so a front end that leaves them empty produces a
//! recipe with no compiler in it. The table below is GNU Make 4.4.1's
//! `default_variables[]` from `src/default.c`, taken through the same
//! preprocessor branches this host's oracle was built with — the generic
//! non-VMS Unix ones, without `GCC_IS_NATIVE`.
//!
//! Three properties make these defaults rather than assignments, and each one
//! is observable from a Makefile:
//!
//! * They are recursive, so `$(COMPILE.c)` picks up a `CC` the Makefile sets
//!   later and `$(flavor COMPILE.c)` answers `recursive`.
//! * They carry [`VarOrigin::Default`], the weakest origin there is, so the
//!   environment, the command line and the Makefile all outrank them and
//!   `$(origin CC)` answers `default` only while nothing else has spoken.
//! * `-R` withholds them entirely, and a `MAKEFLAGS` write that says `-R`
//!   after the fact withdraws them — see [`undefine_default_variables`].

use anyhow::Result;
use bytes::Bytes;

use crate::expr::{ParseExprOpt, parse_expr};
use crate::loc::Loc;
use crate::session::Session;
use crate::symtab::Symbol;
use crate::var::{VarOrigin, Variable};

/// What GNU Make's `configure` found for `MAKE_CXX`, which is the C++ driver
/// the platform's toolchain answers to rather than a name Make invented.
#[cfg(target_os = "macos")]
const DEFAULT_CXX: &str = "c++";
#[cfg(not(target_os = "macos"))]
const DEFAULT_CXX: &str = "g++";

/// How `-lfoo` is spelled on disk, which is the one entry in the table that
/// names a platform's shared-library convention.
#[cfg(target_os = "macos")]
const DEFAULT_LIBPATTERNS: &str = "lib%.dylib lib%.a";
#[cfg(not(target_os = "macos"))]
const DEFAULT_LIBPATTERNS: &str = "lib%.so lib%.a";

/// GNU Make 4.4.1's `default_variables[]`, in its order.
///
/// The order is kept because it is the order a reader compares against
/// `src/default.c`, and because two entries read each other: `COMPILE.C` is
/// `$(COMPILE.cc)` and would be confusing above it.
pub const DEFAULT_VARIABLES: &[(&str, &str)] = &[
    ("AR", "ar"),
    ("ARFLAGS", "-rv"),
    ("AS", "as"),
    ("CC", "cc"),
    ("OBJC", "cc"),
    ("CXX", DEFAULT_CXX),
    // Expands to the checkout command when the target is absent and to nothing
    // when it is already there, which is why it is written as an `$(if)`
    // rather than as a rule.
    (
        "CHECKOUT,v",
        "+$(if $(wildcard $@),,$(CO) $(COFLAGS) $< $@)",
    ),
    ("CO", "co"),
    ("COFLAGS", ""),
    ("CPP", "$(CC) -E"),
    ("FC", "f77"),
    // System V spells the Fortran compiler this way. Explicit rules using it
    // work; no implicit rule can, because implicit rules use `FC`.
    ("F77", "$(FC)"),
    ("F77FLAGS", "$(FFLAGS)"),
    ("GET", "get"),
    ("LD", "ld"),
    ("LEX", "lex"),
    ("LINT", "lint"),
    ("M2C", "m2c"),
    ("PC", "pc"),
    ("YACC", "yacc"),
    ("MAKEINFO", "makeinfo"),
    ("TEX", "tex"),
    ("TEXI2DVI", "texi2dvi"),
    ("WEAVE", "weave"),
    ("CWEAVE", "cweave"),
    ("TANGLE", "tangle"),
    ("CTANGLE", "ctangle"),
    ("RM", "rm -f"),
    ("LINK.o", "$(CC) $(LDFLAGS) $(TARGET_ARCH)"),
    ("COMPILE.c", "$(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c"),
    (
        "LINK.c",
        "$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    (
        "COMPILE.m",
        "$(OBJC) $(OBJCFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c",
    ),
    (
        "LINK.m",
        "$(OBJC) $(OBJCFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    (
        "COMPILE.cc",
        "$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c",
    ),
    ("COMPILE.C", "$(COMPILE.cc)"),
    ("COMPILE.cpp", "$(COMPILE.cc)"),
    (
        "LINK.cc",
        "$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    ("LINK.C", "$(LINK.cc)"),
    ("LINK.cpp", "$(LINK.cc)"),
    ("YACC.y", "$(YACC) $(YFLAGS)"),
    ("LEX.l", "$(LEX) $(LFLAGS) -t"),
    ("YACC.m", "$(YACC) $(YFLAGS)"),
    ("LEX.m", "$(LEX) $(LFLAGS) -t"),
    ("COMPILE.f", "$(FC) $(FFLAGS) $(TARGET_ARCH) -c"),
    ("LINK.f", "$(FC) $(FFLAGS) $(LDFLAGS) $(TARGET_ARCH)"),
    ("COMPILE.F", "$(FC) $(FFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c"),
    (
        "LINK.F",
        "$(FC) $(FFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    ("COMPILE.r", "$(FC) $(FFLAGS) $(RFLAGS) $(TARGET_ARCH) -c"),
    (
        "LINK.r",
        "$(FC) $(FFLAGS) $(RFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    (
        "COMPILE.def",
        "$(M2C) $(M2FLAGS) $(DEFFLAGS) $(TARGET_ARCH)",
    ),
    (
        "COMPILE.mod",
        "$(M2C) $(M2FLAGS) $(MODFLAGS) $(TARGET_ARCH)",
    ),
    ("COMPILE.p", "$(PC) $(PFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c"),
    (
        "LINK.p",
        "$(PC) $(PFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)",
    ),
    ("LINK.s", "$(CC) $(ASFLAGS) $(LDFLAGS) $(TARGET_MACH)"),
    ("COMPILE.s", "$(AS) $(ASFLAGS) $(TARGET_MACH)"),
    (
        "LINK.S",
        "$(CC) $(ASFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_MACH)",
    ),
    (
        "COMPILE.S",
        "$(CC) $(ASFLAGS) $(CPPFLAGS) $(TARGET_MACH) -c",
    ),
    ("PREPROCESS.S", "$(CPP) $(CPPFLAGS)"),
    (
        "PREPROCESS.F",
        "$(FC) $(FFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -F",
    ),
    (
        "PREPROCESS.r",
        "$(FC) $(FFLAGS) $(RFLAGS) $(TARGET_ARCH) -F",
    ),
    ("LINT.c", "$(LINT) $(LINTFLAGS) $(CPPFLAGS) $(TARGET_ARCH)"),
    ("OUTPUT_OPTION", "-o $@"),
    (".LIBPATTERNS", DEFAULT_LIBPATTERNS),
    // Defined so that a Makefile reading it does not trip an
    // undefined-variable warning. Its switches arrive separately.
    ("GNUMAKEFLAGS", ""),
];

/// The values `.POSIX:` substitutes, from IEEE Std 1003.1-2008 by way of GNU
/// Make's `check_specials`.
///
/// Unlike the table above these are simple, not recursive, and they replace
/// entries that are already there — `CC` is `cc` until `.POSIX:` is seen and
/// `c99` afterwards. The standard asks for `-O 1`, and the space is dropped
/// because GCC does not accept it.
const POSIX_VARIABLES: &[(&str, &str)] = &[
    ("CC", "c99"),
    ("CFLAGS", "-O1"),
    ("FC", "fort77"),
    ("FFLAGS", "-O1"),
    ("SCCSGETFLAGS", "-s"),
    // Same value the ordinary catalogue holds, and installed again because
    // `.POSIX:` makes it simple where the catalogue made it recursive. Debian
    // patches this one to `-rvU`, asking `ar` for a non-deterministic archive;
    // that is Debian's answer rather than GNU's, and it is not implemented.
    ("ARFLAGS", "-rv"),
];

/// The name, if a default may still claim it.
///
/// GNU Make defines these through the ordinary origin ladder, and `default` is
/// the bottom of it: anything the environment, the command line, an `override`
/// or the Makefile itself has already said outranks a default and keeps its
/// value. Another default does not, which is how `.POSIX:` replaces `CC`.
fn claimable(session: &mut Session, name: &str) -> Option<Symbol> {
    let sym = session.intern(name.as_bytes().to_vec());
    match session.peek_global_var(sym) {
        Some(existing) if existing.read().origin() != VarOrigin::Default => None,
        _ => Some(sym),
    }
}

/// Install the built-in variable catalogue.
///
/// Called once per evaluation, after the environment is in scope and before
/// any Makefile is read, which is where GNU Make's `define_default_variables`
/// sits.
///
/// # Errors
///
/// Returns a parse failure for a table entry, which is a defect in the table.
pub fn install_default_variables(session: &mut Session) -> Result<()> {
    for (name, value) in DEFAULT_VARIABLES {
        let Some(sym) = claimable(session, name) else {
            continue;
        };
        let text = Bytes::from_static(value.as_bytes());
        let mut loc = Loc::default();
        let parsed = parse_expr(session, &mut loc, text.clone(), ParseExprOpt::Normal)?;
        let var = Variable::new_recursive(parsed, VarOrigin::Default, None, None, text);
        session.globals.define(sym, var);
    }
    Ok(())
}

/// Substitute the POSIX values, as `.POSIX:` appearing as a target does.
///
/// Not withheld by `-R`: the switch withholds the catalogue, and `.POSIX:` is
/// the Makefile asking for a specific standard's values by name. A `-R` that
/// arrives afterwards still withdraws the names the catalogue owns, because
/// what it withdraws is every default binding under those names rather than
/// the ones it installed.
pub fn install_posix_variables(session: &mut Session) {
    for (name, value) in POSIX_VARIABLES {
        let Some(sym) = claimable(session, name) else {
            continue;
        };
        let var = Variable::with_simple_string(
            Bytes::from_static(value.as_bytes()),
            VarOrigin::Default,
            None,
            None,
        );
        session.globals.define(sym, var);
    }
}

/// Withdraw the catalogue after the fact.
///
/// A Makefile that writes `MAKEFLAGS += -rR` — which is how the Linux kernel's
/// build asks for a clean namespace — has already been read by the time the
/// switch is known. GNU Make reads the whole makefile with the defaults in
/// scope and only then removes them, so a `$(origin CC)` during the read still
/// answers `default` while the recipe that runs afterwards sees nothing.
///
/// Only bindings still at [`VarOrigin::Default`] go: a Makefile that assigned
/// `CC` itself keeps its value, exactly as `undefine` under this origin does.
pub fn undefine_default_variables(session: &mut Session) {
    for (name, _) in DEFAULT_VARIABLES {
        let sym = session.intern(name.as_bytes().to_vec());
        let installed = session
            .peek_global_var(sym)
            .is_some_and(|var| var.read().origin() == VarOrigin::Default);
        if installed {
            session.globals.replace(sym, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GNU Make binds each name in the table once. A duplicate would make the
    /// later entry silently win and leave the earlier one as dead text that
    /// still reads like the definition.
    #[test]
    fn default_variable_names_are_unique() {
        let mut names = DEFAULT_VARIABLES
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        let recorded = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), recorded);
    }
}
