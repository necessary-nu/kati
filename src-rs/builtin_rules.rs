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

//! The rules GNU Make knows before a Makefile says anything.
//!
//! `all: hello` beside `hello.c` is the oldest idiom Make has, and nothing in a
//! Makefile makes it work: the rule that links a program from a source file is
//! one GNU Make supplies. The tables below are GNU Make 4.4.1's
//! `default_suffixes`, `default_suffix_rules[]`, `default_pattern_rules[]` and
//! `default_terminal_rules[]` from `src/default.c`, taken through the same
//! preprocessor branches this host's oracle was built with — the generic
//! non-VMS Unix ones, without `GCC_IS_NATIVE` and without `__MSDOS__`. The
//! command macros they expand to are the other half of the same catalogue and
//! are already installed by [`crate::builtins`].
//!
//! Three of the four tables are not rules yet. GNU Make records the suffix
//! rules as *files* named `.c.o` or `.c`, and turns them into pattern rules
//! only once every Makefile has been read, walking `.SUFFIXES` as the read left
//! it: `%.o: %.c` from the pair, `%: %.c` from the single suffix — the link
//! rules — and a prerequisite-less, recipe-less `%.c:` from the suffix alone,
//! which exists solely so that a match-anything rule stops applying to a name
//! that ends in a known suffix. Clearing `.SUFFIXES` therefore withdraws every
//! rule derived from it, and adding a suffix activates whichever pairs then
//! exist. `install_default_implicit_rules` adds the last two tables afterwards,
//! which is why a Makefile's own `%.out: %` outranks the built-in one.
//!
//! The whole catalogue is what `-r` withholds — and `-R`, which implies it.

use anyhow::Result;
use bytes::Bytes;

use crate::expr::{ParseExprOpt, ValueId, parse_expr};
use crate::loc::Loc;
use crate::rule::Rule;
use crate::session::Session;

/// GNU Make 4.4.1's `default_suffixes`, in its order.
///
/// The order is the order the rules derived from it are tried in, so `.c`
/// preceding `.cc` is what makes `foo.o` come from `foo.c` when both sources
/// are there. `.s` is late for the reason `default.c` gives: an object file
/// should be made from a `.c` or a `.p` before it is made from assembler.
pub const DEFAULT_SUFFIXES: &[&str] = &[
    ".out", ".a", ".ln", ".o", ".c", ".cc", ".C", ".cpp", ".p", ".f", ".F", ".m", ".r", ".y", ".l",
    ".ym", ".yl", ".s", ".S", ".mod", ".sym", ".def", ".h", ".info", ".dvi", ".tex", ".texinfo",
    ".texi", ".txinfo", ".w", ".ch", ".web", ".sh", ".elc", ".el",
];

/// GNU Make 4.4.1's `default_suffix_rules[]`, as `(name, recipe)` pairs.
///
/// A name with one suffix is a link rule and a name with two is a compile rule,
/// which is a distinction the reader makes rather than the table: GNU Make
/// stores both as files and lets `convert_to_pattern` tell them apart. A
/// newline inside a recipe separates command lines, exactly as it does in a
/// Makefile.
///
/// `.lm.m` is in the table and never becomes a rule, because `.lm` is not in
/// [`DEFAULT_SUFFIXES`]. It is kept so the table can be read against
/// `default.c` line by line.
pub const DEFAULT_SUFFIX_RULES: &[(&str, &str)] = &[
    (".o", "$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".s", "$(LINK.s) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".S", "$(LINK.S) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".c", "$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".cc", "$(LINK.cc) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".C", "$(LINK.C) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".cpp", "$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".f", "$(LINK.f) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".m", "$(LINK.m) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".p", "$(LINK.p) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".F", "$(LINK.F) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".r", "$(LINK.r) $^ $(LOADLIBES) $(LDLIBS) -o $@"),
    (".mod", "$(COMPILE.mod) -o $@ -e $@ $^"),
    (".def.sym", "$(COMPILE.def) -o $@ $<"),
    (".sh", "cat $< >$@ \n chmod a+x $@"),
    (".s.o", "$(COMPILE.s) -o $@ $<"),
    (".S.o", "$(COMPILE.S) -o $@ $<"),
    (".c.o", "$(COMPILE.c) $(OUTPUT_OPTION) $<"),
    (".cc.o", "$(COMPILE.cc) $(OUTPUT_OPTION) $<"),
    (".C.o", "$(COMPILE.C) $(OUTPUT_OPTION) $<"),
    (".cpp.o", "$(COMPILE.cpp) $(OUTPUT_OPTION) $<"),
    (".f.o", "$(COMPILE.f) $(OUTPUT_OPTION) $<"),
    (".m.o", "$(COMPILE.m) $(OUTPUT_OPTION) $<"),
    (".p.o", "$(COMPILE.p) $(OUTPUT_OPTION) $<"),
    (".F.o", "$(COMPILE.F) $(OUTPUT_OPTION) $<"),
    (".r.o", "$(COMPILE.r) $(OUTPUT_OPTION) $<"),
    (".mod.o", "$(COMPILE.mod) -o $@ $<"),
    (".c.ln", "$(LINT.c) -C$* $<"),
    (
        ".y.ln",
        "$(YACC.y) $< \n $(LINT.c) -C$* y.tab.c \n $(RM) y.tab.c",
    ),
    (
        ".l.ln",
        "@$(RM) $*.c\n $(LEX.l) $< > $*.c\n$(LINT.c) -i $*.c -o $@\n $(RM) $*.c",
    ),
    (".y.c", "$(YACC.y) $< \n mv -f y.tab.c $@"),
    (".l.c", "@$(RM) $@ \n $(LEX.l) $< > $@"),
    (".ym.m", "$(YACC.m) $< \n mv -f y.tab.c $@"),
    (".lm.m", "@$(RM) $@ \n $(LEX.m) $< > $@"),
    (".F.f", "$(PREPROCESS.F) $(OUTPUT_OPTION) $<"),
    (".r.f", "$(PREPROCESS.r) $(OUTPUT_OPTION) $<"),
    // Might make lex.yy.c rather than lex.yy.r when the source has no %R%
    // directive, but then the Makefile had no business asking for a .r.
    (".l.r", "$(LEX.l) $< > $@ \n mv -f lex.yy.r $@"),
    (".S.s", "$(PREPROCESS.S) $< > $@"),
    (".texinfo.info", "$(MAKEINFO) $(MAKEINFO_FLAGS) $< -o $@"),
    (".texi.info", "$(MAKEINFO) $(MAKEINFO_FLAGS) $< -o $@"),
    (".txinfo.info", "$(MAKEINFO) $(MAKEINFO_FLAGS) $< -o $@"),
    (".tex.dvi", "$(TEX) $<"),
    (".texinfo.dvi", "$(TEXI2DVI) $(TEXI2DVI_FLAGS) $<"),
    (".texi.dvi", "$(TEXI2DVI) $(TEXI2DVI_FLAGS) $<"),
    (".txinfo.dvi", "$(TEXI2DVI) $(TEXI2DVI_FLAGS) $<"),
    // The `-` says there is no `.ch` file; the two-prerequisite form below is
    // the one that has one.
    (".w.c", "$(CTANGLE) $< - $@"),
    (".web.p", "$(TANGLE) $<"),
    (".w.tex", "$(CWEAVE) $< - $@"),
    (".web.tex", "$(WEAVE) $<"),
];

/// GNU Make 4.4.1's `default_pattern_rules[]`, as `(target, prerequisites,
/// recipe)`.
///
/// These are already pattern rules in `default.c`, and are installed after the
/// suffix-derived ones so that a Makefile that wrote any of them keeps its own.
/// The `%.out` rule is here rather than among the suffix rules because BSD Make
/// has no null-suffix rules and spells a program `foo.out`.
pub const DEFAULT_PATTERN_RULES: &[(&str, &str, &str)] = &[
    ("(%)", "%", "$(AR) $(ARFLAGS) $@ $<"),
    ("%.out", "%", "@rm -f $@ \n cp $< $@"),
    // Syntax is "ctangle foo.w foo.ch foo.c".
    ("%.c", "%.w %.ch", "$(CTANGLE) $^ $@"),
    ("%.tex", "%.w %.ch", "$(CWEAVE) $^ $@"),
];

/// GNU Make 4.4.1's `default_terminal_rules[]`, as `(target, prerequisites,
/// recipe)`.
///
/// Checking a file out of RCS or SCCS is the one thing Make does that has no
/// source file to work from, so these are terminal: their prerequisites must
/// already be there, and being terminal is also what lets a match-anything rule
/// like these serve as a link in a chain at all.
pub const DEFAULT_TERMINAL_RULES: &[(&str, &str, &str)] = &[
    ("%", "%,v", "$(CHECKOUT,v)"),
    ("%", "RCS/%,v", "$(CHECKOUT,v)"),
    ("%", "RCS/%", "$(CHECKOUT,v)"),
    ("%", "s.%", "$(GET) $(GFLAGS) $(SCCS_OUTPUT_OPTION) $<"),
    ("%", "SCCS/s.%", "$(GET) $(GFLAGS) $(SCCS_OUTPUT_OPTION) $<"),
];

/// The `.SUFFIXES` prerequisite list and the `SUFFIXES` variable's value, which
/// GNU Make writes from one string.
pub fn default_suffix_list() -> String {
    DEFAULT_SUFFIXES.join(" ")
}

/// Bind `SUFFIXES` to the list, or to nothing when the rules are withheld.
///
/// GNU Make's `set_default_suffixes` writes the same string it gives
/// `.SUFFIXES` into a simple variable at `default` origin, so a Makefile can
/// read the list back and an environment or command-line value of that name
/// still outranks it. `-r` leaves the name bound and empty rather than
/// undefined, which is a difference a Makefile can see.
///
/// Called again after the read when a Makefile's own `MAKEFLAGS` asked for
/// `-r`, which is why it must replace a binding it already made.
pub fn install_suffixes_variable(session: &mut Session, withheld: bool) {
    let Some(sym) = crate::builtins::claimable(session, "SUFFIXES") else {
        return;
    };
    let value = if withheld {
        String::new()
    } else {
        default_suffix_list()
    };
    let var = crate::var::Variable::with_simple_string(
        Bytes::from(value.into_bytes()),
        crate::var::VarOrigin::Default,
        None,
        None,
    );
    session.globals.define(sym, var);
}

/// The name a built-in recipe is reported against, which is GNU Make's own:
/// there is no Makefile line to point a reader at.
pub const BUILTIN_LOCATION: &str = "<builtin>";

/// The recipe the catalogue holds for a suffix-rule name, if it holds one.
///
/// GNU Make's `install_default_suffix_rules` writes these onto files it enters
/// by name, so the lookup is by the whole name — `.c` for a link rule and
/// `.c.o` for a compile rule — rather than by a pair of suffixes it took apart.
pub fn suffix_recipe(name: &[u8]) -> Option<&'static str> {
    DEFAULT_SUFFIX_RULES
        .iter()
        .find(|(rule_name, _)| rule_name.as_bytes() == name)
        .map(|(_, recipe)| *recipe)
}

/// Build one pattern rule of the catalogue.
///
/// `prerequisites` is a space-separated list because that is how `default.c`
/// writes it and because a rule with two of them — `%.c: %.w %.ch` — is in the
/// table. `terminal` is spelled as a double-colon rule, which is the same claim
/// written the way a Makefile would write it.
///
/// # Errors
///
/// Returns a parse failure for a table entry, which is a defect in the table.
pub fn pattern_rule(
    session: &mut Session,
    target: &str,
    prerequisites: &str,
    recipe: &'static str,
    terminal: bool,
) -> Result<Rule> {
    let mut rule = Rule::new(builtin_loc(session), terminal, false);
    rule.output_patterns
        .push(session.intern(target.as_bytes().to_vec()));
    for prerequisite in prerequisites.split_whitespace() {
        let sym = session.intern(prerequisite.as_bytes().to_vec());
        rule.inputs.push(sym);
        rule.prerequisite_names.push(sym);
    }
    rule.cmds = crate::rule::Recipe::builtin(recipe);
    Ok(rule)
}

/// Where a built-in rule says it came from.
pub fn builtin_loc(session: &mut Session) -> Loc {
    Loc {
        filename: session.intern(BUILTIN_LOCATION),
        line: 0,
    }
}

/// Split a table recipe into the command lines GNU Make's `chop_commands`
/// makes of it, and parse each one.
///
/// Nothing is trimmed: the space that begins the second line of `.y.c`'s recipe
/// is in `default.c` and reaches the shell. An empty string is no recipe at
/// all rather than one empty command line, which is what the suffix rules that
/// exist only to be matched are written as.
///
/// # Errors
///
/// Returns a parse failure for a table entry, which is a defect in the table.
pub fn recipe_lines(session: &mut Session, recipe: &str) -> Result<Vec<ValueId>> {
    let mut cmds = Vec::new();
    if recipe.is_empty() {
        return Ok(cmds);
    }
    for line in recipe.split('\n') {
        let text = Bytes::copy_from_slice(line.as_bytes());
        let mut loc = builtin_loc(session);
        cmds.push(parse_expr(session, &mut loc, text, ParseExprOpt::Command)?);
    }
    Ok(cmds)
}

/// The variable names the catalogue's recipes refer to that nothing else in a
/// session's startup interns.
///
/// A built-in recipe is parsed when something asks to run it — see
/// [`crate::rule::Recipe`] — and a session that runs none of them parses none.
/// But parsing is also what used to intern the names those recipes refer to,
/// and `$(.VARIABLES)` is a list of every name a session has interned, in the
/// order it interned them (`GlobalVars::matching_named`). Deferring the parse
/// would take these six names out of that list, or move them within it,
/// depending on what a makefile did later — the one way a symbol's ordinal is
/// observable in output.
///
/// So they are interned where the parse used to intern them, and the parse
/// that follows finds them already there. Every other name the tables refer to
/// — `CC`, `CFLAGS`, `TARGET_ARCH` and the rest — is interned before this by
/// [`crate::builtins::install_default_variables`], which parses each built-in
/// variable's value.
///
/// [`recipes_intern_no_name_outside_the_catalogue_list`] is what keeps this
/// list honest: it parses every recipe in the tables and fails if one of them
/// mints a name that is not here.
/// In the order the recipes refer to them, walking the three tables in the
/// order the installers walk them, so that each of the six lands on the
/// ordinal the parse used to give it.
pub const RECIPE_ONLY_VARIABLE_NAMES: &[&str] = &[
    "LOADLIBES",
    "LDLIBS",
    "MAKEINFO_FLAGS",
    "TEXI2DVI_FLAGS",
    "GFLAGS",
    "SCCS_OUTPUT_OPTION",
];

/// Intern the names in [`RECIPE_ONLY_VARIABLE_NAMES`], in table order.
pub fn intern_recipe_variable_names(session: &mut Session) {
    for name in RECIPE_ONLY_VARIABLE_NAMES {
        session.intern(*name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name the catalogue's recipes mint is on
    /// [`RECIPE_ONLY_VARIABLE_NAMES`], and every name on that list is one they
    /// mint.
    ///
    /// The list exists because the recipes are no longer parsed at install —
    /// see [`crate::rule::Recipe`] — and interning them there is what keeps a
    /// symbol's ordinal, and so `$(.VARIABLES)`, where it was. A name added to
    /// a table recipe without being added here would be interned late again;
    /// a name left here after the recipe referring to it went would intern a
    /// symbol nothing asks for. This fails on either.
    #[test]
    fn recipes_intern_no_name_outside_the_catalogue_list() {
        let mut session = Session::new();
        crate::builtins::install_default_variables(&mut session)
            .expect("the built-in variable table parses");
        // `<builtin>` is minted by rule construction rather than by a recipe
        // parse, and it is minted before the first recipe either way.
        let _ = builtin_loc(&mut session);
        let before = session.symtab.count();

        let mut recipes = DEFAULT_SUFFIX_RULES
            .iter()
            .map(|(_, recipe)| *recipe)
            .collect::<Vec<_>>();
        recipes.extend(DEFAULT_PATTERN_RULES.iter().map(|(_, _, recipe)| *recipe));
        recipes.extend(DEFAULT_TERMINAL_RULES.iter().map(|(_, _, recipe)| *recipe));
        for recipe in recipes {
            recipe_lines(&mut session, recipe).expect("a table recipe parses");
        }

        let minted = (before..session.symtab.count())
            .map(|index| {
                let sym = crate::symtab::Symbol::from_index(index)
                    .expect("an index below the interner's count names a symbol");
                String::from_utf8(session.symtab.name(sym).to_vec())
                    .expect("a built-in name is text")
            })
            .collect::<Vec<_>>();
        assert_eq!(minted, RECIPE_ONLY_VARIABLE_NAMES);
    }

    /// Interning the list twice mints nothing the second time, which is what
    /// says [`intern_recipe_variable_names`] is safe to call on a session that
    /// already read a makefile naming one of them.
    #[test]
    fn interning_the_recipe_names_twice_mints_them_once() {
        let mut session = Session::new();
        intern_recipe_variable_names(&mut session);
        let after_first = session.symtab.count();
        intern_recipe_variable_names(&mut session);
        assert_eq!(session.symtab.count(), after_first);
    }

    /// The suffix list is what every derived rule is keyed on, so a duplicate
    /// would silently make the same pair twice and a missing entry would take a
    /// rule away without any table saying so.
    #[test]
    fn default_suffixes_are_unique_and_dotted() {
        let mut seen = DEFAULT_SUFFIXES.to_vec();
        let recorded = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), recorded);
        assert!(DEFAULT_SUFFIXES.iter().all(|s| s.starts_with('.')));
        assert_eq!(recorded, 35);
    }

    /// Two entries under one name would leave the later one as dead text that
    /// still reads like the recipe.
    #[test]
    fn default_suffix_rule_names_are_unique() {
        let mut names = DEFAULT_SUFFIX_RULES
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>();
        let recorded = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), recorded);
        assert_eq!(recorded, 49);
    }

    /// Every table name is one the conversion can reach: a suffix on the list,
    /// or two of them written one after the other. A name nothing constructs is
    /// a recipe that can never run, and `.lm.m` is the one GNU Make has.
    #[test]
    fn every_suffix_rule_name_is_one_the_conversion_constructs() {
        let mut constructed = DEFAULT_SUFFIXES
            .iter()
            .map(|source| (*source).to_owned())
            .collect::<Vec<_>>();
        for source in DEFAULT_SUFFIXES {
            for target in DEFAULT_SUFFIXES {
                if source != target {
                    constructed.push(format!("{source}{target}"));
                }
            }
        }
        let unreachable = DEFAULT_SUFFIX_RULES
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !constructed.iter().any(|built| built == name))
            .collect::<Vec<_>>();
        assert_eq!(unreachable, [".lm.m"]);
    }
}
