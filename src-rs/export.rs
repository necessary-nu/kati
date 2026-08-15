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

//! The environment Make hands to something it starts.
//!
//! This is GNU Make's `target_environment` in `src/variable.c`, which answers
//! one question — given a scope, which names reach a child and with what bytes
//! — for all of the things Make starts: a recipe, a `$(shell)`, and a
//! recursive `$(MAKE)`. GNU passes a `struct file` for the first and `NULL`
//! for the rest, and the difference is only which variable sets are in the
//! chain, so one function serves all of them here too.
//!
//! What crosses out of here is a *delta* rather than a whole environment:
//! `Some(value)` for a name to set and `None` for one to remove. The caller
//! already runs in the environment the invocation was given, so a name Make
//! did not touch needs no entry — and must not get one, because writing an
//! entry means evaluating a value, and an environment variable's bytes are its
//! own rather than something Make expands.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use bytes::Bytes;

use std::os::unix::ffi::OsStrExt;

use crate::eval::Evaluator;
use crate::expr::Evaluable;
use crate::symtab::{Interner, Symbol};
use crate::var::{Var, VarExport, VarOrigin, Vars, is_exportable_name};

/// One name's fate in a child's environment: the bytes to set it to, or
/// nothing, meaning remove whatever the caller inherited under that name.
pub type EnvironmentChange = (Bytes, Option<Bytes>);

/// Quote one word so a shell reads back exactly these bytes.
fn push_shell_word(command: &mut Vec<u8>, word: &[u8]) {
    command.push(b'\'');
    for byte in word {
        if *byte == b'\'' {
            command.extend_from_slice(b"'\\''");
        } else {
            command.push(*byte);
        }
    }
    command.push(b'\'');
}

/// The `env` invocation that imposes `changes` on whatever command follows it,
/// or nothing at all when there is nothing to change.
///
/// A destination that cannot give an edge its own environment has to say the
/// same thing in the command line, and this is how it is said. `changes` is
/// read as a sequence of decisions — a later entry replaces an earlier one for
/// the same name, which is how a target's own `export` overrules the answer
/// its compilation unit reached — and what is emitted is the settled set, by
/// name, so the command line does not depend on how the answer was assembled.
#[must_use]
pub fn environment_prefix(changes: &[EnvironmentChange]) -> Vec<u8> {
    if changes.is_empty() {
        return Vec::new();
    }
    let settled = changes
        .iter()
        .cloned()
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut command = Vec::from(b"env".as_slice());
    for (name, value) in &settled {
        if value.is_none() {
            command.extend_from_slice(b" -u ");
            push_shell_word(&mut command, name);
        }
    }
    for (name, value) in &settled {
        if let Some(value) = value {
            let mut assignment = Vec::with_capacity(name.len() + value.len() + 1);
            assignment.extend_from_slice(name);
            assignment.push(b'=');
            assignment.extend_from_slice(value);
            command.push(b' ');
            push_shell_word(&mut command, &assignment);
        }
    }
    command.push(b' ');
    command
}

/// Which of the two environments GNU Make builds is being asked for.
///
/// The difference is only where the chain of variable sets starts, and it is
/// observable in exactly one place: a `private` binding is reachable from the
/// innermost set and from nothing further out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildKind {
    /// `target_environment(file, …)`: what a recipe runs in. The target's own
    /// set is innermost even when it is empty, so a `private` global is out of
    /// a recipe's reach.
    Recipe,
    /// `target_environment(NULL, …)`: what a `$(shell)` runs in, which is
    /// whatever sets are current where the call is being expanded. At read
    /// time that is the global set itself, so a `private` global does reach it.
    Expansion,
}

/// Whether this binding reaches a child, which is GNU Make's `should_export`.
///
/// A directive that named the variable decides on its own. Otherwise the
/// origin does: what the command line and the environment supplied is exported
/// without being asked, a built-in default or an automatic variable never is,
/// and everything else waits for a bare `export` or `.EXPORT_ALL_VARIABLES` —
/// and then only if its name is one a shell could read back.
fn should_export(name: Symbol, var: &Var, export_all: bool, names: &impl Interner) -> bool {
    let (attribute, origin) = {
        let var = var.read();
        (var.export, var.origin())
    };
    match attribute {
        VarExport::Export => return true,
        VarExport::NoExport => return false,
        VarExport::Default => {}
    }
    match origin {
        VarOrigin::Default | VarOrigin::Automatic => return false,
        VarOrigin::CommandLine | VarOrigin::Environment | VarOrigin::EnvironmentOverride => {}
        VarOrigin::File | VarOrigin::Override if export_all => {}
        VarOrigin::File | VarOrigin::Override => return false,
    }
    is_exportable_name(&name.as_bytes(names))
}

/// The bindings a child's environment is computed from.
///
/// GNU Make walks the scope chain from most specific to least and keeps the
/// first binding it finds for each name, so a target-specific value outranks
/// the global one. The export *attribute* travels the other way when the
/// specific binding has none of its own: `all: V = local` beside a global
/// `export V` is exported, because the target-specific assignment said nothing
/// about exporting and the global binding did.
fn resolved_bindings(
    ev: &Evaluator,
    scope: Option<&Vars>,
    kind: ChildKind,
) -> HashMap<Symbol, Var> {
    let mut resolved: HashMap<Symbol, Var> = HashMap::new();
    if let Some(scope) = scope {
        for (name, var) in scope.0.lock().iter() {
            resolved.insert(*name, var.clone());
        }
    }
    // With no target scope the global set is itself the innermost one, which
    // is why a `private` global reaches a read-time `$(shell)` and reaches no
    // recipe at all — a recipe always has a set of its own in front of it.
    let global_is_local = scope.is_none() && kind == ChildKind::Expansion;
    for (name, var) in ev.session.globals.matching(|_| true) {
        // A `private` binding is invisible from every set but its own — not
        // merely unexported. It lends nothing, so a target-specific binding
        // beside it is left to answer for itself.
        if !global_is_local && var.read().is_private {
            continue;
        }
        match resolved.entry(name) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let inner = entry.get();
                if inner.read().export == VarExport::Default {
                    let inherited = var.read().export;
                    inner.write().export = inherited;
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(var);
            }
        }
    }
    resolved
}

/// The environment changes a child of this evaluation should be started with.
///
/// `scope` is the target-specific set a recipe or a recipe-time `$(shell)` runs
/// under, and `kind` says which of GNU Make's two calls this is — see
/// [`ChildKind`], which is what decides whether a `private` global is in reach.
///
/// # Errors
///
/// Whatever expanding an exported variable's value rejects.
pub fn exported_environment(
    ev: &mut Evaluator,
    scope: Option<&Vars>,
    kind: ChildKind,
) -> Result<Vec<EnvironmentChange>> {
    let export_all = ev.session.flags.export_all_variables;
    let mut candidates = resolved_bindings(ev, scope, kind)
        .into_iter()
        .filter(|(name, var)| should_export(*name, var, export_all, &ev.session))
        .collect::<Vec<_>>();
    // By name, because a map's order is not one and a recipe's environment
    // should not depend on which way the hash fell.
    candidates.sort_by_cached_key(|(name, _)| name.as_bytes(&ev.session));

    let exported = candidates.iter().map(|(name, _)| *name).collect();
    let mut changes = Vec::with_capacity(candidates.len());
    for (name, var) in candidates {
        // An untouched environment variable goes back out as the bytes it came
        // in as. Its origin is still the environment precisely because nothing
        // replaced it, so there is nothing to write and nothing to expand.
        if matches!(
            var.read().origin(),
            VarOrigin::Environment | VarOrigin::EnvironmentOverride
        ) {
            continue;
        }
        // A recipe's environment cannot be waiting on itself; a `$(shell)`'s
        // can, and GNU Make answers that from the invocation's environment
        // rather than refusing the makefile.
        let guarded = kind == ChildKind::Expansion;
        let value = ev.expand_for_environment(name, &var, guarded)?;
        changes.push((name.as_bytes(&ev.session), Some(value)));
    }
    for name in withdrawn_names(ev, &exported) {
        changes.push((name.as_bytes(&ev.session), None));
    }
    changes.retain(|(name, _)| name != MAKELEVEL);
    changes.push((Bytes::from_static(MAKELEVEL), Some(child_makelevel(ev))));
    changes.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(changes)
}

/// The name whose value is one deeper in every child, whatever the makefile
/// did with it.
const MAKELEVEL: &[u8] = b"MAKELEVEL";

/// The depth a child of this evaluation runs at.
///
/// GNU Make sets this unconditionally in every environment it builds, and sets
/// it from its own counter rather than from the variable, so a makefile that
/// assigns `MAKELEVEL` changes what it reads and not what its children are
/// told. The counter is what the invocation was started with, which is why
/// this reads the invocation's environment rather than the variable store.
fn child_makelevel(ev: &Evaluator) -> Bytes {
    let level = invocation_value(ev, MAKELEVEL)
        .and_then(|value| std::str::from_utf8(&value).ok()?.trim().parse().ok())
        .unwrap_or(0usize);
    Bytes::from(level.saturating_add(1).to_string())
}

/// One name as the invocation's own environment carries it.
pub(crate) fn invocation_value(ev: &Evaluator, name: &[u8]) -> Option<Vec<u8>> {
    match &ev.session.invocation_environment {
        Some(environment) => environment
            .iter()
            .find(|(candidate, _)| candidate.as_os_str().as_encoded_bytes() == name)
            .map(|(_, value)| value.as_os_str().as_encoded_bytes().to_vec()),
        None => std::env::var_os(std::ffi::OsStr::from_bytes(name))
            .map(|value| value.as_os_str().as_encoded_bytes().to_vec()),
    }
}

/// What one target's own scope changes about the environment its recipe runs
/// in, over and above what the makefile's global export set already said.
///
/// A recipe's environment is the whole of [`exported_environment`] read with
/// that target's scope, and a front end that has already built the global part
/// once needs only the difference. That difference is bounded by the scope —
/// a name no target-specific assignment mentions cannot have a different
/// answer here — which is why this walks the scope rather than everything.
///
/// # Errors
///
/// Whatever expanding an exported variable's value rejects.
pub fn scoped_environment(ev: &mut Evaluator, scope: &Vars) -> Result<Vec<EnvironmentChange>> {
    let export_all = ev.session.flags.export_all_variables;
    let mut local = scope
        .0
        .lock()
        .iter()
        .map(|(name, var)| (*name, var.clone()))
        .collect::<Vec<_>>();
    local.sort_by_cached_key(|(name, _)| name.as_bytes(&ev.session));
    let mut changes = Vec::new();
    for (name, var) in local {
        // A `private` global is invisible from a target's scope, so it neither
        // lends its export attribute nor leaves anything to withdraw.
        let global = ev
            .session
            .peek_global_var(name)
            .filter(|global| !global.read().is_private);
        // A target-specific assignment that said nothing about exporting takes
        // the global binding's answer, which is how `all: V = local` beside a
        // global `export V` reaches the recipe's environment.
        if var.read().export == VarExport::Default
            && let Some(global) = &global
        {
            let inherited = global.read().export;
            var.write().export = inherited;
        }
        if should_export(name, &var, export_all, &ev.session) {
            let value = var.read().eval_to_buf_mut(ev)?.freeze();
            changes.push((name.as_bytes(&ev.session), Some(value)));
            continue;
        }
        // The scope hides a name the global set exported, so the recipe has to
        // be started without it rather than with the value the global answer
        // already put in the wrapper.
        if global.is_some_and(|global| should_export(name, &global, export_all, &ev.session)) {
            changes.push((name.as_bytes(&ev.session), None));
        }
    }
    Ok(changes)
}

/// What a recipe has since told the export set, as changes to impose on a child
/// started now.
///
/// GNU Make never needs this: it builds a child's environment out of the export
/// set when the job starts, so everything an earlier recipe did is already in
/// it. Ronin settles the compilation unit's environment once, before any recipe
/// runs, and this is the correction for the names a recipe has spoken about
/// since — an `export`, an `unexport`, or a new value for a name already
/// exported.
///
/// A name with no global binding left is withdrawn rather than dropped: the
/// settled environment may still be carrying it.
///
/// # Errors
///
/// Whatever expanding an exported variable's value rejects.
pub fn late_environment(ev: &mut Evaluator, names: &[Symbol]) -> Result<Vec<EnvironmentChange>> {
    let export_all = ev.session.flags.export_all_variables;
    let mut names = names.to_vec();
    // By name, so a recipe's environment does not depend on which way the hash
    // fell.
    names.sort_by_cached_key(|name| name.as_bytes(&ev.session));
    let mut changes = Vec::with_capacity(names.len());
    for name in names {
        let global = ev
            .session
            .peek_global_var(name)
            .filter(|global| !global.read().is_private);
        let exported = global.filter(|global| should_export(name, global, export_all, &ev.session));
        let value = match exported {
            Some(var) => Some(var.read().eval_to_buf_mut(ev)?.freeze()),
            None => None,
        };
        changes.push((name.as_bytes(&ev.session), value));
    }
    Ok(changes)
}

/// The names the invocation's own environment carries that this child must not
/// see: `unexport`ed, `undefine`d, or replaced by a binding that is not
/// exported.
fn withdrawn_names(ev: &Evaluator, exported: &HashSet<Symbol>) -> Vec<Symbol> {
    let inherited = ev
        .session
        .invocation_environment
        .clone()
        .unwrap_or_else(|| std::env::vars_os().collect());
    let mut withdrawn = Vec::new();
    for (name, _) in inherited {
        let bytes = name.as_os_str().as_encoded_bytes();
        // GNU Make refuses to let a makefile's SHELL reach a child and
        // re-emits the one the invocation was given, so the inherited value
        // stands whatever the makefile did with the name.
        if bytes == b"SHELL" {
            continue;
        }
        let Some(symbol) = ev.session.symtab.peek_symbol(bytes) else {
            continue;
        };
        if !exported.contains(&symbol) {
            withdrawn.push(symbol);
        }
    }
    withdrawn
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Evaluator;
    use crate::session::Session;

    /// Read `source` as a makefile in a session with a known environment, and
    /// answer with the environment changes one of its children is started
    /// with.
    fn environment_of(source: &str, kind: ChildKind) -> Vec<(String, Option<String>)> {
        use std::ffi::OsString;
        let mut session = Session::new();
        session.invocation_environment = Some(vec![
            (OsString::from("INHERITED"), OsString::from("kept")),
            (OsString::from("MAKELEVEL"), OsString::from("2")),
        ]);
        let mut ev = Evaluator::new(session);
        let statements = crate::parser::parse_buf(
            &mut ev.session,
            &Bytes::from(source.as_bytes().to_vec()),
            crate::loc::Loc::default(),
        )
        .expect("the makefile parses");
        let statements = statements.lock().clone();
        for statement in statements {
            statement.eval(&mut ev).expect("the makefile evaluates");
        }
        exported_environment(&mut ev, None, kind)
            .expect("an environment")
            .into_iter()
            .map(|(name, value)| {
                (
                    String::from_utf8_lossy(&name).into_owned(),
                    value.map(|value| String::from_utf8_lossy(&value).into_owned()),
                )
            })
            .collect()
    }

    fn recipe_environment(source: &str) -> Vec<(String, Option<String>)> {
        environment_of(source, ChildKind::Recipe)
    }

    /// What the changes say about one name: `None` for no entry at all,
    /// `Some(None)` for a removal, `Some(Some(v))` for a value.
    fn entry(environment: &[(String, Option<String>)], name: &str) -> Option<Option<String>> {
        environment
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value.clone())
    }

    fn set_to(value: &str) -> Option<Option<String>> {
        Some(Some(value.to_owned()))
    }

    /// The name filter is GNU Make's: a leading letter or underscore, then
    /// letters, digits and underscores, and nothing else at all.
    #[test]
    fn test_exportable_names_are_the_ones_a_shell_can_read() {
        assert!(is_exportable_name(b"A"));
        assert!(is_exportable_name(b"_"));
        assert!(is_exportable_name(b"_A1"));
        assert!(is_exportable_name(b"lower_9"));
        assert!(!is_exportable_name(b""));
        assert!(!is_exportable_name(b"1A"));
        assert!(!is_exportable_name(b"A.B"));
        assert!(!is_exportable_name(b"A-B"));
        assert!(!is_exportable_name(b"A B"));
    }

    /// A bare `export` skips a name no shell could read back, and an `export`
    /// that says the name outright still exports it.
    ///
    /// No build-intent case can hold this half of it: the shell a recipe runs
    /// under drops such a name while it starts, so the only place the
    /// difference is visible is the environment as it is handed over.
    #[test]
    fn test_a_named_export_outranks_the_name_filter() {
        let environment = recipe_environment("A.B = dotted\nA_B = fine\nexport\n");
        assert_eq!(entry(&environment, "A_B"), set_to("fine"));
        assert_eq!(entry(&environment, "A.B"), None);

        let environment = recipe_environment("export A.B = dotted\n");
        assert_eq!(entry(&environment, "A.B"), set_to("dotted"));
    }

    /// A variable the makefile never touched is left out of the changes
    /// entirely, so its bytes reach the child as they arrived rather than
    /// through an expansion that would read Make syntax in them. Withdrawing
    /// it is what puts it back in, as a removal.
    #[test]
    fn test_an_untouched_environment_variable_is_not_rewritten() {
        assert_eq!(entry(&recipe_environment("all: ;@:\n"), "INHERITED"), None);
        assert_eq!(
            entry(&recipe_environment("unexport INHERITED\n"), "INHERITED"),
            Some(None),
        );
    }

    /// Every environment Make builds says how deep the child is, counting from
    /// the invocation rather than from whatever the makefile assigned.
    #[test]
    fn test_the_child_is_one_level_deeper_than_the_invocation() {
        assert_eq!(
            entry(&recipe_environment("MAKELEVEL = 9\n"), "MAKELEVEL"),
            set_to("3"),
        );
    }

    /// A `private` global is out of a recipe's reach and within a read-time
    /// `$(shell)`'s, which is the whole of what GNU Make's two calls differ by.
    #[test]
    fn test_a_private_global_reaches_an_expansion_and_no_recipe() {
        let source = "private export P = p\nexport Q = q\n";
        let recipe = environment_of(source, ChildKind::Recipe);
        assert_eq!(entry(&recipe, "P"), None);
        assert_eq!(entry(&recipe, "Q"), set_to("q"));

        let expansion = environment_of(source, ChildKind::Expansion);
        assert_eq!(entry(&expansion, "P"), set_to("p"));
        assert_eq!(entry(&expansion, "Q"), set_to("q"));
    }

    /// The `env` words are the settled set: a later decision replaces an
    /// earlier one for the same name, so a target's own export overrules the
    /// answer its compilation unit reached rather than being overruled by it.
    #[test]
    fn test_a_later_change_settles_a_name() {
        let unit = (Bytes::from_static(b"V"), Some(Bytes::from_static(b"unit")));
        let target = (Bytes::from_static(b"V"), None);
        assert_eq!(
            environment_prefix(&[unit.clone(), target.clone()]),
            b"env -u 'V' ".to_vec(),
        );
        assert_eq!(
            environment_prefix(&[target, unit]),
            b"env 'V=unit' ".to_vec(),
        );
        assert_eq!(environment_prefix(&[]), Vec::<u8>::new());
    }
}
