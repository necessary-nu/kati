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

//! Where a converted Makefile goes.
//!
//! kati turns a Makefile into a dependency graph and then hands that graph to
//! something. Today the only something is [`crate::ninja::NinjaWriter`], which
//! serialises it as `build.ninja`. This module is the seam: the front end
//! computes edges and rules, a [`BuildSink`] decides what they become.
//!
//! Two things deliberately do *not* cross this trait, because both exist only
//! because the destination is a text file:
//!
//! * **Escaping.** `build.ninja` needs `$`, `:` and space escaped in paths, and
//!   the shell command quoted. A sink that builds a graph in memory needs none
//!   of that, and a path that arrived pre-escaped would be silently wrong in a
//!   way no test would notice for a long time. So every name here is the
//!   unescaped one, as a [`Symbol`] or as raw bytes, and escaping happens
//!   inside the writer.
//!
//!   Commands are the case where this is easy to get wrong, because a command
//!   with the wrong escaping still round-trips through a manifest: ninja
//!   unescapes on the way in, so a doubled `$` reaches the shell as one either
//!   way, and only a sink that skips the unescaping sees the difference. The
//!   rule is the same as for paths, and it decides which: what crosses here is
//!   what the *shell* should receive.
//!
//! * **Ninja expressions.** A manifest can say `$out` and have it mean a
//!   different path per edge. That is a property of the format, not of the
//!   build, so nothing shaped like it crosses this trait — the writer keeps
//!   the response-file path kati relies on, and a sink that cannot evaluate it
//!   never receives it.
//!
//! * **`_kati_always_build_`.** A `.PHONY` target names no file, so nothing can
//!   ever decide it is up to date. In a manifest kati expresses that by giving
//!   the edge a synthetic input that is itself always dirty. That is one way to
//!   get the property, not the property itself, so what crosses the trait is
//!   [`SinkEdge::always_dirty`] and the writer invents the synthetic input if
//!   its output format needs one.
//!
//! See `plan/decisions/make-as-graph.md`.

use anyhow::Result;

use crate::loc::Loc;
use crate::symtab::{Interner, Symbol};

/// Names the rule an edge runs. Unique within one generation.
pub type RuleId = usize;

/// When the final value of Make's `$?` automatic variable is determined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NewInputsTiming {
    /// The generated recipe computes `$?` immediately before invoking its
    /// command because the destination cannot represent a scheduler boundary.
    RecipeShell,
    /// The destination computes `$?` after prerequisites settle and before it
    /// launches the edge.
    SchedulerBoundary,
    /// The recipe is being expanded at the moment the destination launches it,
    /// so every prerequisite has already been brought up to date and the
    /// timestamps on disk are the ones GNU Make compares. The value is
    /// computed here rather than deferred to anyone.
    Launch,
}

/// When a recipe is expanded into the command text that will run.
///
/// GNU Make expands a recipe immediately before running it, and only if the
/// target is out of date, so every evaluation-time effect the recipe carries —
/// a `$(shell)`, an `$(info)`, an `$(error)`, what `$(wildcard)` can see —
/// happens at that moment or never. A destination that has to hold command
/// text before anything runs cannot have that, and a destination that runs the
/// build itself can.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecipeExpansion {
    /// Every reachable recipe is expanded while the graph is constructed. A
    /// manifest holds command text, so a manifest writer needs the text before
    /// any of it runs.
    Construction,
    /// A recipe the destination can hold unexpanded is left unexpanded, and the
    /// destination expands it through [`DeferredRecipes`](crate::ninja::DeferredRecipes)
    /// when it is about to run it. A recipe whose text the compiler itself
    /// must read — a recursive `$(MAKE)` line, an automatic depfile, a `$?`
    /// whose value the scheduler binds — is expanded at construction whatever
    /// the destination asked for, because the graph's shape depends on it.
    Launch,
}

/// Which part of a name a settled-name reference stands for.
///
/// The three forms GNU Make reads a prerequisite in: the name, the directory
/// it carries, and the name with that directory taken off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettledNameView {
    /// The whole name.
    Whole,
    /// `$(<D)` and its neighbours: the directory the name carries.
    Directory,
    /// `$(<F)` and its neighbours: the name with the directory taken off.
    Filename,
}

/// One name a recipe read before the build could not spell, and the name it
/// carries instead.
///
/// A prerequisite the directory search answered about has two names, and which
/// of them stands is settled by whether the build has to remake it — later than
/// any expansion the compiler performs. A recipe the compiler had to read for
/// itself therefore cannot hold the prerequisite's name at all, so it holds
/// [`Self::variable`] and the destination substitutes the spelling it settled
/// on. Nothing here reads either name: the pair is carried, not interpreted.
#[derive(Debug)]
pub struct SettledName {
    /// The name the command reads the spelling from.
    pub variable: Symbol,
    /// The prerequisite it stands for.
    pub input: Symbol,
    /// Which part of that prerequisite's settled name to substitute.
    pub view: SettledNameView,
}

/// Names a recipe left unexpanded for the destination to expand at launch.
pub type DeferredRecipeId = usize;

/// Who answers a `$(shell)` written inside a recipe.
///
/// GNU Make always answers it itself, while it expands the recipe. kati grew
/// the other answer for Android: when the destination is a manifest some other
/// program will run, writing the call through as a shell command substitution
/// keeps it out of every regeneration. That is a different language — the
/// value never reaches a Make function, and a quoted one is never substituted
/// at all — so it is the destination's choice rather than kati's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShellEvaluation {
    /// The recipe's own shell answers it, as a command substitution written
    /// into the generated command.
    RecipeShell,
    /// Make answers it while it expands the recipe, which is GNU Make's
    /// `func_shell`.
    Expansion,
}

/// Who performs a `$(file ...)` written inside a recipe.
///
/// GNU Make always performs it itself, while it expands the recipe: a read
/// answers from the file as it stands at that moment, and a write happens
/// before the first line of the recipe runs. kati refused both outright,
/// because its destination is a manifest — the expansion that would do the
/// work happens while the manifest is written, which is a different run of a
/// different build, and the answer it produced would be frozen into a file
/// that outlives it. That refusal is the manifest writer's, not the
/// evaluator's: a destination that runs the build itself expands the recipe
/// where GNU Make expands it, and can do exactly what GNU Make does there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileEvaluation {
    /// Refused wherever a rule can reach it, which is kati's own behaviour and
    /// what a manifest writer keeps.
    Refused,
    /// Performed while the recipe is expanded, which is GNU Make's
    /// `func_file`.
    Expansion,
}

/// Who performs an `$(info ...)`, `$(warning ...)` or `$(error ...)` written
/// inside a recipe.
///
/// GNU Make performs all three while it expands the recipe, in `new_job`,
/// before any command line exists. The text is not a command and never becomes
/// one: a recipe whose whole expansion is `$(info X)` prints X and then has no
/// command line at all, so no shell starts, the target reports up to date, and
/// `-q` answers zero. kati turned each into a `printf` written into the recipe,
/// which is the only thing a manifest writer can do — the text has to survive
/// to a run that happens on another day. A destination that runs the build
/// itself expands the recipe in the process that is about to run it, which is
/// exactly where GNU Make prints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputEvaluation {
    /// Written into the recipe as a command that prints when it runs, which is
    /// kati's own behaviour and what a manifest writer keeps.
    RecipeCommand,
    /// Performed while the recipe is expanded, which is GNU Make's
    /// `func_info`, `func_warning` and `func_error`.
    Expansion,
}

/// A pool kati declares for itself.
///
/// Pools kati only *refers* to — the ones `.KATI_NINJA_POOL` and
/// `--default_pool` name — are declared by whoever wrote the ninja fragment
/// that includes kati's output, and never appear here.
pub struct SinkPool<'a> {
    pub name: &'a [u8],
    /// How many edges bound to this pool may run at once.
    pub depth: usize,
}

/// How a rule's shell script reaches the shell.
///
/// The split is not cosmetic: a command too long to pass as an argument is
/// invoked differently, as the shell reading a script file rather than as the
/// shell given `-c` and a string.
#[derive(Clone, Copy)]
pub enum SinkCommand<'a> {
    /// `<shell> <shell_flags> "<script>"`.
    Inline(&'a [u8]),
    /// The script does not fit in an argument list, so it has to reach the
    /// shell as a file instead: `<shell> <shell_flags less the `c`> <script
    /// file>`.
    ///
    /// The flags still apply — the script is the same script and the shell is
    /// the same shell — but the letter saying the next word is the command
    /// would take the file name for it, so
    /// [`script_file_flags`](crate::ninja::script_file_flags) takes that one
    /// off and copies the rest. Dropping the value here is how a `.POSIX:`
    /// recipe silently lost its `-e` on crossing the length threshold.
    ///
    /// Which file is not said here. A sink that runs the script itself may not
    /// need one at all, and the manifest writer names it in the only terms a
    /// manifest has.
    ResponseFile(&'a [u8]),
}

/// One recursive command extracted from a recipe for semantic graph inclusion.
pub struct SinkSubninja<'a> {
    /// The expanded command line selecting the child Make compilation.
    pub command: &'a [u8],
    /// The value produced by the `MAKE` reference in command position.
    pub make: &'a [u8],
    /// The recipe's own lines written after the previous invocation and before
    /// this one, assembled into one script.
    ///
    /// GNU Make runs a recipe's lines in the order they were written, and a
    /// line before an invocation runs before the Make that invocation starts
    /// reads anything. A sink that compiles the child instead of starting it
    /// therefore has to run this before it compiles: the only state one shell
    /// line can hand the next is on the filesystem, and a Makefile the child
    /// includes is read from there.
    pub preceding: Option<SinkCommand<'a>>,
    /// Whether every line in [`Self::preceding`] ignores failure.
    pub preceding_ignore_errors: bool,
    /// The Makefile and line the recipe line was written on, rendered here for
    /// the same reason [`crate::census::Invocation::location`] is: the file
    /// name lives in an interner the sink does not hold.
    ///
    /// `None` when nobody asked for a report, because rendering it costs an
    /// interner lookup and a string a build would drop.
    pub location: Option<&'a str>,
}

/// The half of a Make dependency node that Ninja binds to a `rule`.
///
/// A `DepNode` is one thing, but Ninja splits what it carries across two
/// declarations, and the split is not arbitrary: a rule is *how to build*, an
/// edge is *what is built from what*. Rules are shareable in Ninja's model even
/// though kati never shares one — it mints exactly one rule per node that has
/// commands — so a sink must be able to receive them separately.
pub struct SinkRule<'a> {
    /// What [`SinkEdge::rule`] will name.
    pub id: RuleId,
    /// The shell to run the script under, unescaped.
    pub shell: &'a [u8],
    /// The flags that make the shell take a script as an argument, unescaped.
    ///
    /// [`SinkCommand::ResponseFile`] reads them too, through
    /// [`script_file_flags`](crate::ninja::script_file_flags): the script
    /// arrives another way, and everything else these say about the shell
    /// still holds.
    pub shell_flags: &'a [u8],
    /// The assembled shell script, as the shell should receive it. A `$` here
    /// is a `$` the shell will act on, never an escape belonging to some
    /// destination format.
    ///
    /// Empty when [`Self::deferred_recipe`] names one: there is no text yet,
    /// because the recipe it comes from has not been expanded.
    pub command: SinkCommand<'a>,
    /// What this target's own scope changes about the environment its recipe
    /// runs in, over the export set the whole compilation unit agreed on.
    ///
    /// A name bound to `Some` bytes is set and one bound to `None` is removed.
    /// Empty for all but a target carrying an `export` of its own, which is
    /// why a sink may treat it as the exception rather than the rule.
    ///
    /// Absent when [`Self::deferred_recipe`] names one: the scope is read as
    /// the recipe is expanded, and arrives with it.
    pub recipe_environment: &'a [crate::export::EnvironmentChange],
    /// The recipe this rule will run, still unexpanded, for a sink that asked
    /// for [`RecipeExpansion::Launch`].
    ///
    /// `Some` means every command-derived field here — the script, the shell
    /// flags, the description, the depfile, whether failure is ignored — is
    /// absent rather than empty, and the sink must obtain them from
    /// [`DeferredRecipes::expand`](crate::ninja::DeferredRecipes::expand)
    /// before it runs the rule. Nothing else about the edge is deferred: its
    /// shape is settled here as it always was.
    pub deferred_recipe: Option<DeferredRecipeId>,
    /// The processes the recipe behind [`Self::command`] really is, in
    /// written order, for a sink that can run several.
    ///
    /// GNU Make runs each command line of a recipe as its own process, and
    /// [`Self::command`] is the whole recipe assembled into one script for a
    /// destination that can only hold one. A destination that can launch them
    /// separately does, and the assembled script stays as the edge's name —
    /// what a progress line, a log entry and a `-n` all want.
    ///
    /// Empty when the recipe's own shape makes the split unsound rather than
    /// merely unwanted: a script the depfile extraction rewrote, a recipe
    /// composed into child graphs whose edge does not run the whole of it, and
    /// a recipe of nothing. A sink is free to refuse the split for reasons of
    /// its own on top of that — a line too long to be an argument is one.
    ///
    /// A recipe the sink asked to have deferred carries its steps on the
    /// expansion instead, where the text finally exists.
    pub steps: &'a [crate::ninja::RecipeStep],
    /// Recursive `$(MAKE)` invocations extracted from the recipe in their
    /// written order. A graph sink compiles these as semantic subninjas rather
    /// than handing nested Make processes to its executor.
    ///
    /// Empty for a recipe that names recursion nothing can lift out of it: a
    /// multi-line `.ONESHELL` recipe, whose lines share a shell a split would
    /// lose, and a line whose invocation sits behind a runtime test or beside
    /// work of its own. Such a recipe is not refused. It reaches the executor
    /// as the script it is, and the Make it names starts — which is what GNU
    /// Make does with it, and the only answer that leaves nothing unbuilt.
    // [spec:ronin:req:make.recursive-invocation+2]
    pub subninjas: &'a [SinkSubninja<'a>],
    /// The recipe's own lines written after the last invocation, assembled
    /// into one script. A graph sink runs this parent action after the child
    /// graphs, which is where the recipe wrote it. Lines written before an
    /// invocation are on that invocation's
    /// [`SinkSubninja::preceding`] instead.
    pub residual_command: Option<SinkCommand<'a>>,
    /// Whether every residual line ignores failure.
    pub residual_ignore_errors: bool,
    /// What to print while the command runs, if the Makefile said — literal
    /// text, with the shell quoting already off.
    ///
    /// `None` is not "print nothing": it is nobody having chosen, which leaves
    /// the choice to the sink. A sink can use a short inline recipe itself and
    /// leave an oversized response-file command undescribed.
    pub description: Option<&'a [u8]>,
    /// A file the command writes its discovered dependencies to, in the format
    /// `cc -MD` produces. kati never emits any other format.
    pub depfile: Option<&'a [u8]>,
    /// The command may leave its output unchanged, so downstream edges should
    /// re-check rather than assume they are dirty.
    pub restat: bool,
    /// A nonzero status from this command is an error Make was told to ignore:
    /// the `-` recipe prefix on every line, `-i`, or `.IGNORE`.
    ///
    /// The script keeps the status rather than throwing it away, so a sink that
    /// runs the build can say what it was and carry on — which is the whole of
    /// GNU Make's `Error N (ignored)`. Ninja has no notion of an edge allowed
    /// to fail, so the manifest writer answers for it instead.
    pub ignore_errors: bool,
    /// Android's ninja fork runs this command outside its sandbox.
    pub sandbox_disabled: bool,
    /// The line of the Makefile this came from.
    pub loc: Option<&'a Loc>,
}

/// The half of a Make dependency node that Ninja binds to a `build` statement:
/// what is produced, what it is produced from, and the bindings that are per
/// edge rather than per rule.
pub struct SinkEdge<'a> {
    /// The rule that builds these outputs, or `None` for a node with no
    /// commands at all — which Ninja expresses with its built-in `phony` rule,
    /// and which has nothing to do with [`Self::always_dirty`].
    pub rule: Option<RuleId>,
    /// The primary output. Every kati edge has exactly one.
    pub output: Symbol,
    /// Outputs the command also writes, which nothing may name as a dependency
    /// of this edge.
    pub implicit_outputs: &'a [Symbol],
    /// Inputs whose contents the command depends on.
    pub inputs: &'a [Symbol],
    /// Inputs that must exist before the command runs but whose timestamps do
    /// not make it dirty.
    pub order_only_inputs: &'a [Symbol],
    /// The subset of [`Self::order_only_inputs`] whose failure this edge is
    /// willing to outlive: the wait holds and the status waited for is
    /// discarded.
    ///
    /// Ninja's manifest has no spelling for it — an order-only input there
    /// blocks its consumer when it fails — so a sink that writes `build.ninja`
    /// ignores this and a sink with scheduler support keeps it as graph
    /// metadata. It exists because GNU Make's double-colon chain under `-k`
    /// runs a target's later entries after an earlier entry failed, which is an
    /// ordering with the status taken out of it.
    pub forgiven_order_only_inputs: &'a [Symbol],
    /// Targets to build alongside this one, whose results it does not consume.
    pub validations: &'a [Symbol],
    /// The Make target is `.PHONY`: it names no file, so nothing can find it up
    /// to date and it must run every time it is reached.
    ///
    /// This says the property, not the trick. The manifest writer either binds
    /// `phony_output` or wires in a synthetic always-dirty input; a sink that
    /// builds a graph directly should do neither.
    pub always_dirty: bool,
    /// Real outputs whose pre-prerequisite state defines this edge's
    /// freshness.  An empty slice means ordinary timestamp semantics.
    ///
    /// The edge's own output may be a private virtual completion identity;
    /// sinks with scheduler support keep this relation as graph metadata.
    pub deferred_freshness_outputs: &'a [Symbol],
    /// One of the deferred freshness outputs is phony, so reaching the action
    /// always requires it to run even if a file with that spelling exists.
    pub deferred_freshness_always_dirty: bool,
    /// Normal inputs that are phony and therefore always belong to the late
    /// new-input set used by the recipe.
    pub deferred_always_new_inputs: &'a [Symbol],
    /// Normal inputs hidden from the late new-input value by Make evaluation,
    /// for example by `$(filter-out $(PHONY),$?)`.
    ///
    /// They still participate in the edge's freshness decision; only the
    /// automatic variable's published value omits them.
    pub deferred_excluded_new_inputs: &'a [Symbol],
    /// Normal inputs the late value spells differently from the name the graph
    /// knows them by, paired with the spelling to publish.
    ///
    /// `lib.a(m.o)` is the one prerequisite whose name in the graph is not the
    /// name the recipe reads: GNU Make puts only the member into `$?`, `$^`,
    /// `$+` and `$<` (src/commands.c), and every one of those but `$?` is
    /// expanded here, where the reduction is already made. `$?` is the one
    /// whose value cannot be known until the prerequisites have settled, so the
    /// spelling has to travel with the deferral rather than be applied to it —
    /// a destination that substitutes the value assigns no meaning to these
    /// names and must not have to know that a parenthesis is one.
    ///
    /// Empty for every edge with nothing to respell, which is every edge that
    /// mentions no archive.
    pub deferred_new_input_names: &'a [(Symbol, Symbol)],
    /// Prerequisite spellings a recipe the compiler had to read left for the
    /// destination to fill in, because the directory search had answered about
    /// them and the build had not yet chosen between the two names.
    ///
    /// Empty for every edge whose recipe the destination expands itself, and
    /// for every edge none of whose prerequisites was searched for — which is
    /// nearly all of them.
    pub settled_names: &'a [SettledName],
    /// This edge publishes a real output only after its private action inputs
    /// have settled.  It runs no command itself.
    pub completion_join: bool,
    /// The output's absence is no reason to remake what reads it: the implicit
    /// rule search invented the name to complete a chain, or `.INTERMEDIATE` or
    /// `.SECONDARY` said so.
    ///
    /// Nothing in a manifest says this — Ninja has no notion of a file the
    /// build is allowed to skip — so the writer ignores it and a sink that runs
    /// the build is the one that answers for it.
    pub intermediate: bool,
    /// The build should delete this output once it has finished with it, which
    /// is every intermediate but a `.SECONDARY` one and a goal.
    pub disposable: bool,
    /// The outputs a stopped recipe may be made to give back.
    ///
    /// The names rather than a flag, because the exclusions are per output and
    /// not per edge: `.PRECIOUS` protects one member of a grouped record while
    /// its peers still go, and a `.PHONY` name stands for no file at all. Which
    /// of them the stopped recipe actually touched is the running build's
    /// question and not this one's, so every eligible name is listed and the
    /// sink compares timestamps.
    ///
    /// Listed whatever the Makefile said about `.DELETE_ON_ERROR`, which is
    /// [`Self::delete_on_error`] and only one of the two reasons to ask for
    /// them. An empty list from a Makefile means there is nothing here to take
    /// back; a manifest says nothing at all and the sink may tell the two
    /// apart, because Ninja withdraws everything a cut-short command wrote.
    ///
    /// Nothing in a manifest says this — Ninja has no notion of an output the
    /// build withdraws when the command fails — so the writer ignores it, as it
    /// ignores [`Self::intermediate`] and [`Self::disposable`], and a sink that
    /// runs the build is the one that answers for it.
    pub withdrawable_outputs: &'a [Symbol],
    /// Whether an ordinary failure is reason enough to withdraw them.
    ///
    /// GNU Make asks `exit_sig != 0 || delete_on_error`: a recipe killed by a
    /// signal is cleaned up after whatever the Makefile said, and this is the
    /// other reason. Read per Makefile unit, so a recursive Make that declares
    /// `.DELETE_ON_ERROR` does not answer for the parent that did not.
    pub delete_on_error: bool,
    /// The outputs among [`Self::implicit_outputs`] this recipe makes only on
    /// the way to making something that was asked for — GNU Make's `also_make`.
    ///
    /// A pattern rule spelling several target patterns is one recipe for all of
    /// them, and GNU Make still decides each of those names from that name
    /// alone. So a peer nobody reached is neither a reason to run the recipe
    /// when it is missing nor something the intermediate sweep may take: it is
    /// written when the recipe runs and otherwise left alone.
    ///
    /// Nothing in a manifest says this, so the writer ignores it as it ignores
    /// [`Self::intermediate`] and [`Self::disposable`], and a sink that runs the
    /// build is the one that answers for it.
    pub peer_outputs: &'a [Symbol],
    /// Where the directory search found this edge's output, when the output is
    /// not where the Makefile named it.
    ///
    /// Both names, because which of them survives is the build's to decide and
    /// not the compiler's. GNU Make searches for a target it cannot find here,
    /// hangs the answer off the file object as `hname` beside the written
    /// `name`, and lets `update_file_1` choose once the prerequisites have
    /// settled: a target it does not have to remake takes the found name and
    /// every dependent reads that path, and one it does have to remake throws
    /// the found name away and is made here under its own. The choice cannot be
    /// folded in here — a target current when the Makefile was read is made
    /// stale by a prerequisite's own recipe — so the compiler says both and the
    /// destination settles it.
    ///
    /// What that asks of a destination is two things. The output's freshness is
    /// read from this path while the output itself is absent, which is GNU
    /// Make's `f_mtime` taking the found file's date for the target. And an
    /// edge that reads this output spells it this way for as long as the output
    /// has not been remade.
    ///
    /// Nothing in a manifest says either, so the writer ignores it as it
    /// ignores [`Self::intermediate`] and [`Self::peer_outputs`], and a sink
    /// that runs the build is the one that answers for it. `None` for every
    /// edge whose output is where it was written, which is nearly all of them,
    /// and for one `GPATH` settled — there the name was already replaced.
    pub searched_at: Option<Symbol>,
    /// The pool that limits how many edges like this run at once, unescaped.
    pub pool: Option<&'a [u8]>,
    /// Opaque per-edge metadata from `.KATI_TAGS`, for consumers of the graph
    /// rather than for the build itself.
    pub tags: Option<&'a [u8]>,
    /// The line of the Makefile this came from.
    pub loc: Option<&'a Loc>,
}

/// A destination for the build graph kati derives from a Makefile.
///
/// Calls arrive in dependency-walk order: [`Self::start`] once, then a
/// [`Self::declare_rule`] immediately before each [`Self::declare_edge`] that
/// uses it, then [`Self::set_default_targets`], then [`Self::finish`].
///
/// Every method that carries a [`Symbol`] is handed the interner it was minted
/// from. Passing the handle rather than the bytes keeps the identity kati
/// already established, so a sink that has to map kati's names onto its own can
/// key that map on a `usize` instead of re-hashing every path on every edge
/// that mentions it.
pub trait BuildSink {
    /// When this sink can compute `$?` after prerequisites have settled.
    ///
    /// A manifest writer cannot express that scheduling boundary and keeps
    /// kati's legacy shell fallback. Ronin's direct graph sink can, so command
    /// evaluation leaves a marker and describes the late comparison on the
    /// edge instead.
    fn new_inputs_timing(&self) -> NewInputsTiming {
        NewInputsTiming::RecipeShell
    }

    /// Who answers a `$(shell)` written inside a recipe.
    ///
    /// A manifest writer keeps kati's deferral, because the value would
    /// otherwise be frozen into a file that outlives the run that wrote it.
    /// Ronin's direct graph sink runs the build itself and so answers where
    /// GNU Make does.
    fn shell_evaluation(&self) -> ShellEvaluation {
        ShellEvaluation::RecipeShell
    }

    /// Who performs a `$(file ...)` written inside a recipe.
    ///
    /// A manifest writer refuses it, because the operation would be performed
    /// while the manifest is written rather than while the build runs: a write
    /// would land on the wrong day, and a read would answer from a tree the
    /// build has not made yet. Ronin's direct graph sink runs the build
    /// itself, so it performs it where GNU Make performs it.
    fn file_evaluation(&self) -> FileEvaluation {
        FileEvaluation::Refused
    }

    /// Who performs an `$(info ...)`, `$(warning ...)` or `$(error ...)`
    /// written inside a recipe.
    ///
    /// A manifest writer defers all three into the recipe, because the
    /// expansion that would print happens while the manifest is written and
    /// the text has to reach whoever runs it later. Ronin's direct graph sink
    /// expands the recipe in the process that runs it, so it prints where GNU
    /// Make prints and leaves no command behind.
    fn output_evaluation(&self) -> OutputEvaluation {
        OutputEvaluation::RecipeCommand
    }

    /// When this sink wants a recipe turned into command text.
    ///
    /// A manifest writer needs all of it before any of it runs, because the
    /// file it writes is the whole of what it produces. Ronin's direct graph
    /// sink runs the build itself, so it can hold a recipe unexpanded and
    /// expand it where GNU Make does: immediately before the command runs, and
    /// not at all for a target that turns out to be up to date.
    fn recipe_expansion(&self) -> RecipeExpansion {
        RecipeExpansion::Construction
    }

    /// Called once, before anything else, with the pools kati declares.
    fn start(&mut self, pools: &[SinkPool<'_>]) -> Result<()>;

    /// Declare a rule. Always immediately followed by the one edge that runs
    /// it.
    fn declare_rule(&mut self, names: &dyn Interner, rule: &SinkRule<'_>) -> Result<()>;

    /// Declare an edge.
    fn declare_edge(&mut self, names: &dyn Interner, edge: &SinkEdge<'_>) -> Result<()>;

    /// The targets to build when the command line names none. Called once,
    /// after every edge.
    fn set_default_targets(&mut self, names: &dyn Interner, targets: &[Symbol]) -> Result<()>;

    /// Called once, after everything else.
    fn finish(&mut self) -> Result<()>;
}
