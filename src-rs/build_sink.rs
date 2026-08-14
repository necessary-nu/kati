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
    /// shell as a file instead: `<shell> <script file>`, with no flags.
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
    /// Unused by [`SinkCommand::ResponseFile`].
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
    /// Recursive `$(MAKE)` invocations extracted from the recipe in their
    /// written order. A graph sink compiles these as semantic subninjas rather
    /// than handing nested Make processes to its executor.
    // [spec:ronin:req:make.recursive-invocation+1]
    pub subninjas: &'a [SinkSubninja<'a>],
    /// At least one line in the recipe is a recursive `$(MAKE)` invocation.
    /// This can be true while [`Self::subninjas`] is empty when a multi-line
    /// `.ONESHELL` recipe, one shell line holding more than one MAKE
    /// reference, or a recursion GNU Make classified but that sits where no
    /// static child invocation can be lifted out, cannot be split safely. A
    /// sink refuses those rather than running them, because running them is a
    /// nested Make process.
    pub contains_recursive: bool,
    /// The non-recursive recipe lines, assembled after extracting subninjas.
    /// A graph sink runs this parent action after the child graphs.
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
    /// The outputs a failed recipe must not leave behind, which is what
    /// `.DELETE_ON_ERROR` asks for. Empty means keep whatever the recipe wrote.
    ///
    /// The names rather than a flag, because the exclusions are per output and
    /// not per edge: `.PRECIOUS` protects one member of a grouped record while
    /// its peers still go. Which of them the failed recipe actually touched is
    /// the running build's question and not this one's, so every eligible name
    /// is listed and the sink compares timestamps.
    ///
    /// Nothing in a manifest says this — Ninja has no notion of an output the
    /// build withdraws when the command fails — so the writer ignores it, as it
    /// ignores [`Self::intermediate`] and [`Self::disposable`], and a sink that
    /// runs the build is the one that answers for it.
    pub delete_on_error_outputs: &'a [Symbol],
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
