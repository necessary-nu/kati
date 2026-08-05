# Emitter differential harnesses

Two harnesses that answer one question: **did this change alter the bytes we
emit?** Both compare two `rkati` binaries rather than comparing output against
a stored expectation, so they stay useful as the emitter is refactored and they
never need updating when the emitter legitimately changes.

Every node that restructures Ninja emission without meaning to change it —
`kati-build-sink` and `kati-command-escapes` so far — is gated on these coming
back clean. Reuse them rather than writing a third.

## Why two

`sinkcmp.py` runs the whole `testcase/` corpus: 455 runs, from 296 `.mk` files
expanded to each `testN` target they declare, plus 50 `.sh` files run twice
(`--ninja` and `--ninja --regen`). Broad, but it only reaches the emitter
branches the corpus happens to exercise — in practice `--default_pool`,
`.KATI_NINJA_POOL`, `.KATI_IMPLICIT_OUTPUTS`, `--empty_ninja_file` and
`--emit_sandbox_disabled`, and nothing else.

`matrix.py` covers the rest deliberately: 5 handwritten Makefiles across 19
flag sets, 95 runs, reaching `rspfile`/`rspfile_content`, `restat`, `tags`,
`depfile`, `deps = gcc`, `phony_output`, `_kati_always_build_`, all three pool
sources, `sandbox_disabled`, `builddir`, `-d` locations, `|@`, `||`, `$`- and
`:`-escaped names, `: phony`, and `-j`-derived pool depth.

Neither is redundant. The corpus finds what you did not think of; the matrix
covers what the corpus cannot reach.

## Running them

```sh
git worktree add -f --detach /tmp/kati-base <baseline-commit>
( cd /tmp/kati-base && cargo build --release --bin rkati )
cargo build --release -p kati --bin rkati        # from the ronin workspace root

python3 tools/sinkcmp.py pass a /tmp/kati-base/target/release/rkati
python3 tools/sinkcmp.py pass b ../target/release/rkati
python3 tools/sinkcmp.py compare                 # exit 1 if anything differs
```

`matrix.py` takes the same three subcommands. Scratch state goes to
`$KATI_DIFF_WORK`, or `$TMPDIR/kati-diff` by default — never into the source
tree.

## Two things that will mislead you

**Validate the harness before trusting a clean result.** Run the *baseline*
binary as both passes. Anything that differs there is noise, not a regression,
and you want that list before you start attributing differences to your change.

**Three sources of noise are known and pre-existing.** `.ninja_log` carries
timestamps. `env.sh` line order varies between two runs of the same binary,
because kati writes it by iterating a `HashMap` — it is compared as a sorted
set. And `kati_cache.sh#regen` has a sub-second mtime race inside the testcase
itself, so it differs against a self-comparison; at the time of writing it is
the only case that does.

Both harnesses already remove the two obvious artefacts by construction rather
than by scrubbing: each pass copies its binary to the *same* absolute path and
runs each case in the *same* absolute directory, so neither `current_exe()`
(which kati embeds in the stamp) nor `$(realpath ../foo)` can differ between
passes for reasons that are about the harness.
