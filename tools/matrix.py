#!/usr/bin/env python3
"""Second differential sweep: the emitter paths the kati corpus never reaches.

The testcase corpus exercises --default_pool, .KATI_NINJA_POOL,
.KATI_IMPLICIT_OUTPUTS, --empty_ninja_file and --emit_sandbox_disabled, and
nothing else the Ninja emitter branches on. This sweeps handwritten Makefiles
against every remaining flag: --use_ninja_phony_output, --no_ninja_prelude,
--ninja_dir, --remote_num_jobs, -j, -d, --gen_all_targets,
--detect_android_echo, --detect_depfiles, --use_ninja_validations, plus
.KATI_DEPFILE, .KATI_RESTAT, .KATI_TAGS, escaping of $ : and space in target
names, and a command over the 100 kB response-file threshold.

Same construction as sinkcmp.py: identical binary path and identical run
directory across both passes.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys

W = os.environ.get("KATI_DIFF_WORK") or os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "kati-diff")
os.makedirs(W, exist_ok=True)
BIN = os.path.join(W, "matrixbin", "rkati")
RUN = os.path.join(W, "matrixrun")

BIG = "x" * 200  # 200 * 600 lines > 100 kB of command text

MAKEFILES = {
    # Everything the emitter branches on that has a per-target spelling.
    "features": r"""
.PHONY: phony_target other_phony
.KATI_RESTAT: restat_target
all: real phony_target implicit_owner restat_target pooled tagged depfiled
	echo all

phony_target: phony_dep
	echo phony

other_phony:

phony_dep:
	echo dep

real: src.txt | order_only.txt
	echo real > real

implicit_owner: .KATI_IMPLICIT_OUTPUTS := extra1 extra2
implicit_owner:
	touch implicit_owner extra1 extra2

restat_target: .KATI_RESTAT := 1
restat_target:
	touch restat_target

pooled: .KATI_NINJA_POOL := mypool
pooled:
	echo pooled

tagged: .KATI_TAGS := tag1 tag2
tagged:
	echo tagged

depfiled: .KATI_DEPFILE := depfiled.d
depfiled:
	echo depfiled

nocommands: real

src.txt:
	touch src.txt

order_only.txt:
	touch order_only.txt
""",
    # Names that need ninja escaping. A colon cannot reach a target name
    # through Make's rule syntax, so it arrives via .KATI_IMPLICIT_OUTPUTS.
    "escaping": r"""
all: a$$b sub/deep$$x weird
	echo all

a$$b:
	echo dollar

sub/deep$$x:
	echo deep

weird: .KATI_IMPLICIT_OUTPUTS := has:colon has$$dollar
weird:
	touch weird
""",
    # Multi-command, ignore-error, silent, subshell, mkdir and echo folding,
    # and a gcc-style command the depfile detector should find.
    "commands": r"""
all: multi ignore silent mkdiry compiled
	echo all

multi:
	echo one
	echo two
	echo three

ignore:
	-false
	echo after

silent:
	@echo quiet

out/sub/mkdiry: mkdiry
mkdiry:
	mkdir -p .
	echo made

compiled:
	echo "host C++: thing" && g++ -c -MD -o compiled.o compiled.cc
""",
    # A command comfortably past the 100 kB response-file threshold.
    "rspfile": "\n".join(
        ["all: big", "\techo all", "", "big:"]
        + ["\techo %s" % BIG for _ in range(600)]
    )
    + "\n",
    # .KATI_VALIDATIONS, which needs --use_ninja_validations.
    "validations": r"""
all: .KATI_VALIDATIONS := checker
all:
	echo all

checker:
	echo checking
""",
}

FLAGSETS = [
    [],
    ["--use_ninja_phony_output"],
    ["--no_ninja_prelude"],
    ["--ninja_dir=out"],
    ["--remote_num_jobs=8"],
    ["--remote_num_jobs=8", "--default_pool=dp"],
    ["--default_pool=dp"],
    ["-j3"],
    ["-d"],
    ["-d", "--use_ninja_phony_output", "--remote_num_jobs=4"],
    ["--gen_all_targets"],
    ["--detect_android_echo"],
    ["--detect_depfiles"],
    ["--detect_android_echo", "--detect_depfiles"],
    ["--use_ninja_validations"],
    ["--emit_sandbox_disabled"],
    ["--empty_ninja_file"],
    ["--ninja_suffix=_sfx"],
    ["--ninja_dir=out", "--no_ninja_prelude", "--remote_num_jobs=2"],
]


def cases():
    for mk in sorted(MAKEFILES):
        for i, fs in enumerate(FLAGSETS):
            yield (f"{mk}#{i}", mk, fs)


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.#-]", "_", s)


def run_case(case_id, mk, flagset, snapdir):
    d = os.path.join(RUN, slug(case_id))
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "out"))
    open(os.path.join(d, "Makefile"), "w").write(MAKEFILES[mk])
    args = [BIN, "--ninja"] + flagset
    if mk == "validations" and "--use_ninja_validations" not in flagset:
        args.append("--use_ninja_validations")
    args.append("SHELL=/bin/bash")
    e = dict(os.environ)
    e.pop("MAKEFLAGS", None)
    e.pop("MAKELEVEL", None)
    e["LC_ALL"] = "C"
    try:
        p = subprocess.run(args, cwd=d, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=90)
        rc, out, err = str(p.returncode), p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        rc, out, err = "TIMEOUT", ex.stdout or b"", ex.stderr or b""

    s = os.path.join(snapdir, slug(case_id))
    os.makedirs(s)
    open(os.path.join(s, "@exit"), "w").write(rc)
    open(os.path.join(s, "@stdout"), "wb").write(out)
    open(os.path.join(s, "@stderr"), "wb").write(err)
    listing = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, d)
            data = open(full, "rb").read()
            listing.append(f"{rel}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
            dst = os.path.join(s, "files", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
    open(os.path.join(s, "@files"), "w").write("\n".join(sorted(listing)))
    shutil.rmtree(d, ignore_errors=True)


def do_pass(which, src_binary):
    snapdir = os.path.join(W, "matrixsnap", which)
    shutil.rmtree(snapdir, ignore_errors=True)
    os.makedirs(snapdir)
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    shutil.copyfile(src_binary, BIN)
    os.chmod(BIN, 0o755)
    os.utime(BIN, (1700000000, 1700000000))
    n = 0
    for case_id, mk, fs in cases():
        run_case(case_id, mk, fs, snapdir)
        n += 1
    print(f"pass {which}: {n} runs", file=sys.stderr)


def volatile(rel):
    b = os.path.basename(rel)
    return b.startswith(".kati_stamp") or b.startswith("kati.")


def compare():
    a = os.path.join(W, "matrixsnap", "base")
    b = os.path.join(W, "matrixsnap", "new")
    ids = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    diffs = []
    ninja_seen = 0
    for i in ids:
        da, db = os.path.join(a, i), os.path.join(b, i)
        for key in ("@exit", "@stdout", "@stderr", "@files"):
            xa = open(os.path.join(da, key), "rb").read()
            xb = open(os.path.join(db, key), "rb").read()
            if key == "@files":
                def norm(x):
                    return "\n".join(
                        (ln.split("\t")[0] if volatile(ln.split("\t")[0]) else ln)
                        for ln in x.decode(errors="replace").split("\n")
                    )
                xa, xb = norm(xa), norm(xb)
            if xa != xb:
                diffs.append((i, f"{key} differs"))
        fa, fb = os.path.join(da, "files"), os.path.join(db, "files")
        rels = set()
        for root in (fa, fb):
            for r, _, fs in os.walk(root):
                for f in fs:
                    rels.add(os.path.relpath(os.path.join(r, f), root))
        for rel in sorted(rels):
            if volatile(rel):
                continue
            pa, pb = os.path.join(fa, rel), os.path.join(fb, rel)
            xa = open(pa, "rb").read() if os.path.exists(pa) else None
            xb = open(pb, "rb").read() if os.path.exists(pb) else None
            if os.path.basename(rel).startswith("build") and rel.endswith(".ninja"):
                ninja_seen += 1
            if xa != xb:
                diffs.append((i, f"file {rel} differs"))
    print(f"runs compared:        {len(ids)}")
    print(f"build.ninja compared: {ninja_seen}")
    print(f"differences:          {len(diffs)}")
    for i, why in diffs:
        print(f"  {i}: {why}")
    return 1 if diffs else 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "count":
        print(len(list(cases())))
    elif cmd == "pass":
        if sys.argv[2] not in ("base", "new"):
            sys.exit("pass takes 'base' or 'new': compare reads exactly those "
                     "two snapshots, so any other label is written and then "
                     "silently ignored while a stale snapshot is compared.")
        do_pass(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        sys.exit(compare())
