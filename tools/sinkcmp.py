#!/usr/bin/env python3
"""Differential harness for the kati Ninja emitter.

Two passes over the whole testcase corpus. Each pass runs a different rkati
binary, but from the *same* absolute binary path and in the *same* absolute
working directory, so neither the binary path (kati embeds current_exe in the
stamp, and testcases can observe it) nor the run directory ($(realpath ../foo))
can differ between passes. That removes both known harness artefacts by
construction rather than by scrubbing.

Each pass snapshots exit code, stdout, stderr, the recursive file listing and
the bytes of every produced file. `compare` diffs the snapshots.
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
KATI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTCASE = os.path.join(KATI, "testcase")
BIN = os.path.join(W, "sinkbin", "rkati")
RUN = os.path.join(W, "sinkrun")
FAKEBIN = os.path.join(W, "sinkpath")

TESTCASE_RE = re.compile(rb"^test\d*")


def unique_testcases(content):
    seen = []
    for line in content.split(b"\n"):
        m = TESTCASE_RE.match(line)
        if m:
            s = m.group(0).decode()
            if s not in seen:
                seen.append(s)
    return sorted(seen) or [""]


def cases():
    for name in sorted(os.listdir(TESTCASE)):
        if name.startswith("."):
            continue
        path = os.path.join(TESTCASE, name)
        if name.endswith(".mk"):
            for tc in unique_testcases(open(path, "rb").read()):
                yield (f"{name}#{tc or 'default'}", "mk", name, tc)
        elif name.endswith(".sh"):
            yield (f"{name}#ninja", "sh", name, ["--ninja"])
            yield (f"{name}#regen", "sh", name, ["--ninja", "--regen"])


def env():
    e = dict(os.environ)
    e.pop("MAKEFLAGS", None)
    e.pop("MAKELEVEL", None)
    e["NINJA_STATUS"] = "NINJACMD: "
    e["PATH"] = FAKEBIN + ":" + e["PATH"]
    e["LC_ALL"] = "C"
    return e


def slug(case_id):
    return re.sub(r"[^A-Za-z0-9_.#-]", "_", case_id)


def run_case(case_id, kind, name, arg, snapdir):
    d = os.path.join(RUN, slug(case_id))
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)

    if kind == "mk":
        shutil.copyfile(os.path.join(TESTCASE, name), os.path.join(d, "Makefile"))
        try:
            os.symlink(os.path.join(TESTCASE, "submake"), os.path.join(d, "submake"))
        except OSError:
            pass
        args = [BIN, "--use_find_emulator", "--ninja"]
        if name.startswith("submake_"):
            args.append("-s")
        args.append("SHELL=/bin/bash")
        if arg:
            args.append(arg)
    else:
        args = ["bash", os.path.join(TESTCASE, name), BIN] + arg + ["SHELL=/bin/bash"]

    try:
        p = subprocess.run(args, cwd=d, env=env(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=90)
        rc, out, err = str(p.returncode), p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "TIMEOUT", e.stdout or b"", e.stderr or b""

    s = os.path.join(snapdir, slug(case_id))
    os.makedirs(s)
    open(os.path.join(s, "@exit"), "w").write(rc)
    open(os.path.join(s, "@stdout"), "wb").write(out)
    open(os.path.join(s, "@stderr"), "wb").write(err)

    listing = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(dirs):
            listing.append(os.path.relpath(os.path.join(root, f), d) + "\tDIR")
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, d)
            if os.path.islink(full):
                listing.append(rel + "\tSYMLINK")
                continue
            try:
                data = open(full, "rb").read()
            except OSError as e:
                listing.append(f"{rel}\tUNREADABLE {e.errno}")
                continue
            listing.append(f"{rel}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
            dst = os.path.join(s, "files", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
    open(os.path.join(s, "@files"), "w").write("\n".join(sorted(listing)))
    shutil.rmtree(d, ignore_errors=True)


def do_pass(which, src_binary):
    snapdir = os.path.join(W, "sinksnap", which)
    shutil.rmtree(snapdir, ignore_errors=True)
    os.makedirs(snapdir)
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    shutil.copyfile(src_binary, BIN)
    os.chmod(BIN, 0o755)
    os.utime(BIN, (1700000000, 1700000000))
    os.makedirs(FAKEBIN, exist_ok=True)
    ninja = os.path.join(FAKEBIN, "ninja")
    if not os.path.exists(ninja):
        # -j1 so that edge output order is deterministic across passes.
        open(ninja, "w").write(
            "#!/bin/sh\nexec /home/brendan/work/samurai/target/release/ronin -j1 \"$@\"\n"
        )
        os.chmod(ninja, 0o755)
    n = 0
    for case_id, kind, name, arg in cases():
        run_case(case_id, kind, name, arg, snapdir)
        n += 1
        if n % 100 == 0:
            print(f"  {n}...", file=sys.stderr, flush=True)
    print(f"pass {which}: {n} runs", file=sys.stderr)


def volatile(rel):
    """Content that legitimately differs run to run."""
    base = os.path.basename(rel)
    return (
        # The stamp embeds the wall-clock time of generation.
        base.startswith(".kati_stamp")
        or base.startswith("kati.")
        # ninja's build log embeds start/end times per edge.
        or base == ".ninja_log"
        or base == ".ninja_deps"
    )


def unordered(rel):
    """Content whose *order* is nondeterministic in kati today.

    env.sh is written by iterating `Evaluator::exports`, a HashMap, so its line
    order varies run to run in the same binary. That predates this work; the
    set of lines is what is being checked.
    """
    return os.path.basename(rel) == "env.sh"


def compare():
    a = os.path.join(W, "sinksnap", "base")
    b = os.path.join(W, "sinksnap", "new")
    ids = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    diffs = []
    ninja_seen = 0
    for i in ids:
        da, db = os.path.join(a, i), os.path.join(b, i)
        if not (os.path.isdir(da) and os.path.isdir(db)):
            diffs.append((i, "case present in only one pass"))
            continue
        for key in ("@exit", "@stdout", "@stderr", "@files"):
            xa = open(os.path.join(da, key), "rb").read()
            xb = open(os.path.join(db, key), "rb").read()
            if key == "@files":
                def norm(x):
                    out = []
                    for line in x.decode(errors="replace").split("\n"):
                        rel = line.split("\t")[0]
                        keep = not (volatile(rel) or unordered(rel))
                        out.append(line if keep else rel)
                    return "\n".join(out)
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
            if os.path.basename(rel) == "build.ninja" and xa is not None:
                ninja_seen += 1
            if unordered(rel):
                xa = b"\n".join(sorted(xa.split(b"\n"))) if xa is not None else None
                xb = b"\n".join(sorted(xb.split(b"\n"))) if xb is not None else None
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
        cs = list(cases())
        print(len(cs), "runs;", sum(1 for c in cs if c[1] == "mk"), "mk;",
              sum(1 for c in cs if c[1] == "sh"), "sh")
    elif cmd == "pass":
        if sys.argv[2] not in ("base", "new"):
            sys.exit("pass takes 'base' or 'new': compare reads exactly those "
                     "two snapshots, so any other label is written and then "
                     "silently ignored while a stale snapshot is compared.")
        do_pass(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        sys.exit(compare())
