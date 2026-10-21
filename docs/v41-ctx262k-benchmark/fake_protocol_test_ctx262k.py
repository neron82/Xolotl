#!/usr/bin/env python3
"""fake_protocol_test_ctx262k.py - no-model validation of the ctx=262144 rerun path.

Mirrors the baseline docs/v41-local-benchmark/fake_protocol_test.py ladder, but:
  * drives the harness with `--outdir <this dir> --ctx 262144` so it proves THE ACTUAL COMMAND SHAPE
    the ctx262k runner will use;
  * builds the fake repo's launcher with a relaxed pre-load MemAvailable default, so the ladder runs
    even while the live DS41 engine holds host RAM (no service is stopped, no real GPU/model is used).

It asserts the real harness produces, for each arm: 2 warm-up rows, 18 measured rows over 6 cells x 3
reps, reuse <= the template allowance (2), exact DONE counts, scope 110G/swap0/floor4, correct DSpark
intent per arm, and a clean summarize.

Usage:  <shared-venv-gpu>/bin/python fake_protocol_test_ctx262k.py
Exit 0 = all assertions passed. No GPU, no model, no service change.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path("/home/neron/projects/xolotl/docs/v41-local-benchmark")
HARNESS = BASE / "benchmark_server_ab.py"
SCRATCH = Path(os.environ.get("FAKE_TEST_DIR", "/home/neron/.hermes/cache/scratch/fake_ctx262k"))
CTX = "262144"

FAKE_ENGINE = r'''#!/usr/bin/env python3
import os, sys
dspark = os.environ.get("STRATA_SERVE_DSPARK")
if "STRATA_ROUTE_BIAS" in os.environ:
    raise SystemExit("harness leaked an inherited STRATA_ROUTE_BIAS override")
ctx = "262144"
sys.stdout.write(f"INFO ctx={ctx} experts_vram=10 arena_experts=20\n")
sys.stdout.write(f"READY {ctx} stop\n")
sys.stdout.flush()
sys.stderr.write(f"serve: ready (ctx {ctx}, prefill chunk 512)\n")
sys.stderr.flush()
prev = []
dspark_logged = False
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    if not line.startswith("GEN "):
        continue
    parts = line.split()
    max_new = int(parts[1])
    ids = [int(x) for x in parts[-1].split(",") if x]
    if dspark and not dspark_logged:
        dspark_logged = True
        if not os.environ.get("FAKE_BAD_DSPARK"):
            sys.stderr.write(f"DSpark: {dspark}, 4.86 GB of weights on the card (3.6 s)\n")
            sys.stderr.flush()
    lcp = 0
    while lcp < len(prev) and lcp < len(ids) and prev[lcp] == ids[lcp]:
        lcp += 1
    if lcp < len(ids):
        lcp = min(lcp, 2)
    reused = lcp
    new = len(ids) - reused
    sys.stderr.write(f"serve: prompt {len(ids)} tokens ({reused} reused, {new} new) in 1.00 s = "
                     f"{len(ids)/1.0:.1f} tok/s (experts 1 hit / 2 cpu / 3 pcie / 4 file)\n")
    sys.stderr.write(f"serve: prompt phases: {new} new = {new} feed (1.0 s: dense 0.5, experts 0.4, logits 0.1) "
                     f"+ DSpark tail 128\n")
    if not dspark:
        sys.stderr.write("serve: VRAM 20.00 GiB free -> 30 experts in VRAM (seed 1.0 s)\n")
    sys.stderr.write("serve: reuse lcp=%d park=0 lp=0 hist=%d snap=0 prompt=%d -> reused=%d\n"
                     % (lcp, len(prev), len(ids), reused))
    for i in range(max_new):
        sys.stdout.write("T %d\n" % (1000 + i))
    sys.stdout.flush()
    dec_ms = 100.0
    sys.stderr.write(f"serve: {max_new} tokens in {dec_ms/1000:.2f} s = {max_new/(dec_ms/1000):.2f} tok/s (length); "
                     f"experts: 5 hit, 6 cpu, 7 pcie, 8 file\n")
    if dspark:
        sys.stderr.write("serve: DSpark 3 passes, 2.00 tokens/pass (all requests 2.00)\n")
        passes, hits, lookups, file_tier = 3, 5, 6, 8
    else:
        passes, hits, lookups, file_tier = 0, 5, 6, 8
    sys.stdout.write("DONE %d %d %.1f %.1f %s 0 0 %d %d %d 0 %d 0 %d\n"
                     % (max_new, len(ids), 1000.0, dec_ms, "length", reused, hits, lookups, file_tier, new))
    sys.stdout.flush()
    prev = ids
'''


def build_fake_repo(root: Path) -> Path:
    if root.is_dir():
        shutil.rmtree(root)
    (root / "docs/v41-local-benchmark").mkdir(parents=True)
    (root / "build-xolotl").mkdir()
    launcher_src = (BASE / "run_engine_fg.sh").read_text()
    # Relax ONLY the copied fake launcher so the ladder runs while the live DS41 service is still resident
    # (it holds both host RAM and the shared GPU lock). The harness pins BENCH_MIN_MEM_AVAIL_GIB and DS4_LOCK
    # in engine_env(), so the relaxation must live inside the fake launcher, not in the test's env.
    launcher_src = launcher_src.replace(
        'BENCH_MIN_MEM_AVAIL_GIB="${BENCH_MIN_MEM_AVAIL_GIB:-70}"',
        'BENCH_MIN_MEM_AVAIL_GIB="${BENCH_MIN_MEM_AVAIL_GIB:-70}"\n'
        '[ "${FAKE_RELAX_RAM:-0}" = 1 ] && BENCH_MIN_MEM_AVAIL_GIB=0')
    launcher_src = launcher_src.replace(
        'LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"',
        'LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"\n'
        '[ "${FAKE_RELAX_LOCK:-0}" = 1 ] && LOCK="${FAKE_LOCK_FILE:-/tmp/fake_ctx262k.lock}"')
    (root / "docs/v41-local-benchmark/run_engine_fg.sh").write_text(launcher_src)
    eng = root / "build-xolotl/ds41_generate"
    eng.write_text(FAKE_ENGINE)
    eng.chmod(eng.stat().st_mode | stat.S_IEXEC)
    for rel in ("m.gguf", "rank.bin", "V41-MTP-Vision.Q2_K.gguf"):
        (root / rel).write_bytes(b"fake")
    for d in ("engram1", "engram2"):
        p = root / d
        p.mkdir(exist_ok=True)
        (p / "model.safetensors.index.json").write_text("{}")
    return root


def main() -> int:
    repo = build_fake_repo(SCRATCH / "repo")
    out = SCRATCH / "out"
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # Use the REAL ctx262k manifest (byte-identical to the baseline prompts_ab.json). Its ids_file paths are
    # absolute and point at the untouched baseline fixtures, so this proves the production command reuses the
    # exact baseline int32 files. No `prepare` here: the ctx262k run must NOT regenerate fixtures.
    shutil.copy(HERE / "prompts_ab.json", out / "prompts_ab.json")
    print("== staged baseline manifest into scratch outdir (no prepare) ==", flush=True)

    print(f"== run --arm both --ctx {CTX} (fake engine, baseline fixtures) ==", flush=True)
    env = dict(os.environ)
    env.update({
        "MEMGUARD_ALLOW_COMFY": "1",
        "FAKE_RELAX_RAM": "1",                 # relax the FAKE launcher's own floor only
        "FAKE_RELAX_LOCK": "1",                # use a scratch lock; the live service holds the real one
        "FAKE_LOCK_FILE": str(SCRATCH / "fake.lock"),
        "BENCH_WATCHDOG_FLOOR_GIB": "1",
        "STRATA_BUILD_DIR": str(repo / "build-xolotl"),
        "STRATA_ROUTE_BIAS": "999",            # harness must strip inherited tuning switches
    })
    cmd = [py, str(HARNESS), "--outdir", str(out), "--repo", str(repo), "--ctx", CTX,
           "--model", str(repo / "m.gguf"), "--rank", str(repo / "rank.bin"),
           "--engram", f"{repo/'engram1'}:{repo/'engram2'}",
           "--sidecar", str(repo / "V41-MTP-Vision.Q2_K.gguf"),
           "--min-mem-available-gib", "0", "--min-free-vram-mib", "0",
           "--physical-gpu", "0", "run", "--arm", "both"]
    print("   CMD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout[-3000:])
    sys.stderr.write(r.stderr[-3000:])
    if r.returncode != 0:
        print("RUN FAILED rc=", r.returncode, flush=True)
        return 1

    print("== summarize ==", flush=True)
    subprocess.run([py, str(HARNESS), "--outdir", str(out), "--repo", str(repo), "summarize"],
                   check=True, stdout=subprocess.DEVNULL)

    results = json.loads((out / "results_ab.json").read_text())
    summary = json.loads((out / "summary_ab.json").read_text())
    manifest = json.loads((out / "prompts_ab.json").read_text())
    allowance = manifest["fixture_policy"]["template_prefix_reuse_tokens"]

    assert results["config"]["ctx"] == 262144, results["config"]["ctx"]
    assert results["config"]["reps"] == 3
    assert results["config"]["warmup_per_arm"] == 2
    for arm in ("mtp", "nomtp"):
        a = results["arms"][arm]
        assert len(a["results"]) == 18, (arm, len(a["results"]))
        assert len(a["warmup"]) == 2, (arm, len(a["warmup"]))
        assert a["engine"]["ready_line"] == f"READY {CTX} stop", a["engine"]["ready_line"]
        assert a["engine"]["scope"] == {"unit": a["engine"]["scope"]["unit"], "max_gib": 110,
                                        "swap_gib": 0, "watchdog_floor_gib": 4}, a["engine"]["scope"]
        assert a["engine"]["dspark_state"]["verified"] is True
        want = "on" if arm == "mtp" else "off"
        assert a["engine"]["dspark_state"]["intent"] == want, a["engine"]["dspark_state"]
        for row in a["results"]:
            assert row["accepted"], (arm, row["case"], row["problems"])
            assert row["engine_prompt_reused"] == allowance, (arm, row["case"], row["engine_prompt_reused"])
            assert row["done_reused"] == allowance
            assert row["done_field_count"] == 15
            assert row["done_total_prompt"] == row["prepared_prompt_tokens"]
            assert row["done_generated"] == row["requested_max_new_tokens"]
        assert a["warmup"][1]["engine_prompt_reused"] > 0

    assert summary["reps"] == 3
    assert summary["arms"]["mtp"]["cells"].keys() == summary["arms"]["nomtp"]["cells"].keys()
    for arm in ("mtp", "nomtp"):
        cells = summary["arms"][arm]["cells"]
        assert len(cells) == 6, (arm, list(cells))
        for cell, c in cells.items():
            assert c["runs_accepted"] == 3, (arm, cell, c["runs_accepted"])
    print("ALL FAKE-PROTOCOL (ctx=262144) ASSERTIONS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
