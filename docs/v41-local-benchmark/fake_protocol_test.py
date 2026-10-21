#!/usr/bin/env python3
"""No-model fake-protocol test for benchmark_server_ab.py.

Builds a throwaway fake repo (local launcher + fake ds41_generate) and dummy model artifacts, then drives the real
harness end-to-end: run --arm both, then summarize.  No model is loaded, no GPU is used, no real engine runs.
The fake engine speaks the documented line protocol and logs the phase lines the parser expects.

Usage:  PYTHONPATH=<benchmark dir> python3 fake_protocol_test.py
Exit 0 = all assertions passed.
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
SCRATCH = Path(os.environ.get("FAKE_TEST_DIR", str(Path.home() / ".hermes/cache/scratch/fake_protocol")))
FAKE_ENGINE = r'''#!/usr/bin/env python3
import os, sys, math
dspark = os.environ.get("STRATA_SERVE_DSPARK")
park = os.environ.get("STRATA_SERVE_PARK")
if "STRATA_ROUTE_BIAS" in os.environ:
    raise SystemExit("harness leaked an inherited STRATA_ROUTE_BIAS override")
sys.stdout.write("INFO ctx=8192 experts_vram=10 arena_experts=20\n")
sys.stdout.write("READY 8192 stop\n")
sys.stdout.flush()
sys.stderr.write("serve: ready (ctx 8192, prefill chunk 512)\n")
sys.stderr.flush()
# state for a tiny reuse model: remember the previous ids
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
    # the real chat template gives every distinct fixture a fixed short prefix; a byte-identical repeat reuses
    # the whole prompt.  Mirror that so the guard is exercised realistically.
    if lcp < len(ids):
        lcp = min(lcp, 2)
    reused = lcp
    new = len(ids) - reused
    sys.stderr.write(f"serve: prompt {len(ids)} tokens ({reused} reused, {new} new) in 1.00 s = "
                     f"{len(ids)/1.0:.1f} tok/s (experts 1 hit / 2 cpu / 3 pcie / 4 file)\n")
    sys.stderr.write(f"serve: prompt phases: {new} new = {new} feed (1.0 s: dense 0.5, experts 0.4, logits 0.1) "
                     f"+ DSpark tail 128\n")
    # MTP seeds during its prompt tail and may not emit a serve-level seed line; the control arm does.
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
    # DONE gen total prompt_ms decode_ms finish 0 0 reused hits lookups 0 file_tier 0 new
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
    shutil.copy(HERE / "run_engine_fg.sh", root / "docs/v41-local-benchmark/run_engine_fg.sh")
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
    out.mkdir(parents=True, exist_ok=True)
    harness = HERE / "benchmark_server_ab.py"
    py = sys.executable
    # prepare (CPU/tokenizer only) - reuse the real manifest generator
    print("== prepare ==", flush=True)
    subprocess.run([py, str(harness), "--outdir", str(out), "prepare"], check=True,
                   stdout=subprocess.DEVNULL)
    manifest = json.loads((out / "prompts_ab.json").read_text())
    assert len(manifest["requests"]) == 18, len(manifest["requests"])
    assert len({r["ids_sha256"] for r in manifest["requests"]}) == 18
    assert manifest["warmup_requests"][0]["ids_sha256"] == manifest["warmup_requests"][1]["ids_sha256"]

    print("== run --arm both (fake engine) ==", flush=True)
    env = dict(os.environ)
    env.update({
        "MEMGUARD_ALLOW_COMFY": "1",
        "BENCH_MIN_MEM_AVAIL_GIB": "0",       # we are not touching RAM in this test
        "BENCH_WATCHDOG_FLOOR_GIB": "1",
        "STRATA_BUILD_DIR": str(repo / "build-xolotl"),
        "STRATA_ROUTE_BIAS": "999",              # harness must strip inherited tuning switches
    })
    # fake artifacts + repo layouts the harness preflight needs
    cmd = [py, str(harness), "--outdir", str(out), "--repo", str(repo),
           "--model", str(repo / "m.gguf"), "--rank", str(repo / "rank.bin"),
           "--engram", f"{repo/'engram1'}:{repo/'engram2'}",
           "--sidecar", str(repo / "V41-MTP-Vision.Q2_K.gguf"),
           "--min-mem-available-gib", "0", "--min-free-vram-mib", "0",
           "--physical-gpu", "0", "run", "--arm", "both"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    sys.stderr.write(r.stderr[-4000:])
    if r.returncode != 0:
        print("RUN FAILED rc=", r.returncode, flush=True)
        return 1

    print("== summarize ==", flush=True)
    subprocess.run([py, str(harness), "--outdir", str(out), "summarize"], check=True,
                   stdout=subprocess.DEVNULL)

    results = json.loads((out / "results_ab.json").read_text())
    summary = json.loads((out / "summary_ab.json").read_text())

    # --- assertions ---
    allowance = manifest["fixture_policy"]["template_prefix_reuse_tokens"]
    for arm in ("mtp", "nomtp"):
        a = results["arms"][arm]
        assert len(a["results"]) == 18, (arm, len(a["results"]))
        assert len(a["warmup"]) == 2
        assert a["engine"]["scope"]["max_gib"] == 110
        assert a["engine"]["scope"]["watchdog_floor_gib"] == 4   # harness pins its LAUNCHER_MEMGUARD floor
        assert a["engine"]["dspark_state"]["verified"] is True
        want = "on" if arm == "mtp" else "off"
        assert a["engine"]["dspark_state"]["intent"] == want, a["engine"]["dspark_state"]
        for r in a["results"]:
            assert r["accepted"], (arm, r["case"], r["problems"])
            assert r["engine_prompt_reused"] == allowance, (arm, r["case"], r["engine_prompt_reused"])
            assert r["done_reused"] == allowance
            assert r["done_total_prompt"] == r["prepared_prompt_tokens"]
            assert r["done_generated"] == r["requested_max_new_tokens"]
            assert r["done_new_prompt"] == r["prepared_prompt_tokens"] - allowance
        # warm-up #2 exercises the reuse path (fake engine reuses the shared warm-up fixture)
        assert a["warmup"][1]["engine_prompt_reused"] > 0, a["warmup"][1].get("engine_prompt_reused")
        if arm == "mtp":
            assert "seed_free_gib" not in a["warmup"][0]  # valid: this arm seeds inside its prompt tail
        else:
            assert a["warmup"][0]["seed_free_gib"] == 20.0  # parser still consumes the optional control-arm line
    # rotation recorded and identical for both arms
    assert summary["reps"] == 3
    for arm in ("mtp", "nomtp"):
        cells = summary["arms"][arm]["cells"]
        assert len(cells) == 6, list(cells)
        for cell, c in cells.items():
            assert c["runs_accepted"] == 3, (arm, cell, c["runs_accepted"])
            m = c["metrics"]["decode_tok_s"]
            assert m["n"] == 3 and "sd" in m and "min" in m and "max" in m and "spread_abs" in m

    # A post-warmup-1 DSpark verification failure must QUIT the engine and release its transient scope.
    print("== startup cleanup after a deliberately bad DSpark state ==", flush=True)
    bad_out = SCRATCH / "cleanup_out"
    if bad_out.exists():
        shutil.rmtree(bad_out)
    bad_out.mkdir(parents=True, exist_ok=True)
    subprocess.run([py, str(harness), "--outdir", str(bad_out), "prepare"], check=True,
                   stdout=subprocess.DEVNULL)
    bad_cmd = list(cmd)
    bad_cmd[bad_cmd.index("--outdir") + 1] = str(bad_out)
    bad_cmd[-1] = "mtp"
    bad_env = dict(env)
    bad_env["FAKE_BAD_DSPARK"] = "1"
    bad = subprocess.run(bad_cmd, env=bad_env, capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0, "bad DSpark state unexpectedly passed startup verification"
    assert "engine printed no DSpark sidecar load line after first warm-up request" in bad.stderr, bad.stderr[-2000:]
    logs = list((bad_out / "runs_ab").glob("*.mtp.server.stdout.log"))
    assert len(logs) == 1, logs
    scope = re.search(r"memguard: scope (\S+) MemoryMax=", logs[0].read_text())
    assert scope, logs[0].read_text()
    status = subprocess.run(["systemctl", "--user", "is-active", f"{scope.group(1)}.scope"],
                            capture_output=True, text=True, timeout=10)
    assert status.returncode != 0 or status.stdout.strip() != "active", status.stdout + status.stderr
    print("startup verification failure cleaned up engine/scope", flush=True)
    print("ALL FAKE-PROTOCOL ASSERTIONS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
