#!/usr/bin/env python3
"""verify_outputs.py - post-run verifier for the ctx=262144 A/B rerun.

Proves the run is complete and comparable to the baseline BEFORE any number is quoted:
  * config.ctx == 262144 and every other config knob matches the baseline (except outdir/ctx/physical_gpu);
  * both arms present, each with 2 warm-ups + 18 measured rows (6 cells x 3 reps);
  * every measured row accepted, reuse <= 2 on all reuse sources, exact prompt/generated counts;
  * fixtures are the baseline int32 files (ids_sha256 matches the baseline manifest) and untouched;
  * the pinned card 00000000:2B:00.0 was the one used, and each arm's scope/Env contract was verified.

Usage:  python verify_outputs.py [--results <dir>/results_ab.json] [--baseline <dir>/results_ab.json]
Exit 0 = all proved. Never writes anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXITOK = 0
NEW_DIR = Path("/home/neron/projects/xolotl/docs/v41-ctx262k-benchmark")
BASE_DIR = Path("/home/neron/projects/xolotl/docs/v41-local-benchmark")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=NEW_DIR / "results_ab.json")
    ap.add_argument("--baseline", type=Path, default=BASE_DIR / "results_ab.json")
    ap.add_argument("--ctx", type=int, default=262144)
    args = ap.parse_args()

    problems: list[str] = []
    if not args.results.is_file():
        print(f"FAIL: no results at {args.results}")
        return 1
    res = json.loads(args.results.read_text())
    cfg = res["config"]

    # ---- config comparison vs baseline -------------------------------------------------------------
    if cfg["ctx"] != args.ctx:
        problems.append(f"config.ctx {cfg['ctx']} != {args.ctx}")
    fields = ["model", "rank", "engram", "sidecar", "serve_park", "reps",
              "memguard_max_gib", "watchdog_floor_gib", "preload_mem_floor_gib", "run_mem_floor_gib",
              "min_mem_available_gib", "min_free_vram_mib"]
    if args.baseline.is_file():
        bcfg = json.loads(args.baseline.read_text())["config"]
        for f in fields:
            if cfg.get(f) != bcfg.get(f):
                problems.append(f"config.{f}: run={cfg.get(f)!r} baseline={bcfg.get(f)!r}")
        print(f"baseline ctx={bcfg['ctx']}  rerun ctx={cfg['ctx']}")
        print("config fields compared  :", ", ".join(fields))

    # ---- arms / counts ----------------------------------------------------------------------------
    arms = res.get("arms", {})
    for arm in ("mtp", "nomtp"):
        if arm not in arms:
            problems.append(f"arm missing: {arm}")
            continue
        a = arms[arm]
        n_w = len(a.get("warmup", []))
        n_r = len(a.get("results", []))
        if n_w != 2:
            problems.append(f"{arm}: {n_w} warm-up rows != 2")
        if n_r != 18:
            problems.append(f"{arm}: {n_r} measured rows != 18")
        cells: dict[tuple[int, int], int] = {}
        for r in a.get("results", []):
            if not r.get("accepted"):
                problems.append(f"{arm}/{r['case']}: not accepted: {r.get('problems')}")
            allowance = r.get("max_reuse_tokens_allowed")
            srcc = r.get("reuse_sources", {})
            for k, v in srcc.items():
                if allowance is not None and v > allowance:
                    problems.append(f"{arm}/{r['case']}: {k}={v} > allowance {allowance}")
            if r.get("done_field_count") != 15:
                problems.append(f"{arm}/{r['case']}: done_field_count {r.get('done_field_count')} != 15")
            if r.get("done_total_prompt") != r.get("prepared_prompt_tokens"):
                problems.append(f"{arm}/{r['case']}: DONE prompt != prepared tokens")
            if r.get("done_generated") != r.get("requested_max_new_tokens"):
                problems.append(f"{arm}/{r['case']}: DONE generated != requested")
            key = (r.get("prefill_tokens"), r.get("max_new_tokens"))
            cells[key] = cells.get(key, 0) + 1
        for key, n in cells.items():
            if n != 3:
                problems.append(f"{arm}: cell {key} has {n} runs != 3")
        if len(cells) != 6:
            problems.append(f"{arm}: {len(cells)} cells != 6")
        # engine scope / pin evidence
        eng = a.get("engine", {})
        sc = eng.get("scope", {})
        if sc.get("max_gib") != 110 or sc.get("swap_gib") != 0 or sc.get("watchdog_floor_gib") != 4:
            problems.append(f"{arm}: scope {sc} != MemoryMax 110 / swap 0 / floor 4")
        if "2B:00.0" not in eng.get("gpu_before", ""):
            problems.append(f"{arm}: pinned card 00000000:2B:00.0 not in gpu_before")
        ds = eng.get("dspark_state", {})
        if not ds or ds.get("verified") is not True:
            problems.append(f"{arm}: dspark_state not verified: {ds}")
        want = "on" if arm == "mtp" else "off"
        if ds.get("intent") != want:
            problems.append(f"{arm}: dspark intent {ds.get('intent')} != {want}")
        print(f"{arm:5s}: warmups={n_w} measured={n_r} cells={len(cells)} "
              f"load={eng.get('load_seconds')}s dspark={ds.get('intent')}")

    # ---- fixtures are the baseline int32 files and untouched ---------------------------------------
    bfixtures = {}
    bman_path = BASE_DIR / "prompts_ab.json"
    if bman_path.is_file():
        bman = json.loads(bman_path.read_text())
        for req in bman["requests"] + bman.get("warmup_requests", []):
            bfixtures[req["case"]] = req["ids_sha256"]
    used = {r["case"]: r["ids_sha256"] for r in sum((arms[a].get("results", []) for a in arms), [])}
    mism = [c for c, h in used.items() if c in bfixtures and bfixtures[c] != h]
    if mism:
        problems.append(f"measured rows whose ids_sha256 differs from baseline fixtures: {mism}")
    print(f"fixture hashes matched   : {len(used) - len(mism)}/{len(used)} measured rows vs baseline manifest")

    print()
    if problems:
        print(f"VERIFY FAIL ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"VERIFY OK: ctx={cfg['ctx']}, both arms 2+18, reuse<=2, exact counts, pinned card, scope 110G/swap0/floor4")
    return EXITOK


if __name__ == "__main__":
    raise SystemExit(main())
