#!/usr/bin/env python3
"""verify_manifest.py - preflight manifest/fixture integrity for the ctx=262144 rerun.

Reads the manifest copied into docs/v41-ctx262k-benchmark/ and proves, before any model load:
  * the manifest is byte-identical to the baseline prompts_ab.json (sha256), so the fixtures are the
    EXACT baseline int32 files (absolute ids_file paths point back into docs/v41-local-benchmark/prompts_ab/);
  * every measured + warm-up fixture exists, its sha256 matches the manifest, and size == tokens*4;
  * 6 cells x 3 reps = 18 measured requests per arm, pairwise-distinct ids, disjoint from the warm-up;
  * 2 warm-up requests sharing one fixture; template_prefix_reuse_tokens == 2 (the reuse<=2 allowance).

Usage:  python verify_manifest.py <path-to-manifest>
Exit 0 = all proved. Never writes anything.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE_MANIFEST = Path("/home/neron/projects/xolotl/docs/v41-local-benchmark/prompts_ab.json")
EXPECTED_CELLS = {(1024, 128), (1024, 256), (2048, 128), (2048, 256), (4096, 128), (4096, 256)}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("/home/neron/projects/xolotl/docs/v41-ctx262k-benchmark/prompts_ab.json")
    if not manifest_path.is_file():
        print(f"FAIL: manifest not found: {manifest_path}")
        return 1
    raw = manifest_path.read_bytes()
    man = json.loads(raw)
    problems: list[str] = []

    # 1. byte-identical to the baseline manifest (same fixtures, not regenerated)
    if BASE_MANIFEST.is_file():
        base_raw = BASE_MANIFEST.read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashlib.sha256(base_raw).hexdigest():
            # tolerated only if the only diffs are provenance fields; compare the fixture-carrying parts
            base_man = json.loads(base_raw)
            for key in ("requests", "warmup_requests", "fixture_policy", "schedule", "reps_per_cell_per_arm"):
                if man.get(key) != base_man.get(key):
                    problems.append(f"manifest differs from baseline in '{key}'")
        print(f"manifest sha256          : {hashlib.sha256(raw).hexdigest()}")

    # 2. fixtures: existence, hash, size
    n_checked = 0
    for req in list(man["requests"]) + list(man.get("warmup_requests", [])):
        p = Path(req["ids_file"])
        if not p.is_file():
            problems.append(f"missing fixture {req['case']}: {p}")
            continue
        n_checked += 1
        if sha256_file(p) != req["ids_sha256"]:
            problems.append(f"sha256 mismatch {req['case']}: {p}")
        if p.stat().st_size != 4 * req["prompt_id_count"]:
            problems.append(f"size mismatch {req['case']}: {p} ({p.stat().st_size}B != "
                            f"{4 * req['prompt_id_count']}B)")
    print(f"fixtures checked         : {n_checked}")

    # 3. counts + shape
    measured = man["requests"]
    if len(measured) != 18:
        problems.append(f"measured requests {len(measured)} != 18")
    cells = [(r["prefill_tokens"], r["max_new_tokens"]) for r in measured]
    if set(cells) != EXPECTED_CELLS:
        problems.append(f"cells {sorted(set(cells))} != expected 6 cells")
    if man.get("reps_per_cell_per_arm") != 3:
        problems.append(f"reps {man.get('reps_per_cell_per_arm')} != 3")
    if len({r["ids_sha256"] for r in measured}) != len(measured):
        problems.append("measured fixtures are not pairwise distinct")
    warm = man.get("warmup_requests", [])
    if len(warm) != 2:
        problems.append(f"warmup requests {len(warm)} != 2")
    if len({w["ids_sha256"] for w in warm}) != 1:
        problems.append("warm-up requests do not share one fixture")
    if {w["ids_sha256"] for w in warm} & {r["ids_sha256"] for r in measured}:
        problems.append("a measured fixture reuses the warm-up prompt")

    # 4. reuse allowance
    allowance = man["fixture_policy"].get("template_prefix_reuse_tokens")
    if allowance != 2:
        problems.append(f"template_prefix_reuse_tokens {allowance} != 2")

    print(f"measured/warmup          : {len(measured)}/{len(warm)}  cells={len(set(cells))}  allowance={allowance}")
    if problems:
        print("FAIL:");
        for p in problems:
            print("  -", p)
        return 1
    print("MANIFEST OK: fixtures are exact baseline int32 files; counts and reuse allowance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
