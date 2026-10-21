#!/usr/bin/env python3
"""Persistent-server, repeated DeepSeek-V4.1 prefill/decode A/B (MTP/DSpark on vs off).

One engine load per arm (`ds41_generate --serve` through tools/v41/serve/ds41-engine.sh: GPU lock, 84 GiB RAM
cap, swap off), three generations of the same six cells per arm, mean + spread per cell.

  benchmark_server_ab.py prepare                  # write prompts_ab.json + prompts_ab/*.i32   (no model)
  benchmark_server_ab.py run --arm both           # 1 load for the DSpark arm, 1 for the control arm
  benchmark_server_ab.py run --arm mtp|nomtp      # one arm (one engine load)
  benchmark_server_ab.py summarize                # aggregates from results_ab.json only

Why there is no no-DSpark control arm in the one-shot grid: `STRATA_SERVE_DSPARK` is read at load time, so the
arm switch costs one engine restart. Nothing else differs between the arms: same binary, model, Engram, rank,
ctx, arena, route biases, same token-id files, same cell order.

Measurement vs order: one generation = one pass over the six measured cells in the fixed order the one-shot
grid used (1024/128, 1024/256, 2048/128, 2048/256, 4096/128, 4096/256); three generations per arm. The order
is never permuted, so any order/VRAM trend is identical in both arms and the spread across generations is the
cell's own uncertainty. Each measured prompt is a distinct id file (rep1..rep3), so every measurement is a
cache miss; the pre-load warm-up is deliberately repeated twice on the same prompt (its state is reused).

What is recorded: the engine's own prompt/generated counts, the engine's per-phase timing lines (prompt tok/s,
decode tok/s, expert tier counters, DSpark passes/tokens-per-pass), the wall time, output id hash, and host/GPU
samples. VRAM-seed and prompt-batch-cap lines are conditional engine diagnostics, not guaranteed per request:
MTP can seed inside its prompt tail, and a batch-cap line is emitted only when the configured width is reduced.
Their absence does not invalidate an otherwise exact request. Every request is appended to results_ab.json as it
completes, so a killed run keeps its data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- paths / constants

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = HERE.parents[1]
MODEL_DEFAULT = Path.home() / "models/Deepseek-V41-GSQ/V41-GSQ.gguf"
RANK_DEFAULT = Path.home() / "models/Deepseek-V41-GSQ/rank.bin"
ENGRAM_DEFAULT = (
    Path.home() / "models/Deepseek-FP8-engram/layers.1",
    Path.home() / "models/Deepseek-FP8-engram/layers.14",
)
TOKENIZER_DEFAULT = MODEL_DEFAULT.parent / "tokenizer"
SIDECAR_NAME = "V41-MTP-Vision.Q2_K.gguf"
# LOCAL launcher (this benchmark dir): 110 GiB cap + live host-MemAvailable watchdog, DSpark env contract kept
# identical to the tracked tools/v41/serve/ds41-engine.sh.  The tracked launcher is NOT used: its MemoryMax=84G is
# below the required 110 GiB.  Not a tracked file - it lives under this benchmark directory only.
LAUNCHER_REL = "docs/v41-local-benchmark/run_engine_fg.sh"
LAUNCHER_MAL_DEFAULT_DSPARK = "/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf"
GENSCRIPT_REL = "tools/v41/engine/ds41_generate.cpp"      # protocol source of truth (documented below)
# the scope the local launcher is expected to print on stdout before READY (host MemAvailable watchdog floor, GiB)
LAUNCHER_MEMGUARD = {"max_gib": 110, "swap_gib": 0, "watchdog_floor_gib": 4}
PRELOAD_MEM_FLOOR_GIB = 70        # gated BEFORE the engine reserves its 60 GiB arena
RUN_MEM_FLOOR_GIB = 8             # in-run sanity floor: the arena makes 70 GiB impossible once resident

SOURCE_FILES = (
    REPO_DEFAULT / "README.md",
    REPO_DEFAULT / "docs/DEEPSEEK_V41.md",
    REPO_DEFAULT / "docs/xolotl/BENCHMARKS.md",
    REPO_DEFAULT / "docs/MULTI_GPU.md",
    REPO_DEFAULT / "docs/DETAILS.md",
    REPO_DEFAULT / "docs/BATCHING.md",
    REPO_DEFAULT / "docs/INSTALL.md",
    REPO_DEFAULT / "docs/HOW_IT_WORKS.md",
    REPO_DEFAULT / "docs/QUANTS.md",
    REPO_DEFAULT / "docs/VRAM_ELASTIC.md",
    REPO_DEFAULT / "tools/v41/hf/README.md",
    REPO_DEFAULT / "tools/v41/DSPARK_SHRINK.md",
)

# the one-shot grid's measured cells, in the one-shot grid's order
GRID = [(1024, 128), (1024, 256), (2048, 128), (2048, 256), (4096, 128), (4096, 256)]
CELL_NAMES = ["Quartz", "Juniper", "Saffron", "Tangent", "Cobalt", "Marigold"]
WARMUP = (1024, 32)
DEFAULT_REPS = 3
DEFAULT_ARMS = ("mtp", "nomtp")

# ---- engine line protocol (from tools/v41/engine/ds41_generate.cpp, the --serve block) -----------------
# in:  GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. seed=..] <ids csv> | STOP | QUIT
# out: INFO ctx=.. experts_vram=.. arena_experts=.. / READY <ctx> stop / RESUME <reused> /
#      PP <read> <total> <ms> <tok/s> / T <id> / DONE ... / ERR <text>
#
# DONE is printed (ds41_generate.cpp, ~line 1651) as:
#   printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n",
#          gen, total, prompt_ms, dec_ms, finish, reused, hits, lookups, file_tier, total - reused);
# -> done.split() has 15 whitespace fields: the literal "DONE" plus 14 values.  VALUES, 1-based:
#   v1  gen
#   v2  total                                 (prompt tokens the engine counted)
#   v3  prompt_ms
#   v4  decode_ms
#   v5  finish
#   v6  0                                     (reserved)
#   v7  0                                     (reserved)
#   v8  reused                                <-- REUSED IS VALUE/FIELD 8, NOT 7
#   v9  hits                                  (expert VRAM hits)
#   v10 lookups                               (file-tier lookups)
#   v11 0                                     (reserved)
#   v12 file_tier                             (file-tier expert reads)
#   v13 0                                     (reserved)
#   v14 total_prompt_minus_reused             (new prompt tokens; v2 - v8)
# 0-based in code: f[0]="DONE"; f[k] = value k (e.g. reused = f[8], file_tier = f[12], new = f[14]).
# stderr (the phase lines this harness parses):
#   serve: ready (ctx %d, prefill chunk %d)
#   serve: prompt %d tokens (%d reused, %s) in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / %lld pcie / %lld file)
#   serve: prompt phases: %d new = %d %s (%.2f s: dense %.2f, experts %.2f, logits %.2f) + DSpark tail %d%s ...
#   serve: %d tokens in %.2f s = %.2f tok/s (%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file
#   serve: VRAM %.2f GiB free -> %lld experts in VRAM (seed %.2f s)
#   serve: VRAM %.2f GiB free -> prompt batch %d (of %d)
#   DSpark: <sidecar>, %.2f GB of weights on the card (%.1f s)
#   serve: DSpark %ld passes, %.2f tokens/pass (all requests %.2f)
#   serve: reuse lcp=%zu park=%zu lp=%zu hist=%zu snap=%d prompt=%zu -> reused=%d
RE_PROMPT = re.compile(
    r"serve: prompt (\d+) tokens \((\d+) reused, ([^)]*)\) in ([0-9.]+) s = ([0-9.]+) tok/s "
    r"\(experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file\)")
RE_PHASES = re.compile(
    r"serve: prompt phases: (\d+) new = (\d+) (\w+) \(([0-9.]+) s: dense ([0-9.]+), experts ([0-9.]+), "
    r"logits ([0-9.]+)\) \+ DSpark tail (\d+)")
RE_DECODE = re.compile(
    r"serve: (\d+) tokens in ([0-9.]+) s = ([0-9.]+) tok/s \(([a-z]+)\); experts: (\d+) hit, (\d+) cpu, "
    r"(\d+) pcie, (\d+) file")
RE_SEED = re.compile(r"serve: VRAM ([0-9.]+) GiB free -> (\d+) experts in VRAM \(seed ([0-9.]+) s\)")
RE_PBATCH = re.compile(r"serve: VRAM ([0-9.]+) GiB free -> prompt batch (\d+) \(of (\d+)\)")
RE_READY = re.compile(r"serve: ready \(ctx (\d+), prefill chunk (\d+)\)")
RE_REUSE = re.compile(
    r"serve: reuse lcp=(\d+) park=(\d+) lp=(\d+) hist=(\d+) snap=(-?\d+) prompt=(\d+) -> reused=(\d+)")
RE_DSPARK_LOAD = re.compile(r"DSpark: (\S+), ([0-9.]+) GB of weights on the card \(([0-9.]+) s\)")
RE_DSPARK_TURN = re.compile(r"serve: DSpark (\d+) passes, ([0-9.]+) tokens/pass \(all requests ([0-9.]+)\)")
RE_DSPARK_OFF = re.compile(r"serve: STRATA_SERVE_DSPARK needs the GPU layer-major prompt path - DSpark off")
RE_LOADED = re.compile(r"loaded in ([0-9.]+) s: (\d+) layers, ctx (\d+), prefill chunk (\d+)")
RE_MEMGUARD = re.compile(r"memguard: scope (\S+) MemoryMax=(\d+)G swap=(\d+) floor=(\d+) GiB")
RE_CUDA_DEV = re.compile(r"Device (\d+): ([^,]+), compute capability")
RE_ERR_LINE = re.compile(r"^(ERR|DONE) ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- prompt preparation
# copied verbatim from benchmark_grid.py so the two prompt sets are generated by the same code path; both
# arms of THIS harness share these fixtures (the one-shot prompt/*.i32 files are left untouched).

def load_tokenizer(repo: Path, tokdir: Path):
    # strata_tokenizer lives in tools/, dsv41_encoding in tools/v41/ -> both directories must be importable
    for mod_dir in (repo / "tools", repo / "tools/v41"):
        p = str(mod_dir)
        if p not in sys.path:
            sys.path.append(p)
    import strata_tokenizer as ST  # type: ignore[import-not-found]
    import dsv41_encoding as E  # type: ignore[import-not-found]

    vocab_file = tokdir / "vocab.json"
    merges_file = tokdir / "merges.txt"
    types_file = tokdir / "token_type.json"
    for path in (vocab_file, merges_file, types_file):
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer component missing: {path}")
    vocab = json.loads(vocab_file.read_text())
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])] if isinstance(vocab, dict) else vocab
    merges = merges_file.read_text().splitlines()
    types = json.loads(types_file.read_text())

    class Ds41Tokenizer(ST.Tokenizer):
        def __init__(self):
            super().__init__(tokens, merges, types, pre="joyai-llm")
            self._literal_specials = {idx: text.encode("utf-8") for text, idx in self.special_tokens.items()}

        def token_bytes(self, idx):
            literal = self._literal_specials.get(idx)
            return literal if literal is not None else super().token_bytes(idx)

    return Ds41Tokenizer(), E


def clean_source(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = re.sub(r"[｜\u200b\ufeff]", " ", text)
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"^\s{0,3}#{1,6}\s*", "", para, flags=re.M)
        para = re.sub(r"\|", " ", para)
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= 80:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def encode_prompt(tokenizer, encoding, target: int, label: str, source: str) -> list[int]:
    instruction = (
        f"{label} is a benchmark case. Carefully analyze the reference material below and write an exhaustive, "
        "well-organized technical discussion with concrete examples, implications, and caveats. Continue until the "
        "response limit forces you to stop; do not write a short summary or conclude early. Aim for at least 2,000 "
        "words. Distinguish explicit evidence from inference.\n\nReference material:\n\n"
    )
    prompt_text = encoding.encode_messages(
        [{"role": "user", "content": instruction + source}], thinking_mode="chat"
    )
    suffix = None
    for candidate in (
        encoding.ASSISTANT_SP_TOKEN + encoding.thinking_end_token,
        encoding.ASSISTANT_SP_TOKEN + encoding.thinking_start_token,
    ):
        if prompt_text.endswith(candidate):
            suffix = candidate
            break
    if suffix is None:
        raise RuntimeError(f"could not identify generation suffix for {label}: {prompt_text[-80:]!r}")
    ids = tokenizer.encode(prompt_text, parse_special=True)
    suffix_ids = tokenizer.encode(suffix, parse_special=True)
    prefix_ids = ids[:-len(suffix_ids)]
    if len(prefix_ids) + len(suffix_ids) < target:
        raise RuntimeError(f"source too short for {label}: got {len(ids)} tokens, need {target}")
    result = prefix_ids[: target - len(suffix_ids)] + suffix_ids
    if len(result) != target:
        raise AssertionError(f"{label}: prompt length {len(result)} != {target}")
    return result


def rotation_offsets(reps: int) -> list[int]:
    """Deterministic balanced rotation of the six cells across `reps` generations.

    Generation g processes cell index (position + offset_g) % 6, so every cell is measured exactly once per
    generation and the offsets spread the cells evenly across positions (with 6 cells: offsets 0, 2, 4 put each
    cell in three distinct positions, one per generation).  Identical for both arms, so a residual
    order/VRAM/state drift is shared by the arms instead of masquerading as an arm difference.
    """
    return [(2 * g) % len(GRID) for g in range(reps)]


def cell_order_by_generation(reps: int) -> list[list[tuple[int, int]]]:
    """One list of (prefill, output) per generation, in the generation's rotated order."""
    return [[GRID[(pos + off) % len(GRID)] for pos in range(len(GRID))] for off in rotation_offsets(reps)]


def prepare(args) -> dict:
    tokenizer, encoding = load_tokenizer(args.repo, args.tokenizer)
    sources = [(path, clean_source(path)) for path in SOURCE_FILES]
    sources = [(path, text) for path, text in sources if text]
    if not sources:
        raise RuntimeError("no benchmark source documents found")
    corpus = "\n\n".join(f"Source {path.name}:\n{text}" for path, text in sources)
    corpus_ids = tokenizer.encode(corpus, parse_special=False)
    if len(corpus_ids) < 20000:
        raise RuntimeError(f"benchmark source corpus too short: {len(corpus_ids)} tokens, expected >=20000")

    args.outdir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.outdir / "prompts_ab"
    if args.force and prompt_dir.is_dir():
        shutil.rmtree(prompt_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    cursor = 0
    measured: list[dict] = []
    warmup_entry: dict | None = None
    # Fixture policy (requirement 1):
    #   * one pre-load warm-up fixture (WARMUP tokens) that the two warm-up REQUESTS share on purpose;
    #   * 6 cells x reps measured fixtures, EVERY ONE freshly encoded from its own source span - so all 18 (for
    #     reps=3) measured id files are pairwise distinct AND distinct from the warm-up fixture.  No measured
    #     request reuses a warm-up prompt.
    jobs: list[tuple[str, int, int, str, int]] = [("warmup", WARMUP[0], WARMUP[1], "Obsidian", 1)]
    for (prefill, output), name in zip(GRID, CELL_NAMES):
        for rep in range(1, args.reps + 1):
            jobs.append((f"pp{prefill}_out{output}.rep{rep}", prefill, output, name, rep))

    # the corpus must supply every job a distinct span
    need_tokens = sum(prefill + 128 + 16 for _, prefill, _, _, _ in jobs)
    if need_tokens > len(corpus_ids):
        raise RuntimeError(
            f"source corpus too short for {len(jobs)} distinct prompts: need ~{need_tokens} tokens, "
            f"have {len(corpus_ids)} (add source documents or reduce --reps)")

    used_hashes: dict[str, str] = {}
    for case, prefill, output, label, rep in jobs:
        span_len = prefill + 128
        if cursor + span_len > len(corpus_ids):
            raise RuntimeError(
                f"source pool exhausted preparing {case}: need {cursor + span_len}, have {len(corpus_ids)}")
        source = tokenizer.decode(corpus_ids[cursor:cursor + span_len])
        cursor += span_len + 16
        ids = encode_prompt(tokenizer, encoding, prefill, label, source)
        if len(ids) != prefill:
            raise AssertionError(f"{case}: prompt length {len(ids)} != {prefill}")
        raw = struct.pack(f"<{len(ids)}i", *ids)
        digest = sha256_bytes(raw)
        if digest in used_hashes:
            raise RuntimeError(f"{case}: duplicate prompt ids, also produced by {used_hashes[digest]}")
        used_hashes[digest] = case
        path = (prompt_dir / f"{case}.i32").resolve()
        path.write_bytes(raw)
        entry = {
            "case": case,
            "base_case": "warmup" if case == "warmup" else re.sub(r"\.rep\d+$", "", case),
            "phase": "warmup" if case == "warmup" else "measure",
            "prefill_tokens": prefill,
            "max_new_tokens": output,
            "ids_file": str(path),
            "ids_sha256": sha256_bytes(raw),
            "prompt_id_count": len(ids),
            "first_ids": ids[:8],
            "last_ids": ids[-8:],
        }
        if case != "warmup":
            entry["rep"] = int(case.rsplit(".rep", 1)[1])
            measured.append(entry)
        else:
            warmup_entry = entry

    warmup_file = str((prompt_dir / "warmup.i32").resolve())
    assert warmup_entry is not None and warmup_entry["ids_file"] == warmup_file
    warmup_ids = [{**warmup_entry, "index": i + 1} for i in range(args.warmup_repeats)]

    # ---- fixture-uniqueness proof (requirement 1): measured fixtures pairwise distinct, and never the warm-up ----
    measured_hashes = {r["ids_sha256"]: r["case"] for r in measured}
    assert len(measured_hashes) == len(measured), "measured fixtures are not pairwise distinct"
    if warmup_entry["ids_sha256"] in measured_hashes:
        raise RuntimeError(f"measured fixture {measured_hashes[warmup_entry['ids_sha256']]} reuses the warm-up prompt")
    assert len({w["ids_sha256"] for w in warmup_ids}) == 1, "the two warm-up requests must share one fixture"

    gens = cell_order_by_generation(args.reps)
    generation_schedule = [
        [{"generation": g + 1, "position": pos + 1,
          "case": f"pp{prefill}_out{output}.rep{g + 1}"}
         for pos, (prefill, output) in enumerate(order)]
        for g, order in enumerate(gens)
    ]

    # ---- template-prefix reality check ---------------------------------------------------------------
    # In a PERSISTENT server the engine's reuse is the longest common prefix of the current prompt vs the
    # previous request's history.  Every fixture from this tokenizer/chat template starts with the same short
    # BOS/role prefix, so a genuinely NEW fixture still reports `reused == template_prefix_len` (a handful of
    # tokens) and NOT zero.  A literal "0 reused" is therefore physically impossible here; what IS provable and
    # what the guard enforces is: no measured request reuses MORE than this fixed template prefix, i.e. zero
    # FIXTURE-level reuse.  We compute the prefix from the real request order and record it for audit.
    ordered_ids: list[list[int]] = []
    for gen in generation_schedule:
        for e in gen:
            case = e["case"]
            raw = (prompt_dir / f"{case}.i32").read_bytes()
            ordered_ids.append(list(struct.unpack(f"<{len(raw)//4}i", raw)))
    lcps = []
    prev = list(struct.unpack(f"<{warmup_entry['prompt_id_count']}i",
                              (prompt_dir / 'warmup.i32').read_bytes()))
    for cur in ordered_ids:
        n = 0
        while n < len(prev) and n < len(cur) and prev[n] == cur[n]:
            n += 1
        lcps.append(n)
        prev = cur
    template_prefix_len = min(lcps) if lcps else 0
    if max(lcps) != template_prefix_len:
        raise RuntimeError(
            f"measured fixtures are not prefix-disjoint: consecutive-prompt lcp varies "
            f"{sorted(set(lcps))} - a fixture shares more than the template prefix with its predecessor")

    manifest = {
        "created_utc": utc_now(),
        "purpose": "persistent-server repeated prefill/decode A/B: MTP/DSpark on vs off",
        "repo": str(args.repo),
        "tokenizer_dir": str(args.tokenizer),
        "tokenizer_vocab": len(tokenizer.tokens),
        "source_files": [str(path) for path, _ in sources],
        "source_corpus_tokens": len(corpus_ids),
        "source_corpus_sha256": sha256_bytes(corpus.encode("utf-8")),
        "prompt_template": "DeepSeek V4.1 chat template; exhaustive response; greedy decode (temperature=0)",
        "launcher": str(args.repo / LAUNCHER_REL),
        "protocol_source": str(args.repo / GENSCRIPT_REL),
        "protocol": "GEN <max_new> temperature=0 <ids csv> | STOP | QUIT",
        "reps_per_cell_per_arm": args.reps,
        "warmup_per_arm": len(warmup_ids),
        "fixture_policy": {
            "measured_fixtures": len(measured),
            "measured_pairwise_distinct": True,
            "measured_disjoint_from_warmup": True,
            "warmup_requests_share_one_fixture": True,
            "fixture_ids_identical_between_arms": True,
            # the fixed, unavoidable template-prefix reuse reported for a brand-new fixture in the persistent
            # server; the reuse guard allows exactly this much and nothing more (zero fixture-level reuse).
            "template_prefix_reuse_tokens": template_prefix_len,
            "measured_reuse_expectation": "== template_prefix_reuse_tokens (no fixture-level reuse); a literal "
                                          "0 is impossible in the persistent server because of the shared "
                                          "chat-template prefix",
        },
        "schedule": {
            "cells_in_order": [{"prefill_tokens": p, "max_new_tokens": o} for p, o in GRID],
            "order_rule": "deterministic balanced rotation: generation g uses offset (2*g) mod 6; identical for "
                          "both arms",
            "rotation_offsets": rotation_offsets(args.reps),
            "generation": "one pass over the six cells in the generation's rotated order",
            "generation_schedule": generation_schedule,
            "measured_requests_per_arm": len(GRID) * args.reps,
        },
        "arms": {
            "mtp": "STRATA_SERVE_DSPARK=" + str((args.model.parent / SIDECAR_NAME).resolve()) +
                   " (explicit local Q2_K sidecar; the launcher would otherwise default to the Mal path "
                   + LAUNCHER_MAL_DEFAULT_DSPARK + "); DSpark defaults DEPTH 4, KEEP file:64, the launcher's "
                   "CONF/LOOKUP/EVICT - not overridden",
            "nomtp": "STRATA_SERVE_DSPARK= (EMPTY) so run_engine_fg.sh unsets it at exec - the control arm runs "
                     "without drafts; the only other difference is the engine restart",
        },
        "requests": measured,
        "warmup_requests": warmup_ids,
    }
    manifest_path = (args.outdir / "prompts_ab.json").resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "prompt_dir": str(prompt_dir.resolve()),
        "source_tokens": len(corpus_ids),
        "cells": [{"case": r["case"], "prefill": r["prefill_tokens"], "output": r["max_new_tokens"],
                   "ids_sha256": r["ids_sha256"]} for r in measured],
    }, indent=2))
    return manifest


# ---------------------------------------------------------------- host probes

def read_mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    raise RuntimeError("could not read MemAvailable from /proc/meminfo")


def read_swap_gib() -> tuple[float, float]:
    total = free = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("SwapTotal:"):
            total = int(line.split()[1]) / 1048576.0
        elif line.startswith("SwapFree:"):
            free = int(line.split()[1]) / 1048576.0
    return total, free


def gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.free,utilization.gpu,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def gpu_cards() -> list[list[str]]:
    return [line.split(",") for line in gpu_snapshot().splitlines() if line.strip()]


def free_vram_mib(cards: list[list[str]], physical_gpu: int) -> int:
    if len(cards) <= physical_gpu:
        raise RuntimeError(f"physical GPU index {physical_gpu} absent from nvidia-smi output")
    return int(cards[physical_gpu][3].strip())


def port_busy(port: int) -> bool:
    out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    return f":{port} " in out


# ---------------------------------------------------------------- the serve driver

class ServeDriver:
    """Owns one engine process (`ds41_generate --serve`) and speaks its line protocol."""

    def __init__(self, args, arm: str, log_prefix: Path, trace_path: Path):
        self.args = args
        self.arm = arm
        self.log_prefix = log_prefix
        self.trace_path = trace_path
        self.proc: subprocess.Popen | None = None
        self.stdout_log: Path | None = None
        self.stderr_log: Path | None = None
        self.load_seconds: float | None = None
        self.ready_line: str | None = None
        self.info_line: str | None = None
        self.dspark_loaded: list[str] = []
        self.notes: list[str] = []
        self.memguard_line: str | None = None
        self.scope: dict | None = None
        self.dspark_state: dict | None = None
        self.stderr_after_ready: list[str] = []

    # ---- lifecycle
    def dspark_intent(self) -> str:
        """The STRATA_SERVE_DSPARK value this arm MUST pass to the launcher (empty string = MTP off)."""
        return "" if self.arm == "nomtp" else str(self.args.sidecar)

    def engine_env(self) -> dict:
        # Strip every inherited STRATA_* tuning switch. Otherwise a shell startup file can quietly change the
        # engine path or make the two arms differ in ways the manifest never records. The local launcher then
        # applies its documented defaults; the only arm-specific setting is STRATA_SERVE_DSPARK.
        env = {key: value for key, value in os.environ.items() if not key.startswith("STRATA_")}
        env["DS4_LOCK"] = str(Path.home() / ".quetza-data/conductor/ds4-gpu.lock")
        env["STRATA_BUILD_DIR"] = str(self.args.repo / "build-xolotl")
        # Never bypass the ComfyUI gate through an inherited override.
        env["MEMGUARD_ALLOW_COMFY"] = "0"
        # the launcher's own defaults would otherwise apply; pin every knob the harness cares about so the two
        # arms are byte-identical apart from DSpark, and no outer shell can silently change the run.
        env["STRATA_NO_LARGEPAGES"] = "1"
        # requirement 4: independent requests must not park each other's conversation - the park path exists for
        # agent harnesses interleaving a main conversation with side requests.  Our prompts are all unrelated, so
        # PARK=0 is the correct, deterministic setting for every arm (and it is pinned identically).
        env["STRATA_SERVE_PARK"] = "0"
        # requirement 3: this harness uses its OWN launcher contract, not the tracked 84 GiB one.  Pin the cap and
        # both MemAvailable floors explicitly so they cannot drift with an ambient env var.
        env["BENCH_MEMGUARD_MAX_GIB"] = str(LAUNCHER_MEMGUARD["max_gib"])
        env["BENCH_MIN_MEM_AVAIL_GIB"] = str(PRELOAD_MEM_FLOOR_GIB)
        env["BENCH_WATCHDOG_FLOOR_GIB"] = str(LAUNCHER_MEMGUARD["watchdog_floor_gib"])
        # requirement 2: pass STRATA_SERVE_DSPARK EXPLICITLY in both arms.  The launcher (like ds41-engine.sh)
        # defaults an *unset* variable to the Mal sidecar path; an *empty* value makes it unset the variable, so
        # the MTP-off arm must pass "" and the MTP-on arm must pass the local Q2_K sidecar.  Never leave it unset.
        env["STRATA_SERVE_DSPARK"] = self.dspark_intent()
        return env

    def command(self) -> list[str]:
        cmd = [
            "/bin/bash", str(self.args.repo / LAUNCHER_REL),
            "-m", str(self.args.model), "--backend", "cuda", "--experts", "gpu",
            "--engram", self.args.engram, "--rank", str(self.args.rank),
            "--ctx", str(self.args.ctx), "--serve",
        ]
        if self.args.prompt_batch > 0:
            cmd += ["--prompt-batch", str(self.args.prompt_batch)]
        if self.args.prefill_chunk > 0:
            cmd += ["--prefill-chunk", str(self.args.prefill_chunk)]
        return cmd

    def start(self) -> None:
        args = self.args
        self.stdout_log = self.log_prefix.with_suffix(".server.stdout.log")
        self.stderr_log = self.log_prefix.with_suffix(".server.stderr.log")
        if port_busy(8188) and not args.allow_comfy:
            raise RuntimeError("ComfyUI is listening on :8188; the launcher would refuse (stop it or pass --allow-comfy)")
        if args.physical_gpu is not None:
            cards = gpu_cards()
            free = free_vram_mib(cards, args.physical_gpu)
            if free < args.min_free_vram_mib:
                raise RuntimeError(
                    f"refusing load: GPU{args.physical_gpu} has only {free} MiB free; need {args.min_free_vram_mib} MiB")
            snapshot = gpu_snapshot()
        else:
            snapshot = gpu_snapshot()
        avail = read_mem_available_gib()
        if avail < args.min_mem_available_gib:
            raise RuntimeError(
                f"refusing load: MemAvailable {avail:.1f} GiB < {args.min_mem_available_gib} GiB")
        self.gpu_before = snapshot
        self.mem_before = avail

        env = self.engine_env()
        dspark_env = env["STRATA_SERVE_DSPARK"]
        self.trace("start", cmd=self.command(), env_dspark=dspark_env, env_park=env.get("STRATA_SERVE_PARK"),
                   env_memguard_max=env.get("BENCH_MEMGUARD_MAX_GIB"),
                   env_watchdog_floor=env.get("BENCH_WATCHDOG_FLOOR_GIB"),
                   mem_available_gib=avail, gpu=snapshot)
        so = self.stdout_log.open("w")
        se = self.stderr_log.open("w")
        hdr = (f"# {utc_now()} arm={self.arm} cmd={' '.join(self.command())}\n"
               f"# STRATA_SERVE_DSPARK={dspark_env!r} STRATA_SERVE_PARK={env.get('STRATA_SERVE_PARK')} "
               f"BENCH_MEMGUARD_MAX_GIB={env.get('BENCH_MEMGUARD_MAX_GIB')} "
               f"BENCH_MIN_MEM_AVAIL_GIB={env.get('BENCH_MIN_MEM_AVAIL_GIB')} "
               f"BENCH_WATCHDOG_FLOOR_GIB={env.get('BENCH_WATCHDOG_FLOOR_GIB')}\n")
        so.write(hdr)
        se.write(hdr)
        so.flush(); se.flush()
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=se,
            text=True, bufsize=1, env=env)
        # The parent descriptor is no longer needed; Popen has duplicated it for the child.
        se.close()
        # The launcher prints the memguard scope line on stdout before it execs the engine; the engine's banner
        # (INFO) and READY follow.  Keep a copy of everything in the log.
        self.memguard_line = None
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    tail = self.stderr_log.read_text(errors="replace")[-2000:]
                    raise RuntimeError(f"engine died before READY (see {self.stderr_log}); tail: {tail}")
                so.write(line)
                so.flush()
                m = RE_MEMGUARD.search(line)
                if m:
                    self.memguard_line = line.strip()
                    self.scope = {"unit": m.group(1), "max_gib": int(m.group(2)),
                                  "swap_gib": int(m.group(3)), "watchdog_floor_gib": int(m.group(4))}
                    self.notes.append(line.strip())
                if line.startswith("INFO"):
                    self.info_line = line.strip()
                if line.startswith("READY"):
                    self.ready_line = line.strip()
                    break
                if line.startswith("ERR"):
                    raise RuntimeError(f"engine error before READY: {line.strip()}")
            self.load_seconds = time.monotonic() - t0
            self.stderr_after_ready = self._read_new_stderr()   # load-time stderr (DSpark sidecar + reuse lines)
            self._verify_scope(dspark_env)
            self.trace("ready", seconds=round(self.load_seconds, 3), line=self.ready_line, info=self.info_line,
                       memguard=self.memguard_line)
        except BaseException:
            # A post-READY verification failure must not leak the loaded model or its 110-GiB scope.
            try:
                self.stop()
            except Exception as cleanup_error:                  # noqa: BLE001
                self.notes.append(f"startup cleanup failed: {cleanup_error}")
            raise
        finally:
            so.close()
        return

    def _verify_scope(self, dspark_env: str) -> None:
        """Prove the 110G/swap-0 scope at READY; DSpark itself initializes lazily on the first GEN."""
        if self.memguard_line is None:
            raise RuntimeError("the local launcher did not print a memguard scope line before READY; refusing to "
                               "continue (cannot prove the 110G cap / watchdog are in force)")
        m = RE_MEMGUARD.search(self.memguard_line)
        assert m is not None
        max_gib, swap_gib, floor_gib = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if (max_gib, swap_gib, floor_gib) != (LAUNCHER_MEMGUARD["max_gib"], LAUNCHER_MEMGUARD["swap_gib"],
                                              LAUNCHER_MEMGUARD["watchdog_floor_gib"]):
            raise RuntimeError(
                f"launcher scope mismatch: got MemoryMax={max_gib}G swap={swap_gib} floor={floor_gib} GiB, "
                f"expected {LAUNCHER_MEMGUARD}")
        self.scope = {"unit": m.group(1), "max_gib": max_gib, "swap_gib": swap_gib,
                      "watchdog_floor_gib": floor_gib}

        if self.arm == "mtp":
            if dspark_env != str(self.args.sidecar):
                raise RuntimeError(f"mtp arm must pass the local sidecar; env has {dspark_env!r}")
            if dspark_env == LAUNCHER_MAL_DEFAULT_DSPARK:
                raise RuntimeError("mtp arm: STRATA_SERVE_DSPARK is the Mal default path, not the local sidecar")
        elif dspark_env != "":
            raise RuntimeError(f"nomtp arm must pass an EMPTY STRATA_SERVE_DSPARK; env has {dspark_env!r}")
        self.dspark_expected_env = dspark_env
        self.dspark_state = {
            "intent": "on" if self.arm == "mtp" else "off",
            "env_value": dspark_env,
            "sidecar_load_lines": [],
            "verified": False,
        }

    def _verify_dspark_activity(self) -> None:
        """After the first GEN, verify lazy sidecar initialization (or its absence for the control arm)."""
        sidecars = list(self.dspark_loaded)
        if self.arm == "mtp":
            if not sidecars:
                raise RuntimeError("mtp arm: engine printed no DSpark sidecar load line after first warm-up request; "
                                   "MTP is not active")
            if str(self.args.sidecar) not in sidecars:
                raise RuntimeError(f"mtp arm: engine loaded {sidecars!r}, expected {self.args.sidecar}")
        elif sidecars:
            raise RuntimeError(f"nomtp arm: engine loaded a DSpark sidecar {sidecars!r}; MTP must be off")
        self.dspark_state = {
            "intent": "on" if self.arm == "mtp" else "off",
            "env_value": self.dspark_expected_env,
            "sidecar_load_lines": sidecars,
            "verified": True,
            "verified_after": "warmup#1",
        }

    def stop(self) -> int:
        rc = None
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                rc = self.proc.wait(timeout=self.args.shutdown_timeout)
            except Exception as exc:                            # noqa: BLE001 - make a best effort to kill the unit too
                # Killing only the shell parent can orphan ds41_generate inside its transient systemd scope.
                if self.scope and self.scope.get("unit"):
                    try:
                        subprocess.run(
                            ["systemctl", "--user", "kill", "--signal=KILL", "--kill-whom=all",
                             self.scope["unit"]], check=False, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=10)
                    except Exception:                            # noqa: BLE001
                        pass
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                rc = self.proc.wait(timeout=30)
                self.notes.append(f"graceful shutdown failed; scope kill attempted ({type(exc).__name__}: {exc})")
            finally:
                for stream in (self.proc.stdin, self.proc.stdout):
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass
        self.trace("stop", returncode=rc)
        return rc if rc is not None else -1

    # ---- protocol
    def request(self, case: str, ids: list[int], max_new: int, output_ids_path: Path,
                max_reuse_tokens: int | None = 0) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        info: dict = {
            "case": case,
            "arm": self.arm,
            "prepared_prompt_tokens": len(ids),
            "requested_max_new_tokens": max_new,
            "temperature": 0,
            "max_reuse_tokens_allowed": max_reuse_tokens,
            "sent_utc": utc_now(),
        }
        err = None
        tokens: list[int] = []
        done = None
        t0 = time.monotonic()
        self.trace("request_start", case=case, prompt_tokens=len(ids), max_new=max_new,
                   mem_available_gib=round(read_mem_available_gib(), 3))
        try:
            line = f"GEN {max_new} temperature=0 " + ",".join(map(str, ids)) + "\n"
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            while True:
                out = self.proc.stdout.readline()
                if not out:
                    raise RuntimeError("engine died mid-request (stdout closed)")
                out = out.strip()
                if out.startswith("T "):
                    tokens.append(int(out[2:]))
                elif out.startswith("RESUME"):
                    info["resume_reused"] = int(out.split()[1])
                elif out.startswith("PP"):
                    pass
                elif out.startswith("DONE"):
                    done = out
                    break
                elif out.startswith("ERR"):
                    err = out[4:].strip()
                    done = None
                    break
        except Exception as exc:                                  # noqa: BLE001 - recorded, not swallowed
            err = f"{type(exc).__name__}: {exc}"
        wall = time.monotonic() - t0

        # the engine's stderr is the only place with the phase lines: read the bytes this request appended and
        # parse them (buffered by the child, so a line may lag by one request - the parser takes the LAST match
        # and each request's case is recorded next to it).
        self.stderr_tail = self._read_new_stderr()
        info["engine_stderr_lines"] = self.stderr_tail[-40:]

        info["wall_seconds"] = round(wall, 3)
        info["generated_tokens_emitted"] = len(tokens)
        info["error"] = err
        if done is not None:
            f = done.split()
            info["done_line"] = done
            info["done_field_count"] = len(f)
            try:
                if len(f) != 15:
                    raise ValueError(f"expected 15 whitespace fields (DONE + 14 values), got {len(f)}")
                # done.split()[0] == "DONE"; the 14 VALUES follow (0-based index = 1 + value number):
                info["done_generated"] = int(f[1])            # value 1  gen
                info["done_total_prompt"] = int(f[2])         # value 2  total
                info["done_prompt_ms"] = float(f[3])          # value 3  prompt_ms
                info["done_decode_ms"] = float(f[4])          # value 4  dec_ms
                info["done_finish"] = f[5]                    # value 5  finish
                # values 6,7 are the reserved "0 0"
                info["done_reused"] = int(f[8])               # value 8  reused  <-- FIELD 8, NOT FIELD 7
                info["done_hits"] = int(f[9])                 # value 9  hits
                info["done_lookups"] = int(f[10])             # value 10 lookups
                # value 11 is the reserved "0"
                info["done_file_tier"] = int(f[12])           # value 12 file_tier
                # value 13 is the reserved "0"
                info["done_new_prompt"] = int(f[14])          # value 14 total - reused
                info["done_experts"] = {"hit": info["done_hits"], "lookups": info["done_lookups"],
                                        "file": info["done_file_tier"], "new": info["done_new_prompt"]}
            except (IndexError, ValueError) as exc:
                info["done_parse_error"] = f"{exc}: {done}"
        text = "\n".join(self.stderr_tail)
        m = RE_PROMPT.search(text)
        if m:
            info.update({
                "engine_prompt_tokens": int(m.group(1)),
                "engine_prompt_reused": int(m.group(2)),
                "engine_prompt_path": m.group(3),
                "prompt_seconds": float(m.group(4)),
                "prompt_tok_s": float(m.group(5)),
                "prompt_expert_hits": int(m.group(6)),
                "prompt_expert_cpu": int(m.group(7)),
                "prompt_expert_pcie": int(m.group(8)),
                "prompt_expert_file": int(m.group(9)),
            })
        m = RE_PHASES.search(text)
        if m:
            info.update({
                "prompt_new_tokens": int(m.group(1)),
                "prompt_fed_tokens": int(m.group(2)),
                "prompt_feed_mode": m.group(3),
                "prompt_feed_seconds": float(m.group(4)),
                "prompt_dense_seconds": float(m.group(5)),
                "prompt_experts_seconds": float(m.group(6)),
                "prompt_logits_seconds": float(m.group(7)),
                "dspark_tail_tokens": int(m.group(8)),
            })
        m = RE_DECODE.search(text)
        if m:
            info.update({
                "decode_tokens": int(m.group(1)),
                "decode_seconds": float(m.group(2)),
                "decode_tok_s": float(m.group(3)),
                "decode_finish": m.group(4),
                "decode_expert_hits": int(m.group(5)),
                "decode_expert_cpu": int(m.group(6)),
                "decode_expert_pcie": int(m.group(7)),
                "decode_expert_file": int(m.group(8)),
            })
        m = RE_SEED.search(text)
        if m:
            info.update({"seed_free_gib": float(m.group(1)), "vram_experts_resident": int(m.group(2)),
                         "seed_seconds": float(m.group(3))})
        m = RE_PBATCH.search(text)
        if m:
            info.update({"prompt_batch_free_gib": float(m.group(1)), "prompt_batch_width": int(m.group(2)),
                         "prompt_batch_requested": int(m.group(3))})
        m = RE_REUSE.search(text)
        if m:
            info.update({"reuse_lcp": int(m.group(1)), "reuse_hist": int(m.group(4)),
                         "reuse_snap": int(m.group(5)), "reuse_reported": int(m.group(7))})
        for dl in RE_DSPARK_LOAD.findall(text):
            if dl[0] not in self.dspark_loaded:
                self.dspark_loaded.append(dl[0])
                info["dspark_sidecar_loaded"] = dl[0]
                info["dspark_weight_gb"] = float(dl[1])
                info["dspark_init_seconds"] = float(dl[2])
        m = RE_DSPARK_TURN.search(text)
        if m:
            info.update({"dspark_passes": int(m.group(1)), "dspark_tokens_per_pass": float(m.group(2)),
                         "dspark_tokens_per_pass_all_requests": float(m.group(3))})
        if RE_DSPARK_OFF.search(text):
            info["dspark_disabled_by_engine"] = True
        info["raw_output_ids"] = tokens
        info["output_ids_sha256"] = sha256_bytes(struct.pack(f"<{len(tokens)}i", *tokens))
        output_ids_path.write_bytes(struct.pack(f"<{len(tokens)}i", *tokens))
        info["output_ids_file"] = str(output_ids_path)

        # ---- verdict
        problems = []
        if err:
            problems.append(f"engine reported: {err}")
        if done is None and not err:
            problems.append("no DONE line")
        if info.get("done_parse_error"):
            problems.append(f"DONE parse: {info['done_parse_error']}")
        if info.get("engine_prompt_tokens") is not None:
            if info["engine_prompt_tokens"] != len(ids):
                problems.append(
                    f"engine counted {info['engine_prompt_tokens']} prompt tokens, prepared {len(ids)}")
        else:
            problems.append("no engine prompt line")
        if info.get("decode_tokens") is not None and info["decode_tokens"] != max_new:
            problems.append(f"engine generated {info['decode_tokens']} tokens, requested {max_new}")
        if len(tokens) != max_new:
            problems.append(f"{len(tokens)} output ids emitted, requested {max_new}")
        # DONE's own counts must agree with the phase lines and the request (an exact-count requirement).
        if info.get("done_total_prompt") is not None and info.get("engine_prompt_tokens") is not None \
                and info["done_total_prompt"] != info["engine_prompt_tokens"]:
            problems.append(f"DONE prompt {info['done_total_prompt']} != prompt line {info['engine_prompt_tokens']}")
        if info.get("done_generated") is not None and info["done_generated"] != max_new:
            problems.append(f"DONE generated {info['done_generated']} != requested {max_new}")
        # reuse guard: a measured cell must NOT reuse more than the fixed chat-template prefix (zero FIXTURE-level
        # reuse).  We check BOTH the DONE field 8 and the 'serve: prompt N tokens (R reused, ...)' line and the
        # 'serve: reuse ... -> reused=R' line.  A literal 0 is not achievable in a persistent server (see prepare).
        if max_reuse_tokens is not None:
            reused_sources = {}
            if info.get("done_reused") is not None:
                reused_sources["done_reused_field8"] = info["done_reused"]
            if info.get("engine_prompt_reused") is not None:
                reused_sources["prompt_line_reused"] = info["engine_prompt_reused"]
            if info.get("reuse_reported") is not None:
                reused_sources["reuse_line_reused"] = info["reuse_reported"]
            if info.get("resume_reused") is not None:
                reused_sources["resume_reused"] = info["resume_reused"]
            info["reuse_sources"] = reused_sources
            if not reused_sources:
                problems.append("measured cell: no reuse field found to prove reuse (DONE/prompt/reuse/RESUME)")
            over = {k: v for k, v in reused_sources.items() if v > max_reuse_tokens}
            if over:
                problems.append(f"measured cell reused {over} prompt tokens, allowance is only "
                                f"{max_reuse_tokens} (the chat-template prefix)")
            # a request that is a fixture repeat would reuse a whole prompt: catch the degenerate case explicitly
            if reused_sources and max(reused_sources.values()) > len(ids) // 2:
                problems.append(f"measured cell reused more than half its prompt "
                                f"({reused_sources}); it is not a fresh fixture")
        # These are conditional engine diagnostics, not a request-success contract. MTP may seed its cache while
        # executing the prompt tail, before the later serve-level seed log; a batch-cap line exists only when the
        # engine actually has to reduce its configured prompt-batch width. Parse them when present, but do not reject
        # exact prompt/DONE/output results merely because neither diagnostic was emitted.
        info["problems"] = problems
        info["accepted"] = not problems
        info["finished_utc"] = utc_now()
        self.trace("request_end", case=case, accepted=info["accepted"], problems=problems,
                   prompt_tok_s=info.get("prompt_tok_s"), decode_tok_s=info.get("decode_tok_s"),
                   wall_seconds=info["wall_seconds"])
        return info

    # ---- plumbing
    def _read_new_stderr(self) -> list[str]:
        assert self.stderr_log is not None
        size = self.stderr_log.stat().st_size
        pos = getattr(self, "_stderr_pos", 0)
        with self.stderr_log.open("r", errors="replace") as f:
            f.seek(pos)
            data = f.read()
        self._stderr_pos = size
        return [ln for ln in data.splitlines() if ln.strip()]

    def trace(self, event: str, **fields) -> None:
        rec = {"utc": utc_now(), "event": event, "arm": self.arm, **fields}
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- the run

def load_manifest(args) -> dict:
    path = args.outdir / "prompts_ab.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest missing: {path} (run `prepare` first)")
    return json.loads(path.read_text())


def verify_fixtures(manifest: dict) -> None:
    """Manifest integrity + the requirement-1 uniqueness proof (run before any engine load)."""
    def check_file(req):
        p = Path(req["ids_file"])
        if not p.is_file():
            raise FileNotFoundError(f"prompt id file missing: {p}")
        raw = p.read_bytes()
        if sha256_bytes(raw) != req["ids_sha256"]:
            raise RuntimeError(f"prompt id file changed since prepare: {p}")
        if len(raw) != 4 * req["prompt_id_count"]:
            raise RuntimeError(f"prompt id file size mismatch: {p}")

    for req in manifest["requests"]:
        check_file(req)
    for req in manifest.get("warmup_requests", []):
        check_file(req)

    measured = manifest["requests"]
    seen: dict[str, list[str]] = {}
    for req in measured:
        seen.setdefault(req["ids_sha256"], []).append(req["case"])
    dupes = {h: c for h, c in seen.items() if len(c) > 1}
    if dupes:
        raise RuntimeError(f"measured prompts are not unique: {dupes}")
    warm_hashes = {req["ids_sha256"] for req in manifest.get("warmup_requests", [])}
    overlap = [req["case"] for req in measured if req["ids_sha256"] in warm_hashes]
    if overlap:
        # requirement 1: NO measured fixture may reuse a warm-up prompt
        raise RuntimeError(f"measured fixtures reuse a warm-up prompt: {overlap}")
    # the schedule must be present and identical for both arms (one manifest drives both arms)
    if len(manifest.get("schedule", {}).get("generation_schedule", [])) != manifest["reps_per_cell_per_arm"]:
        raise RuntimeError("manifest generation_schedule missing or wrong length")


def results_writer(args, path: Path, payload: dict) -> None:
    # Keep the last complete snapshot if the process is killed mid-write (OOM/watchdog/power loss).
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def run_arm(args, manifest: dict, arm: str, results: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = args.outdir / "runs_ab"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = runs_dir / f"{stamp}.{arm}.run"
    trace_path = log_prefix.with_suffix(".trace.jsonl")
    driver = ServeDriver(args, arm, log_prefix, trace_path)

    print(f"== arm {arm}: starting the engine (one load for the whole arm) ==\n"
          f"   STRATA_SERVE_DSPARK={driver.dspark_intent()!r} STRATA_SERVE_PARK=0 "
          f"scope={LAUNCHER_MEMGUARD}", flush=True)
    driver.start()
    print(f"   ready in {driver.load_seconds:.1f} s; {driver.info_line}", flush=True)
    results["arms"][arm]["engine"] = {
        "started_utc": utc_now(),
        "command": driver.command(),
        "stdout_log": str(driver.stdout_log),
        "stderr_log": str(driver.stderr_log),
        "trace": str(trace_path),
        "ready_line": driver.ready_line,
        "info_line": driver.info_line,
        "load_seconds": round(driver.load_seconds, 3),
        "notes": driver.notes,
        "memguard_scope_line": driver.memguard_line,
        "scope": driver.scope,
        "dspark_intent": driver.dspark_intent(),
        "dspark_state": getattr(driver, "dspark_state", None),
        "gpu_before": driver.gpu_before,
        "mem_available_before_gib": round(driver.mem_before, 3),
    }
    results_writer(args, args.outdir / "results_ab.json", results)

    try:
        # ---- warm-up: the two warm-up requests intentionally share ONE fixture (reuse is allowed here) ----
        for i, w in enumerate(manifest["warmup_requests"], 1):
            check_guards(args, f"warmup {i}/{len(manifest['warmup_requests'])}")
            ids = load_ids(Path(w["ids_file"]))
            info = driver.request(f"warmup#{i}", ids, w["max_new_tokens"],
                                  log_prefix.with_suffix(f".warmup{i}.output.i32"), max_reuse_tokens=None)
            info.update({"phase": "warmup", "ids_file": w["ids_file"], "ids_sha256": w["ids_sha256"],
                         "warmup_index": i, "warmup_reuses_fixture": True})
            results["arms"][arm]["warmup"].append(info)
            guard_accepted(args, info)
            results_writer(args, args.outdir / "results_ab.json", results)
            if i == 1:
                driver._verify_dspark_activity()
                results["arms"][arm]["engine"]["dspark_state"] = driver.dspark_state
                results_writer(args, args.outdir / "results_ab.json", results)
            print(f"   warmup {i}: prompt {info.get('engine_prompt_tokens')} reused "
                  f"{info.get('engine_prompt_reused')} -> {info.get('decode_tok_s')} tok/s decode", flush=True)

        # ---- measured generations: the manifest's deterministic balanced rotation (identical for both arms) ----
        by_case = {r["case"]: r for r in manifest["requests"]}
        schedule = manifest["schedule"]["generation_schedule"]
        for generation in range(1, args.reps + 1):
            order = schedule[generation - 1]
            print(f"-- generation {generation}/{args.reps}: "
                  f"{' -> '.join(e['case'] for e in order)} --", flush=True)
            for cell_idx, entry in enumerate(order, 1):
                case = entry["case"]
                req = by_case[case]
                check_guards(args, case)
                ids = load_ids(Path(req["ids_file"]))
                out_path = log_prefix.with_suffix(f".{case}.output.i32")
                t0 = time.monotonic()
                # measured fixtures may reuse at most the fixed chat-template prefix (recorded in the manifest)
                allowance = manifest["fixture_policy"]["template_prefix_reuse_tokens"]
                info = driver.request(case, ids, req["max_new_tokens"], out_path,
                                      max_reuse_tokens=allowance)
                info.update({
                    "generation": generation,
                    "cell_index_in_generation": cell_idx,
                    "rotation_offset": manifest["schedule"]["rotation_offsets"][generation - 1],
                    "base_case": req["base_case"],
                    "prefill_tokens": req["prefill_tokens"],
                    "max_new_tokens": req["max_new_tokens"],
                    "ids_file": req["ids_file"],
                    "ids_sha256": req["ids_sha256"],
                    "wall_seconds_including_request": round(time.monotonic() - t0, 3),
                })
                info["repeats_this_prompt_on_engine"] = 1   # distinct id file per generation -> a cache miss
                results["arms"][arm]["results"].append(info)
                results_writer(args, args.outdir / "results_ab.json", results)
                ok = "ok" if info["accepted"] else "FAILED: " + "; ".join(info["problems"])
                print(f"   [g{generation} p{cell_idx}] {case}: prompt {info.get('engine_prompt_tokens')} "
                      f"(reused {info.get('engine_prompt_reused')}, {info.get('prompt_tok_s')} tok/s, "
                      f"{info.get('prompt_seconds')} s), decode {info.get('decode_tok_s')} tok/s "
                      f"({info.get('decode_seconds')} s), wall {info['wall_seconds']} s -> {ok}", flush=True)
                guard_accepted(args, info)
    finally:
        rc = driver.stop()
        results["arms"][arm]["engine"].update({
            "stopped_utc": utc_now(), "exit_code": rc,
            "dspark_sidecars_loaded": driver.dspark_loaded,
            "mem_available_after_gib": round(read_mem_available_gib(), 3),
            "gpu_after": gpu_snapshot(),
        })
        results_writer(args, args.outdir / "results_ab.json", results)
        print(f"== arm {arm}: engine stopped (exit {rc}); VRAM/MemAvailable logged ==", flush=True)


def load_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise RuntimeError(f"{path}: not a whole number of int32 values")
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def guard_accepted(args, info: dict) -> None:
    if info["accepted"] or args.allow_failures:
        return
    raise RuntimeError(
        f"{info['case']} failed: {'; '.join(info['problems'])}; see {info.get('engine_stderr_lines', [])[-4:]}")


def check_guards(args, label: str) -> None:
    # requirement 3: the 70 GiB floor is a PRE-LOAD gate only.  Once the engine's 60 GiB arena is resident, a
    # 70 GiB MemAvailable is physically impossible; the in-run guard uses RUN_MEM_FLOOR_GIB (sanity/protection),
    # and the launcher's own host watchdog (LAUNCHER_MEMGUARD['watchdog_floor_gib']) is the hard backstop.
    avail = read_mem_available_gib()
    if avail < RUN_MEM_FLOOR_GIB:
        raise RuntimeError(f"refusing request {label}: MemAvailable {avail:.1f} GiB < in-run floor "
                           f"{RUN_MEM_FLOOR_GIB} GiB (the launcher watchdog floor is "
                           f"{LAUNCHER_MEMGUARD['watchdog_floor_gib']} GiB)")
    # VRAM's 22-GiB minimum is a pre-load gate only: the engine itself reserves VRAM before READY.
    # ServeDriver.start() rechecks it before each arm; applying it to warm-up/request checks would reject
    # every valid loaded run.


def run(args) -> None:
    # ---- preflight (all of it BEFORE any engine process exists; nothing here loads a model)
    manifest = load_manifest(args)
    if args.reps != DEFAULT_REPS:
        raise ValueError(f"this benchmark is defined for exactly {DEFAULT_REPS} repetitions per cell; got {args.reps}")
    if manifest.get("reps_per_cell_per_arm") != args.reps:
        raise ValueError(f"manifest repetitions ({manifest.get('reps_per_cell_per_arm')}) do not match --reps {args.reps}; "
                         "rerun prepare with the required schedule")
    expected_cases = {f"pp{p}_out{o}.rep{rep}" for p, o in GRID for rep in range(1, args.reps + 1)}
    manifest_cases = {r.get("case") for r in manifest.get("requests", [])}
    if manifest_cases != expected_cases or len(manifest.get("requests", [])) != len(expected_cases):
        raise ValueError("manifest must contain exactly the six requested grid cells, each repeated three times")
    args.sidecar = args.sidecar or (args.model.parent / SIDECAR_NAME)
    if not args.sidecar.is_file():
        raise FileNotFoundError(f"DSpark sidecar missing: {args.sidecar}")
    launcher = args.repo / LAUNCHER_REL
    if not launcher.is_file():
        raise FileNotFoundError(f"launcher missing: {launcher}")
    for path in (args.model, args.rank):
        if not path.is_file():
            raise FileNotFoundError(f"required model artifact missing: {path}")
    for d in (Path(p) for p in args.engram.split(":")):
        if not (d / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Engram index missing: {d / 'model.safetensors.index.json'}")
    binary = args.repo / "build-xolotl/ds41_generate"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"native engine binary not ready: {binary}")
    # the manifest's own integrity (fail before spending a load on it)
    verify_fixtures(manifest)
    # the host must be able to take a load at all
    if args.physical_gpu is not None:
        free = free_vram_mib(gpu_cards(), args.physical_gpu)
        if free < args.min_free_vram_mib:
            raise RuntimeError(
                f"refusing load: GPU{args.physical_gpu} has only {free} MiB free; need {args.min_free_vram_mib} MiB")
    avail = read_mem_available_gib()
    if avail < args.min_mem_available_gib:
        raise RuntimeError(f"refusing load: MemAvailable {avail:.1f} GiB < {args.min_mem_available_gib} GiB "
                           f"(pre-load floor)")
    if port_busy(8188) and not args.allow_comfy:
        raise RuntimeError("ComfyUI is listening on :8188; the launcher would refuse (stop it or pass --allow-comfy)")

    arms = list(DEFAULT_ARMS) if args.arm == "both" else [args.arm]
    results_path = args.outdir / "results_ab.json"
    if results_path.is_file():
        backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = args.outdir / f"results_ab.{backup_stamp}.json.prev"
        if backup_path.exists():
            raise FileExistsError(f"refusing to replace existing backup: {backup_path}")
        results_path.rename(backup_path)
    results = {
        "created_utc": utc_now(),
        "harness": str(Path(__file__).resolve()),
        "manifest": str(args.outdir / "prompts_ab.json"),
        "config": {
            "repo": str(args.repo), "model": str(args.model), "rank": str(args.rank),
            "engram": args.engram, "sidecar": str(args.sidecar),
            "ctx": args.ctx, "arena_gib": None,
            "prompt_batch": args.prompt_batch if args.prompt_batch > 0 else "engine default (8192 in serve)",
            "prefill_chunk": args.prefill_chunk if args.prefill_chunk > 0 else "engine default (512 with a layer-major batch)",
            "route_bias": "launcher/engine defaults (not overridden by this harness)",
            "serve_park": 0,
            "physical_gpu": args.physical_gpu, "cuda_visible_devices": "the engine has no device flag; "
                              "the launcher is started as-is (CUDA_VISIBLE_DEVICES is inherited)",
            "reps": args.reps, "warmup_per_arm": len(manifest["warmup_requests"]),
            "launcher": str(args.repo / LAUNCHER_REL),
            "memguard": f"local launcher scope (MemoryMax={LAUNCHER_MEMGUARD['max_gib']}G, "
                        f"MemorySwapMax={LAUNCHER_MEMGUARD['swap_gib']}) + shared flock + host watchdog "
                        f"floor {LAUNCHER_MEMGUARD['watchdog_floor_gib']} GiB",
            "memguard_max_gib": LAUNCHER_MEMGUARD["max_gib"],
            "watchdog_floor_gib": LAUNCHER_MEMGUARD["watchdog_floor_gib"],
            "preload_mem_floor_gib": args.min_mem_available_gib,
            "run_mem_floor_gib": RUN_MEM_FLOOR_GIB,
            "min_mem_available_gib": args.min_mem_available_gib,
            "min_free_vram_mib": args.min_free_vram_mib,
            "initial_mem_available_gib": round(read_mem_available_gib(), 3),
            "initial_swap_gib": dict(zip(("total", "free"), read_swap_gib())),
            "initial_nvidia_smi": gpu_snapshot(),
        },
        "arms": {a: {"warmup": [], "results": [], "engine": {}} for a in arms},
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if args.physical_gpu == 1 else str(args.physical_gpu)

    sampler_stop = False
    import threading

    resources_path = args.outdir / "resources_ab.csv"
    if not resources_path.exists():
        resources_path.write_text("utc,arm,mem_available_gib,swap_used_gib,gpu_snapshot\n")

    def sample_resources():
        while not sampler_stop:
            try:
                used = read_swap_gib()
                with resources_path.open("a") as f:
                    f.write(f'"{utc_now()}","{current_arm[0]}",{read_mem_available_gib():.3f},'
                            f'{(used[0]-used[1]):.3f},"{gpu_snapshot().replace(chr(10), " | ")}"\n')
            except Exception as exc:                              # noqa: BLE001
                with resources_path.open("a") as f:
                    f.write(f'"{utc_now()}","{current_arm[0]}",,,"sample_error:{exc}"\n')
            for _ in range(args.sample_seconds * 10):
                if sampler_stop:
                    return
                time.sleep(0.1)

    current_arm = [arms[0]]
    sampler = threading.Thread(target=sample_resources, daemon=True)
    sampler.start()
    try:
        for arm in arms:
            current_arm[0] = arm
            run_arm(args, manifest, arm, results)
    finally:
        sampler_stop = True
        sampler.join(timeout=5)
        summarize(args, results)
        print(f"RESULTS: {results_path}; {args.outdir / 'summary_ab.json'}", flush=True)


# ---------------------------------------------------------------- aggregation

METRICS = ("prompt_tok_s", "decode_tok_s", "prompt_seconds", "decode_seconds", "wall_seconds")


def stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    out = {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals),
           "spread_abs": max(vals) - min(vals)}
    out["spread_rel_halfwidth"] = (max(vals) - min(vals)) / (2 * mean) if mean else None
    out["sd_rel"] = sd / mean if mean else None
    return out


def summarize(args, results: dict | None = None) -> dict:
    if results is None:
        path = args.outdir / "results_ab.json"
        if not path.is_file():
            raise FileNotFoundError(f"no results to summarize: {path}")
        results = json.loads(path.read_text())
    summary = {"created_utc": utc_now(), "source": str(args.outdir / "results_ab.json"),
               "reps": results["config"]["reps"], "arms": {},
               "rotation": "deterministic balanced rotation: generation g uses offset (2*g) mod 6, identical for "
                           "both arms (see prompts_ab.json schedule.generation_schedule)"}
    for arm, data in results["arms"].items():
        per_cell: dict[str, dict] = {}
        for r in data["results"]:
            cell = r["base_case"]
            d = per_cell.setdefault(cell, {"arm": arm, "case": cell, "prefill_tokens": r.get("prefill_tokens"),
                                           "max_new_tokens": r.get("max_new_tokens"), "runs": []})
            d["runs"].append(r)
        cells = {}
        for cell, d in per_cell.items():
            runs = [r for r in d["runs"] if r.get("accepted")]
            failed = [r["case"] for r in d["runs"] if not r.get("accepted")]
            metrics = {m: stats([r.get(m) for r in runs]) for m in METRICS}
            metrics["dspark_tokens_per_pass"] = stats([r.get("dspark_tokens_per_pass") for r in runs])
            metrics["decode_expert_file"] = stats([r.get("decode_expert_file") for r in runs])
            metrics["prompt_batch_width"] = stats([r.get("prompt_batch_width") for r in runs])
            cells[cell] = {
                "prefill_tokens": d["prefill_tokens"], "max_new_tokens": d["max_new_tokens"],
                "runs_accepted": len(runs), "runs_failed": failed, "metrics": metrics,
                "ids_sha256": sorted({r["ids_sha256"] for r in d["runs"]}),
                "output_ids_sha256": sorted({r["output_ids_sha256"] for r in d["runs"]}),
                "outputs_identical_within_cell": len({r["output_ids_sha256"] for r in d["runs"]}) == 1,
            }
        summary["arms"][arm] = {
            "engine": {k: v for k, v in data["engine"].items() if k in
                       ("load_seconds", "ready_line", "exit_code", "notes", "stdout_log", "stderr_log",
                        "mem_available_before_gib", "mem_available_after_gib", "gpu_before", "gpu_after",
                        "dspark_sidecars_loaded", "memguard_scope_line", "scope", "dspark_intent", "dspark_state")},
            "cells": cells,
        }
    # cross-arm delta on the matched cells
    if len(summary["arms"]) == 2:
        a, b = DEFAULT_ARMS
        if a in summary["arms"] and b in summary["arms"]:
            deltas = {}
            for cell in summary["arms"][a]["cells"]:
                if cell not in summary["arms"][b]["cells"]:
                    continue
                ca = summary["arms"][a]["cells"][cell]["metrics"]
                cb = summary["arms"][b]["cells"][cell]["metrics"]
                deltas[cell] = {}
                for m in ("prompt_tok_s", "decode_tok_s"):
                    if ca[m].get("n") and cb[m].get("n"):
                        deltas[cell][m] = {
                            "mtp_mean": ca[m]["mean"], "nomtp_mean": cb[m]["mean"],
                            "ratio_mtp_over_nomtp": ca[m]["mean"] / cb[m]["mean"],
                            "pct": 100.0 * (ca[m]["mean"] / cb[m]["mean"] - 1.0),
                        }
            summary["delta_mtp_vs_nomtp"] = deltas
    (args.outdir / "summary_ab.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (args.outdir / "results_ab.csv").open("w", newline="") as f:
        cols = ["case", "base_case", "arm", "generation", "cell_index_in_generation", "accepted", "problems",
                "engine_prompt_tokens", "engine_prompt_reused", "engine_prompt_path", "prompt_tok_s",
                "prompt_seconds", "decode_tok_s", "decode_seconds", "prompt_expert_file", "decode_expert_file",
                "decode_vram_hit_pct", "prompt_batch_width", "dspark_passes", "dspark_tokens_per_pass",
                "generated_tokens_emitted", "wall_seconds", "ids_sha256", "output_ids_sha256"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for arm, data in results["arms"].items():
            for r in data["results"] + data["warmup"]:
                row = dict(r)
                row["problems"] = "; ".join(r.get("problems", []))
                w.writerow(row)

    with (args.outdir / "summary_ab.csv").open("w", newline="") as f:
        cols = ["arm", "case", "prefill_tokens", "max_new_tokens", "runs_accepted", "metric", "n", "mean", "sd",
                "min", "max", "spread_abs", "spread_rel_halfwidth", "sd_rel"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, data in summary["arms"].items():
            for cell, c in data["cells"].items():
                for metric, s in c["metrics"].items():
                    if not s.get("n"):
                        continue
                    w.writerow({"arm": arm, "case": cell, "prefill_tokens": c["prefill_tokens"],
                                "max_new_tokens": c["max_new_tokens"], "runs_accepted": c["runs_accepted"],
                                "metric": metric, "n": s["n"], "mean": f"{s['mean']:.4f}",
                                "sd": f"{s['sd']:.4f}", "min": f"{s['min']:.4f}", "max": f"{s['max']:.4f}",
                                "spread_abs": f"{s['spread_abs']:.4f}",
                                "spread_rel_halfwidth": f"{s['spread_rel_halfwidth']:.5f}"
                                if s["spread_rel_halfwidth"] is not None else "",
                                "sd_rel": f"{s['sd_rel']:.5f}" if s["sd_rel"] is not None else ""})
    write_markdown(args.outdir / "SUMMARY_AB.md", summary)
    return summary


def write_markdown(path: Path, summary: dict) -> None:
    lines = ["# DeepSeek V4.1 — persistent-server repeated MTP A/B", "",
             f"Generated {summary['created_utc']} from `results_ab.json`; {summary['reps']} generations per cell per arm.", ""]
    for arm, data in summary["arms"].items():
        e = data["engine"]
        lines += [f"## Arm `{arm}`", "",
                  f"- engine: one load, {e.get('load_seconds')} s; ready line `{e.get('ready_line')}`; exit {e.get('exit_code')}",
                  f"- DSpark sidecars loaded: {e.get('dspark_sidecars_loaded')}",
                  "", "| prefill | out | n | prefill tok/s mean ± sd (min–max) | decode tok/s mean ± sd (min–max) | "
                  "wall s mean | DSpark tok/pass |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for cell, c in data["cells"].items():
            m = c["metrics"]
            def f(metric, fmt="{:.2f}"):
                s = m.get(metric, {})
                if not s.get("n"):
                    return "–"
                return (fmt.format(s["mean"]) + " ± " + fmt.format(s["sd"]) + " (" +
                        fmt.format(s["min"]) + "–" + fmt.format(s["max"]) + ")")
            lines.append(f"| {c['prefill_tokens']} | {c['max_new_tokens']} | {c['runs_accepted']} | "
                         f"{f('prompt_tok_s')} | {f('decode_tok_s')} | {f('wall_seconds')} | "
                         f"{f('dspark_tokens_per_pass', '{:.3f}')} |")
        lines.append("")
    if "delta_mtp_vs_nomtp" in summary:
        lines += ["## DSpark on vs off (mean of the accepted runs)", "",
                  "| prefill | out | prefill tok/s mtp / nomtp | Δ mtp vs nomtp | decode tok/s mtp / nomtp | Δ mtp vs nomtp |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for cell, d in summary["delta_mtp_vs_nomtp"].items():
            c = summary["arms"][DEFAULT_ARMS[0]]["cells"][cell]
            p, q = d.get("prompt_tok_s"), d.get("decode_tok_s")

            def pair(x):
                return f"{x['mtp_mean']:.2f} / {x['nomtp_mean']:.2f}" if x else "–"

            def delta(x):
                return f"{x['pct']:+.1f}%" if x else "–"

            lines.append(f"| {c['prefill_tokens']} | {c['max_new_tokens']} | {pair(p)} | {delta(p)} | "
                         f"{pair(q)} | {delta(q)} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- CLI

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("prepare", "run", "summarize"))
    p.add_argument("--arm", choices=("mtp", "nomtp", "both"), default="both")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--warmup-repeats", type=int, default=2,
                   help="warm-up requests per arm (the same prompt: exercises the engine's state reuse)")
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    p.add_argument("--outdir", type=Path, default=HERE)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_DEFAULT)
    p.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    p.add_argument("--rank", type=Path, default=RANK_DEFAULT)
    p.add_argument("--engram", default=":".join(map(str, ENGRAM_DEFAULT)))
    p.add_argument("--sidecar", type=Path, default=None,
                   help=f"default: <model dir>/{SIDECAR_NAME}")
    p.add_argument("--physical-gpu", type=int, default=1)
    p.add_argument("--min-free-vram-mib", type=int, default=22000)
    p.add_argument("--min-mem-available-gib", type=int, default=70)
    p.add_argument("--ctx", type=int, default=8192)
    # 0 = leave the engine's own defaults alone (the one-shot grid did the same: it must stay comparable)
    p.add_argument("--prompt-batch", type=int, default=0)
    p.add_argument("--prefill-chunk", type=int, default=0)
    p.add_argument("--sample-seconds", type=int, default=5)
    p.add_argument("--shutdown-timeout", type=int, default=120)
    p.add_argument("--force", action="store_true", help="prepare: rewrite prompts_ab/ and the manifest")
    p.add_argument("--allow-failures", action="store_true",
                   help="run: record a failed request and continue instead of aborting the run")
    p.add_argument("--allow-comfy", action="store_true",
                   help="run: do not pre-check :8188 (the launcher still refuses unless MEMGUARD_ALLOW_COMFY=1)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    else:
        summary = summarize(args)
        for arm, data in summary["arms"].items():
            for cell, c in data["cells"].items():
                m = c["metrics"]["decode_tok_s"]
                p = c["metrics"]["prompt_tok_s"]
                print(f"{arm:6s} {cell:16s} n={c['runs_accepted']} "
                      f"prompt {p.get('mean', float('nan')):.2f} tok/s (+-{p.get('sd', 0):.2f}) "
                      f"decode {m.get('mean', float('nan')):.2f} tok/s (+-{m.get('sd', 0):.2f})")


if __name__ == "__main__":
    main()
