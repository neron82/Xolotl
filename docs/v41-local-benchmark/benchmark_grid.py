#!/usr/bin/env python3
"""Reproducible one-GPU DeepSeek V4.1 prefill/decode grid using the native one-shot engine."""
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

REPO = Path(__file__).resolve().parents[2]
MODEL_DEFAULT = Path.home() / "models/Deepseek-V41-GSQ/V41-GSQ.gguf"
RANK_DEFAULT = Path.home() / "models/Deepseek-V41-GSQ/rank.bin"
ENGRAM_DEFAULT = (Path.home() / "models/Deepseek-FP8-engram/layers.1", Path.home() / "models/Deepseek-FP8-engram/layers.14")
TOKENIZER_DEFAULT = MODEL_DEFAULT.parent / "tokenizer"
SOURCE_FILES = (
    REPO / "README.md",
    REPO / "docs/DEEPSEEK_V41.md",
    REPO / "docs/xolotl/BENCHMARKS.md",
    REPO / "docs/MULTI_GPU.md",
    REPO / "tools/v41/hf/README.md",
)
GRID = [(1024, 128), (1024, 256), (2048, 128), (2048, 256), (4096, 128), (4096, 256)]
WARMUP = (1024, 32)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tokenizer(repo: Path, tokdir: Path):
    sys.path[:0] = [str(repo / "tools"), str(repo / "tools/v41")]
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


def prepare(args) -> dict:
    tokenizer, encoding = load_tokenizer(args.repo, args.tokenizer)
    sources = [(path, clean_source(path)) for path in SOURCE_FILES]
    sources = [(path, text) for path, text in sources if text]
    if not sources:
        raise RuntimeError("no benchmark source documents found")
    corpus = "\n\n".join(f"Source {path.name}:\n{text}" for path, text in sources)
    corpus_ids = tokenizer.encode(corpus, parse_special=False)
    if len(corpus_ids) < 18000:
        raise RuntimeError(f"benchmark source corpus too short: {len(corpus_ids)} tokens, expected >=18000")

    args.outdir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.outdir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    requests = []
    jobs = [("warmup", WARMUP[0], WARMUP[1], "Obsidian")]
    names = ["Quartz", "Juniper", "Saffron", "Tangent", "Cobalt", "Marigold"]
    for (prefill, output), name in zip(GRID, names):
        jobs.append((f"pp{prefill}_out{output}", prefill, output, name))

    for case, prefill, output, label in jobs:
        span_len = prefill + 128
        if cursor + span_len > len(corpus_ids):
            raise RuntimeError(f"source pool exhausted preparing {case}: need {cursor + span_len}, have {len(corpus_ids)}")
        source = tokenizer.decode(corpus_ids[cursor:cursor + span_len])
        cursor += span_len + 16
        ids = encode_prompt(tokenizer, encoding, prefill, label, source)
        raw = struct.pack(f"<{len(ids)}i", *ids)
        path = prompt_dir / f"{case}.i32"
        path.write_bytes(raw)
        requests.append({
            "case": case,
            "phase": "warmup" if case == "warmup" else "measure",
            "prefill_tokens": prefill,
            "max_new_tokens": output,
            "ids_file": str(path),
            "ids_sha256": sha256(raw),
            "prompt_id_count": len(ids),
            "first_ids": ids[:12],
            "last_ids": ids[-8:],
        })

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(args.repo),
        "tokenizer_dir": str(args.tokenizer),
        "tokenizer_vocab": len(tokenizer.tokens),
        "source_files": [str(path) for path, _ in sources],
        "source_corpus_tokens": len(corpus_ids),
        "source_corpus_sha256": sha256(corpus.encode("utf-8")),
        "prompt_template": "DeepSeek V4.1 chat template; exhaustive response; greedy decode",
        "requests": requests,
    }
    manifest_path = args.outdir / "prompts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "source_tokens": len(corpus_ids),
                      "requests": [{"case": r["case"], "prefill": r["prefill_tokens"],
                                    "output": r["max_new_tokens"], "ids_sha256": r["ids_sha256"]}
                                   for r in requests]}, indent=2))
    return manifest


def read_mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    raise RuntimeError("could not read MemAvailable from /proc/meminfo")


def gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def parse_run(stderr: str, stdout: str, requested_new: int, prompt_tokens: int) -> dict:
    p = re.search(r"prompt: (\d+) tokens in ([0-9.]+) s = ([0-9.]+) tok/s \(experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file\)", stderr)
    d = re.search(r"decode: (\d+) tokens in ([0-9.]+) s = ([0-9.]+) tok/s; experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file \(([0-9.]+)% VRAM hits\)", stderr)
    vram = re.search(r"VRAM after the prompt: ([0-9.]+) GiB free -> ([0-9.]+) GiB of expert slots: (\d+) experts in VRAM", stderr)
    tokens_out = [int(line) for line in stdout.splitlines() if re.fullmatch(r"\d+", line.strip())]
    if not p or not d:
        raise RuntimeError("could not parse native prompt/decode timing lines from stderr")
    prompt = int(p.group(1))
    generated = int(d.group(1))
    if prompt != prompt_tokens:
        raise RuntimeError(f"engine prompt count {prompt} does not match prepared count {prompt_tokens}")
    if generated != requested_new or len(tokens_out) != requested_new:
        raise RuntimeError(f"generated {generated} tokens ({len(tokens_out)} ids); requested {requested_new}")
    return {
        "prompt_tokens": prompt,
        "prompt_seconds": float(p.group(2)),
        "prompt_tok_s": float(p.group(3)),
        "prompt_expert_hits": int(p.group(4)),
        "prompt_expert_cpu": int(p.group(5)),
        "prompt_expert_pcie": int(p.group(6)),
        "prompt_expert_file": int(p.group(7)),
        "generated_tokens": generated,
        "decode_seconds": float(d.group(2)),
        "decode_tok_s": float(d.group(3)),
        "decode_expert_hits": int(d.group(4)),
        "decode_expert_cpu": int(d.group(5)),
        "decode_expert_pcie": int(d.group(6)),
        "decode_expert_file": int(d.group(7)),
        "decode_vram_hit_pct": float(d.group(8)),
        "vram_after_prompt_free_gib": float(vram.group(1)) if vram else None,
        "vram_expert_slots_gib": float(vram.group(2)) if vram else None,
        "vram_experts_resident": int(vram.group(3)) if vram else None,
        "output_ids_sha256": sha256(struct.pack(f"<{len(tokens_out)}i", *tokens_out)),
    }


def run_grid(args) -> None:
    manifest_path = args.outdir / "prompts.json"
    manifest = json.loads(manifest_path.read_text())
    binary = args.repo / "build-xolotl/ds41_generate"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"native engine binary not ready: {binary}")
    for path in (args.model, args.rank):
        if not path.is_file():
            raise FileNotFoundError(f"required model artifact missing: {path}")
    dspark_path = args.model.parent / "V41-MTP-Vision.Q2_K.gguf"
    if not dspark_path.is_file():
        raise FileNotFoundError(f"recommended DSpark sidecar missing: {dspark_path}")
    engram_dirs = [Path(p) for p in args.engram.split(":")]
    for path in engram_dirs:
        if not (path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Engram index missing: {path / 'model.safetensors.index.json'}")

    mem_available = read_mem_available_gib()
    if mem_available < args.min_mem_available_gib:
        raise RuntimeError(f"refusing load: MemAvailable {mem_available:.1f} GiB < {args.min_mem_available_gib} GiB")
    snapshot = gpu_snapshot()
    cards = [line.split(",") for line in snapshot.splitlines() if line.strip()]
    if len(cards) <= args.physical_gpu:
        raise RuntimeError(f"physical GPU index {args.physical_gpu} absent from nvidia-smi output: {snapshot}")
    selected = cards[args.physical_gpu]
    free_mib = int(selected[3].strip())
    if free_mib < args.min_free_vram_mib:
        raise RuntimeError(f"refusing load: GPU{args.physical_gpu} has only {free_mib} MiB free; need {args.min_free_vram_mib} MiB")

    results_path = args.outdir / "results.json"
    csv_path = args.outdir / "results.csv"
    sampler_stop = False
    resources_path = args.outdir / "resources.csv"
    if not resources_path.exists():
        resources_path.write_text("utc,mem_available_gib,gpu_snapshot\n")

    def sample_resources():
        while not sampler_stop:
            try:
                now = datetime.now(timezone.utc).isoformat()
                avail = read_mem_available_gib()
                gpu = gpu_snapshot().replace("\n", " | ").replace('"', '""')
                with resources_path.open("a") as f:
                    f.write(f'"{now}",{avail:.3f},"{gpu}"\n')
            except Exception as e:
                with resources_path.open("a") as f:
                    f.write(f'"{datetime.now(timezone.utc).isoformat()}",,"sample_error:{e}"\n')
            for _ in range(args.sample_seconds * 10):
                if sampler_stop:
                    return
                time.sleep(0.1)

    import threading
    sampler = threading.Thread(target=sample_resources, daemon=True)
    sampler.start()
    results = []
    try:
        ordered = [r for r in manifest["requests"] if r["phase"] == "warmup"] + [
            r for r in manifest["requests"] if r["phase"] == "measure"]
        for idx, req in enumerate(ordered, 1):
            if read_mem_available_gib() < args.min_mem_available_gib:
                raise RuntimeError("MemAvailable fell below guard before next model load")
            run_gpu_before = gpu_snapshot()
            current_gpu = [line.split(",") for line in run_gpu_before.splitlines() if line.strip()]
            free_mib = int(current_gpu[args.physical_gpu][3].strip())
            if free_mib < args.min_free_vram_mib:
                raise RuntimeError(f"GPU{args.physical_gpu} free VRAM fell below guard: {free_mib} MiB")
            log_dir = args.outdir / "runs"
            log_dir.mkdir(exist_ok=True)
            stdout_path = log_dir / f"{req['case']}.stdout.log"
            stderr_path = log_dir / f"{req['case']}.stderr.log"
            output_ids_path = log_dir / f"{req['case']}.output.i32"
            command = [
                "/bin/bash", str(args.repo / "tools/ds4/memguard.sh"),
                str(args.memguard_max_gib), str(args.min_mem_available_gib), "--", "env",
                f"CUDA_VISIBLE_DEVICES={args.physical_gpu}",
                "STRATA_NO_LARGEPAGES=1", "STRATA_DENSE_REQUANT=",
                f"STRATA_DSPARK={dspark_path}", "STRATA_DSPARK_DEPTH=4",
                str(binary), "-m", str(args.model), "--backend", "cuda", "--experts", "gpu",
                "--engram", args.engram, "--rank", str(args.rank), "--ids-file", req["ids_file"],
                "-n", str(req["max_new_tokens"]), "--temp", "0", "--ctx", str(args.ctx),
                "--arena-gib", str(args.arena_gib), "--vram-lru",
                "--route-bias", str(args.route_bias), "--route-bias-file", str(args.route_bias_file),
            ]
            if args.prompt_batch > 0:
                command.extend(["--prompt-batch", str(args.prompt_batch)])
            if args.prefill_chunk > 0:
                command.extend(["--prefill-chunk", str(args.prefill_chunk)])
            print(f"[{idx}/{len(ordered)}] {req['case']}: {req['prefill_tokens']} prompt / {req['max_new_tokens']} output; loading on physical GPU{args.physical_gpu}", flush=True)
            start = time.monotonic()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.run_timeout)
            elapsed = time.monotonic() - start
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{req['case']} exited {completed.returncode}; see {stderr_path}; tail: {completed.stderr[-2000:]}")
            timing = parse_run(completed.stderr, completed.stdout, req["max_new_tokens"], req["prefill_tokens"])
            output_ids = [int(line.strip()) for line in completed.stdout.splitlines() if re.fullmatch(r"\d+", line.strip())]
            output_ids_path.write_bytes(struct.pack(f"<{len(output_ids)}i", *output_ids))
            result = {
                **{k: req[k] for k in ("case", "phase", "prefill_tokens", "max_new_tokens", "ids_file", "ids_sha256")},
                **timing,
                "wall_seconds_including_load": round(elapsed, 3),
                "physical_gpu": args.physical_gpu,
                "gpu_snapshot_before": run_gpu_before,
                "returncode": completed.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "output_ids_file": str(output_ids_path),
            }
            results.append(result)
            results_path.write_text(json.dumps({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "model": str(args.model), "rank": str(args.rank), "engram": args.engram,
                    "dspark": str(dspark_path), "dspark_depth": 4,
                    "cuda_visible_devices": str(args.physical_gpu), "ctx": args.ctx,
                    "arena_gib": args.arena_gib,
                    "prompt_batch": args.prompt_batch if args.prompt_batch > 0 else "engine default (8192; one-shot clamps to prompt length)",
                    "prefill_chunk": args.prefill_chunk if args.prefill_chunk > 0 else "engine default",
                    "kv_fp4": False, "temp": 0,
                    "route_bias": args.route_bias, "route_bias_file": args.route_bias_file,
                    "memguard_max_gib": args.memguard_max_gib,
                    "min_mem_available_gib": args.min_mem_available_gib,
                    "initial_mem_available_gib": mem_available,
                    "initial_nvidia_smi": snapshot,
                },
                "results": results,
            }, indent=2) + "\n")
            print(f"    prefill {timing['prompt_tok_s']:.1f} tok/s ({timing['prompt_seconds']:.2f}s); decode {timing['decode_tok_s']:.2f} tok/s ({timing['decode_seconds']:.2f}s); file-tier {timing['decode_expert_file']}; wall {elapsed:.1f}s", flush=True)
        write_csv(csv_path, results)
        print(f"GRID COMPLETE: {len([r for r in results if r['phase'] == 'measure'])}/6 measured cells; {results_path}; {csv_path}", flush=True)
    finally:
        sampler_stop = True
        sampler.join(timeout=3)


def write_csv(path: Path, results: list[dict]) -> None:
    rows = [r for r in results if r["phase"] == "measure"]
    columns = ["case", "prompt_tokens", "generated_tokens", "prompt_tok_s", "decode_tok_s",
               "prompt_seconds", "decode_seconds", "prompt_expert_hits", "prompt_expert_cpu",
               "prompt_expert_pcie", "prompt_expert_file", "decode_expert_hits", "decode_expert_cpu",
               "decode_expert_pcie", "decode_expert_file", "decode_vram_hit_pct",
               "vram_after_prompt_free_gib", "vram_expert_slots_gib", "vram_experts_resident",
               "wall_seconds_including_load", "ids_sha256", "output_ids_sha256"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--rank", type=Path, default=RANK_DEFAULT)
    parser.add_argument("--engram", default=":".join(map(str, ENGRAM_DEFAULT)))
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--min-free-vram-mib", type=int, default=22000)
    parser.add_argument("--min-mem-available-gib", type=int, default=70)
    parser.add_argument("--memguard-max-gib", type=int, default=84)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--arena-gib", type=float, default=60.0)
    parser.add_argument("--prompt-batch", type=int, default=0)
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--route-bias", type=float, default=1.0)
    parser.add_argument("--route-bias-file", type=float, default=0.6)
    parser.add_argument("--sample-seconds", type=int, default=5)
    parser.add_argument("--run-timeout", type=int, default=1200)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
