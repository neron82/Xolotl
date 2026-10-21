"""Compare EXL3 and native experts with the same engine protocol, fixtures and GPU.

One load, one GEN per binary int32 fixture. Rates exclude reused prompt tokens and
the first generated token (which prefill already computed). Results include the
command, relevant environment, executable hash, token IDs and raw engine log.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=Path("build-xolotl/ds41_generate"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--exl3-dir", default="", help="omit for the native GGUF comparison")
    ap.add_argument("--rank", default="")
    ap.add_argument("--gpu", required=True, help="physical CUDA device index (GPU1 on the local x16 machine)")
    ap.add_argument("--ids-files", required=True, help="comma-separated raw little-endian int32 fixtures")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--arena-gib", type=float, default=60)
    ap.add_argument("--prompt-batch", type=int, default=8192)
    ap.add_argument("--fast", action="store_true", help="enable STRATA_EXL3_FAST=1")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds without an engine response")
    ap.add_argument("--out", type=Path, required=True, help="result JSON (engine log uses the same stem)")
    args = ap.parse_args()
    if args.gen < 1 or args.ctx < 1 or args.prompt_batch < 1:
        ap.error("gen, ctx and prompt-batch must be positive")
    fixtures = []
    for name in args.ids_files.split(","):
        path = Path(name.strip())
        raw = path.read_bytes()
        if not raw or len(raw) % 4:
            ap.error(f"{path}: expected a nonempty int32 fixture")
        ids = struct.unpack(f"<{len(raw) // 4}i", raw)
        fixtures.append((path, ids[:args.limit] if args.limit > 0 else ids))
    exe = args.exe.resolve()
    settings = dict(CUDA_VISIBLE_DEVICES=args.gpu, STRATA_NO_LARGEPAGES="1", STRATA_DENSE_REQUANT="q6_K",
                    STRATA_SERVE_STEP_MAX="96", STRATA_SERVE_PASS_BIAS="0", STRATA_PRESTAGE_GIB="1.5",
                    STRATA_SERVE_PARK="0", STRATA_EXL3_FAST="1" if args.fast else "0")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "DS41_", "DS4_"))}
    env.update(settings)
    cmd = [str(exe), "-m", args.model, "--serve", "--backend", "cuda", "--experts", "gpu",
           "--ctx", str(args.ctx), "--arena-gib", str(args.arena_gib), "--prompt-batch", str(args.prompt_batch),
           "--kv-fp4", "--route-bias", "0.5", "--route-bias-file", "0.3", "--arena-adapt"]
    if args.rank:
        cmd += ["--rank", args.rank]
    if args.exl3_dir:
        cmd += ["--exl3-dir", args.exl3_dir]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(argv=cmd, env=settings, executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), runs=[])
    with args.out.with_suffix(".engine.log").open("w") as log:
        start = time.monotonic()
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        pending = bytearray()

        def readline():
            while b"\n" not in pending:
                if not selector.select(args.timeout):
                    raise TimeoutError("engine response timed out; see the engine log")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("engine closed its output; see the engine log")
                pending.extend(chunk)
            line, _, rest = pending.partition(b"\n")
            pending[:] = rest
            return line.decode()

        def send(line):
            proc.stdin.write((line + "\n").encode())
            proc.stdin.flush()

        try:
            while not readline().startswith("READY"):
                pass
            result["load_s"] = time.monotonic() - start
            for path, ids in fixtures:
                start = time.monotonic()
                send(f"GEN {args.gen} " + ",".join(map(str, ids)))
                tokens = []
                while True:
                    fields = readline().split()
                    if fields and fields[0] == "T":
                        tokens.append(int(fields[1]))
                    if fields and fields[0] == "ERR":
                        raise RuntimeError(" ".join(fields))
                    if fields and fields[0] == "DONE":
                        break
                generated, total = int(fields[1]), int(fields[2])
                pp_ms, tg_ms = float(fields[3]), float(fields[4])
                reused = int(fields[8]) if len(fields) > 8 else 0
                row = dict(fixture=str(path), prompt_tokens=total, reused=reused, generated=generated,
                           prompt_ms=pp_ms, decode_ms=tg_ms, finish=fields[5], tokens=tokens,
                           prompt_tok_s=(total - reused) * 1000 / pp_ms if pp_ms else None,
                           decode_tok_s=max(0, generated - 1) * 1000 / tg_ms if tg_ms else None,
                           wall_s=time.monotonic() - start)
                result["runs"].append(row)
                args.out.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
            send("QUIT")
            proc.wait(timeout=120)
            if proc.returncode:
                raise RuntimeError(f"engine exited {proc.returncode}; see the engine log")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            selector.close()


if __name__ == "__main__":
    main()
