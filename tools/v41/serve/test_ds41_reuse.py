#!/usr/bin/env python3
"""Real line-protocol KV regression on make_tiny_v41's model (no GPU by default).

Run: python tools/v41/serve/test_ds41_reuse.py --engine build/ds41_generate
Uses real experts, Engram, rolling attention and ratio-1/2 compressed caches, both independent
and shared KV/index source layouts. The engine is never mocked. Temporary files are removed.
"""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_tiny_v41 import build


class Engine:
    def __init__(self, exe, model, directory, *, park=False, stop=False):
        self.directory = directory
        self.number = 0
        self.log = (directory / "engine.log").open("w")
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
        env.update(STRATA_SERVE_PARK=str(int(park)), STRATA_SERVE_PARK_MIN="1",
                   STRATA_SERVE_PARK_MAX="512", STRATA_SERVE_PASS_BIAS="0",
                   STRATA_SERVE_DUMP=str(directory))
        args = [str(exe), "-m", str(model), "--backend", "cpu", "--experts", "cpu",
                "--ctx", "512", "--threads", "2", "--prefill-chunk", "16",
                "--prompt-batch", "32", "--route-bias", "0", "--route-bias-file", "0",
                "--vision", "--image-token", "200", "--serve"]
        if stop:
            args += ["--stop", ",".join(map(str, range(256)))]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, env=env, bufsize=1)
        while not self.line().startswith("READY "):
            pass

    def line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError("engine exited: " + (self.directory / "engine.log").read_text()[-4000:])
        if line.startswith("ERR "):
            raise AssertionError(line)
        return line.strip()

    def request(self, ids, *, count=8, image=None, cancel=False):
        command = f"GENI {count} temperature=0 {image} " if image else f"GEN {count} temperature=0 "
        self.proc.stdin.write(command + ",".join(map(str, ids)) + "\n" + ("STOP\n" if cancel else ""))
        self.proc.stdin.flush()
        tokens, reused = [], 0
        while True:
            line = self.line()
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
            elif line.startswith("RESUME "):
                reused = int(line.split()[1])
            elif line.startswith("DONE "):
                fields = line.split()
                if reused != int(fields[8]):
                    raise AssertionError(("RESUME/DONE disagree", line, reused))
                raw = (self.directory / f"req{self.number:03d}.f32").read_bytes()
                self.number += 1
                return tokens, struct.unpack(f"<{len(raw) // 4}f", raw), reused, fields[5]

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.log.close()


def require(ok, detail):
    if not ok:
        raise AssertionError(detail)


def same(got, cold, label, *, exact=False):
    require(got[0] == cold[0], (label, "greedy output differs", got[0], cold[0]))
    require(len(got[1]) == len(cold[1]) > 0, (label, "missing logits"))
    for a, b in zip(got[1], cold[1]):
        require(math.isfinite(a) and math.isfinite(b), (label, "nonfinite logits"))
        require(a == b if exact else abs(a - b) <= 1e-4 + 1e-4 * abs(b),
                (label, "logits differ", a, b))


def exercise(exe, root, *, shared_sources=False):
    model = root / "tiny.gguf"
    build(str(model), 8 if shared_sources else 4, 20261021, shared_sources=shared_sources)
    reset_image = root / "reset.sve"
    reset_image.write_bytes(struct.pack("<5i", 0x31455653, 2, 2, 1, 256) + bytes(512 * 4))
    for park in (False, True):
        directory = root / ("park" if park else "live")
        directory.mkdir()
        engine = Engine(exe, model, directory, park=park)
        try:
            def cold(ids, **kw):
                # A real image request invalidates every text checkpoint and parked entry.
                engine.request([200, 200], image=reset_image, count=1)
                result = engine.request(ids, **kw)
                require(result[2] == 0, ("cold request unexpectedly reused", result[2]))
                return result

            for length in (15, 16, 17, 31, 32, 33):
                prompt = [10 + i % 170 for i in range(length - 1)] + [199]
                first = cold(prompt, count=24)
                regen = engine.request(prompt, count=24)
                require(regen[2] == length, ("regeneration prefills", length, regen[2]))
                same(regen, first, "regeneration", exact=True)
                # Cross both the 16-row dense span and 32-row prompt-batch boundaries after a rewind.
                suffix_length = (2, 17, 35)[length % 3]
                rewritten = prompt[:-1] + [198] + [51 + i % 60 for i in range(suffix_length)]
                if park:
                    # Overwrite even the parked prefix's final Engram look-back tokens, not just position zero.
                    engine.request([249] * (length + 4), count=2)
                reused = engine.request(rewritten)
                require(reused[2] == length - 1, ("rewritten header rereads prefix", length, reused[2]))
                same(reused, cold(rewritten), "rewritten header")
            prompt = list(range(30, 63))
            generated = cold(prompt, count=4)
            extended = prompt + generated[0] + [71, 72]
            continued = engine.request(extended)
            require(continued[2] >= len(prompt), ("extension did not reuse", continued[2]))
            same(continued, cold(extended), "extension")

            cold(prompt)
            cancelled = engine.request(prompt, count=128, cancel=True)
            require(cancelled[3] == "cancel", ("STOP not consumed", cancelled[3]))
            retry = engine.request(prompt)
            require(retry[2] == len(prompt), "cancelled request lost prompt snapshot")
            same(retry, cold(prompt), "cancel then regenerate")

            changed = prompt[:12] + [190] + prompt[13:]
            invalidated = engine.request(changed)
            require(invalidated[2] == 0, "rewind before captured prefix reused stale rings")
            same(invalidated, cold(changed), "changed earlier prefix")

            image_ids = [21, 200, 200, 22]
            for value in (0.125, -0.125):
                image = root / f"image{value}.sve"
                image.write_bytes(struct.pack("<5i", 0x31455653, 2, 2, 1, 256)
                                  + struct.pack("<512f", *([value] * 512)))
                result = engine.request(image_ids, image=image)
                require(result[2] == 0, "image identity matched by token ids only")
                same(result, cold(image_ids, image=image), "changed image")
            text = engine.request(image_ids)
            require(text[2] == 0, "text request reused image embeddings")
            same(text, cold(image_ids), "image to text")
        finally:
            engine.close()

    directory = root / "stop"
    directory.mkdir()
    engine = Engine(exe, model, directory, stop=True)
    try:
        prompt = list(range(30, 63))
        first = engine.request(prompt)
        second = engine.request(prompt)
        require(first[3] == second[3] == "stop" and len(first[0]) == 1, "stop-token case did not stop")
        require(second[2] == len(prompt), "stop token invalidated prompt snapshot")
        same(second, first, "stop then regenerate", exact=True)
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ds41-reuse-") as directory:
        for shared in (False, True):
            root = Path(directory) / ("shared" if shared else "independent")
            root.mkdir()
            exercise(args.engine.resolve(), root, shared_sources=shared)
    print("PASS: real CPU KV reuse with independent/shared sources, rewritten headers, regeneration, extension, STOP, images and parking")


if __name__ == "__main__":
    main()
