# DeepSeek V4.1 — persistent-server, repeated MTP A/B benchmark

This directory holds the **one-shot** grid (`benchmark_grid.py`, `prompts.json`, `results.json`, `REPORT.md`)
and the **persistent-server repeated A/B** harness added afterwards (`benchmark_server_ab.py`).

The one-shot grid loaded the model once per cell (~48–50 s a load, 792 s total). The user asked for every cell
repeated 3× with mean and spread, and explicitly for the persistent server (one load per arm) under a 110 GiB
memguard approach. This harness does that, using its own local launcher (`run_engine_fg.sh`) because the tracked
`ds41-engine.sh` hard-codes an 84 GiB cap.

## Artifacts

| File | Role |
|---|---|
| `benchmark_server_ab.py` | The harness: `prepare` (write the repeated-prompt manifest), `run` (`--arm mtp` / `--arm nomtp` / `--arm both`), `summarize`. |
| `run_engine_fg.sh` | **Local foreground launcher** used by the harness instead of `tools/v41/serve/ds41-engine.sh`: same DSpark env contract, same `:8188` Comfy refusal, same shared flock, but **`MemoryMax=110G`** (the tracked launcher hard-codes 84G, below the required cap), a **live host-MemAvailable watchdog** (floor 4 GiB), and `MemorySwapMax=0`. Prints a `memguard: scope …` line the harness verifies. |
| `fake_protocol_test.py` | No-model end-to-end test: builds a throwaway fake repo + fake `ds41_generate`, drives the real harness through both arms, and asserts the DONE parser, reuse guards, scope and DSpark state. |
| `prompts_ab.json` | Manifest of the repeated prompt set: per-case id file paths, SHA-256, engine-counted prompt length, the rotation schedule, the fixture-uniqueness proof, and the two arm definitions. Written by `prepare`. |
| `prompts_ab/` | Token-id files: one shared warm-up fixture (`warmup.i32`) plus the 18 measured fixtures (`<case>.rep<N>.i32`). |
| `results_ab.json` | Per-run results for every arm (raw, one row per request), atomically rewritten after each request so a killed run keeps its last complete snapshot. |
| `results_ab.csv` | One row per measured request. |
| `summary_ab.json` / `summary_ab.csv` / `SUMMARY_AB.md` | Per arm → per cell: n, mean, sample stdev, min, max, absolute spread, and the relative half-spread `(max-min)/(2*mean)` for prefill tok/s, decode tok/s, prompt s, decode s, wall s; plus cross-arm deltas. Written by `run`. |
| `runs_ab/` | Raw per-arm logs: `<stamp>.<arm>.server.stdout.log`, `.server.stderr.log`, `.trace.jsonl`. |
| `resources_ab.csv` | Host MemAvailable + `nvidia-smi` samples taken by the harness during the run. |

## Method

- **One engine load per arm.** The engine is started once in `--serve` mode (via the local `run_engine_fg.sh`,
  which holds the shared GPU flock inside a `MemoryMax=110G` / `MemorySwapMax=0` scope with a host watchdog) and
  stays up for the whole arm. Only two loads happen in a `--arm both` run. `STRATA_SERVE_DSPARK` is a load-time
  option, so the arm switch requires a restart; nothing else differs between the arms.
- **The DSpark switch is explicit.** `ds41-engine.sh` (and the local launcher, which keeps its contract) defaults
  an **unset** `STRATA_SERVE_DSPARK` to Mal's sidecar path. The harness therefore passes it explicitly in both
  arms: the **MTP-off** arm passes an **empty** value (the launcher *unsets* it, so the engine runs without
  drafts), and the **MTP-on** arm pins the **local Q2_K sidecar** `<model dir>/V41-MTP-Vision.Q2_K.gguf`. The
  launcher's own defaults (DEPTH 4, KEEP file:64, CONF/LOOKUP/EVICT) are left in place. DSpark initializes lazily on
  the first `GEN`, so after the first excluded warm-up the harness **verifies** the arm's DSpark state (sidecar load
  line present & equal to the local path for MTP-on; absent for MTP-off) and refuses to continue otherwise.
- **Fixtures: 18 distinct measured prompts + one shared warm-up.** The six cells × three generations use a
  **distinct token-id fixture per request**, all 18 pairwise distinct and all distinct from the warm-up fixture.
  The two warm-up requests intentionally share the single warm-up fixture. Both arms consume the *identical*
  fixture set and order.
- **Balanced rotation.** Generation *g* processes the six cells in the order `cell[(pos + 2*g) mod 6]`, i.e.
  offsets `0, 2, 4` for three generations — a deterministic, balanced rotation that spreads each cell across
  three distinct positions, identical for both arms. The exact per-generation order is recorded in
  `prompts_ab.json` (`schedule.generation_schedule`) and a per-cell `rotation_offset` is stored with each result.
- **Reuse guard.** In a persistent server the engine's `reused` is the longest common prefix against the previous
  request's history; every fixture shares a fixed short chat-template prefix (recorded as
  `fixture_policy.template_prefix_reuse_tokens`, currently 2), so a *fresh* fixture honestly reports `reused == 2`,
  never 0 — a literal 0 is physically impossible here. The guard therefore enforces `reused ≤ template prefix`
  (i.e. **zero fixture-level reuse**), cross-checks the DONE field 8 against the `serve: prompt … reused` and
  `serve: reuse … -> reused=` lines, and flags any request reusing more than half its prompt (a fixture repeat).
  Only the warm-up is exempt.
- **Engine defaults stay engine defaults.** `--prompt-batch`, `--prefill-chunk`, `--arena-gib`, `--route-bias`,
  `--kv-fp4`, `--arena-adapt` are **not** passed, exactly as in the one-shot grid, so the engine defaults
  (`arena-gib 60`, route-bias 1.0/0.6, no kv-fp4, no arena-adapt) apply; both arms get identical argv, Engram, rank,
  ctx. `STRATA_SERVE_PARK=0` is pinned (the prompts are unrelated independent requests), and both arms get the
  identical environment apart from DSpark. Inherited `STRATA_*` tuning variables are stripped before launch so a
  shell environment cannot silently skew the arms; the local launcher then applies its documented defaults.
- **Accepted result = the engine's own counts.** `results_ab.json` records the prompt count and generated count
  the engine reported and whether they match the prepared count and requested length, plus the output id SHA-256.
  The DONE line is parsed in its **exact 14-value format** (reused = value 8) and its counts are cross-checked
  against the phase lines.

## Safety

- Runs only through the local `run_engine_fg.sh` (same shared lock and DSpark contract as `ds41-engine.sh`, no
  `memguard.sh` nesting). It refuses to start if ComfyUI listens on `:8188` (unless `MEMGUARD_ALLOW_COMFY=1`), if
  the pre-load MemAvailable is below the floor, and it runs the engine in a `MemoryMax=110G` / `MemorySwapMax=0`
  scope.
- **MemAvailable floors.** A **pre-load** floor of **70 GiB** gates every engine start (checked in both the
  harness preflight and the launcher). Once the 60 GiB arena is resident a 70 GiB MemAvailable is impossible, so
  the **per-request** in-run guard uses a lower sanity floor (`RUN_MEM_FLOOR_GIB = 8`), and the launcher's host
  **watchdog** (`BENCH_WATCHDOG_FLOOR_GIB = 4`, the value `memguard.sh` uses on this host) is the hard backstop
  that KILLs the scope if the host gets tight. The harness **verifies** the printed scope line
  (`MemoryMax=110G swap=0 floor=4 GiB`) and refuses to continue on any mismatch.
- Host samples (`MemAvailable`, per-GPU `mem.free`, `util`) go to `resources_ab.csv` throughout.

## Usage

```bash
PY=/home/neron/shared-venv-gpu/bin/python          # needs the repo's tokenizer modules + numpy

# 1. write prompts_ab.json and prompts_ab/*.i32   (CPU only, no model load)
$PY benchmark_server_ab.py --outdir . prepare

# 2. the run (starts the engine; ~20 min for --arm both)
$PY benchmark_server_ab.py --outdir . run --arm both        # or --arm mtp / --arm nomtp

# 3. aggregates only, from an existing results_ab.json
$PY benchmark_server_ab.py --outdir . summarize
```

The run prints progress per request and rewrites `results_ab.json` after every request, so a killed run keeps
its partial data. A request that fails (non-zero exit, missing `DONE`, engine-count mismatch, reuse over the
allowed template prefix) is recorded with its error, and the run refuses to continue; `--allow-failures` records
and continues.

## Testing (no model, no GPU)

```bash
$PY -m py_compile benchmark_server_ab.py fake_protocol_test.py   # syntax
$PY benchmark_server_ab.py --outdir . prepare                    # rewrite prompts_ab.json (CPU + tokenizer only)
$PY fake_protocol_test.py                                        # end-to-end with a fake engine (no model load)
```

`fake_protocol_test.py` builds a throwaway fake repo + fake `ds41_generate`, then drives the real harness through
`run --arm both` + `summarize`, asserting: 18 distinct measured fixtures + a shared warm-up per arm, the DONE
parser (reused = value 8), the reuse guard, the 110 GiB / swap-0 / watchdog scope verification, DSpark on/off
state verification, the rotation schedule, per-cell mean/SD/min/max/spread, and stripping of inherited `STRATA_*`
tuning overrides. A deliberately incorrect DSpark state that remains absent after the first warm-up request must be
rejected and cleanly stop its engine and transient scope. Launcher guards (Comfy `:8188` refusal, pre-load
MemAvailable floor, watchdog kill) were verified separately by direct `run_engine_fg.sh` invocations.
