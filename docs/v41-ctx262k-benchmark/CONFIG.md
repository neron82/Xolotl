# ctx=262144 rerun of the DeepSeek V4.1 persistent-server A/B benchmark

Prepared 2026-10-18. **Not yet run** — this directory holds the prepared, verified runner and config for the
ctx=262144 rerun, to be compared against the ctx=8192 baseline in
`/home/neron/projects/xolotl/docs/v41-local-benchmark/`.

## What is reused verbatim (baseline must stay untouched)

| artifact | path | status |
|---|---|---|
| harness | `docs/v41-local-benchmark/benchmark_server_ab.py` | **reused in place, unmodified** |
| launcher | `docs/v41-local-benchmark/run_engine_fg.sh` | **reused in place** (resolved by the harness as `<repo>/docs/v41-local-benchmark/run_engine_fg.sh`) |
| fixtures | `docs/v41-local-benchmark/prompts_ab/*.i32` (20 files) | **reused verbatim** — the manifest copy points at them by absolute path; `run` re-verifies every sha256 + size |
| fixture manifest | `docs/v41-local-benchmark/prompts_ab.json` | **copied byte-identically** to this dir (same sha256 `3958113edb7409ee…564a3`) |

`prepare` is **not** re-run: the ctx rerun must not regenerate fixtures (a regenerated fixture would turn a ctx
effect into a fixture effect).

## The one command

```
bash /home/neron/projects/xolotl/docs/v41-ctx262k-benchmark/run_ctx262k.sh
```

The script runs a 9-gate preflight **in the same command** that launches, then executes the exact harness
invocation:

```
/home/neron/shared-venv-gpu/bin/python \
  /home/neron/projects/xolotl/docs/v41-local-benchmark/benchmark_server_ab.py \
  --outdir /home/neron/projects/xolotl/docs/v41-ctx262k-benchmark \
  --ctx 262144 --arm both --reps 3 --warmup-repeats 2 \
  --min-free-vram-mib 22000 --min-mem-available-gib 70 \
  --physical-gpu <nvidia-smi index for PCI 00000000:2B:00.0> \
  run --arm both
```

`--physical-gpu` is resolved from the bus id at launch (fails loudly if the bus is absent). On this host
`00000000:2B:00.0` = index 1 = UUID `GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b` — the same card and the same
index the baseline used. `DRY_RUN=1 bash run_ctx262k.sh` runs the preflight without launching.

## Effective-config comparison (baseline vs rerun)

Values read from the baseline's `results_ab.json` (`config` + per-arm engine stderr headers). The rerun passes
every one of these identically except `ctx`, `outdir`, and the launch-time card index resolution.

| knob | baseline (8192) | rerun (262144) | same? |
|---|---|---|---|
| `ctx` | 8192 | 262144 | **differs (the one intended change)** |
| `outdir` | `docs/v41-local-benchmark` | `docs/v41-ctx262k-benchmark` | **differs (separate outputs)** |
| card | `--physical-gpu 1` = PCI `00000000:2B:00.0`, UUID `GPU-7ac9a486…` | resolved from bus `2B` → index 1, same UUID | same card/pin |
| `reps` | 3 | 3 | ✓ |
| warmups/arm | 2 | 2 | ✓ |
| `--arm` | both (mtp + nomtp) | both | ✓ |
| model / rank / engram / sidecar | GSQ gguf, rank.bin, layers.1:layers.14, V41-MTP-Vision.Q2_K | identical absolutes | ✓ |
| `STRATA_SERVE_DSPARK` | mtp=same sidecar, nomtp=empty→unset | identical | ✓ |
| `STRATA_SERVE_PARK` | 0 | 0 | ✓ |
| `STRATA_NO_LARGEPAGES` | 1 | 1 | ✓ |
| `STRATA_SERVE_STEP_MAX` / `PASS_BIAS` | 96 / 0 (launcher defaults) | same | ✓ |
| `STRATA_DSPARK_*` (KEEP/DEPTH/CONF/LOOKUP/EVICT) | launcher defaults | same | ✓ |
| `STRATA_DENSE_REQUANT` / `STRATA_PRESTAGE_GIB` | `q6_K` / 1.5 | same | ✓ |
| `prompt_batch` / `prefill_chunk` | engine defaults (not overridden) | same | ✓ |
| `BENCH_MEMGUARD_MAX_GIB` | 110 | 110 | ✓ |
| `MemorySwapMax` | 0 | 0 | ✓ |
| `BENCH_WATCHDOG_FLOOR_GIB` | 4 | 4 | ✓ |
| `BENCH_MIN_MEM_AVAIL_GIB` (pre-load) | 70 | 70 | ✓ |
| in-run MemAvailable floor | 8 | 8 | ✓ |
| `min_free_vram_mib` | 22000 | 22000 | ✓ |
| fixtures (all 20 sha256) | baseline set | **byte-identical baseline set** | ✓ |
| `CUDA_VISIBLE_DEVICES` | 1 | index for bus 2B (=1) | ✓ |

The harness itself strips **all** inherited `STRATA_*` variables before each arm start and pins the scope/gate
knobs, so an ambient shell cannot silently change the arm (see `engine_env()` in the harness).

## Expected duration

Baseline (ctx 8192) both arms, engine-alive: **18.9 min** (mtp 9.4 min + nomtp 9.5 min; each load ~74–76 s,
prefill ~400 s/arm + decode ~140 s/arm + warmups ~21 s/arm). The ctx rerun has the same grid (≤4096-token
prompts) and the same fixed-size furnace arena, so per-token throughput should be close; the larger KV cache
adds some init/memory cost. **Estimate: ~20–25 min** for `--arm both`, plus ~10 min of operator overhead to stop
and later restore the model service.

## Caveats / notes to carry into the report

- **Load path / page cache**: the second arm inherits the first arm's page cache. This is identical to the
  baseline shape, so it does not make the arms less comparable to baseline, but a "colder second arm" and a
  genuine effect look alike unless cell list, load path and cache state are recorded per arm (they are).
- **Reuse guard is `<= 2`, not `== 0`.** Every fixture shares the chat-template BOS prefix, so a fresh fixture
  still reports `reused == 2`. The harness enforces `reused <= template_prefix_reuse_tokens (2)` on *every* reuse
  source (DONE field 8, prompt line, reuse line, RESUME). The run is accepted only if no measured request reuses
  more than the template prefix.
- **DONE line has 15 fields; `reused` is field 8** (fields 6/7 are reserved zeros). The verifier checks the field
  count and reads field 8.
- **No two-GPU arm exists** — `ds41_generate` has no split selector. Single card only, as in the baseline.
- **This directory does not touch services.** The operator must stop `battlestation-llm.service` before running
  (it currently holds the DS41 engine at ctx 262144 and the shared GPU lock). ComfyUI may remain stopped;
  BGE-M3 on :7777 is protected and must stay up (preflight gate 8 enforces HTTP 200).
- The ctx rerun is **not** poolable with the baseline numbers as one dataset: quote the comparison, not a merged
  grid (different ctx ⇒ different measurement condition).

## Post-run verification

```
/home/neron/shared-venv-gpu/bin/python \
  /home/neron/projects/xolotl/docs/v41-ctx262k-benchmark/verify_outputs.py
```

Proves: ctx==262144; both arms 2 warm-ups + 18 measured rows over 6 cells ×3; every row accepted with
`reused <= 2`; exact DONE prompt/generated counts and field count 15; scope 110G/swap0/floor4; the pinned
`00000000:2B:00.0` card in `gpu_before`; DSpark intent per arm; and that every measured `ids_sha256` equals the
baseline manifest's fixture hash (fixtures untouched). Exit 0 = all proved.

## Files in this directory

- `run_ctx262k.sh` — the runner (preflight + exact harness command). `DRY_RUN=1` = preflight only.
- `prompts_ab.json` — byte-identical copy of the baseline manifest (fixtures stay in the baseline dir).
- `verify_manifest.py` — preflight manifest/fixture integrity check.
- `verify_outputs.py` — post-run results verification.
- `fake_protocol_test_ctx262k.py` — no-model end-to-end ladder at ctx 262144 (validated, see below).
- `CONFIG.md` — this file.
