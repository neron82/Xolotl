# DeepSeek V4.1 Flash — local benchmark report

Run date: 2026-10-18 (CEST)

## Result

The requested six-cell exact-length benchmark completed successfully on physical GPU 1. All six engine processes exited 0. Prompt input IDs matched the manifest hashes; output ID files matched the hashes recorded in `results.json` and their expected byte lengths. The engine logs independently reported the requested prompt and generated-token counts.

| Prompt tokens | Output tokens | Prefill tok/s | Decode tok/s | Prefill s | Decode s | Wall s incl. model load |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 128 | 28.00 | 19.02 | 36.62 | 6.68 | 99.8 |
| 1,024 | 256 | 27.10 | 15.76 | 37.77 | 16.18 | 111.9 |
| 2,048 | 128 | 42.40 | 17.06 | 48.33 | 7.45 | 114.6 |
| 2,048 | 256 | 41.60 | 15.91 | 49.22 | 16.02 | 123.6 |
| 4,096 | 128 | 70.90 | 19.94 | 57.76 | 6.37 | 123.2 |
| 4,096 | 256 | 76.10 | 18.41 | 53.85 | 13.85 | 125.4 |

Each one-shot process loaded the model in 47.8–50.3 seconds. The full sequence (one warm-up plus six measured runs), including each model load, summed to 792 seconds (13m12s). Throughput figures above are the engine's separately reported prefill and decode phases, not wall-time averages.

## Runtime and method

- Engine: `/home/neron/projects/xolotl`, branch `main`, commit `6e4bc8e5c09cf732dfa25ccf3ecfc81e672e6d69`; CUDA build completed successfully for sm_86.
- Model: `/home/neron/models/Deepseek-V41-GSQ/V41-GSQ.gguf`; Engram tables extracted to `layers.1` and `layers.14` from the completed FP8 source shards and verified with 2,000 random rows per tensor.
- Device: `CUDA_VISIBLE_DEVICES=1`, physical RTX 3090 at PCI bus `00000000:2B:00.0` (the CPU-connected x16 card). GPU0 was not part of the run.
- DSpark: recommended `V41-MTP-Vision.Q2_K.gguf` sidecar, depth 4. Temperature 0; context 8,192; 60 GiB arena; engine-default prompt batch/prefill chunk; route biases 1.0 / file 0.6.
- Safety: each run used `tools/ds4/memguard.sh`, an 84 GiB cgroup memory cap, `MemorySwapMax=0`, a shared GPU lock, and a 70 GiB preflight MemAvailable threshold. The resource log sampled a minimum host MemAvailable of 49.837 GiB during the grid; the engine scope itself had swap disabled. No OOM, failed cell, or hung process occurred.
- One separate 1,024-token / 32-output warm-up completed before the measured grid; it is excluded from the table.

## Two-GPU scope

No two-GPU benchmark was run. The custom V4.1 `ds41_generate` path has no multi-GPU selector or layer-split implementation; `CUDA_VISIBLE_DEVICES=1` selects its single CUDA device. The generic Strata layer-split documentation applies to other engine/model paths, not this V4.1 executable. A two-GPU speed figure would therefore be invented, not measured.

## Post-run state and caveats

ComfyUI was restored after the grid: `comfyui.service` is active and `/system_stats` returned HTTP 200. BGE-m3 on port 7777 also returned HTTP 200. No Xolotl process remained and the shared GPU lock was free. At the post-run check, MemAvailable was 114 GiB and memory PSI avg10 was 0.00, but host swap remained nearly full (2.0 GiB used, 46 MiB free); this is distinct from the benchmark cgroup, whose swap limit was zero. Four subsequent one-second `vmstat` samples showed no swap-out (swap-in 4, 4, then 0, 0 KiB/s). The models filesystem had 121 GiB free.

These are single-GPU measurements on this RTX 3090 and this exact one-shot configuration. There was no no-DSpark control arm, and no quality evaluation is implied by the token-generation checks.

## Artifacts

- Harness and prompt manifest: `benchmark_grid.py`, `prompts.json`
- Machine-readable results: `results.json`, `results.csv`
- Resource samples: `resources.csv`
- Per-cell stdout/stderr: files under `runs/`
- Build and process ledger: `build.log`, `PROCESS_LOG.md`
