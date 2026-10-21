# DeepSeek V4.1 Flash GSQ — persistent-server A/B benchmark

Run: 2026-10-18 (CEST)

Dataset source: `results_ab.json`

Generated report companion: `SUMMARY_AB.md`

## Executive result

The repeated, persistent-server benchmark completed on one RTX 3090 (physical GPU 1, PCI bus `00000000:2B:00.0`), with six prompt/output cells and three accepted measured runs per cell for each arm. The MTP/DSpark-on arm improved decode throughput in every cell: +1.5% to +17.6%, averaging +7.4% across the 18 measured requests. It traded that for a 2.7%–5.7% prefill slowdown, averaging −4.5%.

There is no two-GPU result: the V4.1 `ds41_generate` path has no multi-GPU selector or layer-split implementation. Generic Strata layer-splitting documentation applies to other model paths, not this executable.

## Results

Throughput is the engine-reported phase throughput in tokens/s. Values are mean ± sample standard deviation over three runs per cell. The delta is MTP-on relative to MTP-off.

| Prompt | Output | Prefill MTP on | Prefill MTP off | Δ prefill | Decode MTP on | Decode MTP off | Δ decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 128 | 65.37 ± 0.78 | 67.97 ± 0.71 | −3.8% | 26.46 ± 2.12 | 24.63 ± 0.33 | +7.5% |
| 1,024 | 256 | 65.73 ± 0.90 | 68.97 ± 0.76 | −4.7% | 28.22 ± 2.38 | 24.00 ± 0.68 | +17.6% |
| 2,048 | 128 | 95.33 ± 1.10 | 98.00 ± 1.70 | −2.7% | 27.41 ± 3.52 | 24.67 ± 0.59 | +11.1% |
| 2,048 | 256 | 95.77 ± 0.55 | 98.90 ± 1.06 | −3.2% | 25.37 ± 1.73 | 24.99 ± 0.15 | +1.5% |
| 4,096 | 128 | 137.47 ± 0.90 | 145.77 ± 1.53 | −5.7% | 25.03 ± 1.16 | 23.89 ± 0.30 | +4.8% |
| 4,096 | 256 | 137.93 ± 1.21 | 146.10 ± 1.35 | −5.6% | 24.00 ± 1.54 | 23.60 ± 0.62 | +1.7% |
| **Mean over 18 requests** | — | **99.60** | **104.28** | **−4.5%** | **26.08** | **24.30** | **+7.4%** |

Engine load time was 74.448 s for MTP-on and 76.251 s for MTP-off. These load times are reported separately; throughput above excludes model load.

## Method and configuration

- Model: `/home/neron/models/Deepseek-V41-GSQ/V41-GSQ.gguf`; extracted runtime Engram tables in `/home/neron/models/Deepseek-FP8-engram/layers.1` and `layers.14`.
- Engine: Xolotl `ds41_generate --serve`, context 8,192; one persistent engine load per arm.
- MTP-on used `/home/neron/models/Deepseek-V41-GSQ/V41-MTP-Vision.Q2_K.gguf`; MTP-off ran with the sidecar unset.
- Six cells: prompt 1,024/2,048/4,096 × output 128/256; three generations per cell per arm. Two warm-ups per arm were excluded. All 36 measured requests were accepted, with no reported request errors or problems.
- The two arms used identical prepared prompt fixtures and engine/model/Engram/rank/context settings; only the MTP/DSpark arm setting differed. Output token IDs and hashes were recorded per request; outputs were not assumed identical between arms.
- Safety: each arm ran in a 110-GiB memory cgroup with swap disabled, a shared GPU lock, a 70-GiB pre-load MemAvailable floor, and an 8-GiB in-run floor. No model service was left running after the benchmark.

## Interpretation

For this workload, enabling MTP is a reasonable decode-speed trade: the average gain is modest but consistent, while prefill becomes about 4.5% slower. The largest decode benefit was on 1,024/256 (+17.6%); the smallest was 2,048/256 (+1.5%). If prompt processing dominates, MTP-off is faster in this grid; if generation dominates, MTP-on is faster. These are benchmark results for this exact one-GPU setup and context, not a quality evaluation or a claim about larger contexts.

The earlier `REPORT.md` is the separate one-shot grid. Do not pool its figures with this persistent-server repeated A/B: the harness/method differs.

## Artifacts and post-run state

- Full per-cell table: `SUMMARY_AB.md`
- Raw results: `results_ab.json`
- Harness and manifest: `benchmark_server_ab.py`, `prompts_ab.json`
- Per-arm logs and traces: `runs_ab/`

Post-run verification: DS41 unit inactive, no DS41 process or listener on 8080/8150, shared GPU lock free, GPU 1 had 23,785 MiB free, host MemAvailable was 115 GiB, BGE-M3 health returned HTTP 200, and ComfyUI remained stopped. The two extracted runtime Engram tables remain; the original source `.safetensors` shards are absent. The report makes no two-GPU speed claim.
