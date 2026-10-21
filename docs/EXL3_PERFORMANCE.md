# DeepSeek-V4.1 SAGE EXL3 performance

Measured 2026-10-21 on GPU1, an RTX 3090 at `2B:00.0`, PCIe 4.0 x16, with 125 GiB system RAM.
GPU0 is the PCIe 3.0 x4 card and was not used for these runs.

The hybrid uses the Acrogenesis 1017 GGUF for dense weights and Engram tables, and the SAGE 1.62 bpw
safetensors for routed experts. The native comparison uses the same Acrogenesis GGUF, including its IQ2_XXS
experts. All arms used one card, the same GSQ rank profile, a 60 GiB pinned arena, prompt batch 8192,
NVFP4 KV, Q6_K dense requantization, route biases `0.5 / 0.3`, and no MTP or vision encoder.

## Results

| Workload | EXL3 before | EXL3 fast | Native Acrogenesis, same day |
| --- | ---: | ---: | ---: |
| 1024-token prefill, tok/s | 26.4 | **126.6** | 84.5 |
| Decode after 1024 tokens, tok/s | 11.0 | **22.5** | 21.1 |
| 20976-token prefill, tok/s | 149.3 | **413.7** | 353.9 |
| Decode after 20976 tokens, tok/s | 8.0 | **18.0** | 16.2 |

The 1K rows are means of two distinct real-text fixtures, each followed by 128 generated tokens, at context 8192.
Neither request reused a prefix. The long rows are one request per arm, with 16 generated tokens, at context
32768. The long input is the first 20976 token IDs of the saved source-inspection prompt, encoded with the V4.1
tokenizer. Decode excludes the first token, whose logits come from prefill. Loading and cache seeding are excluded.
These are free-generation measurements: different quants and cache-tier bias masks can produce different
continuations. They are not fixed-stream kernel comparisons.

On these runs the changes make EXL3 **4.8x faster at 1K prefill and 2.0x faster at decode**. Long prefill is 2.8x
faster. The same-day native comparison is more useful than comparing requests with different prompt sizes,
batch widths or prefix reuse. The earlier local 85.8/25.7 tok/s Acrogenesis and 436.6 tok/s long-prompt results
used other benchmark settings; these measurements do not establish a win on every workload.

Raw logs, JSON, the old executable's profile, and accuracy dumps are under
`~/models/benchmarks/dsv41-exl3-20261021/`. The final result files are `final-1k.json` and `final-20k.json`;
comparison files are `baseline-1k.json`, `baseline-20k.json`, `native-1k.json`, and `native-20k.json`.

## What changed

Enable the new path with **`STRATA_EXL3_FAST=1`** before starting the engine. Unset or zero keeps the old path.
The local `tools/v41/serve/strata-ds41-sage.json` opts in and selects GPU1.

- **Compact arena slots.** The old reader padded every expert to the largest class, 16.92 MiB. The fast reader
  writes just that expert's class bytes; the arena can use the actual class size. The same 60 GiB now holds
  **8439 experts instead of 3631**. This reduces file-tier reads without changing any expert's weights.
- **Pipelined prefill.** Persistent readers fill one pinned slab while the GPU consumes the other. A separate
  copy stream and paired copy/use events protect the two device slabs. Expert kernels no longer synchronize the
  host after each expert. Slabs contain up to eight experts, with four read tasks per file-tier expert.
- **Batch upload and device gather.** Inputs, pair indices and weights cross PCIe once per layer. A device gather
  selects each expert's rows. This replaces per-row activation copies and repeated tiny metadata transfers.
- **Preserved accumulation.** Products scatter into unique `(token, top-k slot)` rows. A final kernel sums them
  in the original ascending-expert order with the original FP32 fused multiply-add, rather than atomic additions
  or a different reduction order.
- **Decode prefetch is consumed.** The EXL3 path now waits for and uses matching prefetched experts. File-tier
  misses use the persistent reader pool, and dummy prefetch DMAs are skipped. Prefetch metadata is cleared after
  use so a later layer cannot consume stale staging data.
- **Prompt scratch is released before cache seeding.** Device buffers are returned after prefill; pinned reader
  slabs remain available for the next request.

The compact arena changes tier membership. With residency-biased routing enabled, this can change selected
experts, just as changing an arena budget does. Exact-output validation therefore disables both route biases.

## Correctness checks

- The complete **256 x 129280 F32 logit dump is byte-identical** between the old executable, the new executable
  with the flag off, and the new fast path, with exact routing. All three dumps have SHA-256
  `3afded3758a9de838298d979168e712fdd1102e4525c3ca9e851f20e7de6b8f4`.
- That fixture's PPL is **60.3285**, mean NLL **4.0998**, in all three cases. This is a regression control,
  not a claim about general model quality.
- Mixed-K synthetic tests cover multiple token rows, top-k ordering, exact arena capacities, repeated chunk
  allocation/release, real prefetch consumption, and stale-prefetch rejection.
- CUDA Compute Sanitizer memcheck reports **0 errors** on the EXL3 tests. The CPU split-GGUF source tests pass.
- CUDA and CPU tier targets build. HIP/SYCL performance was not tested; this V4.1 EXL3 target is CUDA-specific.

A tensor-core FP16-input prototype reached 484.5 tok/s on the long prompt, but changed the regression logits
(mean KL 0.036 on the 256-token control). It was removed from the implementation. The enabled path retains the
existing EXL3 GEMV arithmetic.

## Reproduce

Build `ds41_generate` and `test_ds41_exl3`, then run:

```bash
CUDA_VISIBLE_DEVICES=1 ./build-xolotl/test_ds41_exl3
CUDA_VISIBLE_DEVICES=1 STRATA_EXL3_FAST=1 ./build-xolotl/test_ds41_exl3

python3 tools/v41/engine/bench_ds41_exl3.py \
  --gpu 1 --model "$DENSE_GGUF" --exl3-dir "$SAGE_DIR" --rank "$RANK" \
  --ids-files docs/v41-local-benchmark/prompts_ab/pp1024_out128.rep1.i32,docs/v41-local-benchmark/prompts_ab/pp1024_out128.rep2.i32 \
  --gen 128 --ctx 8192 --arena-gib 60 --prompt-batch 8192 --fast \
  --out /tmp/exl3-fast.json
```

Omit `--fast` for the old EXL3 path. Omit `--exl3-dir` as well for the native GGUF comparison. The harness records
actual reused-token counts, executable SHA-256, generated IDs and engine logs. Stop the resident model first so
each arm has the same free VRAM and RAM. The benchmark covers up to 20976 input tokens, not a full 512K stress run.
The native comparison needs the original full GGUF, not the dense-only support pack. The old native quants were
removed locally after the measurements and consolidation checks.

## Local service check

The service was restarted on `127.0.0.1:8080`, physical GPU1, with the fast path, context 524288 and a 64 GiB
arena. Its startup report shows **9026 arena experts**, up from 3873 before the change. OpenAI text and image
requests pass. A fresh 2567-token HTTP prompt processed at **185.6 tok/s** after asset consolidation; the model returned `READY` as
requested. A short text request returned `The capital of France is Paris.` and the vision request identified
the three colored shapes. These API checks use the deployed 512K configuration, separate from the table above.
Responses and timings are saved in `deployed-api.json` alongside the benchmark results.

## Consolidated model assets

The local dashboard now uses `~/models/ds41-server.json` through the UUID-pinned `ds41-server.sh` wrapper.
All model assets are under `~/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw/`: the EXL3 shards, a three-shard
dense/Engram GGUF support pack, `tokenizer/`, `vision/`, and `rank.bin`. The first support GGUF contains dense
tensors only; `--exl3-dir` explicitly allows the dense loader to omit native expert tensors. Normal GGUF loading
still requires them. Both Engram payloads were moved intact, and the copied dense payload was fully hashed.
The complete 256-token logit dump remains byte-identical to the reference after consolidation. The local
`README-XOLOTL.md` and `asset-migration.json` record the layout and verification.
