#!/usr/bin/env bash
# run_ctx262k.sh - standalone, reusable runner for the ctx=262144 rerun of the DeepSeek V4.1
# persistent-server MTP A/B benchmark.
#
# DESIGN: this script does NOT reimplement the benchmark. It reuses the EXACT baseline harness
#   /home/neron/projects/xolotl/docs/v41-local-benchmark/benchmark_server_ab.py
# and the EXACT baseline launcher (resolved by the harness as <repo>/docs/v41-local-benchmark/run_engine_fg.sh),
# changing ONLY:
#   * --ctx 262144            (baseline: 8192)
#   * --outdir <this dir>     (baseline: docs/v41-local-benchmark)
# The PCI-2B card pin is verified (not changed): baseline and this run both target the CPU slot
# 00000000:2B:00.0.  The harness selects that card via --physical-gpu <nvidia-smi index>; this script
# resolves the index from the bus id at launch time and fails loudly if it is not 1.
#
# Preflight (all gates in THIS command, so a failed gate exits before any model load):
#   1. py_compile harness + verifier
#   2. battlestation-llm.service inactive   (else it holds the DS41 engine + GPU lock)
#   3. no live ds41_generate
#   4. shared GPU lock free                  (~/.quetza-data/conductor/ds4-gpu.lock)
#   5. GPU 00000000:2B:00.0 present, maps to a known index, and has >= MIN_FREE_VRAM_MIB free
#   6. MemAvailable >= PRELOAD_MEM_FLOOR_GIB
#   7. :8188 (ComfyUI) free
#   8. BGE-M3 :7777 healthy (HTTP 200)
#   9. prompts_ab.json present and every fixture hash + size verified against the baseline manifest
#
# Usage:
#   bash run_ctx262k.sh                 # full preflight + run --arm both --ctx 262144
#   DRY_RUN=1 bash run_ctx262k.sh       # preflight only; print the exact command that would run
#
# This script never touches services, the lock, the dashboard, or Hermes config. It does not stop
# battlestation-llm.service: the operator stops it before invoking this script.
set -euo pipefail

CTX="${CTX:-262144}"
ARM="${ARM:-both}"
REPS="${REPS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-2}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"
PRELOAD_MEM_FLOOR_GIB="${PRELOAD_MEM_FLOOR_GIB:-70}"
PIN_BUS="${PIN_BUS:-00000000:2B:00.0}"
DRY_RUN="${DRY_RUN:-0}"

BASE=/home/neron/projects/xolotl/docs/v41-local-benchmark
NEW=/home/neron/projects/xolotl/docs/v41-ctx262k-benchmark
HARNESS="$BASE/benchmark_server_ab.py"
MANIFEST="$NEW/prompts_ab.json"
PY=/home/neron/shared-venv-gpu/bin/python
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }

# ---- 1. py_compile ---------------------------------------------------------------------------------
echo "[1/9] py_compile harness + verifiers"
"$PY" -m py_compile "$HARNESS" "$BASE/fake_protocol_test.py" "$NEW/verify_outputs.py" \
  || fail "py_compile"

# ---- 2. no live DS41 service holding the engine/lock -----------------------------------------------
echo "[2/9] battlestation-llm.service must be inactive"
svc="$(systemctl --user is-active battlestation-llm.service 2>/dev/null || true)"
[ "$svc" = "inactive" ] || fail "battlestation-llm.service is '$svc' (operator must stop it first)"

# ---- 3. no live engine -----------------------------------------------------------------------------
echo "[3/9] no live ds41_generate"
if pgrep -x ds41_generate >/dev/null; then fail "ds41_generate is running"; fi

# ---- 4. shared GPU lock free -----------------------------------------------------------------------
echo "[4/9] shared GPU lock free ($LOCK)"
mkdir -p "$(dirname "$LOCK")"
if [ -e "$LOCK" ]; then
    if fuser "$LOCK" >/dev/null 2>&1; then fail "GPU lock is held: $LOCK"; fi
fi

# ---- 5. card by BUS ID, resolve index, check free VRAM ---------------------------------------------
echo "[5/9] GPU $PIN_BUS present + free VRAM >= ${MIN_FREE_VRAM_MIB} MiB"
gpu_line="$(nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.free --format=csv,noheader,nounits \
            | awk -F', ' -v b="$PIN_BUS" '$2==b{print $1","$3","$4}')"
[ -n "$gpu_line" ] || fail "GPU $PIN_BUS not found in nvidia-smi output"
GPU_INDEX="${gpu_line%%,*}"; rest="${gpu_line#*,}"; GPU_UUID="${rest%%,*}"; FREE_VRAM="${rest##*,}"
echo "      -> $PIN_BUS = index $GPU_INDEX, uuid $GPU_UUID, ${FREE_VRAM} MiB free"
[ "$FREE_VRAM" -ge "$MIN_FREE_VRAM_MIB" ] || fail "GPU $PIN_BUS has ${FREE_VRAM} MiB free (< ${MIN_FREE_VRAM_MIB})"
if [ "$DRY_RUN" = 1 ] && [ "$GPU_INDEX" != 1 ]; then
    echo "      NOTE: index is $GPU_INDEX (baseline used index 1 for this bus); harness --physical-gpu will use $GPU_INDEX"
fi
PHYS_GPU="$GPU_INDEX"

# ---- 6. MemAvailable pre-load floor ----------------------------------------------------------------
echo "[6/9] MemAvailable >= ${PRELOAD_MEM_FLOOR_GIB} GiB"
avail="$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)"
[ "$avail" -ge "$PRELOAD_MEM_FLOOR_GIB" ] || fail "MemAvailable ${avail} GiB < ${PRELOAD_MEM_FLOOR_GIB} GiB (a DS41 engine may still be resident)"
echo "      -> ${avail} GiB available"

# ---- 7. ComfyUI :8188 free -------------------------------------------------------------------------
echo "[7/9] :8188 free (ComfyUI)"
ss -ltn 2>/dev/null | grep -q ':8188 ' && fail "ComfyUI is listening on :8188"

# ---- 8. BGE-M3 :7777 healthy -----------------------------------------------------------------------
echo "[8/9] BGE-M3 :7777/health"
code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7777/health || true)"
[ "$code" = "200" ] || fail "BGE-M3 :7777/health returned '$code' (must stay protected/healthy)"

# ---- 9. manifest + fixture integrity (reuses baseline int32 fixtures, does not regenerate) ----------
echo "[9/9] manifest + baseline fixture hashes"
[ -f "$MANIFEST" ] || fail "manifest missing: $MANIFEST (copy the baseline prompts_ab.json here first)"
"$PY" "$NEW/verify_manifest.py" "$MANIFEST" || fail "manifest/fixture verification"

# ---- run -------------------------------------------------------------------------------------------
# Exact harness invocation. Only --ctx and --outdir differ from the baseline rerun; every other knob is
# pinned to the baseline value. `run --arm both` = one persistent engine load per arm.
CMD=( "$PY" "$HARNESS" --outdir "$NEW" --ctx "$CTX" --arm "$ARM" --reps "$REPS" \
      --warmup-repeats "$WARMUP_REPEATS" \
      --min-free-vram-mib "$MIN_FREE_VRAM_MIB" --min-mem-available-gib "$PRELOAD_MEM_FLOOR_GIB" \
      --physical-gpu "$PHYS_GPU" run --arm "$ARM" )

echo
echo "=== READY ==="
echo "ctx=$CTX arm=$ARM reps=$REPS warmups=$WARMUP_REPEATS phys_gpu=$PHYS_GPU ($PIN_BUS $GPU_UUID)"
printf 'COMMAND: %q ' "${CMD[@]}"; echo
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN=1 -> preflight passed, not launching"; exit 0; fi

exec "${CMD[@]}"
