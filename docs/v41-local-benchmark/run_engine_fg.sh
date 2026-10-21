#!/usr/bin/env bash
# run_engine_fg.sh - LOCAL, AUDITABLE foreground launcher for the DS41 MTP A/B benchmark.
#
# Why this file exists (do NOT replace it with tools/v41/serve/ds41-engine.sh):
#   * ds41-engine.sh hard-codes `MemoryMax=84G`, but the user requires a 110 GiB hard cap for this A/B.
#   * ds41-engine.sh has no live host-MemAvailable watchdog.
#   * ds41-engine.sh's `exec ... systemd-run --scope` leaves no shell to supervise the run.
# This launcher reproduces ds41-engine.sh's serverside ENV CONTRACT byte-for-byte (so the arm switch means exactly
# one thing: STRATA_SERVE_DSPARK on / off), but adds the required 110 GiB cap and a host watchdog.  It never edits
# tracked source and it does not background the engine: the engine is the foreground child of this shell, so the
# harness's stdin/stdout line protocol is inherited unchanged.
#
# Usage (the harness builds exactly this argv):
#   run_engine_fg.sh -m MODEL --backend cuda --experts gpu --engram A:B --rank R --ctx N --serve
#
# Guards, in order:
#   1. ComfyUI refusal on :8188 (same rule as ds41-engine.sh / memguard.sh) unless MEMGUARD_ALLOW_COMFY=1.
#   2. Pre-load host MemAvailable floor (default 70 GiB) - checked BEFORE the engine reserves its 60 GiB arena.
#   3. Shared GPU/model flock (same lock path as ds41-engine.sh and memguard.sh) held for the engine's lifetime.
#   4. systemd-run --user --scope with MemoryMax=110G (hard cap) and MemorySwapMax=0 (no swap; the engine is
#      OOM-killed (137) instead of freezing the desktop).
#   5. Live watchdog: if host MemAvailable falls below WATCHDOG_FLOOR_GIB (default 4 GiB - the value memguard.sh
#      uses on this host) the scope is KILLed.  This is the same class of protection as memguard.sh; it is the
#      ONLY guard that is allowed to fire while the arena is resident, because a loaded engine's own arena makes
#      a 70 GiB MemAvailable physically impossible.
#
# Tunables (env, defaults chosen for this host: 125 GiB RAM, 2x RTX 3090):
#   BENCH_MEMGUARD_MAX_GIB   (110)  MemoryMax for the scope         -- the required hard cap
#   BENCH_MIN_MEM_AVAIL_GIB  (70)   pre-load MemAvailable floor      -- the required start gate
#   BENCH_WATCHDOG_FLOOR_GIB (4)    live MemAvailable kill floor     -- documented watchdog floor
#   MEMGUARD_ALLOW_COMFY     (0)    set to 1 to skip the :8188 refusal (Mal's call, idle ComfyUI only)
set -u

BENCH_MEMGUARD_MAX_GIB="${BENCH_MEMGUARD_MAX_GIB:-110}"
BENCH_MIN_MEM_AVAIL_GIB="${BENCH_MIN_MEM_AVAIL_GIB:-70}"
BENCH_WATCHDOG_FLOOR_GIB="${BENCH_WATCHDOG_FLOOR_GIB:-4}"
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"

avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

# ---- 1. ComfyUI refusal on :8188 (never bypassed unless explicitly allowed) ----
if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "run_engine_fg: refusing - ComfyUI is running on :8188 (RAM)" >&2
    exit 3
fi

# ---- 2. pre-load MemAvailable floor (only meaningful BEFORE the 60 GiB arena is resident) ----
a="$(avail_gib)"
if [ "$a" -lt "$BENCH_MIN_MEM_AVAIL_GIB" ]; then
    echo "run_engine_fg: refusing - MemAvailable ${a} GiB < ${BENCH_MIN_MEM_AVAIL_GIB} GiB needed before load" >&2
    exit 3
fi

cd "$(dirname "$0")/../.."   # repo root (ds41-engine.sh's cwd contract)
export GGML_NO_BACKTRACE=1
# 4 KiB pages pin the 60 GiB arena at ~1.2 GB/s (load ~50 s); THP pinning is ~0.16 GB/s (~4 min).  Same default as
# ds41-engine.sh: the harness also pins this explicitly, so both arms share it.
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
# The exact/batch prompt split and the route-bias knobs of ds41-engine.sh (defaults, not overridden):
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# ---- the DSpark contract, VERBATIM from ds41-engine.sh: an UNSET STRATA_SERVE_DSPARK defaults to the Mal sidecar
# path; an EMPTY STRATA_SERVE_DSPARK is unset below, so the engine runs the no-draft control arm.  The harness is
# responsible for passing `STRATA_SERVE_DSPARK=` (empty) for the MTP-off arm and the local Q2_K path for MTP-on.
DSPARK_SIDECAR=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-$DSPARK_SIDECAR}"
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
export STRATA_SERVE_LOOKUP="${STRATA_SERVE_LOOKUP:-14}"
export STRATA_SERVE_DSPARK_EVICT="${STRATA_SERVE_DSPARK_EVICT:-4096}"
export STRATA_DENSE_REQUANT="${STRATA_DENSE_REQUANT-q6_K}"
export STRATA_PRESTAGE_GIB="${STRATA_PRESTAGE_GIB:-1.5}"
[ -n "${STRATA_PROMPT_BATCH:-}" ] && set -- "$@" --prompt-batch "$STRATA_PROMPT_BATCH"
# empty STRATA_SERVE_DSPARK -> unset, so the engine sees "no sidecar" instead of an empty path (the engine treats a
# non-null empty path as a path and fails at DSpark init - the exact bug the process log records at 09:52).
[ -z "${STRATA_SERVE_DSPARK:-}" ] && unset STRATA_SERVE_DSPARK

BUILD_DIR="${STRATA_BUILD_DIR:-$PWD/build-xolotl}"
[ -x "$BUILD_DIR/ds41_generate" ] || [ ! -x "$PWD/build-ds4-gpu/ds41_generate" ] || BUILD_DIR="$PWD/build-ds4-gpu"
if [ ! -x "$BUILD_DIR/ds41_generate" ]; then
    echo "run_engine_fg: engine binary not found under $BUILD_DIR" >&2
    exit 3
fi

# ---- 3/4/5. flock -> scope -> watchdog -> engine, protocol fds untouched ----
unit="ds41-bench-$$"
echo "memguard: scope ${unit} MemoryMax=${BENCH_MEMGUARD_MAX_GIB}G swap=0 floor=${BENCH_WATCHDOG_FLOOR_GIB} GiB (avail ${a} GiB, need ${BENCH_MIN_MEM_AVAIL_GIB} GiB pre-load)"
if [ -z "${STRATA_SERVE_DSPARK:-}" ]; then
    echo "run_engine_fg: STRATA_SERVE_DSPARK=<unset> (empty -> unset: MTP off, no drafts)" >&2
else
    echo "run_engine_fg: STRATA_SERVE_DSPARK=${STRATA_SERVE_DSPARK} (MTP on)" >&2
fi

watchdog_stop=0
watchdog() {
    while [ "$watchdog_stop" != 1 ]; do
        if [ "$(avail_gib)" -lt "$BENCH_WATCHDOG_FLOOR_GIB" ]; then
            echo "run_engine_fg: MemAvailable < ${BENCH_WATCHDOG_FLOOR_GIB} GiB - killing ${unit}.scope" >&2
            systemctl --user kill -s KILL "${unit}.scope" 2>/dev/null
            return
        fi
        sleep 1
    done
}
watchdog &
wd_pid=$!

# `flock ... systemd-run --scope` in the FOREGROUND: flock holds the lock fd and execs into systemd-run, which runs
# the engine as a child in the transient scope unit with the harness's inherited stdin/stdout.
set +e
flock "$LOCK" systemd-run --user --scope -q --unit="${unit}" \
    -p MemoryMax="${BENCH_MEMGUARD_MAX_GIB}G" -p MemorySwapMax=0 \
    "$BUILD_DIR/ds41_generate" "$@"
rc=$?
set -e

watchdog_stop=1
kill "$wd_pid" 2>/dev/null || true
wait "$wd_pid" 2>/dev/null || true
[ "$rc" -eq 137 ] && echo "run_engine_fg: process was killed (exit 137) - memory cap or watchdog" >&2
exit "$rc"
