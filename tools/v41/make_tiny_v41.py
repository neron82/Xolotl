#!/usr/bin/env python3
"""tools/v41/make_tiny_v41.py - a TINY but VALID `deepseek41` GGUF the V4.1 engine runs end to end.

The file is the real V4.1 shrunk to a few MB: 4 blocks, IQ2_XXS routed experts in all three roles, the
compressor / indexer / Engram / hyper-connection tensor sets the engine predicts, and every metadata key
`ds41_read_geometry` asks for (a missing key is an error there, never a default).  No third-party deps.

The metadata spelling and the tensor names / type ids are the engine's own - the same set
tools/v41/engine/test_ds41_moe_shards.cpp writes in C++, and the shape rules are the ones that file's
numbers obey.  Sizes are not free, though: every dim here is the smallest one the format allows.

What forces a size (all measured from the engine sources, none invented):
  * `tensor_payload_bytes` (include/strata/artifact/gguf_reader.hpp) requires shape[0] % block_elements == 0,
    and `ds41_check_tensors` calls `in_bounds` on every predicted tensor.  IQ2_XXS is 256 elements per block,
    so the stacked expert tensors [hidden, ff, n_experts] and [ff, hidden, n_experts] need hidden % 256 == 0
    AND ff % 256 == 0 -> hidden 256, moe_intermediate 256 (the CPU tier's `native_fmt` repeats the same rule:
    "expert geometry is not whole blocks").
  * The engine reads the attention / shared-expert matrices as Q8_0 (`want` in ds41_check_tensors), and
    ggml_row_size asserts ne[0] % 32 == 0, so q_lora_rank (attn_q_b's dim 0) and o_groups*o_lora_rank
    (attn_output_b's dim 0) must be multiples of 32 -> q_lora_rank 32, o_lora_rank 16 with o_groups 2.
  * With act_quant on (the default), build_attn fake-quantizes the window KV and the indexer keys in groups of
    32 (`ds41_fake_quant` reshapes to (32, nelements/32)), so head_dim and index_head_dim must be multiples of
    32; a 16-wide row aborts ggml_reshape_2d on the first decode token.
  * On CUDA the two fused kernels only have instances for a fixed list of shapes:
    ggml_cuda_get_best_fattn_kernel (ggml-cuda/fattn.cu:564-610) takes K->ne[0] (head_dim) only in
    {40, 64, 72, 80, 96, 112, 128, 256} - 192/320/512/576 need the GQA optimization, which the engine's
    attention sinks disable by setting max_bias - and ggml_cuda_lightning_indexer
    (ggml-cuda/lightning-indexer.cu:447-536) is dispatched on q->ne[0] == 128 with q->ne[1] in {32, 64}, i.e.
    index_head_dim 128 and index_n_heads 32 or 64.  Anything else is GGML_ABORT("fatal error").  head_dim 64
    (4 x 64 = 256 = hidden) with a 32 x 128 indexer is the release's own shape and the smallest that clears
    both lists; head_dim 32 with a 2 x 32 indexer runs on CPU but aborts on CUDA.
  * The GPU tier speculatively predicts the next layer's routing with `ggml_argsort_top_k(sc, 16)`
    (ds41_dense.cpp:958 and :1004, `sc` having n_routed_experts columns) and ggml_argsort_top_k asserts
    `ne[0] >= k`, so n_routed_experts >= 16.  The CPU tier never calls predict(), so 8 experts runs on CPU and
    aborts on CUDA at the first speculative read.
  * The Engram table the engine reads off the file is Q8_0 [engram_head_dim, rows] and `open_gguf` wants
    dim % 32 == 0 -> engram_head_dim 64.
  * The candidate block's `GGML_ASSERT(cap % candidate_block_size == 0)` runs against a power-of-two cap, so
    candidate_block_size stays a power of two (8, the release's own).
Everything else is the assignment's shape: 4 blocks, 16 experts top-2, 4 heads of 64 with 8 rope dims (heads x
head_dim = hidden exactly), indexer 32 x 128 top-8 (the kernel sizes above), sliding window 8, vocab 256,
ctx 4096, compress_ratios [0,0,2,1] with the kv and index source lists [2,3] (every compressed layer is its
own source, which is what the closure rule in ds41_read_geometry requires), the candidate block on the ratio-1
layer, Engram on layer 1.

Usage: python3 tools/v41/make_tiny_v41.py --out PATH [--blocks N] [--seed S] [--shared-sources]
The optional shared-source layout needs >= 8 blocks: layers 2/5 own ratio-2/1 KV, layers 3/6
reuse both KV and index selections, and layers 4/7 rerank the shared keys. Layer 5 supplies
candidate blocks to the later ratio-1 indexer. Without the flag, the original layout is unchanged.
"""

import argparse
import os
import random
import struct
import sys

# ------------------------------------------------------------------ GGUF v3 (what strata::GgufFile reads)
MAGIC, VERSION, ALIGN = 0x46554747, 3, 32

# ggml type ids and block geometry exactly as include/strata/artifact/gguf_reader.hpp spells them.
TY_F32, TY_F16, TY_Q8_0, TY_IQ2_XXS = 0, 1, 8, 16
BLOCK = {TY_F32: (1, 4), TY_F16: (1, 2), TY_Q8_0: (32, 34), TY_IQ2_XXS: (256, 66)}

# GGUF metadata value types
MV_U8, MV_BOOL, MV_STRING, MV_ARRAY, MV_U32, MV_U64, MV_I64, MV_F64 = 0, 7, 8, 9, 4, 10, 11, 12


def payload_bytes(shape, ttype):
    """The tensor's byte count, the way tensor_payload_bytes computes it (0 = not a whole-block shape)."""
    be, bb = BLOCK[ttype]
    if shape[0] % be:
        return 0
    n = 1
    for d in shape:
        n *= d
    return n // be * bb


# ------------------------------------------------------------------ deterministic payloads
def rng_for(name, seed):
    """FNV-1a over the tensor name (the C++ writer's trick), so each tensor's bytes are stable and distinct."""
    h = 14695981039346656037
    for c in name.encode():
        h = ((h ^ c) * 1099511628211) & ((1 << 64) - 1)
    return random.Random((h ^ seed) & ((1 << 64) - 1))


def floats(rng, n, amp):
    return [rng.uniform(-amp, amp) for _ in range(n)]


def f32_bytes(vals):
    return struct.pack('<%df' % len(vals), *vals)


def f16_bytes(vals):
    return struct.pack('<%de' % len(vals), *vals)


def q8_0_bytes(vals):
    """One Q8_0 block = an fp16 scale d then 32 int8 codes; ggml dequantizes them as d * code."""
    out = bytearray()
    for i in range(0, len(vals), 32):
        blk = vals[i:i + 32]
        amax = max(abs(v) for v in blk)
        d = amax / 127.0 if amax else 0.0
        codes = [max(-127, min(127, int(round(v / d)))) if d else 0 for v in blk]
        out += struct.pack('<e', d) + struct.pack('<32b', *codes)
    return bytes(out)


def iq2_xxs_bytes(rng, n, scale=0.02):
    """One IQ2_XXS block = an fp16 scale then 64 code bytes (256 values, 2 bits each).  Any code sequence is a
    valid one - each 2-bit index picks a codebook entry - so random codes with a small pinned scale are real
    weights in a narrow band, which is what keeps the forward pass finite."""
    out = bytearray()
    packed = rng.getrandbits(8 * 64 * (n // 256)).to_bytes(64 * (n // 256), 'little')
    for i in range(0, n // 256):
        out += struct.pack('<e', scale) + packed[64 * i:64 * (i + 1)]
    return bytes(out)


# ------------------------------------------------------------------ GGUF v3 writer
class Writer:
    def __init__(self):
        self.head = bytearray(struct.pack('<IIQQ', MAGIC, VERSION, 0, 0))   # counts patched at finish
        self.data = bytearray()
        self.n_kv = 0
        self.n_tensors = 0

    def kv(self, key, vtype, payload):
        self.head += struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', vtype) + payload
        self.n_kv += 1

    def kv_str(self, key, val):
        b = val.encode()
        self.kv(key, MV_STRING, struct.pack('<Q', len(b)) + b)

    def kv_u32(self, key, val):
        self.kv(key, MV_U32, struct.pack('<I', val))

    def kv_u64(self, key, val):
        self.kv(key, MV_U64, struct.pack('<Q', val))

    def kv_i64(self, key, val):
        self.kv(key, MV_I64, struct.pack('<q', val))

    def kv_f64(self, key, val):
        self.kv(key, MV_F64, struct.pack('<d', val))

    def kv_bool(self, key, val):
        self.kv(key, MV_BOOL, struct.pack('<B', 1 if val else 0))

    def kv_arr(self, key, elem, fmt, values):
        body = struct.pack('<%s' % (fmt * len(values)), *values) if values else b''
        self.kv(key, MV_ARRAY, struct.pack('<I', elem) + struct.pack('<Q', len(values)) + body)

    def tensor(self, name, shape, ttype, payload):
        assert payload_bytes(shape, ttype) == len(payload), '%s: payload does not match its shape' % name
        self.head += struct.pack('<Q', len(name)) + name.encode() + struct.pack('<I', len(shape))
        for d in shape:
            self.head += struct.pack('<Q', d)
        self.head += struct.pack('<I', ttype) + struct.pack('<Q', len(self.data))
        self.data += payload
        self.data += b'\0' * (-len(self.data) % ALIGN)
        self.n_tensors += 1

    def finish(self, path):
        struct.pack_into('<Q', self.head, 8, self.n_tensors)
        struct.pack_into('<Q', self.head, 16, self.n_kv)
        self.head += b'\0' * (-len(self.head) % ALIGN)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.head)
            f.write(self.data)
        return len(self.head) + len(self.data)


# ------------------------------------------------------------------ the model
class Model:
    """Collects the tensors in the engine's names and shapes (GGUF order: dim 0 varies fastest) and writes them
    with payloads drawn from a fixed seed."""

    def __init__(self, seed):
        self.seed = seed
        self.ts = []

    def add(self, name, shape, ttype, amp=0.05, scale=0.02):
        n = 1
        for d in shape:
            n *= d
        rng = rng_for(name, self.seed)
        if ttype == TY_F32:
            payload = f32_bytes(floats(rng, n, amp))
        elif ttype == TY_F16:
            payload = f16_bytes(floats(rng, n, amp))
        elif ttype == TY_Q8_0:
            payload = q8_0_bytes(floats(rng, n, amp))
        elif ttype == TY_IQ2_XXS:
            payload = iq2_xxs_bytes(rng, n, scale)
        else:
            raise SystemExit('no payload writer for type %d' % ttype)
        self.ts.append((name, shape, ttype, payload))


def ratios_for(n_layer):
    """The release's [0, 0, 2, 1] cycled, with the last block always ratio 1: it is the candidate source and a
    kv / index source of its own, which is what the closure rule in ds41_read_geometry asks for."""
    r = ([0, 0, 2, 1][i % 4] for i in range(n_layer))
    out = list(r)
    out[-1] = 1
    return out


def build(out, n_layer, seed, *, shared_sources=False):
    D, FF, NE, TOPK = 256, 256, 16, 2         # hidden, moe intermediate, experts (>= 16: the engine's routing
                                              # prefetch argsorts 16), experts per token
    NH, DH, DROPE = 4, 64, 8                  # heads, head dim (CUDA flash-attn size), rope dims of each head
    RQ, OL, OGRP, HCM = 32, 16, 2, 4          # q lora, o lora, o groups, hyper-connection count
    IH, ID, ITOP = 32, 128, 8                 # indexer heads / key dim (CUDA kernel sizes), indexer top-k
    VOCAB, CTX, SWA = 256, 4096, 8
    EG_HEADS, EG_DIM, EG_NGRAM, EG_PAD, EG_CVOCAB = 2, 64, 4, 0, 32
    EG_PRIMES = [719, 727, 733, 739, 743, 751]          # 6 = (ngram-1) * heads; they tile the table
    EG_ROWS = sum(EG_PRIMES)                            # 4412 rows, the Engram layer's table
    EG_MULT = [1, 257, 65537, 16908291]                 # ngram multipliers, all positive
    MIX = (2 + HCM) * HCM                               # hc_mix()

    ratio = ratios_for(n_layer)
    sources = [l for l in range(n_layer) if ratio[l]]   # every compressed layer is its own source
    cand_layer = max(l for l in range(n_layer) if ratio[l] == 1)
    index_sources = sources
    if shared_sources:
        if n_layer < 8:
            raise ValueError('shared sources need at least 8 blocks')
        ratio = [0, 0, 2, 2, 2] + [1] * (n_layer - 5)
        sources = [2, 5]
        index_sources = [2, 4, 5, 7]
        cand_layer = 5
    engram_layers = [1] if n_layer >= 2 else []

    m = Model(seed)
    m.add('token_embd.weight', [D, VOCAB], TY_F16)
    m.add('output_norm.weight', [D], TY_F32, amp=1.0)
    m.add('output.weight', [D, VOCAB], TY_Q8_0)
    for l in range(n_layer):
        b = 'blk.%d.' % l
        for site in ('attn', 'ffn'):
            m.add(b + 'hc_%s_fn.weight' % site, [HCM * D, MIX], TY_F16)
            m.add(b + 'hc_%s_base.weight' % site, [MIX], TY_F32)
            m.add(b + 'hc_%s_scale.weight' % site, [3], TY_F32)
            m.add(b + '%s_norm.weight' % site, [D], TY_F32, amp=1.0)
        m.add(b + 'attn_sinks.weight', [NH], TY_F32)
        m.add(b + 'attn_q_a.weight', [D, RQ], TY_Q8_0)
        m.add(b + 'attn_q_b.weight', [RQ, NH * DH], TY_Q8_0)
        m.add(b + 'attn_q_a_norm.weight', [RQ], TY_F32, amp=1.0)
        m.add(b + 'attn_kv.weight', [D, DH], TY_Q8_0)
        m.add(b + 'attn_kv_a_norm.weight', [DH], TY_F32, amp=1.0)
        m.add(b + 'attn_output_a.weight', [NH * DH // OGRP, OGRP * OL], TY_Q8_0)
        m.add(b + 'attn_output_b.weight', [OGRP * OL, D], TY_Q8_0)
        if l in sources:
            m.add(b + 'attn_compressor_kv.weight', [D, DH], TY_F16)
            m.add(b + 'attn_compressor_norm.weight', [DH], TY_F32, amp=1.0)
            m.add(b + 'indexer.attn_k.weight', [DH, ID], TY_F16)
            m.add(b + 'indexer.k_norm.weight', [ID], TY_F32, amp=1.0)
            if ratio[l] > 1:
                m.add(b + 'attn_compressor_gate.weight', [D, DH], TY_F16)
        if l in index_sources:
            m.add(b + 'indexer.attn_q_b.weight', [RQ, IH * ID], TY_F16)
            m.add(b + 'indexer.proj.weight', [D, IH], TY_F16)
        m.add(b + 'ffn_gate_inp.weight', [D, NE], TY_F32)
        m.add(b + 'exp_probs_b.bias', [NE], TY_F32)
        m.add(b + 'exp_probs_b_vl.bias', [NE], TY_F32)
        m.add(b + 'ffn_gate_shexp.weight', [D, FF], TY_Q8_0)
        m.add(b + 'ffn_up_shexp.weight', [D, FF], TY_Q8_0)
        m.add(b + 'ffn_down_shexp.weight', [FF, D], TY_Q8_0)
        if l in engram_layers:
            m.add(b + 'engram_q_norm.weight', [D, HCM], TY_F32, amp=1.0)
            m.add(b + 'engram_k_norm.weight', [D, HCM], TY_F32, amp=1.0)
            m.add(b + 'engram_kv.weight', [(EG_NGRAM - 1) * EG_HEADS * EG_DIM, (HCM + 1) * D], TY_F16)
            # the table: not in the predicted map, but ds41_check_tensors accepts it as this model's Q8_0
            # Engram table, and ds41_engram reads its rows straight off the shard
            m.add(b + 'engram_embd.weight', [EG_DIM, EG_ROWS], TY_Q8_0)
        m.add(b + 'ffn_gate_exps.weight', [D, FF, NE], TY_IQ2_XXS)
        m.add(b + 'ffn_up_exps.weight', [D, FF, NE], TY_IQ2_XXS)
        m.add(b + 'ffn_down_exps.weight', [FF, D, NE], TY_IQ2_XXS)

    w = Writer()
    P = 'deepseek41.'
    w.kv_str('general.architecture', 'deepseek41')
    w.kv_str('general.name', 'v41-tiny')
    for k, v in (('hidden_size', D), ('num_hidden_layers', n_layer), ('num_attention_heads', NH),
                 ('head_dim', DH), ('qk_rope_head_dim', DROPE), ('q_lora_rank', RQ), ('o_lora_rank', OL),
                 ('o_groups', OGRP), ('sliding_window', SWA), ('n_routed_experts', NE),
                 ('num_experts_per_tok', TOPK), ('moe_intermediate_size', FF), ('n_shared_experts', 1),
                 ('hc_mult', HCM), ('hc_sinkhorn_iters', 20), ('index_n_heads', IH), ('index_head_dim', ID),
                 ('index_topk', ITOP), ('max_position_embeddings', CTX), ('vocab_size', VOCAB),
                 ('candidate_source_layer_id', cand_layer), ('candidate_topk_blocks', 2048),
                 ('candidate_block_size', 8)):
        w.kv_u32(P + k, v)
    for k, v in (('routed_scaling_factor', 2.5), ('hc_eps', 1e-6), ('rms_norm_eps', 1e-6),
                 ('rope_theta', 160000.0), ('compress_rope_theta', 160000.0), ('rope_scaling.factor', 16.0),
                 ('rope_scaling.beta_fast', 32.0), ('rope_scaling.beta_slow', 1.0),
                 ('rope_scaling.original_max_position_embeddings', 4096.0), ('swiglu_limit', 10.0)):
        w.kv_f64(P + k, v)
    w.kv_bool(P + 'norm_topk_prob', True)
    w.kv_str(P + 'scoring_func', 'sqrtsoftplus')
    w.kv_arr(P + 'compress_ratios', MV_U32, 'I', ratio)
    w.kv_arr(P + 'kv_source_layer_ids', MV_U32, 'I', sources)
    w.kv_arr(P + 'index_source_layer_ids', MV_U32, 'I', index_sources)
    # the Engram shape the flat keys lack (ds41_read_geometry reads these three out of this JSON)
    w.kv_str(P + 'config', '{"engram_n_heads": %d, "engram_head_dim": %d, "engram_max_ngram_size": %d}'
             % (EG_HEADS, EG_DIM, EG_NGRAM))
    w.kv_u32(P + 'engram.pad_id', EG_PAD)
    w.kv_u32(P + 'engram.compressed_vocab_size', EG_CVOCAB)
    # the rows of blk.N.engram_embd.weight are Q8_0 blocks (fp16 scale + 32 int8 codes per 32 values), which is
    # the encoding ds41_engram::open_gguf reads; the HF sidecar's e4m3_e8m0_32_row264 is the other one.
    w.kv_str(P + 'engram.encoding', 'q8_0_32_row%d' % (EG_DIM // 32 * 34))
    w.kv_arr(P + 'engram.layer_ids', MV_U32, 'I', engram_layers)
    w.kv_arr(P + 'engram.rows', MV_U32, 'I', [EG_ROWS] * len(engram_layers))
    w.kv_arr(P + 'engram.token_map', MV_U32, 'I', [i % EG_CVOCAB for i in range(VOCAB)])
    w.kv_arr(P + 'engram.primes', MV_U64, 'Q', EG_PRIMES * len(engram_layers))
    w.kv_arr(P + 'engram.multipliers', MV_I64, 'q', EG_MULT * len(engram_layers))

    total = 0
    for name, shape, ttype, payload in m.ts:
        w.tensor(name, shape, ttype, payload)
        total += len(payload)
    size = w.finish(out)
    return size, total, len(m.ts), ratio, sources, cand_layer, engram_layers, EG_ROWS, (D, FF, NE, TOPK)


def main():
    ap = argparse.ArgumentParser(description='write a tiny valid deepseek41 GGUF')
    ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261019)
    ap.add_argument('--shared-sources', action='store_true', help='share ratio-1/2 KV and index sources (needs >= 8 blocks)')
    a = ap.parse_args()
    if a.blocks < 1:
        raise SystemExit('--blocks must be >= 1')
    if a.shared_sources and a.blocks < 8:
        ap.error('--shared-sources needs --blocks >= 8')
    size, payload, nt, ratio, sources, cand, eg_layers, eg_rows, (D, FF, NE, TOPK) = build(
        a.out, a.blocks, a.seed, shared_sources=a.shared_sources)
    print('%s: %d B (%d tensors, %d B of payload), %d blocks hidden %d ff %d experts %d/%d, '
          'ratios %s, %s sources %s, candidate L%d 2048x8, engram %s rows %d, IQ2_XXS experts'
          % (a.out, size, nt, payload, a.blocks, D, FF, NE, TOPK, ratio, 'kv' if a.shared_sources else 'kv/index',
             sources, cand, eg_layers or 'off', eg_rows))
    if a.shared_sources:
        print('shared index sources [2, 4, 5, 7]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
