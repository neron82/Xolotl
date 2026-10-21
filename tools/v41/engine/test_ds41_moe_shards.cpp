// tools/v41/engine/test_ds41_moe_shards.cpp - the V4.1 file tier against a SPLIT (multi-shard) GGUF.
//
// One synthetic `deepseek41` model - one routed layer, 8 experts, gate/up AND down IQ2_XXS (16), which is what the
// acrogenesis conversion wrote for `ffn_{gate,up,down}_shards.weight` - is written twice: once as a single file,
// once as two `-0000N-of-00002.gguf` shards with the expert tensors split across them, so every blob is assembled
// from two files.  The tier must see exactly the same bytes either way: the same geometry, the same blob bytes
// through `read` / `read_part` (1..8 parts) / `read_staged`, the same CPU-tier outputs, and a blob mirror whose
// header records the model's size (the sum of the shard sizes for a split model, the file's size for a single one).
// CPU-only: no CUDA, no model file, no network.
#include "ds41_moe.hpp"
#include "ds41_geometry.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

int fails = 0;
void say(bool ok, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsprintf(buf, fmt, ap);
    va_end(ap);
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", buf);
    fails += !ok;
}

// ------------------------------------------------------------------ the model
// GGUF / ggml type ids.  IQ2_XXS in every expert role: the acrogenesis file's expert format, and the format
// change 1 adds to the CUDA down list (STRATA_D_FMTS) and to the grouped-parity gate's down list.
constexpr uint32_t TY_F32 = 0, TY_F16 = 1, TY_Q8_0 = 8, TY_IQ2_XXS = 16;

// The real V4.1 shrunk to a test: hidden 512, ONE routed layer (`strata.v41.partial_layers`), 8 experts, FF 256,
// top-2.  512 and 256 are whole 256-element IQ2_XXS blocks, and every dense shape `ds41_check_tensors` predicts is
// derived from these numbers, so the file it describes is the file we write.
constexpr int64_t D = 512, FF = 256, NE = 8, TOPK = 2, VOCAB = 64, NL = 1;
constexpr int64_t NH = 4, DH = 128, DROPE = 64, RQ = 64, OLORA = 64, OGRP = 2, HCM = 4;
constexpr int64_t MIX = (2 + HCM) * HCM;   // hc_mix()

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    block_geometry((uint32_t) type, be, bb);
    return (size_t) (n / be) * (size_t) bb;
}

// The payload of one tensor, generated from its name so the single-file and the split file hold the same bytes.
// An IQ2_XXS block is an F16 scale followed by 64 code bytes; the scale is pinned to 1.0 so the tier's math on
// random codes stays finite (and large enough that the SwiGLU clamp binds).
std::vector<uint8_t> payload(const std::string& name, uint32_t type, uint64_t n) {
    uint64_t h = 14695981039346656037ull;   // FNV-1a over the name
    for (unsigned char c : name) h = (h ^ c) * 1099511628211ull;
    std::mt19937_64 rng(h);
    std::vector<uint8_t> b((size_t) n);
    for (auto& x : b) x = (uint8_t) (rng() >> 24);
    if (type == TY_IQ2_XXS)
        for (size_t i = 0; i + 66 <= b.size(); i += 66) { b[i] = 0x00; b[i + 1] = 0x3C; }   // 1.0h
    return b;
}

struct Model {
    struct T {
        std::string name;
        std::vector<uint64_t> shape;
        uint32_t type;
        std::vector<uint8_t> bytes;
    };
    std::vector<T> ts;
    void add(const std::string& name, std::vector<uint64_t> sh, uint32_t ty) {
        TensorInfo probe;
        probe.shape = sh;
        probe.type = ty;
        const uint64_t n = tensor_payload_bytes(probe);
        if (!n) {
            std::fprintf(stderr, "%s: no byte count for that shape/type\n", name.c_str());
            std::exit(1);
        }
        ts.push_back({name, std::move(sh), ty, payload(name, ty, n)});
    }
    const T& at(const std::string& name) const {
        for (const T& t : ts)
            if (t.name == name) return t;
        std::fprintf(stderr, "%s: not in the model\n", name.c_str());
        std::exit(1);
    }
};

Model make_model() {
    Model m;
    m.add("token_embd.weight", {D, (uint64_t) VOCAB}, TY_F16);
    m.add("output_norm.weight", {D}, TY_F32);
    m.add("output.weight", {D, (uint64_t) VOCAB}, TY_Q8_0);
    const std::string b = "blk.0.";
    for (const char* site : {"attn", "ffn"}) {
        m.add(b + "hc_" + site + "_fn.weight", {(uint64_t) HCM * D, MIX}, TY_F16);
        m.add(b + "hc_" + site + "_base.weight", {MIX}, TY_F32);
        m.add(b + "hc_" + site + "_scale.weight", {3}, TY_F32);
        m.add(b + site + "_norm.weight", {D}, TY_F32);
    }
    m.add(b + "attn_sinks.weight", {(uint64_t) NH}, TY_F32);
    m.add(b + "attn_q_a.weight", {D, (uint64_t) RQ}, TY_Q8_0);
    m.add(b + "attn_q_b.weight", {(uint64_t) RQ, (uint64_t) NH * DH}, TY_Q8_0);
    m.add(b + "attn_q_a_norm.weight", {(uint64_t) RQ}, TY_F32);
    m.add(b + "attn_kv.weight", {D, (uint64_t) DH}, TY_Q8_0);
    m.add(b + "attn_kv_a_norm.weight", {(uint64_t) DH}, TY_F32);
    m.add(b + "attn_output_a.weight", {(uint64_t) NH * DH / OGRP, (uint64_t) OGRP * OLORA}, TY_Q8_0);
    m.add(b + "attn_output_b.weight", {(uint64_t) OGRP * OLORA, D}, TY_Q8_0);
    m.add(b + "ffn_gate_inp.weight", {D, (uint64_t) NE}, TY_F32);
    m.add(b + "exp_probs_b.bias", {(uint64_t) NE}, TY_F32);
    m.add(b + "exp_probs_b_vl.bias", {(uint64_t) NE}, TY_F32);
    m.add(b + "ffn_gate_shexp.weight", {D, (uint64_t) FF}, TY_Q8_0);
    m.add(b + "ffn_up_shexp.weight", {D, (uint64_t) FF}, TY_Q8_0);
    m.add(b + "ffn_down_shexp.weight", {(uint64_t) FF, D}, TY_Q8_0);
    // routed experts, uniform roles (no `ffn_exps_{gu,down}_type` maps): gate/up [D, FF, NE], down [FF, D, NE]
    m.add(b + "ffn_gate_exps.weight", {D, (uint64_t) FF, (uint64_t) NE}, TY_IQ2_XXS);
    m.add(b + "ffn_up_exps.weight", {D, (uint64_t) FF, (uint64_t) NE}, TY_IQ2_XXS);
    m.add(b + "ffn_down_exps.weight", {(uint64_t) FF, D, (uint64_t) NE}, TY_IQ2_XXS);
    return m;
}

// ------------------------------------------------------------------ GGUF v3 writer
constexpr uint32_t MV_U32 = 4, MV_U64 = 10, MV_I64 = 11, MV_F64 = 12, MV_BOOL = 7, MV_STRING = 8, MV_ARRAY = 9;

class Writer {
public:
    explicit Writer(uint64_t align = 32) : align_(align) {
        put((uint32_t) 0x46554747u);   // "GGUF"
        put((uint32_t) 3);             // v3, what strata::GgufFile reads
        n_t_ = head_.size();
        put((uint64_t) 0);   // n_tensors, patched at finish
        put((uint64_t) 0);   // n_kv, ditto
    }
    void kv_str(const std::string& k, const std::string& v) {
        std::vector<uint8_t> p;
        put(p, (uint64_t) v.size());
        p.insert(p.end(), v.begin(), v.end());
        kv(k, MV_STRING, p);
    }
    template <class T> void kv_num(const std::string& k, uint32_t type, T v) {
        std::vector<uint8_t> p(sizeof(T));
        std::memcpy(p.data(), &v, sizeof(T));
        kv(k, type, p);
    }
    template <class T> void kv_arr(const std::string& k, uint32_t elem, const std::vector<T>& v) {
        std::vector<uint8_t> p;
        put(p, elem);
        put(p, (uint64_t) v.size());
        for (const T& x : v) {
            const size_t at = p.size();
            p.resize(at + sizeof(T));
            std::memcpy(p.data() + at, &x, sizeof(T));
        }
        kv(k, MV_ARRAY, p);
    }
    void tensor(const Model::T& t) {
        put((uint64_t) t.name.size());
        head_.insert(head_.end(), t.name.begin(), t.name.end());
        put((uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put(d);
        put(t.type);
        put(data_.size());   // offset from the start of this file's data section
        data_.insert(data_.end(), t.bytes.begin(), t.bytes.end());
        pad(data_);
        ++n_t;
    }
    bool finish(const std::string& path) {
        std::memcpy(head_.data() + n_t_, &n_t, 8);
        const uint64_t nkv = n_kv;
        std::memcpy(head_.data() + n_t_ + 8, &nkv, 8);
        pad(head_);
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write((const char*) head_.data(), (std::streamsize) head_.size());
        f.write((const char*) data_.data(), (std::streamsize) data_.size());
        return (bool) f;
    }

private:
    template <class T> void put(T v) {
        const size_t at = head_.size();
        head_.resize(at + sizeof(T));
        std::memcpy(head_.data() + at, &v, sizeof(T));
    }
    template <class T> static void put(std::vector<uint8_t>& v, T x) {
        const size_t at = v.size();
        v.resize(at + sizeof(T));
        std::memcpy(v.data() + at, &x, sizeof(T));
    }
    void pad(std::vector<uint8_t>& v) const {
        const size_t r = v.size() % align_;
        if (r) v.resize(v.size() + align_ - r);
    }
    void kv(const std::string& k, uint32_t type, const std::vector<uint8_t>& p) {
        put((uint64_t) k.size());
        head_.insert(head_.end(), k.begin(), k.end());
        put(type);
        head_.insert(head_.end(), p.begin(), p.end());
        ++n_kv;
    }
    uint64_t align_, n_t_ = 0;
    uint64_t n_t = 0, n_kv = 0;
    std::vector<uint8_t> head_, data_;
};

// The model's metadata, in the engine's own (v41_gguf.py) spelling - the file that satisfies the native contract,
// so ds41_read_geometry never consults its alias table.
void add_meta(Writer& w) {
    w.kv_str("general.architecture", "deepseek41");
    w.kv_str("general.name", "ds41-shards-test");
    const std::string P = "deepseek41.";
    struct U { const char* k; int64_t v; };
    for (const U& u : std::initializer_list<U>{
             {"hidden_size", D}, {"num_hidden_layers", NL}, {"num_attention_heads", NH}, {"head_dim", DH},
             {"qk_rope_head_dim", DROPE}, {"q_lora_rank", RQ}, {"o_lora_rank", OLORA}, {"o_groups", OGRP},
             {"sliding_window", 128}, {"n_routed_experts", NE}, {"num_experts_per_tok", TOPK},
             {"moe_intermediate_size", FF}, {"n_shared_experts", 1}, {"hc_mult", HCM}, {"hc_sinkhorn_iters", 20},
             {"index_n_heads", NH}, {"index_head_dim", DH}, {"index_topk", 512}, {"vocab_size", VOCAB},
             {"max_position_embeddings", 4096}, {"candidate_source_layer_id", 0}, {"candidate_topk_blocks", 2048},
             {"candidate_block_size", 8}})
        w.kv_num(P + u.k, MV_U32, (uint32_t) u.v);
    struct F { const char* k; double v; };
    for (const F& f : std::initializer_list<F>{{"routed_scaling_factor", 2.5}, {"hc_eps", 1e-6}, {"rms_norm_eps", 1e-6},
                                               {"rope_theta", 160000.0}, {"compress_rope_theta", 160000.0},
                                               {"rope_scaling.factor", 16.0}, {"rope_scaling.beta_fast", 32.0},
                                               {"rope_scaling.beta_slow", 1.0},
                                               {"rope_scaling.original_max_position_embeddings", 4096.0},
                                               {"swiglu_limit", 10.0}})
        w.kv_num(P + f.k, MV_F64, f.v);
    w.kv_num(P + "norm_topk_prob", MV_BOOL, (uint8_t) 1);
    w.kv_str(P + "scoring_func", "sqrtsoftplus");
    // one layer, uncompressed: no kv/index source layers, no Engram layer, so no Engram tensors either
    w.kv_arr(P + "compress_ratios", MV_U32, std::vector<uint32_t>{0});
    w.kv_arr(P + "kv_source_layer_ids", MV_U32, std::vector<uint32_t>{});
    w.kv_arr(P + "index_source_layer_ids", MV_U32, std::vector<uint32_t>{});
    w.kv_str(P + "config", "{\"engram_n_heads\": 2, \"engram_head_dim\": 64, \"engram_max_ngram_size\": 4}");
    w.kv_num(P + "engram.pad_id", MV_U32, 0u);
    w.kv_num(P + "engram.compressed_vocab_size", MV_U32, 32u);
    w.kv_arr(P + "engram.layer_ids", MV_U32, std::vector<uint32_t>{});
    w.kv_arr(P + "engram.rows", MV_U32, std::vector<uint32_t>{});
    std::vector<uint32_t> tm((size_t) VOCAB);
    for (int64_t i = 0; i < VOCAB; ++i) tm[(size_t) i] = (uint32_t) (i % 32);
    w.kv_arr(P + "engram.token_map", MV_U32, tm);
    w.kv_arr(P + "engram.primes", MV_U64, std::vector<uint64_t>{});
    w.kv_arr(P + "engram.multipliers", MV_I64, std::vector<int64_t>{});
    // the file holds layer 0 only
    w.kv_arr("strata.v41.partial_layers", MV_U32, std::vector<uint32_t>{0});
}

void write_single(const std::string& path, const Model& m) {
    Writer w;
    add_meta(w);
    for (const Model::T& t : m.ts) w.tensor(t);
    if (!w.finish(path)) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }
}

// Two shards, named the way strata::gguf_split_paths expects.  Shard 1 takes every block tensor except the gate
// experts, so each expert blob is gate rows from shard 0 and up + down rows from shard 1.
void write_split(const std::string& a, const std::string& b, const Model& m) {
    const uint64_t total = m.ts.size();
    auto in_second = [](const std::string& n) {
        return n.rfind("blk.0.", 0) == 0 && n.find("ffn_gate_exps") == std::string::npos;
    };
    Writer w0, w1;
    add_meta(w0);
    w0.kv_num("split.count", MV_U32, 2u);
    w0.kv_num("split.no", MV_U32, 0u);
    w0.kv_num("split.tensors.count", MV_U32, (uint32_t) total);
    w1.kv_num("split.count", MV_U32, 2u);
    w1.kv_num("split.no", MV_U32, 1u);
    w1.kv_num("split.tensors.count", MV_U32, (uint32_t) total);
    for (const Model::T& t : m.ts) (in_second(t.name) ? w1 : w0).tensor(t);
    if (!w0.finish(a) || !w1.finish(b)) {
        std::fprintf(stderr, "cannot write %s / %s\n", a.c_str(), b.c_str());
        std::exit(1);
    }
}

// ------------------------------------------------------------------ fixtures
uint64_t fsize(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? (uint64_t) st.st_size : 0;
}

// the mirror header + index (ds41_moe.cpp's own layout, re-declared here so the test can read the file back)
struct Head { char magic[8]; uint64_t src, n, data; };
struct Ent { int32_t l, e; uint64_t off; };

std::vector<uint8_t> slurp(const std::string& p, size_t from = 0) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const size_t n = (size_t) f.tellg();
    std::vector<uint8_t> b(n - from);
    f.seekg((std::streamoff) from);
    f.read((char*) b.data(), (std::streamsize) b.size());
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "ds41_shards_test";
    ::mkdir(dir.c_str(), 0775);
    const Model m = make_model();
    const std::string one = dir + "/v41_shards_one.gguf";
    const std::string s0 = dir + "/v41_shards_two-00001-of-00002.gguf";
    const std::string s1 = dir + "/v41_shards_two-00002-of-00002.gguf";
    write_single(one, m);
    write_split(s0, s1, m);
    std::printf("single %s (%llu B); split %s (%llu B) + %s (%llu B), %zu tensors\n", one.c_str(),
                (unsigned long long) fsize(one), s0.c_str(), (unsigned long long) fsize(s0), s1.c_str(),
                (unsigned long long) fsize(s1), m.ts.size());

    std::string err;
    Ds41FileBlobs a(one), b(s0);
    if (!a.open(err)) { std::fprintf(stderr, "open %s: %s\n", one.c_str(), err.c_str()); return 1; }
    if (!b.open(err)) { std::fprintf(stderr, "open %s: %s\n", s0.c_str(), err.c_str()); return 1; }

    // ---- the geometry the tier is built from
    const Ds41MoeGeom& ga = a.geom();
    const Ds41MoeGeom& gb = b.geom();
    say(ga.n_layers == NL && ga.n_experts == NE && ga.n_embd == D && ga.n_ff == FF && ga.top_k == TOPK,
        "single-file geometry: %lld layers x %lld experts, %lld x %lld, top-%lld", (long long) ga.n_layers,
        (long long) ga.n_experts, (long long) ga.n_embd, (long long) ga.n_ff, (long long) ga.top_k);
    say(ga.cls_gu == gb.cls_gu && ga.cls_d == gb.cls_d && ga.eclass == gb.eclass && ga.swiglu_limit == gb.swiglu_limit &&
                ga.gate_types == gb.gate_types && ga.up_types == gb.up_types && ga.down_types == gb.down_types,
        "split model reports the same geometry (class %d/%d = IQ2_XXS gate+up / IQ2_XXS down)", ga.cls_gu[0],
        ga.cls_d[0]);
    say(a.blob_bytes() == b.blob_bytes() && a.blob_bytes(0) == b.blob_bytes(0), "blob_bytes the same: %zu B",
        a.blob_bytes(0));
    bool same_eb = true;
    for (int64_t e = 0; e < NE; ++e) same_eb = same_eb && a.expert_bytes(0, e) == b.expert_bytes(0, e);
    say(same_eb, "expert_bytes the same for all %lld experts", (long long) NE);

    // ---- read(): the whole blob, every expert
    const size_t bb = a.blob_bytes(0);
    const size_t gb_len = row_bytes(ga.cls_gu[0], D) * (size_t) FF, db_len = row_bytes(ga.cls_d[0], FF) * (size_t) D;
    std::vector<uint8_t> x(bb), y(bb);
    bool same = true;
    for (int64_t e = 0; e < NE; ++e) {
        std::memset(x.data(), 0xA5, bb);
        std::memset(y.data(), 0x5A, bb);
        a.read(0, e, x.data());
        b.read(0, e, y.data());
        same = same && std::memcmp(x.data(), y.data(), bb) == 0;
    }
    say(same, "read(): all %lld blobs identical between the two layouts", (long long) NE);

    // ...and they are the tensors' own rows for that slot, not merely equal to each other
    const Model::T &gt = m.at("blk.0.ffn_gate_exps.weight"), &ut = m.at("blk.0.ffn_up_exps.weight"),
                     &dt = m.at("blk.0.ffn_down_exps.weight");
    same = true;
    for (int64_t e = 0; e < NE; ++e) {
        b.read(0, e, y.data());
        same = same && std::memcmp(y.data(), gt.bytes.data() + (size_t) e * gb_len, gb_len) == 0 &&
                       std::memcmp(y.data() + gb_len, ut.bytes.data() + (size_t) e * gb_len, gb_len) == 0 &&
                       std::memcmp(y.data() + 2 * gb_len, dt.bytes.data() + (size_t) e * db_len, db_len) == 0;
    }
    say(same, "the split model's blobs are the gate/up/down tensors' rows for slot e (%zu|%zu|%zu B spans)", gb_len,
        gb_len, db_len);

    // ---- read_part(): the parts tile the blob, from one or from two files
    for (int np : {1, 2, 3, 5, 8}) {
        same = true;
        for (int64_t e = 0; e < NE; ++e) {
            std::memset(x.data(), 0xA5, bb);
            std::memset(y.data(), 0x5A, bb);
            for (int p = 0; p < np; ++p) {
                a.read_part(0, e, p, np, x.data());
                b.read_part(0, e, p, np, y.data());
            }
            same = same && std::memcmp(x.data(), y.data(), bb) == 0;
        }
        say(same, "read_part with %d parts: identical for all %lld blobs", np, (long long) NE);
    }

    // ---- read_staged(): the chunk reader's O_DIRECT path, per shard
    const size_t sc = a.read_staged_scratch();
    say(sc == b.read_staged_scratch() && sc > 0, "read_staged_scratch the same: %zu B", sc);
    {
        std::vector<uint8_t> sa(sc), sb(sc);
        same = true;
        for (int64_t e = 0; e < NE; ++e) {
            std::memset(x.data(), 0xA5, bb);
            std::memset(y.data(), 0x5A, bb);
            a.read_staged(0, e, x.data(), sa.data(), sc);
            b.read_staged(0, e, y.data(), sb.data(), sb.size());
            same = same && std::memcmp(x.data(), y.data(), bb) == 0;
        }
        say(same, "read_staged (O_DIRECT bounces): identical for all %lld blobs", (long long) NE);
    }
    for (int64_t e = 0; e < NE; ++e) {   // fadvise the shard that holds each span
        a.advise(0, e);
        b.advise(0, e);
    }
    say(true, "advise() ran on both sources (per-span fadvise on the holding shard)");

    // ---- the CPU tier on both: same bytes in, same numbers out
    Ds41MoeConfig cfg;
    cfg.cpu_only = true;
    cfg.threads = 4;
    cfg.slots = 8;
    cfg.arena_gib = 0.05;
    cfg.max_arena_gib = 0.05;
    cfg.pcie_frac = 0.5;
    cfg.pf_b = 0;
    Ds41MoeTier ta, tb;
    if (!ta.init(ga, &a, cfg, err)) { std::fprintf(stderr, "tier init (single): %s\n", err.c_str()); return 1; }
    if (!tb.init(gb, &b, cfg, err)) { std::fprintf(stderr, "tier init (split): %s\n", err.c_str()); return 1; }
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    const int64_t K = ga.top_k;
    std::vector<float> in((size_t) D), oa((size_t) D), ob((size_t) D);
    same = true;
    bool finite = true;
    for (int t = 0; t < 4; ++t) {
        for (auto& v : in) v = nd(rng);
        std::vector<int32_t> ids((size_t) K);
        std::vector<float> w((size_t) K);
        for (int64_t k = 0; k < K; ++k) {   // deterministic picks that walk every expert, both shards
            ids[(size_t) k] = (int32_t) ((t * 3 + k * 3) % NE);
            w[(size_t) k] = 0.5f + 0.1f * (float) k;
        }
        if (!ta.run(0, ids.data(), w.data(), in.data(), oa.data()) ||
            !tb.run(0, ids.data(), w.data(), in.data(), ob.data())) {
            say(false, "tier run failed");
            return 1;
        }
        same = same && std::memcmp(oa.data(), ob.data(), (size_t) D * sizeof(float)) == 0;
        for (float v : oa) finite = finite && std::isfinite(v);
    }
    say(same, "CPU tier run(): bitwise-identical outputs from the two layouts (4 tokens, top-%lld)", (long long) K);
    say(finite, "the tier's outputs are finite (the IQ2_XXS blobs are real weights, not a NaN pass)");

    // ---- the blob mirror: built from either layout, sized by the model (the sum of the shards for a split one)
    std::vector<std::pair<int32_t, int32_t>> order;
    for (int64_t e = 0; e < NE; ++e) order.emplace_back(0, (int32_t) e);
    const std::string ma = dir + "/mirror_one.bin", mb = dir + "/mirror_two.bin";
    if (!a.build_mirror(ma, order, ~0ull, err)) { std::fprintf(stderr, "build_mirror (single): %s\n", err.c_str()); return 1; }
    if (!b.build_mirror(mb, order, ~0ull, err)) { std::fprintf(stderr, "build_mirror (split): %s\n", err.c_str()); return 1; }
    {
        std::vector<uint8_t> ha = slurp(ma, 0), hb = slurp(mb, 0);
        Head A{}, B{};
        std::memcpy(&A, ha.data(), sizeof A);
        std::memcpy(&B, hb.data(), sizeof B);
        say(std::memcmp(A.magic, "S41MIRR1", 8) == 0 && std::memcmp(B.magic, "S41MIRR1", 8) == 0 && A.n == B.n &&
                    A.data == B.data,
            "both mirrors carry the S41MIRR1 index (%llu blobs, data at %llu)", (unsigned long long) A.n,
            (unsigned long long) A.data);
        say(A.src == fsize(one) && B.src == fsize(s0) + fsize(s1),
            "src_size: single file %llu = its size; split %llu = the sum of the %llu + %llu of its shards",
            (unsigned long long) A.src, (unsigned long long) B.src, (unsigned long long) fsize(s0),
            (unsigned long long) fsize(s1));
        say(ha.size() == hb.size() && std::memcmp(ha.data() + A.data, hb.data() + B.data, ha.size() - A.data) == 0 &&
                    std::memcmp(ha.data() + sizeof(Head), hb.data() + sizeof(Head), A.n * sizeof(Ent)) == 0,
            "the two mirrors hold the same index and the same %zu blob bytes", ha.size() - A.data);
    }
    // the mirror in use: read_part serves its share of every part, from the same bytes
    {
        ::setenv("STRATA_BLOB_MIRROR", mb.c_str(), 1);
        Ds41FileBlobs c(s0);
        if (!c.open(err)) { std::fprintf(stderr, "open with mirror: %s\n", err.c_str()); return 1; }
        say(c.mirrored() == NE, "the split model opens with its mirror (%lld experts)", (long long) c.mirrored());
        same = true;
        for (int64_t e = 0; e < NE; ++e) {
            std::memset(x.data(), 0xA5, bb);
            std::memset(y.data(), 0x5A, bb);
            for (int p = 0; p < 8; ++p) {
                b.read_part(0, e, p, 8, y.data());
                c.read_part(0, e, p, 8, x.data());
            }
            same = same && std::memcmp(x.data(), y.data(), bb) == 0;
        }
        say(same, "read_part through the mirror: identical to the shards themselves");
        ::unsetenv("STRATA_BLOB_MIRROR");
        // the size check is real: the split model's mirror does not fit the single-file model
        ::setenv("STRATA_BLOB_MIRROR", mb.c_str(), 1);
        Ds41FileBlobs d(one);
        if (!d.open(err)) { std::fprintf(stderr, "open single with mirror: %s\n", err.c_str()); return 1; }
        say(d.mirrored() == 0, "a mirror built from the split model is refused for the single-file model (size)");
        ::unsetenv("STRATA_BLOB_MIRROR");
    }

    if (fails == 0)
        for (const std::string& p : {one, s0, s1, ma, mb}) ::unlink(p.c_str());
    std::printf("test_ds41_moe_shards: %d failures\n", fails);
    return fails ? 1 : 0;
}
