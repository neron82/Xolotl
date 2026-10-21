// tools/v41/engine/ds41_geometry.hpp - DeepSeek-V4.1-Flash (`deepseek41`) geometry, tensor map and expert tiers.
//
// The GGUF is the one tools/v41/v41_gguf.py writes (DwarfStar's dense names, per-expert routed tiers).  Every number
// comes from the file's `deepseek41.*` keys (or the `deepseek41.config` JSON for the Engram shape the flat keys lack);
// a missing key is an error, never a default.  `ds41_check_tensors` resolves the full tensor map: every name in the
// file must be predicted and every predicted name present, with the conditional sets exactly as model.py builds them:
// compressor tensors on kv source layers (gate only where ratio > 1), indexer query/proj on index source layers,
// indexer key on kv sources, Engram on its layers, routed experts either one stacked tensor per role or one per type
// + the I32 type maps.
//
// A file written by llama.cpp's converter - acrogenesis' DeepSeek-V4.1-Flash-1017 IQ2_XXS build, and the releases
// the recipe kit points at - is the same model in llama.cpp's spellings: `deepseek41.embedding_length` for
// `hidden_size`, `attention.head_count` for `num_attention_heads`, and so on (detail::canon).  Its Engram shape is
// in flat engram.head_count / key_length keys instead of the config JSON. Maya uses layers / head_dim /
// num_embeddings aliases and a pad_token_id in the original vocabulary (mapped through token_map here).
// Missing source-layer lists, candidate blocks, table rows and compressed vocabulary are derived from the
// tensors, release geometry, primes and token map. Native/canonical keys always take precedence over aliases.
// Types are not a spelling: a dense matrix stored as BF16 where the engine computes in F16 is converted at load
// (ds41_dense.cpp); mmap token embeddings are decoded one requested row at a time, and GGUF Engram tables may
// be Q8_0 or Q6_K (ds41_engram.cpp). Maya's F32 Engram gate stores the exact folded Q/K norm product.
//
// Header-only; reads the header region only (GgufModel mmaps it), no tensor data except the two I32 type maps.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace strata::v41 {

namespace ty {
constexpr uint32_t F32 = 0, F16 = 1, Q8_0 = 8, Q3_K = 11, Q6_K = 14, I8 = 24, I32 = 26, Q2_0 = 42, PTQ1_0 = 143;
}

/// One layer's routed experts: the type of each expert's gate+up and of its down, and where each expert's rows are
/// (tensor name + slot inside it).  Uniform roles have a single tensor (slot == expert id).
struct Ds41ExpertMap {
    std::vector<int> gu_type, d_type;              ///< [n_expert] ggml type ids
    std::vector<int> gu_slot, d_slot;              ///< [n_expert] index inside the type's stacked tensor
    std::map<int, std::string> gate_t, up_t, down_t;   ///< type -> tensor name
};

struct Ds41Geometry {
    int64_t n_embd = 0, n_layer = 0, n_head = 0, d_head = 0, d_rope = 0, r_q = 0, o_lora = 0, o_groups = 0;
    int64_t n_swa = 0, n_expert = 0, n_used = 0, n_ff = 0, n_shared = 0;
    double weights_scale = 0;
    bool norm_topk = false;
    int64_t hc = 0, hc_iters = 0;
    double hc_eps = 0, rms_eps = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0;
    int64_t cand_layer = -1, cand_blocks = 0, cand_bsize = 0;
    double rope_base = 0, compress_rope_base = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0, orig_ctx = 0;
    int64_t vocab = 0, ctx_len = 0;
    double swiglu_limit = 0;
    std::vector<int64_t> ratio;                      ///< [n_layer] 0 / 2 / 1
    std::vector<int64_t> kv_sources, index_sources;
    // Engram
    std::vector<int64_t> engram_layers, engram_rows; ///< layer ids, table rows per layer
    int64_t engram_heads = 0, engram_dim = 0, engram_ngram = 0, engram_pad = 0, engram_cvocab = 0;
    std::vector<uint32_t> engram_token_map;          ///< [vocab] compressed id
    std::vector<uint64_t> engram_primes;             ///< [n_engram_layers * (ngram-1) * heads]
    std::vector<int64_t> engram_mult;                ///< [n_engram_layers * ngram]
    int64_t bos = -1, eos = -1;
    std::vector<Ds41ExpertMap> experts;              ///< [n_layer]
    // A file in llama.cpp's spelling (the note at the top): what the file calls each tensor and what type it
    // stores it as, which the loader needs because they are not the engine's.  Empty for a file v41_gguf.py wrote.
    std::map<std::string, std::string> rename;         ///< file tensor name -> the engine's name for it
    std::map<std::string, uint32_t> want;              ///< engine tensor name -> the type the engine reads it as
    std::vector<std::string> engram_tables;            ///< the file's Engram table tensors, in engram_layers order

    bool is_kv_source(int64_t l) const { return contains(kv_sources, l); }
    bool is_index_source(int64_t l) const { return contains(index_sources, l); }
    int64_t engram_slot(int64_t l) const {
        for (size_t i = 0; i < engram_layers.size(); ++i) if (engram_layers[i] == l) return (int64_t) i;
        return -1;
    }
    /// The kv source whose compressed KV layer l reads (model.py shared_attn.compress_kv: the latest source <= l), or -1.
    int64_t kv_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : kv_sources) if (k <= l) s = k;
        return s;
    }
    /// The index source whose top-k layer l uses (the latest index source <= l), or -1.
    int64_t idx_src(int64_t l) const {
        if (ratio[(size_t) l] == 0) return -1;
        int64_t s = -1;
        for (int64_t k : index_sources) if (k <= l) s = k;
        return s;
    }
    int64_t hc_mix() const { return (2 + hc) * hc; }

private:
    static bool contains(const std::vector<int64_t> & v, int64_t x) {
        for (int64_t y : v) if (y == x) return true;
        return false;
    }
};

// ------------------------------------------------------------------ full metadata arrays
// strata::GgufFile keeps 64 items of an array (a sample); the Engram token map has 129,280.  This walks the KV
// section of the file itself and returns one key's whole array (integers only), or an error.
namespace detail {
struct Rd {
    int fd = -1;
    uint64_t pos = 0;
    std::vector<uint8_t> buf;
    uint64_t buf_at = 0;
    bool fill(uint64_t at, size_t n) {
        if (at >= buf_at && at + n <= buf_at + buf.size()) return true;
        const size_t want = std::max<size_t>(n, 1 << 20);
        buf.resize(want);
        const ssize_t r = pread(fd, buf.data(), want, (off_t) at);
        if (r < (ssize_t) n) return false;
        buf.resize((size_t) r);
        buf_at = at;
        return true;
    }
    bool get(void * dst, size_t n) {
        if (!fill(pos, n)) return false;
        std::memcpy(dst, buf.data() + (pos - buf_at), n);
        pos += n;
        return true;
    }
    template <class T> bool get(T & v) { return get(&v, sizeof(T)); }
};
inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;   // U8 I8 BOOL
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}
}  // namespace detail

inline std::string ds41_read_int_array(const std::string & path, const std::string & key, std::vector<int64_t> & out) {
    detail::Rd r;
    r.fd = ::open(path.c_str(), O_RDONLY);
    if (r.fd < 0) return "cannot open " + path;
    struct Closer { int fd; ~Closer() { ::close(fd); } } closer { r.fd };
    uint32_t magic = 0, version = 0;
    uint64_t n_t = 0, n_kv = 0;
    if (!r.get(magic) || magic != 0x46554747u || !r.get(version) || !r.get(n_t) || !r.get(n_kv)) return "bad GGUF header";
    auto skip_str = [&]() { uint64_t n; if (!r.get(n)) return false; r.pos += n; return true; };
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t kl = 0;
        if (!r.get(kl) || kl > 4096) return "bad key";
        std::string k(kl, '\0');
        if (!r.get(k.data(), kl)) return "bad key";
        uint32_t t = 0;
        if (!r.get(t)) return "bad type";
        if (t == 8) { if (!skip_str()) return "bad string"; continue; }
        if (t != 9) { const size_t s = detail::scalar_size(t); if (!s) return "bad scalar type"; r.pos += s; continue; }
        uint32_t et = 0;
        uint64_t cnt = 0;
        if (!r.get(et) || !r.get(cnt)) return "bad array";
        if (k != key) {
            if (et == 8) { for (uint64_t j = 0; j < cnt; ++j) if (!skip_str()) return "bad string array"; }
            else { const size_t s = detail::scalar_size(et); if (!s) return "bad array type"; r.pos += s * cnt; }
            continue;
        }
        const size_t s = detail::scalar_size(et);
        if (!s || et == 7 || et == 6 || et == 12) return key + ": not an integer array";
        out.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) {
            uint8_t b[8] = {};
            if (!r.get(b, s)) return key + ": truncated";
            int64_t v = 0;
            switch (et) {
                case 0: v = b[0]; break;
                case 1: v = (int8_t) b[0]; break;
                case 2: { uint16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 3: { int16_t x; std::memcpy(&x, b, 2); v = x; break; }
                case 4: { uint32_t x; std::memcpy(&x, b, 4); v = x; break; }
                case 5: { int32_t x; std::memcpy(&x, b, 4); v = x; break; }
                default: std::memcpy(&v, b, 8); break;
            }
            out[j] = v;
        }
        return {};
    }
    return "missing " + key;
}

// ------------------------------------------------------------------ key readers (missing key == error)
namespace detail {
// The same model in two spellings.  v41_gguf.py writes the name HuggingFace's config.json uses; llama.cpp's
// converter writes the name llama.cpp's own gguf keys use.  The left column is what this engine asks for.
inline const char * canon(const std::string & hf) {
    static const struct { const char * hf, * llama; } tab[] = {
        { "hidden_size", "embedding_length" },
        { "num_hidden_layers", "block_count" },
        { "num_attention_heads", "attention.head_count" },
        { "num_key_value_heads", "attention.head_count_kv" },
        { "head_dim", "attention.key_length" },
        { "qk_rope_head_dim", "rope.dimension_count" },
        { "q_lora_rank", "attention.q_lora_rank" },
        { "o_lora_rank", "attention.output_lora_rank" },
        { "o_groups", "attention.output_group_count" },
        { "sliding_window", "attention.sliding_window" },
        { "n_routed_experts", "expert_count" },
        { "num_experts_per_tok", "expert_used_count" },
        { "moe_intermediate_size", "expert_feed_forward_length" },
        { "n_shared_experts", "expert_shared_count" },
        { "hc_mult", "hyper_connection.count" },
        { "hc_sinkhorn_iters", "hyper_connection.sinkhorn_iterations" },
        { "hc_eps", "hyper_connection.epsilon" },
        { "index_n_heads", "attention.indexer.head_count" },
        { "index_head_dim", "attention.indexer.key_length" },
        { "index_topk", "attention.indexer.top_k" },
        { "compress_ratios", "attention.compress_ratios" },
        { "rms_norm_eps", "attention.layer_norm_rms_epsilon" },
        { "rope_theta", "rope.freq_base" },
        { "compress_rope_theta", "attention.compress_rope_freq_base" },
        { "rope_scaling.factor", "rope.scaling.factor" },
        { "rope_scaling.beta_fast", "rope.scaling.yarn_beta_fast" },
        { "rope_scaling.beta_slow", "rope.scaling.yarn_beta_slow" },
        { "rope_scaling.original_max_position_embeddings", "rope.scaling.original_context_length" },
        { "max_position_embeddings", "context_length" },
        { "routed_scaling_factor", "expert_weights_scale" },
        { "norm_topk_prob", "expert_weights_norm" },
        { "engram.layer_ids", "engram.layers" },
        { "engram.key_length", "engram.head_dim" },
        { "engram.rows", "engram.num_embeddings" },
    };
    for (const auto & a : tab) if (hf == a.hf) return a.llama;
    return nullptr;
}
/// `deepseek41.<hf>` under llama.cpp's spelling of it ("" when the key has no other spelling).
inline std::string canon_key(const std::string & k) {
    static const std::string P = "deepseek41.";
    if (k.compare(0, P.size(), P) != 0) return {};
    const char * c = canon(k.substr(P.size()));
    return c ? P + c : std::string();
}
/// A key, or the same key in the other spelling.  The engine's own spelling is always tried first.
inline const MetaValue * get2(const GgufFile & g, const std::string & k) {
    if (const MetaValue * v = g.get(k)) return v;
    const std::string a = canon_key(k);
    return a.empty() ? nullptr : g.get(a);
}
inline std::string u64(const GgufFile & g, const std::string & k, int64_t & out) {
    const MetaValue * v = get2(g, k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = (int64_t) v->num();
    return {};
}
inline std::string f64(const GgufFile & g, const std::string & k, double & out) {
    const MetaValue * v = get2(g, k);
    if (!v) return "missing " + k;
    if (!v->is_num()) return k + " is not a number";
    out = v->num();
    return {};
}
inline std::string arr(const GgufFile & g, const std::string & k, std::vector<int64_t> & out) {
    const MetaValue * v = get2(g, k);
    if (!v) return "missing " + k;
    if (v->type != MetaType::ARRAY) return k + " is not an array";
    if (v->count > v->items.size()) return k + ": longer than the reader's sample (use ds41_read_int_array)";
    out.clear();
    for (const auto & e : v->items) out.push_back((int64_t) e.num());
    return {};
}
/// "2, 8, 14" - for the lines that name what the file did not say.
inline std::string join(const std::vector<int64_t> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += std::to_string(v[i]); }
    return s;
}
/// The name llama.cpp's converter gives a tensor the engine names after model.py ("" when it has no other name).
/// The three Engram tensors are the only ones: the converter copies the checkpoint's attribute names.
inline std::string tensor_name(const std::string & n) {
    static const struct { const char * ours, * llama; } tab[] = {
        { "engram_q_norm.weight", "engram_q.weight" },
        { "engram_k_norm.weight", "engram_k.weight" },
        { "engram_kv.weight", "engram_wkv.weight" },
    };
    for (const auto & a : tab) {
        const size_t L = std::strlen(a.ours);
        if (n.size() > L && n.compare(n.size() - L, L, a.ours) == 0) return n.substr(0, n.size() - L) + a.llama;
    }
    return {};
}
/// `"key":<number>` out of the deepseek41.config JSON (flat search; the keys are unique in DeepSeek's config)
inline std::string json_num(const std::string & js, const std::string & key, double & out) {
    const std::string pat = "\"" + key + "\":";
    const size_t p = js.find(pat);
    if (p == std::string::npos) return "deepseek41.config has no " + key;
    out = std::strtod(js.c_str() + p + pat.size(), nullptr);
    return {};
}
}  // namespace detail

inline std::string ds41_read_geometry(const GgufModel & model, Ds41Geometry & G) {
    const GgufFile & g = model.meta();
    const MetaValue * arch = g.get("general.architecture");
    if (!arch || arch->s != "deepseek41") return "not a deepseek41 GGUF";
    std::string e;
    const std::string P = "deepseek41.";
    struct U { const char * k; int64_t * v; };
    for (const U & u : std::initializer_list<U>{
             { "hidden_size", &G.n_embd }, { "num_hidden_layers", &G.n_layer }, { "num_attention_heads", &G.n_head },
             { "head_dim", &G.d_head }, { "qk_rope_head_dim", &G.d_rope }, { "q_lora_rank", &G.r_q },
             { "o_lora_rank", &G.o_lora }, { "o_groups", &G.o_groups }, { "sliding_window", &G.n_swa },
             { "n_routed_experts", &G.n_expert }, { "num_experts_per_tok", &G.n_used },
             { "moe_intermediate_size", &G.n_ff }, { "n_shared_experts", &G.n_shared }, { "hc_mult", &G.hc },
             { "hc_sinkhorn_iters", &G.hc_iters }, { "index_n_heads", &G.idx_heads }, { "index_head_dim", &G.idx_dim },
             { "index_topk", &G.idx_topk }, { "max_position_embeddings", &G.ctx_len } })
        if (!(e = detail::u64(g, P + u.k, *u.v)).empty()) return e;
    if (detail::get2(g, P + "vocab_size")) {   // the engine's own file states the vocabulary; llama.cpp's does not
        if (!(e = detail::u64(g, P + "vocab_size", G.vocab)).empty()) return e;
    } else {
        const MetaValue * tk = g.get("tokenizer.ggml.tokens");
        if (!tk || tk->type != MetaType::ARRAY || tk->count < 1) return "no deepseek41.vocab_size, no usable tokenizer";
        G.vocab = (int64_t) tk->count;
        std::fprintf(stderr, "v41 geometry: no vocab_size key - using the tokenizer's %lld tokens\n", (long long) G.vocab);
    }
    struct F { const char * k; double * v; };
    for (const F & f : std::initializer_list<F>{
             { "routed_scaling_factor", &G.weights_scale }, { "hc_eps", &G.hc_eps }, { "rms_norm_eps", &G.rms_eps },
             { "rope_theta", &G.rope_base }, { "compress_rope_theta", &G.compress_rope_base },
             { "rope_scaling.factor", &G.rope_factor }, { "rope_scaling.beta_fast", &G.beta_fast },
             { "rope_scaling.beta_slow", &G.beta_slow }, { "rope_scaling.original_max_position_embeddings", &G.orig_ctx } })
        if (!(e = detail::f64(g, P + f.k, *f.v)).empty()) return e;
    if (detail::get2(g, P + "swiglu_limit")) {   // the engine's own file states one shared clamp
        if (!(e = detail::f64(g, P + "swiglu_limit", G.swiglu_limit)).empty()) return e;
    } else {   // llama.cpp writes the clamp per layer, never one shared limit
        const MetaValue * cl = g.get(P + "swiglu_clamp_exp");
        if (!cl || cl->type != MetaType::ARRAY || cl->count < 1 || cl->count > cl->items.size())
            return "no deepseek41.swiglu_limit, no readable swiglu_clamp_exp";
        for (const auto & x : cl->items) if (x.num() != cl->items[0].num()) return "swiglu_clamp_exp is not one value";
        G.swiglu_limit = cl->items[0].num();
        std::fprintf(stderr, "v41 geometry: swiglu_limit from the file's per-layer clamp: %.1f\n", G.swiglu_limit);
    }
    {
        const MetaValue * v = detail::get2(g, P + "norm_topk_prob");
        if (!v) return "missing deepseek41.norm_topk_prob";
        G.norm_topk = v->u != 0;
        const MetaValue * sf = g.get(P + "scoring_func");
        if (sf && sf->s != "sqrtsoftplus") return "scoring_func is not sqrtsoftplus";
        if (!sf) {
            // llama.cpp spells the activation as its own enum; 4 is SQRT_SOFTPLUS, the one this engine runs.
            int64_t gf = 0;
            if (!(e = detail::u64(g, P + "expert_gating_func", gf)).empty()) return e;
            if (gf != 4) return "expert_gating_func " + std::to_string(gf) + " is not sqrtsoftplus (4)";
        }
    }
    if (!(e = detail::arr(g, P + "compress_ratios", G.ratio)).empty()) return e;
    if ((int64_t) G.ratio.size() > G.n_layer) G.ratio.resize((size_t) G.n_layer);   // llama.cpp lists the MTP blocks too
    const bool have_kv = detail::arr(g, P + "kv_source_layer_ids", G.kv_sources).empty();
    const bool have_ix = detail::arr(g, P + "index_source_layer_ids", G.index_sources).empty();
    if (have_kv != have_ix) return "the file gives only one of the two source-layer lists";
    if (!have_kv) {
        // A llama.cpp-converted file writes no source-layer keys at all: a layer is a kv source when it carries
        // the compressor's kv tensor, and an index source when it carries the indexer's proj tensor.
        for (int64_t l = 0; l < G.n_layer; ++l) {
            const std::string s = "blk." + std::to_string(l) + ".";
            if (model.find(s + "attn_compressor_kv.weight")) G.kv_sources.push_back(l);
            if (model.find(s + "indexer.proj.weight")) G.index_sources.push_back(l);
        }
        std::fprintf(stderr, "v41 geometry: source layers read off the tensors - kv [%s], index [%s]\n",
                     detail::join(G.kv_sources).c_str(), detail::join(G.index_sources).c_str());
    }
    if ((int64_t) G.ratio.size() != G.n_layer) return "compress_ratios has " + std::to_string(G.ratio.size()) + " entries";
    for (int64_t r : G.ratio) if (r != 0 && r != 1 && r != 2) return "compress ratio " + std::to_string(r) + " (V4.1 has 0/1/2)";
    for (int64_t l : G.kv_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad kv source";
    for (int64_t l : G.index_sources) if (l < 0 || l >= G.n_layer || G.ratio[(size_t) l] == 0) return "bad index source";
    for (int64_t l = 0; l < G.n_layer; ++l)
        if (G.ratio[(size_t) l] && (G.kv_src(l) < 0 || G.idx_src(l) < 0 || G.ratio[(size_t) G.kv_src(l)] != G.ratio[(size_t) l]))
            return "layer " + std::to_string(l) + " has no kv/index source of its ratio";
    if (detail::get2(g, P + "candidate_source_layer_id")) {   // the engine's own file states the candidate block
        if (!(e = detail::u64(g, P + "candidate_source_layer_id", G.cand_layer)).empty()) return e;
        if (!(e = detail::u64(g, P + "candidate_topk_blocks", G.cand_blocks)).empty()) return e;
        if (!(e = detail::u64(g, P + "candidate_block_size", G.cand_bsize)).empty()) return e;
    } else {
        // The candidate block is the ratio-1 block the router prefetch fills.  V4.1 puts it on the first layer
        // whose compress ratio is 1; the count and size are the release's own (its config.json: 2048 of 8).
        G.cand_layer = -1;
        for (int64_t l = 0; l < G.n_layer; ++l) if (G.ratio[(size_t) l] == 1) { G.cand_layer = l; break; }
        if (G.cand_layer < 0) return "no candidate_source_layer_id and no ratio-1 layer to read it off";
        G.cand_blocks = 2048; G.cand_bsize = 8;
        std::fprintf(stderr, "v41 geometry: candidate block from the release's defaults - layer %lld, %lld of %lld\n",
                     (long long) G.cand_layer, (long long) G.cand_blocks, (long long) G.cand_bsize);
    }
    if (G.n_shared != 1 || G.hc < 1 || G.d_head <= G.d_rope || (G.n_head * G.d_head) % G.o_groups) return "unsupported shape";
    // Engram: arrays (full, not the reader's sample) + the shape.  v41_gguf.py puts the shape in the config JSON;
    // llama.cpp writes it as flat engram.* keys and writes no config key at all.
    double d = 0;
    if (const MetaValue * cfg = g.get(P + "config")) {
        if (!(e = detail::json_num(cfg->s, "engram_n_heads", d)).empty()) return e;
        G.engram_heads = (int64_t) d;
        if (!(e = detail::json_num(cfg->s, "engram_head_dim", d)).empty()) return e;
        G.engram_dim = (int64_t) d;
        if (!(e = detail::json_num(cfg->s, "engram_max_ngram_size", d)).empty()) return e;
        G.engram_ngram = (int64_t) d;
    } else {
        if (!(e = detail::u64(g, P + "engram.head_count", G.engram_heads)).empty()) return "no deepseek41.config and " + e;
        if (!(e = detail::u64(g, P + "engram.key_length", G.engram_dim)).empty()) return "no deepseek41.config and " + e;
        if (!(e = detail::u64(g, P + "engram.max_ngram_size", G.engram_ngram)).empty()) return "no deepseek41.config and " + e;
    }
    detail::u64(g, P + "engram.compressed_vocab_size", G.engram_cvocab);   // llama.cpp writes no such key
    if (!(e = detail::arr(g, P + "engram.layer_ids", G.engram_layers)).empty()) return e;
    detail::arr(g, P + "engram.rows", G.engram_rows);   // llama.cpp writes none: the primes tile the table exactly
    const std::string & path = g.path();
    std::vector<int64_t> tm, pr;
    if (!(e = ds41_read_int_array(path, P + "engram.token_map", tm)).empty()) return e;
    if (!(e = ds41_read_int_array(path, P + "engram.primes", pr)).empty()) return e;
    if (!(e = ds41_read_int_array(path, P + "engram.multipliers", G.engram_mult)).empty()) return e;
    if ((int64_t) tm.size() != G.vocab) return "engram.token_map size";
    G.engram_token_map.assign(tm.begin(), tm.end());
    // pad_id is already compressed in the canonical format. Maya's pad_token_id is a tokenizer id,
    // not an alias for that value: even when both happen to be equal in a real release, map it explicitly.
    if (g.get(P + "engram.pad_id")) {
        if (!(e = detail::u64(g, P + "engram.pad_id", G.engram_pad)).empty()) return e;
    } else {
        int64_t token = 0;
        if (!(e = detail::u64(g, P + "engram.pad_token_id", token)).empty()) return e;
        if (token < 0 || token >= (int64_t) tm.size()) return "engram.pad_token_id out of range";
        G.engram_pad = G.engram_token_map[(size_t) token];
    }
    G.engram_primes.assign(pr.begin(), pr.end());
    const int64_t nel = (int64_t) G.engram_layers.size();
    if ((int64_t) G.engram_primes.size() != nel * (G.engram_ngram - 1) * G.engram_heads ||
        (int64_t) G.engram_mult.size() != nel * G.engram_ngram ||
        ((int64_t) G.engram_rows.size() != nel && !G.engram_rows.empty()))
        return "engram arrays do not match its shape";
    bool derived = G.engram_rows.empty();
    for (int64_t i = 0; i < nel; ++i) {   // the bucket ranges tile the table exactly (v41_gguf_meta checks the same)
        uint64_t s = 0;
        for (int64_t j = 0; j < (G.engram_ngram - 1) * G.engram_heads; ++j)
            s += G.engram_primes[(size_t) (i * (G.engram_ngram - 1) * G.engram_heads + j)];
        if (derived) { G.engram_rows.push_back((int64_t) s); continue; }
        if ((int64_t) s != G.engram_rows[(size_t) i]) return "engram primes do not tile the table";
    }
    if (G.engram_cvocab <= 0) {   // with no compressed_vocab_size key, the token map is what defines it
        uint32_t mx = 0;
        for (uint32_t c : G.engram_token_map) if (c > mx) mx = c;
        G.engram_cvocab = (int64_t) mx + 1;
        derived = true;
    }
    if (derived)
        std::fprintf(stderr, "v41 geometry: Engram rows / compressed vocabulary derived - %s rows, %lld ids\n",
                     detail::join(G.engram_rows).c_str(), (long long) G.engram_cvocab);
    detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos);
    detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos);
    return {};
}

/// The full tensor map + per-layer expert tiers.  `layers`: the blocks the file must hold (a partial test file
/// holds a subset - pass them; the real file holds 0..n_layer-1).  Fills `G.rename` / `G.want` / `G.engram_tables`
/// with what the file actually calls and stores, which is not always what the engine asks for (see the note above).
inline std::string ds41_check_tensors(const GgufModel & m, Ds41Geometry & G, const std::vector<int64_t> & layers,
                                     bool external_experts = false) {
    std::map<std::string, std::pair<std::vector<uint64_t>, uint32_t>> want;   // name -> (shape GGUF order, type; ~0 = any)
    const uint32_t ANY = ~0u;
    auto add = [&](const std::string & n, std::vector<uint64_t> sh, uint32_t t) { want[n] = { std::move(sh), t }; };
    const uint64_t D = (uint64_t) G.n_embd, HC = (uint64_t) G.hc, MIX = (uint64_t) G.hc_mix();
    const uint64_t DH = (uint64_t) G.d_head, NH = (uint64_t) G.n_head, RQ = (uint64_t) G.r_q;
    const uint64_t OG = (uint64_t) G.o_groups, OL = (uint64_t) G.o_lora, NE = (uint64_t) G.n_expert, FF = (uint64_t) G.n_ff;
    const uint64_t IH = (uint64_t) G.idx_heads, ID = (uint64_t) G.idx_dim;
    add("token_embd.weight", { D, (uint64_t) G.vocab }, ty::F16);
    add("output_norm.weight", { D }, ty::F32);
    add("output.weight", { D, (uint64_t) G.vocab }, ty::Q8_0);
    G.experts.assign((size_t) G.n_layer, {});
    for (int64_t l : layers) {
        const std::string b = "blk." + std::to_string(l) + ".";
        for (const char * site : { "attn", "ffn" }) {
            add(b + "hc_" + site + "_fn.weight", { HC * D, MIX }, ty::F16);
            add(b + "hc_" + site + "_base.weight", { MIX }, ty::F32);
            add(b + "hc_" + site + "_scale.weight", { 3 }, ty::F32);
            add(b + site + "_norm.weight", { D }, ty::F32);
        }
        add(b + "attn_sinks.weight", { NH }, ty::F32);
        add(b + "attn_q_a.weight", { D, RQ }, ty::Q8_0);
        add(b + "attn_q_b.weight", { RQ, NH * DH }, ty::Q8_0);
        add(b + "attn_q_a_norm.weight", { RQ }, ty::F32);
        add(b + "attn_kv.weight", { D, DH }, ty::Q8_0);
        add(b + "attn_kv_a_norm.weight", { DH }, ty::F32);
        add(b + "attn_output_a.weight", { NH * DH / OG, OG * OL }, ty::Q8_0);
        add(b + "attn_output_b.weight", { OG * OL, D }, ty::Q8_0);
        if (G.is_kv_source(l)) {
            add(b + "attn_compressor_kv.weight", { D, DH }, ty::F16);
            add(b + "attn_compressor_norm.weight", { DH }, ty::F32);
            add(b + "indexer.attn_k.weight", { DH, ID }, ty::F16);
            add(b + "indexer.k_norm.weight", { ID }, ty::F32);
            if (G.ratio[(size_t) l] > 1) add(b + "attn_compressor_gate.weight", { D, DH }, ty::F16);
        }
        if (G.is_index_source(l)) {
            add(b + "indexer.attn_q_b.weight", { RQ, IH * ID }, ty::F16);
            add(b + "indexer.proj.weight", { D, IH }, ty::F16);
        }
        add(b + "ffn_gate_inp.weight", { D, NE }, ty::F32);
        add(b + "exp_probs_b.bias", { NE }, ty::F32);
        add(b + "exp_probs_b_vl.bias", { NE }, ty::F32);
        add(b + "ffn_gate_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_up_shexp.weight", { D, FF }, ty::Q8_0);
        add(b + "ffn_down_shexp.weight", { FF, D }, ty::Q8_0);
        if (G.engram_slot(l) >= 0) {
            if (m.find(b + "engram_gate.weight")) {
                add(b + "engram_gate.weight", { D, HC }, ty::F32);
            } else {
                add(b + "engram_q_norm.weight", { D, HC }, ty::F32);
                add(b + "engram_k_norm.weight", { D, HC }, ty::F32);
            }
            add(b + "engram_kv.weight", { (uint64_t) ((G.engram_ngram - 1) * G.engram_heads * G.engram_dim), (HC + 1) * D }, ty::F16);
        }
        // routed experts: uniform role tensors or per-type tensors + the I32 maps
        if (external_experts && !m.find(b + "ffn_gate_exps.weight") && !m.find(b + "ffn_up_exps.weight") &&
            !m.find(b + "ffn_down_exps.weight") && !m.find(b + "ffn_exps_gu_type") && !m.find(b + "ffn_exps_down_type"))
            continue;
        Ds41ExpertMap & X = G.experts[(size_t) l];
        X.gu_type.assign(NE, -1); X.d_type.assign(NE, -1); X.gu_slot.assign(NE, -1); X.d_slot.assign(NE, -1);
        for (int k = 0; k < 2; ++k) {   // 0 = gate+up, 1 = down
            const std::string mapn = b + (k ? "ffn_exps_down_type" : "ffn_exps_gu_type");
            std::vector<int> & tv = k ? X.d_type : X.gu_type;
            std::vector<int> & sv = k ? X.d_slot : X.gu_slot;
            const char * roles[2] = { k ? "down" : "gate", k ? nullptr : "up" };
            size_t si = 0;
            const TensorInfo * tm = m.find(mapn, &si);
            if (!tm) {   // uniform: one stacked tensor per role
                const TensorInfo * t0 = m.find(b + "ffn_" + roles[0] + "_exps.weight");
                if (!t0) return "missing " + b + "ffn_" + roles[0] + "_exps.weight (and no " + mapn + ")";
                for (uint64_t e = 0; e < NE; ++e) { tv[e] = (int) t0->type; sv[e] = (int) e; }
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string n = b + "ffn_" + r + "_exps.weight";
                    add(n, k ? std::vector<uint64_t>{ FF, D, NE } : std::vector<uint64_t>{ D, FF, NE }, t0->type);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[(int) t0->type] = n;
                }
                continue;
            }
            add(mapn, { NE }, ty::I32);
            if (tm->type != ty::I32 || tm->shape.size() != 1 || tm->shape[0] != NE) return mapn + ": bad shape/type";
            const int32_t * mv = reinterpret_cast<const int32_t *>(m.shard(si).tensor_data(*tm));
            std::map<int, int> count;
            for (uint64_t e = 0; e < NE; ++e) { tv[e] = mv[e]; sv[e] = count[mv[e]]++; }
            for (const auto & [t, n] : count) {
                if (t != (int) ty::Q2_0 && t != (int) ty::Q3_K && t != (int) ty::PTQ1_0)
                    return mapn + ": type " + std::to_string(t) + " is not an expert tier";
                std::string tn = ggml_type_name((uint32_t) t);
                for (char & c : tn) c = (char) std::tolower((unsigned char) c);
                for (const char * r : roles) {
                    if (!r) continue;
                    const std::string nm = b + "ffn_" + r + "_exps." + tn + ".weight";
                    add(nm, k ? std::vector<uint64_t>{ FF, D, (uint64_t) n } : std::vector<uint64_t>{ D, FF, (uint64_t) n }, (uint32_t) t);
                    (k ? X.down_t : (std::strcmp(r, "gate") ? X.up_t : X.gate_t))[t] = nm;
                }
            }
        }
    }
    (void) ANY;
    // Every predicted tensor present with its shape; every file tensor predicted.  A file in llama.cpp's spelling
    // calls three tensors by the checkpoint's names and stores dense matrices in BF16 where the engine computes in
    // F16: the alias is recorded so the loader can find the tensor, and converting the type is the loader's job.
    G.rename.clear(); G.want.clear(); G.engram_tables.clear();
    for (const auto & [n, st] : want) {
        size_t s = 0;
        const TensorInfo * t = m.find(n, &s);
        if (!t) {
            const std::string a = detail::tensor_name(n);
            if (a.empty() || !(t = m.find(a, &s))) return "missing tensor " + n;
            G.rename[a] = n;
        }
        if (t->shape != st.first) {
            std::string a, w;
            for (auto x : t->shape) a += std::to_string(x) + " ";
            for (auto x : st.first) w += std::to_string(x) + " ";
            return n + ": shape [" + a + "] expected [" + w + "]";
        }
        G.want[n] = st.second;   // the type the engine reads it as; the file may store another.  Whether that type
                                 // can be dequantized is the loader's knowledge, not the header's (ds41_dense.cpp
                                 // checks ggml's traits before it converts anything).
        if (!m.in_bounds(*t, s)) return n + ": out of bounds";
    }
    // The Engram tables: the engine's own file keeps them out of the GGUF (HF safetensors, --engram DIRS);
    // llama.cpp releases carry them as Q8_0 or Q6_K tensors, which ds41_engram reads straight off the shard.
    std::set<std::string> tables;
    for (size_t i = 0; i < G.engram_layers.size(); ++i) {
        const std::string n = "blk." + std::to_string(G.engram_layers[i]) + ".engram_embd.weight";
        size_t s = 0;
        const TensorInfo * t = m.find(n, &s);
        if (!t) continue;
        if ((t->type != ty::Q8_0 && t->type != ty::Q6_K) || t->shape.size() != 2 || t->shape[0] != (uint64_t) G.engram_dim ||
            t->shape[1] != (uint64_t) G.engram_rows[i] || !m.in_bounds(*t, s))
            return "Engram table " + n + " is not this model's Q8_0 or Q6_K table";
        tables.insert(n);
        G.engram_tables.push_back(n);
    }
    // Every tensor in the file is one the engine asked for: by its own name, as an Engram table, or under the
    // alias the loop above resolved (a llama.cpp file calls the three Engram tensors by the checkpoint's names).
    for (size_t s = 0; s < m.size(); ++s)
        for (const auto & t : m.shard(s).tensors())
            if (!want.count(t.name) && !tables.count(t.name) && !G.rename.count(t.name))
                return "unexpected tensor " + t.name;
    return {};
}

/// The blocks a file holds: `strata.v41.partial_layers` when the writer made a partial test file, else all.
inline std::vector<int64_t> ds41_file_layers(const GgufModel & m, const Ds41Geometry & G) {
    std::vector<int64_t> v;
    if (detail::arr(m.meta(), "strata.v41.partial_layers", v).empty()) return v;
    for (int64_t l = 0; l < G.n_layer; ++l) v.push_back(l);
    return v;
}

}  // namespace strata::v41
