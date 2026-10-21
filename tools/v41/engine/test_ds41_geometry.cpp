// test_ds41_geometry <file.gguf> - P1 gate: geometry + full tensor map + expert tiers and missing-layer mutations.
// --compat-self-test: synthetic GGUF metadata aliases/pad conversion and folded/original Engram tensor maps.
#include "ds41_geometry.hpp"
#include "../../../tests/core/gguf_fixture.hpp"

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <filesystem>

using namespace strata;
using namespace strata::v41;

namespace {

struct TempDir {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ds41-geometry-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { std::filesystem::create_directories(path); }
    ~TempDir() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

// Small but internally consistent geometry. In particular tokenizer pad 2 maps to compressed id 1,
// unlike Maya's real pad where the two happen to match; a simple rename must not pass this regression.
std::vector<fixture::Kv> compatibility_metadata(bool maya) {
    using namespace fixture;
    const std::string p = "deepseek41.";
    return {
        str("general.architecture", "deepseek41"),
        u32(p + "embedding_length", 32), u32(p + "block_count", 1),
        u32(p + "attention.head_count", 2), u32(p + "attention.key_length", 32),
        u32(p + "rope.dimension_count", 8), u32(p + "attention.q_lora_rank", 32),
        u32(p + "attention.output_lora_rank", 32), u32(p + "attention.output_group_count", 1),
        u32(p + "attention.sliding_window", 4), u32(p + "expert_count", 2), u32(p + "expert_used_count", 1),
        u32(p + "expert_feed_forward_length", 32), u32(p + "expert_shared_count", 1),
        u32(p + "hyper_connection.count", 2), u32(p + "hyper_connection.sinkhorn_iterations", 3),
        u32(p + "attention.indexer.head_count", 1), u32(p + "attention.indexer.key_length", 32),
        u32(p + "attention.indexer.top_k", 2), u32(p + "context_length", 32), u32(p + "vocab_size", 4),
        f32(p + "expert_weights_scale", 1), f32(p + "hyper_connection.epsilon", 1e-6f),
        f32(p + "attention.layer_norm_rms_epsilon", 1e-6f), f32(p + "rope.freq_base", 10000),
        f32(p + "attention.compress_rope_freq_base", 10000), f32(p + "rope.scaling.factor", 1),
        f32(p + "rope.scaling.yarn_beta_fast", 32), f32(p + "rope.scaling.yarn_beta_slow", 1),
        u32(p + "rope.scaling.original_context_length", 32), f32(p + "swiglu_limit", 10),
        u32(p + "expert_weights_norm", 1), u32(p + "expert_gating_func", 4),
        u32s(p + "attention.compress_ratios", {0}), u32s(p + "kv_source_layer_ids", {}),
        u32s(p + "index_source_layer_ids", {}), u32(p + "candidate_source_layer_id", 0),
        u32(p + "candidate_topk_blocks", 2), u32(p + "candidate_block_size", 2),
        u32(p + "engram.head_count", 1), u32(p + "engram.max_ngram_size", 2),
        u32(p + (maya ? "engram.head_dim" : "engram.key_length"), 256),
        u32(p + (maya ? "engram.pad_token_id" : "engram.pad_id"), 2),
        u32s(p + (maya ? "engram.layers" : "engram.layer_ids"), {0}),
        u32s(p + (maya ? "engram.num_embeddings" : "engram.rows"), {3}),
        u32s(p + "engram.token_map", {3, 0, 1, 2}), u32s(p + "engram.primes", {3}),
        u32s(p + "engram.multipliers", {1, 3})
    };
}

std::vector<fixture::Tensor> compatibility_tensors(bool folded, uint32_t table_type) {
    using fixture::Tensor;
    std::vector<Tensor> ts = {
        {"token_embd.weight", {32, 4}, folded ? ty::Q8_0 : 30u},
        {"output_norm.weight", {32}, ty::F32}, {"output.weight", {32, 4}, ty::Q8_0}
    };
    const std::string b = "blk.0.";
    auto add = [&](const std::string & n, std::vector<uint64_t> shape, uint32_t type) {
        ts.push_back({b + n, std::move(shape), type});
    };
    for (const char * site : {"attn", "ffn"}) {
        add(std::string("hc_") + site + "_fn.weight", {64, 8}, ty::F16);
        add(std::string("hc_") + site + "_base.weight", {8}, ty::F32);
        add(std::string("hc_") + site + "_scale.weight", {3}, ty::F32);
        add(std::string(site) + "_norm.weight", {32}, ty::F32);
    }
    add("attn_sinks.weight", {2}, ty::F32);
    add("attn_q_a.weight", {32, 32}, ty::Q8_0); add("attn_q_b.weight", {32, 64}, ty::Q8_0);
    add("attn_q_a_norm.weight", {32}, ty::F32); add("attn_kv.weight", {32, 32}, ty::Q8_0);
    add("attn_kv_a_norm.weight", {32}, ty::F32);
    add("attn_output_a.weight", {64, 32}, ty::Q8_0); add("attn_output_b.weight", {32, 32}, ty::Q8_0);
    add("ffn_gate_inp.weight", {32, 2}, ty::F32);
    add("exp_probs_b.bias", {2}, ty::F32); add("exp_probs_b_vl.bias", {2}, ty::F32);
    for (const char * role : {"gate", "up", "down"}) {
        add(std::string("ffn_") + role + "_shexp.weight", {32, 32}, ty::Q8_0);
        add(std::string("ffn_") + role + "_exps.weight", {32, 32, 2}, ty::Q8_0);
    }
    if (folded) {
        add("engram_gate.weight", {32, 2}, ty::F32);
        add("engram_kv.weight", {256, 96}, ty::Q8_0);
    } else {
        add("engram_q.weight", {32, 2}, 30); add("engram_k.weight", {32, 2}, 30);
        add("engram_wkv.weight", {256, 96}, ty::F16);
    }
    add("engram_embd.weight", {256, 3}, table_type);
    return ts;
}

int compatibility_self_test() {
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & e = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, e.empty() ? "" : "  ", e.c_str());
        fails += !ok;
    };
    TempDir dir;
    const auto path = dir.path / "compat.gguf";
    auto read = [&](const std::vector<fixture::Kv> & kv, Ds41Geometry & g) {
        fixture::write(path, kv, {});
        const GgufModel m = GgufModel::open(path.string());
        return ds41_read_geometry(m, g);
    };
    {
        Ds41Geometry g;
        const std::string e = read(compatibility_metadata(true), g);
        check("Maya aliases and nonidentity pad conversion", e.empty() && g.engram_pad == 1 &&
              g.engram_dim == 256 && g.engram_layers == std::vector<int64_t>{0} &&
              g.engram_rows == std::vector<int64_t>{3}, e);
    }
    {
        Ds41Geometry g;
        auto kv = compatibility_metadata(false);
        kv.push_back(fixture::u32("deepseek41.engram.pad_token_id", 1));
        kv.push_back(fixture::u32("deepseek41.engram.head_dim", 128));
        kv.push_back(fixture::u32s("deepseek41.engram.layers", {9}));
        kv.push_back(fixture::u32s("deepseek41.engram.num_embeddings", {99}));
        const std::string e = read(kv, g);
        check("canonical keys win and canonical pad remains compressed", e.empty() && g.engram_pad == 2 &&
              g.engram_dim == 256 && g.engram_layers == std::vector<int64_t>{0} &&
              g.engram_rows == std::vector<int64_t>{3}, e);
    }
    {
        Ds41Geometry g;
        auto kv = compatibility_metadata(true);
        for (auto & k : kv) if (k.key == "deepseek41.engram.pad_token_id") k.u = 4;
        const std::string e = read(kv, g);
        check("out-of-range tokenizer pad rejected", !e.empty(), e);
    }
    for (bool folded : {false, true}) {
        for (uint32_t table_type : {ty::Q8_0, ty::Q6_K}) {
            fixture::write(path, compatibility_metadata(folded), compatibility_tensors(folded, table_type));
            const GgufModel m = GgufModel::open(path.string());
            Ds41Geometry g;
            std::string e = ds41_read_geometry(m, g);
            if (e.empty()) e = ds41_check_tensors(m, g, {0});
            check(folded ? "folded gate tensor map" : "original Q/K tensor aliases", e.empty(), e);
        }
    }
    {
        auto tensors = compatibility_tensors(true, ty::Q8_0);
        tensors.erase(std::remove_if(tensors.begin(), tensors.end(), [](const auto & t) {
            return t.name.find("_exps.weight") != std::string::npos;
        }), tensors.end());
        fixture::write(path, compatibility_metadata(true), tensors);
        const GgufModel m = GgufModel::open(path.string());
        Ds41Geometry g;
        std::string e = ds41_read_geometry(m, g);
        check("dense-only geometry", e.empty(), e);
        e = ds41_check_tensors(m, g, {0});
        check("dense-only file requires external experts opt-in", !e.empty(), e);
        e = ds41_check_tensors(m, g, {0}, true);
        check("dense-only map with external experts", e.empty() && g.engram_tables.size() == 1, e);
        tensors.erase(std::remove_if(tensors.begin(), tensors.end(), [](const auto & t) {
            return t.name == "blk.0.ffn_gate_inp.weight";
        }), tensors.end());
        const auto broken = dir.path / "missing-router.gguf";
        fixture::write(broken, compatibility_metadata(true), tensors);
        const GgufModel bm = GgufModel::open(broken.string());
        e = ds41_check_tensors(bm, g, {0}, true);
        check("external experts still require the dense router", !e.empty(), e);
    }
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--compat-self-test") return compatibility_self_test();
    if (argc < 2) { std::fprintf(stderr, "usage: test_ds41_geometry <file.gguf>\n"); return 2; }
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & detail = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  ", detail.c_str());
        fails += !ok;
    };
    GgufModel m = GgufModel::open(argv[1]);
    Ds41Geometry G;
    std::string e = ds41_read_geometry(m, G);
    check("geometry", e.empty(), e);
    if (!e.empty()) return 1;
    std::printf("D %lld layers %lld heads %lld d_head %lld rope %lld q_lora %lld o %lldx%lld swa %lld experts %lld/%lld ff %lld "
                "hc %lld idx %lldx%lld top %lld cand L%lld %lldx%lld eps %g vocab %lld ctx %lld clamp %g\n",
                (long long) G.n_embd, (long long) G.n_layer, (long long) G.n_head, (long long) G.d_head, (long long) G.d_rope,
                (long long) G.r_q, (long long) G.o_groups, (long long) G.o_lora, (long long) G.n_swa, (long long) G.n_expert,
                (long long) G.n_used, (long long) G.n_ff, (long long) G.hc, (long long) G.idx_heads, (long long) G.idx_dim,
                (long long) G.idx_topk, (long long) G.cand_layer, (long long) G.cand_blocks, (long long) G.cand_bsize,
                G.rms_eps, (long long) G.vocab, (long long) G.ctx_len, G.swiglu_limit);
    std::printf("engram layers %zu heads %lld dim %lld ngram %lld pad %lld cvocab %lld token_map %zu primes %zu mult %zu\n",
                G.engram_layers.size(), (long long) G.engram_heads, (long long) G.engram_dim, (long long) G.engram_ngram,
                (long long) G.engram_pad, (long long) G.engram_cvocab, G.engram_token_map.size(), G.engram_primes.size(),
                G.engram_mult.size());
    check("V4.1 shape", G.n_embd == 5120 && G.n_layer == 40 && G.n_expert == 384 && G.n_used == 6 && G.n_ff == 2304 &&
                        G.r_q == 1280 && G.d_head == 512 && G.idx_heads == 32 && G.idx_dim == 128 && G.hc == 4);
    uint32_t mx = 0;
    for (uint32_t v : G.engram_token_map) mx = std::max(mx, v);
    check("engram token map within compressed vocab", (int64_t) mx < G.engram_cvocab && mx + 1 == (uint32_t) G.engram_cvocab);
    std::printf("sources: kv_src/idx_src per layer:");
    for (int64_t l = 0; l < G.n_layer; ++l) std::printf(" %lld:%lld/%lld", (long long) l, (long long) G.kv_src(l), (long long) G.idx_src(l));
    std::printf("\n");
    const std::vector<int64_t> layers = ds41_file_layers(m, G);
    e = ds41_check_tensors(m, G, layers);
    check("full tensor map", e.empty(), e);
    for (int64_t l : layers) {
        std::map<std::pair<int, int>, int> pairs;
        const auto & X = G.experts[(size_t) l];
        for (int64_t x = 0; x < G.n_expert; ++x) pairs[{ X.gu_type[(size_t) x], X.d_type[(size_t) x] }]++;
        std::printf("L%lld tiers:", (long long) l);
        for (const auto & [p, n] : pairs) std::printf(" %s/%s x%d", ggml_type_name((uint32_t) p.first), ggml_type_name((uint32_t) p.second), n);
        std::printf("\n");
    }
    // mutations: a layer missing from the expectation -> its tensors are unexpected; one the file lacks -> missing
    if (layers.size() > 1) {
        std::vector<int64_t> fewer(layers.begin() + 1, layers.end());
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, fewer);
        check("mutation: unpredicted layer rejected", e.rfind("unexpected tensor", 0) == 0, e);
    }
    if ((int64_t) layers.size() < G.n_layer) {
        std::vector<int64_t> more = layers;
        for (int64_t l = 0; l < G.n_layer; ++l) if (std::find(layers.begin(), layers.end(), l) == layers.end()) { more.push_back(l); break; }
        Ds41Geometry G2 = G;
        e = ds41_check_tensors(m, G2, more);
        check("mutation: absent layer rejected", e.find("missing") != std::string::npos, e);
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
