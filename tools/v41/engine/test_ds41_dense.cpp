// test_ds41_dense <file.gguf> --tokens 1,2,3 --out DIR [--engram DIR1:DIR2:...] [--mode decode|prefill|both]
//                 [--threads N] [--chunk N]
// Runs the V4.1 dense half on the CPU over the tokens (decode: one token per pass; prefill: chunks of --chunk),
// with the routed experts computed by a plain reference (rows dequantized, F32 matvec) and writes per-layer block
// outputs, router ids / weights and the logits as raw files for compare_ds41.py (vs ds41_golden.py's golden.npz):
//   DIR/<mode>/x<L>.f32 [S, HC, D], pre<L>.f32 [S, HC], ids<L>.i32 [S, 6], wts<L>.f32 [S, 6], logits.f32 [S, V]
// --compat-self-test: model-free CPU regressions for mmap embedding row formats and folded Engram gates.
#include "ds41_dense.hpp"

#include "strata/kernels/cpu/ptq1_0.hpp"

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::v41;


// Implementation-private helpers; the same convention as test_ds41_route, without expanding the public API.
namespace strata::v41 {
ggml_tensor * ds41_engram_gate_weights(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
bool ds41_embedding_layout(uint32_t, int64_t, size_t &, ggml_to_float_t &);
bool ds41_gather_embedding_rows(const uint8_t *, size_t, ggml_to_float_t, int64_t, int64_t, const int *, int, float *);
}
namespace {

int compatibility_self_test() {
    int fails = 0;
    auto check = [&](const char * name, bool ok) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
        fails += !ok;
    };
    constexpr int D = 64, V = 4;
    const int ids[] = {3, 1, 0, 3};
    for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
        size_t stride = 0;
        ggml_to_float_t decode = nullptr;
        const bool layout = ds41_embedding_layout(type, D, stride, decode);
        check("supported embedding layout", layout);
        if (!layout) continue;
        const size_t expected_stride = type == GGML_TYPE_Q8_0 ? D / 32 * 34 : D * (type == GGML_TYPE_F32 ? 4 : 2);
        // Leave enough trailing bytes that the old uint16[D] stride gives wrong values, not an accidental OOB.
        std::vector<uint8_t> table(V * D * 4, 0);
        std::vector<float> expected(V * D), got(4 * D);
        for (int r = 0; r < V; ++r) {
            for (int i = 0; i < D; ++i) {
                const int q = (i * 7 + r * 11) % 61 - 30;
                const float scale = 0.5f * (1 + r + i / 32);
                expected[r * D + i] = scale * q;   // exactly representable in all three float formats
            }
            uint8_t * row = table.data() + r * expected_stride;
            if (type == GGML_TYPE_Q8_0) {
                for (int b = 0; b < D / 32; ++b) {
                    const ggml_fp16_t scale = ggml_fp32_to_fp16(0.5f * (1 + r + b));
                    std::memcpy(row + b * 34, &scale, 2);
                    for (int j = 0; j < 32; ++j) {
                        const int8_t q = (int8_t) (((b * 32 + j) * 7 + r * 11) % 61 - 30);
                        std::memcpy(row + b * 34 + 2 + j, &q, 1);
                    }
                }
            } else if (type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(expected.data() + r * D, reinterpret_cast<ggml_fp16_t *>(row), D);
            } else if (type == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row_ref(expected.data() + r * D, reinterpret_cast<ggml_bf16_t *>(row), D);
            } else {
                std::memcpy(row, expected.data() + r * D, D * sizeof(float));
            }
        }
        bool ok = ds41_gather_embedding_rows(table.data(), stride, decode, D, V, ids, 4, got.data());
        for (int t = 0; t < 4; ++t)
            for (int i = 0; i < D; ++i) ok &= got[t * D + i] == expected[ids[t] * D + i];
        check(type == GGML_TYPE_Q8_0 ? "Q8_0 nonzero/duplicate row values" :
              type == GGML_TYPE_F16 ? "F16 row values" :
              type == GGML_TYPE_BF16 ? "BF16 row values" : "F32 row values", ok);
        for (int invalid : {-1, V})
            check("embedding id range rejected", !ds41_gather_embedding_rows(table.data(), stride, decode, D, V,
                                                                           &invalid, 1, got.data()));
    }
    size_t stride = 0;
    ggml_to_float_t decode = nullptr;
    check("partial quant block rejected", !ds41_embedding_layout(GGML_TYPE_Q8_0, D - 1, stride, decode));
    check("unknown embedding type rejected", !ds41_embedding_layout(GGML_TYPE_COUNT, D, stride, decode));
    check("non-row-decodable embedding type rejected", !ds41_embedding_layout(GGML_TYPE_Q8_1, D, stride, decode));

    ggml_init_params ip = {2 * 1024 * 1024, nullptr, false};
    ggml_context * gc = ggml_init(ip);
    check("Engram graph context", gc != nullptr);
    if (!gc) return 1;
    constexpr int GD = 8, HC = 2;
    ggml_tensor * q = ggml_new_tensor_2d(gc, GGML_TYPE_F32, GD, HC);
    ggml_tensor * k = ggml_new_tensor_2d(gc, GGML_TYPE_F32, GD, HC);
    ggml_tensor * folded = ggml_new_tensor_2d(gc, GGML_TYPE_F32, GD, HC);
    ggml_tensor * h = ggml_new_tensor_2d(gc, GGML_TYPE_F32, GD, HC);
    ggml_tensor * key = ggml_new_tensor_2d(gc, GGML_TYPE_F32, GD, HC);
    float oracle[HC] = {};
    for (int i = 0; i < GD * HC; ++i) {
        const float qv = 0.25f * (i % 5 - 2), kv = 0.5f * (i % 3 + 1);
        const float hv = 0.125f * (i - 5), keyv = 0.25f * (3 - i % 7);
        static_cast<float *>(q->data)[i] = qv;
        static_cast<float *>(k->data)[i] = kv;
        static_cast<float *>(folded->data)[i] = qv * kv;
        static_cast<float *>(h->data)[i] = hv;
        static_cast<float *>(key->data)[i] = keyv;
        oracle[i / GD] += hv * qv * kv * keyv;
    }
    auto gate = [&](ggml_tensor * weights) {
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, h, weights), key)),
                                      1.0f / std::sqrt((float) GD));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        return ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
    };
    ggml_tensor * separate = gate(ds41_engram_gate_weights(gc, nullptr, q, k));
    ggml_tensor * product = gate(ds41_engram_gate_weights(gc, folded, nullptr, nullptr));
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 128, false);
    ggml_build_forward_expand(gf, separate);
    ggml_build_forward_expand(gf, product);
    const bool computed = ggml_graph_compute_with_ctx(gc, gf, 1) == GGML_STATUS_SUCCESS;
    check("Engram gate graph compute", computed);
    if (computed) {
        bool equivalent = true, correct = true;
        for (int c = 0; c < HC; ++c) {
            const float dot = oracle[c] / std::sqrt((float) GD);
            const float signed_root = dot == 0 ? 0 : std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6f)), dot);
            const float expected = 1.0f / (1.0f + std::exp(-signed_root));
            const float a = static_cast<float *>(separate->data)[c], b = static_cast<float *>(product->data)[c];
            equivalent &= std::fabs(a - b) <= 1e-6f;
            correct &= std::fabs(b - expected) <= 1e-6f;
        }
        check("folded and separate Engram gates equivalent", equivalent);
        check("folded Engram gate uses exact stored product", correct);
    }
    ggml_free(gc);
    return fails ? 1 : 0;
}

void save(const std::string & p, const void * d, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char *) d, (std::streamsize) bytes);
}

// the routed experts, by the book: rows dequantized to F32, clamped SwiGLU, weighted sum
struct RefExperts {
    const GgufModel * m = nullptr;
    const Ds41Geometry * g = nullptr;
    int threads = 8;
    std::vector<float> row(int type, const uint8_t * src, int64_t n) const {
        std::vector<float> y((size_t) n);
        if (type == 143) kernels::cpu::ptq1_0_dequant(src, y.data(), n);
        else ggml_get_type_traits((ggml_type) type)->to_float(src, y.data(), n);
        return y;
    }
    size_t row_bytes(int type, int64_t n) const {
        int be = 0, bb = 0;
        block_geometry((uint32_t) type, be, bb);
        return (size_t) (n / be) * (size_t) bb;
    }
    // y[D] += w * expert_e(x)
    void expert(int64_t L, int e, float w, const float * x, float * y) const {
        const Ds41ExpertMap & X = g->experts[(size_t) L];
        const int gt = X.gu_type[(size_t) e], dt = X.d_type[(size_t) e];
        const int64_t D = g->n_embd, FF = g->n_ff;
        auto base = [&](const std::string & name, int slot, int64_t rows, int64_t cols, int type) {
            size_t s = 0;
            const TensorInfo * t = m->find(name, &s);
            return m->shard(s).tensor_data(*t) + (size_t) slot * (size_t) rows * row_bytes(type, cols);
        };
        const uint8_t * Wg = base(X.gate_t.at(gt), X.gu_slot[(size_t) e], FF, D, gt);
        const uint8_t * Wu = base(X.up_t.at(gt), X.gu_slot[(size_t) e], FF, D, gt);
        const uint8_t * Wd = base(X.down_t.at(dt), X.d_slot[(size_t) e], D, FF, dt);
        std::vector<float> a((size_t) FF);
        const size_t rg = row_bytes(gt, D), rd = row_bytes(dt, FF);
        const float lim = (float) g->swiglu_limit;
        auto par = [&](int64_t n, auto fn) {
            std::vector<std::thread> th;
            for (int k = 0; k < threads; ++k) th.emplace_back([&, k] { for (int64_t i = k; i < n; i += threads) fn(i); });
            for (auto & t : th) t.join();
        };
        par(FF, [&](int64_t i) {
            const std::vector<float> rg_ = row(gt, Wg + (size_t) i * rg, D), ru = row(gt, Wu + (size_t) i * rg, D);
            double sg = 0, su = 0;
            for (int64_t k = 0; k < D; ++k) { sg += (double) rg_[(size_t) k] * x[k]; su += (double) ru[(size_t) k] * x[k]; }
            float gg = (float) sg, uu = (float) su;
            if (lim > 0) { uu = std::min(std::max(uu, -lim), lim); gg = std::min(gg, lim); }
            a[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu * w;
        });
        std::vector<float> out((size_t) D);
        par(D, [&](int64_t i) {
            const std::vector<float> r = row(dt, Wd + (size_t) i * rd, FF);
            double s = 0;
            for (int64_t k = 0; k < FF; ++k) s += (double) r[(size_t) k] * a[(size_t) k];
            out[(size_t) i] = (float) s;
        });
        for (int64_t i = 0; i < D; ++i) y[i] += out[(size_t) i];
    }
};

std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::stoi(x));
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--compat-self-test") return compatibility_self_test();
    if (argc < 2) { std::fprintf(stderr, "usage: see the file header\n"); return 2; }
    std::string path = argv[1], out = "/tmp/ds41", mode = "both";
    std::vector<int> toks;
    std::vector<std::string> edirs;
    int threads = 8, chunk = 0;
    bool zero_experts = false, act_quant = true, kv_f32 = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--tokens") toks = parse_ints(next());
        else if (a == "--out") out = next();
        else if (a == "--mode") mode = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--zero-experts") zero_experts = true;
        else if (a == "--no-actquant") act_quant = false;
        else if (a == "--kv-f32") kv_f32 = true;
        else if (a == "--engram") {
            std::stringstream ss(next());
            std::string d;
            while (std::getline(ss, d, ':')) edirs.push_back(d);
        }
    }
    const int S = (int) toks.size();
    if (!S) { std::fprintf(stderr, "no tokens\n"); return 2; }
    if (chunk <= 0) chunk = S;
    for (const std::string m : { std::string("decode"), std::string("prefill") }) {
        if (mode != "both" && mode != m) continue;
        Ds41DenseConfig cfg;
        cfg.n_threads = threads;
        cfg.ctx_max = 4096;
        cfg.max_pass = m == "decode" ? 1 : chunk;
        cfg.engram_dirs = edirs;
        cfg.act_quant = act_quant;
        if (kv_f32) cfg.kv_type = GGML_TYPE_F32;
        Ds41Dense dn;
        std::string err;
        if (!dn.init(path, cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        const Ds41Geometry & G = dn.geom();
        GgufModel model = GgufModel::open(path);
        RefExperts rx { &model, &G, threads };
        const int64_t nl = dn.n_layer(), D = G.n_embd, HC = G.hc, NU = G.n_used, V = G.vocab;
        std::vector<std::vector<float>> X((size_t) nl), P((size_t) nl), WT((size_t) nl);
        std::vector<std::vector<int>> ID((size_t) nl);
        std::vector<float> logits((size_t) S * (size_t) V);
        ::mkdir(out.c_str(), 0755);
        const auto t0 = std::chrono::steady_clock::now();
        const int step = m == "decode" ? 1 : chunk;
        for (int p0 = 0; p0 < S; p0 += step) {
            const int n = std::min(step, S - p0);
            if (!dn.begin(&toks[(size_t) p0], n, p0)) { std::fprintf(stderr, "begin: %s\n", dn.last_error().c_str()); return 1; }
            std::vector<int> ids((size_t) (n * NU));
            std::vector<float> wts((size_t) (n * NU)), routed((size_t) (n * D));
            for (int64_t l = 0; l < nl; ++l) {
                const float * fn = nullptr;
                if (!dn.attn_router((int) l, ids.data(), wts.data(), &fn)) { std::fprintf(stderr, "attn L%lld: %s\n", (long long) l, dn.last_error().c_str()); return 1; }
                std::fill(routed.begin(), routed.end(), 0.0f);
                if (!zero_experts)
                    for (int t = 0; t < n; ++t)
                        for (int k = 0; k < NU; ++k)
                            rx.expert(l, ids[(size_t) (t * NU + k)], wts[(size_t) (t * NU + k)], fn + (size_t) t * D, &routed[(size_t) t * D]);
                if (!dn.finish((int) l, routed.data())) { std::fprintf(stderr, "finish: %s\n", dn.last_error().c_str()); return 1; }
                {   // index source layers: the visible top-k rows per query (gate of the indexer / candidates)
                    std::vector<int> sl;
                    std::vector<float> sm;
                    int K = 0;
                    dn.read_sel((int) l, sl, sm, &K);
                    if (K > 0) {
                        std::ofstream f(out + "/" + m + "_sel" + std::to_string(l) + ".txt", std::ios::app);
                        for (int t = 0; t < n; ++t) {
                            f << (p0 + t);
                            for (int j = 0; j < K; ++j)
                                if (sm[(size_t) (t * K + j)] == 0.0f) f << ' ' << sl[(size_t) (t * K + j)];
                            f << '\n';
                        }
                    }
                }
                std::vector<float> xs, pr;
                dn.read_state(xs, pr);
                X[(size_t) l].insert(X[(size_t) l].end(), xs.begin(), xs.end());
                P[(size_t) l].insert(P[(size_t) l].end(), pr.begin(), pr.end());
                ID[(size_t) l].insert(ID[(size_t) l].end(), ids.begin(), ids.end());
                WT[(size_t) l].insert(WT[(size_t) l].end(), wts.begin(), wts.end());
            }
            if (!dn.logits(0, n, &logits[(size_t) p0 * (size_t) V])) { std::fprintf(stderr, "logits: %s\n", dn.last_error().c_str()); return 1; }
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::string dir = out + "/" + m;
        ::mkdir(out.c_str(), 0755);
        ::mkdir(dir.c_str(), 0755);
        for (int64_t l = 0; l < nl; ++l) {
            save(dir + "/x" + std::to_string(l) + ".f32", X[(size_t) l].data(), X[(size_t) l].size() * 4);
            save(dir + "/pre" + std::to_string(l) + ".f32", P[(size_t) l].data(), P[(size_t) l].size() * 4);
            save(dir + "/ids" + std::to_string(l) + ".i32", ID[(size_t) l].data(), ID[(size_t) l].size() * 4);
            save(dir + "/wts" + std::to_string(l) + ".f32", WT[(size_t) l].data(), WT[(size_t) l].size() * 4);
        }
        save(dir + "/logits.f32", logits.data(), logits.size() * 4);
        std::printf("%s: %d tokens, %lld layers in %.1f s -> %s (D %lld HC %lld)\n", m.c_str(), S, (long long) nl, sec, dir.c_str(),
                    (long long) D, (long long) HC);
    }
    return 0;
}
