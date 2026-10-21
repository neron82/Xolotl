// tools/v41/engine/ds41_dense.cpp - see ds41_dense.hpp.  Line references are to DeepSeek's inference/model.py.
#include "ds41_dense.hpp"

#include "ds41_engram.hpp"
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef DS41_HAVE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>   // cudaHostAlloc / cudaHostGetDevicePointer: the split hand-off buffer
#endif   // ggml_cgraph::uid (graph reuse)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <thread>

#include <strings.h>

using namespace strata;
using namespace strata::v41;

namespace strata::v41 {

// Implementation-private graph helpers, also exercised by test_ds41_route.  The bounded policy ranks only the
// model's corrected top-R, and reserves the first P by ID rather than adding a large score.  Consequently zero
// runtime bias retains the exact order (including the protected prefix), and no selected expert has native rank > R.
// n_out may exceed n_used for prefetch: its first n_used IDs match routing; extra IDs are hints, never route slots.
ggml_tensor * ds41_bounded_route_ids(ggml_context * gc, ggml_tensor * s0, ggml_tensor * rb,
                                    int n_used, int max_rank, int protect, float margin, int n_out) {
    const int ne = (int) s0->ne[0];
    const int64_t n = s0->ne[1];
    GGML_ASSERT(n_used > 0 && n_used <= ne && n_out >= n_used && n_out <= ne);
    const int rank = std::clamp(max_rank, n_used, ne);
    const int p = std::clamp(protect, 0, n_used);
    // -1e30 is the caller's hard-pruning mask, not a cache preference: exclude it before forming the candidate set.
    ggml_tensor * hard = ggml_mul(gc, rb, ggml_step(gc, ggml_scale_bias(gc, rb, -1.0f, -1e20f)));
    ggml_tensor * native = ggml_add(gc, s0, hard);
    const int ntop = std::min(ne, std::max({rank, n_out, margin >= 0.0f ? n_used + 1 : n_used}));
    ggml_tensor * ti = ggml_argsort_top_k(gc, native, ntop);
    auto id_span = [&](int first, int count) {
        ggml_tensor * span = ggml_view_2d(gc, ti, count, n, ti->nb[1], (size_t) first * ti->nb[0]);
        return ggml_is_contiguous(span) ? span : ggml_cont(gc, span);
    };
    ggml_tensor * ids = p ? id_span(0, p) : nullptr;
    const int take = std::min(rank, n_out) - p;
    if (take > 0) {
        ggml_tensor * sel = nullptr;
        if (margin >= 0.0f && n_used < ne) {
            ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, native, 1, ne, n), ti), ntop, n);
            auto row = [&](int r) {
                return ggml_cont(gc, ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]));
            };
            ggml_tensor * cut = ggml_scale(gc, ggml_add(gc, row(n_used - 1), row(n_used)), 0.5f);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);
            ggml_tensor * outside = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, dv), 1.0f, -margin));
            ggml_tensor * gate = ggml_scale_bias(gc, outside, -1.0f, 1.0f);
            // Unlike the legacy margin gate, a penalty cannot push an in-band expert below an unchanged out-of-band
            // candidate.  Clip the adjusted score to the band, and add only the delta: rb=0 adds exactly zero.
            ggml_tensor * delta = ggml_sub(gc, ggml_clamp(gc, ggml_add(gc, dv, rb), -margin, margin), dv);
            sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, gate, delta)), hard);
        } else {
            sel = ggml_add(gc, s0, rb);
        }
        ggml_tensor * candidates = id_span(p, rank - p);
        ggml_tensor * scores = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, ne, n), candidates), rank - p, n);
        ggml_tensor * local = ggml_argsort_top_k(gc, scores, take);
        ggml_tensor * chosen = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, candidates, 1, rank - p, n), local), take, n);
        ids = ids ? ggml_concat(gc, ids, chosen, 0) : chosen;
    }
    // A prefetch request wider than R gets the remaining native-ranked candidates, without enlarging routing's pool.
    if (n_out > rank) ids = ggml_concat(gc, ids, id_span(rank, n_out - rank), 0);
    return ids;
}

ggml_tensor * ds41_route_weights(ggml_context * gc, ggml_tensor * sc, ggml_tensor * ids, bool norm, float scale) {
    ggml_tensor * wts = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sc, 1, sc->ne[0], sc->ne[1]), ids),
                                        ids->ne[0], sc->ne[1]);
    if (norm) wts = ggml_div(gc, wts, ggml_scale_bias(gc, ggml_sum_rows(gc, wts), 1.0f, 1e-20f));
    return ggml_scale(gc, wts, scale);
}

// Implementation-private format helpers, exercised by test_ds41_dense --compat-self-test.
// The folded tensor is the converter's exact Q*K product, not a replacement norm or an approximation.
ggml_tensor * ds41_engram_gate_weights(ggml_context * gc, ggml_tensor * folded, ggml_tensor * q, ggml_tensor * k) {
    return folded ? folded : ggml_mul(gc, q, k);
}

bool ds41_embedding_layout(uint32_t type, int64_t width, size_t & row_bytes, ggml_to_float_t & to_float) {
    if (type >= GGML_TYPE_COUNT || width <= 0) return false;
    const ggml_type t = (ggml_type) type;
    const ggml_type_traits * traits = ggml_get_type_traits(t);
    if (traits->blck_size <= 0 || traits->type_size == 0 || width % traits->blck_size ||
        (t != GGML_TYPE_F32 && !traits->to_float)) return false;
    row_bytes = ggml_row_size(t, width);
    to_float = traits->to_float;   // F16 / BF16 retain ggml's efficient row conversion.
    return true;
}

bool ds41_gather_embedding_rows(const uint8_t * table, size_t row_bytes, ggml_to_float_t to_float,
                                int64_t width, int64_t vocab, const int * ids, int n, float * out) {
    for (int t = 0; t < n; ++t) {
        if (ids[t] < 0 || ids[t] >= vocab) return false;
        const uint8_t * row = table + (size_t) ids[t] * row_bytes;
        float * dst = out + (size_t) t * width;
        if (to_float) to_float(row, dst, width);
        else std::memcpy(dst, row, (size_t) width * sizeof(float));   // F32 identity
    }
    return true;
}

}  // namespace strata::v41

namespace {

constexpr float NEG_INF = -INFINITY;

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

bool env_on(const char * k, bool dflt) {
    const char * e = std::getenv(k);
    return e ? std::atoi(e) != 0 : dflt;
}

int rb_max_rank() {
    static const int value = std::getenv("STRATA_RB_MAX_RANK") ? std::atoi(std::getenv("STRATA_RB_MAX_RANK")) : 0;
    return value;
}

int rb_protected() {
    static const int value = std::getenv("STRATA_RB_PROTECT") ? std::atoi(std::getenv("STRATA_RB_PROTECT")) : 0;
    return value;
}

float rb_score_margin() {
    static const float value = std::getenv("STRATA_RB_MARGIN") ? (float) std::atof(std::getenv("STRATA_RB_MARGIN")) : -1.0f;
    return value;
}

// The decode graphs are built once and reused; stamping a uid lets ggml-cuda reuse its captured CUDA graph.  One
// allocator serves every cached graph (they run one at a time); when it has to grow, every graph allocated before
// is re-allocated on its next use (the epoch), as tools/ds4/ds4_dense.cpp alloc_graph does.
struct Allo {
    ggml_gallocr_t a = nullptr;
    uint64_t epoch = 0;
    std::map<const ggml_cgraph *, uint64_t> graph_epoch;
    std::set<const void *> compute_bufs;
    ~Allo() { if (a) ggml_gallocr_free(a); }
    bool alloc(ggml_cgraph * g, bool reuse) {
        auto it = graph_epoch.find(g);
        if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == epoch) return true;
        if (it != graph_epoch.end() && it->second != epoch) {
            auto computed = [&](const ggml_tensor * t) { return t && t->buffer && compute_bufs.count(t->buffer); };
            auto unbind = [&](ggml_tensor * t) {
                if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                ggml_tensor * t = ggml_graph_node(g, i);
                for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
                unbind(t);
            }
            for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
        }
        const size_t before = ggml_gallocr_get_buffer_size(a, 0);
        if (!ggml_gallocr_alloc_graph(a, g)) return false;
        if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++epoch;
        auto note = [&](const ggml_tensor * t) {
            if (!t || !t->buffer) return;
            if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) compute_bufs.insert(t->buffer);
            else compute_bufs.erase(t->buffer);
        };
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            const ggml_tensor * t = ggml_graph_node(g, i);
            note(t);
            for (int j = 0; j < GGML_MAX_SRC; ++j) note(t->src[j]);
        }
        for (int i = 0; i < g->n_leafs; ++i) note(g->leafs[i]);
        graph_epoch[g] = epoch;
        if (reuse) g->uid = ggml_graph_next_uid();
        return true;
    }
};

struct WStore {
    ggml_context * ctx = nullptr, * ctx2 = nullptr;   // ctx2: host buffers for tensors the file stores in another type
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
        if (ctx2) ggml_free(ctx2);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "[ds41_dense] missing weight %s\n", n.c_str()); std::abort(); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

bool is_routed(const std::string & n) {
    return n.find("_exps.") != std::string::npos || n.find("ffn_exps_") != std::string::npos;
}

}  // namespace

// ================================================================== Impl

struct Ds41Dense::Impl {
    std::function<bool()> oom_hook;       // set_oom_hook: frees device memory when a graph buffer does not fit
    Ds41DenseConfig cfg;
    Ds41Geometry g;
    std::unique_ptr<GgufModel> model;
    std::vector<int64_t> layers;          // blocks in the file (0..nl-1)
    int64_t nl = 0;
    ggml_backend_t backend = nullptr;
    int64_t lb = 0, le = 0;           // effective range: 0 / nl when the cfg range is unset
    bool range_set = false;           // cfg.layer_le set: the split path (every new branch is gated on this)
    bool eng_used[2] = { true, true };// Engram slot used by a layer in the range: alloc_cols makes that i_eng
    bool own_backend = false, cpu = true;
    WStore w;
    Ds41Engram engram;
    std::string err;

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, RQ = 0, SWA = 0, RAW = 0;
    int64_t IH = 0, ID = 0, ITOP = 0, NE = 0, NU = 0, NFF = 0, NC = 0, CB = 0, CBS = 0, EC = 0, ED = 0;
    int64_t cmax[3] = { 0, 0, 0 };        // compressed rows per ratio (index 1 / 2)

    // ---- persistent state (sctx / sbuf)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor * x_state = nullptr;      // [D, HC, NC]
    ggml_tensor * pre_state = nullptr;    // [HC, NC]   the next attention's pre-mix
    ggml_tensor * hap = nullptr;          // [D, HC, NC] the FFN residual (hc_post of the attention)
    ggml_tensor * post_f = nullptr;       // [HC, NC]
    ggml_tensor * comb_f = nullptr;       // [HC, HC, NC]
    ggml_tensor * shexp = nullptr;        // [D, NC]
    ggml_tensor * o_fn = nullptr;         // [D, NC]    ffn_norm activation (the routed experts' input)
    ggml_tensor * o_ids = nullptr;        // I32 [NU, NC]
    ggml_tensor * o_wts = nullptr;        // [NU, NC]
    ggml_tensor * routed = nullptr;       // [D, NC]
    ggml_tensor * cand = nullptr;         // I32 [CB, NC]  candidate blocks of cand_layer
    struct Layer {
        int64_t ratio = 0;
        bool kv_src = false, idx_src = false, cand_src = false, uses_cand = false;
        int64_t kvl = -1, idxl = -1;      // the layers whose compressed KV / top-k this one reads
        int64_t eslot = -1;               // Engram slot
        ggml_tensor * raw = nullptr;      // [DH, RAW]       window K (= V), post-RoPE
        ggml_tensor * comp = nullptr;     // [DH, cmax + 1]  compressed KV (kv sources); row cmax = spare
        ggml_tensor * idxk = nullptr;     // [ID, cmax + 1]  indexer keys (kv sources)
        ggml_tensor * st_kv = nullptr;    // [DH, 2]         ratio-2 compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [DH, 2]         ... (scores)
        ggml_tensor * sel = nullptr;      // I32 [ITOP, NC]  top-k rows (index sources)
        ggml_tensor * selm = nullptr;     // F32 [ITOP, NC]  0 / -inf (row visible to the query)
        ggml_tensor * raw_snap = nullptr, * st_kv_snap = nullptr, * st_sc_snap = nullptr;   // snapshot()
        ggml_tensor * rbias = nullptr;    // F32 [NE]        selection-only routing bias (set_route_bias), 0 = model routing
    };
    int snap_pos = -1;
    struct ParkedRows { ggml_tensor * t; size_t off; std::vector<uint8_t> b; };
    std::vector<ParkedRows> parked;     // park(): host copies of the snapshot rings + the cache rows an unrelated request overwrites
    int parked_pos = -1;
    std::vector<ParkedRows> prefix, parked_prefix;   // independent host-backed pre-tail rings
    int prefix_pos = -1, parked_prefix_pos = -1;
    std::vector<Layer> ly;
    // ---- the split hand-off (open_handoff): this stage's pinned+mapped buffer (x_state then pre_state, NCmax
    // columns each) + its device pointer; hand_in is the PREVIOUS stage's buffer, set by the driver.  The payload
    // is the pass's cross-layer state, columns [0, n).
    float * hand_buf = nullptr, * hand_dev = nullptr, * hand_in = nullptr;
    bool hand_pinned = false;         // hand_buf came from cudaHostAlloc (else malloc, the CPU backend)

    // ---- per-pass inputs (ictx / ibuf), uploaded by begin()
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    // ---- the NC-column tensors (state columns + their per-pass inputs) in their own context / buffer, so
    // set_columns() can resize them: --serve keeps S columns while it decodes and grows to the prompt batch only for a
    // layer-major batch (~0.31 MiB a column: 8192 columns = 2.6 GiB that decode gives to expert slots instead)
    ggml_context * nctx = nullptr;
    ggml_backend_buffer_t nbuf = nullptr;
    int64_t NCmax = 0;
    ggml_tensor * i_emb = nullptr;        // [D, NC]
    ggml_tensor * i_pos = nullptr;        // I32 [NC]
    ggml_tensor * i_widx = nullptr;       // I32 [SWA * NC]   per query: rows of [ring | pass rows]
    ggml_tensor * i_wmask = nullptr;      // F32 [SWA, NC]
    ggml_tensor * i_wslot = nullptr;      // I32 [min(NC, RAW)]  ring slots of the pass's last rows
    ggml_tensor * i_nvis[3] = {};         // F32 [NC]   visible compressed rows per query (ratio 1 / 2)
    ggml_tensor * i_cslot[3] = {};        // I32 [NC]   compressed row each token writes (spare if its group is open)
    ggml_tensor * i_cpos[3] = {};         // I32 [NC]   RoPE position of that row (group start)
    ggml_tensor * i_sidx = nullptr;       // I32 [2 * NC] ratio-2 state rows per token, of [state ring | pass states]
    ggml_tensor * i_sslot = nullptr;      // I32 [min(NC, 2)]
    // upload_span's last (pos0, col0, n): its four index tensors depend only on these, not on the layer, so the 39
    // layers after the first skip identical re-uploads.  Those tiny H2D copies queued behind the chunk prestage's
    // ~1.5 GiB DMA of the next layer's experts and cost ~65 ms a layer (~2.5 s a turn).  Any other writer of the four
    // tensors (upload_inputs) or their reallocation (alloc_state) resets it.
    int64_t span_memo_pos0 = -1;
    int span_memo_col0 = -1, span_memo_n = -1;
    ggml_tensor * i_lastblk = nullptr;    // F32 [NC]   (nvis - 1) / block size: the pinned candidate block (ratio 1)
    ggml_tensor * i_eng[2] = {};          // F32 [EC * ED, NC]  Engram rows
    ggml_tensor * neg_ramp = nullptr;     // F32 [cmax1]  -i     (constant)
    ggml_tensor * blk_ramp = nullptr;     // F32 [cmax1 / CBS + 1]  i (constant)

    // ---- the pass
    int pos0 = 0, n = 0;
    int64_t pass_cap[3] = { 0, 0, 0 };
    // ---- the layer-major prompt batch (begin_batch): S = the dense sub-batch size, batch = B, span_cap per sub-batch
    int64_t S = 1, batch = 0, col0 = 0;
    int batch_pos0 = 0;                             // the batch's first position
    std::vector<std::array<int64_t, 3>> span_cap;   // [sub-batch][ratio] pass_cap
    std::vector<float> host_fn, host_wts, host_logits, host_emb;
    std::vector<int32_t> host_ids;
    const uint8_t * embd = nullptr;       // token_embd rows on the mmap
    size_t embd_row_bytes = 0;            // the stored ggml row stride, not width * sizeof(uint16_t)
    ggml_to_float_t embd_to_float = nullptr;
    // ---- vision (DeepSeek-V4.1 image input): the request's image span rows, by absolute position.  When enabled,
    // set_vision() swaps in exp_probs_b_vl on those positions and
    // marks them DEAD for Engram.  Empty (n_vis = 0) for a text request.
    bool vision = false;                  // opt-in; requires per-layer exp_probs_b_vl.bias
    ggml_tensor * i_img = nullptr;        // F32 [NC]  1.0 at an image position, 0.0 elsewhere (the VL bias / gate mask)
    const float * vis_rows = nullptr;     // n_vis * D floats, already in prompt order
    const int64_t * vis_pos = nullptr;    // n_vis ascending absolute prompt positions
    int n_vis = 0;
    std::vector<std::vector<float>> eng_rows;
    // The pass's Engram reads, started (async) by upload_inputs and waited at the first Engram layer's span /
    // attn_router: the whole pass's rows are read from disk while the embeddings, the init graph and the layers below
    // the first Engram layer (1) run.  Null once uploaded (or when Engram is off) - see engram_ready().
    std::shared_ptr<Ds41Engram::Batch> eng_batch;

    // ---- graphs
    ggml_context * gctx = nullptr;        // node structs of the cached (n = 1) graphs
    std::vector<uint8_t> pf_meta;         // node structs of a pass graph with n > 1 (built per pass)
    Allo allo_dec, allo_pass;
    struct GKey { int il, kind; int64_t cap, n; bool operator<(const GKey & o) const {
        return std::tie(il, kind, cap, n) < std::tie(o.il, o.kind, o.cap, o.n); } };
    std::map<GKey, ggml_cgraph *> graphs;
    std::map<int, ggml_tensor *> logits_out;   // decode head graph output, by nrows
    std::map<int, ggml_tensor *> predict_out;  // predict graph output, by layer
    std::map<std::pair<int, int>, ggml_tensor *> predict_out_n;   // predict_multi output, by (layer, pass width)
    std::map<int, ggml_tensor *> fpredict_out; // finish_predict graph output, by the predicted layer
    std::map<std::pair<int, int>, std::vector<ggml_tensor *>> fpredict_ahead; // finish_predict_ahead extra outputs

    ~Impl() {
        if (nbuf) ggml_backend_buffer_free(nbuf);
        if (nctx) ggml_free(nctx);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
#ifdef DS41_HAVE_CUDA
        if (hand_buf && hand_pinned) cudaFreeHost(hand_buf);
#endif
        if (hand_buf && !hand_pinned) std::free(hand_buf);
    }
    ggml_tensor * W(const std::string & n) const { return w.get(n); }
    ggml_tensor * BL(int64_t il, const std::string & s) const { return w.get("blk." + std::to_string(il) + "." + s); }
    ggml_tensor * engram_gate_weights(ggml_context * gc, int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".engram_gate.weight");
        return it != w.t.end() ? ds41_engram_gate_weights(gc, it->second, nullptr, nullptr)
                              : ds41_engram_gate_weights(gc, nullptr, BL(il, "engram_q_norm.weight"), BL(il, "engram_k_norm.weight"));
    }
    // the router as the prediction graphs read it: the STRATA_PRED_ROUTER copy when the loader made one, else the F32
    ggml_tensor * BLpred(int64_t il) const {
        const auto it = w.t.find("blk." + std::to_string(il) + ".ffn_gate_inp_pred.weight");
        return it != w.t.end() ? it->second : BL(il, "ffn_gate_inp.weight");
    }
    bool compute(ggml_cgraph * gf, bool cached);
    bool upload_inputs(const int * tids, int n, int pos0);
    // absolute layer id -> this instance's range (the unset range is every block, so this is true for every layer)
    bool in_range(int64_t il) const { return il >= lb && il < le; }
    bool upload_pass_maps(int n, int pos0);
};

// ================================================================== weights

static bool load_weights(Ds41Dense::Impl & im, std::string & err) {
    WStore & w = im.w;
    const GgufModel & model = *im.model;
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    w.ctx = ggml_init(ip);
    // What the file calls a tensor and what type it stores it as are not always what the engine wants: a file in
    // llama.cpp's spelling renames the three Engram tensors and stores dense matrices in BF16 where the engine
    // computes in F16 (ds41_geometry.hpp).  Both maps are empty for a file v41_gguf.py wrote, so the file's own
    // name and type are used and nothing below changes for it.
    const Ds41Geometry & G = im.g;
    auto ename = [&](const std::string & n) -> std::string {
        const auto it = G.rename.find(n);
        return it == G.rename.end() ? n : it->second;
    };
    auto wtype = [&](const std::string & n) -> uint32_t {   // ~0u = the engine has no opinion; take the file's type
        const auto it = G.want.find(ename(n));
        return it == G.want.end() ? ~0u : it->second;
    };
    auto make = [&](const TensorInfo & ti, int type = -1) {   // type: the in-memory type (default the file's)
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
        ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) (type >= 0 ? type : (int) ti.type), (int) ti.shape.size(), ne);
        ggml_set_name(t, ename(ti.name).c_str());
        return t;
    };
    // routed experts never reach ggml (PTQ1_0 = 143 is past GGML_TYPE_COUNT); the I32 maps are read by the geometry;
    // the Engram tables a llama.cpp release carries are read by ds41_engram off the shard, never through ggml
    auto skip = [&](const std::string & n) {
        if (is_routed(n) || n == "token_embd.weight") return true;
        for (const std::string & e : G.engram_tables) if (e == n) return true;
        // the split range (src/core/native_dense.cpp's in_range): blk.<n> only inside [lb, le); the non-block
        // tensors left here are the head (output.weight + output_norm.weight), which only the last stage loads
        if (!im.range_set) return false;
        if (n.rfind("blk.", 0) == 0) { const int64_t l = std::atoll(n.c_str() + 4); return l < im.lb || l >= im.le; }
        return im.le != im.nl;
    };
    // the file stores this tensor in another type than the engine reads it in: convert it, do not bind or copy it
    auto converts = [&](const TensorInfo & ti) { const uint32_t wt = wtype(ti.name); return wt != ~0u && wt != ti.type; };
    // A file whose dense matrix is in a type ggml cannot dequantize, or that the engine wants in a type ggml
    // cannot write, is an error here - not a null trait pointer three hundred tensors later.  F32 is the identity
    // at both ends: ggml leaves its trait pointers null, and convert_upload / quantize_chunk memcpy it.
    auto readable = [](uint32_t t) { return (ggml_type) t == GGML_TYPE_F32 || ggml_get_type_traits((ggml_type) t)->to_float; };
    auto writable = [](uint32_t t) {
        return (ggml_type) t == GGML_TYPE_F32 || ggml_get_type_traits((ggml_type) t)->from_float_ref;
    };
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            const uint32_t wt = wtype(ti.name);
            if (skip(ti.name) || wt == ~0u || wt == ti.type) continue;
            if (!readable(ti.type) || !writable(wt)) {
                err = ename(ti.name) + ": cannot convert " + ggml_type_name(ti.type) + " to " + ggml_type_name(wt);
                return false;
            }
        }

    // rows of the file's tensor `ti` (bytes `src`) converted to t->type and written to t; returns the bytes written
    auto convert_upload = [&](ggml_tensor * t, const TensorInfo & ti, const uint8_t * src) -> size_t {
        const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
        const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
        std::vector<uint8_t> dst((size_t) nr * dst_row);
        const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int k = 0; k < nth; ++k)
            th.emplace_back([&, k] {
                std::vector<float> f((size_t) n0);
                for (int64_t r = k; r < nr; r += nth) {
                    if ((ggml_type) ti.type == GGML_TYPE_F32) std::memcpy(f.data(), src + (size_t) r * src_row, (size_t) n0 * 4);
                    else ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                    ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                }
            });
        for (auto & x : th) x.join();
        ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        return dst.size();
    };
    size_t cv_n = 0;
    double cv_from = 0, cv_to = 0;
    if (im.cpu) {
        std::vector<std::pair<size_t, const TensorInfo *>> cvt;
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & ts = sh.tensors();
            if (ts.empty()) continue;
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            w.bufs.push_back(buf);
            for (const auto & ti : ts) {
                if (skip(ti.name)) continue;
                if (converts(ti)) { cvt.push_back({ s, &ti }); continue; }
                ggml_tensor * t = make(ti);
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ename(ti.name)] = t;
            }
        }
        // A tensor the file stores in another type cannot bind to the mmap: it gets a host buffer of its own,
        // filled row by row.  A file v41_gguf.py wrote has none, and this block does not run for it.
        if (!cvt.empty()) {
            ggml_context * cc = ggml_init({ 8ull * 1024 * 1024, nullptr, true });
            w.ctx2 = cc;
            for (const auto & [s, ti] : cvt) {
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti->shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti->shape[d];
                ggml_tensor * t = ggml_new_tensor(cc, (ggml_type) wtype(ti->name), (int) ti->shape.size(), ne);
                ggml_set_name(t, ename(ti->name).c_str());
                w.t[ename(ti->name)] = t;
            }
            ggml_backend_buffer_t b2 = ggml_backend_alloc_ctx_tensors(cc, im.backend);
            if (!b2) { err = "cannot allocate the converted dense weights on the host"; return false; }
            w.bufs.push_back(b2);
            for (const auto & [s, ti] : cvt) {
                ggml_tensor * t = w.t[ename(ti->name)];
                cv_from += (double) ggml_nrows(t) * ggml_row_size((ggml_type) ti->type, t->ne[0]);
                cv_to += (double) convert_upload(t, *ti, model.shard(s).tensor_data(*ti));
                ++cv_n;
            }
        }
    } else {
        // STRATA_DENSE_REQUANT=<ggml type name> (q6_K, q5_K, q4_K, ...; quality experiment, OFF by default): every
        // Q8_0 dense matrix (attention, shared expert; output.weight unless STRATA_DENSE_REQUANT_HEAD=0) is
        // re-quantized in memory at load - rows dequantized, quantized to that type.  The file is not touched.
        ggml_type rq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_DENSE_REQUANT"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) rq = (ggml_type) t;
        const bool rq_head = !(std::getenv("STRATA_DENSE_REQUANT_HEAD") && std::getenv("STRATA_DENSE_REQUANT_HEAD")[0] == '0');
        auto requant = [&](const TensorInfo & ti) {
            return rq != GGML_TYPE_COUNT && (ggml_type) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() == 2 &&
                   (int64_t) ti.shape[0] % ggml_blck_size(rq) == 0 && (rq_head || ti.name != "output.weight") &&
                   !converts(ti);   // a type the file does not already match is the file's problem, not requant's
        };
        // STRATA_PRED_ROUTER=<ggml type> (q8_0, f16; OFF by default): the PREDICTION graphs (the prefetch top-16 and the
        // speculative reads' rows) read a copy of each layer's F32 router in that type (q8_0: 7.9 -> 2.1 MB a read); the
        // routing itself (build_attn) keeps the file's F32.  Only which experts are prefetched can change.
        ggml_type pq = GGML_TYPE_COUNT;
        if (const char * e = std::getenv("STRATA_PRED_ROUTER"); e && *e)
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && strcasecmp(ggml_type_name((ggml_type) t), e) == 0) pq = (ggml_type) t;
        auto pred_name = [&](const TensorInfo & ti) -> std::string {   // "" = no prediction copy for this tensor
            static const std::string suf = "ffn_gate_inp.weight";
            const std::string & n = ti.name;
            if (pq == GGML_TYPE_COUNT || (ggml_type) ti.type != GGML_TYPE_F32 || ti.shape.size() != 2 ||
                (int64_t) ti.shape[0] % ggml_blck_size(pq) != 0 || n.size() <= suf.size() ||
                n.compare(n.size() - suf.size(), suf.size(), suf) != 0) return std::string();
            return n.substr(0, n.size() - suf.size()) + "ffn_gate_inp_pred.weight";
        };
        size_t rq_n = 0, rq_from = 0, rq_to = 0, pq_n = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                const uint32_t wt = wtype(ti.name);   // ~0u -> (int) -1: make() keeps the file's own type
                w.t[ename(ti.name)] = make(ti, requant(ti) ? (int) rq : (int) wt);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    ggml_tensor * tp = make(ti, (int) pq);
                    ggml_set_name(tp, pn.c_str());
                    w.t[pn] = tp;
                }
            }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!buf) { err = "cannot allocate the dense weights on the backend"; return false; }
        w.bufs.push_back(buf);
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto & ti : model.shard(s).tensors()) {
                if (skip(ti.name)) continue;
                ggml_tensor * t = w.t[ename(ti.name)];
                const uint8_t * src = model.shard(s).tensor_data(ti);
                if (const std::string pn = pred_name(ti); !pn.empty()) {
                    convert_upload(w.t[pn], ti, src);
                    ++pq_n;
                }
                if ((int) t->type == (int) ti.type) {
                    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
                    continue;
                }
                const double from = (double) ggml_nrows(t) * ggml_row_size((ggml_type) ti.type, t->ne[0]);
                const double to = (double) convert_upload(t, ti, src);
                if (converts(ti)) { cv_from += from; cv_to += to; ++cv_n; continue; }
                rq_to += to;
                ++rq_n;
                rq_from += from;
            }
        if (rq_n) std::fprintf(stderr, "dense requant: %zu Q8_0 matrices -> %s, %.2f -> %.2f GB\n", rq_n, ggml_type_name(rq),
                               rq_from / 1e9, rq_to / 1e9);
        if (pq_n) std::fprintf(stderr, "prediction routers: %zu F32 copies as %s\n", pq_n, ggml_type_name(pq));
    }
    if (cv_n) std::fprintf(stderr, "dense types: %zu matrices converted from the file's type to the engine's, %.2f -> %.2f GB\n",
                           cv_n, cv_from / 1e9, cv_to / 1e9);
    size_t si = 0;
    // token_embd.weight is the mmap pointer begin() reads; a stage past 0 does not load it (the loader rule), so
    // its absence is not an error here - upload_inputs reports it if a begin() asks for embeddings on such a stage
    const TensorInfo * te = model.find("token_embd.weight", &si);
    if (te) {
        if (!ds41_embedding_layout(te->type, im.D, im.embd_row_bytes, im.embd_to_float)) {
            err = "token_embd.weight: unsupported row type or width";
            return false;
        }
        im.embd = model.shard(si).tensor_data(*te);
    }
    return true;
}

// ================================================================== graph helpers

namespace {

struct B {
    Ds41Dense::Impl * im;
    ggml_context * c;
    const Ds41Geometry * g;

    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * cols2(ggml_tensor * t, int64_t n0, int64_t n1, int64_t c0 = 0) const {   // columns c0.. of a [n0, *] tensor
        return ggml_view_2d(c, t, n0, n1, t->nb[1], (size_t) c0 * t->nb[1]);
    }
    ggml_tensor * cols3(ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t c0 = 0) const {
        return ggml_view_3d(c, t, n0, n1, n2, t->nb[1], t->nb[2], (size_t) c0 * t->nb[2]);
    }
    ggml_tensor * flat2(ggml_tensor * t, int64_t ne0, int64_t ne1) const {
        return ggml_is_contiguous(t) ? ggml_reshape_2d(c, t, ne0, ne1) : ggml_cont_2d(c, t, ne0, ne1);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wt);
    }
    // hc_mixes (model.py Block.hc_mixes + kernel hc_split_sinkhorn): x [D, HC, n] -> pre [HC, n], post [HC, n],
    // comb [HC(dst), HC(src), n]
    void hc_mixes(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                  ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t HC = g->hc, nt = x->ne[2], D = g->n_embd;
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                  // [24, n]
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
#if defined(STRATA_DSV41_HC_MIX)
        // -DSTRATA_DSV41_HC_MIX=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fuse.patch: the two mixes as
        // ONE op (GGML_OP_DSV41_HC_MIXES), the same mul/add/sigmoid/scale steps bit for bit (test_ds41_hcmix).  It
        // replaces 8 of the layer graph's ~82 tiny kernels per call.  STRATA_HC_MIX_COMPOSITE=1: the composite.
        static const bool composite = std::getenv("STRATA_HC_MIX_COMPOSITE") != nullptr;
        if (!composite) {
            ggml_tensor * mix = ggml_dsv41_hc_mixes(c, mixes, v1(sc, 2, 0), v1(bs, 2 * HC, 0), (float) g->hc_eps);
            *pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
            *post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
            return;
        }
#endif
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, v1(sc, 1, 0)), v1(bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) g->hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, v1(sc, 1, sizeof(float))), v1(bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_iters);
    }
    // RoPE over the last `nd` dims of each `ne0` row (model.py apply_rotary_emb: adjacent pairs = NORMAL mode).
    // ratio > 0: compress_rope_theta with YaRN; ratio 0: rope_theta, no YaRN (model.py Attention.__init__)
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t ratio, bool back = false) const {
        const bool yarn = ratio != 0;
        const float base = (float) (yarn ? g->compress_rope_base : g->rope_base);
        const float fscale = yarn ? (float) (1.0 / g->rope_factor) : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        // ggml multiplies by mscale = attn_factor * (1 + 0.1 ln(1/fscale)) under YaRN; model.py has no such scale
        const float attn = yarn ? 1.0f / (1.0f + 0.1f * std::log(1.0f / fscale)) : 1.0f;
        const float bf = yarn ? (float) g->beta_fast : 0.0f, bsl = yarn ? (float) g->beta_slow : 0.0f;
        const int n_ctx = yarn ? (int) g->orig_ctx : 0;
        const int nd = (int) g->d_rope;
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bsl);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    }
};

}  // namespace

// STRATA_DENSE_PROF=1 (diagnostic, off by default): where a layer-major batch's dense time goes - graph build, gallocr
// allocation, GPU compute, Engram wait, readback - summed per prompt and printed by prof_report()
static const bool g_dprof_on = std::getenv("STRATA_DENSE_PROF") != nullptr;
static double g_dprof[8] = {};   // build, alloc, compute, engram, readback, finish build, upload, calls
static double dprof_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_dprof_slot = -1;    // which of build/alloc/compute buckets compute() charges: set by the span callers

bool Ds41Dense::Impl::compute(ggml_cgraph * gf, bool cached) {
    Allo & a = cached ? allo_dec : allo_pass;
    const double tq0 = g_dprof_on ? dprof_now() : 0;
    bool ok_alloc = a.alloc(gf, cached && env_on("DS41_GRAPH_REUSE", true));
    const double tq1 = g_dprof_on ? dprof_now() : 0;
    if (!ok_alloc && oom_hook && oom_hook()) {   // something gave VRAM back: once more
        std::fprintf(stderr, "ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying\n");
        ok_alloc = a.alloc(gf, false);
    }
    if (!ok_alloc) { err = "graph allocation failed"; return false; }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; return false; }
    if (g_dprof_on && g_dprof_slot >= 0) { g_dprof[1] += tq1 - tq0; g_dprof[2] += dprof_now() - tq1; }
    return true;
}

// ================================================================== the attention graph (+ router + shared expert)

// the model's activation rounding: ds41_ops.hpp's stock-op composite (~45 ops per call), or - built with
// -DSTRATA_DSV41_FQ_OP=ON on a llama.cpp tree patched with tools/v41/ggml/dsv41-fake-quant.patch - the fused
// GGML_OP_DSV41_FAKE_QUANT, one kernel per call (bit-identical on CUDA and CPU: test_ds41_fq; ~3,000 fewer tiny
// kernels per decode token).  STRATA_FQ_COMPOSITE=1 in such a build: the composite.
static ggml_tensor * fake_quant(ggml_context * gc, ggml_tensor * x, Ds41Quant mode) {
#if defined(STRATA_DSV41_FQ_OP)
    static const bool composite = std::getenv("STRATA_FQ_COMPOSITE") != nullptr;
    return composite ? ds41_fake_quant(gc, x, mode) : ggml_dsv41_fake_quant(gc, x, (int) mode);
#else
    return ds41_fake_quant(gc, x, mode);
#endif
}

static ggml_cgraph * build_attn(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH, NH = im.NH, SWA = im.SWA, RAW = im.RAW;
    const int64_t r = L.ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    std::vector<ggml_tensor *> outs, late;    // late: ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    // ---- Engram (model.py Engram.forward), before the block, on the hc state
    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);          // [(HC+1) D, n]
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = im.engram_gate_weights(gc, il);   // [D, HC], folded or the original Q*K
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn));    // [1, HC, n]
        dot = ggml_scale(gc, dot, 1.0f / std::sqrt((float) D));
        // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))  (signed sqrt, as the training kernel)
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));   // gate * (1 - image)
        }
        ggml_tensor * gv = ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate);
        x = ggml_add(gc, x, gv);
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));   // the block input (residual) = state + Engram
    }

    // ---- attention hc: mixes of the block input; its pre-mix is the previous FFN's (pre_state)
    ggml_tensor * a_pre = nullptr, * a_post = nullptr, * a_comb = nullptr;
    b.hc_mixes(x, im.BL(il, "hc_attn_fn.weight"), im.BL(il, "hc_attn_scale.weight"), im.BL(il, "hc_attn_base.weight"),
               &a_pre, &a_post, &a_comb);
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));             // [D, n]
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- q (no per-head norm in V4.1)
    ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_q_a.weight"), xn), im.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_reshape_3d(gc, ggml_mul_mat(gc, im.BL(il, "attn_q_b.weight"), qr), DH, NH, n);
    q = b.rope(q, pos, r);

    // ---- window K (= V): [ring | this pass's rows] -> each query's SWA rows; the ring keeps the last RAW
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);   // model.py _window_kv
    ggml_tensor * wsrc = ggml_concat(gc, L.raw->type == GGML_TYPE_F32 ? L.raw : ggml_cast(gc, L.raw, GGML_TYPE_F32), kv2, 1);     // [DH, RAW + n]
    ggml_tensor * Kw = ggml_reshape_3d(gc, ggml_get_rows(gc, wsrc, b.v1(im.i_widx, SWA * n, (size_t) col0 * SWA * 4)), DH, SWA, n);
    ggml_tensor * Mw = b.cols2(im.i_wmask, SWA, n, col0);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV (kv sources write it; every ratio > 0 layer reads its source's)
    ggml_tensor * Kc = nullptr, * Mc = nullptr;
    ggml_tensor * comp_w = nullptr, * idxk_w = nullptr;   // this pass's writes (later reads depend on them)
    if (r) {
        if (L.kv_src) {
            ggml_tensor * latent = nullptr;                                                  // [DH, n]
            if (r == 1) {   // model.py Compressor.forward: ratio 1 = a plain projection, no gate
                latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            } else {        // ratio 2: softmax-gated pooling over the group's 2 states, per dim (fp32)
                ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
                ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
                ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
                ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
                ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);                     // [DH, 2 + n]
                ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
                ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
                ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
                ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
                vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));                       // [2, DH, n]
                vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
                latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
                const int64_t k = std::min<int64_t>(n, 2);
                late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
                late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                             b.v1(im.i_sslot, k, 0)));
            }
            latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
            ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
            ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
            // indexer keys from the UNROTATED latent (model.py Indexer.forward, owns_k)
            ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
            ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
            if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
            idxk_w = ggml_set_rows(gc, L.idxk, ik, cslot);
            outs.push_back(idxk_w);
            ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
            if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
            comp_w = ggml_set_rows(gc, L.comp, lr, cslot);
            outs.push_back(comp_w);
        }
        if (cap > 0) {
            const L_t & KS = im.ly[(size_t) L.kvl];
            const int64_t K = std::min<int64_t>(cap, im.ITOP);
            ggml_tensor * comp = comp_w ? comp_w : KS.comp;
            ggml_tensor * idxk = idxk_w ? idxk_w : KS.idxk;
            ggml_tensor * sel = nullptr, * selm = nullptr;
            if (L.idx_src) {
                // ---- indexer (model.py Indexer.forward): scores over the visible rows of the shared index keys
                ggml_tensor * iq = ggml_mul_mat(gc, im.BL(il, "indexer.attn_q_b.weight"), qr);
                iq = b.rope(ggml_reshape_3d(gc, iq, im.ID, im.IH, n), pos, r);
                if (im.cfg.act_quant) iq = fake_quant(gc, ggml_cont(gc, iq), Ds41Quant::FP4_POW2_32);
                ggml_tensor * iw = ggml_mul_mat(gc, im.BL(il, "indexer.proj.weight"), xn);
                iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.ID * im.IH)));         // softmax_scale * heads^-0.5
                ggml_tensor * ksrc = ggml_view_3d(gc, idxk, im.ID, 1, cap, idxk->nb[1], idxk->nb[1], 0);
                // visibility: row i visible to query t iff i < nvis_t  -> 0 / -inf
                ggml_tensor * nvs = ggml_reshape_2d(gc, b.v1(im.i_nvis[r], n, (size_t) col0 * 4), 1, n);
                ggml_tensor * ramp = ggml_view_2d(gc, im.neg_ramp, cap, 1, (size_t) cap * 4, 0);
                ggml_tensor * vis = ggml_log(gc, ggml_step(gc, ggml_add(gc, ggml_repeat(gc, ramp, ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, n)), nvs)));
                ggml_tensor * sc = ggml_lightning_indexer(gc, iq, ksrc, iw, ggml_cast(gc, vis, GGML_TYPE_F16));   // [cap, n]
                sc = ggml_reshape_2d(gc, sc, cap, n);
                if (L.uses_cand) {   // rank only inside cand_layer's candidate blocks
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    ggml_tensor * m = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, nblk, n), NEG_INF);
                    ggml_tensor * z = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, im.CBS, ncb, n), 0.0f);
                    m = ggml_set_rows(gc, m, z, ggml_cont(gc, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                    m = ggml_view_2d(gc, m, cap, n, m->nb[2], 0);
                    sc = ggml_add(gc, sc, m);
                }
                if (L.cand_src) {   // model.py select_candidate_blocks: block score = max, newest block pinned
                    const int64_t nblk = (cap + im.CBS - 1) / im.CBS, ncb = std::min<int64_t>(im.CB, nblk);
                    GGML_ASSERT(cap % im.CBS == 0);
                    ggml_tensor * bm = ggml_pool_2d(gc, sc, GGML_OP_POOL_MAX, (int) im.CBS, 1, (int) im.CBS, 1, 0, 0);   // [nblk, n]
                    // pin: + 1e30 at block (nvis - 1) / CBS of each query
                    ggml_tensor * br = ggml_view_2d(gc, im.blk_ramp, nblk, 1, (size_t) nblk * 4, 0);
                    ggml_tensor * lb = ggml_reshape_2d(gc, b.v1(im.i_lastblk, n, (size_t) col0 * 4), 1, n);
                    ggml_tensor * eq = ggml_step(gc, ggml_scale_bias(gc, ggml_abs(gc, ggml_sub(gc, ggml_repeat(gc, br, bm), lb)), -1.0f, 0.5f));
                    bm = ggml_add(gc, bm, ggml_scale(gc, eq, 1e30f));
                    ggml_tensor * cb = ggml_cont(gc, ggml_top_k(gc, bm, (int) ncb));                     // I32 [ncb, n]
                    outs.push_back(ggml_cpy(gc, cb, ggml_view_2d(gc, im.cand, ncb, n, im.cand->nb[1], (size_t) col0 * im.cand->nb[1])));
                }
                sel = ggml_cont(gc, ggml_top_k(gc, sc, (int) K));                                     // I32 [K, n]
                // Canonicalise the top-k order.  ggml_cuda_op_top_k asks CUB's DeviceTopK for
                // output_ordering::unsorted, and CUB hands the k ids back in an order its atomically assigned
                // output slots produce - run-to-run unstable, while the SET is stable (measured: order-only on
                // every differing row).  The ids drive the gather of the compressed rows, so that order is the
                // flash-attn's accumulation order; left alone the whole run stops being reproducible from the
                // first query whose compressed rows are attended (the SWA window's 128).  Sort ascending.
                sel = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, sel, 1, K, n),
                                                        ggml_argsort(gc, ggml_cast(gc, sel, GGML_TYPE_F32),
                                                                     GGML_SORT_ORDER_ASC)), K, n);
                selm = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, vis, 1, cap, n), sel), K, n);   // vis at sel
                outs.push_back(ggml_cpy(gc, sel, ggml_view_2d(gc, L.sel, K, n, L.sel->nb[1], (size_t) col0 * L.sel->nb[1])));
                outs.push_back(ggml_cpy(gc, selm, ggml_view_2d(gc, L.selm, K, n, L.selm->nb[1], (size_t) col0 * L.selm->nb[1])));
            } else {
                const L_t & IS = im.ly[(size_t) L.idxl];
                sel = ggml_view_2d(gc, IS.sel, K, n, IS.sel->nb[1], (size_t) col0 * IS.sel->nb[1]);
                selm = ggml_view_2d(gc, IS.selm, K, n, IS.selm->nb[1], (size_t) col0 * IS.selm->nb[1]);
            }
            Kc = ggml_reshape_3d(gc, ggml_get_rows(gc, comp, ggml_cont(gc, ggml_reshape_1d(gc, ggml_cont(gc, sel), K * n))), DH, K, n);
            Mc = ggml_cont(gc, selm);
            // the source's rows must be read AFTER this pass wrote its own (a kv source attends to its new row)
        }
    }

    // ---- attention: per query its window rows + its top-k compressed rows, one flash-attn with the query as batch
    ggml_tensor * Kall = Kc ? ggml_concat(gc, Kw, Kc, 1) : Kw;                                 // [DH, M, n]
    ggml_tensor * Mall = Mc ? ggml_concat(gc, Mw, Mc, 0) : Mw;                                 // [M, n]
    int64_t M = Kall->ne[1];
    if (!im.cpu && M % 256) {   // ggml-cuda's flash-attn for d 512 wants the key count a multiple of 256
        const int64_t pad = 256 - M % 256;
        Kall = ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        Mall = ggml_concat(gc, Mall, ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, pad, n), NEG_INF), 0);
        M += pad;
    }
    ggml_tensor * k4 = ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, M, 1, n), GGML_TYPE_F16);
    ggml_tensor * q4 = ggml_permute(gc, ggml_reshape_4d(gc, q, DH, NH, 1, n), 0, 2, 1, 3);    // [DH, 1, NH, n]
    ggml_tensor * m4 = ggml_cast(gc, ggml_reshape_4d(gc, Mall, M, 1, 1, n), GGML_TYPE_F16);
    ggml_tensor * o = ggml_flash_attn_ext(gc, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, im.BL(il, "attn_sinks.weight"));
    o = ggml_reshape_3d(gc, o, DH, NH, n);
    o = b.rope(o, pos, r, true);                                                               // de-RoPE the output
    // grouped output LoRA (model.py: einsum over groups) then wo_b
    o = ggml_permute(gc, ggml_reshape_3d(gc, o, im.OGD, im.OG, n), 0, 2, 1, 3);               // [OGD, n, OG]
    ggml_tensor * wo_a = ggml_reshape_3d(gc, im.BL(il, "attn_output_a.weight"), im.OGD, im.OL, im.OG);
    ggml_tensor * oa = ggml_permute(gc, ggml_mul_mat(gc, wo_a, o), 0, 2, 1, 3);              // [OL, OG, n]
    ggml_tensor * attn = ggml_mul_mat(gc, im.BL(il, "attn_output_b.weight"), b.flat2(oa, im.OL * im.OG, n));   // [D, n]

    // ---- hc_post(attention) -> FFN hc: its mixes give the next layer's pre-mix; its input uses the attention's
    ggml_tensor * h = ggml_dsv4_hc_post(gc, attn, x, a_post, a_comb);                       // [D, HC, n]
    ggml_tensor * f_pre = nullptr, * f_post = nullptr, * f_comb = nullptr;
    b.hc_mixes(h, im.BL(il, "hc_ffn_fn.weight"), im.BL(il, "hc_ffn_scale.weight"), im.BL(il, "hc_ffn_base.weight"),
               &f_pre, &f_post, &f_comb);
    ggml_tensor * fn = b.rms_w(ggml_dsv4_hc_pre(gc, h, a_pre), im.BL(il, "ffn_norm.weight"));   // [D, n]

    // ---- router (model.py Gate): sqrt(softplus), top-k on score + bias, weights from the raw scores
    ggml_tensor * rl = ggml_mul_mat(gc, im.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = ggml_sqrt(gc, ggml_softplus(gc, rl));                                  // [NE, n]
    // selection = score + exp_probs_b + the selection-only rbias (soft prune / cache bonus); the weights below come
    // from the unbiased scores, as model.py's Gate - all-zero rbias (the default) = the model's routing bit for bit
    // vision: image positions select with exp_probs_b_vl.bias instead (model.py Gate: torch.where(image_mask, bias_vl, bias))
    ggml_tensor * bias_sel = im.BL(il, "exp_probs_b.bias");
    if (im.vision && im.i_img) {
        ggml_tensor * bt = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b.bias"), im.NE, 1);
        ggml_tensor * bv = ggml_reshape_2d(gc, im.BL(il, "exp_probs_b_vl.bias"), im.NE, 1);
        ggml_tensor * m = ggml_reshape_2d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, n);
        bias_sel = ggml_add(gc, ggml_mul(gc, ggml_repeat(gc, ggml_sub(gc, bv, bt), sc), m), bt);
    }
    ggml_tensor * s0 = ggml_add(gc, sc, bias_sel);                                             // the model's own selection score
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    ggml_tensor * sel = nullptr;
    ggml_tensor * ids = nullptr;
    // STRATA_RB_MAX_RANK=R opts into a bounded candidate pool (R is clamped to [NU, NE]); cache preference can
    // reorder only the model-corrected top-R.  STRATA_RB_PROTECT=P reserves the exact top-P by ID, so at most NU-P
    // route slots can change.  STRATA_RB_MARGIN additionally confines score changes to the cutoff band.
    // This trades some cache hits for a rank/slot bound, rather than weakening every cache bonus.  Unset retains the
    // legacy graph below, including its historical PROTECT/MARGIN behavior.  Neither policy changes raw weights.
    const int rb_protect = rb_protected();
    const float rb_margin = rb_score_margin();
    if (rb_max_rank() > 0) {
        ids = ds41_bounded_route_ids(gc, s0, rb, (int) im.NU, rb_max_rank(), rb_protect, rb_margin, (int) im.NU);
    } else if (rb_protect > 0 || rb_margin >= 0.0f) {
        const int kt = (int) im.NU + 1;
        ggml_tensor * ti = ggml_argsort_top_k(gc, s0, kt);                                         // [kt, n]
        ggml_tensor * tv = ggml_reshape_2d(gc, ggml_get_rows(gc, ggml_reshape_3d(gc, s0, 1, im.NE, n), ti), kt, n);   // [kt, n] desc
        auto row = [&](int r) { return ggml_view_2d(gc, tv, 1, n, tv->nb[1], (size_t) r * tv->nb[0]); };   // [1, n]
        ggml_tensor * like = ggml_new_tensor_2d(gc, GGML_TYPE_F32, im.NE, n);
        ggml_tensor * protect = nullptr;
        if (rb_protect > 0) {   // s0 >= the P-th largest exact score
            ggml_tensor * vp = ggml_repeat(gc, ggml_cont(gc, row(std::min(rb_protect, (int) im.NU) - 1)), like);
            protect = ggml_step(gc, ggml_scale_bias(gc, ggml_sub(gc, s0, vp), 1.0f, 1e-6f));
        }
        ggml_tensor * gate = nullptr;
        if (rb_margin >= 0.0f) {
            ggml_tensor * cut = ggml_repeat(gc, ggml_scale(gc, ggml_add(gc, ggml_cont(gc, row((int) im.NU - 1)), ggml_cont(gc, row((int) im.NU))), 0.5f), like);
            ggml_tensor * dv = ggml_sub(gc, s0, cut);                                              // [NE, n]
            ggml_tensor * above = ggml_step(gc, ggml_scale_bias(gc, dv, 1.0f, -rb_margin));          // dv > d
            ggml_tensor * below = ggml_step(gc, ggml_scale_bias(gc, dv, -1.0f, -rb_margin));         // dv < -d
            protect = protect ? ggml_clamp(gc, ggml_add(gc, protect, above), 0.0f, 1.0f) : above;
            gate = ggml_scale_bias(gc, ggml_clamp(gc, ggml_add(gc, above, below), 0.0f, 1.0f), -1.0f, 1.0f);   // in the band
        }
        ggml_tensor * unprot = ggml_scale_bias(gc, protect, -1.0f, 1.0f);
        if (gate) unprot = ggml_mul(gc, unprot, gate);
        sel = ggml_add(gc, ggml_add(gc, s0, ggml_mul(gc, ggml_repeat(gc, rb, like), unprot)), ggml_scale(gc, protect, 1e4f));
    } else {
        sel = ggml_add(gc, s0, rb);
    }
    if (!ids) ids = ggml_argsort_top_k(gc, sel, (int) im.NU);   // [NU, n]
    ggml_tensor * wts = ds41_route_weights(gc, sc, ids, G.norm_topk, (float) G.weights_scale);

    // ---- shared expert (model.py Expert, clamped SwiGLU)
    const float lim = (float) G.swiglu_limit;
    ggml_tensor * up = ggml_mul_mat(gc, im.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gt = ggml_mul_mat(gc, im.BL(il, "ffn_gate_shexp.weight"), fn);
    if (lim > 0) {
        up = ggml_clamp(gc, up, -lim, lim);
        gt = ggml_clamp(gc, gt, NEG_INF, lim);
    }
    ggml_tensor * se = ggml_mul_mat(gc, im.BL(il, "ffn_down_shexp.weight"), ggml_swiglu_split(gc, gt, up));

    outs.push_back(ggml_cpy(gc, h, b.cols3(im.hap, D, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, f_post, b.cols2(im.post_f, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, ggml_reshape_3d(gc, f_comb, HC, HC, n), b.cols3(im.comb_f, HC, HC, n, col0)));
    outs.push_back(ggml_cpy(gc, se, b.cols2(im.shexp, D, n, col0)));
    outs.push_back(ggml_cpy(gc, fn, b.cols2(im.o_fn, D, n, col0)));
    outs.push_back(ggml_cpy(gc, ids, b.cols2(im.o_ids, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, wts, b.cols2(im.o_wts, im.NU, n, col0)));
    outs.push_back(ggml_cpy(gc, f_pre, b.cols2(im.pre_state, HC, n, col0)));   // reads of pre_state happen above
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// enc-skip: the persistent-KV half of build_attn for one span.  Runs hc_pre / attn_norm, writes the raw window K/V
// ring, and on a kv source layer writes the compressed-KV row + indexer-key row (+ the ratio-2 compressor state).
// Everything else build_attn does (q, the attention + output LoRA, hc_post, the router, the shared expert, the indexer
// selection / candidate blocks, and the pre_state / hap / shexp / o_* writes) only ever flows into THIS layer's own
// output columns, so a prompt token whose later layers are never read does not need them.  The state it does write is
// exactly what build_attn writes for those positions, so a later query's window / compressed reads see the same
// values as the full path (verified by the enc-skip gate).
static ggml_cgraph * build_attn_kv_only(Ds41Dense::Impl & im, ggml_context * gc, int il, int64_t n, int64_t col0 = 0) {
    using L_t = Ds41Dense::Impl::Layer;
    L_t & L = im.ly[(size_t) il];
    const Ds41Geometry & G = im.g;
    B b { &im, gc, &G };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 8192, false);
    const int64_t D = im.D, HC = im.HC, DH = im.DH;
    const int64_t r = L.ratio;
    std::vector<ggml_tensor *> outs, late;   // late: the ring / state write-backs, after every read
    ggml_tensor * pos = b.v1(im.i_pos, n, (size_t) col0 * 4);

    ggml_tensor * x = b.cols3(im.x_state, D, HC, n, col0);
    if (L.eslot >= 0 && im.engram.enabled()) {   // the Engram add still feeds the K/V + compressor projections
        ggml_tensor * rows = b.cols2(im.i_eng[L.eslot], im.EC * im.ED, n, col0);
        ggml_tensor * kv = ggml_mul_mat(gc, im.BL(il, "engram_kv.weight"), rows);
        ggml_tensor * key = ggml_view_3d(gc, kv, D, HC, n, (size_t) D * 4, kv->nb[1], 0);
        ggml_tensor * val = ggml_view_3d(gc, kv, D, 1, n, (size_t) D * 4, kv->nb[1], (size_t) HC * D * 4);
        ggml_tensor * wqk = im.engram_gate_weights(gc, il);
        ggml_tensor * hn = ggml_rms_norm(gc, x, (float) G.rms_eps);
        ggml_tensor * kn = ggml_rms_norm(gc, ggml_cont(gc, key), (float) G.rms_eps);
        ggml_tensor * dot = ggml_scale(gc, ggml_sum_rows(gc, ggml_mul(gc, ggml_mul(gc, hn, wqk), kn)), 1.0f / std::sqrt((float) D));
        ggml_tensor * mag = ggml_sqrt(gc, ggml_clamp(gc, ggml_abs(gc, dot), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(gc, ggml_mul(gc, ggml_sgn(gc, dot), mag));
        if (im.vision && im.i_img) {   // image positions take no Engram contribution (model.py: gate masked_fill ~token_mask)
            ggml_tensor * m = ggml_reshape_3d(gc, b.v1(im.i_img, n, (size_t) col0 * 4), 1, 1, n);
            gate = ggml_mul(gc, gate, ggml_scale_bias(gc, m, -1.0f, 1.0f));
        }
        x = ggml_add(gc, x, ggml_mul(gc, ggml_repeat(gc, ggml_cont(gc, val), x), gate));
        outs.push_back(ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    }
    ggml_tensor * xin = ggml_dsv4_hc_pre(gc, x, b.cols2(im.pre_state, HC, n, col0));
    ggml_tensor * xn = b.rms_w(xin, im.BL(il, "attn_norm.weight"));

    // ---- window K (= V): the same post-RoPE raw row build_attn writes into the ring
    ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, im.BL(il, "attn_kv.weight"), xn), im.BL(il, "attn_kv_a_norm.weight"));
    kv = b.rope(ggml_reshape_3d(gc, kv, DH, 1, n), pos, r);
    ggml_tensor * kv2 = ggml_reshape_2d(gc, kv, DH, n);
    if (im.cfg.act_quant) kv2 = fake_quant(gc, ggml_cont(gc, kv2), Ds41Quant::FP8_POW2_32);
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        ggml_tensor * last = ggml_view_2d(gc, kv2, DH, k, kv2->nb[1], (size_t) (n - k) * kv2->nb[1]);
        late.push_back(ggml_set_rows(gc, L.raw, last, b.v1(im.i_wslot, k, 0)));
    }

    // ---- compressed KV + indexer keys (kv source layers; the only other persistent state a later query reads)
    if (r && L.kv_src) {
        ggml_tensor * latent = nullptr;
        if (r == 1) {
            latent = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
        } else {
            ggml_tensor * skv = ggml_mul_mat(gc, im.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * ssc = ggml_mul_mat(gc, im.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_mul_mat_set_prec(skv, GGML_PREC_F32);
            ggml_mul_mat_set_prec(ssc, GGML_PREC_F32);
            ggml_tensor * src_kv = ggml_concat(gc, L.st_kv, skv, 1);
            ggml_tensor * src_sc = ggml_concat(gc, L.st_sc, ssc, 1);
            ggml_tensor * gi = b.v1(im.i_sidx, 2 * n, (size_t) col0 * 2 * 4);
            ggml_tensor * vk = ggml_reshape_3d(gc, ggml_get_rows(gc, src_kv, gi), DH, 2, n);
            ggml_tensor * vs = ggml_reshape_3d(gc, ggml_get_rows(gc, src_sc, gi), DH, 2, n);
            vk = ggml_cont(gc, ggml_permute(gc, vk, 1, 0, 2, 3));
            vs = ggml_cont(gc, ggml_permute(gc, vs, 1, 0, 2, 3));
            latent = ggml_reshape_2d(gc, ggml_sum_rows(gc, ggml_mul(gc, vk, ggml_soft_max(gc, vs))), DH, n);
            const int64_t k = std::min<int64_t>(n, 2);
            late.push_back(ggml_set_rows(gc, L.st_kv, ggml_view_2d(gc, skv, DH, k, skv->nb[1], (size_t) (n - k) * skv->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
            late.push_back(ggml_set_rows(gc, L.st_sc, ggml_view_2d(gc, ssc, DH, k, ssc->nb[1], (size_t) (n - k) * ssc->nb[1]),
                                         b.v1(im.i_sslot, k, 0)));
        }
        latent = b.rms_w(latent, im.BL(il, "attn_compressor_norm.weight"));
        ggml_tensor * cpos = b.v1(im.i_cpos[r], n, (size_t) col0 * 4);
        ggml_tensor * cslot = b.v1(im.i_cslot[r], n, (size_t) col0 * 4);
        ggml_tensor * ik = b.rms_w(ggml_mul_mat(gc, im.BL(il, "indexer.attn_k.weight"), latent), im.BL(il, "indexer.k_norm.weight"));
        ik = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, ik, im.ID, 1, n), cpos, r), im.ID, n);
        if (im.cfg.act_quant) ik = fake_quant(gc, ggml_cont(gc, ik), Ds41Quant::FP4_POW2_32);
        outs.push_back(ggml_set_rows(gc, L.idxk, ik, cslot));
        ggml_tensor * lr = ggml_reshape_2d(gc, b.rope(ggml_reshape_3d(gc, latent, DH, 1, n), cpos, r), DH, n);
        if (im.cfg.act_quant) lr = fake_quant(gc, ggml_cont(gc, lr), Ds41Quant::FP4_E4M3_16);
        outs.push_back(ggml_set_rows(gc, L.comp, lr, cslot));
    }
    for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
    for (ggml_tensor * t : late) ggml_build_forward_expand(gf, t);
    return gf;
}

// Keep prefetch's first NU candidates consistent with the bounded selector.  The unrequested default still stacks
// scores and does one argsort below; the opt-in path stacks already-ranked ID rows, each with its own candidate pool.
static ggml_tensor * predict_route(Ds41Dense::Impl & im, ggml_context * gc, ggml_tensor * logits, int il) {
    ggml_tensor * s0 = ggml_add(gc, ggml_sqrt(gc, ggml_softplus(gc, logits)), im.BL(il, "exp_probs_b.bias"));
    ggml_tensor * rb = ggml_reshape_2d(gc, im.ly[(size_t) il].rbias, im.NE, 1);
    return rb_max_rank() > 0
        ? ds41_bounded_route_ids(gc, s0, rb, (int) im.NU, rb_max_rank(), rb_protected(), rb_score_margin(), 16)
        : ggml_add(gc, s0, rb);
}

// predict: router(rms_norm(hc_pre(state, pre)) * ffn_norm) + bias, top 16 of token 0
static ggml_cgraph * build_predict(Ds41Dense::Impl & im, ggml_context * gc, int il, ggml_tensor ** out, int64_t n = 1) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, rb_max_rank() > 0 ? 256 : 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_dsv4_hc_pre(gc, b.cols3(im.x_state, D, HC, n), b.cols2(im.pre_state, HC, n));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(x, im.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = predict_route(im, gc, rl, il);
    *out = rb_max_rank() > 0 ? sc : ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*out);
    ggml_build_forward_expand(gf, *out);
    return gf;
}

// finish: x_state = hc_post(shexp + routed, hap, post_f, comb_f).  col0: the batch column (0 on the single-pass paths).
static ggml_cgraph * build_finish(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, n, col0), b.cols2(im.routed, D, n, col0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, n, col0), b.cols2(im.post_f, HC, n, col0),
                                        b.cols3(im.comb_f, HC, HC, n, col0));
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, n, col0)));
    return gf;
}

// finish (n = 1) + the next layer's routing prediction on the state it writes (build_predict's ops on the cpy result)
static ggml_cgraph * build_finish_predict(Ds41Dense::Impl & im, ggml_context * gc, int next_il, ggml_tensor ** pout,
                                          int ahead = 0, std::vector<ggml_tensor *> * aout = nullptr) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, rb_max_rank() > 0 ? 256 * (ahead + 2) : 128, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * out = ggml_add(gc, b.cols2(im.shexp, D, 1, 0), b.cols2(im.routed, D, 1, 0));
    ggml_tensor * x = ggml_dsv4_hc_post(gc, out, b.cols3(im.hap, D, HC, 1, 0), b.cols2(im.post_f, HC, 1, 0),
                                        b.cols3(im.comb_f, HC, HC, 1, 0));
    ggml_tensor * st = ggml_cpy(gc, x, b.cols3(im.x_state, D, HC, 1, 0));
    ggml_build_forward_expand(gf, st);
    ggml_tensor * xp = ggml_dsv4_hc_pre(gc, st, b.cols2(im.pre_state, HC, 1));
    ggml_tensor * rl = ggml_mul_mat(gc, im.BLpred(next_il), b.rms_w(xp, im.BL(next_il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rl, GGML_PREC_F32);
    ggml_tensor * sc = predict_route(im, gc, rl, next_il);
    // The layers after next_il on the same state (speculative reads): stack scores (legacy) or ranked IDs (bounded),
    // then ONE readback fetches all the rows ([16, 1 + ahead] I32).
    for (int a = 1; a <= ahead && aout; ++a) {
        const int il = next_il + a;
        if (il >= im.le) break;   // past the file (unset range) or past this stage's last layer: the row is -1
        ggml_tensor * ra = ggml_mul_mat(gc, im.BLpred(il), b.rms_w(xp, im.BL(il, "ffn_norm.weight")));
        ggml_mul_mat_set_prec(ra, GGML_PREC_F32);
        ggml_tensor * sa = predict_route(im, gc, ra, il);
        sc = ggml_concat(gc, sc, sa, 1);
        aout->push_back(nullptr);   // a row of *pout
    }
    *pout = rb_max_rank() > 0 ? sc : ggml_argsort_top_k(gc, sc, 16);
    ggml_set_output(*pout);
    ggml_build_forward_expand(gf, *pout);
    return gf;
}

// head: hc_pre(state, pre_state) -> output_norm -> output, rows row0..row0+nr-1
static ggml_cgraph * build_head(Ds41Dense::Impl & im, ggml_context * gc, int64_t row0, int64_t nr, ggml_tensor ** lg) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = b.cols3(im.x_state, D, HC, nr, row0);
    ggml_tensor * p = b.cols2(im.pre_state, HC, nr, row0);
    ggml_tensor * h = b.rms_w(ggml_dsv4_hc_pre(gc, x, p), im.W("output_norm.weight"));
    *lg = ggml_mul_mat(gc, im.W("output.weight"), h);
    ggml_set_output(*lg);
    ggml_build_forward_expand(gf, *lg);
    return gf;
}

// begin: x_state = repeat(embd), pre_state = one-hot (model.py make_identity_pre_mix).  col0: the batch column the
// pass's n tokens live at (0 on the single-pass paths).
static ggml_cgraph * build_init(Ds41Dense::Impl & im, ggml_context * gc, int64_t n, int64_t col0 = 0) {
    B b { &im, gc, &im.g };
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * e = ggml_reshape_3d(gc, b.cols2(im.i_emb, D, n, col0), D, 1, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_repeat_4d(gc, e, D, HC, n, 1), b.cols3(im.x_state, D, HC, n, col0)));
    ggml_tensor * one = ggml_fill(gc, ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, n), 1.0f);
    ggml_tensor * oh = ggml_pad(gc, one, (int) (HC - 1), 0, 0, 0);                            // [1, 0, 0, 0] per token
    ggml_build_forward_expand(gf, ggml_cpy(gc, oh, b.cols2(im.pre_state, HC, n, col0)));
    return gf;
}

// ================================================================== public API

Ds41Dense::Ds41Dense() : p_(new Impl()) {}
Ds41Dense::~Ds41Dense() = default;
const Ds41Geometry & Ds41Dense::geom() const { return p_->g; }
ggml_tensor * Ds41Dense::weight(const std::string & name) const { return p_->w.get(name); }
int64_t Ds41Dense::n_layer() const { return p_->nl; }
ggml_backend_t Ds41Dense::backend() const { return p_->backend; }
const std::string & Ds41Dense::last_error() const { return p_->err; }
bool Ds41Dense::vision() const { return p_->vision; }

void Ds41Dense::set_vision(const float * rows, const int64_t * pos, int n_rows) {
    Impl & im = *p_;
    im.vis_rows = rows;
    im.vis_pos = pos;
    im.n_vis = n_rows;
}

void Ds41Dense::clear_vision() {
    Impl & im = *p_;
    im.vis_rows = nullptr;
    im.vis_pos = nullptr;
    im.n_vis = 0;
}

// The NC-column tensors (see Impl::nctx): (re)created for `NC` columns, zeroed.  Every one is per-pass scratch (a pass
// writes the columns it reads), so a resize between passes loses nothing.
static bool alloc_cols(Ds41Dense::Impl & im, int64_t NC, std::string & err) {
    if (im.nbuf) { ggml_backend_buffer_free(im.nbuf); im.nbuf = nullptr; }
    if (im.nctx) { ggml_free(im.nctx); im.nctx = nullptr; }
    im.NC = NC;
    ggml_init_params cp = { ggml_tensor_overhead() * 256, nullptr, true };
    im.nctx = ggml_init(cp);
    ggml_context * c = im.nctx;
    const int64_t D = im.D, HC = im.HC;
    im.x_state = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.pre_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.hap = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NC);
    im.post_f = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NC);
    im.comb_f = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NC);
    im.shexp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_fn = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.o_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.NU, NC);
    im.o_wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.NU, NC);
    im.routed = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.cand = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.CB, NC);
    for (auto & L : im.ly)
        if (L.idx_src && im.in_range(&L - im.ly.data())) { L.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, im.ITOP, NC); L.selm = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.ITOP, NC); }
    im.i_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NC);
    im.i_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    im.i_img = im.vision ? ggml_new_tensor_1d(c, GGML_TYPE_F32, NC) : nullptr;   // vision: the image-position mask
    im.i_widx = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA * NC);
    im.span_memo_pos0 = -1;   // new tensors: nothing uploaded yet
    im.i_wmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.SWA, NC);
    im.i_wslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, im.RAW));
    for (int r = 1; r <= 2; ++r) {
        im.i_nvis[r] = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
        im.i_cslot[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
        im.i_cpos[r] = ggml_new_tensor_1d(c, GGML_TYPE_I32, NC);
    }
    im.i_sidx = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2 * NC);
    im.i_sslot = ggml_new_tensor_1d(c, GGML_TYPE_I32, std::min<int64_t>(NC, 2));
    im.i_lastblk = ggml_new_tensor_1d(c, GGML_TYPE_F32, NC);
    // a slot no layer in the range reads has no i_eng tensor (the unset range uses both slots, as today)
    for (int e = 0; e < 2; ++e)
        if (!im.range_set || im.eng_used[e]) im.i_eng[e] = ggml_new_tensor_2d(c, GGML_TYPE_F32, im.EC * im.ED, NC);
    // STRATA_VRAM_TRACE: WHICH state tensor owns the columns' VRAM (s13c: a batch's state alloc was 2.45-2.76 GiB and
    // a 2193.81 MiB allocation later failed with 6.83 GiB "free" - name the owner before sizing anything).
    if (std::getenv("STRATA_VRAM_TRACE")) {
        double tot = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) tot += (double) ggml_nbytes(t);
        std::fprintf(stderr, "vram trace: state NC=%d wants %.2f GiB (i_eng %lld x %d = %.2f GiB each, i_wmask %.1f MiB, "
                             "i_widx %.1f MiB)\n", NC, tot / 1073741824.0, (long long) im.EC, (int) im.ED,
                     (double) (im.i_eng[0] ? ggml_nbytes(im.i_eng[0]) : 0) / 1073741824.0, (double) ggml_nbytes(im.i_wmask) / 1048576.0,
                     (double) ggml_nbytes(im.i_widx) / 1048576.0);
    }
    im.nbuf = ggml_backend_alloc_ctx_tensors(c, im.backend);
    if (!im.nbuf) { err = "cannot allocate " + std::to_string(NC) + " state columns"; return false; }
    ggml_backend_buffer_clear(im.nbuf, 0);
    return true;
}

bool Ds41Dense::set_columns(int nc) {
    Impl & im = *p_;
    const int64_t want = std::max<int64_t>(im.S, std::min<int64_t>(nc, im.NCmax));
    if (want == im.NC) return true;
    ggml_backend_synchronize(im.backend);
    // every cached graph holds views of the old columns (their addresses are fixed at the first allocation), and the
    // allocators remember graphs by address - all of it goes, and is rebuilt lazily for the new tensors
    im.graphs.clear();
    im.logits_out.clear();
    im.predict_out.clear();
    im.predict_out_n.clear();
    im.fpredict_out.clear();
    im.fpredict_ahead.clear();
    for (Allo * al : { &im.allo_dec, &im.allo_pass }) {
        if (al->a) ggml_gallocr_free(al->a);
        al->a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
        al->graph_epoch.clear();
        al->compute_bufs.clear();
        ++al->epoch;
    }
    if (im.gctx) ggml_free(im.gctx);
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    return alloc_cols(im, want, im.err);
}

int Ds41Dense::columns() const { return (int) p_->NC; }
int64_t Ds41Dense::layer_lb() const { return p_->lb; }
int64_t Ds41Dense::layer_le() const { return p_->le; }

// ================================================================== the split hand-off

// The boundary buffer, allocated by the stage that WRITES it: pinned + mapped + portable host memory and its device
// alias (the pattern at src/program/generate.cpp:7818-7833), x_state then pre_state, NCmax columns each - the same
// column count the state tensors use (max(max_pass, prompt_batch)).  The CPU backend: plain malloc, the device
// pointer is the host pointer and every copy below is a memcpy.
bool Ds41Dense::open_handoff(std::string & err) {
    Impl & im = *p_;
    if (!im.range_set || im.le == im.nl) return true;   // today / the last stage: no boundary to write
    if (im.hand_buf) return true;
    const size_t bytes = (size_t) im.NCmax * (im.HC * im.D + im.HC) * sizeof(float);   // cols * 2 * hc * n_embd
    if (im.cpu) {
        im.hand_buf = (float *) std::malloc(bytes);
        if (!im.hand_buf) { err = "cannot allocate the hand-off buffer"; return false; }
        im.hand_dev = im.hand_buf;
    }
#ifdef DS41_HAVE_CUDA
    else {
        void * dp = nullptr;
        if (cudaHostAlloc((void **) &im.hand_buf, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "cannot allocate the mapped hand-off buffer";
            return false;
        }
        im.hand_pinned = true;   // set before the alias query: the destructor must cudaFreeHost, not free
        if (cudaHostGetDevicePointer(&dp, im.hand_buf, 0) != cudaSuccess) { err = "cannot map the hand-off buffer"; return false; }
        im.hand_dev = (float *) dp;
    }
#endif
    std::memset(im.hand_buf, 0, bytes);
    return true;
}

float * Ds41Dense::handoff_dev() { return p_->hand_dev; }

void Ds41Dense::set_handoff_in(const float * dev) { p_->hand_in = const_cast<float *>(dev); }

// x_state [D, HC, NC] and pre_state [HC, NC]: a column is contiguous in each, so columns [0, n) of both are two
// runs and the buffer's x-then-pre layout is the same pair of runs.  CUDA: async copies through the mapped buffer,
// one sync; CPU: ggml_backend_tensor_get / _set are host memcpys.
bool Ds41Dense::handoff_out(int n, std::string & err) {
    Impl & im = *p_;
    if (!im.hand_buf) return true;   // no boundary open (today, or this stage owns the last block)
    if (n < 1 || n > im.NCmax) { err = "handoff_out: bad column count"; return false; }
    const size_t xb = (size_t) n * im.HC * im.D * 4, pb = (size_t) n * im.HC * 4;
    if (im.cpu) {
        ggml_backend_tensor_get(im.x_state, im.hand_buf, 0, xb);
        ggml_backend_tensor_get(im.pre_state, im.hand_buf + im.NCmax * im.HC * im.D, 0, pb);
    }
#ifdef DS41_HAVE_CUDA
    else {
        ggml_backend_tensor_get_async(im.backend, im.x_state, im.hand_buf, 0, xb);
        ggml_backend_tensor_get_async(im.backend, im.pre_state, im.hand_buf + im.NCmax * im.HC * im.D, 0, pb);
        ggml_backend_synchronize(im.backend);
    }
#endif
    return true;
}

bool Ds41Dense::handoff_in(int n, std::string & err) {
    Impl & im = *p_;
    if (im.lb == 0) return true;   // stage 0 (and today): begin() fills the columns, there is nothing to import
    if (!im.hand_in) { err = "handoff_in: set_handoff_in() was not called"; return false; }
    if (n < 1 || n > im.NCmax) { err = "handoff_in: bad column count"; return false; }
    const size_t xb = (size_t) n * im.HC * im.D * 4, pb = (size_t) n * im.HC * 4;
    if (im.cpu) {
        ggml_backend_tensor_set(im.x_state, im.hand_in, 0, xb);
        ggml_backend_tensor_set(im.pre_state, im.hand_in + im.NCmax * im.HC * im.D, 0, pb);
    }
#ifdef DS41_HAVE_CUDA
    else {
        ggml_backend_tensor_set_async(im.backend, im.x_state, im.hand_in, 0, xb);
        ggml_backend_tensor_set_async(im.backend, im.pre_state, im.hand_in + im.NCmax * im.HC * im.D, 0, pb);
        ggml_backend_synchronize(im.backend);
    }
#endif
    return true;
}

bool Ds41Dense::init(const std::string & path, const Ds41DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.cfg = cfg;
    try {
        im.model = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (!(err = ds41_read_geometry(*im.model, im.g)).empty()) return false;
    im.layers = ds41_file_layers(*im.model, im.g);
    if (!(err = ds41_check_tensors(*im.model, im.g, im.layers, cfg.external_experts)).empty()) return false;
    for (size_t i = 0; i < im.layers.size(); ++i)
        if (im.layers[i] != (int64_t) i) { err = "the file's blocks must be 0..k"; return false; }
    im.nl = (int64_t) im.layers.size();
    // the split range (opt-in): le 0 = every block (today, byte-identical); a stage runs [lb, le) and loads only
    // those blocks' weights (load_weights' filter) + the state they need
    im.lb = 0; im.le = im.nl; im.range_set = cfg.layer_le != 0;
    if (im.range_set) {
        im.lb = cfg.layer_lb; im.le = std::min<int64_t>(cfg.layer_le, im.nl);
        if (im.lb < 0 || im.lb >= im.le) { err = "bad layer range"; return false; }
    }
    // which Engram slots a layer in the range reads: alloc_cols makes an i_eng only for those
    im.eng_used[0] = im.eng_used[1] = false;
    for (int64_t l = im.lb; l < im.le; ++l)
        if (const int64_t e = im.g.engram_slot(l); e >= 0 && e < 2) im.eng_used[(size_t) e] = true;
    const Ds41Geometry & G = im.g;
    im.D = G.n_embd; im.HC = G.hc; im.DH = G.d_head; im.DHR = G.d_rope; im.NH = G.n_head; im.OG = G.o_groups;
    im.OL = G.o_lora; im.OGD = G.n_head * G.d_head / G.o_groups; im.RQ = G.r_q; im.SWA = G.n_swa;
    im.IH = G.idx_heads; im.ID = G.idx_dim; im.ITOP = G.idx_topk; im.NE = G.n_expert; im.NU = G.n_used; im.NFF = G.n_ff;
    im.CB = G.cand_blocks; im.CBS = G.cand_bsize; im.EC = (G.engram_ngram - 1) * G.engram_heads; im.ED = G.engram_dim;
    // gate-only overrides (the golden takes the same): small top-k / candidate counts make the selection logic bite
    // on a short prompt
    if (const char * e = std::getenv("DS41_TEST_TOPK")) im.ITOP = std::atoi(e);
    if (const char * e = std::getenv("DS41_TEST_CAND_BLOCKS")) im.CB = std::atoi(e);
    im.S = std::max(1, cfg.max_pass);                       // the dense sub-batch size of the layer-major walk
    im.NCmax = std::max<int64_t>(im.S, std::max(1, cfg.prompt_batch));   // state columns: the whole prompt batch, or the pass
    im.NC = cfg.elastic_cols ? im.S : im.NCmax;                           // elastic: S until a batch asks for more
    // the window ring: every row is gathered BY POSITION (slot = pos % RAW), so its size never changes results.  DSpark
    // verify passes write up to k rejected draft positions; a rejected q evicts q - RAW, which a later query q' (>= the
    // last accepted position, re-fed) needs iff q - RAW >= q' - SWA + 1  ->  safe for k <= RAW - SWA = 16 drafts.
    im.RAW = im.SWA + 16;
    im.cmax[1] = cfg.ctx_max; im.cmax[2] = cfg.ctx_max / 2;
    if (im.ITOP > im.cmax[2] || im.CBS <= 0 || cfg.ctx_max % 256) { err = "ctx_max must be a multiple of 256 and >= 1024"; return false; }

    if (cfg.backend) { im.backend = cfg.backend; }
    else {
        im.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(im.backend, cfg.n_threads);
        im.own_backend = true;
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (!load_weights(im, err)) return false;
    im.vision = cfg.vision;
    if (im.vision) {
        for (int64_t l = im.lb; l < im.le; ++l) {   // a split stage only loads its range's biases
            if (!im.w.has("blk." + std::to_string(l) + ".exp_probs_b_vl.bias")) {
                err = "--vision requires exp_probs_b_vl.bias in every layer";
                return false;
            }
        }
    }
    // The Engram tables: the HF sidecar directories (--engram), or - when the model file carries them itself as
    // Q8_0 tensors, which is what a llama.cpp release does - read straight off the shard that holds them.  A
    // partial dev file caps the readable rows with its own key; a full file has none and reads every row.
    if (cfg.engram_dirs.empty() && !G.engram_tables.empty()) {
        int64_t rows_limit = 0;
        detail::u64(im.model->meta(), "strata.v41.engram_rows_limit", rows_limit);   // absent: the tensor's own count
        if (!im.engram.open_gguf(G, *im.model, rows_limit, cfg.engram_threads, err)) return false;
    }
    if (!cfg.engram_dirs.empty() && !im.engram.open(G, cfg.engram_dirs, cfg.engram_threads, err)) return false;

    // layers
    im.ly.resize((size_t) im.nl);
    for (int64_t l = 0; l < im.nl; ++l) {
        auto & L = im.ly[(size_t) l];
        L.ratio = G.ratio[(size_t) l];
        L.kv_src = G.is_kv_source(l);
        L.idx_src = G.is_index_source(l);
        L.cand_src = l == G.cand_layer;
        L.uses_cand = G.cand_layer >= 0 && l > G.cand_layer && L.idx_src;
        L.kvl = G.kv_src(l);
        L.idxl = G.idx_src(l);
        L.eslot = G.engram_slot(l);
    }
    // persistent state
    if (cfg.kv_fp4 && (!cfg.act_quant || im.cpu || im.DH % 64 || im.ID % 32)) {
        err = "--kv-fp4 needs act_quant (the FP4 grids), the CUDA backend, d_head % 64 == 0 and idx_dim % 32 == 0";
        return false;
    }
    const ggml_type kvt = cfg.kv_type >= 0 ? (ggml_type) cfg.kv_type : cfg.act_quant ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    ggml_init_params sp = { ggml_tensor_overhead() * 512, nullptr, true };
    im.sctx = ggml_init(sp);
    auto T = [&](ggml_type t, std::initializer_list<int64_t> ne) {
        const int64_t * d = ne.begin();
        switch (ne.size()) {
            case 1: return ggml_new_tensor_1d(im.sctx, t, d[0]);
            case 2: return ggml_new_tensor_2d(im.sctx, t, d[0], d[1]);
            default: return ggml_new_tensor_3d(im.sctx, t, d[0], d[1], d[2]);
        }
    };
    const int64_t D = im.D, NCM = im.NCmax;
    for (int64_t l = im.lb; l < im.le; ++l) {   // only the range's layers get rings; sctx (hence sbuf) is the range's bytes
        auto & L = im.ly[(size_t) l];
        L.raw = T(kvt, { im.DH, im.RAW });
        L.raw_snap = T(kvt, { im.DH, im.RAW });
        L.rbias = T(GGML_TYPE_F32, { im.NE });   // zeroed with the buffer
        if (L.kv_src) {
            // STRATA_KV_FP4_ONLY=comp|idx (bisect tap): pack only one of the two caches
            static const char * only = std::getenv("STRATA_KV_FP4_ONLY");
            const bool fc = cfg.kv_fp4 && !(only && std::string(only) == "idx");
            const bool fi = cfg.kv_fp4 && !(only && std::string(only) == "comp");
            L.comp = T(fc ? GGML_TYPE_NVFP4 : kvt, { im.DH, im.cmax[L.ratio] + 1 });
            L.idxk = T(fi ? GGML_TYPE_MXFP4 : kvt, { im.ID, im.cmax[L.ratio] + 1 });
            if (L.ratio == 2) {
                L.st_kv = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_kv_snap = T(GGML_TYPE_F32, { im.DH, 2 });
                L.st_sc_snap = T(GGML_TYPE_F32, { im.DH, 2 });
            }
        }
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors(im.sctx, im.backend);
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);
    // the ratio-2 state scores start at -1e30 (an unwritten state gets softmax weight 0); sel = 0 (a valid row)
    for (auto & L : im.ly)
        if (L.st_sc) { std::vector<float> v((size_t) (im.DH * 2), -1.0e30f); ggml_backend_tensor_set(L.st_sc, v.data(), 0, v.size() * 4); }
    // inputs
    ggml_init_params ipp = { ggml_tensor_overhead() * 64, nullptr, true };
    im.ictx = ggml_init(ipp);
    auto I = [&](ggml_type t, int64_t a, int64_t b2 = 1) { return b2 == 1 ? ggml_new_tensor_1d(im.ictx, t, a) : ggml_new_tensor_2d(im.ictx, t, a, b2); };
    im.neg_ramp = I(GGML_TYPE_F32, im.cmax[1] + 1);
    im.blk_ramp = I(GGML_TYPE_F32, im.cmax[1] / im.CBS + 2);
    im.ibuf = ggml_backend_alloc_ctx_tensors(im.ictx, im.backend);
    if (!im.ibuf) { err = "cannot allocate the inputs"; return false; }
    ggml_backend_buffer_clear(im.ibuf, 0);
    {
        std::vector<float> v((size_t) im.neg_ramp->ne[0]);
        for (size_t i = 0; i < v.size(); ++i) v[i] = -(float) i;
        ggml_backend_tensor_set(im.neg_ramp, v.data(), 0, v.size() * 4);
        v.assign((size_t) im.blk_ramp->ne[0], 0);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (float) i;
        ggml_backend_tensor_set(im.blk_ramp, v.data(), 0, v.size() * 4);
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(8192, false) * 400, nullptr, true };
    im.gctx = ggml_init(gp);
    im.allo_dec.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    if (!alloc_cols(im, im.NC, err)) return false;
    // the host mirrors are sized for the largest batch once (pinned registrations must not move)
    im.host_fn.resize((size_t) (D * NCM));
#ifdef DS41_HAVE_CUDA
    // pinned: the fn rows go device -> host -> the tier's device input every layer (pageable copies ran at ~2 GB/s)
    if (!im.cpu && !im.host_fn.empty()) ggml_backend_cuda_register_host_buffer(im.host_fn.data(), im.host_fn.size() * 4);
#endif
    im.host_ids.resize((size_t) (im.NU * NCM));
    im.host_wts.resize((size_t) (im.NU * NCM));
#ifdef DS41_HAVE_CUDA
    // pinned too: attn_router reads fn / ids / wts back with three async copies and one sync (truly async only from
    // pinned memory)
    if (!im.cpu) {
        ggml_backend_cuda_register_host_buffer(im.host_ids.data(), im.host_ids.size() * 4);
        ggml_backend_cuda_register_host_buffer(im.host_wts.data(), im.host_wts.size() * 4);
    }
#endif
    return true;
}

void Ds41Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    ggml_backend_tensor_set(im.ly[(size_t) il].rbias, bias, 0, (size_t) im.NE * 4);
}

void Ds41Dense::set_oom_hook(std::function<bool()> hook) { p_->oom_hook = std::move(hook); }

void Ds41Dense::release_pass() {
    Impl & im = *p_;
    if (!im.allo_pass.a) return;
    ggml_gallocr_free(im.allo_pass.a);
    im.allo_pass.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    im.allo_pass.graph_epoch.clear();
    im.allo_pass.compute_bufs.clear();
    ++im.allo_pass.epoch;
}

// pass_cap (per ratio) of a pass of n tokens ending at pos0+n: the visible compressed rows it ranks over.
static void span_pass_caps(const Ds41Dense::Impl & im, int pos0, int n, int64_t out[3]) {
    out[0] = 0;
    for (int r = 1; r <= 2; ++r) {
        const int64_t nvis_last = (pos0 + n) / r;
        out[r] = nvis_last == 0 ? 0 : std::min<int64_t>(next_pow2(std::max<int64_t>(nvis_last, im.CBS)), im.cmax[r]);
    }
}

// Wait for the pass's Engram reads (started by upload_inputs) and push the rows into the state tensor's columns.  A
// no-op once done (eng_batch is null), so it is safe to call at every layer - but the caller gates it on L.eslot >= 0
// so the layers BELOW the first Engram layer run while the disk is still reading.  The whole pass's rows (both Engram
// slots, all n columns) go up in one shot, so the second Engram layer (14) is a plain read of the same tensor.
static bool engram_ready(Ds41Dense::Impl & im) {
    if (!im.eng_batch) return true;
    if (!im.engram.wait(im.eng_batch, im.err)) { im.eng_batch.reset(); return false; }
    im.eng_batch.reset();
    for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
        if (im.i_eng[e]) ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);   // a slot outside the range has no tensor
    return true;
}

// The pass's position-derived maps: i_pos, the window rows / mask / ring slots, the compressed-row maps per ratio
// and the ratio-2 state rows / slots.  upload_inputs uploads these after the embeddings; begin_from_handoff uploads
// exactly these and nothing else (a mid-model stage has no embeddings to upload).  Same order, same tensors.
bool Ds41Dense::Impl::upload_pass_maps(int n, int pos0) {
    Impl & im = *this;
    const int64_t SWA = im.SWA, RAW = im.RAW;
    std::vector<int32_t> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    ggml_backend_tensor_set(im.i_pos, pos.data(), 0, (size_t) n * 4);
    // window: per query the SWA positions [p - SWA + 1, p] as rows of [ring (RAW) | pass rows]
    std::vector<int32_t> widx((size_t) (SWA * n));
    std::vector<float> wm((size_t) (SWA * n));
    for (int t = 0; t < n; ++t) {
        const int64_t p = pos0 + t;
        for (int64_t j = 0; j < SWA; ++j) {
            const int64_t q = p - SWA + 1 + j;
            int32_t row = 0;
            float m = 0.0f;
            if (q < 0) m = NEG_INF;
            else if (q >= pos0) row = (int32_t) (RAW + (q - pos0));
            else row = (int32_t) (q % RAW);
            widx[(size_t) (t * SWA + j)] = row;
            wm[(size_t) (t * SWA + j)] = m;
        }
    }
    ggml_backend_tensor_set(im.i_widx, widx.data(), 0, widx.size() * 4);
    ggml_backend_tensor_set(im.i_wmask, wm.data(), 0, wm.size() * 4);
    {
        const int64_t k = std::min<int64_t>(n, RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pos0 + n - k + j) % RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    // compressed rows per ratio
    for (int r = 1; r <= 2; ++r) {
        std::vector<float> nv(n);
        std::vector<int32_t> cs(n), cp(n);
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t;
            nv[t] = (float) ((p + 1) / r);
            const bool done = (p + 1) % r == 0;
            cs[t] = (int32_t) (done ? p / r : im.cmax[r]);   // an open group writes the spare row
            cp[t] = (int32_t) ((p / r) * r);
        }
        ggml_backend_tensor_set(im.i_nvis[r], nv.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cslot[r], cs.data(), 0, (size_t) n * 4);
        ggml_backend_tensor_set(im.i_cpos[r], cp.data(), 0, (size_t) n * 4);
        if (r == 1) {
            std::vector<float> lb(n);
            for (int t = 0; t < n; ++t) lb[t] = (float) ((nv[t] - 1) >= 0 ? (int64_t) (nv[t] - 1) / im.CBS : -1);
            ggml_backend_tensor_set(im.i_lastblk, lb.data(), 0, (size_t) n * 4);
        }
    }
    {   // ratio-2 state: rows of [ring (2) | pass states] for each token's group (2g, 2g+1); open groups repeat 2g
        std::vector<int32_t> si((size_t) (2 * n)), ss;
        auto row = [&](int64_t q) { return (int32_t) (q >= pos0 ? 2 + (q - pos0) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pos0 + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        const int64_t k = std::min<int64_t>(n, 2);
        for (int64_t j = 0; j < k; ++j) ss.push_back((int32_t) ((pos0 + n - k + j) % 2));
        ggml_backend_tensor_set(im.i_sidx, si.data(), 0, si.size() * 4);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

// Per-token inputs of n tokens at positions pos0.. (embeddings, positions, the window / compressed / indexer index
// maps, the Engram rows and the per-pass ring slots).  Shared by begin() and begin_batch().
bool Ds41Dense::Impl::upload_inputs(const int * tids, int n, int pos0) {
    Impl & im = *this;
    im.span_memo_pos0 = -1;   // this writes i_widx / i_wslot / i_sidx / i_sslot: upload_span's memo is stale
    const int64_t D = im.D, SWA = im.SWA, RAW = im.RAW;
    // Decode only the requested mmap rows directly into the reusable pass buffer. Quantized embeddings
    // use their stored ggml byte stride; no whole-table conversion or per-row scratch allocation.
    im.host_emb.resize((size_t) (D * n));
    if (!im.embd) { im.err = "token_embd.weight is not loaded for this layer range"; return false; }   // a stage past 0 (the loader rule)
    if (!ds41_gather_embedding_rows(im.embd, im.embd_row_bytes, im.embd_to_float, D, im.g.vocab,
                                    tids, n, im.host_emb.data())) {
        im.err = "token id out of range";
        return false;
    }
    // vision: overwrite the image span positions with the encoder's rows (model.py merge_image_embeddings) and mark
    // them, so the router picks exp_probs_b_vl there and Engram skips them.  vis_pos is ascending, the pass contiguous.
    std::vector<float> img;
    if (im.vision) {
        img.assign((size_t) n, 0.0f);
        for (int v = 0; v < im.n_vis; ++v) {
            const int64_t p = im.vis_pos[v];
            if (p < pos0 || p >= (int64_t) pos0 + n) continue;
            const int t = (int) (p - pos0);
            std::memcpy(im.host_emb.data() + (size_t) t * D, im.vis_rows + (size_t) v * D, (size_t) D * 4);
            img[(size_t) t] = 1.0f;
        }
        ggml_backend_tensor_set(im.i_img, img.data(), 0, (size_t) n * 4);
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (!upload_pass_maps(n, pos0)) return false;
    // Engram rows: hash the whole pass now and START the (dedup/sort/coalesce) reads; they are waited - and the rows
    // moved to the state tensor - by the first Engram layer's span (engram_ready), so the disk read overlaps the
    // embeddings upload, the init graph and the layers below it.  A tiny batch (decode) still runs on the caller.
    if (im.engram.enabled()) {
        if (!engram_ready(im)) return false;    // flush a still-pending batch from a previous pass first
        for (int t = 0; t < n; ++t) {
            if (!img.empty() && img[(size_t) t] > 0.0f) im.engram.set_dead(pos0 + t);   // an image span: n-grams never span it
            else im.engram.set_token(pos0 + t, tids[t]);
        }
        if (n <= 64) {   // decode / verify: the rows are needed at layer 1 - read them now (synchronously)
            static const bool eprof = std::getenv("STRATA_ENG_PROF") != nullptr;
            static double eng_ms = 0; static long eng_n = 0;
            const auto te = std::chrono::steady_clock::now();
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err)) return false;
            if (eprof) {
                eng_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
                if (++eng_n % 64 == 0) std::fprintf(stderr, "engram decode reads: %.3f ms/call over %ld calls\n", eng_ms / eng_n, eng_n);
            }
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                if (im.i_eng[e]) ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);   // a slot outside the range has no tensor
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

bool Ds41Dense::begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    im.pos0 = pos0;
    im.n = n;
    if (!im.upload_inputs(tids, n, pos0)) return false;
    span_pass_caps(im, pos0, n, im.pass_cap);
    // embeddings -> state
    ggml_context * gc = nullptr;
    ggml_cgraph * gf = nullptr;
    const bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 0, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 0, 0, 1 }, build_init(im, im.gctx, 1)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_init(im, gc, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    return ok;
}

// begin() for a stage that starts a pass mid-model (the split): the pass's columns/positions + this stage's own
// Engram rows, no embeddings and no stage-0 work.  The state columns arrive separately via handoff_in(n); the hash
// needs the ngram-1 tokens before pos0, which tids carries ahead of the pass's own n.
bool Ds41Dense::begin_from_handoff(const int * tids, int ntids, int n, int pos0) {
    Impl & im = *p_;
    if (!im.range_set) { im.err = "begin_from_handoff needs a layer range"; return false; }
    if (n < 1 || n > im.NC || pos0 < 0 || pos0 + n > im.cfg.ctx_max) { im.err = "bad pass"; return false; }
    const int64_t N = im.g.engram_ngram;
    const int nh = (int) std::min<int64_t>(N - 1, pos0);   // the hash's look-back stops at position 0
    if (ntids < n + nh) { im.err = "begin_from_handoff: tids must cover pos0-3..pos0+n-1"; return false; }
    im.pos0 = pos0;
    im.n = n;
    if (!im.upload_pass_maps(n, pos0)) return false;
    span_pass_caps(im, pos0, n, im.pass_cap);
    // this stage's Engram rows: the reads are issued only when a layer in [lb, le) is an Engram layer (a stage
    // between them has no i_eng tensor and never waits).  Same sync/async split as upload_inputs.
    if (im.engram.enabled() && (im.eng_used[0] || im.eng_used[1])) {
        if (!engram_ready(im)) return false;   // flush a still-pending batch from a previous pass first
        for (int t = 0; t < ntids; ++t) {
            const int64_t p = (int64_t) pos0 - nh + t;
            if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
            im.engram.set_token(p, tids[t]);
        }
        if (n <= 64) {   // decode / verify: read now (synchronously)
            if (!im.engram.rows(pos0, n, im.eng_rows, im.err)) return false;
            for (size_t e = 0; e < im.eng_rows.size() && e < 2; ++e)
                if (im.i_eng[e]) ggml_backend_tensor_set(im.i_eng[e], im.eng_rows[e].data(), 0, im.eng_rows[e].size() * 4);
        } else {
            im.eng_batch = im.engram.start(pos0, n, im.eng_rows, im.err);
            if (!im.eng_batch) return false;
        }
    }
    return true;
}

// ================================================================== the layer-major prompt batch

bool Ds41Dense::begin_batch(const int * tids, int B, int pos0) {
    Impl & im = *p_;
    // the layer-major batch is a stage-0 path: it uploads embeddings and runs build_init (x_state = embd, pre_state =
    // one-hot), which a mid-model stage neither has (no token_embd) nor wants (it would clobber the imported hand-off).
    // The split's per-pass flow is begin() / begin_from_handoff(); a stage past 0 starts a pass with the latter.
    if (im.lb > 0) { im.err = "begin_batch: a mid-model stage starts a pass with begin_from_handoff"; return false; }
    if (B < 1 || B > im.cfg.prompt_batch || B > im.NC || pos0 < 0 || pos0 + B > im.cfg.ctx_max) { im.err = "bad batch"; return false; }
    if (!im.upload_inputs(tids, B, pos0)) return false;
    im.pos0 = pos0; im.batch_pos0 = pos0; im.batch = B; im.n = B; im.col0 = 0;
    // per-sub-batch pass_cap (the compressed rows each sub-batch ranks over) - depends only on (pos0, len)
    const int64_t nb = (B + im.S - 1) / im.S;
    im.span_cap.assign((size_t) nb, { 0, 0, 0 });
    for (int64_t i = 0; i < nb; ++i)
        span_pass_caps(im, pos0 + (int) (i * im.S), (int) std::min<int64_t>(im.S, (int64_t) B - i * im.S),
                       im.span_cap[(size_t) i].data());
    // the batch's initial state, one sub-batch at a time (x_state = embd, pre_state = one-hot)
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(64, false));
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t c0 = i * im.S, sn = std::min<int64_t>(im.S, (int64_t) B - c0);
        ggml_init_params p = { meta.size(), meta.data(), true };
        ggml_context * gc = ggml_init(p);
        ggml_cgraph * gf = build_init(im, gc, sn, c0);
        const bool ok = im.compute(gf, false);
        ggml_free(gc);
        if (!ok) { im.err = "batch init failed"; return false; }
    }
    return true;
}

// The per-span inputs of one layer-major sub-batch: the sub-batch's own pass_cap (the graphs read it) and the
// PASS-LOCAL row maps (the pass's own rows / states sit at the end of the [ring | pass rows] they index, so they are
// recomputed from the sub-batch's own position - upload_inputs laid them out for the whole batch, correct only at
// col0 = 0).  Shared by attn_router_span and attn_router_span_kv.
static bool upload_span(Ds41Dense::Impl & im, int col0, int n) {
    if (col0 < 0 || n < 1 || im.S < 1 || (int64_t) col0 + n > im.batch
        || col0 / im.S != (col0 + n - 1) / im.S) { im.err = "bad span"; return false; }
    const size_t si = (size_t) (col0 / im.S);
    if (si >= im.span_cap.size()) { im.err = "bad span"; return false; }
    // A pre-tail checkpoint splits the final sub-batch into two spans. Each must use its OWN
    // visible compressed-row bounds and pass-local maps, not the unsplit sub-batch's bounds.
    if (col0 % im.S || n != std::min<int64_t>(im.S, im.batch - col0))
        span_pass_caps(im, im.batch_pos0 + col0, n, im.pass_cap);
    else {
        im.pass_cap[0] = im.span_cap[si][0];
        im.pass_cap[1] = im.span_cap[si][1];
        im.pass_cap[2] = im.span_cap[si][2];
    }
    const int64_t pbase = (int64_t) im.batch_pos0 + col0;
    static const bool memo_on = !(std::getenv("STRATA_SPAN_MEMO") && std::getenv("STRATA_SPAN_MEMO")[0] == '0');
    if (memo_on && im.span_memo_pos0 == (int64_t) im.batch_pos0 && im.span_memo_col0 == col0 && im.span_memo_n == n) return true;
    im.span_memo_pos0 = (int64_t) im.batch_pos0;
    im.span_memo_col0 = col0;
    im.span_memo_n = n;
    {
        std::vector<int32_t> widx((size_t) (im.SWA * n));
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t;
            for (int64_t j = 0; j < im.SWA; ++j) {
                const int64_t q = p - im.SWA + 1 + j;
                widx[(size_t) (t * im.SWA + j)] = q < 0 ? 0 : (int32_t) (q >= pbase ? im.RAW + (q - pbase) : q % im.RAW);
            }
        }
        ggml_backend_tensor_set(im.i_widx, widx.data(), (size_t) col0 * im.SWA * 4, widx.size() * 4);
    }
    {
        std::vector<int32_t> si((size_t) (2 * n));
        auto row = [&](int64_t q) { return (int32_t) (q >= pbase ? 2 + (q - pbase) : q % 2); };
        for (int t = 0; t < n; ++t) {
            const int64_t p = pbase + t, g0 = (p / 2) * 2;
            si[(size_t) (2 * t)] = row(g0);
            si[(size_t) (2 * t + 1)] = row(p >= g0 + 1 ? g0 + 1 : g0);
        }
        ggml_backend_tensor_set(im.i_sidx, si.data(), (size_t) col0 * 2 * 4, si.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, im.RAW);
        std::vector<int32_t> ws((size_t) k);
        for (int64_t j = 0; j < k; ++j) ws[(size_t) j] = (int32_t) ((pbase + n - k + j) % im.RAW);
        ggml_backend_tensor_set(im.i_wslot, ws.data(), 0, ws.size() * 4);
    }
    {
        const int64_t k = std::min<int64_t>(n, 2);
        std::vector<int32_t> ss((size_t) k);
        for (int64_t j = 0; j < k; ++j) ss[(size_t) j] = (int32_t) ((pbase + n - k + j) % 2);
        ggml_backend_tensor_set(im.i_sslot, ss.data(), 0, ss.size() * 4);
    }
    return true;
}

bool Ds41Dense::attn_router_span_kv(int il, int col0, int n) {
    Impl & im = *p_;
    if (!im.in_range(il)) { im.err = "attn_router_span_kv: layer outside the range"; return false; }   // absolute id
    if (!upload_span(im, col0, n)) return false;
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // the Engram add feeds this layer's K/V
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params pk = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gck = ggml_init(pk);
    ggml_cgraph * gk = build_attn_kv_only(im, gck, il, n, col0);
    const bool okk = im.compute(gk, false);
    ggml_free(gck);
    return okk;
}

bool Ds41Dense::attn_router_span(int il, int col0, int n) {
    Impl & im = *p_;
    double tp = g_dprof_on ? dprof_now() : 0;
    if (!im.in_range(il)) { im.err = "attn_router_span: layer outside the range"; return false; }   // absolute id
    auto lap = [&](int i) { if (g_dprof_on) { const double t = dprof_now(); g_dprof[i] += t - tp; tp = t; } };
    if (!upload_span(im, col0, n)) return false;
    if (g_dprof_on && dprof_now() - tp > 50) std::fprintf(stderr, "dense prof: slow upload_span layer %d col0 %d n %d: %.0f ms\n", il, col0, n, dprof_now() - tp);
    lap(6);
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;   // wait the pass's Engram reads here
    lap(3);
    im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_attn(im, gc, il, n, col0);
    lap(0);
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    if (g_dprof_on) { tp = dprof_now(); g_dprof[7] += 1; }
    ggml_free(gc);
    if (!ok) return false;
    // the sub-batch's ids / wts / fn land at its own batch columns (token-major: token col0+t at t * NU / t * D)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data() + (size_t) col0 * im.D, (size_t) col0 * im.D * 4, (size_t) n * im.D * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data() + (size_t) col0 * im.NU, (size_t) col0 * im.NU * 4, (size_t) n * im.NU * 4);
    ggml_backend_synchronize(im.backend);
    lap(4);
    return true;
}

bool Ds41Dense::upload_routed(const float * routed_host, int B) {
    Impl & im = *p_;
    if (!routed_host || B < 1 || B > im.batch) { im.err = "bad routed batch"; return false; }
    ggml_backend_tensor_set(im.routed, routed_host, 0, (size_t) B * im.D * 4);
    return true;
}

float * Ds41Dense::routed_dev() { return p_->cpu || !p_->routed ? nullptr : (float *) p_->routed->data; }

bool Ds41Dense::finish_span(int il, int col0, int n) {
    Impl & im = *p_;
    (void) il;
    if (col0 < 0 || n < 1 || im.S < 1 || col0 % im.S != 0 || (int64_t) col0 + n > im.batch) { im.err = "bad span"; return false; }
    const double tf0 = g_dprof_on ? dprof_now() : 0;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_cgraph * gf = build_finish(im, gc, n, col0);
    if (g_dprof_on) g_dprof[5] += dprof_now() - tf0;
    g_dprof_slot = 0;
    const bool ok = im.compute(gf, false);
    g_dprof_slot = -1;
    ggml_free(gc);
    return ok;
}

void Ds41Dense::prof_report(const char * tag) {
    if (!g_dprof_on) return;
    std::fprintf(stderr, "dense prof %s: %d attn spans: build %.0f ms, alloc %.0f, gpu compute %.0f, engram wait %.0f, "
                 "readback %.0f, upload %.0f, finish build %.0f\n", tag, (int) g_dprof[7], g_dprof[0], g_dprof[1], g_dprof[2],
                 g_dprof[3], g_dprof[4], g_dprof[6], g_dprof[5]);
    for (double & v : g_dprof) v = 0;
}

int Ds41Dense::batch_size() const { return (int) p_->batch; }
const int32_t * Ds41Dense::batch_ids() const { return p_->host_ids.data(); }
const float * Ds41Dense::batch_wts() const { return p_->host_wts.data(); }
const float * Ds41Dense::batch_fn() const { return p_->host_fn.data(); }

bool Ds41Dense::attn_router(int il, int * ids, float * wts, const float ** fn_host, ggml_tensor ** fn_dev) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (!im.in_range(il)) { im.err = "attn_router: layer outside the range"; return false; }   // absolute id; the unset range is every layer
    // Engram layer: wait the pass's reads (started in upload_inputs) before the graph that reads i_eng runs.  For the
    // chunk path this is the first Engram layer too, so the layers below it ran while the disk was reading.
    if (im.ly[(size_t) il].eslot >= 0 && !engram_ready(im)) return false;
    const int64_t r = im.ly[(size_t) il].ratio;
    const int64_t cap = r ? im.pass_cap[r] : 0;
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1;
    if (cached) {
        const Impl::GKey key { il, 1, cap, 1 };
        auto it = im.graphs.find(key);
        if (it == im.graphs.end()) it = im.graphs.emplace(key, build_attn(im, im.gctx, il, 1)).first;
        gf = it->second;
    } else {
        im.pf_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
        ggml_init_params p = { im.pf_meta.size(), im.pf_meta.data(), true };
        gc = ggml_init(p);
        gf = build_attn(im, gc, il, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    if (!ok) return false;
    // one sync for the three readbacks (was one per tensor_get: 3 x 40 stream syncs per decode token)
    ggml_backend_tensor_get_async(im.backend, im.o_fn, im.host_fn.data(), 0, (size_t) (im.D * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_ids, im.host_ids.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_tensor_get_async(im.backend, im.o_wts, im.host_wts.data(), 0, (size_t) (im.NU * n) * 4);
    ggml_backend_synchronize(im.backend);
    std::memcpy(ids, im.host_ids.data(), (size_t) (im.NU * n) * 4);
    std::memcpy(wts, im.host_wts.data(), (size_t) (im.NU * n) * 4);
    *fn_host = im.host_fn.data();
    if (fn_dev) *fn_dev = im.o_fn;
    return true;
}

bool Ds41Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (im.n != 1 || n_top < 1 || n_top > 16) return false;
    if (!im.in_range(il)) { im.err = "predict: layer outside the range"; return false; }   // absolute id; the unset range is every layer
    const Impl::GKey key { il, 3, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out[il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.predict_out[il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::predict_multi(int il, int * top_ids) {
    Impl & im = *p_;
    const int64_t n = im.n;
    if (n < 1) return false;
    if (!im.in_range(il)) { im.err = "predict_multi: layer outside the range"; return false; }   // absolute id; the unset range is every layer
    if (n == 1) return predict(il, top_ids, 16);
    const Impl::GKey key { il, 3, 0, (int) n };   // one graph per (layer, pass width): verify passes reuse a few widths
    auto it = im.graphs.find(key);
    ggml_tensor * o = nullptr;
    if (it == im.graphs.end()) {
        ggml_cgraph * gf = build_predict(im, im.gctx, il, &o, n);
        it = im.graphs.emplace(key, gf).first;
        im.predict_out_n[{ il, (int) n }] = o;
    }
    o = im.predict_out_n[{ il, (int) n }];
    if (!im.compute(it->second, true)) return false;
    ggml_backend_tensor_get(o, top_ids, 0, sizeof(int32_t) * 16 * (size_t) n);
    return true;
}

bool Ds41Dense::finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    if (!im.in_range(il)) { im.err = "finish: layer outside the range"; return false; }   // absolute id; the unset range is every layer
    const int64_t n = im.n;
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) (im.D * n) * 4);
    ggml_cgraph * gf = nullptr;
    ggml_context * gc = nullptr;
    const bool cached = n == 1;
    std::vector<uint8_t> meta;
    if (cached) {
        auto it = im.graphs.find({ -1, 2, 0, 1 });
        if (it == im.graphs.end()) it = im.graphs.emplace(Impl::GKey { -1, 2, 0, 1 }, build_finish(im, im.gctx, 1)).first;
        gf = it->second;
    } else {
        meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        ggml_init_params p = { meta.size(), meta.data(), true };
        gc = ggml_init(p);
        gf = build_finish(im, gc, n);
    }
    const bool ok = im.compute(gf, cached);
    if (gc) ggml_free(gc);
    return ok;
}

bool Ds41Dense::finish_predict(int il, const float * routed_sum, int next_il, int * top_ids, int n_top) {
    Impl & im = *p_;
    (void) il;
    if (im.n != 1 || n_top < 1 || n_top > 16 || !im.in_range(next_il)) { im.err = "finish_predict: next layer outside the range"; return false; }   // the fused router lives on this stage
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 5, 0, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o);
        it = im.graphs.emplace(key, gf).first;
        im.fpredict_out[next_il] = o;
    }
    if (!im.compute(it->second, true)) return false;
    int32_t buf[16];
    ggml_backend_tensor_get(im.fpredict_out[next_il], buf, 0, sizeof(buf));
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[i];
    return true;
}

bool Ds41Dense::finish_predict_ahead(int il, const float * routed_sum, int next_il, int * top_ids, int n_top, int ahead,
                                     int * ahead_ids) {
    Impl & im = *p_;
    (void) il;
    if (ahead <= 0 || !ahead_ids) return finish_predict(il, routed_sum, next_il, top_ids, n_top);
    if (im.n != 1 || n_top < 1 || n_top > 16 || !im.in_range(next_il)) { im.err = "finish_predict_ahead: next layer outside the range"; return false; }   // the fused router lives on this stage
    if (routed_sum) ggml_backend_tensor_set(im.routed, routed_sum, 0, (size_t) im.D * 4);
    const Impl::GKey key { next_il, 6, ahead, 1 };
    auto it = im.graphs.find(key);
    if (it == im.graphs.end()) {
        ggml_tensor * o = nullptr;
        std::vector<ggml_tensor *> ao;
        ggml_cgraph * gf = build_finish_predict(im, im.gctx, next_il, &o, ahead, &ao);
        it = im.graphs.emplace(key, gf).first;
        ao.insert(ao.begin(), o);   // [0] = next_il's prediction, [1 + a] = next_il + 1 + a's
        im.fpredict_ahead[{ next_il, ahead }] = ao;
    }
    if (!im.compute(it->second, true)) return false;
    const auto & ao = im.fpredict_ahead[{ next_il, ahead }];   // [0] = the output, then one entry per extra row
    // argsort_top_k is a VIEW of the full argsort: row r starts at r * nb[1] (NE ids apart), not r * 16 - one copy of
    // the span covering every row's first 16
    const int rows = (int) ao.size();
    const size_t rs = ao[0]->nb[1] / 4;
    std::vector<int32_t> buf(rs * (size_t) (rows - 1) + 16);
    ggml_backend_tensor_get(ao[0], buf.data(), 0, buf.size() * 4);
    for (int i = 0; i < n_top; ++i) top_ids[i] = buf[(size_t) i];
    for (int a = 0; a < ahead; ++a) {
        if (a + 1 < rows) std::copy(buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1)), buf.begin() + (ptrdiff_t) (rs * (size_t) (a + 1) + 16),
                                    ahead_ids + (size_t) a * 16);
        else std::fill(ahead_ids + (size_t) a * 16, ahead_ids + (size_t) (a + 1) * 16, -1);
    }
    return true;
}

bool Ds41Dense::logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (row0 < 0 || nrows < 1 || row0 + nrows > im.n) { im.err = "bad logits rows"; return false; }
    // only the last stage loads the head (output.weight + output_norm.weight); today it is always loaded
    if (!im.w.has("output.weight") || !im.w.has("output_norm.weight")) { im.err = "logits: this stage does not hold the head"; return false; }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params p = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(p);
    ggml_tensor * lg = nullptr;
    ggml_cgraph * gf = build_head(im, gc, row0, nrows, &lg);
    const bool ok = im.compute(gf, false);
    if (ok) ggml_backend_tensor_get(lg, out, 0, (size_t) nrows * (size_t) im.g.vocab * 4);
    ggml_free(gc);
    return ok;
}

bool Ds41Dense::read_state(std::vector<float> & x, std::vector<float> & pre) {
    Impl & im = *p_;
    x.resize((size_t) (im.D * im.HC * im.n));
    pre.resize((size_t) (im.HC * im.n));
    ggml_backend_tensor_get(im.x_state, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_get(im.pre_state, pre.data(), 0, pre.size() * 4);
    return true;
}

bool Ds41Dense::read_state_cols(int c0, int n, std::vector<float> & x) {
    Impl & im = *p_;
    if (c0 < 0 || n < 1 || c0 + n > im.NC) { im.err = "read_state_cols: bad range"; return false; }
    const size_t col = (size_t) (im.D * im.HC) * sizeof(float);   // x_state [D, HC, NC]: a column is contiguous
    x.resize((size_t) (im.D * im.HC * n));
    ggml_backend_tensor_get(im.x_state, x.data(), (size_t) c0 * col, (size_t) n * col);
    return true;
}

bool Ds41Dense::snapshot(int pos) {
    Impl & im = *p_;
    for (auto & L : im.ly) {
        if (L.raw) ggml_backend_tensor_copy(L.raw, L.raw_snap);   // a layer outside this stage's range has no rings
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv, L.st_kv_snap); ggml_backend_tensor_copy(L.st_sc, L.st_sc_snap); }
    }
    ggml_backend_synchronize(im.backend);
    im.snap_pos = pos;
    return true;
}

bool Ds41Dense::restore() {
    Impl & im = *p_;
    if (im.snap_pos < 0) return false;
    for (auto & L : im.ly) {
        if (L.raw) ggml_backend_tensor_copy(L.raw_snap, L.raw);   // as in snapshot(): out-of-range layers are null here
        if (L.st_kv) { ggml_backend_tensor_copy(L.st_kv_snap, L.st_kv); ggml_backend_tensor_copy(L.st_sc_snap, L.st_sc); }
    }
    ggml_backend_synchronize(im.backend);
    return true;
}

int Ds41Dense::snapshot_pos() const { return p_->snap_pos; }

bool Ds41Dense::snapshot_prefix(int pos, int il) {
    Impl & im = *p_;
    im.prefix_pos = -1;
    if (pos < 0) return true;
    if (il >= im.nl || (il >= 0 && !im.in_range(il))) return false;
    im.prefix.resize((size_t) im.nl * 3);
    ggml_backend_synchronize(im.backend);
    const int lb = il < 0 ? (int) im.lb : il, le = il < 0 ? (int) im.le : il + 1;
    for (int l = lb; l < le; ++l) {
        auto & L = im.ly[(size_t) l];
        int k = 0;
        for (ggml_tensor * t : { L.raw, L.st_kv, L.st_sc }) {
            auto & r = im.prefix[(size_t) l * 3 + k++];
            r.t = t;
            r.off = 0;
            if (!t) continue;
            r.b.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, r.b.data(), 0, r.b.size());
        }
    }
    im.prefix_pos = pos;
    return true;
}

bool Ds41Dense::restore_prefix() {
    Impl & im = *p_;
    if (im.prefix_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    for (auto & r : im.prefix)
        if (r.t) ggml_backend_tensor_set(r.t, r.b.data(), 0, r.b.size());
    ggml_backend_synchronize(im.backend);
    return true;
}

int Ds41Dense::prefix_snapshot_pos() const { return p_->prefix_pos; }

void Ds41Dense::restore_token_history(const int * tids, int n) {
    Impl & im = *p_;
    if (!im.engram.enabled()) return;
    for (int p = std::max(0, n - (int) im.g.engram_ngram + 1); p < n; ++p)
        im.engram.set_token(p, tids[p]);
}

bool Ds41Dense::park(int64_t positions) {
    Impl & im = *p_;
    if (im.snap_pos < 0 || positions < 1) return false;
    ggml_backend_synchronize(im.backend);
    im.parked.clear();
    auto put = [&](ggml_tensor * t, size_t off, size_t n) {
        Impl::ParkedRows r{ t, off, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(t, r.b.data(), off, n);
        im.parked.push_back(std::move(r));
    };
    for (auto & L : im.ly) {
        for (ggml_tensor * t : { L.raw_snap, L.st_kv_snap, L.st_sc_snap })
            if (t) put(t, 0, ggml_nbytes(t));   // a layer outside the range has no rings
        for (ggml_tensor * t : { L.comp, L.idxk }) {   // row r holds a position >= r (ratio 1 or 2): rows [0, n] cover n positions
            if (!t) continue;
            const int64_t rows = t->ne[1];
            const int64_t n = std::min<int64_t>(rows - 1, positions + 1);
            put(t, 0, (size_t) n * t->nb[1]);
            put(t, (size_t) (rows - 1) * t->nb[1], t->nb[1]);   // the spare row (open compressor groups write it)
        }
    }
    im.parked_pos = im.snap_pos;
    im.parked_prefix = im.prefix;
    im.parked_prefix_pos = im.prefix_pos;
    return true;
}

bool Ds41Dense::unpark() {
    Impl & im = *p_;
    if (im.parked_pos < 0) return false;
    ggml_backend_synchronize(im.backend);
    for (auto & r : im.parked) ggml_backend_tensor_set(r.t, r.b.data(), r.off, r.b.size());
    ggml_backend_synchronize(im.backend);
    im.snap_pos = im.parked_pos;
    im.prefix = std::move(im.parked_prefix);
    im.prefix_pos = im.parked_prefix_pos;
    drop_park();
    return true;
}

void Ds41Dense::drop_park() {
    p_->parked.clear();
    p_->parked.shrink_to_fit();
    p_->parked_pos = -1;
    p_->parked_prefix.clear();
    p_->parked_prefix_pos = -1;
}

bool Ds41Dense::parked() const { return p_->parked_pos >= 0; }

bool Ds41Dense::read_sel(int il, std::vector<int> & sel, std::vector<float> & mask, int * K) {
    Impl & im = *p_;
    if (!im.in_range(il)) { im.err = "read_sel: layer outside the range"; return false; }   // absolute id; the unset range is every layer
    const auto & L = im.ly[(size_t) il];
    sel.clear();
    mask.clear();
    *K = 0;
    if (!L.idx_src || !L.ratio) return true;
    const int64_t cap = im.pass_cap[L.ratio];
    if (cap == 0) return true;
    const int64_t k = std::min<int64_t>(cap, im.ITOP);
    std::vector<int32_t> s((size_t) (im.ITOP * im.n));
    std::vector<float> m((size_t) (im.ITOP * im.n));
    ggml_backend_tensor_get(L.sel, s.data(), 0, s.size() * 4);
    ggml_backend_tensor_get(L.selm, m.data(), 0, m.size() * 4);
    for (int t = 0; t < im.n; ++t)
        for (int64_t j = 0; j < k; ++j) {
            sel.push_back(s[(size_t) (t * im.ITOP + j)]);
            mask.push_back(m[(size_t) (t * im.ITOP + j)]);
        }
    *K = (int) k;
    return true;
}
