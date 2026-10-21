// test_ds41_route [--cuda]: deterministic route-level gates, no model files.  Link against ds41_dense.
// Covers corrected-score rank bounds, protected slots, exact-zero IDs/weights, hard masks, cutoff-band clipping,
// raw-score normalization, and prefetch prefix consistency.  This is not a generation-quality or looping test.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// Implementation-private helpers in ds41_dense.cpp; not part of the dense engine's public API.
namespace strata::v41 {
ggml_tensor * ds41_bounded_route_ids(ggml_context *, ggml_tensor *, ggml_tensor *, int, int, int, float, int);
ggml_tensor * ds41_route_weights(ggml_context *, ggml_tensor *, ggml_tensor *, bool, float);
}
using namespace strata::v41;

namespace {
constexpr int NE = 24, NU = 6, NT = 3, NP = 16;
constexpr float SCALE = 1.5f;

struct Fixture {
    std::vector<float> raw = std::vector<float>(NE * NT);
    std::vector<float> correction = std::vector<float>(NE * NT);
    std::vector<float> bias = std::vector<float>(NE);
};

Fixture fixture() {
    Fixture f;
    for (int e = 0; e < NE; ++e) {
        const int rank0 = (e * 7) % NE;
        f.bias[e] = rank0 >= 12 ? 20.0f : rank0 >= NU ? 1.0f : rank0 < 2 ? -0.6f : 0.0f;
        for (int t = 0; t < NT; ++t) {
            const int i = t * NE + e;
            const float logit = -2.0f + 0.17f * ((e * 5 + t * 3) % NE);
            f.raw[i] = std::sqrt(std::log1p(std::exp(logit)));
            const float corrected = 2.4f - 0.04f * ((e * 7 + t * 5) % NE);
            f.correction[i] = corrected - f.raw[i];
        }
    }
    return f;
}

std::vector<int> native_order(const Fixture & f, int t) {
    std::vector<int> ids(NE);
    std::iota(ids.begin(), ids.end(), 0);
    auto score = [&](int e) {
        return f.raw[t * NE + e] + f.correction[t * NE + e] + (f.bias[e] < -1e20f ? f.bias[e] : 0.0f);
    };
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return score(a) > score(b); });
    return ids;
}

std::vector<int> oracle(const Fixture & f, int t, int rank, int protect, float margin) {
    std::vector<int> ids = native_order(f, t);
    const int p = std::clamp(protect, 0, NU);
    rank = std::clamp(rank, NU, NE);
    auto native = [&](int e) { return f.raw[t * NE + e] + f.correction[t * NE + e]; };
    const float cut = 0.5f * (native(ids[NU - 1]) + native(ids[NU]));
    auto score = [&](int e) {
        const float s = native(e);
        if (margin < 0.0f) return s + f.bias[e];
        const float dv = s - cut;
        const float delta = std::fabs(dv) > margin ? 0.0f : std::clamp(dv + f.bias[e], -margin, margin) - dv;
        return s + delta + (f.bias[e] < -1e20f ? f.bias[e] : 0.0f);
    };
    std::stable_sort(ids.begin() + p, ids.begin() + rank, [&](int a, int b) { return score(a) > score(b); });
    ids.resize(NU);
    return ids;
}

bool run(ggml_backend_t be, const char * name, const Fixture & f, int rank, int protect, float margin,
         bool exact_zero = false, bool expect_swap = false, bool expect_unbounded_outside = false) {
    ggml_init_params ip = {ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(1024, false), nullptr, true};
    ggml_context * gc = ggml_init(ip);
    if (!gc) return false;
    ggml_tensor * raw = ggml_new_tensor_2d(gc, GGML_TYPE_F32, NE, NT);
    ggml_tensor * correction = ggml_new_tensor_2d(gc, GGML_TYPE_F32, NE, NT);
    ggml_tensor * rb = ggml_new_tensor_2d(gc, GGML_TYPE_F32, NE, 1);
    ggml_set_input(raw); ggml_set_input(correction); ggml_set_input(rb);
    ggml_tensor * s0 = ggml_add(gc, raw, correction);
    ggml_tensor * ids = ggml_cont(gc, ds41_bounded_route_ids(gc, s0, rb, NU, rank, protect, margin, NU));
    ggml_tensor * weights = ds41_route_weights(gc, raw, ids, true, SCALE);
    ggml_tensor * unnormalized = ds41_route_weights(gc, raw, ids, false, SCALE);
    ggml_tensor * exact = ggml_cont(gc, ggml_argsort_top_k(gc, s0, NU));
    ggml_tensor * exact_weights = ds41_route_weights(gc, raw, exact, true, SCALE);
    ggml_tensor * flat = ggml_cont(gc, ggml_argsort_top_k(gc, ggml_add(gc, s0, rb), NU));
    ggml_tensor * prefetch = ggml_cont(gc, ds41_bounded_route_ids(gc, s0, rb, NU, rank, protect, margin, NP));
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 1024, false);
    for (ggml_tensor * out : {ids, weights, unnormalized, exact, exact_weights, flat, prefetch}) {
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
    }
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(al, gf)) {
        std::fprintf(stderr, "%s: graph allocation failed\n", name);
        ggml_gallocr_free(al); ggml_free(gc);
        return false;
    }
    ggml_backend_tensor_set(raw, f.raw.data(), 0, f.raw.size() * sizeof(float));
    ggml_backend_tensor_set(correction, f.correction.data(), 0, f.correction.size() * sizeof(float));
    ggml_backend_tensor_set(rb, f.bias.data(), 0, f.bias.size() * sizeof(float));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: graph compute failed\n", name);
        ggml_gallocr_free(al); ggml_free(gc);
        return false;
    }
    std::vector<int32_t> got(NU * NT), ref(NU * NT), unbounded(NU * NT), predicted(NP * NT);
    std::vector<float> w(NU * NT), ew(NU * NT), uw(NU * NT);
    ggml_backend_tensor_get(ids, got.data(), 0, got.size() * sizeof(int32_t));
    ggml_backend_tensor_get(exact, ref.data(), 0, ref.size() * sizeof(int32_t));
    ggml_backend_tensor_get(flat, unbounded.data(), 0, unbounded.size() * sizeof(int32_t));
    ggml_backend_tensor_get(prefetch, predicted.data(), 0, predicted.size() * sizeof(int32_t));
    ggml_backend_tensor_get(weights, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_get(exact_weights, ew.data(), 0, ew.size() * sizeof(float));
    ggml_backend_tensor_get(unnormalized, uw.data(), 0, uw.size() * sizeof(float));
    bool ok = true, swapped = false, flat_outside = false;
    auto require = [&](bool condition, const char * what) {
        if (!condition) { std::fprintf(stderr, "%s: %s\n", name, what); ok = false; }
    };
    const int r = std::clamp(rank, NU, NE), p = std::clamp(protect, 0, NU);
    for (int t = 0; t < NT; ++t) {
        const std::vector<int> order = native_order(f, t);
        const std::vector<int> want = oracle(f, t, rank, protect, margin);
        int displaced = 0;
        float sum_raw = 0.0f;
        for (int k = 0; k < NU; ++k) sum_raw += f.raw[t * NE + got[t * NU + k]];
        float sum_weights = 0.0f;
        for (int k = 0; k < NU; ++k) {
            const int i = t * NU + k, e = got[i];
            // Exact ties have no promised expert-ID tie break.  The zero-bias contract below compares against the
            // backend's native selector, not std::stable_sort's incidental index ordering.
            if (!exact_zero) {
                require(e == want[k], "IDs differ from independent bounded-policy oracle");
                require(std::find(order.begin(), order.begin() + r, e) != order.begin() + r, "native rank exceeded R");
            }
            require(e == predicted[t * NP + k], "prefetch prefix differs from route IDs");
            if (k < p && !exact_zero) require(e == order[k], "protected native prefix changed");
            if (exact_zero
                    ? std::find(ref.begin() + t * NU, ref.begin() + (t + 1) * NU, e) == ref.begin() + (t + 1) * NU
                    : std::find(order.begin(), order.begin() + NU, e) == order.begin() + NU) ++displaced;
            flat_outside |= std::find(order.begin(), order.begin() + r, unbounded[i]) == order.begin() + r;
            const float expected = SCALE * f.raw[t * NE + e] / (sum_raw + 1e-20f);
            require(std::fabs(w[i] - expected) <= 2e-6f, "weights were not normalized from unmodified raw scores");
            require(std::fabs(uw[i] - SCALE * f.raw[t * NE + e]) <= 2e-6f, "unnormalized raw weights changed");
            sum_weights += w[i];
        }
        require(displaced <= NU - p, "more than NU-P exact slots displaced");
        require(std::fabs(sum_weights - SCALE) <= 2e-6f, "normalized weights do not sum to route_scale");
        swapped |= displaced > 0;
        std::vector<int32_t> unique(predicted.begin() + t * NP, predicted.begin() + (t + 1) * NP);
        std::sort(unique.begin(), unique.end());
        require(std::adjacent_find(unique.begin(), unique.end()) == unique.end(), "duplicate prefetch IDs");
    }
    if (exact_zero) {
        require(std::memcmp(got.data(), ref.data(), got.size() * sizeof(int32_t)) == 0, "zero bias changed exact ID order");
        require(std::memcmp(w.data(), ew.data(), w.size() * sizeof(float)) == 0, "zero bias changed exact weight bits");
    }
    if (expect_swap) require(swapped, "fixture did not exercise a cache-preferred rank displacement");
    if (expect_unbounded_outside) require(flat_outside, "fixture did not demonstrate unbounded rank displacement");
    std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    ggml_gallocr_free(al); ggml_free(gc);
    return ok;
}
}  // namespace

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!cuda) be = ggml_backend_cpu_init();
    if (!be) { std::fprintf(stderr, "requested backend unavailable\n"); return 1; }
    int failed = 0;
    Fixture f = fixture();
    failed += !run(be, "bounded-corrected-rank", f, 12, 2, -1.0f, false, true, true);
    failed += !run(be, "R-clamped-to-NU", f, 1, 2, -1.0f);
    failed += !run(be, "R-clamped-to-NE", f, 999, 2, -1.0f);
    failed += !run(be, "negative-protect-clamped", f, 12, -3, -1.0f);
    failed += !run(be, "all-slots-protected", f, 12, 999, -1.0f);
    f.bias.assign(NE, 0.0f);
    failed += !run(be, "zero-bias-exact", f, 12, 2, -1.0f, true);
    failed += !run(be, "zero-bias-margin-exact", f, 12, 2, 0.05f, true);
    std::fill(f.raw.begin(), f.raw.end(), 0.5f);
    std::fill(f.correction.begin(), f.correction.end(), 0.5f);
    failed += !run(be, "zero-bias-tied-exact", f, 12, 2, -1.0f, true);
    f = fixture();
    // A hard-pruned native leader must not be resurrected by protection or candidate truncation.
    f.bias[native_order(f, 0)[0]] = -1e30f;
    failed += !run(be, "hard-mask-before-protect", f, 12, 6, -1.0f);
    f = fixture();
    // Extreme negative bias inside the cutoff band and extreme positive bias outside it exercise margin leakage.
    // The independent oracle checks that only the in-band ranking changes, with adjustments clipped to the band.
    failed += !run(be, "bounded-cutoff-margin", f, NE, 2, 0.09f);
    f.bias[native_order(f, 0)[NU - 1]] = -10.0f;
    failed += !run(be, "negative-cutoff-margin", f, NE, 2, 0.09f);
    // Several leaders fit inside the old epsilon-based protection threshold; only exactly P IDs may be reserved.
    for (int e = 0; e < NE; ++e) {
        f.bias[e] = e >= NU ? 1.0f : 0.0f;
        for (int t = 0; t < NT; ++t) {
            f.raw[t * NE + e] = 0.5f;
            f.correction[t * NE + e] = 0.5f - 2.5e-7f * ((e + t) % NE);
        }
    }
    failed += !run(be, "near-tied-protected-prefix", f, 12, 2, -1.0f, false, true);
    ggml_backend_free(be);
    return failed ? 1 : 0;
}
