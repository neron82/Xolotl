// tools/v41/engine/test_ds41_exl3.cpp - checks the SAGE EXL3 pieces without a model: the safetensors header scan,
// the per-expert blob layout/read, and that mv()/the clamped SwiGLU run on a zero-weight tiled matrix.
#include "ds41_moe.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace strata::v41;
void ds41_exl3_swiglu(const float*, const float*, float*, long long, float, cudaStream_t);

namespace {

void put64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}

struct StWriter {
    std::string h;
    std::vector<uint8_t> data;
    void add(const std::string& name, const char* dtype, const std::vector<int64_t>& shape, size_t bytes) {
        const size_t off = data.size();
        data.resize(off + bytes, 0);
        if (!h.empty()) h += ",";
        h += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) h += ",";
            h += std::to_string(shape[i]);
        }
        h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    }
    bool write(const std::string& path, std::string& err) const {
        const std::string body = "{" + h + "}";
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { err = "cannot create " + path; return false; }
        std::vector<uint8_t> head;
        put64(head, (uint64_t) body.size());
        head.insert(head.end(), body.begin(), body.end());
        bool ok = std::fwrite(head.data(), 1, head.size(), f) == head.size() &&
                  std::fwrite(data.data(), 1, data.size(), f) == data.size();
        std::fclose(f);
        if (!ok) err = "short write " + path;
        return ok;
    }
};

int test_scan() {
    std::string err;
    char tmpl[] = "/tmp/ds41_exl3_XXXXXX";
    const char* dir = ::mkdtemp(tmpl);
    if (!dir) { std::fprintf(stderr, "mkdtemp failed\n"); return 1; }
    const int64_t D = 256, FF = 256;
    StWriter w;
    for (int e = 0; e < 2; ++e) {
        const int k2 = e == 0 ? 2 : 6;
        for (int role = 0; role < 3; ++role) {
            const int wgt = role == 0 ? 1 : role == 2 ? 3 : 2;
            const std::string pre = "layers.0.ffn.experts." + std::to_string(e) + ".w" + std::to_string(wgt) + ".";
            w.add(pre + "trellis", "I16", {D / 16, FF / 16, 8 * k2}, (size_t) (D / 16) * (FF / 16) * 16 * k2);
            w.add(pre + "suh", "F16", {D}, (size_t) D * 2);
            w.add(pre + "svh", "F16", {FF}, (size_t) FF * 2);
        }
    }
    const std::string path = std::string(dir) + "/model-00001-of-00001.safetensors";
    if (!w.write(path, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    Ds41MoeGeom base;
    base.n_layers = 1;
    base.n_experts = 2;
    base.n_embd = D;
    base.n_ff = FF;
    base.top_k = 1;
    base.swiglu_limit = 10.0f;
    base.gate_types.assign(1, -1);   // a first-shard GGUF marks layers whose experts live in a later shard
    Ds41FileBlobs src("/nonexistent");
    if (!src.open_exl3(dir, base, err)) { std::fprintf(stderr, "open_exl3: %s\n", err.c_str()); return 1; }
    if (!src.geom().routed(0) || src.geom().per_layer()) {
        std::fprintf(stderr, "open_exl3 kept the base's no-experts mark\n");
        return 1;
    }
    if (!src.exl3() || src.geom().xl_k2_gate.size() != 2 || src.geom().xl_k2_gate[0] != 2 ||
        src.geom().xl_k2_gate[1] != 6) {
        std::fprintf(stderr, "class/k2 wrong\n");
        return 1;
    }
    const size_t tb0 = (size_t) (D / 16) * (FF / 16) * 16 * 2;
    const size_t role0 = ((D * 2 + FF * 2 + tb0 + 255) / 256) * 256;
    const size_t bytes0 = role0 * 3;
    const size_t tb1 = (size_t) (D / 16) * (FF / 16) * 16 * 6;
    const size_t role1 = ((D * 2 + FF * 2 + tb1 + 255) / 256) * 256;
    const size_t bytes = role1 * 3;
    if (src.blob_bytes() != bytes || src.expert_bytes(0, 0) != bytes0 || src.expert_bytes(0, 1) != bytes) {
        std::fprintf(stderr, "blob bytes %zu/%zu/%zu expected %zu/%zu\n", src.blob_bytes(), src.expert_bytes(0, 0),
                     src.expert_bytes(0, 1), bytes0, bytes);
        return 1;
    }
    std::vector<uint8_t> got(bytes, 0xff), half(bytes, 0xff);
    src.read(0, 1, got.data());
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i]) { std::fprintf(stderr, "read byte %zu not zero\n", i); return 1; }
    std::fill(got.begin(), got.end(), 0xff);
    src.read(0, 0, got.data());
    const bool fast = std::getenv("STRATA_EXL3_FAST") && std::atoi(std::getenv("STRATA_EXL3_FAST")) > 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const int want = fast && i >= bytes0 ? 0xff : 0;
        if (got[i] != want) { std::fprintf(stderr, "small class byte %zu wrong (compact read bounds)\n", i); return 1; }
    }
    const size_t p0 = bytes / 2, p1 = bytes;
    src.read_part(0, 1, 1, 2, half.data());
    for (size_t i = p0; i < p1; ++i)
        if (half[i]) { std::fprintf(stderr, "part byte %zu not zero\n", i); return 1; }
    std::string rm = std::string("rm -rf ") + dir;
    if (std::system(rm.c_str()) != 0) std::fprintf(stderr, "warning: %s failed\n", rm.c_str());
    return 0;
}

int test_mv() {
    const int D = 256, FF = 256, k2 = 2;
    const size_t tb = (size_t) (D / 16) * (FF / 16) * 16 * k2 * sizeof(uint16_t);
    void *trellis = nullptr, *suh = nullptr, *svh = nullptr, *x = nullptr, *y = nullptr, *g = nullptr, *u = nullptr, *h = nullptr;
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess || cudaMalloc(&trellis, tb) != cudaSuccess ||
        cudaMalloc(&suh, (size_t) D * 2) != cudaSuccess || cudaMalloc(&svh, (size_t) FF * 2) != cudaSuccess ||
        cudaMalloc(&x, (size_t) D * 4) != cudaSuccess || cudaMalloc(&y, (size_t) FF * 4) != cudaSuccess ||
        cudaMalloc(&g, (size_t) FF * 4) != cudaSuccess || cudaMalloc(&u, (size_t) FF * 4) != cudaSuccess ||
        cudaMalloc(&h, (size_t) FF * 4) != cudaSuccess) {
        std::fprintf(stderr, "cuda alloc failed\n");
        return 1;
    }
    cudaMemsetAsync(trellis, 0, tb, s);
    cudaMemsetAsync(suh, 0, (size_t) D * 2, s);
    cudaMemsetAsync(svh, 0, (size_t) FF * 2, s);
    cudaMemsetAsync(x, 0, (size_t) D * 4, s);
    cudaMemsetAsync(g, 0, (size_t) FF * 4, s);
    cudaMemsetAsync(u, 0, (size_t) FF * 4, s);
    if (!strata::kernels::exl3::prepare(s)) { std::fprintf(stderr, "prepare failed\n"); return 1; }
    strata::kernels::exl3::Mat m;
    m.trellis = (const uint32_t*) trellis;
    m.suh = (const uint16_t*) suh;
    m.svh = (const uint16_t*) svh;
    m.k = D;
    m.n = FF;
    m.k2 = k2;
    m.cb = 2;
    m.ld = 0;
    strata::kernels::exl3::MvJob j;
    j.m = &m;
    j.xf = (const float*) x;
    j.y = (float*) g;
    if (!strata::kernels::exl3::mv(&j, 1, s)) { std::fprintf(stderr, "mv failed\n"); return 1; }
    ds41_exl3_swiglu((const float*) g, (const float*) u, (float*) h, FF, 10.0f, s);
    j.y = (float*) y;
    if (!strata::kernels::exl3::mv(&j, 1, s)) { std::fprintf(stderr, "mv 2 failed\n"); return 1; }
    cudaStreamSynchronize(s);
    std::vector<float> out(FF, 1.0f);
    cudaMemcpy(out.data(), y, (size_t) FF * 4, cudaMemcpyDeviceToHost);
    for (float v : out)
        if (v != 0.0f) { std::fprintf(stderr, "zero weights produced %g\n", v); return 1; }
    cudaFree(trellis); cudaFree(suh); cudaFree(svh); cudaFree(x); cudaFree(y); cudaFree(g); cudaFree(u); cudaFree(h);
    cudaStreamDestroy(s);
    return 0;
}

int test_tier() {
    const int64_t D = 256, FF = 256, k2 = 2;
    const size_t tb = (size_t) (D / 16) * (FF / 16) * 16 * k2;
    size_t suh[3] = {}, svh[3] = {}, trellis[3] = {}, off = 0;
    for (int r = 0; r < 3; ++r) {
        suh[r] = off; off += (size_t) D * 2;
        svh[r] = off; off += (size_t) FF * 2;
        off = (off + 255) & ~(size_t) 255;
        trellis[r] = off; off += tb;
        off = (off + 255) & ~(size_t) 255;
    }
    const size_t bytes = off;
    std::vector<uint8_t> blob(bytes, 0);
    for (int r = 0; r < 3; ++r) {
        for (int i = 0; i < D; ++i) {
            const uint16_t bits = i % 3 == 0 ? 0x3c00 : 0x3800;
            std::memcpy(&blob[suh[r] + (size_t) i * 2], &bits, 2);
        }
        for (int i = 0; i < FF; ++i) {
            const uint16_t bits = i % 5 == 0 ? 0x4000 : 0x3c00;
            std::memcpy(&blob[svh[r] + (size_t) i * 2], &bits, 2);
        }
        for (size_t i = 0; i < tb; ++i) blob[trellis[r] + i] = (uint8_t) ((i * 7 + (size_t) r * 13) & 0xff);
    }
    Ds41MoeGeom g;
    g.n_layers = 1;
    g.n_experts = 2;
    g.n_embd = D;
    g.n_ff = FF;
    g.top_k = 1;
    g.swiglu_limit = 10.0f;
    g.gu_type = g.d_type = kDs41Exl3Type;
    g.exl3 = true;
    g.eclass.assign(2, 0);
    g.cls_gu.assign(1, kDs41Exl3Type);
    g.cls_d.assign(1, kDs41Exl3Type);
    g.xl_k2_gate.assign(1, (int) k2);
    g.xl_k2_up.assign(1, (int) k2);
    g.xl_k2_down.assign(1, (int) k2);
    Ds41MemoryBlobs src(g, bytes);
    src.set(0, 0, blob.data());
    std::vector<uint8_t> blob2 = blob;
    for (int i = 0; i < FF; ++i) {
        const uint16_t bits = i % 3 == 0 ? 0x3800 : 0x3c00;
        std::memcpy(&blob2[svh[1] + (size_t) i * 2], &bits, 2);
    }
    src.set(0, 1, blob2.data());
    Ds41MoeConfig cfg;
    cfg.device = -1;
    cfg.cpu_only = false;
    cfg.no_cache = true;
    cfg.slots = 0;
    cfg.arena_gib = 0;
    cfg.pf_b = 0;
    cfg.threads = 1;
    Ds41MoeTier tier;
    std::string err;
    if (!tier.init(g, &src, cfg, err)) { std::fprintf(stderr, "tier init: %s\n", err.c_str()); return 1; }
    const int32_t ids[1] = {0};
    const float w[1] = {1.0f};
    std::vector<float> x((size_t) D, 0.0f), out((size_t) D, 0.0f);
    for (int i = 0; i < D; ++i) x[(size_t) i] = 0.001f * (float) (i % 17);
    if (!tier.run(0, ids, w, x.data(), out.data())) { std::fprintf(stderr, "tier run failed\n"); return 1; }

    // the same blob, decoded by hand through one gate/up/down job: catches a role/offset swap in the tier path
    cudaStream_t s = nullptr;
    void *db = nullptr, *dx = nullptr, *dg = nullptr, *du = nullptr, *dh = nullptr, *dy = nullptr, *dw = nullptr, *dout = nullptr;
    cudaStreamCreate(&s);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dx, (size_t) D * 4);
    cudaMalloc(&dg, (size_t) FF * 4);
    cudaMalloc(&du, (size_t) FF * 4);
    cudaMalloc(&dh, (size_t) FF * 4);
    cudaMalloc(&dy, (size_t) D * 4);
    cudaMalloc(&dw, 4);
    cudaMalloc(&dout, (size_t) D * 4);
    cudaMemcpy(db, blob.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), (size_t) D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, 4, cudaMemcpyHostToDevice);
    strata::kernels::exl3::prepare(s);
    strata::kernels::exl3::Mat gm, um, dm;
    for (int r = 0; r < 3; ++r) {
        strata::kernels::exl3::Mat& m = r == 0 ? gm : r == 1 ? um : dm;
        m.trellis = (const uint32_t*) ((uint8_t*) db + trellis[r]);
        m.suh = (const uint16_t*) ((uint8_t*) db + suh[r]);
        m.svh = (const uint16_t*) ((uint8_t*) db + svh[r]);
        m.k = r < 2 ? (int) D : (int) FF;
        m.n = r < 2 ? (int) FF : (int) D;
        m.k2 = (int) k2;
        m.cb = 2;
        m.ld = 0;
    }
    strata::kernels::exl3::MvJob jg, ju, jd;
    jg.m = &gm; jg.xf = (const float*) dx; jg.y = (float*) dg;
    ju.m = &um; ju.xf = (const float*) dx; ju.y = (float*) du;
    jd.m = &dm; jd.xf = (const float*) dh; jd.y = (float*) dy;
    if (!strata::kernels::exl3::mv(&jg, 1, s) || !strata::kernels::exl3::mv(&ju, 1, s)) {
        std::fprintf(stderr, "manual gate/up failed\n");
        return 1;
    }
    ds41_exl3_swiglu((const float*) dg, (const float*) du, (float*) dh, FF, 10.0f, s);
    if (!strata::kernels::exl3::mv(&jd, 1, s)) { std::fprintf(stderr, "manual down failed\n"); return 1; }
    strata::kernels::decode_rows_sum((const float*) dy, (const float*) dw, 1, D, (float*) dout, s);
    std::vector<float> ref((size_t) D, 0.0f);
    cudaMemcpy(ref.data(), dout, (size_t) D * 4, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(s);
    for (int i = 0; i < D; ++i)
        if (out[(size_t) i] != ref[(size_t) i]) {
            std::fprintf(stderr, "tier %g != manual %g at %d\n", out[(size_t) i], ref[(size_t) i], i);
            return 1;
        }
    cudaFree(db); cudaFree(dx); cudaFree(dg); cudaFree(du); cudaFree(dh); cudaFree(dy); cudaFree(dw); cudaFree(dout);
    cudaStreamDestroy(s);
    std::vector<float> x2((size_t) 3 * D, 0.0f), out2((size_t) 3 * D, 1.0f), out3((size_t) 3 * D, 1.0f);
    const int32_t ids2[3] = {0, 1, 0};
    const float w2[3] = {0.5f, 0.75f, 0.25f};
    for (int t = 0; t < 3; ++t)
        for (int i = 0; i < D; ++i) x2[(size_t) t * D + i] = 0.001f * (float) ((i + 7 * t) % 19);
    if (!tier.run_multi(0, 3, ids2, w2, x2.data(), out2.data())) { std::fprintf(stderr, "tier run_multi failed\n"); return 1; }
    if (!tier.run_chunk(0, 3, ids2, w2, x2.data(), out3.data())) { std::fprintf(stderr, "tier run_chunk failed\n"); return 1; }
    for (size_t i = 0; i < out2.size(); ++i)
        if (out2[i] != out3[i]) {
            std::fprintf(stderr, "chunk %g != per-token %g at %zu\n", out3[i], out2[i], i);
            return 1;
        }
    void* dout3 = nullptr;
    std::vector<float> out4((size_t) 3 * D, 0.0f);
    cudaMalloc(&dout3, out4.size() * 4);
    if (!tier.run_chunk_dev(0, 3, ids2, w2, x2.data(), dout3)) { std::fprintf(stderr, "tier run_chunk_dev failed\n"); return 1; }
    cudaMemcpy(out4.data(), dout3, out4.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dout3);
    for (size_t i = 0; i < out2.size(); ++i)
        if (out2[i] != out4[i]) {
            std::fprintf(stderr, "chunk_dev %g != per-token %g at %zu\n", out4[i], out2[i], i);
            return 1;
        }
    return 0;
}


size_t make_blob(int D, int FF, int k2, int seed, std::vector<uint8_t>& blob) {
    const size_t tb = (size_t) (D / 16) * (FF / 16) * 16 * k2;
    size_t suh[3] = {}, svh[3] = {}, tr[3] = {}, off = 0;
    for (int r = 0; r < 3; ++r) {
        suh[r] = off; off += (size_t) D * 2;
        svh[r] = off; off += (size_t) FF * 2;
        off = (off + 255) & ~(size_t) 255;
        tr[r] = off; off += tb;
        off = (off + 255) & ~(size_t) 255;
    }
    blob.assign(off, 0);
    for (int r = 0; r < 3; ++r) {
        for (int i = 0; i < D; ++i) {
            const uint16_t bits = (uint16_t) (0x3c00 + ((i + seed) % 3) * 0x200);
            std::memcpy(&blob[suh[r] + (size_t) i * 2], &bits, 2);
        }
        for (int i = 0; i < FF; ++i) {
            const uint16_t bits = (uint16_t) (0x3c00 + ((i + seed) % 5) * 0x100);
            std::memcpy(&blob[svh[r] + (size_t) i * 2], &bits, 2);
        }
        for (size_t i = 0; i < tb; ++i) blob[tr[r] + i] = (uint8_t) ((i * 7 + (size_t) r * 13 + (size_t) seed) & 0xff);
    }
    return off;
}

int test_mixed_k() {
    const int D = 256, FF = 256;
    std::vector<uint8_t> b0, b1;
    const size_t n0 = make_blob(D, FF, 2, 1, b0), n1 = make_blob(D, FF, 4, 9, b1);
    const size_t bytes = std::max(n0, n1);
    b0.resize(bytes, 0);
    b1.resize(bytes, 0);
    Ds41MoeGeom g;
    g.n_layers = 1;
    g.n_experts = 2;
    g.n_embd = D;
    g.n_ff = FF;
    g.top_k = 1;
    g.swiglu_limit = 10.0f;
    g.gu_type = g.d_type = kDs41Exl3Type;
    g.exl3 = true;
    g.eclass.assign(2, 0);
    g.eclass[1] = 1;
    g.cls_gu.assign(2, kDs41Exl3Type);
    g.cls_d.assign(2, kDs41Exl3Type);
    g.xl_k2_gate = {2, 4};
    g.xl_k2_up = {2, 4};
    g.xl_k2_down = {2, 4};
    Ds41MemoryBlobs src(g, bytes);
    src.set(0, 0, b0.data());
    src.set(0, 1, b1.data());
    Ds41MoeConfig cfg;
    cfg.device = -1;
    cfg.cpu_only = false;
    cfg.no_cache = true;
    cfg.slots = 0;
    cfg.arena_gib = 0;
    cfg.pf_b = 0;
    cfg.threads = 1;
    Ds41MoeTier tier;
    std::string err;
    if (!tier.init(g, &src, cfg, err)) { std::fprintf(stderr, "mixed init: %s\n", err.c_str()); return 1; }
    const int32_t ids[3] = {0, 1, 0};
    const float w[3] = {0.5f, 0.75f, 0.25f};
    std::vector<float> x((size_t) 3 * D), per((size_t) 3 * D, 0.0f), ch((size_t) 3 * D, 0.0f);
    for (int t = 0; t < 3; ++t)
        for (int i = 0; i < D; ++i) x[(size_t) t * D + i] = 0.001f * (float) ((i + 5 * t) % 17);
    if (!tier.run_multi(0, 3, ids, w, x.data(), per.data())) { std::fprintf(stderr, "mixed run_multi failed\n"); return 1; }
    if (!tier.run_chunk(0, 3, ids, w, x.data(), ch.data())) { std::fprintf(stderr, "mixed run_chunk failed\n"); return 1; }
    for (size_t i = 0; i < per.size(); ++i)
        if (per[i] != ch[i]) { std::fprintf(stderr, "mixed chunk %g != per-token %g at %zu\n", ch[i], per[i], i); return 1; }
    const bool fast = std::getenv("STRATA_EXL3_FAST") && std::atoi(std::getenv("STRATA_EXL3_FAST")) > 0;
    if (fast) {
        tier.close();
        g.top_k = 2;
        cfg.arena_gib = (double) (n0 + n1) / 1073741824.0;
        cfg.pin = false;
        if (!tier.init(g, &src, cfg, err) || !tier.build_arena({}, err)) {
            std::fprintf(stderr, "mixed arena: %s\n", err.c_str()); return 1;
        }
        tier.reset_stats();
        constexpr int N = 48;
        std::vector<int32_t> bi(N * 2);
        std::vector<float> bw(N * 2), bx(N * D), ref(N * D), result(N * D);
        for (int t = 0; t < N; ++t) {
            bi[t * 2] = t % 2; bi[t * 2 + 1] = 1 - t % 2;
            bw[t * 2] = 0.75f; bw[t * 2 + 1] = 0.25f;
            for (int j = 0; j < D; ++j) bx[t * D + j] = 0.001f * ((j + t) % 17 - 8);
            if (!tier.run(0, bi.data() + t * 2, bw.data() + t * 2, bx.data() + t * D, ref.data() + t * D)) return 1;
        }
        if (!tier.run_chunk(0, N, bi.data(), bw.data(), bx.data(), result.data())) return 1;
        double diff = 0, norm = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            diff += (double) (result[i] - ref[i]) * (result[i] - ref[i]);
            norm += (double) ref[i] * ref[i];
        }
        const double rel = std::sqrt(diff / std::max(norm, 1e-30));
        if (!std::isfinite(rel) || rel > 1e-7) {
            std::fprintf(stderr, "mixed batch relative L2 %.6g\n", rel); return 1;
        }
        std::fprintf(stderr, "mixed batch: relative L2 %.6g\n", rel);
        const auto stats = tier.stats();
        if (stats.file_tier != 0) { std::fprintf(stderr, "compact arena missed an expert\n"); return 1; }
        tier.release_chunk();
        if (!tier.run_chunk(0, N, bi.data(), bw.data(), bx.data(), result.data())) return 1;
        tier.close();
        cfg.no_cache = false;
        cfg.slots = 1;
        cfg.pf_b = 1;
        cfg.dither = false;
        Ds41MoeTier cached;
        if (!cached.init(g, &src, cfg, err) || !cached.seed_from_ranked({{0, 0}, {0, 1}}, err)) {
            std::fprintf(stderr, "prefetch init: %s\n", err.c_str()); return 1;
        }
        cached.reset_stats();
        const int32_t predicted = 1;
        cached.prefetch(0, &predicted, 1);
        if (!cached.run(0, bi.data(), bw.data(), bx.data(), result.data())) return 1;
        if (cached.stats().prefetched_useful != 1) {
            std::fprintf(stderr, "EXL3 did not consume its prefetch\n"); return 1;
        }
        for (int j = 0; j < D; ++j)
            if (result[j] != ref[j]) { std::fprintf(stderr, "prefetched output differs\n"); return 1; }
        cached.reset_stats();
        if (!cached.run(0, bi.data(), bw.data(), bx.data(), result.data())) return 1;
        if (cached.stats().prefetched_useful != 0) {
            std::fprintf(stderr, "EXL3 reused stale prefetch metadata\n"); return 1;
        }
    }
    return 0;
}
int test_real_run(const char* dir) {
    Ds41MoeGeom base;
    base.n_layers = 1;
    base.n_experts = 2;
    base.n_embd = 5120;
    base.n_ff = 2304;
    base.top_k = 1;
    base.swiglu_limit = 10.0f;
    Ds41FileBlobs src("/nonexistent");
    std::string err;
    if (!src.open_exl3(dir, base, err)) { std::fprintf(stderr, "real run open: %s\n", err.c_str()); return 1; }
    Ds41MoeConfig cfg;
    cfg.device = -1;
    cfg.cpu_only = false;
    cfg.no_cache = true;
    cfg.slots = 0;
    cfg.arena_gib = 0;
    cfg.pf_b = 0;
    cfg.threads = 1;
    Ds41MoeTier tier;
    if (!tier.init(src.geom(), &src, cfg, err)) { std::fprintf(stderr, "real run tier: %s\n", err.c_str()); return 1; }
    const int32_t ids[1] = {0};
    const float w[1] = {1.0f};
    std::vector<float> x(5120), out(5120, 0.0f);
    for (int i = 0; i < 5120; ++i) x[(size_t) i] = 0.001f * (float) (i % 23);
    if (!tier.run(0, ids, w, x.data(), out.data())) { std::fprintf(stderr, "real run failed\n"); return 1; }
    double sum = 0;
    for (float v : out) { if (!(v == v) || v > 1e30f || v < -1e30f) { std::fprintf(stderr, "real run non-finite\n"); return 1; } sum += v; }
    std::vector<float> x2(2 * 5120), per(2 * 5120, 0.0f), chunk(2 * 5120, 0.0f);
    for (int t = 0; t < 2; ++t)
        for (int i = 0; i < 5120; ++i) x2[(size_t) t * 5120 + i] = 0.001f * (float) ((i + 11 * t) % 29);
    const int32_t ids2[3] = {0, 1, 0};
    const float w2[3] = {0.75f, 0.25f, 0.5f};
    x2.resize((size_t) 3 * 5120);
    per.assign((size_t) 3 * 5120, 0.0f);
    chunk.assign((size_t) 3 * 5120, 0.0f);
    for (int t = 0; t < 3; ++t)
        for (int i = 0; i < 5120; ++i) x2[(size_t) t * 5120 + i] = 0.001f * (float) ((i + 11 * t) % 29);
    if (!tier.run_multi(0, 3, ids2, w2, x2.data(), per.data())) { std::fprintf(stderr, "real run_multi failed\n"); return 1; }
    if (!tier.run_chunk(0, 3, ids2, w2, x2.data(), chunk.data())) { std::fprintf(stderr, "real run_chunk failed\n"); return 1; }
    for (size_t i = 0; i < per.size(); ++i)
        if (per[i] != chunk[i]) { std::fprintf(stderr, "real chunk %g != per-token %g at %zu\n", chunk[i], per[i], i); return 1; }
    std::fprintf(stderr, "real run: sum %.6f, chunk == per-token\n", sum);
    return sum == 0.0 ? 1 : 0;
}
int test_real_header(const char* dir, bool full) {
    Ds41MoeGeom base;
    base.n_layers = full ? 40 : 1;
    base.n_experts = full ? 384 : 1;
    base.n_embd = 5120;
    base.n_ff = 2304;
    base.top_k = full ? 6 : 1;
    base.swiglu_limit = 10.0f;
    Ds41FileBlobs src("/nonexistent");
    std::string err;
    if (!src.open_exl3(dir, base, err)) { std::fprintf(stderr, "real header: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom& g = src.geom();
    int64_t mixed = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_experts; ++e) {
            const int c = g.cls(l, e);
            mixed += g.xl_k2_gate[(size_t) c] != g.xl_k2_up[(size_t) c];
        }
    std::fprintf(stderr, "real header: %zu classes, %lld experts with gate/up K2 differing\n", g.n_classes(), (long long) mixed);
    if (full && (g.n_classes() != 27 || mixed != 820)) {
        std::fprintf(stderr, "real header: expected 27 classes / 820 mixed experts\n");
        return 1;
    }
    if (full) {
        std::fprintf(stderr, "real header: %zu bytes, largest class\n", src.blob_bytes());
        return 0;
    }
    std::vector<uint8_t> blob((size_t) src.blob_bytes(), 0);
    src.read(0, 0, blob.data());
    if (std::getenv("DS41_EXL3_DUMP"))
        if (std::FILE* f = std::fopen("/tmp/ds41_exl3_blob.bin", "wb")) { std::fwrite(blob.data(), 1, blob.size(), f); std::fclose(f); }
    std::fprintf(stderr, "real header: k2 %d/%d/%d, %zu bytes, first suh %02x%02x\n", src.geom().xl_k2_gate[0],
                 src.geom().xl_k2_up[0], src.geom().xl_k2_down[0], src.blob_bytes(), blob[0], blob[1]);
    return src.geom().xl_k2_gate[0] > 0 && src.blob_bytes() > 0 ? 0 : 1;
}

}  // namespace

int main() {
    const int a = test_scan();
    const int b = test_mv();
    const int c = test_tier();
    const char* real = std::getenv("DS41_EXL3_REAL");
    const int d = real ? test_real_header(real, std::getenv("DS41_EXL3_REAL_FULL") != nullptr) : 0;
    const int e = test_mixed_k();
    const char* real_run = std::getenv("DS41_EXL3_REAL_RUN");
    const int f = real_run ? test_real_run(real_run) : 0;
    std::fprintf(stderr, "test_ds41_exl3: scan %s, mv %s, tier %s, mixed %s%s%s\n", a ? "FAIL" : "ok",
                 b ? "FAIL" : "ok", c ? "FAIL" : "ok", e ? "FAIL" : "ok",
                 real ? (d ? ", real FAIL" : ", real ok") : "",
                 real_run ? (f ? ", real-run FAIL" : ", real-run ok") : "");
    return a || b || c || d || e || f ? 1 : 0;
}
