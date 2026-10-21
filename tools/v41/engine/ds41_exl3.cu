// tools/v41/engine/ds41_exl3.cu - the one elementwise piece the EXL3 expert path needs beside Maya's kernels:
// DeepSeek-V4.1's clamped SwiGLU, over `n` values, on the caller's stream.
#include <cuda_runtime.h>

namespace {

__global__ void ds41_exl3_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                        float* __restrict__ out, long long n, float limit) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    out[i] = (g / (1.0f + __expf(-g))) * fminf(fmaxf(up[i], -limit), limit);
}

}  // namespace

void ds41_exl3_swiglu(const float* gate, const float* up, float* out, long long n, float limit, cudaStream_t s) {
    if (n <= 0) return;
    ds41_exl3_swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(gate, up, out, n, limit);
}

namespace {

__global__ void ds41_exl3_scatter_add_kernel(const float* __restrict__ part, const int* __restrict__ tok,
                                             const float* __restrict__ w, int n, int H, float* __restrict__ out) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long) n * H;
    if (i >= total) return;
    const int p = (int) (i / H), j = (int) (i % H);
    out[(long long) tok[p] * H + j] += w[p] * part[i];
}

}  // namespace

void ds41_exl3_scatter_add(const float* part, const int* tok, const float* w, int n, int H, float* out,
                           cudaStream_t s) {
    if (n <= 0 || H <= 0) return;
    const long long total = (long long) n * H;
    ds41_exl3_scatter_add_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(part, tok, w, n, H, out);
}

namespace {
__global__ void gather_kernel(const float* x, const int* pairs, int K, int n, int H, float* out) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long) n * H) out[i] = x[(long long) (pairs[i / H] / K) * H + i % H];
}
__global__ void scatter_kernel(const float* part, const int* pairs, int n, int H, float* out) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long) n * H) out[(long long) pairs[i / H] * H + i % H] = part[i];
}
__global__ void sum_kernel(const float* parts, const float* weights, const int* order, int n, int K, int H, float* out) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) n * H) return;
    const int t = i / H, j = i % H;
    float sum = 0;
    for (int k = 0; k < K; ++k) {
        const int p = order[t * K + k];
        sum = fmaf(weights[p], parts[(long long) p * H + j], sum);
    }
    out[i] = sum;
}
}

void ds41_exl3_gather(const float* x, const int* pairs, int K, int n, int H, float* out, cudaStream_t s) {
    gather_kernel<<<(unsigned) (((long long) n * H + 255) / 256), 256, 0, s>>>(x, pairs, K, n, H, out);
}
void ds41_exl3_scatter(const float* part, const int* pairs, int n, int H, float* out, cudaStream_t s) {
    scatter_kernel<<<(unsigned) (((long long) n * H + 255) / 256), 256, 0, s>>>(part, pairs, n, H, out);
}
void ds41_exl3_sum(const float* parts, const float* w, const int* order, int n, int K, int H, float* out, cudaStream_t s) {
    sum_kernel<<<(unsigned) (((long long) n * H + 255) / 256), 256, 0, s>>>(parts, w, order, n, K, H, out);
}
