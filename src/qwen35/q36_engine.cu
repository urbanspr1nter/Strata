// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__HIP_PLATFORM_AMD__)
#include <cub/cub.cuh>   // the device sort of the sampled path (the HIP build samples from the host as 0.1.44)
#define STRATA_Q36_DEVICE_TOPK 1
#else
#define STRATA_Q36_DEVICE_TOPK 0
#endif

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
// the 8-bit KV cache (--kv q8_0): a K or V row of one head at one cell is 8 fp16 scales (one per 32 values) followed by
// 256 int8 - llama.cpp's q8_0 cache in all but the layout (272 bytes a row instead of 1,024 as fp32)
constexpr int kKvRowQ8 = 16 + 256;
__device__ __forceinline__ void kv_q8_store(uint8_t* row, float v, int i) {   // every thread of a 256-wide block: i = the value
    float a = fabsf(v);
    for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));   // the warp's 32 values share a scale
    const float scale = a / 127.f;
    const float inv = scale > 0.f ? 1.f / scale : 0.f;
    const int qv = max(-127, min(127, __float2int_rn(v * inv)));
    row[16 + i] = (uint8_t) (int8_t) qv;
    if ((i & 31) == 0) reinterpret_cast<__half*>(row)[i >> 5] = __float2half(scale);
}
__device__ __forceinline__ float kv_q8_load(const uint8_t* row, int i) {
    return __half2float(reinterpret_cast<const __half*>(row)[i >> 5]) * (float) (int8_t) row[16 + i];
}

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// grid (H + HKV, T), block 256 (HD <= 256).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, bool q8) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const size_t cell = (size_t) kh * ctx + pos;
        if (!q8) {
            kc[cell * HD + i] = v;
            vc[cell * HD + i] = vx[((size_t) t * HKV + kh) * HD + i];
        } else {   // (HD == 256: every thread of the block takes part in its warp's scale)
            kv_q8_store(reinterpret_cast<uint8_t*>(kc) + cell * kKvRowQ8, v, i);
            kv_q8_store(reinterpret_cast<uint8_t*>(vc) + cell * kKvRowQ8, vx[((size_t) t * HKV + kh) * HD + i], i);
        }
    }
}

/// Split-K attention as 0.1.44 had it (STRATA_Q36_ATTN=v1, the bit-exact reference): one block per (chunk of 256
/// cells, query row), every thread walks its own cell's K row - 32 rows a warp, a fraction of the card's bandwidth
/// (32 tok/s after a 14K prompt on an RTX 2080).
__global__ void k_attn_part_v1(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                               int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const float* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The 8 query heads of one kv head and one query token in one block (STRATA_Q36_ATTN=v3, the default): the
/// chunk's K and V rows are read once for all eight (v2 read them once per query head, eight times over, mostly
/// from DRAM again).  grid (n_chunk, n_tok * HKV), block 256; H / HKV == 8, HD == 256.  Each warp scores 32
/// cells: a cell's K row as 32 x 8 floats across the lanes, eight dot products reduced with shuffles.
template <bool Q8>
__global__ void __launch_bounds__(256) k_attn_part_gqa(const float* q, const float* kc, const float* vc,
                                                       const int32_t* pos0, int t0, int H, int HKV, int ctx,
                                                       float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256, G = 8;
    __shared__ float sc[G][kChunkCells];
    __shared__ float sh[kThreads];
    const int c = blockIdx.x, grp = blockIdx.y;   // grp = (t - t0) * HKV + kvh
    const int t = t0 + grp / HKV, kvh = grp % HKV;
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    const int n = min(kChunkCells, npos - p0);
    float4 qa[G], qb[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qr = q + ((size_t) t * H + kvh * G + g) * HD + lane * 8;
        qa[g] = *reinterpret_cast<const float4*>(qr);
        qb[g] = *reinterpret_cast<const float4*>(qr + 4);
    }
    const float* kb = kc + ((size_t) kvh * ctx + p0) * HD + lane * 8;
    const uint8_t* kb8 = reinterpret_cast<const uint8_t*>(kc) + ((size_t) kvh * ctx + p0) * kKvRowQ8;
#pragma unroll 2
    for (int w = 0; w < 32; ++w) {
        const int p = warp * 32 + w;
        float a[G];
#pragma unroll
        for (int g = 0; g < G; ++g) a[g] = 0.f;
        if (p < n) {
            float4 k0, k1;
            if constexpr (Q8) {   // this lane's 8 int8 of the row and their block's scale
                const uint8_t* kr = kb8 + (size_t) p * kKvRowQ8;
                const float sc = __half2float(reinterpret_cast<const __half*>(kr)[lane >> 2]);
                const uint2 raw = *reinterpret_cast<const uint2*>(kr + 16 + lane * 8);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
                k0 = make_float4(b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc);
                k1 = make_float4(b[4] * sc, b[5] * sc, b[6] * sc, b[7] * sc);
            } else {
                const float* kr = kb + (size_t) p * HD;
                k0 = *reinterpret_cast<const float4*>(kr);
                k1 = *reinterpret_cast<const float4*>(kr + 4);
            }
#pragma unroll
            for (int g = 0; g < G; ++g)
                a[g] = qa[g].x * k0.x + qa[g].y * k0.y + qa[g].z * k0.z + qa[g].w * k0.w + qb[g].x * k1.x + qb[g].y * k1.y +
                       qb[g].z * k1.z + qb[g].w * k1.w;
        }
#pragma unroll
        for (int g = 0; g < G; ++g)
            for (int m = 16; m > 0; m >>= 1) a[g] += __shfl_xor_sync(0xffffffffu, a[g], m);
        if (lane == 0)
#pragma unroll
            for (int g = 0; g < G; ++g) sc[g][p] = p < n ? a[g] * scale : -INFINITY;
    }
    __syncthreads();
    float mx[G], ls[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float s = sc[g][j];
        mx[g] = block_max(s, sh);
        const float e = j < n ? expf(s - mx[g]) : 0.f;
        sc[g][j] = e;
        ls[g] = block_sum(e, sh);
    }
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    const uint8_t* vp8 = reinterpret_cast<const uint8_t*>(vc) + ((size_t) kvh * ctx + p0) * kKvRowQ8;
    for (int k = 0; k < n; ++k) {
        float v;
        if constexpr (Q8) v = kv_q8_load(vp8 + (size_t) k * kKvRowQ8, j);
        else v = vp[(size_t) k * HD];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] += sc[g][k] * v;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int row = (t - t0) * H + kvh * G + g;
        const size_t o = (size_t) row * n_chunk + c;
        pacc[o * HD + j] = acc[g];
        if (j == 0) {
            pm[o] = mx[g];
            pl[o] = ls[g];
        }
    }
}
/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.  Each warp scores 32
/// cells, a cell's K row read as 32 x 8 floats across the lanes (coalesced) and the dot product reduced with
/// shuffles (STRATA_Q36_ATTN=v2; not bit-identical to v1's sum).
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256;
    __shared__ float sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    const int n = min(kChunkCells, npos - p0);
    {   // this lane's 8 of the query's 256 dims against the same 8 of each cell's K row
        const float* qr = q + ((size_t) t * H + h) * HD + lane * 8;
        const float4 q0 = *reinterpret_cast<const float4*>(qr), q1 = *reinterpret_cast<const float4*>(qr + 4);
        const float* kb = kc + ((size_t) kvh * ctx + p0) * HD + lane * 8;
#pragma unroll 4
        for (int w = 0; w < 32; ++w) {
            const int p = warp * 32 + w;
            float a = 0.f;
            if (p < n) {
                const float* kr = kb + (size_t) p * HD;
                const float4 k0 = *reinterpret_cast<const float4*>(kr), k1 = *reinterpret_cast<const float4*>(kr + 4);
                a = q0.x * k0.x + q0.y * k0.y + q0.z * k0.z + q0.w * k0.w + q1.x * k1.x + q1.y * k1.y + q1.z * k1.z + q1.w * k1.w;
            }
            for (int m = 16; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
            if (lane == 0) sc[p] = p < n ? a * scale : -INFINITY;
        }
    }
    __syncthreads();
    const float s = sc[j];
    const float m = block_max(s, sh);
    const float e = j < n ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    float acc = 0.f;
    const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).
__global__ void k_kv_f16(const float* kc, const float* vc, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    k16[i] = __float2half(kc[src]);
    v16[i] = __float2half(vc[src]);
}
/// The same from the 8-bit cache.
__global__ void k_kv_f16_q8(const uint8_t* kc, const uint8_t* vc, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t cell = (size_t) kvh * ctx + (size_t) (rest / HD);
    const int d = (int) (rest % HD);
    k16[i] = __float2half(kv_q8_load(kc + cell * kKvRowQ8, d));
    v16[i] = __float2half(kv_q8_load(vc + cell * kKvRowQ8, d));
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// The prompt's causal softmax over ONE key chunk (flash-style): row (h, t) of S (ld floats, n cells starting at
/// cell c0) -> exp(s * scale - m_new) as FP16 for the visible cells (< pos_base + t + 1), 0 past them; m / l per
/// (t, h) carried across the chunks (m_new = max(m_old, the chunk's max)), alpha = exp(m_old - m_new) rescales the
/// accumulator.  A chunk with no visible cell for the row writes zeros and leaves m / l alone.  grid (nt, H), block 256.
__global__ void k_attn_softmax_chunk(const float* S, __half* S16, int ld, int nt, int n, int c0, int pos_base,
                                     float scale, int G, float* m_run, float* l_run, float* alpha) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y, H = gridDim.y;
    const size_t r = (size_t) (h / G) * nt * G + (size_t) t * G + h % G;   // the grouped row order (kvh, t, g)
    const float* row = S + r * ld;
    __half* out = S16 + r * ld;
    const int idx = t * H + h;
    const int nv = min(n, max(0, pos_base + t + 1 - c0));   // this row's visible cells of the chunk
    if (nv == 0) {
        for (int p = threadIdx.x; p < n; p += blockDim.x) out[p] = __float2half(0.f);
        if (threadIdx.x == 0) alpha[idx] = 1.f;
        return;
    }
    const float mo = m_run[idx], lo = l_run[idx];
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < nv; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    const float mn = fmaxf(mo, mx);
    float sum = 0.f;
    for (int p = threadIdx.x; p < nv; p += blockDim.x) sum += expf(row[p] * scale - mn);
    sum = block_sum(sum, sh);
    for (int p = threadIdx.x; p < n; p += blockDim.x) out[p] = __float2half(p < nv ? expf(row[p] * scale - mn) : 0.f);
    if (threadIdx.x == 0) {
        const float a = mo == -INFINITY ? 0.f : expf(mo - mn);
        m_run[idx] = mn;
        l_run[idx] = lo * a + sum;
        alpha[idx] = a;
    }
}
/// The slice's queries regrouped by kv head for one GEMM per kv head: qg[(kvh * nt + t) * G + g][:] =
/// q16[(t0 + t) * H + kvh * G + g][:] (then K is read once for the G query heads, not once per head as a batched
/// GEMM over the heads does), n = nt * H * HD values.
__global__ void k_q_group(const __half* q16, __half* qg, int t0, int nt, int H, int G, int HD, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = (int) (i % HD), r = (int) (i / HD);   // r = (kvh * nt + t) * G + g
    const int g = r % G, t = (r / G) % nt, kvh = r / (G * nt);
    qg[i] = q16[((size_t) (t0 + t) * H + kvh * G + g) * HD + d];
}
/// acc[t][h][:] = acc * alpha[t][h] + part[(kvh * nt + t) * G + g][:] (the chunk's P V product in the grouped
/// order, h = kvh * G + g), n = nt * H * HD values.
__global__ void k_attn_acc(float* acc, const float* part, const float* alpha, int nt, int H, int G, int HD, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = (int) (i % HD), t = (int) (i / ((int64_t) H * HD)), h = (int) ((i / HD) % H);
    const int kvh = h / G, g = h % G;
    acc[i] = acc[i] * alpha[t * H + h] + part[((size_t) (kvh * nt + t) * G + g) * HD + d];
}
/// att[t][h][:] = acc / l[t][h] * sigmoid(gate) (the softmax's denominator and the output gate), n values.
__global__ void k_attn_fin(const float* acc, const float* l, const float* gate, float* att, int H, int HD, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = (int) (i / ((int64_t) H * HD)), h = (int) ((i / HD) % H);
    att[i] = acc[i] / l[t * H + h] * sigm(gate[i]);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x[i] = v for n values.
__global__ void k_fill(float* x, float v, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}
/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write) {
    __shared__ float part[4];
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write) return;   // a verify window: the history is commit()'s to move
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write) {
    constexpr int S = 128, RG = 4, RPG = 32;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
        __syncthreads();
        if (rg == 0) {
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) t * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;   // (unused since the table moved to the host: see embd_host)
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<uint8_t> embd_host;  // the token-embedding table in host memory: a batch's rows are staged per card
    std::vector<int> dev_of;         // the device of each layer's weights, state and experts (index n_layers: the MTP block's)
    int last_dev = 0;                // the device of the MTP block, the output head and the logits
};

size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

size_t stage_batch_bytes(const strata::core::ModelGeometry& g, int64_t top_k, bool use_mmq, size_t T);
constexpr size_t kMmqTail = 4096;

/// How the layers are shared between two cards (--layer-split): `k` the first layer of the second card (0: one
/// card), or `automatic`: the largest trailing block of layers whose experts the second card can hold entirely
/// (so it never streams them: it may sit on a slow slot), beside their dense weights, the KV cache of `ctx` cells,
/// the head, the MTP block and the chunk's buffers.
struct SplitSpec {
    int k = 0;
    bool automatic = false;
    int dev = 1, n_dev = 1;
    int ctx = 32768, tmax_est = 2048;
    bool small = false, kv_q8 = false;
};

/// The split point for `automatic` (see SplitSpec), priced with the file's tensor sizes; 0 when nothing fits.
int plan_split(const strata::GgufModel& gm, const Model& m, const SplitSpec& sp) {
    const auto& g = m.g;
    const bool packed = K::native_q8_0_packed_enabled();
    auto bytes_of = [&](const std::string& name) -> size_t {
        size_t s = 0;
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) return 0;
        size_t b = (size_t) strata::tensor_payload_bytes(*t);
        if (packed && t->type == 8 && t->shape.size() >= 2 &&
            K::native_q8_0_packed_eligible((int) t->shape[0], (int) t->shape[1]))
            b *= 2;   // the packed decode copy beside the GGUF blocks
        return b;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    const size_t kv_layer = 2 * (size_t) g.n_head_kv * (size_t) sp.ctx * (sp.kv_q8 ? (size_t) kKvRowQ8 : (size_t) g.head_dim * 4);
    const size_t gdn_state = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 +
                             (size_t) g.ssm_conv_channels * 3 * 4 + 8 * (2 * (size_t) g.ssm_conv_channels + 2 * (size_t) g.ssm_v_heads) * 4;
    auto layer_bytes = [&](int64_t l, bool attn) -> size_t {
        size_t b = 0;
        for (const char* n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) b += bytes_of(L(l, n));
        b += (size_t) g.n_expert * 2048 + kMmqTail;   // the blobs' stride rounding, the tail
        for (const char* n : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight", "attn_k.weight",
                              "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                              "ffn_down_shexp.weight", "ffn_gate_inp.weight"})
            b += bytes_of(L(l, n));
        b += (size_t) (g.n_expert + 1) * g.n_embd * 4;   // router_ext
        b += attn ? kv_layer : gdn_state;
        return b;
    };
    size_t fr = 0, tot = 0;
    {
        strata::core::OnDevice od(sp.dev);
        CK(cudaMemGetInfo(&fr, &tot));
    }
    const size_t n_chunk = ((size_t) sp.ctx + kChunkCells - 1) / kChunkCells;
    size_t fixed = (size_t) 768 << 20;                                                   // the reserve
    fixed += stage_batch_bytes(g, m.top_k, true, (size_t) sp.tmax_est);                  // the chunk's buffers
    fixed += (size_t) (sp.small ? 16 << 20 : 64 << 20) * 6;                              // the attention scores
    fixed += 2 * (size_t) g.n_head_kv * (size_t) sp.ctx * (size_t) g.head_dim * 2;       // k16 / v16
    fixed += (size_t) g.n_head * 8 * n_chunk * ((size_t) g.head_dim * 4 + 8);            // pacc, pm, pl
    fixed += (size_t) 64 << 20;                                                          // the GEMM's workspace, scratch
    fixed += (size_t) 9 * g.n_vocab * 4;                                                 // logits, logits_m
    fixed += bytes_of("output.weight");                                                  // the head
    if (g.n_mtp_layers > 0)
        fixed += layer_bytes(g.n_layers, true) + bytes_of(L(g.n_layers, "nextn.eh_proj.weight")) + 4 * (size_t) g.n_embd * 4;
    fixed += (size_t) 160 << 20;                                                         // headroom
    if (fr <= fixed) {
        std::fprintf(stderr, "strata-q36: layer split: CUDA%d has %.2f GiB free, the fixed buffers alone need %.2f: one card\n",
                     sp.dev, (double) fr / (1 << 30), (double) fixed / (1 << 30));
        return 0;
    }
    size_t room = fr - fixed, used = 0;
    int count = 0;
    for (int64_t l = g.n_layers - 1; l >= 1; --l) {
        const size_t b = layer_bytes(l, strata::core::is_qsa_layer(g, l));
        if (b > room) break;
        room -= b;
        used += b;
        ++count;
    }
    if (count == 0) {
        std::fprintf(stderr, "strata-q36: layer split: CUDA%d has no room for a whole layer beside its buffers: one card\n", sp.dev);
        return 0;
    }
    std::fprintf(stderr, "strata-q36: layer split (auto): the last %d layers' experts, weights and state (%.2f GiB) on CUDA%d, "
                 "every one of their experts resident; %.2f GiB left there for the buffers and %.2f GiB to spare\n",
                 count, (double) used / (1 << 30), sp.dev, (double) fixed / (1 << 30), (double) room / (1 << 30));
    return (int) g.n_layers - count;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp, const SplitSpec& sp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15)";
        return false;
    }
    // the layer split: the device of every layer (and of the MTP block, the head and the logits: the last one)
    m.dev_of.assign((size_t) g.n_layers + 1, 0);
    m.last_dev = 0;
    {
        int k = sp.automatic ? plan_split(gm, m, sp) : sp.k;
        if (k != 0 && (sp.n_dev < 2 || sp.dev < 1 || sp.dev >= sp.n_dev)) {
            std::fprintf(stderr, "strata-q36: --layer-split needs a second visible GPU (CUDA_VISIBLE_DEVICES, --gpu 0,1): one card\n");
            k = 0;
        }
        if (k < 0 || k >= g.n_layers) {
            std::fprintf(stderr, "strata-q36: --layer-split %d is not a layer of 1 .. %lld: one card\n", k, (long long) g.n_layers - 1);
            k = 0;
        }
        if (k > 0) {
            for (int64_t l = k; l <= g.n_layers; ++l) m.dev_of[(size_t) l] = sp.dev;
            m.last_dev = sp.dev;
            std::fprintf(stderr, "strata-q36: layer split: layers 0-%d on CUDA0; layers %d-%lld%s and the output head on CUDA%d "
                         "(the residual crosses once per batch through pinned host memory)\n", k - 1, k, (long long) g.n_layers - 1,
                         g.n_mtp_layers > 0 ? ", the MTP block" : "", sp.dev);
        }
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        CK(cudaSetDevice(m.dev_of[(size_t) l]));   // the layer's weights on its card
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    CK(cudaSetDevice(m.last_dev));   // the MTP block's projections, the output norm and the head: the last card
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        // the table stays in host memory (515 MiB of VRAM on Qwen3.6): a batch's rows are gathered into pinned
        // staging and uploaded by the stage that embeds them (Stage::embed_host / embed_dev)
        const uint8_t* src = gm.shard(s).tensor_data(*t);
        m.embd_host.assign(src, src + nb);
        m.embd = nullptr;
    }
    CK(cudaSetDevice(m.dev_of[0]));
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU%s in %.1f s (the routed "
                 "experts are placed after the engine's buffers; the token table in host memory)\n", path.c_str(), checked,
                 (double) bytes / (1 << 30), m.last_dev != m.dev_of[0] ? "s" : "", ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
using strata::core::OnDevice;

/// The device bytes Stage::init allocates for a chunk of T rows (the auto layer split prices a card's buffers with
/// it before anything is uploaded; kept beside init so the two agree: the tile-padded MMQ and scratch buffers are
/// sized once per batch, as init sizes them).
size_t stage_batch_bytes(const strata::core::ModelGeometry& g, int64_t top_k, bool use_mmq, size_t T) {
    const size_t N = (size_t) g.n_embd, Kp = (size_t) top_k, FF = (size_t) g.n_ff, C = (size_t) g.ssm_conv_channels;
    const size_t HD = (size_t) g.head_dim, H = (size_t) g.n_head, HKV = (size_t) g.n_head_kv;
    size_t b = 0;
    b += T * (4 * N * 4 + N * 4);                                           // x, h, y, hmain; moe_out
    b += T * (2 * C * 4 + 2 * (size_t) g.ssm_value_dim * 4);                // qkv, hc, z, gdn_y
    b += T * 2 * (size_t) g.ssm_v_heads * 4;                                // alpha, beta
    b += T * (2 * H * HD * 4 + 2 * HKV * HD * 4 + 3 * H * HD * 4);          // qg, kx, vx, q, gate, att
    b += T * ((size_t) (g.n_expert + 1) * 4 + Kp * N * 4 + N * 4);          // logits_r, eparts, sy
    b += T * (3 * (size_t) g.n_ff_shexp * 4 + 4 + Kp * 4);                  // sg, su, shh, sgate, ewts
    const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
    b += (K::native_q8_1_bytes(max_in, 1) + K::native_q8_1_bytes((int) N, 1) + (size_t) max_in * 2) * T;   // q8, xq, x16
    b += T * (16 * (size_t) (g.n_expert + 1) * 4 + H * HD * 2);             // sk_part, q16
    const int64_t ne = (int64_t) (T * Kp);
    if (use_mmq) b += MMQ::q8_bytes(ne, g.n_embd) + MMQ::q8_bytes(ne, g.n_ff) + (size_t) ne * (2 * FF * 4 + FF * 4) + (size_t) ne * 8;
    b += K::native_expert_scratch_bytes(ne, (int64_t) FF) + (size_t) ne * 4 * 3 + T * 4;   // escr, ids, ent_*, tok
    b += T * 3 * N * 4;                                                     // mtp_cat, hmtp (the last stage)
    return b;
}

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

struct Engine;

/// One card's share of the model: the layers [lb, le) (and on the last stage the MTP block, the output head and the
/// logits), with every buffer, stream, event and handle on its device.  The single-device engine of 0.1.44 less the
/// orchestration: Engine (below) drives the stages in layer order and hands the residual over between them through
/// pinned host memory (no peer-to-peer needed).
struct Stage {
    const Model* m = nullptr;
    int dev = 0, lb = 0, le = 0;
    bool first = true, last = true;
    bool kv_q8 = false;   // the KV cache as 8-bit rows (kKvRowQ8) instead of fp32
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits = nullptr;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds; last stage)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first)
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S, kc, vc;   // per layer (global index; null off this stage); the MTP block's K / V at n_layers
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (the last stage, when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats on a small card, or when the experts' fit is close)
    // the prompt attention over a long prefix, flash-style: the keys in chunks of kAttKeyChunk cells so that the
    // query slices stay wide (0.1.44 sliced the queries so that a slice's scores over the WHOLE prefix fit the
    // buffer: 10 rows a GEMM at 100K cells); the running max / sum per (row, head), the rescale factor, and the
    // chunk's P V product before it is folded into the accumulator (`att`)
    static constexpr int kAttKeyChunk = 16384;
    int att_tq_chunk = 0;              // query rows a slice takes on the chunked path
    float *att_m = nullptr, *att_l = nullptr, *att_alpha = nullptr, *att_part = nullptr;
    __half* att_qg = nullptr;          // the slice's queries grouped by kv head
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    // the token embeddings: the table stays in host memory (Model::embd_host); a batch's rows are gathered into a
    // pinned staging buffer (free again whenever a batch starts: every batch begins with a stream sync), uploaded,
    // and read by iq_embed_rows through an identity index
    uint8_t* embd_pin = nullptr;
    uint8_t* embd_d = nullptr;
    int32_t* iota_d = nullptr;
    // the residual handed over from the previous stage (pinned, portable: written by that card's copy engine)
    float* hand_in = nullptr;
    // the hybrid decode in graphs (Engine::forward_graphs): per batch size T (1: a decode step; 2 .. kVerify: a
    // verify window) and per layer, graph A (the previous layer's merge and combine when that layer ran on the pool,
    // this layer's mixer, router and routing copies) and graph B (its resident and shared experts); a layer with
    // every expert resident is graph A alone; the stage's tail (the last merge and combine, then the head or the
    // hand-off) is one more.  ~70 graph launches per token instead of ~700 kernel launches.
    struct LayerGraphs {
        std::vector<cudaGraphExec_t> a, b;
        cudaGraphExec_t tail = nullptr;
    };
    LayerGraphs lgr[2][kVerify + 1];   // [verify][rows]: a verify window's GDN kernels leave the state alone
    template <class F> void run_graph(cudaGraphExec_t& gx, F body) {
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }
    // sampled requests: the vocabulary's logits sorted on the device (a radix sort), the top candidates copied to
    // the host - the host sampler keeps only its top-k anyway (0.1.44 copied the 248,320 logits and partial-sorted
    // them on the host for every token)
    static constexpr int kTopMax = 256;
    float* sort_v = nullptr;
    int32_t *sort_i = nullptr, *iota_v = nullptr;
    void* sort_tmp = nullptr;
    size_t sort_tmp_bytes = 0;
    float* h_topv = nullptr;
    int32_t* h_topi = nullptr;

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    int hyT = 0;                // rows the hybrid buffers hold (hybrid_t, or tmax when every batch is a hybrid batch)
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool warming = false;
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
    /// state index li is this stage's (its layers, and on the last stage the MTP block)
    bool mine(size_t li) const { return li < m->layers.size() ? (int) li >= lb && (int) li < le : last && m->has_mtp; }
#if defined(STRATA_Q36_CPU_POOL)
    strata::kernels::cpu::ExpertPool* pool = nullptr;   // shared by the stages (they run one after the other)
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
    std::vector<char> quantized;   // per row of the batch: its activation quantized for the pool
    std::vector<int> open_of;      // per expert: its job with room, or -1
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }

    /// The buffers of this stage: its layers' state, the chunk's activations, the prompt buffers; `small`: a card
    /// with little VRAM (the attention scores buffers at 64 + 32 MB instead of 256 + 128).
    void init(const Model& model, int ctx_, int tmax_, int dev_, int lb_, int le_, bool last_, bool small, bool kvq8) {
        m = &model;
        kv_q8 = kvq8;
        dev = dev_;
        lb = lb_;
        le = le_;
        first = lb == 0;
        last = last_;
        OnDevice od(dev);
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        if (small && std::getenv("STRATA_Q36_BIG_ATT") == nullptr) kAttScoreFloats = (size_t) 16 << 20;   // (STRATA_Q36_BIG_ATT=1: the 256 MB buffers, for A/B runs)
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N); hlast = dalloc(N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp);
        if (last) logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        att_tq_chunk = (int) std::max<size_t>(1, std::min<size_t>(T, kAttScoreFloats / ((size_t) g.n_head * kAttKeyChunk)));
        att_m = dalloc(T * (size_t) g.n_head);
        att_l = dalloc(T * (size_t) g.n_head);
        att_alpha = dalloc(T * (size_t) g.n_head);
        att_part = dalloc((size_t) att_tq_chunk * g.n_head * g.head_dim);
        att_qg = dalloc_t<__half>((size_t) att_tq_chunk * g.n_head * g.head_dim);
        k16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        v16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        if (first) std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (int l = lb; l < le; ++l) take(m->layers[(size_t) l]);
            if (last && m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocPortable));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        // the split-K decode attention runs for batches of <= kGemmT (8) rows only (a chunk takes the GEMMs), so its
        // partials hold kVerify query rows per head, not 64 (0.1.44: 4 KiB per context cell, now 512 B)
        att_rows = (int) (g.n_head * kVerify);
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        const size_t nl = m->layers.size() + 1;
        conv.assign(nl, nullptr); S.assign(nl, nullptr); kc.assign(nl, nullptr); vc.assign(nl, nullptr);
        for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->assign(nl, nullptr);
        for (int l = lb; l < le; ++l) {
            const Layer& L = m->layers[(size_t) l];
            if (L.attn) {
                kc[(size_t) l] = kv_alloc();
                vc[(size_t) l] = kv_alloc();
            } else {
                conv[(size_t) l] = dalloc(C * (g.ssm_d_conv - 1));
                S[(size_t) l] = dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size);
                v_qkv[(size_t) l] = dalloc((size_t) kVerify * C);
                v_hc[(size_t) l] = dalloc((size_t) kVerify * C);
                v_alpha[(size_t) l] = dalloc((size_t) kVerify * g.ssm_v_heads);
                v_beta[(size_t) l] = dalloc((size_t) kVerify * g.ssm_v_heads);
            }
        }
        if (last && m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            const size_t li = m->layers.size();
            kc[li] = kv_alloc();
            vc[li] = kv_alloc();
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        if (first || (last && m->has_mtp)) {   // the token rows this stage embeds (the prompt / decode rows, the MTP's)
            CK(cudaHostAlloc((void**) &embd_pin, T * m->embd_row, cudaHostAllocPortable));
            CK(cudaMalloc(&embd_d, T * m->embd_row));
            std::vector<int32_t> io(T);
            for (size_t i = 0; i < T; ++i) io[i] = (int32_t) i;
            iota_d = dalloc_t<int32_t>(T);
            CK(cudaMemcpy(iota_d, io.data(), T * 4, cudaMemcpyHostToDevice));
        }
        if (!first) CK(cudaHostAlloc((void**) &hand_in, T * N * 4, cudaHostAllocPortable));
#if STRATA_Q36_DEVICE_TOPK
        if (last) {   // the sampled-request sort (see topk)
            const size_t V = (size_t) g.n_vocab;
            sort_v = dalloc(V);
            sort_i = dalloc_t<int32_t>(V);
            iota_v = dalloc_t<int32_t>(V);
            std::vector<int32_t> io(V);
            for (size_t i = 0; i < V; ++i) io[i] = (int32_t) i;
            CK(cudaMemcpy(iota_v, io.data(), V * 4, cudaMemcpyHostToDevice));
            CK(cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_tmp_bytes, (const float*) nullptr, sort_v,
                                                         (const int32_t*) nullptr, sort_i, (int) V, 0, 32, st));
            CK(cudaMalloc(&sort_tmp, sort_tmp_bytes));
            CK(cudaHostAlloc((void**) &h_topv, kTopMax * sizeof(float), cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_topi, kTopMax * sizeof(int32_t), cudaHostAllocPortable));
        }
#endif
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: CUDA%d: layers %d-%d%s, context %d cells (%s K/V), prompt chunk %d, %.2f GiB of VRAM free\n",
                     dev, lb, le - 1, last && m->has_mtp ? " + the MTP block" : "", ctx, kv_q8 ? "8-bit" : "fp32", tmax,
                     (double) fr / (1 << 30));
    }

    /// One attention layer's K (or V) cache: HKV x ctx cells, fp32 rows or 8-bit rows.
    float* kv_alloc() {
        const auto& g = m->g;
        if (!kv_q8) return dalloc((size_t) g.n_head_kv * ctx * g.head_dim);
        return reinterpret_cast<float*>(dalloc_t<uint8_t>((size_t) g.n_head_kv * ctx * kKvRowQ8));
    }
    /// The recurrent state and the cache to a fresh sequence (this stage's layers).
    void reset() {
        OnDevice od(dev);
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Token ids (host) -> their embedding rows in the pinned staging buffer.
    void embed_host(const int32_t* toks, int T) {
        uint8_t* dst = embd_pin;
        for (int t = 0; t < T; ++t)
            std::memcpy(dst + (size_t) t * m->embd_row, m->embd_host.data() + (size_t) toks[t] * m->embd_row, m->embd_row);
    }
    /// The staged rows uploaded and dequantized into `out` [T][N] (after embed_host).
    void embed_dev(int T, float* out) {
        CK(cudaMemcpyAsync(embd_d, embd_pin, (size_t) T * m->embd_row, cudaMemcpyHostToDevice, st));
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, iota_d, T, (int64_t) m->g.n_embd, out, st);
    }

    /// The k largest of the logits row `lg` and their ids into h_topv / h_topi (synced).
    void topk(const float* lg, int k) {
#if STRATA_Q36_DEVICE_TOPK
        size_t tmp = sort_tmp_bytes;
        CK(cub::DeviceRadixSort::SortPairsDescending(sort_tmp, tmp, lg, sort_v, (const int32_t*) iota_v, sort_i,
                                                     (int) m->g.n_vocab, 0, 32, st));
        CK(cudaMemcpyAsync(h_topv, sort_v, (size_t) k * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_topi, sort_i, (size_t) k * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
#else
        (void) lg;
        (void) k;
#endif
    }

    /// logits [n][V] = W_out hn rows, and each row's greedy pick (the last stage).
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs), for this stage's layers.
    void commit(int n_keep) {
        OnDevice od(dev);
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (int l = lb; l < le; ++l) {
            const Layer& L = m->layers[(size_t) l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[(size_t) l], v_qkv[(size_t) l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[(size_t) l], v_hc[(size_t) l], C, v_alpha[(size_t) l], v_beta[(size_t) l], z,
                                                   L.ssm_norm, m->eps, nullptr, HK, HV, n_keep, true);
        }
    }

    void shrink_att(size_t floats) {
        OnDevice od(dev);
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// The first part of a layer for T rows: x (the residual) += mixer(h), the post norm, the router and the top-k
    /// (nothing in it needs the host).  `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the
    /// same on the host (eager grids); `verify`: a verify window - the GDN state is read, not written (commit()
    /// writes it for the kept rows).
    void layer_mix(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep(h, N, T, xq);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
            // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
            // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
            // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
            // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
            if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                  be_b, gdn_y, HK, HV, T);
                k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
            } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                       T, !verify);
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep(h, N, T, xq);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            k_attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, kc[li], vc[li], H,
                                                              HKV, HD, m->n_rot, dpos, ctx, m->rope_base, m->eps, kv_q8);
            if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                if (kv_q8)
                    k_kv_f16_q8<<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(
                        reinterpret_cast<const uint8_t*>(kc[li]), reinterpret_cast<const uint8_t*>(vc[li]), k16, v16, ctx, P, HD, HKV);
                else
                    k_kv_f16<<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, ctx, P, HD, HKV);
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                static const bool chunked = std::getenv("STRATA_Q36_PROMPT_KEY_CHUNK_OFF") == nullptr;
                if (chunked && P > kAttKeyChunk) {
                    // the keys in chunks: wide query slices, the softmax carried across the chunks (online), the
                    // chunk's P V product folded into the accumulator with the rescale factor
                    const int tqc = att_tq_chunk;
                    const float scale = 1.0f / sqrtf((float) HD);
                    for (int t0 = 0; t0 < T; t0 += tqc) {
                        const int nt = std::min(tqc, T - t0);
                        const int last_vis = (int) p0 + t0 + nt;   // cells < last_vis are visible to some row of the slice
                        float* Oacc = att + ((size_t) t0 * H) * HD;
                        CK(cudaMemsetAsync(Oacc, 0, (size_t) nt * H * HD * 4, st));
                        CK(cudaMemsetAsync(att_l, 0, (size_t) nt * H * 4, st));
                        k_fill<<<grid1((int64_t) nt * H), kThreads, 0, st>>>(att_m, -INFINITY, (int64_t) nt * H);
                        k_q_group<<<grid1((int64_t) nt * H * HD), kThreads, 0, st>>>(q16, att_qg, t0, nt, H, G, HD,
                                                                                     (int64_t) nt * H * HD);
                        for (int c0 = 0; c0 < last_vis; c0 += kAttKeyChunk) {
                            const int n = std::min(kAttKeyChunk, P - c0);
                            const int ldc = (n + 7) / 8 * 8;
                            const int rows = nt * G;   // a kv head's query rows: (t, g)
                            for (int kvh = 0; kvh < HKV; ++kvh) {   // scores [rows][ldc] = Qg . K^T, K read once for the group
                                const __half* Kh = k16 + ((size_t) kvh * P + c0) * HD;
                                const __half* Qg = att_qg + (size_t) kvh * rows * HD;
                                float* Sh = att_s + (size_t) kvh * rows * ldc;
                                if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, HD, &one, Kh, CUDA_R_16F, HD, Qg, CUDA_R_16F,
                                                 HD, &zero, Sh, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                                    CUBLAS_STATUS_SUCCESS) {
                                    std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                                    std::exit(1);
                                }
                            }
                            k_attn_softmax_chunk<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldc, nt, n, c0, (int) p0 + t0,
                                                                                   scale, G, att_m, att_l, att_alpha);
                            for (int kvh = 0; kvh < HKV; ++kvh) {   // part [rows][HD] = P . V, V read once for the group
                                const __half* Vh = v16 + ((size_t) kvh * P + c0) * HD;
                                const __half* Sh = att_p16 + (size_t) kvh * rows * ldc;
                                float* Oh = att_part + (size_t) kvh * rows * HD;
                                if (cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, rows, n, &one, Vh, CUDA_R_16F, HD, Sh, CUDA_R_16F,
                                                 ldc, &zero, Oh, CUDA_R_32F, HD, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                                    CUBLAS_STATUS_SUCCESS) {
                                    std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                                    std::exit(1);
                                }
                            }
                            k_attn_acc<<<grid1((int64_t) nt * H * HD), kThreads, 0, st>>>(Oacc, att_part, att_alpha, nt, H, G, HD,
                                                                                         (int64_t) nt * H * HD);
                        }
                        k_attn_fin<<<grid1((int64_t) nt * H * HD), kThreads, 0, st>>>(Oacc, att_l, gate + ((size_t) t0 * H) * HD, Oacc,
                                                                                     H, HD, (int64_t) nt * H * HD);
                    }
                } else
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = k16 + (size_t) kvh * P * HD;
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = v16 + (size_t) kvh * P * HD;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                if (!(chunked && P > kAttKeyChunk))   // (the chunked path applied the gate in k_attn_fin)
                    k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    static const int attn_v = [] {   // STRATA_Q36_ATTN=v1 (0.1.44's kernel) | v2 (coalesced) | v3 (default)
                        const char* e = std::getenv("STRATA_Q36_ATTN");
                        return e && e[0] == 'v' ? std::atoi(e + 1) : 3;
                    }();
                    if (kv_q8)   // (the 8-bit cache: the shared-head kernel only; init checks the head counts)
                        k_attn_part_gqa<true><<<dim3(nc, nt * HKV), kChunkCells, 0, st>>>(q, kc[li], vc[li], dpos, t0, H, HKV, ctx,
                                                                                         1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl);
                    else if (attn_v == 1)
                        k_attn_part_v1<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, kc[li], vc[li], dpos, t0, H, HKV, ctx,
                                                                                1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl);
                    else if (attn_v == 2 || H != 8 * HKV)
                        k_attn_part<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, kc[li], vc[li], dpos, t0, H, HKV, ctx,
                                                                             1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl);
                    else
                        k_attn_part_gqa<false><<<dim3(nc, nt * HKV), kChunkCells, 0, st>>>(q, kc[li], vc[li], dpos, t0, H, HKV, ctx,
                                                                                          1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl);
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att);
                }
            }
            mm(L.wo, att, y, T);
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        if (one) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, 1);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
    }
    /// The hybrid path's routing to the host: ids and h copied and (`record`) the event the host waits on recorded
    /// (a graph records it after its launch instead).
    void route_to_host(int T, bool record = true) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * Kp * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        if (record) CK(cudaEventRecord(ev_route, st));
    }
    /// The resident experts of a hybrid batch: the entries whose expert has a VRAM slot, grouped and run.
    void resident_experts(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        k_group_hy<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, (int) g.n_expert, hy[li].slot_of_d, L.blobs, L.stride,
                                           grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
        quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// Every routed expert in VRAM (no hybrid): the entries grouped per expert and run.
    void grouped_experts(const Layer& L, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        k_group<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, (int) g.n_expert, L.blobs, L.stride, grp_ptr, grp_start,
                                        n_groups, ent_dst, ent_tok);
        quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The shared expert, scaled later by sigmoid(w . h) (its gate: the router product's last row for one row).
    void shared_expert(const Layer& L, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, FS = (int) g.n_ff_shexp;
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        apply(L.sh_gate, xq, sg, T);
        apply(L.sh_up, xq, su, T);
        if (T <= kGemmT) {   // SwiGLU and its q8_1 image in one launch
            K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);
            mmq(L.sh_down, q8, sy, T);
        } else {
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (T != 1) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
    }
    /// The CPU pool's rows into the entries' rows (the hybrid path): uploaded and merged.
    void merge_cpu_rows(int T) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
    }
    /// The combine, the shared expert's gate, the residual add and the next norm in one launch (prow: the prompt's
    /// MMQ order of the parts, null otherwise).
    void combine(const Layer& L, int T, const int32_t* prow, const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        (void) L;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, sy, T == 1 ? logits_r + g.n_expert : sgate, 1, wn,
                                          hout, N, m->eps);
    }

    /// One layer for T rows, eagerly: the mixer and the router, the routed experts (a hybrid batch: the resident ones
    /// here, the others on the CPU pool meanwhile; a prompt chunk: llama.cpp's MMQ over the streamed or resident
    /// blobs; else the grouped kernels), the shared expert and the combine.  The kernel order of 0.1.44.
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff;
        layer_mix(L, li, T, p0, dpos, graph_mode, verify);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        if (hy_rows) {
            route_to_host(T);
            resident_experts(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            grouped_experts(L, T);
        }
        shared_expert(L, T);
        if (hy_rows && moe_hybrid_cpu(li, T)) merge_cpu_rows(T);   // the CPU pool's experts (while the GPU runs the rest)
        combine(L, T, prow, wn, hout);
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats); true
    /// when there were any (their rows wait in cpu_out for merge_cpu_rows).
    bool moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        // (no allocation per layer: the per-call tables are members, cleared for the rows and experts in play)
        if (quantized.size() < (size_t) T) quantized.resize((size_t) T);
        std::fill(quantized.begin(), quantized.begin() + T, 0);
        if (open_of.size() < (size_t) g.n_expert) open_of.resize((size_t) g.n_expert);
        std::fill(open_of.begin(), open_of.end(), -1);   // expert -> its job with room
        int n_miss = 0;
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            if (open_of[(size_t) x] < 0 || jobs[(size_t) open_of[(size_t) x]].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                open_of[(size_t) x] = (int) jobs.size() - 1;
            }
            C::ExpertJobMulti& j = jobs[(size_t) open_of[(size_t) x]];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return false;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        return true;
#else
        (void) li;
        (void) T;
        return false;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            if (!mine(li)) continue;   // another stage's layer
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) y] >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                       cudaMemcpyHostToDevice, st));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }

};   // Stage

/// The engine: the stages in layer order (one card each), the CPU expert pool they share, the position, the drafts
/// and the graphs of a single-stage engine.  A forward runs the stages one after the other; the residual crosses
/// from one card to the next through the next stage's pinned hand-off buffer (8 KiB per row), with a host sync.
struct Engine {
    const Model* m = nullptr;
    std::vector<std::unique_ptr<Stage>> stg;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    bool use_graph = true;
    bool hybrid_graph = true;   // the hybrid decode in per-layer graphs (--no-graph / STRATA_Q36_NO_LAYER_GRAPH=1: eager)
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    int hybrid_t = 32;
    bool warming = false;
    static constexpr int kVerify = Stage::kVerify;
    static constexpr int kRebalanceEvery = 32;
    int hybrid_swaps_after_prompt = 2048;
    // graphs: [0] the decode step, [T] a verify window of T rows (one stage only), mtp1 the MTP draft step
    cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
#endif

    Stage& first() { return *stg.front(); }
    Stage& last() { return *stg.back(); }
    Stage& stage_of(size_t li) {
        for (auto& s : stg)
            if (s->mine(li)) return *s;
        return last();
    }
    bool lhy(size_t li) { return stage_of(li).lhy(li); }
    bool use_mmq() { return first().use_mmq; }
    long long hy_cpu() { long long n = 0; for (auto& s : stg) n += s->hy_cpu; return n; }
    long long hy_gpu() { long long n = 0; for (auto& s : stg) n += s->hy_gpu; return n; }
    long long hy_swaps() { long long n = 0; for (auto& s : stg) n += s->hy_swaps; return n; }
    double hy_cpu_ms() { double n = 0; for (auto& s : stg) n += s->hy_cpu_ms; return n; }

    /// The stages from Model::dev_of (one per run of layers on the same device, in layer order).
    bool kv_q8 = false;
    void init(const Model& model, int ctx_, int tmax_, bool small, bool q8) {
        m = &model;
        kv_q8 = q8;
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        std::vector<std::pair<int, int>> runs;   // (device, first layer)
        for (size_t l = 0; l < m->layers.size(); ++l)
            if (l == 0 || m->dev_of[l] != m->dev_of[l - 1]) runs.push_back({m->dev_of[l], (int) l});
        for (size_t i = 0; i < runs.size(); ++i) {
            stg.push_back(std::make_unique<Stage>());
            const int le = i + 1 < runs.size() ? runs[i + 1].second : (int) m->layers.size();
            stg.back()->init(model, ctx, tmax, runs[i].first, runs[i].second, le, i + 1 == runs.size(), small, q8);
        }
        if (stg.size() > 1) use_graph = false;   // the hand-off's host sync cannot be captured
    }

    void sync_all() {
        for (auto& s : stg) {
            OnDevice od(s->dev);
            CK(cudaStreamSynchronize(s->st));
        }
    }
    /// The batch's first position on every stage (pos0, read by the attention kernels).
    void set_pos(int64_t p) {
        for (auto& s : stg) {
            OnDevice od(s->dev);
            s->hpin[1] = (int32_t) p;
            CK(cudaMemcpyAsync(s->pos0, s->hpin + 1, 4, cudaMemcpyHostToDevice, s->st));
        }
    }
    void set_deferred(bool v) { for (auto& s : stg) s->rows_deferred = v; }
    void set_warming(bool v) { warming = v; for (auto& s : stg) s->warming = v; }

    /// The main model over T rows (the rows' tokens staged by first().embed_host; positions p0 .. p0 + T - 1, also at
    /// every stage's *pos0): the last stage's hmain gets every row's final normed hidden; `heads` rows from the end
    /// get logits (rows [T - heads, T): logits row r - (T - heads)) and their greedy picks (`pick`).  `verify`: a
    /// verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const int N = (int) m->g.n_embd;
        if (!graph_mode && hybrid_graph && m->hybrid && (T == 1 || verify) && T <= kVerify && T <= hybrid_t) {
            forward_graphs(T, p0, heads, verify);
            return;
        }
        for (size_t i = 0; i < stg.size(); ++i) {
            Stage& s = *stg[i];
            OnDevice od(s.dev);
            if (s.main_hybrid && T > s.hybrid_t && s.use_mmq) {
                std::vector<int> seq;
                for (int l = s.lb; l < s.le; ++l)
                    if (s.lhy((size_t) l)) seq.push_back(l);
                if (!seq.empty()) s.stream_begin(seq);
            }
            if (s.first) s.embed_dev(T, s.x);
            else CK(cudaMemcpyAsync(s.x, s.hand_in, (size_t) T * N * 4, cudaMemcpyHostToDevice, s.st));
            // (the fused norm at the end of k_moe_out and k_rms accumulate the same way: a stage's first norm from the
            // handed-over residual is the one the previous card would have fused)
            k_rms<<<T, kThreads, 0, s.st>>>(s.x, m->layers[(size_t) s.lb].attn_norm, s.h, N, m->eps);
            for (int l = s.lb; l < s.le; ++l) {
                const bool last_layer = (size_t) l + 1 == m->layers.size();
                const float* wn = last_layer ? m->out_norm : (l + 1 < s.le ? m->layers[(size_t) l + 1].attn_norm : nullptr);
                s.layer(m->layers[(size_t) l], (size_t) l, T, p0, s.pos0, graph_mode, verify, wn, last_layer ? s.hmain : s.h);
            }
            if (!s.last) {
                Stage& nx = *stg[i + 1];
                CK(cudaMemcpyAsync(nx.hand_in, s.x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, s.st));
                CK(cudaStreamSynchronize(s.st));
            } else if (heads > 0) {
                s.head_rows(s.hmain + (size_t) (T - heads) * N, heads, s.logits, s.pick);
            }
        }
    }

    /// The hybrid decode step (T == 1) or verify window in per-layer graphs: the same kernels in the same order as
    /// the eager forward (Stage::layer), the host taking part between a layer's graph A and graph B (Stage::LayerGraphs).
    void forward_graphs(int T, int64_t p0, int heads, bool verify) {
        const int N = (int) m->g.n_embd;
        for (size_t i = 0; i < stg.size(); ++i) {
            Stage& s = *stg[i];
            OnDevice od(s.dev);
            Stage::LayerGraphs& G = s.lgr[verify ? 1 : 0][T];
            if (G.a.empty()) {
                G.a.assign(m->layers.size() + 1, nullptr);
                G.b.assign(m->layers.size() + 1, nullptr);
            }
            // a layer's merge and combine run at the start of the next graph (after the host's pool work)
            auto pending = [&](int l) {
                const Layer& P = m->layers[(size_t) l];
                const bool last_layer = (size_t) l + 1 == m->layers.size();
                const float* wn = last_layer ? m->out_norm : (l + 1 < s.le ? m->layers[(size_t) l + 1].attn_norm : nullptr);
                s.merge_cpu_rows(T);
                s.combine(P, T, nullptr, wn, last_layer ? s.hmain : s.h);
            };
            for (int l = s.lb; l < s.le; ++l) {
                const Layer& L = m->layers[(size_t) l];
                const bool last_layer = (size_t) l + 1 == m->layers.size();
                const float* wn = last_layer ? m->out_norm : (l + 1 < s.le ? m->layers[(size_t) l + 1].attn_norm : nullptr);
                float* hout = last_layer ? s.hmain : s.h;
                const bool hy = s.lhy((size_t) l), prev_hy = l > s.lb && s.lhy((size_t) l - 1);
                s.run_graph(G.a[(size_t) l], [&] {
                    if (l == s.lb) {   // the stage's input: the embedded rows, or the handed-over residual
                        if (s.first) s.embed_dev(T, s.x);
                        else CK(cudaMemcpyAsync(s.x, s.hand_in, (size_t) T * N * 4, cudaMemcpyHostToDevice, s.st));
                        k_rms<<<T, kThreads, 0, s.st>>>(s.x, L.attn_norm, s.h, N, m->eps);
                    }
                    if (prev_hy) pending(l - 1);
                    s.layer_mix(L, (size_t) l, T, p0, s.pos0, true, verify);
                    if (hy) {
                        s.route_to_host(T, false);
                    } else {
                        s.grouped_experts(L, T);
                        s.shared_expert(L, T);
                        s.combine(L, T, nullptr, wn, hout);
                    }
                });
                if (hy) {
                    CK(cudaEventRecord(s.ev_route, s.st));
                    s.run_graph(G.b[(size_t) l], [&] {
                        s.resident_experts(L, (size_t) l, T);
                        s.shared_expert(L, T);
                    });
                    s.moe_hybrid_cpu((size_t) l, T);   // waits for the routing, runs the pool (its rows merge next)
                }
            }
            const bool tail_hy = s.lhy((size_t) s.le - 1);
            s.run_graph(G.tail, [&] {
                if (tail_hy) pending(s.le - 1);
                if (!s.last) CK(cudaMemcpyAsync(stg[i + 1]->hand_in, s.x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, s.st));
                else if (heads > 0) s.head_rows(s.hmain + (size_t) (T - heads) * N, heads, s.logits, s.pick);
            });
            if (!s.last) CK(cudaStreamSynchronize(s.st));
        }
    }

    /// The MTP block over T rows (the last stage): row t reads the token staged by last().embed_host at position
    /// p0 + t (also at *pos0_m) and the seed hidden seeds[t] (the main model's normed hidden of the position before,
    /// or the block's own for a chained draft); hmtp gets each row's normed output.  `draft`: logits of the last row
    /// and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        Stage& s = last();
        OnDevice od(s.dev);
        const int N = (int) m->g.n_embd;
        const size_t li = m->layers.size();
        if (s.main_hybrid && s.lhy(li) && T > s.hybrid_t && s.use_mmq) s.stream_begin({(int) li});
        s.embed_dev(T, s.y);
        k_rms_ld<<<T, kThreads, 0, s.st>>>(s.y, N, m->mtp_enorm, s.mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, s.st>>>(seeds, N, m->mtp_hnorm, s.mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        s.mm(m->eh_proj, s.mtp_cat, s.x, T);
        k_rms<<<T, kThreads, 0, s.st>>>(s.x, m->mtp.attn_norm, s.h, N, m->eps);
        s.layer(m->mtp, li, T, p0, s.pos0_m, graph_mode, false, m->mtp_shnorm, s.hmtp);
        if (draft) s.head_rows(s.hmtp + (size_t) (T - 1) * N, 1, s.logits_m, s.pick_m);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows.
    void commit(int n_keep) {
        for (auto& s : stg) s->commit(n_keep);
    }
    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        for (auto& s : stg) s->reset();
        pos = 0;
    }
    void count_rows(int n) {
        for (auto& s : stg) s->count_rows(n);
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            sync_all();
            rebalance(8);
        }
    }
    /// The slots to the experts the routing used most: at most max_swaps replacements per stage, between forwards.
    void rebalance(int max_swaps) {
        for (auto& s : stg) {
            OnDevice od(s->dev);
            s->rebalance(max_swaps);
        }
    }
    void reset_graphs() {
        for (cudaGraphExec_t* g : {&g1, &gm1})
            if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
        for (auto& g : gv)
            if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
    }
    template <class F> void graph_run(cudaGraphExec_t& gx, cudaStream_t s, bool on, F body) {
        if (!on) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(s, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, s));
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph (one stage,
    /// every expert in VRAM) or eagerly.  The last token's logits stay in the last stage's `logits` (its greedy pick
    /// in `pick`), hlast its normed hidden.  With the MTP block the prompt also runs through it (its K / V for every
    /// position: row q reads h_{q-1} and token q).
    template <class F> void process(const int32_t* toks, int64_t n, F on_chunk) {
        const int N = (int) m->g.n_embd;
        Stage& L = last();
        if (n == 1) {
            sync_all();   // the pinned staging is free again
            set_pos(pos);
            first().embed_host(toks, 1);
            set_deferred(true);
            graph_run(g1, first().st, use_graph, [&] { forward(1, pos, 1, use_graph, false); });
            set_deferred(false);
            {
                OnDevice od(L.dev);
                CK(cudaMemcpyAsync(L.hlast, L.hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            }
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            sync_all();
            set_pos(pos);
            first().embed_host(toks + i, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `moe_out`, free here)
                OnDevice od(L.dev);
                CK(cudaMemcpyAsync(mtp_seed(), L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, L.hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, L.st));
                L.embed_host(toks + i, T);
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            {
                OnDevice od(L.dev);
                CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                CK(cudaStreamSynchronize(L.st));
            }
            pos += T;
            if (checkpoint_chunks && T == tmax && i + T < n) take_checkpoint();
            on_chunk(i + T);
        }
        if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
    }
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return last().moe_out; }

    // ---- checkpoints: the recurrent state (every GDN layer's conv history and state matrix, hlast) at a position,
    // in host memory, so a later request that shares the prefix up to there continues from it instead of reading
    // the whole prompt again (the recurrent state cannot step back; the KV cache keeps its cells below the position).
    // serve() takes one after every full prompt chunk and a few tokens before a prompt's end (the next turn of a
    // conversation is rendered again by the chat template and diverges at the answer's start, just before the
    // generation prompt's end), and restores the latest one at or before the common prefix.  ~63 MiB each here.
    struct Ckpt {
        int64_t pos = 0;
        std::vector<std::vector<float>> conv, S;   // per layer (empty for an attention layer)
        std::vector<float> hlast;
    };
    std::deque<Ckpt> ckpts;
    static constexpr size_t kCkpts = 8;
    bool checkpoint_chunks = false;   // process(): a checkpoint after every full prompt chunk (serve sets it)
    void take_checkpoint() {
        const auto& g = m->g;
        sync_all();
        Ckpt c;
        c.pos = pos;
        c.conv.resize(m->layers.size());
        c.S.resize(m->layers.size());
        for (auto& s : stg) {
            OnDevice od(s->dev);
            for (int l = s->lb; l < s->le; ++l) {
                if (!s->conv[(size_t) l]) continue;
                c.conv[(size_t) l].resize((size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1));
                c.S[(size_t) l].resize((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size);
                CK(cudaMemcpyAsync(c.conv[(size_t) l].data(), s->conv[(size_t) l], c.conv[(size_t) l].size() * 4,
                                   cudaMemcpyDeviceToHost, s->st));
                CK(cudaMemcpyAsync(c.S[(size_t) l].data(), s->S[(size_t) l], c.S[(size_t) l].size() * 4, cudaMemcpyDeviceToHost,
                                   s->st));
            }
        }
        {
            Stage& L = last();
            OnDevice od(L.dev);
            c.hlast.resize((size_t) g.n_embd);
            CK(cudaMemcpyAsync(c.hlast.data(), L.hlast, c.hlast.size() * 4, cudaMemcpyDeviceToHost, L.st));
        }
        sync_all();
        while (!ckpts.empty() && ckpts.back().pos >= c.pos) ckpts.pop_back();   // (the same position again)
        ckpts.push_back(std::move(c));
        while (ckpts.size() > kCkpts) ckpts.pop_front();
    }
    /// The latest checkpoint at or before `at` (the common prefix with the held tokens), or null.
    const Ckpt* checkpoint_at_or_before(int64_t at) const {
        const Ckpt* best = nullptr;
        for (const Ckpt& c : ckpts)
            if (c.pos <= at && c.pos > 0 && (!best || c.pos > best->pos)) best = &c;
        return best;
    }
    /// The state back at checkpoint c (the checkpoints after it are dropped: the sequence diverges there).
    void restore_checkpoint(const Ckpt& c) {
        sync_all();
        for (auto& s : stg) {
            OnDevice od(s->dev);
            for (int l = s->lb; l < s->le; ++l) {
                if (!s->conv[(size_t) l]) continue;
                CK(cudaMemcpyAsync(s->conv[(size_t) l], c.conv[(size_t) l].data(), c.conv[(size_t) l].size() * 4,
                                   cudaMemcpyHostToDevice, s->st));
                CK(cudaMemcpyAsync(s->S[(size_t) l], c.S[(size_t) l].data(), c.S[(size_t) l].size() * 4, cudaMemcpyHostToDevice,
                                   s->st));
            }
        }
        {
            Stage& L = last();
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hlast, c.hlast.data(), c.hlast.size() * 4, cudaMemcpyHostToDevice, L.st));
        }
        const int64_t at = c.pos;
        sync_all();
        pos = at;
        while (!ckpts.empty() && ckpts.back().pos > at) ckpts.pop_back();
    }
    void clear_checkpoints() { ckpts.clear(); }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks) {
        const int N = (int) m->g.n_embd;
        Stage& L = last();
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        sync_all();
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        std::vector<int32_t> mt(fix);
        mt.push_back(x_tok);
        L.embed_host(mt.data(), T0);
        drafts.assign((size_t) D, 0);
        {
            OnDevice od(L.dev);
            L.hpin[1] = (int32_t) (pos - (int64_t) fix.size());
            CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
            // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
            float* seeds = mtp_seed();
            if (!fix.empty())
                CK(cudaMemcpyAsync(seeds, L.hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
            if (D == 0) CK(cudaStreamSynchronize(L.st));   // (its uploads done before the window's staging is rewritten)
            if (D > 0) {
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[0] = L.hpin[2];
            }
            int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
            for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
                L.embed_host(&drafts[(size_t) d - 1], 1);
                L.hpin[1] = (int32_t) (pos + d);
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                CK(cudaMemcpyAsync(seeds, L.hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                last_row = 0;
                graph_run(gm1, L.st, mtp_graph, [&] { forward_mtp(1, pos + d, seeds, true, mtp_graph); });
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[(size_t) d] = L.hpin[2];
            }
        }
        // the verify window
        const int T = D + 1;
        std::vector<int32_t> win;
        win.push_back(x_tok);
        for (int d = 0; d < D; ++d) win.push_back(drafts[(size_t) d]);
        first().embed_host(win.data(), T);
        set_pos(pos);
        set_deferred(true);
        graph_run(gv[T], first().st, use_graph, [&] { forward(T, pos, T, use_graph, true); });
        set_deferred(false);
        {
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hpin + 2, L.pick, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
            CK(cudaStreamSynchronize(L.st));
        }
        int a = 0;
        while (a < D && L.hpin[2 + a] == drafts[(size_t) a]) ++a;
        for (int i = 0; i <= a; ++i) picks[i] = L.hpin[2 + i];
        commit(a + 1);
        {
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        }
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    void warmup() {
        const auto t0 = Clock::now();
        set_warming(true);
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        if (tmax > Stage::kGemmT + 1) process(toks.data(), Stage::kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        sync_all();
        set_warming(false);
        for (auto& s : stg) {
            for (Stage::Hy& H : s->hy) std::fill(H.score.begin(), H.score.end(), 0.0);
            s->hy_cpu = s->hy_gpu = s->hy_swaps = 0;
            s->hy_cpu_ms = 0;
        }
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        Stage& L = last();
        OnDevice od(L.dev);
        CK(cudaMemcpyAsync(L.hpin + 2, L.pick, 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
        return L.hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        Stage& L = last();
        OnDevice od(L.dev);
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), L.logits, out.size() * 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
    }
    /// The k largest logits of logits row `row` (0: the last token's; a verify window's rows) and their ids
    /// (k <= Stage::kTopMax).
    void host_topk(int k, std::vector<float>& v, std::vector<int32_t>& ids, int row = 0) {
        Stage& L = last();
        OnDevice od(L.dev);
        k = std::max(1, std::min(k, std::min(Stage::kTopMax, (int) m->g.n_vocab)));
        L.topk(L.logits + (size_t) row * m->g.n_vocab, k);
        v.assign(L.h_topv, L.h_topv + k);
        ids.assign(L.h_topi, L.h_topi + k);
    }

    /// One speculative step for a SAMPLED request (the sampler `s` without penalties): the drafts are the MTP block's
    /// greedy tokens, as in spec_step; the window's rows give the target's distributions, and draft d_r is kept with
    /// probability p_r(d_r) (the draft is a point mass, so that is the exact acceptance rule), else the step ends
    /// with a token drawn from p_r without d_r; after D kept drafts the last row's draw.  The output distribution is
    /// the sampler's, token for token.  Returns the accepted count a; picks[0 .. a] as spec_step.
    int spec_step_sampled(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts,
                          int32_t* picks, const Sampling& s, std::mt19937_64& rng, std::vector<float>& tv,
                          std::vector<int32_t>& ti) {
        const int N = (int) m->g.n_embd;
        Stage& L = last();
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;
        sync_all();
        std::vector<int32_t> mt(fix);
        mt.push_back(x_tok);
        L.embed_host(mt.data(), T0);
        drafts.assign((size_t) D, 0);
        {
            OnDevice od(L.dev);
            L.hpin[1] = (int32_t) (pos - (int64_t) fix.size());
            CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
            float* seeds = mtp_seed();
            if (!fix.empty())
                CK(cudaMemcpyAsync(seeds, L.hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
            if (D == 0) CK(cudaStreamSynchronize(L.st));   // (its uploads done before the window's staging is rewritten)
            if (D > 0) {
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[0] = L.hpin[2];
            }
            int last_row = T0 - 1;
            for (int d = 1; d < D; ++d) {
                L.embed_host(&drafts[(size_t) d - 1], 1);
                L.hpin[1] = (int32_t) (pos + d);
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                CK(cudaMemcpyAsync(seeds, L.hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                last_row = 0;
                graph_run(gm1, L.st, mtp_graph, [&] { forward_mtp(1, pos + d, seeds, true, mtp_graph); });
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[(size_t) d] = L.hpin[2];
            }
        }
        // the verify window: every row's logits
        const int T = D + 1;
        std::vector<int32_t> win;
        win.push_back(x_tok);
        for (int d = 0; d < D; ++d) win.push_back(drafts[(size_t) d]);
        first().embed_host(win.data(), T);
        set_pos(pos);
        set_deferred(true);
        graph_run(gv[T], first().st, use_graph, [&] { forward(T, pos, T, use_graph, true); });
        set_deferred(false);
        sync_all();
        // row r: keep draft r with probability p_r(d_r), else draw from p_r without it and stop
        const int kk = std::max(1, s.top_k > 0 ? s.top_k : 64);
        static const bool dbg = std::getenv("STRATA_Q36_SPEC_DEBUG") != nullptr;
        if (dbg) {   // the device's greedy picks of the window rows beside the sorted rows' top ids
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hpin + 2, L.pick, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
            CK(cudaStreamSynchronize(L.st));
            std::fprintf(stderr, "spec: x %d drafts", x_tok);
            for (int d = 0; d < D; ++d) std::fprintf(stderr, " %d", drafts[(size_t) d]);
            std::fprintf(stderr, " | picks");
            for (int r = 0; r < T; ++r) std::fprintf(stderr, " %d", L.hpin[2 + r]);
            std::fprintf(stderr, "\n");
        }
        int a = 0;
        for (int r = 0; r <= D; ++r) {
            host_topk(kk, tv, ti, r);
            std::vector<double> p;
            size_t keep = 0;
            truncated(tv.data(), (int) tv.size(), s, p, keep);
            if (dbg) {
                std::fprintf(stderr, "spec:   row %d top:", r);
                for (size_t i = 0; i < std::min<size_t>(5, tv.size()); ++i) std::fprintf(stderr, " %d(%.2f)", ti[i], tv[i]);
                std::fprintf(stderr, " keep %zu\n", keep);
            }
            if (r < D) {
                const int d = drafts[(size_t) r];
                size_t at = keep;
                for (size_t i = 0; i < keep; ++i)
                    if (ti[i] == d) { at = i; break; }
                double tot = 0;
                for (size_t i = 0; i < keep; ++i) tot += p[i];
                std::uniform_real_distribution<double> u(0.0, 1.0);
                if (at < keep && u(rng) < p[at] / tot) {
                    picks[r] = d;   // kept
                    a = r + 1;
                    continue;
                }
                if (at < keep) p[at] = 0.0;   // the residual: p without the draft
                picks[r] = draw(p.data(), ti.data(), keep, rng);
                break;
            }
            picks[r] = draw(p.data(), ti.data(), keep, rng);   // every draft kept: the row after them
        }
        commit(a + 1);
        {
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        }
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }
    /// The sampler's truncated distribution over the top candidates (descending logits): p (unnormalized) and the
    /// number kept after min-p / top-p.
    static void truncated(const float* top, int kk, const Sampling& s, std::vector<double>& p, size_t& keep) {
        p.assign((size_t) kk, 0.0);
        const double mx = top[0];
        double sum = 0;
        for (int i = 0; i < kk; ++i) sum += (p[(size_t) i] = std::exp((top[i] - mx) / s.temperature));
        keep = (size_t) kk;
        double cum = 0;
        for (size_t i = 0; i < (size_t) kk; ++i) {
            if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
            cum += p[i] / sum;
            if (cum >= s.top_p) { keep = i + 1; break; }
        }
        keep = std::max<size_t>(1, keep);
    }
    static int draw(const double* p, const int32_t* ids, size_t keep, std::mt19937_64& rng) {
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += p[i];
        if (tot <= 0) return ids[0];
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(rng);
        for (size_t i = 0; i < keep; ++i) {
            if ((r -= p[i]) <= 0) return ids[i];
        }
        return ids[keep - 1];
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// Every expert of a stage's layer into the layer's VRAM blobs (all resident).
void upload_layer(Layer& L, int64_t ne) {
    CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
    CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
    std::vector<uint8_t> blob(L.L.bytes, 0);
    for (int64_t e = 0; e < ne; ++e) {
        expert_blob(L, e, blob.data());
        CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
    }
    L.n_slots = (int) ne;
}

/// The routed experts after the engine's buffers, stage by stage: all of a stage's in its VRAM when they fit in
/// `vram_gib` (< 0: the free VRAM less a reserve), else the hybrid mode for that stage - every expert in host memory
/// (one pinned, portable arena for all stages), the same number of VRAM slots per layer, a decode batch's other
/// experts on the CPU pool (shared) and a prompt chunk's streamed through two staging buffers on that card.
#if defined(STRATA_Q36_CPU_POOL)
strata::kernels::cpu::PoolAffinity pool_affinity_of(const std::string& v) {
    namespace C = strata::kernels::cpu;
    if (v == "auto") return C::PoolAffinity::Auto;
    if (v == "p-cores" || v == "pcores") return C::PoolAffinity::PCores;
    return C::PoolAffinity::All;
}
#endif

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, std::string& err,
                   int pool_tasks = 0, const std::string& affinity = "all") {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    const size_t reserve = (size_t) 768 << 20;
    struct Plan {
        std::vector<Layer*> all;
        bool hybrid = false, mtp_only = false;
        size_t total = 0, total_main = 0, per_slot = 0, max_layer = 0, budget = 0, fr = 0;
    };
    std::vector<Plan> plans(E.stg.size());
    for (size_t i = 0; i < E.stg.size(); ++i) {
        Stage& s = *E.stg[i];
        Plan& P = plans[i];
        OnDevice od(s.dev);
        for (int l = s.lb; l < s.le; ++l) P.all.push_back(&m.layers[(size_t) l]);
        if (s.last && m.has_mtp) P.all.push_back(&m.mtp);
        for (Layer* L : P.all) {
            P.total += L->stride * (size_t) ne + kMmqTail;
            P.per_slot += L->stride;
            P.max_layer = std::max(P.max_layer, L->stride * (size_t) ne);
            if (L != &m.mtp) P.total_main += L->stride * (size_t) ne + kMmqTail;
        }
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                            : (fr > reserve ? fr - reserve : 0);
        // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the
        // card's room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same
        // outputs, a little slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.
        // STRATA_Q36_KEEP_PACKED=1 keeps them (and the hybrid mode).  One stage only: the copies are per device.
        size_t packed_bytes = 0;
        for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
        const size_t slack = ((size_t) 512 << 20) + (s.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
        const bool mtp_cpu_ok = s.last && m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
        if (E.stg.size() == 1 && slots <= 0 && vram_gib < 0 && P.total > budget && packed_bytes > 0 &&
            (P.total <= budget + packed_bytes || (mtp_cpu_ok && P.total_main <= budget + packed_bytes + slack)) &&
            std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
            CK(cudaDeviceSynchronize());
            for (const auto& pk : m.packed) {
                K::native_q8_0_packed_unregister(std::get<0>(pk));
                CK(cudaFree(std::get<1>(pk)));
            }
            m.packed.clear();
            E.reset_graphs();
            CK(cudaMemGetInfo(&fr, &tot));
            std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                         "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                         (double) packed_bytes / (1 << 30));
        }
        size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
        // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM -
        // the text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU
        // pool (its drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more,
        // smaller slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram
        // sets the budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
        bool mtp_only = false;
        if (slots <= 0 && vram_gib < 0 && mtp_cpu_ok && P.total > budget2) {
            const size_t att_save = (s.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
            const size_t small_reserve = (size_t) 256 << 20;
            const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
            if (P.total_main + ((size_t) 64 << 20) <= budget2) {
                mtp_only = true;
            } else if (P.total_main + ((size_t) 64 << 20) <= room) {
                s.shrink_att((size_t) 16 << 20);
                CK(cudaMemGetInfo(&fr, &tot));
                budget2 = fr > small_reserve ? fr - small_reserve : 0;
                mtp_only = P.total_main + ((size_t) 64 << 20) <= budget2;
                std::fprintf(stderr, "strata-q36: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                             "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                             "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                             "budget instead\n");
            }
        }
        if (slots <= 0 && P.total <= budget2 && P.total + ((size_t) 256 << 20) <= fr) {
            for (Layer* L : P.all) upload_layer(*L, ne);
            std::fprintf(stderr, "strata-q36: CUDA%d: every routed expert of layers %d-%d%s in VRAM (%.2f GiB) in %.1f s\n",
                         s.dev, s.lb, s.le - 1, s.last && m.has_mtp ? " and the MTP block" : "", (double) P.total / (1 << 30),
                         ms_since(t0) / 1000.0);
            continue;
        }
        P.hybrid = true;
        P.mtp_only = mtp_only;
        P.budget = budget2;
        P.fr = fr;
    }
    bool any = false;
    for (const Plan& P : plans) any |= P.hybrid;
    if (!any) return true;
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts do not fit in VRAM, and this build has no CPU expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = false;
    for (const Plan& P : plans) E.main_hybrid |= P.hybrid && !P.mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = !E.lhy(m.layers.size());    // (the MTP block's layer resident: its draft step may still be a graph)
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.tmax));
    if (!E.use_mmq()) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    // host memory: every expert of every hybrid layer (an mtp_only stage: the MTP block's), pinned where the driver
    // allows it (the prompt's stream and the slot copies), portable so that every card's copy engine reads it
    size_t host = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const Plan& P = plans[i];
        if (!P.hybrid) continue;
        for (Layer* L : P.all)
            if (!P.mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    }
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocPortable) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata-q36: the driver would not pin %.2f GiB of host memory for the experts: pageable memory "
                     "instead (prompt streaming and slot copies go through the driver's staging buffer, slower)\n",
                     (double) host / (1 << 30));
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    // row tasks per pool phase: nine per thread unless --pool-tasks says otherwise (the pool's own default is three;
    // measured on an i5-13400 (6 P + 4 E cores), two cards: tg64 60.8 / 64.7 / 64.3 tok/s with three, 62.0 / 67.3 /
    // 67.3 with nine - the smaller tasks shorten the E-cores' tail at every phase barrier)
    if (pool_tasks <= 0) {
        const int n = threads > 0 ? threads : (int) C::physical_cores(false).size();
        pool_tasks = std::min(C::ExpertPool::kMaxTasks, 9 * std::max(1, n));
    }
    E.pool = std::make_unique<C::ExpertPool>(threads, true, true, pool_affinity_of(affinity), pool_tasks);
    size_t off = 0;
    const size_t nl = m.layers.size() + 1;
    for (size_t i = 0; i < plans.size(); ++i) {
        const Plan& P = plans[i];
        if (!P.hybrid) continue;
        Stage& s = *E.stg[i];
        OnDevice od(s.dev);
        s.main_hybrid = P.hybrid && !P.mtp_only;   // this stage's own main layers (streamed prompt chunks, the batch size)
        s.hybrid_t = E.hybrid_t;
        s.hyT = s.main_hybrid ? std::max(E.hybrid_t, Stage::kVerify) : E.tmax;   // every batch is a hybrid batch when only the MTP block is
        const size_t staging = s.use_mmq && E.main_hybrid && !P.mtp_only ? 2 * (P.max_layer + kMmqTail) : 0;
        size_t slot_bytes = P.budget > staging ? P.budget - staging : 0;
        if (slot_bytes + staging + ((size_t) 256 << 20) > P.fr)
            slot_bytes = P.fr > staging + ((size_t) 512 << 20) ? P.fr - staging - ((size_t) 512 << 20) : 0;
        size_t per_slot = 0;   // one slot in every hybrid layer of this stage
        for (Layer* L : P.all)
            if (!P.mtp_only || L == &m.mtp) per_slot += L->stride;
        // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
        int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
        if (P.mtp_only) {   // the main layers all, the MTP block what is left
            const size_t left = P.budget > P.total_main ? P.budget - P.total_main : 0;
            S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
        }
        s.hy.assign(nl, Stage::Hy());
        for (Layer* L : P.all) {
            const size_t li = L == &m.mtp ? m.layers.size() : (size_t) (L - m.layers.data());
            if (P.mtp_only && L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
                upload_layer(*L, ne);
                continue;
            }
            L->hblobs = arena + off;
            off += L->stride * (size_t) ne + kMmqTail;
            std::memset(L->hblobs, 0, L->stride * (size_t) ne + kMmqTail);
            for (int64_t e = 0; e < ne; ++e) expert_blob(*L, e, L->hblobs + (size_t) e * L->stride);
            L->n_slots = S;
            if (S > 0) {
                CK(cudaMalloc(&L->blobs, L->stride * (size_t) S + kMmqTail));
                CK(cudaMemset(L->blobs, 0, L->stride * (size_t) S + kMmqTail));
                CK(cudaMemcpy(L->blobs, L->hblobs, L->stride * (size_t) S, cudaMemcpyHostToDevice));
            }
            Stage::Hy& H = s.hy[li];
            H.slot_of.assign((size_t) ne, -1);
            H.exp_in_slot.assign((size_t) S, 0);
            for (int x = 0; x < S; ++x) {
                H.slot_of[(size_t) x] = x;
                H.exp_in_slot[(size_t) x] = x;
            }
            H.score.assign((size_t) ne, 0.0);
            CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
            CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
            if (!C::native_fmt(L->L.gu_type, L->L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
            if (H.fmt.up_off != L->L.up_off || H.fmt.down_off != L->L.down_off || H.fmt.bytes != L->L.bytes) {
                err = "the CPU and GPU expert layouts differ";
                return false;
            }
        }
        const size_t T = (size_t) s.hyT, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
        CK(cudaHostAlloc((void**) &s.h_h, T * N * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &s.cpu_out, T * Kp * N * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &s.h_ids, T * Kp * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &s.h_miss, T * Kp * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &s.h_cnt, nl * (size_t) ne * 4, cudaHostAllocPortable));
        CK(cudaMalloc(&s.ecpu, T * Kp * N * 4));
        CK(cudaMalloc(&s.d_miss, T * Kp * 4));
        s.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
        if (!s.acts) {
            err = "no host memory for the activations";
            return false;
        }
        CK(cudaEventCreateWithFlags(&s.ev_route, cudaEventDisableTiming));
        if (staging) {
            CK(cudaStreamCreateWithFlags(&s.cst, cudaStreamNonBlocking));
            for (int b = 0; b < 2; ++b) {
                CK(cudaMalloc(&s.stage[b], P.max_layer + kMmqTail));
                CK(cudaMemset(s.stage[b], 0, P.max_layer + kMmqTail));
                CK(cudaEventCreateWithFlags(&s.ev_copy[b], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&s.ev_used[b], cudaEventDisableTiming));
                CK(cudaEventRecord(s.ev_used[b], s.st));
            }
        }
        s.pool = E.pool.get();
        CK(cudaDeviceSynchronize());
        if (P.mtp_only)
            std::fprintf(stderr, "strata-q36: CUDA%d: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                         "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", s.dev, (double) P.total_main / (1 << 30), S,
                         (long long) ne);
        else
            std::fprintf(stderr, "strata-q36: CUDA%d: hybrid experts of layers %d-%d%s: %d of %lld per layer in VRAM (%.2f GiB), "
                         "their %.2f GiB in %s host memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt "
                         "chunks streamed (%.2f GiB staging)\n", s.dev, s.lb, s.le - 1,
                         s.last && m.has_mtp ? " and the MTP block" : "", S, (long long) ne, (double) S * per_slot / (1 << 30),
                         (double) (P.total) / (1 << 30), pinned ? "pinned" : "pageable", s.hyT,
                         E.pool->workers() + (E.pool->host_works() ? 1 : 0), (double) staging / (1 << 30));
    }
    std::fprintf(stderr, "strata-q36: experts placed in %.1f s\n", ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling

/// A draw from the kk largest logits (descending, with their ids): temperature, min-p and top-p over them.
int sample_from(const float* top, const int32_t* ids, int kk, const Sampling& s, std::mt19937_64& rng) {
    std::vector<double> p((size_t) kk);
    const double mx = top[0];
    double sum = 0;
    for (int i = 0; i < kk; ++i) sum += (p[(size_t) i] = std::exp((top[i] - mx) / s.temperature));
    size_t keep = (size_t) kk;
    double cum = 0;
    for (size_t i = 0; i < (size_t) kk; ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return ids[i];
    }
    return ids[keep - 1];
}

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int32_t> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int32_t) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int32_t a, int32_t b) { return lg[(size_t) a] > lg[(size_t) b]; });
    std::vector<float> top((size_t) kk);
    for (int i = 0; i < kk; ++i) top[(size_t) i] = lg[(size_t) idx[(size_t) i]];
    return sample_from(top.data(), idx.data(), kk, s, rng);
}

/// The next token of a sampled request: the top candidates sorted on the device when no penalty needs the whole
/// vocabulary on the host (the sampler keeps only its top-k anyway), else the full logits.
int sample_next(Engine& E, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                std::vector<float>& lg, std::vector<float>& tv, std::vector<int32_t>& ti) {
    const bool penal = s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f);
    const int kk = std::max(1, s.top_k > 0 ? s.top_k : 64);
    if (STRATA_Q36_DEVICE_TOPK && !penal && s.temperature > 0.f && kk <= Stage::kTopMax) {
        E.host_topk(kk, tv, ti);
        return sample_from(tv.data(), ti.data(), (int) tv.size(), s, rng);
    }
    E.host_logits(lg);
    return sample(lg, s, hist, rng);
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ serve
int serve(Engine& E) {
    std::printf("READY %d stop\n", E.ctx);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;   // the tokens whose state the engine holds (cell i = live[i])
    std::vector<float> lg, tv;
    std::vector<int32_t> ti;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        stop.store(false);
        // the held state continues when the request extends it (the recurrent state cannot step back)
        size_t reuse = 0;
        if (live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin())) reuse = live.size();
        else {
            // not an extension: the latest checkpoint at or before the common prefix, else from the start
            size_t common = 0;
            while (common < live.size() && common < ids.size() && live[common] == ids[common]) ++common;
            const Engine::Ckpt* c = common > 0 ? E.checkpoint_at_or_before((int64_t) std::min(common, ids.size() - 1)) : nullptr;
            if (c) {
                reuse = (size_t) c->pos;
                E.restore_checkpoint(*c);
                live.resize(reuse);
            } else {
                E.reset();
                E.clear_checkpoints();
                live.clear();
            }
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        auto progress = [&](int64_t base) {
            return [&, base](int64_t done) {
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) (base + done), (long long) fresh, ms_since(t0),
                            (double) (base + done) / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
            };
        };
        // the prompt but its last kTail tokens in chunks (a checkpoint after every full one), a checkpoint, then the
        // tail on its own (a small batch: slots + the pool, no streaming): the next turn of a conversation shares the
        // prompt up to the answer's start, a few tokens before the end, and continues from that checkpoint
        static const int64_t kTail = [] {   // STRATA_Q36_PROMPT_TAIL=N (0: the whole prompt in one go, as 0.1.44)
            const char* e = std::getenv("STRATA_Q36_PROMPT_TAIL");
            return e ? std::max<int64_t>(0, std::atoll(e)) : 16;
        }();
        E.checkpoint_chunks = true;
        if (kTail > 0 && fresh > kTail + 8) {
            E.process(ids.data() + reuse, fresh - kTail, progress(0));
            E.take_checkpoint();
            E.process(ids.data() + reuse + (fresh - kTail), kTail, progress(fresh - kTail));
        } else {
            E.process(ids.data() + reuse, fresh, progress(0));
        }
        E.checkpoint_chunks = false;
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        const double prompt_ms = ms_since(t0);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones one token at a time
        const bool spec = E.mtp_d > 0 && !smp.needs_logits();
        // sampled ones too (no penalties, a top-k the device sort covers): the drafts verified against the sampled
        // distribution, so the text is a draw from the same sampler
        const bool penal = smp.last_n != 0 && (smp.repeat != 1.f || smp.freq != 0.f || smp.present != 0.f);
        const bool spec_s = STRATA_Q36_DEVICE_TOPK && E.mtp_d > 0 && smp.needs_logits() && !penal && smp.temperature > 0.f &&
                            std::max(1, smp.top_k > 0 ? smp.top_k : 64) <= Stage::kTopMax &&
                            std::getenv("STRATA_Q36_NO_SPEC_SAMPLING") == nullptr;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        bool done = false, have_next = false;   // have_next: a speculative step decided nx already (its own draw)
        while (!done) {
            if (smp.needs_logits() && !have_next) nx = sample_next(E, smp, hist, rng, lg, tv, ti);
            have_next = false;
            auto emit = [&](int t) {   // false: the answer ends here
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if ((spec || spec_s) && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = spec ? E.spec_step(nx, fix, E.mtp_d, drafts, picks)
                                   : E.spec_step_sampled(nx, fix, E.mtp_d, drafts, picks, smp, rng, tv, ti);
                offered += (long long) drafts.size();   // (clamped at a rebalance boundary)
                accepted += a;
                live.push_back(nx);
                fix.assign(drafts.begin(), drafts.begin() + a);
                live.insert(live.end(), drafts.begin(), drafts.begin() + a);   // (the engine holds them all, ended or not)
                for (int i = 0; i < a; ++i)
                    if (!emit(drafts[(size_t) i])) { done = true; break; }
                nx = picks[a];
                have_next = true;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
    }
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    std::string layer_split;   // "auto" (the default with two visible GPUs), a first layer of the second card, or 0
    int split_dev = -1;
    int pool_tasks = 0;        // row tasks per pool phase (0: three per thread)
    std::string kv = "f32";    // the KV cache: f32 (upstream's) or q8_0 (8-bit rows, a quarter of the VRAM)
    std::string pool_affinity = "all", pin_host = "auto";
    bool warm = true;
    int expert_slots = 0;
    bool serve_mode = false, graph = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--layer-split") layer_split = next();
        else if (a == "--pool-tasks") pool_tasks = std::atoi(next().c_str());
        else if (a == "--kv") kv = next();
        else if (a == "--pool-affinity") pool_affinity = next();
        else if (a == "--pin-host") pin_host = next();   // yes | no | auto (yes when the pool runs)
        else if (a == "--split-device") split_dev = std::atoi(next().c_str());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    // the visible cards: the first runs the first layers (and the whole model on its own); the packed Q8_0 decode
    // copies (a second copy of the eligible dense matrices, 1.8 GiB on Qwen3.6) only where the first card has room
    // for them (20 GB or more) unless STRATA_Q8_PACKED says otherwise; a small card (under 12 GB) takes the smaller
    // attention-scores buffers
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess) n_dev = 0;
    (void) cudaGetLastError();
    size_t fr0 = 0, tot0 = 0;
    if (n_dev > 0) {
        CK(cudaSetDevice(0));
        CK(cudaMemGetInfo(&fr0, &tot0));
    }
    const bool small = tot0 < ((size_t) 12 << 30);
    const char* packed_default = tot0 >= ((size_t) 20 << 30) ? "1" : "0";
#if !defined(_WIN32)
    setenv("STRATA_Q8_PACKED", packed_default, 0);
#else
    if (std::getenv("STRATA_Q8_PACKED") == nullptr) _putenv_s("STRATA_Q8_PACKED", packed_default);
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    SplitSpec sp;
    sp.n_dev = n_dev;
    sp.dev = split_dev >= 0 ? split_dev : 1;
    sp.ctx = ctx;
    sp.tmax_est = chunk > 0 ? chunk : 2048;
    sp.small = small;
    if (kv != "f32" && kv != "q8_0") { std::fprintf(stderr, "strata-q36: --kv takes f32 or q8_0\n"); return 2; }
    sp.kv_q8 = kv == "q8_0";
    if (layer_split.empty()) layer_split = n_dev >= 2 ? "auto" : "0";
    sp.automatic = layer_split == "auto";
    sp.k = sp.automatic ? 0 : std::atoi(layer_split.c_str());
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0, sp)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    Engine E;
    E.use_graph = graph;
    E.hybrid_graph = graph && std::getenv("STRATA_Q36_NO_LAYER_GRAPH") == nullptr;
    if (chunk == 0) {
        // auto: 512 with every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaSetDevice(m.dev_of[0]));
        CK(cudaMemGetInfo(&fr, &tot));
        for (size_t l = 0; l < m.layers.size(); ++l)   // the first card's layers (a second card's are placed whole)
            if (m.dev_of[l] == m.dev_of[0]) experts += m.layers[l].stride * (size_t) m.g.n_expert;
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        if (m.last_dev == m.dev_of[0])   // (two cards never give them back: place_experts)
            for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (with %.2f packed), main experts %.2f GiB\n",
                         fr / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0);
    }
    if (sp.kv_q8 && (m.g.n_head != 8 * m.g.n_head_kv || m.g.head_dim != 256)) {
        std::fprintf(stderr, "strata-q36: --kv q8_0 needs 8 query heads per kv head and a head size of 256\n");
        return 1;
    }
    E.init(m, ctx, chunk, small, sp.kv_q8);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err, pool_tasks, pool_affinity)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
#if defined(STRATA_Q36_CPU_POOL)
    // the host thread spins on the routing event and drains pool tasks itself: pinned to the pool's reserved core
    // (the first physical core), so it never lands on a worker's core or SMT sibling (Flash-Next's session does
    // the same; --pin-host no leaves it to the scheduler)
    if (E.pool && pin_host != "no") {
        namespace C = strata::kernels::cpu;
        const int core = C::planned_host_core(pool_affinity_of(pool_affinity));
        if (core >= 0) {
            const C::ThreadAffinity ta = C::pin_current_thread(core);
            if (ta.valid) std::fprintf(stderr, "strata-q36: the host thread on CPU %d (--pin-host no leaves it free)\n", core);
        }
    }
#endif
    struct Report {
        Engine& E;
        ~Report() {
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu() + E.hy_gpu();
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu(), n, n ? 100.0 * E.hy_gpu() / n : 0.0,
                         E.hy_cpu(), E.hy_cpu_ms(), E.hy_swaps());
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            E.sync_all();
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            if (E.mtp_d > 0 && bench_tg > 0) {   // the same count again with MTP drafts (greedy): the effective rate
                const auto t2 = Clock::now();
                long long offered = 0, accepted = 0;
                int produced = 0;
                std::vector<int32_t> fix, drafts;
                int32_t picks[16];
                while (produced < bench_tg) {
                    const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
                    offered += E.mtp_d;
                    accepted += a;
                    fix.assign(drafts.begin(), drafts.begin() + a);
                    produced += 1 + a;
                    nx = picks[a];
                }
                const double sp_ms = ms_since(t2);
                std::printf("bench %d: tg%d with %d MTP drafts %.1f tok/s (%lld of %lld drafts accepted)\n", rep, produced,
                            E.mtp_d, (double) produced / (sp_ms / 1000.0), accepted, offered);
            }
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
