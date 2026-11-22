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
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/q36_prompt_route.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/fused_tile_rule.hpp"
#include "strata/prefill/own_only_switches.hpp"
#include "strata/prefill/moe_fused_q36.hpp"
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "strata/qwen35/expert_info.hpp"
#include "strata/qwen35/sampling.hpp"
#include "q36_kernels.cuh"
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
namespace PF = strata::prefill::fused;          // the grouping and the int8 sizes (shape-free)
namespace FQ = strata::prefill::fused_q36;      // the fused kernels for Qwen3.6's shape (CUDA)

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
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)
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

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session (8-bit rows with --kv q8_0)
    float* const* vc = nullptr;
};

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, bool q8, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
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
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
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
                               int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
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
                                                       float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256, G = 8;
    __shared__ float sc[G][kChunkCells];
    __shared__ float sh[kThreads];
    const int c = blockIdx.x, grp = blockIdx.y;   // grp = (t - t0) * HKV + kvh
    const int t = t0 + grp / HKV, kvh = grp % HKV;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
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
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
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
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
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
                           const uint8_t* blobs, const uint8_t* blobs2, int S1, unsigned long long blob,
                           unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                           int32_t* ent_tok) {
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
            const int s = slot_of[x];
            grp_ptr[g] = (unsigned long long) (s < S1 ? blobs + (size_t) s * blob : blobs2 + (size_t) (s - S1) * blob);
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

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
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
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
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
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
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
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
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
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
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
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM (n_slots1 in `blobs`, the rest in blobs2)
    int n_slots1 = 0;           // the slots in `blobs`; slots n_slots1 .. n_slots - 1 are in blobs2 (the elastic region)
    uint8_t* blobs2 = nullptr;  // the extra slots a stage keeps while no prompt chunk needs its prompt buffers
    uint8_t* blobs2_prev = nullptr;   // where they were last time (the decode graphs hold the address)
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
    } else if (((t.type >= 11 && t.type <= 14) || t.type == 23) && h.size() % 256 == 0) {
        // Q3_K..Q6_K, IQ4_XS (mixed GGUFs store ssm_alpha / ssm_beta so): 256-value super-blocks
        static const struct { void (*f)(const uint8_t*, float*); size_t bytes; } kq[] = {
            {strata::dequantize_q3_K, 110}, {strata::dequantize_q4_K, 144}, {strata::dequantize_q5_K, 176},
            {strata::dequantize_q6_K, 210}, {strata::dequantize_iq4_xs, 136}};
        const auto& k = kq[t.type == 23 ? 4 : t.type - 11];
        for (size_t b = 0; b < h.size() / 256; ++b) k.f(src + b * k.bytes, h.data() + b * 256);
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16, Q8_0, Q3_K..Q6_K or IQ4_XS\n", t.name.c_str(), t.type_name());
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
    int n_sess = 1;   // sessions (--batch): each keeps its own K / V and recurrent state on every card
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
    const size_t kv_layer = (size_t) sp.n_sess * 2 * (size_t) g.n_head_kv * (size_t) sp.ctx *
                            (sp.kv_q8 ? (size_t) kKvRowQ8 : (size_t) g.head_dim * 4);
    const size_t gdn_state = (size_t) sp.n_sess * ((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 +
                                                   (size_t) g.ssm_conv_channels * 3 * 4) +
                             8 * (2 * (size_t) g.ssm_conv_channels + 2 * (size_t) g.ssm_v_heads) * 4;
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

using strata::qwen35::Sampling;

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
    // the buffers sized by the rows of a batch: one set for the prompt chunk (tmax rows) and, on an elastic stage,
    // a second for the hybrid batches (hybrid_t rows) that is the active one while no streamed prompt runs - the
    // chunk's set and the streaming buffers are freed then and their VRAM holds extra expert slots (blobs2)
    struct RowSet {
        int T = 0, att_tq_chunk = 0;
        bool carved = false;   // from the elastic arena (nothing to free)
        float *x = nullptr, *h = nullptr, *y = nullptr, *hmain = nullptr, *qkv = nullptr, *hc = nullptr, *z = nullptr,
              *alpha = nullptr, *beta = nullptr, *gdn_y = nullptr, *qg = nullptr, *kx = nullptr, *vx = nullptr, *q = nullptr,
              *gate = nullptr, *att = nullptr, *logits_r = nullptr, *eparts = nullptr, *moe_out = nullptr, *sg = nullptr,
              *su = nullptr, *shh = nullptr, *sy = nullptr, *sgate = nullptr, *ewts = nullptr, *sk_part = nullptr,
              *mtp_cat = nullptr, *hmtp = nullptr, *att_m = nullptr, *att_l = nullptr, *att_alpha = nullptr, *att_part = nullptr,
              *mq_gu = nullptr, *mq_h = nullptr;
        void *q8 = nullptr, *xq = nullptr, *escr = nullptr, *mq_x = nullptr, *mq_hq = nullptr;
        __half *x16 = nullptr, *q16 = nullptr, *att_qg = nullptr;
        int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr,
                *tok = nullptr, *ident = nullptr, *rinv = nullptr;
        unsigned long long* grp_ptr = nullptr;
        // the prompt chunk's fused / int8 expert routes (upstream's, see use_pf): the grouping tables, the input and H in
        // the fused kernels' int8 form, the entries' rows, and the int8 kernels' activation image
        void *f_scr = nullptr, *f_xq = nullptr, *f_ha = nullptr, *i8x = nullptr;
        int32_t *f_slot = nullptr, *f_src = nullptr;
    };
    RowSet rows_big, rows_small;
    bool elastic = false;      // the prompt chunk's buffers only while a streamed prompt runs (see enter_*_phase)
    bool prompt_phase = true;  // which set is active
    size_t per_slot = 0;       // one slot in every hybrid layer of this stage, in bytes
    size_t max_layer_bytes = 0;
    // the elastic region: one allocation (the free VRAM after the base slots but 512 MiB) that holds the chunk's set
    // and the streaming buffers while a streamed prompt runs and the extra slots otherwise - carved, never freed,
    // so the switch costs no allocation and the extra slots keep their addresses (the decode graphs stay valid)
    uint8_t* arena_d = nullptr;
    size_t arena_bytes = 0, carve_off = 0;
    bool carving = false;      // alloc_rows / alloc_staging carve from the arena instead of cudaMalloc
    cudaStream_t fst = nullptr;   // the extra slots' fill stream (host memory to VRAM behind the first decode steps)
    struct Pending { int li = 0; std::vector<int> experts; cudaEvent_t ev = nullptr; };
    std::vector<Pending> pending;   // layers whose extra slots are still filling: published when their copies are done
    std::vector<cudaEvent_t> ev_fill;   // per layer
    cudaEvent_t ev_fill_start = nullptr, ev_fill_end = nullptr;   // (timing) the fill's span on the stream
    Clock::time_point fill_t0;
    double fill_ms = 0;   // the fills' time on the stream, summed (stats)
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
    // the prompt's experts on Strata's fused int8 kernels (moe_fused_iq_q36.cu) where they cover a layer (fused_on), else our
    // int8 expert kernels (i8_on, native_expert_i8), else MMQ where the build has it, else the native grouped kernels (the route
    // of each layer: q36_layer_route, include/strata/kernels/q36_prompt_route.hpp; upstream's since 0.2.16, which has no MMQ in
    // the tree).  use_pf: every layer of this stage has one of the fast routes, so a hybrid stage streams each layer's experts
    // for a chunk (the strategy MMQ had alone before)
    std::vector<char> fused_on, i8_on;   // by state index (the MTP block last)
    size_t n_i8 = 0, n_fused = 0;
    int i8_few = 0;                      // native_expert_i8_few_rows_limit()
    bool use_pf = false;
    void *f_scr = nullptr, *f_xq = nullptr, *f_ha = nullptr, *i8x = nullptr;
    int32_t *f_slot = nullptr, *f_src = nullptr;
    // the prompt's MMQ experts (a build with llama.cpp's MMQ only)
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
    // [verify][rows] per session (the graphs hold the session's state pointers as kernel arguments): a verify window's GDN
    // kernels leave the state alone; the batch windows' graphs (rows of several sessions, the state through the tables) by
    // the window's shape (Engine::run_rows' key)
    struct LGSet { LayerGraphs g[2][kVerify + 1]; };
    std::vector<LGSet> lgs;
    std::map<int, LayerGraphs> rgr;
    // ---- sessions (--batch N): per session this stage's layers' recurrent state and K / V cache, and hlast (the last
    // stage's is the one read); conv / S / kc / vc / hlast above are the CURRENT session's (select_sess)
    struct SessState { std::vector<float*> conv, S, kc, vc; float* hlast = nullptr; };
    std::vector<SessState> sst;
    int n_sess = 1, cur = 0;
    void select_sess(int i) {
        cur = i;
        const SessState& ss = sst[(size_t) i];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
    }
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers, on this card
    float **d_conv_tab = nullptr, **d_S_tab = nullptr, **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;     // per group (a slot): its session, first row, rows
    int rows_G = 0;            // groups of the rows being run
    bool rows_mode = false;    // the rows are sessions' (not consecutive cells)
    // pictures (--vision): the M-RoPE table of the request's prompt cells (every stage: its attention layers read it); the
    // picture rows and a chunk's row -> picture row map on the stages that embed tokens (the first, and the MTP block's)
    int32_t* d_mtab = nullptr;
    float* d_img = nullptr;
    size_t img_cap = 0;
    int32_t* d_map = nullptr;
    int32_t* h_map = nullptr;
    bool map_on = false;
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
    // ---- the attention's row selection and rotary inputs: the batch rows' tables (sessions) and the pictures' M-RoPE table
    // and rotary offset (set by the engine; null: one session's consecutive cells, text positions)
    int32_t* d_delta = nullptr;          // device scalar: the current session's rotary offset (decode graphs read it)
    const int32_t* mtab_now = nullptr;   // [ctx][3] t, h, w of the current request's prompt cells, for a forward over them
    RowSel rsel(size_t li) const {
        RowSel r;
        if (rows_mode) {
            r.cell = d_rcell;
            r.slot = d_rslot;
            r.delta = d_rdelta;
            r.kc = d_kc_tab + li * (size_t) n_sess;
            r.vc = d_vc_tab + li * (size_t) n_sess;
        }
        return r;
    }
    // ---- the decode kernels of upstream's q36_kernels.cuh (each the old kernel's bits; STRATA_Q36_OLD_KERNELS=1: the old ones)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    // the one-warp-per-row Q8_0 products (q8_small): measured slower on Turing - 2x RTX 2080 + i5-13400, UD-Q4_K_XL, 32K, --kv
    // q8_0, engine bench tg128 75.7 / 76.5 tok/s with them against 78.5-78.9 / 79.8-79.9 without (drafted 87.4 / 97.4 against
    // 91.3-91.4 / 102.4-102.7), while every other kernel of the set was level or faster there (k_moe_out_small +2.6%) - so they
    // run from compute capability 8 on (upstream measured the set on an R9700); STRATA_Q36_Q8_ROWS=1 / 0 decides either way
    bool q8_rows = true;
    bool fk(unsigned b) const { return fast_k && (b != 1 || q8_rows); }
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
    int sw_n = 0;                            // elastic phase switches (prompt -> decode) and their time (the buffers'
    double sw_ms = 0, sw_fill_ms = 0;        // release and return); the time until the extra slots were all in use
    bool said_extra = false;                 // the extra slots' line printed (once per stage)
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
    void init(const Model& model, int ctx_, int tmax_, int dev_, int lb_, int le_, bool last_, bool small, bool kvq8,
              int n_sessions, bool vision) {
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
        {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) major = 8;
            const char* qr = std::getenv("STRATA_Q36_Q8_ROWS");
            q8_rows = qr ? std::atoi(qr) != 0 : major >= 8;
        }
        if (small && std::getenv("STRATA_Q36_BIG_ATT") == nullptr) kAttScoreFloats = (size_t) 16 << 20;   // (STRATA_Q36_BIG_ATT=1: the 256 MB buffers, for A/B runs)
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        if (last) logits = dalloc((size_t) kVerify * g.n_vocab);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        k16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        v16 = dalloc_t<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        {
            const bool mmq_avail = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
            auto mmq_ok = [&](const Layer& L) {
                return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                       MMQ::fits(L.L.d_type, g.n_embd);
            };
            const size_t n_state = m->layers.size() + 1;
            fused_on.assign(n_state, 0);
            i8_on.assign(n_state, 0);
            if (g.n_embd == FQ::kQ36Embd && g.n_ff == FQ::kQ36FF)
                for (size_t i = 0; i < n_state; ++i)
                    if (mine(i) && (i < m->layers.size() || m->has_mtp)) {
                        const Layer& L = layer_at((int) i);
                        fused_on[i] = FQ::native_layer_ext(L.L.gu_type, L.L.d_type) ? 1 : 0;
                    }
            // the route of each layer: fused > our int8 kernels (STRATA_EXPERT_I8=1) > MMQ > our int8 kernels > the grouped kernels.
            // A build without MMQ reads an unset STRATA_EXPERT_I8 as 1 (the kernel library is told here, as upstream's engine does)
            const bool mmq_free = !MMQ::built();
            if (mmq_free) K::native_expert_i8_set_mmq_free();
            const int i8_raw = K::i8_switch_parse(std::getenv("STRATA_EXPERT_I8"));
            i8_few = K::native_expert_i8_few_rows_limit();
            bool any_mmq = false;
            int n_mine = 0, n_grouped = 0;
            for (size_t i = 0; i < n_state; ++i) {
                if (!mine(i) || (i == m->layers.size() && !m->has_mtp)) continue;
                const Layer& Li = layer_at((int) i);
                const bool mmq_takes = !fused_on[i] && mmq_avail && mmq_ok(Li);
                const K::Q36Route r = K::q36_layer_route(fused_on[i] != 0, i8_raw, mmq_free, mmq_takes,
                                                         !fused_on[i] && K::native_expert_i8_supported(Li.L));
                i8_on[i] = r == K::Q36Route::Int8 ? 1 : 0;
                n_grouped += r == K::Q36Route::Grouped ? 1 : 0;
                any_mmq = any_mmq || r == K::Q36Route::Mmq;
                n_fused += fused_on[i] ? 1 : 0;
                n_i8 += (size_t) i8_on[i];
                ++n_mine;
            }
            use_pf = K::q36_prompt_path_ok(n_mine, n_grouped);
            if (!use_pf) {   // a layer has no route faster than the grouped kernels: every prompt chunk on them, as before
                std::fill(fused_on.begin(), fused_on.end(), 0);
                std::fill(i8_on.begin(), i8_on.end(), 0);
                n_fused = n_i8 = 0;
                any_mmq = false;
            }
            use_mmq = use_pf && any_mmq;
            if (use_pf) {
                rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
                rcur = dalloc_t<int32_t>((size_t) g.n_expert);
                rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
                rmax = dalloc_t<int32_t>(1);
            }
            if (use_mmq) mctx = std::make_unique<MMQ::Context>();
            const int n_mmq = n_mine - (int) n_fused - (int) n_i8 - n_grouped;
            std::fprintf(stderr, "strata-q36: CUDA%d: prompt experts: %zu of %d layers on the fused int8 kernels, %zu on our int8 kernels, "
                         "%d on MMQ, %d on the native grouped kernels\n", dev_, n_fused, n_mine, n_i8, n_mmq, n_grouped);
        }
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
        alloc_rows(rows_big, T);   // the prompt chunk's set (an elastic stage adds a small one in place_experts)
        use_rows(rows_big);
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
        for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->assign(nl, nullptr);
        for (int l = lb; l < le; ++l)   // the verify windows' GDN inputs (one set, whichever session runs)
            if (!m->layers[(size_t) l].attn) {
                v_qkv[(size_t) l] = dalloc((size_t) kVerify * C);
                v_hc[(size_t) l] = dalloc((size_t) kVerify * C);
                v_alpha[(size_t) l] = dalloc((size_t) kVerify * g.ssm_v_heads);
                v_beta[(size_t) l] = dalloc((size_t) kVerify * g.ssm_v_heads);
            }
        // the sessions: this stage's layers' state and cache per session (one without --batch); the MTP block's own K / V
        // (an attention layer at index n_layers) on the last stage
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        sst.resize((size_t) n_sess);
        lgs.resize((size_t) n_sess);
        for (SessState& ss : sst) {
            ss.conv.assign(nl, nullptr); ss.S.assign(nl, nullptr); ss.kc.assign(nl, nullptr); ss.vc.assign(nl, nullptr);
            for (int l = lb; l < le; ++l) {
                if (m->layers[(size_t) l].attn) {
                    ss.kc[(size_t) l] = kv_alloc();
                    ss.vc[(size_t) l] = kv_alloc();
                } else {
                    ss.conv[(size_t) l] = dalloc(C * (g.ssm_d_conv - 1));
                    ss.S[(size_t) l] = dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size);
                }
            }
            if (last && m->has_mtp) {
                ss.kc[nl - 1] = kv_alloc();
                ss.vc[nl - 1] = kv_alloc();
            }
            ss.hlast = dalloc(N);
        }
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on this card
            std::vector<float*> hc_(nl * (size_t) n_sess), hs(hc_), hk(hc_), hv(hc_);
            for (size_t li = 0; li < nl; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc_[li * n_sess + s_] = sst[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sst[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sst[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sst[(size_t) s_].vc[li];
                }
            const std::pair<float***, std::vector<float*>*> tabs[4] = {{&d_conv_tab, &hc_}, {&d_S_tab, &hs},
                                                                          {&d_kc_tab, &hk}, {&d_vc_tab, &hv}};
            for (const auto& tb : tabs) {
                CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*), cudaMemcpyHostToDevice));
            }
            for (int32_t** d : {&d_rcell, &d_rslot, &d_rdelta, &d_gslot, &d_grow0, &d_gcnt}) *d = dalloc_t<int32_t>(kVerify);
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            if (first || (last && m->has_mtp)) {
                d_map = dalloc_t<int32_t>(T);
                CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocPortable));
            }
        }
        d_delta = dalloc_t<int32_t>(1);
        select_sess(0);
        if (last && m->has_mtp) {
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

    /// A device buffer: from the elastic arena while `carving` (zeroed on the stream), else its own allocation.
    void* carve(size_t bytes) {
        const size_t off = (carve_off + 255) & ~(size_t) 255;
        if (off + bytes > arena_bytes) {
            std::fprintf(stderr, "strata-q36: CUDA%d: the elastic region is too small (%zu + %zu of %zu bytes)\n", dev, off,
                         bytes, arena_bytes);
            std::exit(1);
        }
        carve_off = off + bytes;
        void* p = arena_d + off;
        CK(cudaMemsetAsync(p, 0, bytes, st));
        return p;
    }
    float* ralloc(size_t n) { return carving ? (float*) carve(n * 4) : dalloc(n); }
    template <class T> T* ralloc_t(size_t n) { return carving ? (T*) carve(n * sizeof(T)) : dalloc_t<T>(n); }
    void* rallocb(size_t bytes) {
        if (carving) return carve(bytes);
        void* p = nullptr;
        CK(cudaMalloc(&p, bytes));
        return p;
    }
    /// The buffers of a batch of up to T rows.
    void alloc_rows(RowSet& r, size_t T) {
        const auto& g = m->g;
        const size_t N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff, C = (size_t) g.ssm_conv_channels;
        r.T = (int) T;
        r.carved = carving;
        r.x = ralloc(T * N); r.h = ralloc(T * N); r.y = ralloc(T * N); r.hmain = ralloc(T * N);
        r.qkv = ralloc(T * C); r.hc = ralloc(T * C); r.z = ralloc(T * g.ssm_value_dim);
        r.alpha = ralloc(T * g.ssm_v_heads); r.beta = ralloc(T * g.ssm_v_heads); r.gdn_y = ralloc(T * g.ssm_value_dim);
        r.qg = ralloc(T * 2 * g.n_head * g.head_dim); r.kx = ralloc(T * g.n_head_kv * g.head_dim);
        r.vx = ralloc(T * g.n_head_kv * g.head_dim); r.q = ralloc(T * g.n_head * g.head_dim);
        r.gate = ralloc(T * g.n_head * g.head_dim); r.att = ralloc(T * g.n_head * g.head_dim);
        r.logits_r = ralloc(T * (g.n_expert + 1));
        r.eparts = ralloc(T * Kp * N); r.moe_out = ralloc(T * N);
        r.sg = ralloc(T * g.n_ff_shexp); r.su = ralloc(T * g.n_ff_shexp); r.shh = ralloc(T * g.n_ff_shexp); r.sy = ralloc(T * N);
        r.sgate = ralloc(T); r.ewts = ralloc(T * Kp);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        r.q8 = rallocb(K::native_q8_1_bytes(max_in, 1) * T);
        r.xq = rallocb(K::native_q8_1_bytes((int) N, 1) * T);
        r.x16 = ralloc_t<__half>(T * (size_t) max_in);
        r.sk_part = ralloc((size_t) kSplitK * T * (g.n_expert + 1));
        r.q16 = ralloc_t<__half>(T * g.n_head * g.head_dim);
        if (use_pf) {
            const int64_t ne = (int64_t) T * Kp;
            r.rinv = ralloc_t<int32_t>((size_t) ne);
            if (n_fused > 0) {
                r.f_scr = rallocb(PF::group_bytes(ne, (int) g.n_expert));
                r.f_xq = rallocb(PF::act_bytes((int64_t) T, g.n_embd));
                r.f_ha = rallocb(PF::act_bytes(ne, g.n_ff));
                r.f_slot = ralloc_t<int32_t>((size_t) ne);
                r.f_src = ralloc_t<int32_t>((size_t) ne);
            }
            if (n_i8 > 0) r.i8x = rallocb(K::native_q8_1_bytes((int) N, 1) * T);
        }
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            // the row table is read past its end by llama.cpp's MMQ (a whole J-column tile, and the stream-k fixup up to
            // roundup(max_rows, J) from an expert's first row): mmq_ids_table_rows adds that margin (upstream a041fd038)
            const int64_t ids_rows = (int64_t) MMQ::mmq_ids_table_rows((uint64_t) ne, (uint64_t) T);
            if (ids_rows <= 0) { std::fprintf(stderr, "strata-q36: the MMQ row table does not fit\n"); std::exit(1); }
            r.ident = ralloc_t<int32_t>((size_t) ids_rows);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(r.ident, ids_rows, st);
            r.mq_x = rallocb(MMQ::q8_bytes(ne, g.n_embd));
            r.mq_hq = rallocb(MMQ::q8_bytes(ne, g.n_ff));
            r.mq_gu = ralloc((size_t) ne * 2 * FF);
            r.mq_h = ralloc((size_t) ne * FF);
        }
        r.escr = rallocb(K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF));
        r.ids = ralloc_t<int32_t>(T * Kp);
        r.grp_ptr = ralloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        r.grp_start = ralloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        r.n_groups = ralloc_t<int32_t>(1);
        r.ent_dst = ralloc_t<int32_t>(T * Kp);
        r.ent_tok = ralloc_t<int32_t>(T * Kp);
        r.tok = ralloc_t<int32_t>(T);
        if (last && m->has_mtp) {
            r.mtp_cat = ralloc(T * 2 * N);
            r.hmtp = ralloc(T * N);
        }
        r.att_tq_chunk = (int) std::max<size_t>(1, std::min<size_t>(T, kAttScoreFloats / ((size_t) g.n_head * kAttKeyChunk)));
        r.att_m = ralloc(T * (size_t) g.n_head);
        r.att_l = ralloc(T * (size_t) g.n_head);
        r.att_alpha = ralloc(T * (size_t) g.n_head);
        r.att_part = ralloc((size_t) r.att_tq_chunk * g.n_head * g.head_dim);
        r.att_qg = ralloc_t<__half>((size_t) r.att_tq_chunk * g.n_head * g.head_dim);
    }
    void free_rows(RowSet& r) {
        for (void* p : {(void*) r.x, (void*) r.h, (void*) r.y, (void*) r.hmain, (void*) r.qkv, (void*) r.hc, (void*) r.z,
                        (void*) r.alpha, (void*) r.beta, (void*) r.gdn_y, (void*) r.qg, (void*) r.kx, (void*) r.vx, (void*) r.q,
                        (void*) r.gate, (void*) r.att, (void*) r.logits_r, (void*) r.eparts, (void*) r.moe_out, (void*) r.sg,
                        (void*) r.su, (void*) r.shh, (void*) r.sy, (void*) r.sgate, (void*) r.ewts, (void*) r.sk_part,
                        (void*) r.mtp_cat, (void*) r.hmtp, (void*) r.att_m, (void*) r.att_l, (void*) r.att_alpha,
                        (void*) r.att_part, (void*) r.mq_gu, (void*) r.mq_h, r.q8, r.xq, r.escr, r.mq_x, r.mq_hq,
                        (void*) r.x16, (void*) r.q16, (void*) r.att_qg, (void*) r.ids, (void*) r.grp_start, (void*) r.n_groups,
                        (void*) r.ent_dst, (void*) r.ent_tok, (void*) r.tok, (void*) r.ident, (void*) r.rinv, (void*) r.grp_ptr,
                        r.f_scr, r.f_xq, r.f_ha, r.i8x, (void*) r.f_slot, (void*) r.f_src})
            if (p && !r.carved) CK(cudaFree(p));
        r = RowSet();
    }
    /// The set the kernels read and write.
    void use_rows(const RowSet& r) {
        x = r.x; h = r.h; y = r.y; hmain = r.hmain; qkv = r.qkv; hc = r.hc; z = r.z; alpha = r.alpha; beta = r.beta;
        gdn_y = r.gdn_y; qg = r.qg; kx = r.kx; vx = r.vx; q = r.q; gate = r.gate; att = r.att; logits_r = r.logits_r;
        eparts = r.eparts; moe_out = r.moe_out; sg = r.sg; su = r.su; shh = r.shh; sy = r.sy; sgate = r.sgate; ewts = r.ewts;
        q8 = r.q8; xq = r.xq; escr = r.escr; x16 = r.x16; sk_part = r.sk_part; q16 = r.q16; ids = r.ids; grp_ptr = r.grp_ptr;
        grp_start = r.grp_start; n_groups = r.n_groups; ent_dst = r.ent_dst; ent_tok = r.ent_tok; tok = r.tok;
        mtp_cat = r.mtp_cat; hmtp = r.hmtp; att_tq_chunk = r.att_tq_chunk; att_m = r.att_m; att_l = r.att_l;
        att_alpha = r.att_alpha; att_part = r.att_part; att_qg = r.att_qg; ident = r.ident; rinv = r.rinv; mq_x = r.mq_x;
        mq_hq = r.mq_hq; mq_gu = r.mq_gu; mq_h = r.mq_h;
        f_scr = r.f_scr; f_xq = r.f_xq; f_ha = r.f_ha; i8x = r.i8x; f_slot = r.f_slot; f_src = r.f_src;
    }
    /// The streaming buffers of a hybrid stage's prompt chunks (two whole layers of experts).
    void alloc_staging() {
        if (!cst) CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        staging_carved = carving;
        for (int b = 0; b < 2; ++b) {
            stage[b] = (uint8_t*) rallocb(max_layer_bytes + kMmqTail);
            if (!carving) CK(cudaMemset(stage[b], 0, max_layer_bytes + kMmqTail));
            if (!ev_copy[b]) CK(cudaEventCreateWithFlags(&ev_copy[b], cudaEventDisableTiming));
            if (!ev_used[b]) CK(cudaEventCreateWithFlags(&ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(ev_used[b], st));
        }
    }
    bool staging_carved = false;
    void free_staging() {
        for (int b = 0; b < 2; ++b) {
            if (stage[b] && !staging_carved) CK(cudaFree(stage[b]));
            stage[b] = nullptr;
        }
    }
    void reset_layer_graphs() {
        auto drop = [](LayerGraphs& G) {
            for (auto& g : G.a) if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
            for (auto& g : G.b) if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
            if (G.tail) { CK(cudaGraphExecDestroy(G.tail)); G.tail = nullptr; }
            G.a.clear();
            G.b.clear();
        };
        for (auto& kv : rgr) drop(kv.second);
        rgr.clear();
        for (LGSet& ls : lgs)
          for (auto& set : ls.g)
            for (auto& G : set) {
                for (auto& g : G.a) if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
                for (auto& g : G.b) if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
                if (G.tail) { CK(cudaGraphExecDestroy(G.tail)); G.tail = nullptr; }
                G.a.clear();
                G.b.clear();
            }
    }
    /// An elastic stage before a streamed prompt: the extra slots go (their experts leave the slot tables), the
    /// chunk's row set and the streaming buffers come back.
    void enter_prompt_phase() {
        if (!elastic || prompt_phase) return;
        const auto t0 = Clock::now();
        OnDevice od(dev);
        CK(cudaStreamSynchronize(st));
        if (fst) CK(cudaStreamSynchronize(fst));
        // the fills still pending are dropped (their region is about to be overwritten)
        for (const Pending& p : pending)
            for (int x : p.experts) hy[(size_t) p.li].slot_of[(size_t) x] = -1;
        pending.clear();
        const auto& g = m->g;
        for (int l = lb; l < le; ++l) {
            Layer& L = const_cast<Layer&>(m->layers[(size_t) l]);
            if (!L.blobs2) continue;
            Hy& H = hy[(size_t) l];
            for (int s = L.n_slots1; s < L.n_slots; ++s) H.slot_of[(size_t) H.exp_in_slot[(size_t) s]] = -1;
            H.exp_in_slot.resize((size_t) L.n_slots1);
            L.n_slots = L.n_slots1;
            CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice));
            L.blobs2 = nullptr;   // (the arena's: not freed)
        }
        carve_off = 0;
        carving = true;
        alloc_rows(rows_big, (size_t) tmax);
        alloc_staging();
        carving = false;
        use_rows(rows_big);
        prompt_phase = true;
        CK(cudaDeviceSynchronize());
        sw_ms += ms_since(t0);
    }
    /// An elastic stage after a streamed prompt: the chunk's buffers go, their VRAM holds extra slots.  The first
    /// time the region is allocated (the free VRAM but 512 MiB; at least the chunk's buffers' size).  The slots are
    /// filled by routing score from the host arena on the fill stream and put to use (publish_fills) before the
    /// next token, or behind the next steps with STRATA_Q36_ASYNC_FILL=1 (then their experts are pending, -2, so a
    /// rebalance does not bring them in twice, and they go into the tables at a rebalance boundary).
    void enter_decode_phase() {
        if (!elastic || !prompt_phase) return;
        const auto t0 = Clock::now();
        OnDevice od(dev);
        CK(cudaDeviceSynchronize());
        size_t fr0 = 0, fr = 0, tot = 0;
        if (!arena_d) CK(cudaMemGetInfo(&fr0, &tot));
        free_rows(rows_big);
        free_staging();
        use_rows(rows_small);
        prompt_phase = false;
        const auto& g = m->g;
        const int64_t ne = g.n_expert;
        if (!arena_d) {
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t need = (fr > fr0 ? fr - fr0 : 0) + ((size_t) 1 << 20);   // what the chunk's buffers took
            const size_t keep = (size_t) 512 << 20;
            arena_bytes = std::max(fr > keep ? fr - keep : 0, need);
            if (arena_bytes > fr) arena_bytes = fr;
            CK(cudaMalloc(&arena_d, arena_bytes));
            CK(cudaStreamCreateWithFlags(&fst, cudaStreamNonBlocking));
            ev_fill.assign(m->layers.size() + 1, nullptr);
            for (auto& e : ev_fill) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            CK(cudaEventCreate(&ev_fill_start));
            CK(cudaEventCreate(&ev_fill_end));
        }
        // the slot count: the region over every hybrid layer's share (the layers with every expert resident take none)
        size_t per = 0, tails = 0;
        for (int l = lb; l < le; ++l) {
            const Layer& L = m->layers[(size_t) l];
            if (L.n_slots1 >= ne || !L.hblobs) continue;
            per += L.stride;
            tails += kMmqTail + 256;
        }
        const int S2 = per > 0 && arena_bytes > tails ? (int) std::min<size_t>((size_t) ne, (arena_bytes - tails) / per) : 0;
        carve_off = 0;
        carving = true;
        int added = 0;
        bool moved = false;
        CK(cudaEventRecord(ev_fill_start, fst));
        for (int l = lb; l < le && S2 > 0; ++l) {
            Layer& L = const_cast<Layer&>(m->layers[(size_t) l]);
            if (L.n_slots1 >= ne || !L.hblobs) continue;   // not hybrid, or every expert resident already
            const int S2l = std::min<int>(S2, (int) ne - L.n_slots1);
            if (S2l <= 0) continue;
            const size_t off = (carve_off + 255) & ~(size_t) 255;   // (carve zeroes; the slots are written whole)
            carve_off = off + L.stride * (size_t) S2l + kMmqTail;
            uint8_t* b2 = arena_d + off;
            moved |= L.blobs2_prev != nullptr && L.blobs2_prev != b2;
            L.blobs2 = L.blobs2_prev = b2;
            Hy& H = hy[(size_t) l];
            std::vector<int> order;
            for (int x = 0; x < ne; ++x)
                if (H.slot_of[(size_t) x] == -1) order.push_back(x);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            Pending P;
            P.li = l;
            P.ev = ev_fill[(size_t) l];
            // one copy per slot (the decode's small uploads wait for the copy in flight, so a decode step beside the
            // fill runs at about half speed; smaller pieces measured worse: 256 KB pieces cost 440 ms to issue)
            for (int j = 0; j < S2l; ++j) {
                const int x = order[(size_t) j];
                CK(cudaMemcpyAsync(b2 + (size_t) j * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                   cudaMemcpyHostToDevice, fst));
                H.slot_of[(size_t) x] = -2;   // pending: no slot yet, and not a rebalance candidate
                P.experts.push_back(x);
            }
            CK(cudaEventRecord(P.ev, fst));
            pending.push_back(std::move(P));
            added = S2l;
        }
        carving = false;
        CK(cudaEventRecord(ev_fill_end, fst));
        if (moved) reset_layer_graphs();   // (the slots' addresses are the same from one switch to the next: never)
        fill_t0 = Clock::now();
        // the slots before the next token (the default): STRATA_Q36_ASYNC_FILL=1 leaves the copies running behind
        // the next steps and puts the slots to use at the next rebalance boundary (or the prompt tail's end) - the
        // same prompt time through serve (its tail batch runs beside the fill, at the half speed the copy queue
        // leaves it), 15% less on the engine bench's first 64 tokens after a prompt
        static const bool async_fill = std::getenv("STRATA_Q36_ASYNC_FILL") != nullptr;
        if (!async_fill) publish_fills();
        sw_ms += ms_since(t0);
        ++sw_n;
        if (!said_extra) {
            said_extra = true;
            size_t fr2 = 0;
            CK(cudaMemGetInfo(&fr2, &tot));
            std::fprintf(stderr, "strata-q36: CUDA%d: %d extra expert slots per layer between prompts (the prompt chunk's "
                         "buffers' VRAM, %.2f GiB; filled after each streamed prompt; %.2f GiB of VRAM free)\n",
                         dev, added, (double) arena_bytes / (1 << 30), (double) fr2 / (1 << 30));
        }
    }
    /// The extra slots go into the slot tables (at a rebalance boundary, outside graph capture; the copies still
    /// running are waited for, so the tables depend on the position alone).
    void publish_fills() {
        if (pending.empty()) return;
        OnDevice od(dev);
        const int64_t ne = m->g.n_expert;
        bool any = false;
        for (auto it = pending.begin(); it != pending.end();) {
            CK(cudaEventSynchronize(it->ev));
            Layer& L = const_cast<Layer&>(m->layers[(size_t) it->li]);
            Hy& H = hy[(size_t) it->li];
            for (size_t j = 0; j < it->experts.size(); ++j) {
                const int x = it->experts[j];
                H.slot_of[(size_t) x] = L.n_slots1 + (int) j;
                H.exp_in_slot.push_back(x);
            }
            L.n_slots = L.n_slots1 + (int) it->experts.size();
            CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            any = true;
            it = pending.erase(it);
        }
        if (any) CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable
        if (any && pending.empty()) {
            sw_fill_ms += ms_since(fill_t0);
            float ms = 0;
            CK(cudaEventSynchronize(ev_fill_end));
            CK(cudaEventElapsedTime(&ms, ev_fill_start, ev_fill_end));
            fill_ms += ms;
        }
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

    /// The batch rows' tables from the engine's pinned staging (hr: [0..T) tokens, [8..) cells, [16..) sessions, [24..)
    /// rotary offsets, [40..) / [48..) / [56..) per group its session / first row / rows).
    void put_rows(const int32_t* hr, int T, int G) {
        CK(cudaMemcpyAsync(d_rcell, hr + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hr + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hr + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hr + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hr + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hr + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
    }
    /// After a batch verify window: group g's GDN state takes its first keep rows (the counts already in d_gcnt), as commit().
    void commit_rows(int G) {
        OnDevice od(dev);
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (int l = lb; l < le; ++l) {
            const Layer& L = m->layers[(size_t) l];
            if (L.attn) continue;
            k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[(size_t) l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                         d_conv_tab + (size_t) l * n_sess, d_gslot, d_grow0, d_gcnt);
            k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[(size_t) l], C, v_alpha[(size_t) l], v_beta[(size_t) l],
                                                            z, L.ssm_norm, m->eps, nullptr, HK, HV, 1, true,
                                                            d_S_tab + (size_t) l * n_sess, d_gslot, d_grow0, d_gcnt);
        }
    }
    /// Session dst takes session src's state for this stage's layers (the recurrent state, the cache's first n cells, hlast).
    void copy_sess(int src, int dst, int64_t n) {
        OnDevice od(dev);
        const auto& g = m->g;
        const SessState& a = sst[(size_t) src];
        SessState& b = sst[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t cell = kv_q8 ? (size_t) kKvRowQ8 : (size_t) g.head_dim * 4;   // one head's row at one cell
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && n > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], (size_t) ctx * cell, a.kc[l], (size_t) ctx * cell, (size_t) n * cell,
                                     (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], (size_t) ctx * cell, a.vc[l], (size_t) ctx * cell, (size_t) n * cell,
                                     (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
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
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    // ---- the decode kernels of upstream's q36_kernels.cuh (c9e7f6dd5), each written to give the old kernel's bits (upstream
    // checks them with STRATA_Q36_VERIFY=1; here the parity runs' greedy texts are the check); STRATA_Q36_OLD_KERNELS=1 keeps
    // the old ones
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048 rows) for
    /// T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of native_small_mmvq_kernel.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (!fk(1) || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        return true;
    }
    // the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is already
    // quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q).  Host flags: a
    // stage's graphs of one kind are captured in one pass, so a captured graph and its neighbours agree on them
    bool xq_ready = false, xq_post = false;
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fk(2) && T <= kGemmT) {
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            xq_post = true;
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fk(4) || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(a.d, nullptr, b.d, nullptr, (const q36k::QB*) xq,     \
                                                            (q36k::QB*) q8, nb, FS);                         \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        return true;
    }
    /// A decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when
    /// the fused router follows (upstream 93f0f48f1); one row always did.  The same answer for every caller of a batch.
    bool use_ext(int T) const {
        static const bool no_ext = std::getenv("STRATA_Q36_NO_EXT") != nullptr;
        return T == 1 || (fk(32) && !no_ext && T <= kGemmT && m->g.n_expert == 256 && m->top_k == 8);
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in
    /// logits_r with row stride ld.  slot_of: the hybrid mode's slot table (null: every expert resident).
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        grouped_by_route = false;
        if (!fk(8) || T > 8 || m->g.n_expert != 256 || m->top_k != 8) return false;
        if (slot_of)
            q36k::k_route_group<true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr,
                                                                 grp_start, n_groups, ent_dst, ent_tok, L.blobs2, L.n_slots1);
        else
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr,
                                                                  grp_start, n_groups, ent_dst, ent_tok);
        grouped_by_route = true;
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy over both regions).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fk(8) && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok, L.blobs2, L.n_slots1);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            return;
        }
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.blobs2, L.n_slots1,
                                               L.stride, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, grp_ptr, grp_start,
                                            n_groups, ent_dst, ent_tok);
    }
    /// The MoE block's end for T rows: k_moe_out, or for a decode batch k_moe_out_small (1024 threads), which also writes the
    /// next layer's q8_1 input image when the next layer is on this stage (xq_ready).
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fk(16) && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            xq_ready = mkq;
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// The draft head (upstream's --draft-vocab N, opt-in): the MTP block's pick over the first draft_n rows of the output
    /// matrix only.  A draft only proposes - the verify window decides - so the answer is the greedy answer whatever draft_n is.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 != 64) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
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
            prep_in(T);
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
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            const RowSel rs = rsel(li);
            k_attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, rs.kc ? nullptr : kc[li],
                                                              rs.kc ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                                                              m->rope_base, m->eps, kv_q8, d_delta, mtab_now, rs);
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
                        k_attn_part_gqa<true><<<dim3(nc, nt * HKV), kChunkCells, 0, st>>>(q, rs.kc ? nullptr : kc[li], rs.kc ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                                         1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    else if (attn_v == 1)
                        k_attn_part_v1<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, rs.kc ? nullptr : kc[li], rs.kc ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                                1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    else if (attn_v == 2 || H != 8 * HKV)
                        k_attn_part<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, rs.kc ? nullptr : kc[li], rs.kc ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                             1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    else
                        k_attn_part_gqa<false><<<dim3(nc, nt * HKV), kChunkCells, 0, st>>>(q, rs.kc ? nullptr : kc[li], rs.kc ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                                          1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        // routed experts (a decode batch: the router and the shared expert's gate in one product, the gate its last row)
        const bool ext = use_ext(T);
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the router's top-k and (a decode batch) the grouping of its picks in one launch; a prompt chunk's routes group
        // their own entries
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool pf_rows = !hy_rows && T > kGemmT && use_pf;
        if (pf_rows || !route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert))
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
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// Every routed expert in VRAM (no hybrid): the entries grouped per expert and run.
    void grouped_experts(const Layer& L, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        group_entries(L, nullptr, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The shared expert, scaled later by sigmoid(w . h) (its gate: the router product's last row for one row).
    void shared_expert(const Layer& L, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, FS = (int) g.n_ff_shexp;
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        if (T <= kGemmT) {
            if (!shared_gu(L, T)) {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
            }
            mmq(L.sh_down, q8, sy, T);
        } else {
            apply(L.sh_gate, xq, sg, T);
            apply(L.sh_up, xq, su, T);
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (!use_ext(T)) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
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
        (void) Kp;
        const bool ext = use_ext(T);
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
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
        } else if (T > kGemmT && use_pf) {
            // a prompt chunk: each expert's weights read once for all its rows, on Strata's fused int8 kernels where they
            // cover the layer (fused_on), else our int8 kernels (i8_on), else llama.cpp's MMQ (a build with it).  In the
            // hybrid mode the layer's experts come streamed (stream_take); the routing counts feed the slots' scores there
            const int ne = (int) g.n_expert;
            const bool fz = fused_on[li] != 0;
            const bool i8l = i8_on[li] != 0;   // our int8 kernels (no sorted-row tables: they collect their own groups)
            const bool mql = !fz && !i8l;       // MMQ
            int64_t maxr = 0;
            if (mql || lhy(li)) {
                CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
                k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            }
            if (mql) {
                // MMQ: the entries sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down
                // into rows the combine reads through entry -> sorted row
                k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
                k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
                CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            }
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            if (mql || lhy(li)) {
                CK(cudaStreamSynchronize(st));
                if (lhy(li))
                    for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
                maxr = hpin[10];
            }
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            if (fz) {
                // the layer's input to int8 once per token, the rows of each expert grouped on the device (no host wait),
                // then per batch of experts gate/up + SwiGLU + H to int8 + down into the entries' rows (f_slot)
                FQ::quantize_act_native(h, T, N, f_xq, st);
                PF::group(ids, n_ent, Kp, ne, f_scr, f_slot, f_src, st);
                const FQ::NativeGeom ng{L.L.gu_type, L.L.d_type, L.L.gu_row, L.L.d_row, L.L.up_off, L.L.down_off};
                for (int e = 0; e < ne; e += FQ::kQ36MaxBatch) {
                    FQ::Batch b;
                    b.e0 = e;
                    b.e1 = std::min(ne, e + FQ::kQ36MaxBatch);
                    for (int x = b.e0; x < b.e1; ++x) b.blob[x - b.e0] = wbase + (size_t) x * L.stride;
                    FQ::experts_native_ext(b, ng, ne, n_ent, f_scr, f_xq, f_src, f_ha, eparts, st);
                }
                prow = f_slot;
            } else if (i8l) {
                // our int8 expert kernels: the layer's rows quantized once, the groups per expert (blobs from wbase: the
                // streamed buffer in the hybrid mode), gate/up + SwiGLU + H to int8 and the down rows into eparts by entry
                // (prow stays null, as the grouped kernels).  A chunk whose experts average few rows takes the grouped kernels.
                k_group<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, (int) g.n_expert, wbase, L.stride, grp_ptr, grp_start,
                                                n_groups, ent_dst, ent_tok);
                if (K::q36_i8_chunk_takes(n_ent, ne, i8_few)) {
                    K::native_expert_i8_quantize(h, T, N, i8x, st);
                    K::native_expert_i8(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(n_ent, g.n_expert),
                                        n_ent, i8x, escr, eparts, st);
                } else {
                    quant(h, N, T, xq);
                    K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                             std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
                }
            } else {
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
                prow = rinv;
            }
            if (lhy(li)) stream_done();
        } else {
            grouped_experts(L, T);
        }
        shared_expert(L, T);
        if (hy_rows && moe_hybrid_cpu(li, T)) merge_cpu_rows(T);   // the CPU pool's experts (while the GPU runs the rest)
        combine(L, T, prow, wn, hout);
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }
    /// Slot s of layer L (the base region, or the elastic one).
    static uint8_t* slot_ptr(const Layer& L, int s) {
        return s < L.n_slots1 ? L.blobs + (size_t) s * L.stride : L.blobs2 + (size_t) (s - L.n_slots1) * L.stride;
    }

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
                if (dev)   // consecutive slots holding consecutive experts (a run stays inside one slot region)
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1 &&
                           (H.slot_of[(size_t) y] < L.n_slots1) == (H.slot_of[(size_t) x] < L.n_slots1))
                        ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, slot_ptr(L, H.slot_of[(size_t) x]), n,
                                       cudaMemcpyDeviceToDevice, cst));
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
            int npend = 0;   // experts on their way into extra slots: in the top ranks, but not in the S slots yet
            for (int x = 0; x < ne; ++x) {
                order[(size_t) x] = x;
                npend += H.slot_of[(size_t) x] == -2;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            const int scan = (int) std::min<int64_t>(ne, S + npend);
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < scan && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] != -1) continue;   // resident, or pending in an extra slot
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    // a clear win only (twice the count, and two): the counts of a few tokens reorder the middle
                    // ranks every time, and a swap is 1.8 MiB over PCIe plus a sync
                    if (H.score[(size_t) y] * 2.0 + 2.0 >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(slot_ptr(L, s), L.hblobs + (size_t) x * L.stride, L.L.bytes, cudaMemcpyHostToDevice, st));
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

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

/// The engine: the stages in layer order (one card each), the CPU expert pool they share, the position, the drafts
/// and the graphs of a single-stage engine.  A forward runs the stages one after the other; the residual crosses
/// from one card to the next through the next stage's pinned hand-off buffer (8 KiB per row), with a host sync.
bool is_eos(int t);

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
    int rb_swaps = 0;      // slot replacements right after prompts, and their time (stats)
    double rb_ms = 0;
    bool (*draft_stop)(int) = nullptr;   // a draft this returns true for ends the window (serve: the end of the answer)
    // #5012: the expert placement, for the INFO line (the Monitor's "expert cache: slots / experts"; strata/qwen35/expert_info.hpp)
    strata::qwen35::ExpertInfo expert_info;
    // graphs of a one-stage engine: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step (they hold
    // the session's state pointers as kernel arguments: one set per session), and the batch windows' by their shape
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    std::map<int, cudaGraphExec_t> row_graphs;

    // ---- checkpoints: the recurrent state (every GDN layer's conv history and state matrix, hlast) at a position,
    // in host memory, so a later request that shares the prefix up to there continues from it instead of reading
    // the whole prompt again (the recurrent state cannot step back; the KV cache keeps its cells below the position).
    // serve() takes one after every full prompt chunk and a few tokens before a prompt's end (the next turn of a
    // conversation is rendered again by the chat template and diverges at the answer's start, just before the
    // generation prompt's end), and restores the latest one at or before the common prefix.  ~63 MiB each here;
    // per session (fewer each with --batch).
    struct Ckpt {
        int64_t pos = 0;
        std::vector<std::vector<float>> conv, S;   // per layer (empty for an attention layer)
        std::vector<float> hlast;
    };
    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache on every stage (Stage::sst), so N requests run at once: their rows share one decode
    // window (run_rows).  `pos` above is the CURRENT session's (select() swaps it).
    struct Sess {
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
        std::deque<Ckpt> ckpts;
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    bool bmtp = false;          // --batch-mtp: batch windows verify MTP proposals
    int32_t* hrow = nullptr;    // pinned (portable) staging for the batch rows' tables
    int rows_key = -1;          // the batch window being run (its per-layer graphs: Stage::rgr)
    bool vision = false;
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
    bool use_pf() { return first().use_pf; }
    long long hy_cpu() { long long n = 0; for (auto& s : stg) n += s->hy_cpu; return n; }
    long long hy_gpu() { long long n = 0; for (auto& s : stg) n += s->hy_gpu; return n; }
    long long hy_swaps() { long long n = 0; for (auto& s : stg) n += s->hy_swaps; return n; }
    int sw_n() { int n = 0; for (auto& s : stg) n += s->sw_n; return n; }
    double sw_ms() { double n = 0; for (auto& s : stg) n += s->sw_ms; return n; }
    double sw_fill_ms() { double n = 0; for (auto& s : stg) n += s->sw_fill_ms; return n; }
    double fill_ms() { double n = 0; for (auto& s : stg) n += s->fill_ms; return n; }
    double hy_cpu_ms() { double n = 0; for (auto& s : stg) n += s->hy_cpu_ms; return n; }

    /// The stages from Model::dev_of (one per run of layers on the same device, in layer order).
    bool kv_q8 = false;
    void init(const Model& model, int ctx_, int tmax_, bool small, bool q8, int n_sessions, bool vision_) {
        m = &model;
        kv_q8 = q8;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        ctx = ctx_;
        tmax = std::max(tmax_, kVerify);
        std::vector<std::pair<int, int>> runs;   // (device, first layer)
        for (size_t l = 0; l < m->layers.size(); ++l)
            if (l == 0 || m->dev_of[l] != m->dev_of[l - 1]) runs.push_back({m->dev_of[l], (int) l});
        for (size_t i = 0; i < runs.size(); ++i) {
            stg.push_back(std::make_unique<Stage>());
            const int le = i + 1 < runs.size() ? runs[i + 1].second : (int) m->layers.size();
            stg.back()->init(model, ctx, tmax, runs[i].first, runs[i].second, le, i + 1 == runs.size(), small, q8, n_sess,
                             vision);
        }
        if (stg.size() > 1) use_graph = false;   // the hand-off's host sync cannot be captured
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocPortable));
        cur = 0;
        load_cur();
    }

    /// The current session: `i`'s state, cache and position become the engine's (every stage's conv / S / kc / vc / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        for (auto& s : stg) s->select_sess(cur);
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar on every stage), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        for (auto& s : stg) {
            if (!s->d_delta) continue;
            OnDevice od(s->dev);
            s->hpin[60] = d;
            CK(cudaMemcpyAsync(s->d_delta, s->hpin + 60, 4, cudaMemcpyHostToDevice, s->st));
            CK(cudaStreamSynchronize(s->st));   // (the pinned word is reused)
        }
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
    void enter_prompt_phase() { for (auto& s : stg) s->enter_prompt_phase(); }
    void enter_decode_phase() { for (auto& s : stg) s->enter_decode_phase(); }
    void publish_fills() { for (auto& s : stg) s->publish_fills(); }   // (at a rebalance boundary)
    void set_warming(bool v) { warming = v; for (auto& s : stg) s->warming = v; }

    /// The main model over T rows (the rows' tokens staged by first().embed_host; positions p0 .. p0 + T - 1, also at
    /// every stage's *pos0): the last stage's hmain gets every row's final normed hidden; `heads` rows from the end
    /// get logits (rows [T - heads, T): logits row r - (T - heads)) and their greedy picks (`pick`).  `verify`: a
    /// verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const int N = (int) m->g.n_embd;
        // the hybrid decode step, verify window or batch window in per-layer graphs - not over a picture's cells, and not
        // while an elastic stage holds the prompt chunk's buffers (a batch window between a read's chunks: the graphs hold
        // the decode buffers' addresses)
        bool prompt_bufs = false;
        for (auto& s : stg) prompt_bufs = prompt_bufs || (s->elastic && s->prompt_phase);
        if (!graph_mode && hybrid_graph && m->hybrid && (T == 1 || verify || stg[0]->rows_mode) && T <= kVerify &&
            T <= hybrid_t && !pictures_now() && !prompt_bufs) {
            forward_graphs(T, p0, heads, verify);
            return;
        }
        for (size_t i = 0; i < stg.size(); ++i) {
            Stage& s = *stg[i];
            OnDevice od(s.dev);
            if (s.main_hybrid && T > s.hybrid_t && s.use_pf) {
                std::vector<int> seq;
                for (int l = s.lb; l < s.le; ++l)
                    if (s.lhy((size_t) l)) seq.push_back(l);
                if (!seq.empty()) s.stream_begin(seq);
            }
            if (s.first) {
                s.embed_dev(T, s.x);
                if (s.map_on) k_put_rows<<<T, kThreads, 0, s.st>>>(s.x, s.d_img, s.d_map, N);   // a picture's cells take its rows
            } else {
                CK(cudaMemcpyAsync(s.x, s.hand_in, (size_t) T * N * 4, cudaMemcpyHostToDevice, s.st));
            }
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
            Stage::LayerGraphs& G = s.rows_mode ? s.rgr[rows_key] : s.lgs[(size_t) s.cur].g[verify ? 1 : 0][T];
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
        if (s.main_hybrid && s.lhy(li) && T > s.hybrid_t && s.use_pf) s.stream_begin({(int) li});
        s.embed_dev(T, s.y);
        if (s.map_on) k_put_rows<<<T, kThreads, 0, s.st>>>(s.y, s.d_img, s.d_map, N);
        k_rms_ld<<<T, kThreads, 0, s.st>>>(s.y, N, m->mtp_enorm, s.mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, s.st>>>(seeds, N, m->mtp_hnorm, s.mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        s.mm(m->eh_proj, s.mtp_cat, s.x, T);
        k_rms<<<T, kThreads, 0, s.st>>>(s.x, m->mtp.attn_norm, s.h, N, m->eps);
        s.layer(m->mtp, li, T, p0, s.pos0_m, graph_mode, false, m->mtp_shnorm, s.hmtp);
        if (draft) s.head_draft(s.hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows.
    void commit(int n_keep) {
        for (auto& s : stg) s->commit(n_keep);
    }
    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        for (auto& s : stg) s->reset();
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }
    void count_rows(int n) {
        for (auto& s : stg) s->count_rows(n);
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            sync_all();
            publish_fills();   // the extra slots come into use at a boundary (the fill waited for): the slot tables
            rebalance(8);      // stay a function of the position, as the rebalance keeps them
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
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
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
    /// on_chunk(done) runs after every chunk; false ends the read there (a cancelled request): the tokens processed are
    /// returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        const int N = (int) m->g.n_embd;
        Stage& L = last();
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            sync_all();   // the pinned staging is free again
            set_pos(pos);
            first().embed_host(toks, 1);
            chunk_pictures(pos, 1);
            set_deferred(true);
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (pictures_now()) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, first().st, use_graph, [&] { forward(1, pos, 1, use_graph, false); });
            clear_chunk_pictures();
            set_deferred(false);
            {
                OnDevice od(L.dev);
                CK(cudaMemcpyAsync(L.hlast, L.hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            }
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        if (n > hybrid_t) enter_prompt_phase();   // a streamed chunk: the elastic stages' prompt buffers
        int64_t done_n = n;
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            sync_all();
            set_pos(pos);
            first().embed_host(toks + i, T);
            chunk_pictures(pos, T);
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
            clear_chunk_pictures();
            if (checkpoint_chunks && T == tmax && i + T < n) take_checkpoint();
            if (!on_chunk(i + T)) {
                done_n = i + T;
                break;
            }
        }
        enter_decode_phase();
        if (n <= hybrid_t) publish_fills();   // a batch that ran beside the fill (serve's prompt tail): the slots now
        if (m->hybrid) {
            // the slots to the prompt's routing: on an elastic stage the extra slots' fill is that (its base slots
            // follow at the boundaries); after a small batch as at a boundary
            const auto t1 = Clock::now();
            const long long before = hy_swaps();
            for (auto& s : stg) {
                OnDevice od(s->dev);
                s->rebalance(s->elastic && n > hybrid_t ? 0 : n <= hybrid_t ? 8 : hybrid_swaps_after_prompt);
            }
            rb_ms += ms_since(t1);
            rb_swaps += (int) (hy_swaps() - before);
            static const bool dbg = std::getenv("STRATA_Q36_SWAP_DEBUG") != nullptr;
            if (dbg)
                std::fprintf(stderr, "strata-q36: batch of %lld: %lld slot replacements after it in %.1f ms\n", (long long) n,
                             hy_swaps() - before, ms_since(t1));
        }
        return done_n;
    }
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return last().moe_out; }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        sync_all();
        for (auto& sp : stg) {
            Stage& s = *sp;
            OnDevice od(s.dev);
            if (s.d_map) {   // the stages that embed tokens
                if (rows.size() > s.img_cap) {
                    if (s.d_img) CK(cudaFree(s.d_img));
                    CK(cudaMalloc((void**) &s.d_img, rows.size() * 4));
                    s.img_cap = rows.size();
                }
                CK(cudaMemcpy(s.d_img, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
            }
            CK(cudaMemcpy(s.d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        }
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        clear_chunk_pictures();
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        clear_chunk_pictures();
        if (!img_active) return;
        bool any = false;
        std::vector<int32_t> mp((size_t) T);
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            mp[(size_t) j] = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            any = any || mp[(size_t) j] >= 0;
        }
        for (auto& sp : stg) {
            Stage& s = *sp;
            if (c0 < img_prompt_cells) s.mtab_now = s.d_mtab;
            if (any && s.h_map) {
                OnDevice od(s.dev);
                std::memcpy(s.h_map, mp.data(), (size_t) T * 4);
                CK(cudaMemcpyAsync(s.d_map, s.h_map, (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
                s.map_on = true;
            }
        }
    }
    void clear_chunk_pictures() {
        for (auto& s : stg) {
            s->mtab_now = nullptr;
            s->map_on = false;
        }
    }
    bool pictures_now() const {
        for (auto& s : stg)
            if (s->mtab_now || s->map_on) return true;
        return false;
    }

    // ---- batch rows (--batch N): one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    long long win_count = 0;
    int32_t rows_pick[kVerify] = {};
    /// One window over the groups' rows on every stage (the rows' cells, sessions and rotary offsets in each stage's
    /// tables).  `verify`: a verify window - the GDN state is read, not written (commit_rows writes it for the kept rows)
    /// and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every row's logits in the
    /// last stage's `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        sync_all();   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        first().embed_host(hrow, T);
        for (auto& sp : stg) {
            OnDevice od(sp->dev);
            sp->put_rows(hrow, T, G);
            sp->rows_mode = true;
            sp->rows_G = G;
        }
        set_pos(pmax);
        clear_chunk_pictures();
        rows_key = (verify ? 1000 : 0) + G * 10 + (T - G);
        set_deferred(true);
        graph_run(row_graphs[rows_key], first().st, use_graph, [&] { forward(T, pmax, T, use_graph, verify); });
        set_deferred(false);
        for (auto& sp : stg) sp->rows_mode = false;
        {
            Stage& L = last();
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(hrow + 32, L.pick, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
            CK(cudaStreamSynchronize(L.st));
        }
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) {
                sync_all();
                publish_fills();
                rebalance(8);
            }
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows and the session advances.
    void commit_rows(const Group* gr, const int* keep, int G) {
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (the stages' slot / first-row words are the window's)
        for (auto& sp : stg) {
            OnDevice od(sp->dev);
            CK(cudaMemcpyAsync(sp->d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, sp->st));
            sp->rows_mode = true;
            sp->rows_G = G;
            sp->commit_rows(G);
            sp->rows_mode = false;
        }
        sync_all();
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }
    /// The MTP block over rows of several sessions (the last stage, eagerly): row r reads token tok at cell `cell` of session
    /// `slot` and the seed hidden at `seed` (device, on the last stage); its draft - the block's greedy pick after that row -
    /// ends up in mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        Stage& L = last();
        sess[(size_t) cur].pos = pos;
        sync_all();
        OnDevice od(L.dev);
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            hrow[40 + r] = rows[r].slot;
            hrow[48 + r] = r;
            hrow[56 + r] = 1;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        }
        L.embed_host(hrow, T);
        L.put_rows(hrow, T, T);
        L.rows_mode = true;
        L.rows_G = T;
        clear_chunk_pictures();
        L.hpin[1] = (int32_t) pmax;
        CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
        set_deferred(true);
        L.embed_dev(T, L.y);
        k_rms_ld<<<T, kThreads, 0, L.st>>>(L.y, N, m->mtp_enorm, L.mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
        k_rms_ld<<<T, kThreads, 0, L.st>>>(mtp_seed(), N, m->mtp_hnorm, L.mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
        L.mm(m->eh_proj, L.mtp_cat, L.x, T);
        k_rms<<<T, kThreads, 0, L.st>>>(L.x, m->mtp.attn_norm, L.h, N, m->eps);
        L.layer(m->mtp, m->layers.size(), T, pmax, L.pos0_m, false, false, m->mtp_shnorm, L.hmtp);
        L.mm(m->head, L.hmtp, logits_mr, T);
        k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, L.st>>>(logits_mr, (int) g.n_vocab, pick_mr, L.amv, L.ami, false);
        k_argmax<<<dim3(1, T), kThreads, 0, L.st>>>(logits_mr, kArgBlocks, pick_mr, L.amv, L.ami, true);
        set_deferred(false);
        L.rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last window, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        Stage& L = last();
        OnDevice od(L.dev);
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), L.logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
    }
    /// Session b's normed hidden of its last processed position (the MTP seed), on the last stage.
    float* sess_hlast(int b) { return last().sst[(size_t) b].hlast; }
    /// The last window's row r's normed hidden (the last stage's hmain).
    const float* row_hidden(int r) { return last().hmain + (size_t) r * m->g.n_embd; }
    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        sync_all();
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        for (auto& sp : stg) sp->copy_sess(src, dst, a.pos);
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        b.ckpts.clear();
        if (dst == cur) load_cur();
    }

    // ---- checkpoints (see Ckpt): the current session's
    size_t ckpt_cap() const { return n_sess > 1 ? std::max<size_t>(2, 8 / (size_t) n_sess) : 8; }
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
        std::deque<Ckpt>& ckpts = sess[(size_t) cur].ckpts;
        while (!ckpts.empty() && ckpts.back().pos >= c.pos) ckpts.pop_back();   // (the same position again)
        ckpts.push_back(std::move(c));
        while (ckpts.size() > ckpt_cap()) ckpts.pop_front();
    }
    /// The latest checkpoint at or before `at` (the common prefix with the held tokens), or null.
    const Ckpt* checkpoint_at_or_before(int64_t at) const {
        const Ckpt* best = nullptr;
        for (const Ckpt& c : sess[(size_t) cur].ckpts)
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
        std::deque<Ckpt>& ckpts = sess[(size_t) cur].ckpts;
        while (!ckpts.empty() && ckpts.back().pos > at) ckpts.pop_back();
    }
    void clear_checkpoints() { sess[(size_t) cur].ckpts.clear(); }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown (upstream 3921f57ee).
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr) {
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
                if (draft_stop && draft_stop(drafts[0])) D = 1;   // nothing follows the end of the answer
            }
            int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
            for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
                L.embed_host(&drafts[(size_t) d - 1], 1);
                L.hpin[1] = (int32_t) (pos + d);
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                CK(cudaMemcpyAsync(seeds, L.hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                last_row = 0;
                graph_run(gs().gm1, L.st, mtp_graph, [&] { forward_mtp(1, pos + d, seeds, true, mtp_graph); });
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[(size_t) d] = L.hpin[2];
                if (draft_stop && draft_stop(drafts[(size_t) d])) D = d + 1;
            }
        }
        drafts.resize((size_t) D);
        // the verify window
        const int T = D + 1;
        std::vector<int32_t> win;
        win.push_back(x_tok);
        for (int d = 0; d < D; ++d) win.push_back(drafts[(size_t) d]);
        first().embed_host(win.data(), T);
        set_pos(pos);
        set_deferred(true);
        graph_run(gs().gv[T], first().st, use_graph, [&] { forward(T, pos, T, use_graph, true); });
        set_deferred(false);
        {
            OnDevice od(L.dev);
            CK(cudaMemcpyAsync(L.hpin + 2, L.pick, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
            CK(cudaStreamSynchronize(L.st));
        }
        int a = 0;
        while (a < D && L.hpin[2 + a] == drafts[(size_t) a]) ++a;
        if (cancel != nullptr && cancel->load()) a = 0;
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
        // #4708: the prompt chunk may equal or exceed the context (--context 2048 with a 2048-token chunk): warm only cells that
        // exist, and start the small-prompt, decode and speculative probes in a fresh sequence instead of appending past the cache.
        std::vector<int32_t> toks((size_t) std::max(1, std::min(tmax, ctx)), filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        reset();
        if (tmax > Stage::kGemmT + 1 && ctx > Stage::kGemmT + 1) process(toks.data(), Stage::kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
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
            s->sw_n = 0;
            s->sw_ms = s->sw_fill_ms = s->fill_ms = 0;
            rb_swaps = 0;
            rb_ms = 0;
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
                if (draft_stop && draft_stop(drafts[0])) D = 1;
            }
            int last_row = T0 - 1;
            for (int d = 1; d < D; ++d) {
                L.embed_host(&drafts[(size_t) d - 1], 1);
                L.hpin[1] = (int32_t) (pos + d);
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                CK(cudaMemcpyAsync(seeds, L.hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                last_row = 0;
                graph_run(gs().gm1, L.st, mtp_graph, [&] { forward_mtp(1, pos + d, seeds, true, mtp_graph); });
                CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
                CK(cudaStreamSynchronize(L.st));
                drafts[(size_t) d] = L.hpin[2];
                if (draft_stop && draft_stop(drafts[(size_t) d])) D = d + 1;
            }
        }
        drafts.resize((size_t) D);
        // the verify window: every row's logits
        const int T = D + 1;
        std::vector<int32_t> win;
        win.push_back(x_tok);
        for (int d = 0; d < D; ++d) win.push_back(drafts[(size_t) d]);
        first().embed_host(win.data(), T);
        set_pos(pos);
        set_deferred(true);
        graph_run(gs().gv[T], first().st, use_graph, [&] { forward(T, pos, T, use_graph, true); });
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
    if (!E.use_pf()) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
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
        Plan& P = plans[i];
        if (!P.hybrid) continue;
        Stage& s = *E.stg[i];
        OnDevice od(s.dev);
        s.main_hybrid = P.hybrid && !P.mtp_only;   // this stage's own main layers (streamed prompt chunks, the batch size)
        s.hybrid_t = E.hybrid_t;
        s.hyT = s.main_hybrid ? std::max(E.hybrid_t, Stage::kVerify) : E.tmax;   // every batch is a hybrid batch when only the MTP block is
        // elastic: the prompt chunk's buffers and the streaming buffers only while a streamed prompt runs, their VRAM
        // holds extra slots otherwise (STRATA_Q36_NO_ELASTIC=1, --expert-slots or --expert-vram: as placed, always)
        s.elastic = s.main_hybrid && s.use_pf && slots <= 0 && vram_gib < 0 && std::getenv("STRATA_Q36_NO_ELASTIC") == nullptr;
        s.max_layer_bytes = P.max_layer;
        if (s.elastic) {   // the small set is allocated now so that the base slots leave room for it
            s.alloc_rows(s.rows_small, (size_t) s.hyT);
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            P.fr = fr;
            P.budget = fr > reserve ? fr - reserve : 0;
        }
        const size_t staging = s.use_pf && E.main_hybrid && !P.mtp_only ? 2 * (P.max_layer + kMmqTail) : 0;
        size_t slot_bytes = P.budget > staging ? P.budget - staging : 0;
        if (slot_bytes + staging + ((size_t) 256 << 20) > P.fr)
            slot_bytes = P.fr > staging + ((size_t) 512 << 20) ? P.fr - staging - ((size_t) 512 << 20) : 0;
        size_t per_slot = 0;   // one slot in every hybrid layer of this stage
        for (Layer* L : P.all)
            if (!P.mtp_only || L == &m.mtp) per_slot += L->stride;
        s.per_slot = per_slot;
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
            L->n_slots1 = S;
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
        if (staging) s.alloc_staging();
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
    E.enter_decode_phase();   // (the elastic stages: the chunk's buffers go, extra slots come)
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
    strata::qwen35::apply_penalties(lg, s, hist);
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

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.  (upstream b4ef48e94)
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block, greedy or sampled), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into
/// that slot's session - the decode windows of the other slots go on between its chunks - then its tokens come as BT
/// lines, one window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the
/// request on the control lines.  (upstream's sessions, batch slots and pictures, b4ef48e94 / 741903725, with this fork's
/// checkpoints, prompt tail, sampled speculative steps and device top-k sampling)
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg, tv;
    std::vector<int32_t> ti;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1)
    int read_windows = 1;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// The hidden state after a slot's last kept row of a window: its session's seed for a later MTP proposal or draft.
    void keep_hidden(int b, int row) {
        Stage& L = E.last();
        OnDevice od(L.dev);
        CK(cudaMemcpyAsync(E.sess_hlast(b), E.row_hidden(row), (size_t) E.m->g.n_embd * 4, cudaMemcpyDeviceToDevice, L.st));
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, E.sess_hlast(act[(size_t) i])});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new && ss.pos + 3 <= E.ctx;
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            keep_hidden(b, r0 + keep[(size_t) i] - 1);
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.row_hidden(r0)});
                nx.push_back({b, last[(size_t) i], p_new, E.row_hidden(r0 + 1)});
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.row_hidden(r0)});
            }
            nx_last.push_back((int) nx.size() - 1);
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            keep_hidden(b, (int) r);   // (the seed of a later proposal, or of the drafts when the session continues alone)
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }
    /// The prefix a session's checkpoints can give a request whose prompt does not extend its held tokens: the common prefix
    /// (not past the prompt's last token, nor into a picture the two do not share), then the latest checkpoint at or
    /// before it; 0 for none.
    size_t ckpt_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        size_t common = 0;
        while (common < ss.live.size() && common < ids.size() && ss.live[common] == ids[common]) ++common;
        common = std::min(common, ids.size() - 1);
        for (const ImgKey& k : imgs)
            if (k.cell < (int64_t) common && std::find(ss.imgs.begin(), ss.imgs.end(), k) == ss.imgs.end()) common = (size_t) k.cell;
        for (const ImgKey& k : ss.imgs)
            if (k.cell < (int64_t) common && std::find(imgs.begin(), imgs.end(), k) == imgs.end()) common = (size_t) k.cell;
        int64_t best = 0;
        for (const Engine::Ckpt& c : ss.ckpts)
            if (c.pos <= (int64_t) common && c.pos > best) best = c.pos;
        return (size_t) best;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt (its held tokens, or a
    /// checkpoint), else an empty one, else the least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = std::max(match_len(ss, ids, imgs), ckpt_len(ss, ids, imgs));
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
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
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back), or
        // from a checkpoint at or before the common prefix
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            const size_t ck = ckpt_len(ss, ids, ir.imgs);
            const Engine::Ckpt* c = ck > 0 ? E.checkpoint_at_or_before((int64_t) ck) : nullptr;
            if (c) {
                reuse = (size_t) c->pos;
                E.restore_checkpoint(*c);
                ss.live.resize(reuse);
                std::vector<ImgKey> kept;
                for (const ImgKey& k : ss.imgs)
                    if (k.cell + k.n <= (int64_t) reuse) kept.push_back(k);
                ss.imgs = kept;
            } else {
                E.reset();
                E.clear_checkpoints();
                ss.live.clear();
                ss.imgs.clear();
            }
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        auto progress = [&](int64_t base) {
            return [&, base](int64_t done) -> bool {
                const int64_t at = base + done;
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) fresh, ms_since(t0),
                            (double) at / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
                poll_control();
                if (n_active > 0 && at < fresh)   // the slots decoding go on between a read's chunks (the last chunk's
                    for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
                if (stop.load() && at < fresh) { cancelled = true; return false; }
                return true;
            };
        };
        // the prompt but its last kTail tokens in chunks (a checkpoint after every full one), a checkpoint, then the tail on
        // its own (a small batch: slots + the pool, no streaming): the next turn of a conversation shares the prompt up to
        // the answer's start, a few tokens before the end, and continues from that checkpoint.  Not with pictures in the
        // tail's reach (a picture's cells are read in one go).
        static const int64_t kTail = [] {   // STRATA_Q36_PROMPT_TAIL=N (0: the whole prompt in one go, as 0.1.44)
            const char* e = std::getenv("STRATA_Q36_PROMPT_TAIL");
            return e ? std::max<int64_t>(0, std::atoll(e)) : 16;
        }();
        bool tail_ok = kTail > 0 && fresh > kTail + 8;
        for (const ImgKey& k : ir.imgs)
            if (k.cell + k.n > (int64_t) ids.size() - kTail) tail_ok = false;
        E.checkpoint_chunks = true;
        E.draft_stop = is_eos;   // a window ends at an end-of-answer draft: the state never runs past the answer
        int64_t got = 0;
        if (tail_ok) {
            got = E.process(ids.data() + reuse, fresh - kTail, progress(0));
            if (!cancelled) {
                E.take_checkpoint();
                got += E.process(ids.data() + reuse + (fresh - kTail), kTail, progress(fresh - kTail));
            }
        } else {
            got = E.process(ids.data() + reuse, fresh, progress(0));
        }
        E.checkpoint_chunks = false;
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        // #4590: the penalties count the reply's tokens (and the fed-back last prompt token), not the prompt; STRATA_PENALTY_PROMPT=1
        // restores the old window.  `ids` is the request's full prompt whether or not the state was resumed.
        static const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, pen_prompt);
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {   // a batch slot: its first token now, the rest in the windows
            if (smp.needs_logits()) nx = sample_next(E, smp, hist, rng, lg, tv, ti);
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones too (no penalties, a top-k the device sort covers): the drafts verified against the sampled
        // distribution, so the text is a draw from the same sampler.  Beside decoding slots: one token at a time.
        const bool spec = E.mtp_d > 0 && !smp.needs_logits();
        const bool penal = smp.last_n != 0 && (smp.repeat != 1.f || smp.freq != 0.f || smp.present != 0.f);
        const bool spec_s = STRATA_Q36_DEVICE_TOPK && E.mtp_d > 0 && smp.needs_logits() && !penal && smp.temperature > 0.f &&
                            std::max(1, smp.top_k > 0 ? smp.top_k : 64) <= Stage::kTopMax &&
                            std::getenv("STRATA_Q36_NO_SPEC_SAMPLING") == nullptr;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        bool have_next = false;   // a speculative step decided nx already (its own draw)
        for (;;) {
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
            if (n_active > 0) window();
            // at most the tokens the limit still allows (the state then ends where the answer does, so a request that
            // extends the answer continues from it instead of a checkpoint before it)
            const int D = (int) std::min<long long>(E.mtp_d, max_new - produced);
            if ((spec || spec_s) && n_active == 0 && E.pos + D + 1 <= E.ctx) {
                const int a = spec ? E.spec_step(nx, fix, D, drafts, picks, &stop)
                                   : E.spec_step_sampled(nx, fix, D, drafts, picks, smp, rng, tv, ti);
                offered += (long long) drafts.size();   // (clamped at a rebalance boundary)
                accepted += a;
                ss.live.push_back(nx);
                fix.assign(drafts.begin(), drafts.begin() + a);
                ss.live.insert(ss.live.end(), drafts.begin(), drafts.begin() + a);   // (the engine holds them all, ended or not)
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                have_next = true;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        if (E.expert_info.experts > 0) std::fputs(strata::qwen35::expert_info_line(E.expert_info).c_str(), stdout);   // #5012
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
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
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    Serve sv(E);
    return sv.run();
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    int draft_vocab = -1;   // --draft-vocab N / STRATA_Q36_DRAFT_VOCAB: the MTP drafts over the first N token ids (0: all)
    double expert_vram = -1.0;
    std::string layer_split;   // "auto" (the default with two visible GPUs), a first layer of the second card, or 0
    int split_dev = -1;
    int pool_tasks = 0;        // row tasks per pool phase (0: three per thread)
    std::string kv = "f32";    // the KV cache: f32 (upstream's) or q8_0 (8-bit rows, a quarter of the VRAM)
    std::string pool_affinity = "all", pin_host = "auto";
    bool warm = true;
    int expert_slots = 0, batch = 0;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    std::string emb_file;
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
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
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
    // Strata's fused prompt kernels for this family's formats (moe_fused_iq_q36.cu): the K-quant, Q6_K and Q8_0 gate/up
    // instances are opt-in for Flash-Next's packs; this engine sets them up by default (=0 keeps a format off them), and in a
    // build without MMQ the Turing set-up too (own_only_switches.hpp, q36_entry_switches: the same switches on both entries)
    for (const char* v : strata::prefill::q36_entry_switches(MMQ::built())) {
#if !defined(_WIN32)
        setenv(v, "1", 0);
#else
        if (std::getenv(v) == nullptr) _putenv_s(v, "1");
#endif
    }
    // a build without llama.cpp's MMQ (0.2.16 on) has no MMQ tile for any layer: a layer takes the fused kernels, our int8
    // kernels or the grouped kernels.  `strata` sets STRATA_PF_FUSED=1 before main in such a build (prefill.cpp); the
    // standalone strata-q36 does not link that code, so it does the same here.  A value you set is kept.
    {
        const int pf_now = strata::prefill::fused_tile::pf_env(std::getenv("STRATA_PF_FUSED"));
        if (pf_now < 0 && strata::prefill::fused_tile::q36_start_env(MMQ::built(), pf_now) == 1) {
#if !defined(_WIN32)
            setenv("STRATA_PF_FUSED", "1", 1);
#else
            _putenv_s("STRATA_PF_FUSED", "1");
#endif
        }
    }
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
    sp.n_sess = std::max(1, std::min(batch, Stage::kVerify));
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
    if (batch > 1) {   // the sessions' caches, said before they are allocated
        size_t n_attn = m.has_mtp ? 1 : 0, n_gdn = 0;
        for (const Layer& L : m.layers) (L.attn ? n_attn : n_gdn) += 1;
        const auto& gg = m.g;
        const size_t per = (size_t) gg.n_head_kv * 2 * (sp.kv_q8 ? (size_t) kKvRowQ8 : (size_t) gg.head_dim * 4) * (size_t) ctx * n_attn +
                           (size_t) gg.ssm_v_heads * gg.ssm_state_size * gg.ssm_state_size * 4 * n_gdn;
        std::fprintf(stderr, "strata-q36: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in all, over "
                     "the cards)\n", sp.n_sess, (double) per / (1 << 30), ctx, (double) per * sp.n_sess / (1 << 30));
    }
    E.init(m, ctx, chunk, small, sp.kv_q8, batch, vision);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err, pool_tasks, pool_affinity)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    E.expert_info = strata::qwen35::expert_info(m.layers, m.g.n_expert);   // #5012: what the placement left in VRAM (base slots)
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
            if (E.sw_n() > 0)
                std::fprintf(stderr, "strata-q36: elastic VRAM: %d prompt/decode switches in %.0f ms; the extra slots' fill "
                             "%.0f ms, all in use %.0f ms after a prompt (averages); %d slot replacements after prompts in "
                             "%.0f ms\n", E.sw_n(), E.sw_ms(), E.fill_ms() / E.sw_n(), E.sw_fill_ms() / E.sw_n(), E.rb_swaps,
                             E.rb_ms);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.last().draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) &&
             E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
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
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-q36: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-q36: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
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
