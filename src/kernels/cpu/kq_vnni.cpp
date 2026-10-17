// src/kernels/cpu/kq_vnni.cpp - the K-quant expert rows of Qwen3.6-35B-A3B UD-Q4_K_XL (Q4_K gate/up, Q5_K and Q6_K
// down) against Q8_K activations, 1..8 activation rows per pass over the weights, on AVX-VNNI.  BIT-EXACT against
// ggml-cpu's AVX2 dots (ggml-cpu/arch/x86/quants.c, the __AVX2__ branches of ggml_vec_dot_q4_K_q8_K, _q5_K_q8_K and
// _q6_K_q8_K), which is what the pool computes by default (native_expert.cpp).
//
// What ggml computes, per 32-bit lane k of a block: an int32 sumi[k] = sum over the block's sub-blocks of
// scale * (maddubs pair sum 2k + pair sum 2k+1), then acc[k] = fma(d, float(sumi[k]), acc[k]) block after block,
// and one horizontal sum at the end (plus the mins' correction from the bsums).  The integers are exact - maddubs
// cannot saturate here (2 * 63 * 128 < 32768) and the int32 sums stay far below 2^31 - so any instruction that puts
// the same integer in the same lane gives the same bits, and a float operation applied lane by lane to eight tokens
// at once is the same operation as ggml's scalar one on each.  Here
//   - the u8 x s8 pair products stay maddubs; ggml's madd_epi16 + add_epi32 (scale times pair sums, accumulated)
//     become one vpdpwssd: two instructions per 32 weights and row instead of three;
//   - the float steps are ggml's, in ggml's order, for each activation row on its own: the block scale d (and -dmin)
//     of eight rows come from two vector multiplies, the mins' correction (madd of the mins with the bsums, convert,
//     fma) runs two rows per 256-bit register;
//   - the loop over the activation rows sits INSIDE the loop over the blocks: a block's nibbles, high bits, 6-bit
//     scales and mins are unpacked once and serve up to 8 rows (a decode window's verify batch), where ggml's
//     per-row dot unpacks them again for every row - the pool's cost was linear in the rows (AGENTS_FORK.md);
//   - the int32 accumulators of a row are split over several chains (vpdpwssd has a 5-cycle latency on Golden Cove)
//     and added at the block's end: the same integer, whichever chain a sub-block landed in.
// The row count is a compile-time constant (rows_nt<NT>), so the accumulators are registers: the 16 ymm registers
// hold the chains plus two weight vectors and their two scale vectors, which is why every format is unpacked 64
// weights at a time.  Measured on an i5-13400 P-core, a 2048 x 512 expert in cache (kq_vnni_parity --bench), us per
// expert, ggml's dots -> these: q4_K/q6_K 79 -> 82 at one row, 155 -> 111 at two, 307 -> 161 at four, 612 -> 302 at
// eight; q4_K/q5_K 88 -> 86, 171 -> 117, 341 -> 167, 680 -> 311.  One row is ggml's speed (its chain, not a port,
// is what binds there; an unpack hoisted before the block loop measured no faster); the pool's DRAM-streaming numbers
// are in AGENTS_FORK.md (kq_cpu_bench).
//
// Only reached behind cpu_avxvnni_ok() (on by default there; STRATA_KQ_VNNI=0 keeps ggml's path), from two rows on
// (native_expert.cpp: at one row ggml's dot is as fast on a P-core and faster on an E-core).  Compiled for AVX2 (-mavx2 -mfma -mf16c); the VNNI
// instructions are inside functions with target("avxvnni"), as in iq_avx2.cpp / q2_avx2.cpp (never -mavxvnni on a
// file: CMakeLists.txt).  A compiler without the intrinsics (STRATA_AVXVNNI=0) gets ggml's madd + add in their
// place, the same bits.
#include "strata/kernels/cpu/kq_vnni.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <cstring>

#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_AVXVNNI_FN
#endif
#if defined(__GNUC__) || defined(__clang__)
#define STRATA_PRAGMA_STR_(x) #x
#define STRATA_UNROLL(n) _Pragma(STRATA_PRAGMA_STR_(GCC unroll n))
#else
#define STRATA_UNROLL(n)
#endif

namespace strata::kernels::cpu {
namespace {

constexpr int MAXT = 8;    // activation rows per pass (the verify window's tokens per group, kVerifyMaxT)
constexpr int MAXB = 16;   // Q8_K blocks per row: native_fmt caps a row at kNativeActBytes = 14 blocks (2048: 8, 2560: 10)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} a Q4_K / Q5_K block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ggml's hsum_float_8 (ggml-cpu/arch/x86/quants.c), verbatim: the final reduction's order is part of the bits
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Tables written with scalar or vector stores and read back with broadcast loads (vpbroadcastw / vbroadcastss from
// memory: a load port, nothing else).  GCC turns such a store-then-load back into register shuffles (vpextrw, vmovd,
// vpbroadcastw: three port-5 uops per scale, 24 per block, measured 0.82x ggml at one row on an i5-13400); this
// tells it the object has been written and must be read from memory.
template <typename T>
inline void keep_in_memory(T& object) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+m"(object));
#else
    (void) object;
#endif
}

// acc[k] += a[2k] * b[2k] + a[2k+1] * b[2k+1] (int16 x int16 -> int32): ggml's _mm256_add_epi32(acc,
// _mm256_madd_epi16(a, b)) as one instruction.  Exact in both forms, so the same integers.  On an i5-13400 the
// instruction alone is no faster than the pair (eight independent chains, P-core: 2.4 ns per eight vpdpwssd against
// 2.1 per eight madd + add - one port for it, two for madd; E-core 5.2 against 4.0), but inside these kernels the one
// uop wins: the madd + add form measured 10-15% slower at 4-8 rows on the P-core (196 vs 168 us, 358 vs 310 us per
// q4_K/q5_K expert) and no faster on the E-core.
STRATA_AVXVNNI_FN inline __m256i dpwssd(__m256i acc, __m256i a, __m256i b) {
#if STRATA_AVXVNNI
    return _mm256_dpwssd_avx_epi32(acc, a, b);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
#endif
}

// int32 chains per activation row: 4 at one or two rows, 2 at three or four, one per row above (8 at most)
template <int NT> constexpr int chains() { return NT >= 5 ? 1 : NT >= 3 ? 2 : 4; }

// ggml's scalar float a * b + c on each compiler: GCC and Clang contract ggml's `summs += dmin * hsum` (Q5_K) to one
// fused multiply-add (checked in the engine's libggml-cpu.a: vcvtsi2ss + vfmadd231ss), MSVC under /fp:precise does
// not.  Lane-wise on a vector here (the lanes that matter hold the same operands as ggml's scalars).
inline __m256 ggml_fma_ps(__m256 a, __m256 b, __m256 c) {
#if defined(_MSC_VER) && !defined(__clang__)
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#else
    return _mm256_fmadd_ps(a, b, c);
#endif
}
inline float ggml_fma(float a, float b, float c) {
#if defined(_MSC_VER) && !defined(__clang__)
    return a * b + c;
#else
    return std::fma(a, b, c);
#endif
}

// ---- the activation rows of one pass (up to 8), prepared once per call
struct Pass {
    const block_q8_K* y[MAXT];
    alignas(32) float yd[MAXB][8];    // y[t][i].d per block, lane t (0 past nt)
    alignas(32) float ynd[MAXB][8];   // -y[t][i].d: ggml's `-y[i].d * dmin` negates first, exactly
    // ggml's q8s (the hadd of a block's 16 bsums into its 8 sub-block sums) of rows 2k (low half) and 2k+1 (high
    // half): the mins' correction then runs two rows per madd (zero past nt)
    __m256i q8s2[MAXB][MAXT / 2];
};

STRATA_AVXVNNI_FN void make_pass(const void* const* act, int nt, int nb, Pass& p) {
    for (int t = 0; t < nt; ++t) p.y[t] = (const block_q8_K*) act[t];
    for (int i = 0; i < nb; ++i) {
        for (int t = 0; t < MAXT; ++t) {
            const float d = t < nt ? p.y[t][i].d : 0.f;
            p.yd[i][t] = d;
            p.ynd[i][t] = -d;
        }
        for (int k = 0; k < MAXT / 2; ++k) {
            __m128i lo = _mm_setzero_si128(), hi = _mm_setzero_si128();
            if (2 * k < nt) {
                const __m256i s = _mm256_loadu_si256((const __m256i*) p.y[2 * k][i].bsums);
                lo = _mm_hadd_epi16(_mm256_extracti128_si256(s, 0), _mm256_extracti128_si256(s, 1));
            }
            if (2 * k + 1 < nt) {
                const __m256i s = _mm256_loadu_si256((const __m256i*) p.y[2 * k + 1][i].bsums);
                hi = _mm_hadd_epi16(_mm256_extracti128_si256(s, 0), _mm256_extracti128_si256(s, 1));
            }
            p.q8s2[i][k] = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        }
    }
}

// lanes 0-3 = v[2k], lanes 4-7 = v[2k+1]: the d (or dmin) pair of rows 2k and 2k+1 for one 256-bit float op
alignas(32) const int32_t k_pair_idx[MAXT / 2][8] = {{0, 0, 0, 0, 1, 1, 1, 1}, {2, 2, 2, 2, 3, 3, 3, 3},
                                                      {4, 4, 4, 4, 5, 5, 5, 5}, {6, 6, 6, 6, 7, 7, 7, 7}};
inline __m256 pair_of(__m256 v, int k) {
    return _mm256_permutevar8x32_ps(v, _mm256_load_si256((const __m256i*) k_pair_idx[k]));
}

// Q4_K / Q5_K's 6-bit scales and mins (ggml's utmp unpack; the two formats share the {d, dmin, scales[12]} head):
// sc16 = the 8 scales as int16, in memory for the broadcasts; returns the 8 mins as int16
inline __m128i unpack_scales_k4(const uint8_t* scales12, int16_t* sc16) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, scales12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1],
                                                                       (int) utmp[0]));
    _mm_store_si128((__m128i*) sc16, _mm256_castsi256_si128(mins_and_scales));
    return _mm256_extracti128_si256(mins_and_scales, 1);
}
// a block's scales into sc16 (the caller's array, kept in memory) and its mins into `mins`
#define STRATA_SCALES_K4(x, sc16, mins) \
    alignas(16) int16_t sc16[8]; \
    const __m128i mins = unpack_scales_k4((x).scales, sc16); \
    keep_in_memory(sc16)

// Q6_K: ggml's cvtepi8_epi16(shuffle_epi8(scales, get_scale_shuffle(k))) = scale 2k in the low eight int16 lanes,
// scale 2k+1 in the high eight: two broadcast loads and a blend instead of two port-5 shuffles
inline __m256i scale_pair_q6(const int16_t* sc16, int k) {
    return _mm256_blend_epi32(_mm256_set1_epi16(sc16[2 * k]), _mm256_set1_epi16(sc16[2 * k + 1]), 0xF0);
}

// the block scale d (and -dmin) of up to eight rows: ggml's `y[i].d * d` lane by lane
STRATA_AVXVNNI_FN inline __m256 row_scales(const float* yd, float xd) {
    return _mm256_mul_ps(_mm256_load_ps(yd), _mm256_broadcast_ss(&xd));
}

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__): res[t] = x . y[t]
template <int NT>
STRATA_AVXVNNI_FN void q4k_rows(const block_q4_K* x, int nb, const Pass& p, float* res) {
    constexpr int NA = chains<NT>(), NP = (NT + 1) / 2;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NT], acc_m2[NP];   // acc_m2[k]: ggml's acc_m of row 2k in the low half, of row 2k+1 in the high half
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int k = 0; k < NP; ++k) acc_m2[k] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        STRATA_SCALES_K4(x[i], sc16, mins);
        __m256i sumi[NT][NA];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < NA; ++a) sumi[t][a] = _mm256_setzero_si256();
        STRATA_UNROLL(4)
        for (int j = 0; j < QK_K / 64; ++j) {
            // ggml's scale_l / scale_h: shuffle_epi8 of the scales with its k_shuffle table = scale 2j / 2j+1 in
            // every int16 lane, a broadcast
            const __m256i scale_l = _mm256_set1_epi16(sc16[2 * j]);
            const __m256i scale_h = _mm256_set1_epi16(sc16[2 * j + 1]);
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            STRATA_UNROLL(8)
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = p.y[t][i].qs + 64 * j;
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                sumi[t][(2 * j) % NA] = dpwssd(sumi[t][(2 * j) % NA], scale_l, _mm256_maddubs_epi16(q4l, q8l));
                sumi[t][(2 * j + 1) % NA] = dpwssd(sumi[t][(2 * j + 1) % NA], scale_h, _mm256_maddubs_epi16(q4h, q8h));
            }
        }
        if constexpr (NT == 1) {   // one row: ggml's scalar d / dmin and 128-bit mins step (an E-core pays for 256-bit ones)
            const float d = p.y[0][i].d * xd, dmin = -p.y[0][i].d * xdmin;
            const __m128i prod = _mm_madd_epi16(mins, _mm256_castsi256_si128(p.q8s2[i][0]));
            const __m128 m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), _mm256_castps256_ps128(acc_m2[0]));
            acc_m2[0] = _mm256_castps128_ps256(m);   // the high half is never read
            __m256i s = sumi[0][0];
            for (int a = 1; a < NA; ++a) s = _mm256_add_epi32(s, sumi[0][a]);
            acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s), acc[0]);
        } else {
            alignas(32) float d8[8];
            _mm256_store_ps(d8, row_scales(p.yd[i], xd));
            keep_in_memory(d8);
            const __m256 dmin8 = row_scales(p.ynd[i], xdmin);
            const __m256i mins2 = _mm256_broadcastsi128_si256(mins);
            for (int k = 0; k < NP; ++k) {
                const __m256i prod = _mm256_madd_epi16(mins2, p.q8s2[i][k]);
                acc_m2[k] = _mm256_fmadd_ps(pair_of(dmin8, k), _mm256_cvtepi32_ps(prod), acc_m2[k]);
            }
            for (int t = 0; t < NT; ++t) {
                __m256i s = sumi[t][0];
                for (int a = 1; a < NA; ++a) s = _mm256_add_epi32(s, sumi[t][a]);
                acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d8[t]), _mm256_cvtepi32_ps(s), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = t & 1 ? _mm256_extractf128_ps(acc_m2[t / 2], 1) : _mm256_castps256_ps128(acc_m2[t / 2]);
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, __AVX2__)
template <int NT>
STRATA_AVXVNNI_FN void q5k_rows(const block_q5_K* x, int nb, const Pass& p, float* res) {
    constexpr int NA = chains<NT>(), NP = (NT + 1) / 2;
    const __m256i m4 = _mm256_set1_epi8(0xF), mone = _mm256_set1_epi8(1);
    __m256 acc[NT], summs2[NP];   // summs2[k]: ggml's scalar summs of row 2k in lane 0, of row 2k+1 in lane 4
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int k = 0; k < NP; ++k) summs2[k] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        STRATA_SCALES_K4(x[i], sc16, mins);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i sumi[NT][NA];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < NA; ++a) sumi[t][a] = _mm256_setzero_si256();
        STRATA_UNROLL(4)
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_set1_epi16(sc16[2 * j]);
            const __m256i scale_1 = _mm256_set1_epi16(sc16[2 * j + 1]);
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            // the fifth bits, ggml's way: bit 2j of each qh byte for the low nibbles, bit 2j+1 for the high ones
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), 2 * j), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), 2 * j + 1), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
            STRATA_UNROLL(8)
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = p.y[t][i].qs + 64 * j;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                sumi[t][(2 * j) % NA] = dpwssd(sumi[t][(2 * j) % NA], scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
                sumi[t][(2 * j + 1) % NA] = dpwssd(sumi[t][(2 * j + 1) % NA], scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
            }
        }
        // ggml: prod = madd(mins, q8s); hsum = hadd(hadd(prod, 0), 0) lane 0 = (p0 + p1) + (p2 + p3), an exact int,
        // converted to float; `summs += dmin * hsum` (ggml_fma).  Two rows per register: lane 0 for row 2k, lane 4
        // for 2k+1; one row: ggml's scalar step (an E-core pays for the 256-bit ones)
        if constexpr (NT == 1) {
            const float d = p.y[0][i].d * xd, dmin = -p.y[0][i].d * xdmin;
            const __m128i prod = _mm_madd_epi16(mins, _mm256_castsi256_si128(p.q8s2[i][0]));
            const __m128i h2 = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, 0xEE));
            const int hsum = _mm_cvtsi128_si32(_mm_add_epi32(h2, _mm_shuffle_epi32(h2, 0x55)));
            summs2[0] = _mm256_castps128_ps256(_mm_set_ss(ggml_fma(dmin, (float) hsum, _mm256_cvtss_f32(summs2[0]))));
            __m256i s = sumi[0][0];
            for (int a = 1; a < NA; ++a) s = _mm256_add_epi32(s, sumi[0][a]);
            acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s), acc[0]);
        } else {
            alignas(32) float d8[8];
            _mm256_store_ps(d8, row_scales(p.yd[i], xd));
            keep_in_memory(d8);
            const __m256 dmin8 = row_scales(p.ynd[i], xdmin);
            const __m256i mins2 = _mm256_broadcastsi128_si256(mins);
            for (int k = 0; k < NP; ++k) {
                const __m256i prod = _mm256_madd_epi16(mins2, p.q8s2[i][k]);
                const __m256i h2 = _mm256_add_epi32(prod, _mm256_shuffle_epi32(prod, 0xEE));   // p0+p2, p1+p3 per half
                const __m256i h = _mm256_add_epi32(h2, _mm256_shuffle_epi32(h2, 0x55));
                summs2[k] = ggml_fma_ps(pair_of(dmin8, k), _mm256_cvtepi32_ps(h), summs2[k]);
            }
            for (int t = 0; t < NT; ++t) {
                __m256i s = sumi[t][0];
                for (int a = 1; a < NA; ++a) s = _mm256_add_epi32(s, sumi[t][a]);
                acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d8[t]), _mm256_cvtepi32_ps(s), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        const float summs = t & 1 ? _mm_cvtss_f32(_mm256_extractf128_ps(summs2[t / 2], 1)) : _mm256_cvtss_f32(summs2[t / 2]);
        res[t] = hsum_float_8(acc[t]) + summs;
    }
}

// ---- Q6_K x Q8_K (ggml_vec_dot_q6_K_q8_K, __AVX2__): the -32 offset of the weights is taken off through the
// bsums (32 * scale * bsum per sub-block, `q8sclsub`), as ggml does; the 16 scales are int8
template <int NT>
STRATA_AVXVNNI_FN void q6k_rows(const block_q6_K* x, int nb, const Pass& p, float* res) {
    constexpr int NA = chains<NT>();
    const __m256i m3 = _mm256_set1_epi8(3), m15 = _mm256_set1_epi8(15);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) x[i].scales));
        // 32 * scale: ggml's (madd(bsums, scales) << 5) as madd(bsums, 32 * scales), exact, one shift per block
        const __m256i scales32 = _mm256_slli_epi16(scales_16, 5);
        alignas(32) int16_t sc16[16];
        _mm256_store_si256((__m256i*) sc16, scales_16);
        keep_in_memory(sc16);
        __m256i sumi[NT][NA];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < NA; ++a) sumi[t][a] = _mm256_setzero_si256();
        STRATA_UNROLL(2)
        for (int j = 0; j < QK_K / 128; ++j) {   // halves of 128 weights: 64 low nibbles' bytes hold 128 weights
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j));
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j + 32));
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) (x[i].qh + 32 * j));
            {   // weights 0..63 of the half: the low nibbles, with qh bits 0-1 and 2-3
                const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15),
                                                     _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4));
                const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15),
                                                     _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2));
                const __m256i scale_0 = scale_pair_q6(sc16, 4 * j + 0);
                const __m256i scale_1 = scale_pair_q6(sc16, 4 * j + 1);
                STRATA_UNROLL(8)
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = p.y[t][i].qs + 128 * j;
                    const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                    const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                    sumi[t][(4 * j) % NA] = dpwssd(sumi[t][(4 * j) % NA], scale_0, _mm256_maddubs_epi16(q4_0, q8_0));
                    sumi[t][(4 * j + 1) % NA] = dpwssd(sumi[t][(4 * j + 1) % NA], scale_1, _mm256_maddubs_epi16(q4_1, q8_1));
                }
            }
            {   // weights 64..127: the high nibbles, with qh bits 4-5 and 6-7
                const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),
                                                     _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48)));
                const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),
                                                     _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2));
                const __m256i scale_2 = scale_pair_q6(sc16, 4 * j + 2);
                const __m256i scale_3 = scale_pair_q6(sc16, 4 * j + 3);
                STRATA_UNROLL(8)
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = p.y[t][i].qs + 128 * j + 64;
                    const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8);
                    const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                    sumi[t][(4 * j + 2) % NA] = dpwssd(sumi[t][(4 * j + 2) % NA], scale_2, _mm256_maddubs_epi16(q4_2, q8_2));
                    sumi[t][(4 * j + 3) % NA] = dpwssd(sumi[t][(4 * j + 3) % NA], scale_3, _mm256_maddubs_epi16(q4_3, q8_3));
                }
            }
        }
        alignas(32) float d8[8];
        if constexpr (NT > 1) {
            _mm256_store_ps(d8, row_scales(p.yd[i], xd));
            keep_in_memory(d8);
        } else {
            d8[0] = p.y[0][i].d * xd;   // one row: ggml's scalar multiply
        }
        for (int t = 0; t < NT; ++t) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) p.y[t][i].bsums);
            const __m256i q8sclsub = _mm256_madd_epi16(q8sums, scales32);
            __m256i s = sumi[t][0];
            for (int a = 1; a < NA; ++a) s = _mm256_add_epi32(s, sumi[t][a]);
            s = _mm256_sub_epi32(s, q8sclsub);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d8[t]), _mm256_cvtepi32_ps(s), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

template <int NT>
STRATA_AVXVNNI_FN void dot_rows(int type, const uint8_t* row, int nb, const Pass& p, float* res) {
    switch (type) {
        case 12: q4k_rows<NT>((const block_q4_K*) row, nb, p, res); break;
        case 13: q5k_rows<NT>((const block_q5_K*) row, nb, p, res); break;
        case 14: q6k_rows<NT>((const block_q6_K*) row, nb, p, res); break;
        default: for (int t = 0; t < NT; ++t) res[t] = 0.f; break;   // unreachable: kq_vnni_supported gates the callers
    }
}

template <int NT>
STRATA_AVXVNNI_FN void gu_rows_nt(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int nb, const Pass& p,
                                  float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        dot_rows<NT>(type, blob + (size_t) r * gu_row, nb, p, g);
        dot_rows<NT>(type, blob + up_off + (size_t) r * gu_row, nb, p, u);
        // native_gu_rows' own SwiGLU expression: the same bits as the per-row ggml path
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
STRATA_AVXVNNI_FN void rows_nt(int type, const uint8_t* w, size_t row_bytes, int nb, const Pass& p, float* const* out,
                               int r0, int r1) {
    float s[NT];
    for (int r = r0; r < r1; ++r) {
        dot_rows<NT>(type, w + (size_t) r * row_bytes, nb, p, s);
        for (int t = 0; t < NT; ++t) out[t][r] = s[t];
    }
}

}  // namespace

void kq_vnni_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    assert(nb <= MAXB);
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        Pass p;
        make_pass(act + t0, m, nb, p);
        float* const* f = ff + t0;
        switch (m) {
            case 1: gu_rows_nt<1>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 2: gu_rows_nt<2>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 3: gu_rows_nt<3>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 4: gu_rows_nt<4>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 5: gu_rows_nt<5>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 6: gu_rows_nt<6>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            case 7: gu_rows_nt<7>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
            default: gu_rows_nt<8>(type, blob, gu_row, up_off, nb, p, f, r0, r1); break;
        }
    }
}

void kq_vnni_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    assert(nb <= MAXB);
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        Pass p;
        make_pass(act + t0, m, nb, p);
        float* const* o = out + t0;
        switch (m) {
            case 1: rows_nt<1>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 2: rows_nt<2>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 3: rows_nt<3>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 4: rows_nt<4>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 5: rows_nt<5>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 6: rows_nt<6>(type, w, row_bytes, nb, p, o, r0, r1); break;
            case 7: rows_nt<7>(type, w, row_bytes, nb, p, o, r0, r1); break;
            default: rows_nt<8>(type, w, row_bytes, nb, p, o, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
