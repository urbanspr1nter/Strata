// src/kernels/cpu/kq_vnni_parity.cpp - the AVX-VNNI K-quant rows (kq_vnni.cpp) against ggml-cpu's vec_dot, bit for
// bit: Q4_K, Q5_K and Q6_K rows of 512, 2048 (Qwen3.6-35B-A3B's n_ff and n_embd) and 2560 values against Q8_K
// activations, 1..8 activation rows, over the pool's whole and partial row ranges, on blocks from ggml's quantizers
// (random weights at three magnitudes) and on random bytes (every scale, min and quant the formats can hold, with
// activations down to -128, which ggml's quantizer never writes); then the engine's entry points (native_gu_rows /
// native_down_rows with STRATA_KQ_VNNI=1, which take the kernels from two rows on) on a Q4_K/Q5_K and a Q4_K/Q6_K
// expert against the same dots.  No GPU, no model.
//
//     kq_vnni_parity            (a CPU without AVX-VNNI: the kernels are skipped, the entry points still checked)
//     kq_vnni_parity --bench    us per expert (2048 x 512 gate/up, 512 x 2048 down) on one thread, weights in cache
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_vnni.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int MAXT = 8;
constexpr int QK = 256;
constexpr size_t kQ8KBlock = 292;   // sizeof(block_q8_K): float d, 256 int8, 16 int16 bsums

#if defined(_WIN32)
void set_env(const char* k, const char* v) { _putenv_s(k, v); }
#else
void set_env(const char* k, const char* v) { setenv(k, v, 1); }
#endif

struct Stat {
    long checked = 0, mism = 0;
    double max_rel = 0;
    void add(float got, float want) {
        ++checked;
        if (std::memcmp(&got, &want, 4) == 0) return;
        ++mism;
        const double rel = std::fabs((double) got - want) / (std::fabs((double) want) + 1e-30);
        max_rel = std::max(max_rel, rel);
    }
};

// a finite fp16 (exponent 1..30) over random bytes: a random-byte block must not run into Inf or NaN scales
void fix_half(uint8_t* p, std::mt19937& rng) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    h = (uint16_t) ((h & 0x83FFu) | (uint16_t) ((1 + rng() % 30) << 10));
    std::memcpy(p, &h, 2);
}

// a Q8_K row from random bytes: qs in [-128, 127], bsums the true sums of 16, d random
void raw_q8k_row(std::mt19937& rng, int n, uint8_t* dst) {
    std::uniform_int_distribution<int> qd(-128, 127);
    std::uniform_real_distribution<float> dd(1e-3f, 1.f);
    for (int b = 0; b < n / QK; ++b) {
        uint8_t* blk = dst + (size_t) b * kQ8KBlock;
        const float d = dd(rng);
        std::memcpy(blk, &d, 4);
        int8_t* qs = (int8_t*) (blk + 4);
        for (int k = 0; k < QK; ++k) qs[k] = (int8_t) qd(rng);
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += qs[16 * g + k];
            const int16_t s16 = (int16_t) s;
            std::memcpy(blk + 4 + QK + 2 * g, &s16, 2);
        }
    }
}

struct Rows {
    int type = 0, n = 0, R = 0;
    size_t row_bytes = 0;
    std::vector<uint8_t> w;                  // R weight rows
    std::vector<std::vector<uint8_t>> act;   // MAXT Q8_K rows
    const void* act_p[MAXT] = {};
};

// R rows of `type` x n: the first half from ggml's quantizer on random weights at three magnitudes, the second
// random bytes (any byte pattern is a block of these formats; the fp16 scales kept finite); MAXT activation rows,
// half from ggml's Q8_K quantizer, half raw
Rows make_rows(int type, int n, int R, std::mt19937& rng) {
    Rows rs;
    rs.type = type;
    rs.n = n;
    rs.R = R;
    rs.row_bytes = ggml_row_size((ggml_type) type, n);
    rs.w.resize(rs.row_bytes * (size_t) R);
    const auto* tr = ggml_get_type_traits_cpu((ggml_type) type);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n);
    const size_t block = ggml_type_size((ggml_type) type);
    for (int r = 0; r < R; ++r) {
        uint8_t* row = rs.w.data() + rs.row_bytes * (size_t) r;
        if (r < R / 2) {
            const float scale = r % 3 == 0 ? 0.02f : r % 3 == 1 ? 1.f : 30.f;
            for (auto& v : x) v = nd(rng) * scale;
            tr->from_float(x.data(), row, n);
        } else {
            for (size_t k = 0; k < rs.row_bytes; ++k) row[k] = (uint8_t) rng();
            for (int b = 0; b < n / QK; ++b) {
                uint8_t* blk = row + (size_t) b * block;
                if (type == 14) fix_half(blk + block - 2, rng);   // Q6_K: d at the end
                else { fix_half(blk, rng); fix_half(blk + 2, rng); }   // Q4_K / Q5_K: d, dmin at the start
            }
        }
    }
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t abytes = ggml_row_size(GGML_TYPE_Q8_K, n);
    rs.act.assign(MAXT, std::vector<uint8_t>(abytes));
    for (int t = 0; t < MAXT; ++t) {
        if (t < MAXT / 2) {
            const float scale = t == 1 ? 1e-2f : t == 2 ? 50.f : 1.f;
            for (auto& v : x) v = nd(rng) * scale;
            q8->from_float(x.data(), rs.act[(size_t) t].data(), n);
        } else {
            raw_q8k_row(rng, n, rs.act[(size_t) t].data());
        }
        rs.act_p[t] = rs.act[(size_t) t].data();
    }
    return rs;
}

float silu_mul(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }   // native_gu_rows' expression

// kq_vnni_rows and kq_vnni_gu_rows (the rows as gate rows, the same rows shifted by one as up rows) against ggml's
// dot, at 1..8 activation rows (a different subset of the eight for every count), whole and partial row ranges
void check_rows(const Rows& rs, Stat& st) {
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) rs.type)->vec_dot;
    std::vector<float> ref((size_t) MAXT * rs.R);
    for (int t = 0; t < MAXT; ++t)
        for (int r = 0; r < rs.R; ++r)
            dot(rs.n, &ref[(size_t) t * rs.R + r], 0, rs.w.data() + rs.row_bytes * (size_t) r, 0, rs.act_p[t], 0, 1);
    std::vector<std::vector<float>> outb(MAXT, std::vector<float>((size_t) rs.R));
    float* out[MAXT];
    for (int t = 0; t < MAXT; ++t) out[t] = outb[(size_t) t].data();
    const int Rg = rs.R - 1;   // gate rows 0..Rg-1, up rows 1..Rg (up_off = one row)
    for (int nt = 1; nt <= MAXT; ++nt) {
        const void* ap[MAXT];
        int src[MAXT];
        for (int t = 0; t < nt; ++t) { src[t] = (t + nt) % MAXT; ap[t] = rs.act_p[src[t]]; }
        for (int pass = 0; pass < 2; ++pass) {
            const int r0 = pass == 0 ? 0 : 3, r1 = pass == 0 ? rs.R : rs.R - 2;
            for (auto& o : outb) std::fill(o.begin(), o.end(), -12345.f);
            cpu::kq_vnni_rows(rs.type, rs.w.data(), rs.row_bytes, rs.n, ap, nt, out, r0, r1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) st.add(out[t][r], ref[(size_t) src[t] * rs.R + r]);
            const int g1 = std::min(r1, Rg);
            for (auto& o : outb) std::fill(o.begin(), o.end(), -12345.f);
            cpu::kq_vnni_gu_rows(rs.type, rs.w.data(), rs.row_bytes, rs.row_bytes, rs.n, ap, nt, out, r0, g1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < g1; ++r)
                    st.add(out[t][r], silu_mul(ref[(size_t) src[t] * rs.R + r], ref[(size_t) src[t] * rs.R + r + 1]));
        }
    }
}

// one expert in the GGUF form (gate, up, down), ggml's quantizers on random weights, as kq_cpu_bench builds them
std::vector<uint8_t> make_blob(const cpu::NativeFmt& f, std::mt19937& rng) {
    std::vector<uint8_t> blob(f.bytes);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(f.n_embd, f.n_ff));
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    for (int64_t r = 0; r < 2 * f.n_ff; ++r) {
        for (int64_t i = 0; i < f.n_embd; ++i) row[(size_t) i] = nd(rng);
        tg->from_float(row.data(), blob.data() + (size_t) r * f.gu_row, f.n_embd);
    }
    for (int64_t r = 0; r < f.n_embd; ++r) {
        for (int64_t i = 0; i < f.n_ff; ++i) row[(size_t) i] = nd(rng);
        td->from_float(row.data(), blob.data() + f.down_off + (size_t) r * f.d_row, f.n_ff);
    }
    return blob;
}

struct Expert {
    cpu::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<uint8_t>> xa, ha;
    const void* act[MAXT];
    const void* hq[MAXT];
};

Expert make_expert(int gu_type, int d_type, std::mt19937& rng) {
    Expert e;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, 2048, 512, e.f, err)) {
        std::printf("native_fmt: %s\n", err.c_str());
        std::exit(1);
    }
    e.blob = make_blob(e.f, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    e.xa.assign(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    e.ha.assign(MAXT, std::vector<uint8_t>(cpu::kNativeHBytes));
    for (int t = 0; t < MAXT; ++t) {
        std::vector<float> x((size_t) e.f.n_embd), h((size_t) e.f.n_ff);
        for (auto& v : x) v = nd(rng);
        for (auto& v : h) v = 0.1f * nd(rng);
        cpu::native_quant_act(e.f, x.data(), e.xa[(size_t) t].data());
        cpu::native_quant_h(e.f, h.data(), e.ha[(size_t) t].data());
        e.act[t] = e.xa[(size_t) t].data();
        e.hq[t] = e.ha[(size_t) t].data();
    }
    return e;
}

// ggml's per-row dots, as native_gu_rows / native_down_rows compute them without STRATA_KQ_VNNI
void ggml_expert(const Expert& e, int nt, float* const* ff, float* const* out) {
    const ggml_vec_dot_t dg = ggml_get_type_traits_cpu((ggml_type) e.f.gu_type)->vec_dot;
    const ggml_vec_dot_t dd = ggml_get_type_traits_cpu((ggml_type) e.f.d_type)->vec_dot;
    for (int r = 0; r < (int) e.f.n_ff; ++r)
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dg((int) e.f.n_embd, &g, 0, e.blob.data() + (size_t) r * e.f.gu_row, 0, e.act[t], 0, 1);
            dg((int) e.f.n_embd, &u, 0, e.blob.data() + e.f.up_off + (size_t) r * e.f.gu_row, 0, e.act[t], 0, 1);
            ff[t][r] = silu_mul(g, u);
        }
    for (int r = 0; r < (int) e.f.n_embd; ++r)
        for (int t = 0; t < nt; ++t)
            dd((int) e.f.n_ff, &out[t][r], 0, e.blob.data() + e.f.down_off + (size_t) r * e.f.d_row, 0, e.hq[t], 0, 1);
}

// the engine's entry points (STRATA_KQ_VNNI=1 is set before the first call) against ggml's dots
void check_expert(const Expert& e, Stat& st) {
    std::vector<std::vector<float>> fa(MAXT, std::vector<float>((size_t) e.f.n_ff)), fb = fa,
        oa(MAXT, std::vector<float>((size_t) e.f.n_embd)), ob = oa;
    float* ffa[MAXT];
    float* ffb[MAXT];
    float* outa[MAXT];
    float* outb[MAXT];
    for (int t = 0; t < MAXT; ++t) { ffa[t] = fa[(size_t) t].data(); ffb[t] = fb[(size_t) t].data(); outa[t] = oa[(size_t) t].data(); outb[t] = ob[(size_t) t].data(); }
    for (int nt = 1; nt <= MAXT; ++nt) {
        ggml_expert(e, nt, ffa, outa);
        cpu::native_gu_rows(e.f, e.blob.data(), e.act, nt, ffb, 0, (int) e.f.n_ff);
        cpu::native_down_rows(e.f, e.blob.data(), e.hq, nt, outb, 0, (int) e.f.n_embd);
        for (int t = 0; t < nt; ++t) {
            for (int r = 0; r < (int) e.f.n_ff; ++r) st.add(ffb[t][r], ffa[t][r]);
            for (int r = 0; r < (int) e.f.n_embd; ++r) st.add(outb[t][r], outa[t][r]);
        }
    }
}

void bench(const Expert& e) {
    std::vector<std::vector<float>> fa(MAXT, std::vector<float>((size_t) e.f.n_ff)), oa(MAXT, std::vector<float>((size_t) e.f.n_embd));
    float* ff[MAXT];
    float* out[MAXT];
    for (int t = 0; t < MAXT; ++t) { ff[t] = fa[(size_t) t].data(); out[t] = oa[(size_t) t].data(); }
    auto best_us = [&](auto fn) {
        double best = 1e30;
        for (int rep = 0; rep < 15; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    std::printf("  %s/%s expert (%lld x %lld), one thread, weights in cache: us per expert\n",
                ggml_type_name((ggml_type) e.f.gu_type), ggml_type_name((ggml_type) e.f.d_type), (long long) e.f.n_embd,
                (long long) e.f.n_ff);
    for (int nt : {1, 2, 3, 4, 8}) {
        const double g = best_us([&] { ggml_expert(e, nt, ff, out); });
        const double v = best_us([&] {
            cpu::kq_vnni_gu_rows(e.f.gu_type, e.blob.data(), e.f.gu_row, e.f.up_off, (int) e.f.n_embd, e.act, nt, ff, 0, (int) e.f.n_ff);
            cpu::kq_vnni_rows(e.f.d_type, e.blob.data() + e.f.down_off, e.f.d_row, (int) e.f.n_ff, e.hq, nt, out, 0, (int) e.f.n_embd);
        });
        std::printf("    nt %d: ggml %7.1f us  vnni %7.1f us  %.2fx  (%.1f / %.1f us per row)\n", nt, g, v, g / v, g / nt, v / nt);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    set_env("STRATA_KQ_VNNI", "1");   // before the first native_* call: the entry points then take the VNNI path
    ggml_cpu_init();
    const bool vnni = cpu::cpu_avxvnni_ok();
    std::printf("kq_vnni_parity: %s (avx2 %d, avx-vnni %d)\n", cpu::cpu_name().c_str(), (int) cpu::cpu_avx2_ok(), (int) vnni);
    std::mt19937 rng(2026);
    int failures = 0;
    if (vnni) {
        for (const int type : {12, 13, 14}) {
            Stat st;
            for (const int n : {512, 2048, 2560}) {
                const Rows rs = make_rows(type, n, 24, rng);
                check_rows(rs, st);
            }
            std::printf("  %s rows x Q8_K, nt 1..8: %ld values, %ld differ from ggml-cpu's dot%s\n", ggml_type_name((ggml_type) type),
                        st.checked, st.mism, st.mism ? (" (max rel " + std::to_string(st.max_rel) + ")").c_str() : " - identical");
            failures += st.mism != 0;
        }
    } else {
        std::printf("  no AVX-VNNI on this CPU (or STRATA_NO_AVXVNNI / STRATA_FORCE_ISA): the kernels are skipped\n");
    }
    for (const int d_type : {13, 14}) {
        const Expert e = make_expert(12, d_type, rng);
        Stat st;
        check_expert(e, st);
        std::printf("  native_gu_rows / native_down_rows, q4_K/%s expert, nt 1..8, STRATA_KQ_VNNI=1 (%s): %ld values, %ld differ%s\n",
                    ggml_type_name((ggml_type) d_type), vnni ? "the VNNI path from two rows on" : "ggml's path, no AVX-VNNI here", st.checked, st.mism,
                    st.mism ? (" (max rel " + std::to_string(st.max_rel) + ")").c_str() : " - identical");
        failures += st.mism != 0;
        if (do_bench && vnni) bench(e);
    }
    std::printf("kq_vnni_parity: %s\n", failures ? "FAILED" : "identical");
    return failures ? 1 : 0;
}
