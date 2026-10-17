// include/strata/kernels/cpu/kq_vnni.hpp - AVX-VNNI dot products for the K-quant expert rows of Qwen3.6-35B-A3B
// UD-Q4_K_XL (Q4_K gate/up, Q5_K / Q6_K down) against Q8_K activations, 1..8 activation rows per pass over the
// weights, bit-exact against ggml-cpu's AVX2 vec_dot for every row.  See src/kernels/cpu/kq_vnni.cpp.  On where
// cpu_avxvnni_ok() (native_expert.cpp; STRATA_KQ_VNNI=0 keeps ggml's path).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q4_K (12), Q5_K (13), Q6_K (14).  Answered in native_expert.cpp, not in the kernel file (#795).
bool kq_vnni_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows [r0, r1); gate rows at blob, up rows at blob + up_off;
/// a[t] a Q8_K row of n values.  Any nt (passes of up to 8 rows inside).
void kq_vnni_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t] for rows [r0, r1).
void kq_vnni_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
