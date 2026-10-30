#include <metal_stdlib>
using namespace metal;

kernel void empty() {}

// y[row] = sum_k bf16(W[row,k]) * x[k]. Each simdgroup owns 4 rows; lanes
// stride along k reading 8 bf16 (16 bytes) at a time, then reduce. Assumes
// rows % 32 == 0 and k % 256 == 0, which the spike's shapes satisfy.
kernel void gemv_bf16(device const ushort *w [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      device float *y [[buffer(2)]],
                      constant int &k [[buffer(3)]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint tg [[threadgroup_position_in_grid]]) {
    uint row0 = (tg * 8 + sg) * 4;
    float acc[4] = {0, 0, 0, 0};
    for (int j = lane * 8; j < k; j += 32 * 8) {
        float4 xa = *(device const float4 *)(x + j);
        float4 xb = *(device const float4 *)(x + j + 4);
        for (int r = 0; r < 4; r++) {
            ushort4 a = *(device const ushort4 *)(w + (row0 + r) * k + j);
            ushort4 b = *(device const ushort4 *)(w + (row0 + r) * k + j + 4);
            float4 fa = as_type<float4>(uint4(a) << 16);
            float4 fb = as_type<float4>(uint4(b) << 16);
            acc[r] += dot(fa, xa) + dot(fb, xb);
        }
    }
    for (int r = 0; r < 4; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0) y[row0 + r] = s;
    }
}
