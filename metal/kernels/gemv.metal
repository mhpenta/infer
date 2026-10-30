// Matrix-vector products for decoding: y = W·x (+ bias), W [n, k]
// row-major, x [k] float32, y [n] float32. W is bf16, fp16 or float32 as a
// checkpoint stores it, or int8 with a float32 scale per row (output),
// quantized as ops.PackInt8 quantizes: y[j] = scale[j]·Σ q[j,k]·x[k].
// Arguments are GemmArgs, of which m, ldx and ldy are not read.

// gemv gives each simdgroup R rows of W. Lane l reads 8 consecutive weights
// of each row per step (two 4-value loads: bf16 rows are only guaranteed
// 8-byte aligned), so a simdgroup streams 256 contiguous weights of a row at
// a time, and x is loaded once for all R rows. Rows past n reread row n-1
// rather than branch, and are not written. scale is read only for int8 and
// bias only when a.bias is set (the launcher binds any buffer otherwise).
// Grid: ceil(n / (R·SG)) threadgroups of 32·SG threads.
template <typename T, uint R, uint SG>
kernel void gemv(device const T *w [[buffer(0)]],
                 device const float *x [[buffer(1)]],
                 device float *y [[buffer(2)]],
                 constant GemmArgs &a [[buffer(3)]],
                 device const float *scale [[buffer(4)]],
                 device const float *bias [[buffer(5)]],
                 uint tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * SG + sg) * R;
    if (row0 >= a.n) {
        return;
    }
    device const T *rows[R];
    for (uint r = 0; r < R; r++) {
        rows[r] = w + size_t(min(row0 + r, a.n - 1)) * a.k;
    }
    float acc[R];
    for (uint r = 0; r < R; r++) {
        acc[r] = 0;
    }
    uint full = a.k - a.k % 256;
    for (uint j = lane * 8; j < full; j += 256) {
        float4 x0 = *(device const float4 *)(x + j);
        float4 x1 = *(device const float4 *)(x + j + 4);
        for (uint r = 0; r < R; r++) {
            acc[r] += dot(widen4(rows[r] + j), x0) + dot(widen4(rows[r] + j + 4), x1);
        }
    }
    for (uint j = full + lane * 4; j < a.k; j += 128) {
        float4 x0 = *(device const float4 *)(x + j);
        for (uint r = 0; r < R; r++) {
            acc[r] += dot(widen4(rows[r] + j), x0);
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < a.n) {
            y[row0 + r] = epilogue<T>(s, row0 + r, scale, bias, a);
        }
    }
}

#define GEMV(T, NAME, R, SG) \
    template [[host_name("gemv_" NAME "_r" #R "_sg" #SG)]] [[kernel]] decltype(gemv<T, R, SG>) gemv<T, R, SG>;

// The one Kernels uses for each weight type, then others for
// BenchmarkGemvVariants.
GEMV(ushort, "bf16", 4, 8)
GEMV(half, "f16", 4, 8)
GEMV(float, "f32", 4, 8)
GEMV(char, "int8", 4, 8)
GEMV(ushort, "bf16", 2, 4)
GEMV(ushort, "bf16", 2, 8)
GEMV(ushort, "bf16", 4, 4)
GEMV(ushort, "bf16", 8, 4)
GEMV(char, "int8", 8, 4)
