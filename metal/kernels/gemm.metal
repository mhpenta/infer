// Matrix products for prompts: y = x·Wᵀ for many rows of x, on the
// simdgroup matrix units. x [m, k] float32 (rows ldx apart), y [m, n] (rows
// ldy apart), W [n, k] row-major in any of gemv's weight types, with gemv's
// scale and bias.

// gemm_mma_bf16 gives each threadgroup a BM×BN block of y and steps through
// k BK at a time. Each step, the threadgroup's threads copy the BM×BK block
// of x and the BN×BK block of W, widened to float32, into threadgroup
// memory (zero past the edges of x, W or k), and each of its WM×WN
// simdgroups multiplies 8×8 tiles out of them into its (BM/WM)×(BN/WN)
// share of the block, accumulating in float32. Outputs are scaled (int8)
// and biased on the way out. Grid: (ceil(n / BN), ceil(m / BM)) threadgroups of
// 32·WM·WN threads.
template <typename T, uint BM, uint BN, uint BK, uint WM, uint WN>
kernel void gemm_mma(device const T *w [[buffer(0)]],
                     device const float *x [[buffer(1)]],
                     device float *y [[buffer(2)]],
                     constant GemmArgs &a [[buffer(3)]],
                     device const float *scale [[buffer(4)]],
                     device const float *bias [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SM = BM / WM, SN = BN / WN; // a simdgroup's share
    constexpr uint TM = SM / 8, TN = SN / 8;   // in 8×8 tiles
    constexpr uint THREADS = 32 * WM * WN;
    constexpr uint LD = BK + 4; // padded rows, still 16-byte aligned
    threadgroup float xs[BM * LD];
    threadgroup float ws[BN * LD];
    threadgroup float stage[WM * WN][64];

    uint m0 = tg.y * BM, n0 = tg.x * BN;
    uint sm = (sg / WN) * SM, sn = (sg % WN) * SN;
    simdgroup_float8x8 acc[TM][TN];
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k0 = 0; k0 < a.k; k0 += BK) {
        for (uint e = t; e < BM * BK / 4; e += THREADS) {
            uint r = e / (BK / 4), c = e % (BK / 4) * 4;
            float4 v = 0;
            if (m0 + r < a.m && k0 + c < a.k) {
                v = *(device const float4 *)(x + size_t(m0 + r) * a.ldx + k0 + c);
            }
            *(threadgroup float4 *)(xs + r * LD + c) = v;
        }
        for (uint e = t; e < BN * BK / 4; e += THREADS) {
            uint r = e / (BK / 4), c = e % (BK / 4) * 4;
            float4 v = 0;
            if (n0 + r < a.n && k0 + c < a.k) {
                v = widen4(w + size_t(n0 + r) * a.k + k0 + c);
            }
            *(threadgroup float4 *)(ws + r * LD + c) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 A[TM], B[TN];
            for (uint i = 0; i < TM; i++) {
                simdgroup_load(A[i], xs + (sm + 8 * i) * LD + kk, LD);
            }
            for (uint j = 0; j < TN; j++) {
                // W's rows are y's columns: load the tile transposed.
                simdgroup_load(B[j], ws + (sn + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool full = !is_same<T, char>::value && !a.bias && m0 + BM <= a.m && n0 + BN <= a.n;
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            uint r0 = m0 + sm + 8 * i, c0 = n0 + sn + 8 * j;
            if (full) {
                simdgroup_store(acc[i][j], y + size_t(r0) * a.ldy + c0, a.ldy);
                continue;
            }
            // An edge block, or int8, or a bias: through threadgroup
            // memory, keeping what fits.
            simdgroup_store(acc[i][j], stage[sg], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint r = r0 + e / 8, c = c0 + e % 8;
                if (r < a.m && c < a.n) {
                    y[size_t(r) * a.ldy + c] = epilogue<T>(stage[sg][e], c, scale, bias, a);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define GEMM_MMA(T, NAME, BM, BN, BK, WM, WN) \
    template [[host_name("gemm_mma_" NAME "_" #BM "x" #BN "x" #BK "_" #WM "x" #WN)]] [[kernel]] \
    decltype(gemm_mma<T, BM, BN, BK, WM, WN>) gemm_mma<T, BM, BN, BK, WM, WN>;

// The ones Kernels uses, then others for BenchmarkGemmVariants. A
// simdgroup's share is 32×16 (8 tiles) in the first four and the last,
// 16×16 in the fifth, and 32×32 in the sixth, which is several times
// slower: 16 accumulator tiles do not fit in registers.
GEMM_MMA(ushort, "bf16", 64, 64, 16, 2, 4)
GEMM_MMA(half, "f16", 64, 64, 16, 2, 4)
GEMM_MMA(float, "f32", 64, 64, 16, 2, 4)
GEMM_MMA(char, "int8", 64, 64, 16, 2, 4)
GEMM_MMA(ushort, "bf16", 32, 32, 16, 2, 2)
GEMM_MMA(ushort, "bf16", 64, 64, 16, 2, 2)
GEMM_MMA(ushort, "bf16", 128, 64, 16, 4, 4)
