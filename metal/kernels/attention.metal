// Rotary embeddings, the key/value cache, and attention over it: causal
// (decoders), or over every cached position (encoders, cross-attention),
// with an optional additive bias (T5's relative positions).
//
// A layer's cache is two buffers, keys and values, each [kvHeads, capacity,
// dim] float32: a head's positions are contiguous rows of dim values.

struct RopeArgs {
    uint ld;    // floats from one row to the next
    uint half_; // dim / 2
};

// rope rotates, in place, consecutive heads of 2·half values at the start
// of each row of x: dimension j with j+half by the angle whose cosine and
// sine are cos[r·half + j] and sin[r·half + j] (ops.Rotation's tables).
// Each value is two rounded products and a rounded sum, as ops.Rotation.Apply
// computes it: fma(x, c, 0) is the product rounded once, and writing it so
// keeps the compiler from fusing a product into the sum. Grid: (half, heads,
// rows).
kernel void rope(device float *x [[buffer(0)]],
                 device const float *cos [[buffer(1)]],
                 device const float *sin [[buffer(2)]],
                 constant RopeArgs &a [[buffer(3)]],
                 uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    device float *p = x + size_t(gid.z) * a.ld + size_t(gid.y) * 2 * a.half_;
    float c = cos[gid.z * a.half_ + j], s = sin[gid.z * a.half_ + j];
    float x1 = p[j], x2 = p[j + a.half_];
    p[j] = fma(x1, c, 0.0f) + fma(-x2, s, 0.0f);
    p[j + a.half_] = fma(x2, c, 0.0f) + fma(x1, s, 0.0f);
}

struct AppendArgs {
    uint ld;       // floats from one source row to the next
    uint dim;      // values per head
    uint pos;      // cache position of the first row
    uint capacity; // positions per head in the cache
};

// kv_append copies rows of keys and values, each heads·dim wide at the start
// of the row, into the caches at positions pos.. Grid: (dim, heads, rows).
kernel void kv_append(device const float *k [[buffer(0)]],
                      device const float *v [[buffer(1)]],
                      device float *keys [[buffer(2)]],
                      device float *values [[buffer(3)]],
                      constant AppendArgs &a [[buffer(4)]],
                      uint3 gid [[thread_position_in_grid]]) {
    size_t src = size_t(gid.z) * a.ld + gid.y * a.dim + gid.x;
    size_t dst = (size_t(gid.y) * a.capacity + a.pos + gid.z) * a.dim + gid.x;
    keys[dst] = k[src];
    values[dst] = v[src];
}

struct AttendArgs {
    uint ldq, ldo;       // floats between rows of q and of out
    uint heads, kvHeads; // query head h reads key/value head h / (heads / kvHeads)
    uint dim;            // a multiple of 32, at most 256 (attend_tiled: of 8, at most 128)
    uint capacity;       // positions per head in the cache
    uint offset;         // with causal, the cache position of query row 0
    uint rows;           // query rows
    uint keys;           // cache positions in use: rows see positions below it
    uint causal;         // 1: row r sees only positions up to offset+r
    uint bias;           // 1: add bias[(h·rows + r)·keys + j] to the scaled scores
    float scale;
};

// sees returns the last position query row r sees.
static inline uint sees(constant AttendArgs &a, uint r) {
    return a.causal ? a.offset + r : a.keys - 1;
}

constant constexpr uint ATTEND_SG = 8;

// attend computes attention for each query row and head: softmax(scale·q·kᵀ
// + bias)·v over the cache positions the row sees. One threadgroup per
// (head, row). Each simdgroup takes every ATTEND_SG-th position and keeps
// an online softmax: the running maximum, the sum of exponentials, and the
// weighted sum of values, rescaled when the maximum grows. Lane l holds
// dimensions l, l+32, ... The simdgroups' partial results are then combined.
kernel void attend(device const float *q [[buffer(0)]],
                   device const float *keys [[buffer(1)]],
                   device const float *values [[buffer(2)]],
                   device float *out [[buffer(3)]],
                   constant AttendArgs &a [[buffer(4)]],
                   device const float *bias [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxes[ATTEND_SG], sums[ATTEND_SG];
    threadgroup float partial[ATTEND_SG][256];
    uint h = tg.x, row = tg.y, per = a.dim / 32;
    size_t head = size_t(h / (a.heads / a.kvHeads)) * a.capacity * a.dim;
    device const float *qp = q + size_t(row) * a.ldq + h * a.dim;
    float qv[8], acc[8];
    for (uint i = 0; i < per; i++) {
        qv[i] = qp[lane + 32 * i];
        acc[i] = 0;
    }
    float m = -INFINITY, l = 0;
    uint n = sees(a, row) + 1;
    device const float *rowBias = bias + (size_t(h) * a.rows + row) * a.keys;
    for (uint j = sg; j < n; j += ATTEND_SG) {
        device const float *kp = keys + head + size_t(j) * a.dim;
        device const float *vp = values + head + size_t(j) * a.dim;
        float s = 0;
        for (uint i = 0; i < per; i++) {
            s += qv[i] * kp[lane + 32 * i];
        }
        s = simd_sum(s) * a.scale;
        if (a.bias) {
            s += rowBias[j];
        }
        float mNew = max(m, s);
        float rescale = exp(m - mNew), p = exp(s - mNew);
        l = l * rescale + p;
        for (uint i = 0; i < per; i++) {
            acc[i] = acc[i] * rescale + p * vp[lane + 32 * i];
        }
        m = mNew;
    }
    if (lane == 0) {
        maxes[sg] = m;
        sums[sg] = l;
    }
    for (uint i = 0; i < per; i++) {
        partial[sg][lane + 32 * i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float mAll = -INFINITY;
    for (uint s = 0; s < ATTEND_SG; s++) {
        mAll = max(mAll, maxes[s]);
    }
    float lAll = 0, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint s = 0; s < ATTEND_SG; s++) {
        float c = exp(maxes[s] - mAll); // 0 for a simdgroup that saw no position
        lAll += sums[s] * c;
        for (uint i = 0; i < per; i++) {
            o[i] += partial[s][lane + 32 * i] * c;
        }
    }
    device float *op = out + size_t(row) * a.ldo + h * a.dim;
    for (uint i = 0; i < per; i++) {
        op[lane + 32 * i] = o[i] / lAll;
    }
}

constant constexpr uint TILE_Q = 32; // query rows per threadgroup, 8 per simdgroup
constant constexpr uint TILE_K = 16; // cache positions per step
constant constexpr uint TILE_D = 128; // the largest head dimension

// attend_tiled is attend for many query rows at once, FlashAttention-style,
// on the simdgroup matrix units. A threadgroup takes TILE_Q rows of one
// head; each of its 4 simdgroups owns 8 of them, holding their queries and
// their running context as 8×8 matrices. The threadgroup steps through the
// cache TILE_K positions at a time (as far as its last row can see),
// copying keys and values into threadgroup memory. Each step, a simdgroup
// computes its 8×TILE_K scores (q·kᵀ), and its lanes, four per row, mask and
// scale them, update the row's running maximum and sum of exponentials,
// and turn the scores into weights. The context is rescaled by the change
// in maximum (a product with a diagonal matrix) and the weights' product
// with the values is added. dim must be a multiple of 8, at most TILE_D.
// Grid: (heads, ceil(rows / TILE_Q)) threadgroups of 128 threads.
kernel void attend_tiled(device const float *q [[buffer(0)]],
                         device const float *keys [[buffer(1)]],
                         device const float *values [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         constant AttendArgs &a [[buffer(4)]],
                         device const float *bias [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    // Queries, then keys and values, then the final context, share mem.
    threadgroup float mem[TILE_Q * TILE_D];
    threadgroup float scores[4][8 * TILE_K];
    threadgroup float diag[4][64];
    threadgroup float sums[4][8];
    uint h = tg.x, r0 = tg.y * TILE_Q, dim = a.dim, dt = dim / 8, rows = a.rows;
    size_t head = size_t(h / (a.heads / a.kvHeads)) * a.capacity * dim;

    for (uint e = t; e < TILE_Q * dim; e += 128) {
        uint r = e / dim, d = e % dim;
        mem[r * dim + d] = r0 + r < rows ? q[size_t(r0 + r) * a.ldq + h * dim + d] : 0.0f;
    }
    for (uint e = lane; e < 64; e += 32) {
        diag[sg][e] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[TILE_D / 8], O[TILE_D / 8];
    for (uint d = 0; d < dt; d++) {
        simdgroup_load(Q[d], mem + sg * 8 * dim + 8 * d, dim);
        O[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane l works on row l/4 of its simdgroup's 8, columns 4·(l%4)...
    uint row = lane / 4, part = lane % 4;
    uint r = r0 + sg * 8 + row, see = sees(a, r); // this lane's row, and the last position it sees
    device const float *rowBias = bias + (size_t(h) * rows + r) * a.keys;
    bool biased = a.bias && r < rows;
    float m = -INFINITY, l = 0;
    uint last = min(sees(a, r0 + TILE_Q - 1) + 1, a.keys); // positions the threadgroup needs
    threadgroup float *ks = mem, *vs = mem + TILE_K * dim;
    for (uint j0 = 0; j0 < last; j0 += TILE_K) {
        for (uint e = t; e < TILE_K * dim; e += 128) {
            uint j = j0 + e / dim;
            bool in = j < a.keys;
            ks[e] = in ? keys[head + size_t(j) * dim + e % dim] : 0.0f;
            vs[e] = in ? values[head + size_t(j) * dim + e % dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint jt = 0; jt < TILE_K / 8; jt++) {
            simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), K;
            for (uint d = 0; d < dt; d++) {
                simdgroup_load(K, ks + jt * 8 * dim + 8 * d, dim, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S, Q[d], K, S);
            }
            simdgroup_store(S, scores[sg] + jt * 8, TILE_K);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[4], peak = -INFINITY;
        for (uint c = 0; c < 4; c++) {
            uint j = j0 + part * 4 + c;
            s[c] = j <= see ? scores[sg][row * TILE_K + part * 4 + c] * a.scale : -INFINITY;
            if (biased && j <= see) {
                s[c] += rowBias[j];
            }
            peak = max(peak, s[c]);
        }
        peak = max(peak, simd_shuffle_xor(peak, 1));
        peak = max(peak, simd_shuffle_xor(peak, 2));
        float mNew = max(m, peak), sum = 0;
        for (uint c = 0; c < 4; c++) {
            float p = exp(s[c] - mNew);
            scores[sg][row * TILE_K + part * 4 + c] = p;
            sum += p;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        float rescale = exp(m - mNew);
        l = l * rescale + sum;
        m = mNew;
        if (part == 0) {
            diag[sg][row * 9] = rescale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 D, P[TILE_K / 8], V;
        simdgroup_load(D, diag[sg], 8);
        for (uint jt = 0; jt < TILE_K / 8; jt++) {
            simdgroup_load(P[jt], scores[sg] + jt * 8, TILE_K);
        }
        for (uint d = 0; d < dt; d++) {
            simdgroup_multiply(O[d], D, O[d]);
            for (uint jt = 0; jt < TILE_K / 8; jt++) {
                simdgroup_load(V, vs + jt * 8 * dim + 8 * d, dim);
                simdgroup_multiply_accumulate(O[d], P[jt], V, O[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (part == 0) {
        sums[sg][row] = l;
    }
    threadgroup float *ctx = mem + sg * 8 * dim;
    for (uint d = 0; d < dt; d++) {
        simdgroup_store(O[d], ctx + 8 * d, dim);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * dim; e += 32) {
        uint ro = r0 + sg * 8 + e / dim;
        if (ro < rows) {
            out[size_t(ro) * a.ldo + h * dim + e % dim] = ctx[e] / sums[sg][e / dim];
        }
    }
}
