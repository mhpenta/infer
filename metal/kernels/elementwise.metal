// Normalization, activations and the other small kernels of a decoder layer.

struct NormArgs {
    uint ld;     // floats from one row to the next
    uint groups; // groups normalized separately per row (heads, or 1)
    uint dim;    // values per group
    float eps;
};

// rmsnorm sets dst to src scaled to unit root mean square, then by w, for
// each group of dim values: group g of row r starts at r·ld + g·dim in both
// src and dst, which may be the same buffer. One threadgroup per (row,
// group). Sums are float32 (Metal has no float64), reduced as a tree.
kernel void rmsnorm(device const float *src [[buffer(0)]],
                    device float *dst [[buffer(1)]],
                    device const float *w [[buffer(2)]],
                    constant NormArgs &a [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint t [[thread_position_in_threadgroup]],
                    uint nt [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    size_t off = size_t(tg / a.groups) * a.ld + (tg % a.groups) * a.dim;
    float sum = 0;
    for (uint j = t; j < a.dim; j += nt) {
        float v = src[off + j];
        sum += v * v;
    }
    float inv = 1 / sqrt(tg_sum(sum, scratch, sg, lane, nsg) / float(a.dim) + a.eps);
    for (uint j = t; j < a.dim; j += nt) {
        dst[off + j] = w[j] * (src[off + j] * inv);
    }
}

// swiglu sets out [rows, ff] to SiLU(gate)·up, where row r of gateUp holds
// gate in its first ff columns and up in the next ff. Grid: (ff, rows).
kernel void swiglu(device const float *gateUp [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   constant uint &ff [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    size_t row = size_t(gid.y) * 2 * ff;
    float g = gateUp[row + gid.x];
    out[size_t(gid.y) * ff + gid.x] = g / (1 + exp(-g)) * gateUp[row + ff + gid.x];
}

// add adds src to dst element-wise. Grid: (n).
kernel void add(device float *dst [[buffer(0)]],
                device const float *src [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    dst[i] += src[i];
}

// embed sets row r of dst [rows, cols] to row ids[r] of a table of bf16,
// fp16 or float32 values, widened exactly. Grid: (cols, rows).
template <typename T>
kernel void embed(device const T *table [[buffer(0)]],
                  device const int *ids [[buffer(1)]],
                  device float *dst [[buffer(2)]],
                  constant uint &cols [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]]) {
    dst[size_t(gid.y) * cols + gid.x] = widen(table[size_t(ids[gid.y]) * cols + gid.x]);
}

template [[host_name("embed_bf16")]] [[kernel]] decltype(embed<ushort>) embed<ushort>;
template [[host_name("embed_f16")]] [[kernel]] decltype(embed<half>) embed<half>;
template [[host_name("embed_f32")]] [[kernel]] decltype(embed<float>) embed<float>;

// add_row adds row [cols] to every row of x [rows, cols]. Grid: (cols, rows).
kernel void add_row(device float *x [[buffer(0)]],
                    device const float *row [[buffer(1)]],
                    constant uint &cols [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    x[size_t(gid.y) * cols + gid.x] += row[gid.x];
}

struct LayerNormArgs {
    uint dim;
    uint residual; // 1: add residual to x first
    float eps;
};

// layernorm sets each row of x [rows, dim] to LayerNorm(x (+ residual))·w +
// b, in place: the sum is stored back, its mean taken, then the mean of
// squared deviations, as ops.LayerNorm computes them, but in float32. One
// threadgroup per row.
kernel void layernorm(device float *x [[buffer(0)]],
                      device const float *residual [[buffer(1)]],
                      device const float *w [[buffer(2)]],
                      device const float *b [[buffer(3)]],
                      constant LayerNormArgs &a [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint t [[thread_position_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    device float *row = x + size_t(tg) * a.dim;
    float sum = 0;
    for (uint j = t; j < a.dim; j += nt) {
        float v = row[j];
        if (a.residual) {
            v += residual[size_t(tg) * a.dim + j];
            row[j] = v;
        }
        sum += v;
    }
    float mean = tg_sum(sum, scratch, sg, lane, nsg) / float(a.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup); // scratch is reused
    float sq = 0;
    for (uint j = t; j < a.dim; j += nt) {
        float d = row[j] - mean;
        sq += d * d;
    }
    float inv = 1 / sqrt(tg_sum(sq, scratch, sg, lane, nsg) / float(a.dim) + a.eps);
    for (uint j = t; j < a.dim; j += nt) {
        row[j] = (row[j] - mean) * inv * w[j] + b[j];
    }
}

// erf_approx is ops' erff: a ratio of an odd and an even polynomial (the
// minimax approximation Eigen and TensorFlow use), on x clamped to [-4, 4].
static inline float erf_approx(float x) {
    x = clamp(x, -4.0f, 4.0f);
    float x2 = x * x;
    float p = -2.72614225801306e-10f;
    p = p * x2 + 2.77068142495902e-08f;
    p = p * x2 - 2.10102402082508e-06f;
    p = p * x2 - 5.69250639462346e-05f;
    p = p * x2 - 7.34990630326855e-04f;
    p = p * x2 - 2.95459980854025e-03f;
    p = p * x2 - 1.60960333262415e-02f;
    p *= x;
    float q = -1.45660718464996e-05f;
    q = q * x2 - 2.13374055278905e-04f;
    q = q * x2 - 1.68282697438203e-03f;
    q = q * x2 - 7.37332916720468e-03f;
    q = q * x2 - 1.42647390514189e-02f;
    return p / q;
}

// gelu applies the exact (erf) GELU in place, as ops.GELU. Grid: (n).
kernel void gelu(device float *x [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
    float v = x[i];
    x[i] = 0.5f * v * (1 + erf_approx(v * 0.70710678118654752f));
}

// relu replaces negative values with zero, in place. Grid: (n).
kernel void relu(device float *x [[buffer(0)]],
                 uint i [[thread_position_in_grid]]) {
    x[i] = max(x[i], 0.0f);
}

// argmax writes the index of the largest of x[0..n) to out[0], the first one
// on ties, as ops.Argmax. One threadgroup: each thread scans a stride, then
// pairs combine, the larger value winning and the smaller index on a tie.
kernel void argmax(device const float *x [[buffer(0)]],
                   device int *out [[buffer(1)]],
                   constant uint &n [[buffer(2)]],
                   uint t [[thread_position_in_threadgroup]],
                   uint nt [[threads_per_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float values[32];
    threadgroup int indices[32];
    float best = -INFINITY;
    int at = INT_MAX;
    if (t < n) {
        best = x[t];
        at = t;
    }
    for (uint j = t + nt; j < n; j += nt) {
        if (x[j] > best) {
            best = x[j];
            at = j;
        }
    }
    for (int round = 0; round < 2; round++) {
        for (ushort d = 16; d > 0; d /= 2) {
            float v = simd_shuffle_down(best, d);
            int i = simd_shuffle_down(at, d);
            if (lane + d < 32 && (v > best || (v == best && i < at))) {
                best = v;
                at = i;
            }
        }
        if (round == 1) {
            break;
        }
        if (lane == 0) {
            values[sg] = best;
            indices[sg] = at;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        best = lane < nsg ? values[lane] : -INFINITY;
        at = lane < nsg ? indices[lane] : INT_MAX;
    }
    if (t == 0) {
        out[0] = at;
    }
}

// scale sets dst = s·src over n values. Grid: (n).
kernel void scale(device const float *src [[buffer(0)]],
                  device float *dst [[buffer(1)]],
                  constant float &s [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    dst[i] = src[i] * s;
}
