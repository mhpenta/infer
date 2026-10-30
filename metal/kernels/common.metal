#include <metal_stdlib>
using namespace metal;

// tg_sum returns the sum of v over the threadgroup, on every thread. scratch
// holds one float per simdgroup; the call must be reached by every thread.
static inline float tg_sum(float v, threadgroup float *scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_sum(lane < nsg ? scratch[lane] : 0.0f);
}

// bf16 widens a bfloat16 bit pattern to float32 exactly.
static inline float bf16(ushort h) { return as_type<float>(uint(h) << 16); }

static inline float4 bf16x4(ushort4 h) { return as_type<float4>(uint4(h) << 16); }

// GemmArgs are the arguments of the products, gemv and gemm_mma:
// y = x·Wᵀ (+ bias).
struct GemmArgs {
    uint n;    // rows of W
    uint k;    // a multiple of 4
    uint m;    // rows of x and y
    uint ldx;  // floats between rows of x (a multiple of 4)
    uint ldy;  // floats between rows of y
    uint bias; // 1: add bias[j] to column j of y
};

// epilogue finishes output j of a product from its sum: int8 sums are
// multiplied by the row's scale, then the bias is added, each rounded.
template <typename T>
static inline float epilogue(float s, uint j, device const float *scale, device const float *bias, constant GemmArgs &a) {
    if (is_same<T, char>::value) {
        s *= scale[j];
    }
    return a.bias ? s + bias[j] : s;
}

// widen4 loads 4 consecutive weights and widens them to float32: bf16 and
// fp16 exactly, int8 to its integer value (a product's scale comes after
// the sum). Loads need alignment to 4 values: 16 bytes for float32, 8 for
// bf16 and fp16, 4 for int8.
static inline float4 widen4(device const ushort *p) { return bf16x4(*(device const ushort4 *)p); }
static inline float4 widen4(device const half *p) { return float4(*(device const half4 *)p); }
static inline float4 widen4(device const float *p) { return *(device const float4 *)p; }
static inline float4 widen4(device const char *p) { return float4(*(device const char4 *)p); }

// widen widens one value of an embedding table.
static inline float widen(ushort h) { return bf16(h); }
static inline float widen(half h) { return float(h); }
static inline float widen(float f) { return f; }
