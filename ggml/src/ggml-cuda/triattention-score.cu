/*
 * triattention-score.cu — GPU trig scoring for TriAttention KV eviction.
 *
 * Hybrid path: the CPU still dequantizes turbo K, applies the inverse WHT and
 * extracts the post-RoPE complex pairs (k_real/k_imag).  Those are uploaded
 * here and this file runs the expensive part on the GPU:
 *   - the 17-offset cos/sin "trig" scoring (score_keys_single_head)
 *   - the two-level GQA aggregation (per-query-head z-normalize + max, then
 *     per-kv-head z-normalize + layer weighting)
 *   - the commutative max-merge into the persistent global score buffer.
 *
 * It is a faithful port of the CPU math in src/triattention.c and the kvi loop
 * in src/triattention-runtime.c.  Because both paths consume identical
 * k_real/k_imag, GPU vs CPU scores can be compared directly (TRIA_GPU_VERIFY).
 *
 * Entry points are extern "C" with default visibility so the `llama` library
 * (triattention-runtime.c) can call them across the ggml-cuda .so boundary.
 */

#include "common.cuh"
#include "turbo-quant.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TRIA_N_OFFSETS 17
#define TRIA_GPU_MAX_FC 64   /* GPU path only handles fc <= this; CPU fallback otherwise */

// ----------------------------------------------------------------------------
// Persistent device state (allocated lazily, reused across passes).
// ----------------------------------------------------------------------------
struct tria_cuda_ctx {
    // per-head stats, uploaded once: layout [ (li*nh + h) * 4 * fc ]
    //   block order per head: q_mean_real[fc], q_mean_imag[fc], q_abs_mean[fc], qma[fc]
    float * d_head_stats = nullptr;
    float * d_omega      = nullptr;   // [fc]
    int     stats_nl     = 0;
    int     stats_nh     = 0;
    int     stats_fc     = 0;

    // per-pass buffers
    float * d_kr      = nullptr;      // [nkv * n_new * fc]
    float * d_ki      = nullptr;      // [nkv * n_new * fc]
    int   * d_key_pos = nullptr;      // [n_new]
    float * d_raw     = nullptr;      // [nkv * gqa * n_new]
    float * d_lscores = nullptr;      // [nkv * n_new]
    float * d_global  = nullptr;      // [n_old]
    size_t  cap_kr    = 0;            // capacity (elements) of d_kr
    size_t  cap_ki    = 0;            // capacity (elements) of d_ki (separate: shared cap left d_ki under-sized on growth)
    size_t  cap_raw   = 0;
    size_t  cap_lsc   = 0;
    size_t  cap_keys  = 0;
    size_t  cap_glob  = 0;

    int     n_new   = 0;
    int     n_old   = 0;
    int     nkv     = 0;
    int     gqa     = 0;
    int     fc      = 0;
    int     cur_pos = 0;

    // physical KV compaction (gather) buffers
    int   * d_keep      = nullptr;   // [num_kept] kept physical row indices
    size_t  cap_keep    = 0;
    void  * d_scratch   = nullptr;   // [num_kept * max_row_bytes] gather destination
    size_t  cap_scratch = 0;         // capacity in bytes
    int     num_kept    = 0;
};

static tria_cuda_ctx g_ctx;

#define TRIA_CUDA_OK(call) do {                                            \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
        fprintf(stderr, "tria_cuda: %s failed: %s\n", #call,              \
                cudaGetErrorString(err__));                                \
        return -1;                                                         \
    }                                                                      \
} while (0)

static int ensure_capacity(float ** ptr, size_t * cap, size_t need) {
    if (*cap >= need && *ptr) return 0;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    TRIA_CUDA_OK(cudaMalloc((void **)ptr, need * sizeof(float)));
    *cap = need;
    return 0;
}

static int ensure_capacity_i(int ** ptr, size_t * cap, size_t need) {
    if (*cap >= need && *ptr) return 0;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    TRIA_CUDA_OK(cudaMalloc((void **)ptr, need * sizeof(int)));
    *cap = need;
    return 0;
}

static int ensure_capacity_bytes(void ** ptr, size_t * cap, size_t need_bytes) {
    if (*cap >= need_bytes && *ptr) return 0;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    TRIA_CUDA_OK(cudaMalloc(ptr, need_bytes));
    *cap = need_bytes;
    return 0;
}

// ----------------------------------------------------------------------------
// atomic max on float (CAS loop). Handles the max-merge faithfully.
// ----------------------------------------------------------------------------
static __device__ __forceinline__ float atomicMaxFloat(float * addr, float val) {
    int * iaddr = (int *)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (old != assumed);
    return __int_as_float(old);
}

// ----------------------------------------------------------------------------
// Kernel 1: raw per (head=kvi*gqa+g, token) score = score_keys_single_head.
// One thread per (head, token).  kr/ki indexed [kvi][token][f] with kvi stride.
// ----------------------------------------------------------------------------
__global__ void tria_k_raw_scores(
        const float * __restrict__ d_kr,     // [nkv * n_new * fc]
        const float * __restrict__ d_ki,
        const int   * __restrict__ d_key_pos,// [n_new]
        const float * __restrict__ d_stats,  // head stats base for this layer
        const float * __restrict__ d_omega,  // [fc]
        float       * __restrict__ d_raw,    // [nkv * gqa * n_new]
        int n_new, int nkv, int gqa, int fc,
        int cur_pos, float max_beta) {

    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int head  = blockIdx.y;            // 0 .. nkv*gqa-1
    if (token >= n_new) return;

    const int kvi = head / gqa;
    // per-head stats: block of 4*fc floats
    const float * qmr  = d_stats + (size_t)head * 4 * fc + 0 * fc;
    const float * qmi  = d_stats + (size_t)head * 4 * fc + 1 * fc;
    const float * qabs = d_stats + (size_t)head * 4 * fc + 2 * fc;
    const float * qma  = d_stats + (size_t)head * 4 * fc + 3 * fc;

    const float * kr = d_kr + ((size_t)kvi * n_new + token) * fc;
    const float * ki = d_ki + ((size_t)kvi * n_new + token) * fc;

    float rel_r[TRIA_GPU_MAX_FC];
    float rel_i[TRIA_GPU_MAX_FC];
    float extra = 0.0f;

    for (int f = 0; f < fc; f++) {
        const float krf = kr[f];
        const float kif = ki[f];
        rel_r[f] = qmr[f] * krf + qmi[f] * kif;
        rel_i[f] = qmi[f] * krf - qmr[f] * kif;
        const float residual = qabs[f] - qma[f];
        if (residual > 0.0f) {
            const float ka = sqrtf(krf * krf + kif * kif);
            extra += residual * ka;
        }
    }

    const float base_delta = (float)(cur_pos - d_key_pos[token]);

    float trig_sum = 0.0f;
    float trig_max = -1e30f;
    int   offset   = 1;
    for (int o = 0; o < TRIA_N_OFFSETS; o++) {
        const float delta = base_delta + (float)offset;
        float trig = 0.0f;
        for (int f = 0; f < fc; f++) {
            float s_, c_;
            sincosf(delta * d_omega[f], &s_, &c_);
            trig += rel_r[f] * c_ - rel_i[f] * s_;
        }
        trig_sum += trig;
        if (trig > trig_max) trig_max = trig;
        offset <<= 1;
    }

    const float trig_mean = trig_sum / (float)TRIA_N_OFFSETS;
    const float out = trig_mean + max_beta * (trig_max - trig_mean) + extra;
    d_raw[(size_t)head * n_new + token] = out;
}

// ----------------------------------------------------------------------------
// Kernel 2: per (kvi, g) z-normalize raw over tokens, then atomic-max into
// d_lscores[kvi][token].  One block per (kvi, g).  d_lscores pre-set to -inf.
// ----------------------------------------------------------------------------
__global__ void tria_k_level1(
        const float * __restrict__ d_raw,    // [nkv*gqa*n_new]
        float       * __restrict__ d_lscores,// [nkv*n_new]
        int n_new, int gqa) {

    const int head = blockIdx.x;            // 0 .. nkv*gqa-1
    const int kvi  = head / gqa;
    const float * raw = d_raw + (size_t)head * n_new;
    float * lsc = d_lscores + (size_t)kvi * n_new;

    __shared__ float s_red[256];
    __shared__ float s_mean;
    __shared__ float s_std;

    // pass 1: mean
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < n_new; t += blockDim.x) local_sum += raw[t];
    s_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_mean = s_red[0] / (float)n_new;
    __syncthreads();
    const float mean = s_mean;

    // pass 2: variance from deviations (matches CPU: var = sum((x-mean)^2)/n)
    float local_var = 0.0f;
    for (int t = threadIdx.x; t < n_new; t += blockDim.x) {
        float d = raw[t] - mean;
        local_var += d * d;
    }
    s_red[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float std = sqrtf(s_red[0] / (float)n_new);
        if (std < 1e-6f) std = 1e-6f;
        s_std = std;
    }
    __syncthreads();
    const float std = s_std;

    // pass 3: z-normalize + max-merge across g into lscores
    for (int t = threadIdx.x; t < n_new; t += blockDim.x) {
        float z = (raw[t] - mean) / std;
        atomicMaxFloat(&lsc[t], z);
    }
}

// ----------------------------------------------------------------------------
// Kernel 3: per kvi z-normalize lscores, apply layer weight, atomic-max into
// global scores at [score_start..].  One block per kvi.
// ----------------------------------------------------------------------------
__global__ void tria_k_level2(
        const float * __restrict__ d_lscores,// [nkv*n_new]
        float       * __restrict__ d_global, // [n_old]
        int n_new, int score_start, float layer_weight) {

    const int kvi = blockIdx.x;
    const float * lsc = d_lscores + (size_t)kvi * n_new;

    __shared__ float s_red[256];
    __shared__ float s_mean;
    __shared__ float s_std;

    // pass 1: mean
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < n_new; t += blockDim.x) local_sum += lsc[t];
    s_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_mean = s_red[0] / (float)n_new;
    __syncthreads();
    const float mean = s_mean;

    // pass 2: variance; CPU level-2 std = sqrt(var/n + 1e-8), no floor
    float local_var = 0.0f;
    for (int t = threadIdx.x; t < n_new; t += blockDim.x) {
        float d = lsc[t] - mean;
        local_var += d * d;
    }
    s_red[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_std = sqrtf(s_red[0] / (float)n_new + 1e-8f);
    __syncthreads();
    const float std = s_std;

    for (int t = threadIdx.x; t < n_new; t += blockDim.x) {
        float z  = (lsc[t] - mean) / std;
        float wz = z > 0.0f ? z * layer_weight : z;
        atomicMaxFloat(&d_global[score_start + t], wz);
    }
}

__global__ void tria_k_fill(float * p, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = val;
}

// ----------------------------------------------------------------------------
// Physical KV compaction gather kernel.
// Each output row `r` in [0, num_kept) copies `row_size` bytes from source row
// keep[r] of the live tensor into the contiguous scratch buffer. Gathering into
// a separate scratch (rather than the live tensor) avoids the read-after-write
// race of an in-place parallel gather (thread for row r' < r may read source
// row r while the thread for row r overwrites it). The 16-byte fast path is used
// when both the row stride and the data pointers are 16-byte aligned (turbo K/V
// rows are padded to a 128-byte multiple, so this holds in practice); a byte
// tail handles any remainder. blockIdx.y strides over rows to stay within the
// 65535 gridDim.y limit at large num_kept.
// ----------------------------------------------------------------------------
__global__ void tria_gather_rows(
        char       * __restrict__ dst,   // scratch [num_kept * row_size]
        const char * __restrict__ src,   // live tensor data
        const int  * __restrict__ keep,  // [num_kept] source row indices
        int num_kept, size_t row_size) {
    // When row_size is a multiple of 16 every per-row base (row*row_size) is also
    // 16-byte aligned (the CUDA allocation base is 256-aligned), so the uint4 fast
    // path is safe. Otherwise fall back to a plain byte copy to avoid a misaligned
    // vector access.
    const bool vec = (row_size & (size_t)0xF) == 0;
    for (int row = blockIdx.y; row < num_kept; row += gridDim.y) {
        const size_t base_dst = (size_t) row * row_size;
        const size_t base_src = (size_t) keep[row] * row_size;
        const size_t stride   = (size_t) gridDim.x * blockDim.x;
        const size_t t0       = blockIdx.x * blockDim.x + threadIdx.x;

        if (vec) {
            const size_t n16 = row_size >> 4;
            const uint4 * s16 = reinterpret_cast<const uint4 *>(src + base_src);
            uint4       * d16 = reinterpret_cast<uint4 *>(dst + base_dst);
            for (size_t w = t0; w < n16; w += stride) {
                d16[w] = s16[w];
            }
        } else {
            for (size_t b = t0; b < row_size; b += stride) {
                dst[base_dst + b] = src[base_src + b];
            }
        }
    }
}

// ============================================================================
// Turbo K dequant + inverse WHT + complex-pair extraction (GPU offload).
//
// Replaces the CPU stage in triattention-runtime.c (ggml_backend_tensor_get +
// traits->to_float + tria_inverse_wht_row + the per-kvi complex-pair extract)
// for turbo K caches. One block per (kv head, scored token): 128 threads
// dequantize the head's first 128-element WHT group from the quantized K tensor
// in device memory, thread 0 runs the inverse WHT (matches the CPU
// tria_inverse_wht_group exactly), then threads 0..fc-1 write the post-RoPE
// complex pairs straight into the scorer's d_kr/d_ki buffers.
//
// Only the first 128-group of each head is needed: the scorer reads logical
// indices 0..2*fc-1 (fc <= 64), all within that group, so padded heads spanning
// multiple 128-groups still only require one inverse WHT here.
// ============================================================================
template <typename block_t, float (*deq_elem)(const block_t *, int, float)>
__global__ void tria_k_dequant(
        const char * __restrict__ d_base, int phys_base, size_t row_size,
        int blocks_per_head, int n_new, int fc, int rope_neox,
        float * __restrict__ d_kr, float * __restrict__ d_ki) {
    const int kvi = blockIdx.x;
    const int s   = blockIdx.y;
    const int t   = threadIdx.x;            // 0..127

    __shared__ float g[128];

    const char *    row = d_base + (size_t)(phys_base + s) * row_size;
    const block_t * blk = reinterpret_cast<const block_t *>(row) + (size_t)kvi * blocks_per_head;

    const float norm = __half2float(blk->norm);
    g[t] = deq_elem(blk, t, norm);
    __syncthreads();

    // Inverse WHT on the 128-element group (serial; matches CPU bit-for-bit).
    if (t == 0) {
        turbo_inverse_wht_128(g);
    }
    __syncthreads();

    if (t < fc) {
        const size_t out = ((size_t)kvi * n_new + s) * fc + t;
        if (rope_neox) {
            d_kr[out] = g[t];
            d_ki[out] = g[fc + t];
        } else {
            d_kr[out] = g[2 * t + 0];
            d_ki[out] = g[2 * t + 1];
        }
    }
}

// ============================================================================
// Host entry points (extern "C")
// ============================================================================
extern "C" {

// Upload per-head stats once. head_stats layout: [ (li*nh + h) * 4 * fc ]
// with per-head blocks q_mean_real[fc], q_mean_imag[fc], q_abs_mean[fc], qma[fc].
int tria_cuda_upload_stats(const float * omega, int fc,
                           const float * head_stats, int nl, int nh) {
    if (fc <= 0 || fc > TRIA_GPU_MAX_FC || nl <= 0 || nh <= 0) return -1;
    if (g_ctx.d_head_stats && g_ctx.stats_nl == nl &&
        g_ctx.stats_nh == nh && g_ctx.stats_fc == fc) {
        return 0; // already uploaded
    }
    if (g_ctx.d_head_stats) { cudaFree(g_ctx.d_head_stats); g_ctx.d_head_stats = nullptr; }
    if (g_ctx.d_omega)      { cudaFree(g_ctx.d_omega);      g_ctx.d_omega = nullptr; }

    const size_t n_stats = (size_t)nl * nh * 4 * fc;
    TRIA_CUDA_OK(cudaMalloc((void **)&g_ctx.d_head_stats, n_stats * sizeof(float)));
    TRIA_CUDA_OK(cudaMalloc((void **)&g_ctx.d_omega, (size_t)fc * sizeof(float)));
    TRIA_CUDA_OK(cudaMemcpy(g_ctx.d_head_stats, head_stats, n_stats * sizeof(float), cudaMemcpyHostToDevice));
    TRIA_CUDA_OK(cudaMemcpy(g_ctx.d_omega, omega, (size_t)fc * sizeof(float), cudaMemcpyHostToDevice));
    g_ctx.stats_nl = nl;
    g_ctx.stats_nh = nh;
    g_ctx.stats_fc = fc;
    return 0;
}

// Begin a scoring pass: size buffers, upload key positions and the initial
// global-score state (already initialized by the caller to -1e30 / prior).
int tria_cuda_global_begin(int n_old, int n_new, const int * key_pos_scored,
                           int cur_pos, int nkv, int gqa, int fc,
                           const float * global_init) {
    if (fc <= 0 || fc > TRIA_GPU_MAX_FC) return -1;

    // Entry fence: this scorer issues all its work on the legacy default stream
    // (stream 0), but ggml-cuda creates its compute/copy streams with
    // cudaStreamNonBlocking (common.cuh), which explicitly opt OUT of the
    // implicit synchronization that stream 0 normally has with other streams.
    // So in-flight ggml work (the just-completed decode forward pass) can still
    // be running when we start allocating/copying/launching here, and the CUDA
    // allocator may recycle device memory across the two unsynchronized streams
    // -> use-after-free -> a deferred "illegal memory access". tria_cuda_global_download
    // already fences at the END; mirror it at the START so the whole scoring pass
    // is bracketed. (Previously the only entry sync was the incidental device sync
    // inside cudaMalloc/cudaFree during ensure_capacity GROWTH passes, so the first
    // pass at a new context length was safe but every subsequent steady-state pass
    // — which skips the malloc — raced. Matches CUDA_LAUNCH_BLOCKING=1 being immune.)
    TRIA_CUDA_OK(cudaDeviceSynchronize());

    g_ctx.n_new = n_new; g_ctx.n_old = n_old; g_ctx.nkv = nkv;
    g_ctx.gqa = gqa; g_ctx.fc = fc; g_ctx.cur_pos = cur_pos;

    if (ensure_capacity(&g_ctx.d_kr, &g_ctx.cap_kr, (size_t)nkv * n_new * fc)) return -1;
    if (ensure_capacity(&g_ctx.d_ki, &g_ctx.cap_ki, (size_t)nkv * n_new * fc)) return -1;
    if (ensure_capacity_i(&g_ctx.d_key_pos, &g_ctx.cap_keys, (size_t)n_new)) return -1;
    if (ensure_capacity(&g_ctx.d_raw, &g_ctx.cap_raw, (size_t)nkv * gqa * n_new)) return -1;
    if (ensure_capacity(&g_ctx.d_lscores, &g_ctx.cap_lsc, (size_t)nkv * n_new)) return -1;
    if (ensure_capacity(&g_ctx.d_global, &g_ctx.cap_glob, (size_t)n_old)) return -1;

    TRIA_CUDA_OK(cudaMemcpy(g_ctx.d_key_pos, key_pos_scored, (size_t)n_new * sizeof(int), cudaMemcpyHostToDevice));
    TRIA_CUDA_OK(cudaMemcpy(g_ctx.d_global, global_init, (size_t)n_old * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

// Launch the 3 scoring kernels over d_kr/d_ki (already populated, either via the
// host-copy path in tria_cuda_score_layer or the device dequant in
// tria_cuda_dequant_layer). Returns 0 on success.
static int tria_score_kernels(int n_new, int nkv, int fc, int gqa,
                              int layer_idx, int num_heads,
                              float max_beta, float layer_weight, int score_start) {
    const float * d_stats_layer = g_ctx.d_head_stats + (size_t)layer_idx * num_heads * 4 * fc;

    // Kernel 1: raw scores
    {
        const int threads = 128;
        dim3 grid((n_new + threads - 1) / threads, nkv * gqa);
        tria_k_raw_scores<<<grid, threads>>>(
            g_ctx.d_kr, g_ctx.d_ki, g_ctx.d_key_pos,
            d_stats_layer, g_ctx.d_omega, g_ctx.d_raw,
            n_new, nkv, gqa, fc, g_ctx.cur_pos, max_beta);
    }

    // reset lscores to -inf
    {
        const int threads = 256;
        const int n = nkv * n_new;
        tria_k_fill<<<(n + threads - 1) / threads, threads>>>(g_ctx.d_lscores, n, -1e30f);
    }

    // Kernel 2: level-1 z-norm + max across gqa
    tria_k_level1<<<nkv * gqa, 256>>>(g_ctx.d_raw, g_ctx.d_lscores, n_new, gqa);

    // Kernel 3: level-2 z-norm + weight + global merge
    tria_k_level2<<<nkv, 256>>>(g_ctx.d_lscores, g_ctx.d_global, n_new, score_start, layer_weight);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tria_score_kernels: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Score one attention layer. kr/ki are host [nkv][kvi_stride][fc] (kvi_stride in
// elements-of-fc-rows, i.e. row index stride per KV head).
int tria_cuda_score_layer(const float * kr_host, const float * ki_host,
                          int kvi_stride_rows,
                          int n_new, int nkv, int fc, int gqa,
                          int layer_idx, int num_heads,
                          float max_beta, float layer_weight, int score_start) {
    if (fc <= 0 || fc > TRIA_GPU_MAX_FC) return -1;

    // Upload kr/ki compacted to [nkv * n_new * fc]: copy per-kv slice (host has
    // a larger n_old row stride, we only need the first n_new rows).
    for (int kvi = 0; kvi < nkv; kvi++) {
        const float * src_r = kr_host + (size_t)kvi * kvi_stride_rows * fc;
        const float * src_i = ki_host + (size_t)kvi * kvi_stride_rows * fc;
        float * dst_r = g_ctx.d_kr + (size_t)kvi * n_new * fc;
        float * dst_i = g_ctx.d_ki + (size_t)kvi * n_new * fc;
        TRIA_CUDA_OK(cudaMemcpy(dst_r, src_r, (size_t)n_new * fc * sizeof(float), cudaMemcpyHostToDevice));
        TRIA_CUDA_OK(cudaMemcpy(dst_i, src_i, (size_t)n_new * fc * sizeof(float), cudaMemcpyHostToDevice));
    }

    return tria_score_kernels(n_new, nkv, fc, gqa, layer_idx, num_heads,
                              max_beta, layer_weight, score_start);
}

// Dequantize one turbo K layer directly from device memory into d_kr/d_ki,
// applying the inverse WHT and extracting post-RoPE complex pairs on the GPU.
// d_k is the K tensor's device data pointer; ktype is the ggml_type. padded_hd
// is the physical per-head width (multiple of 128); fc/rope_neox match the
// scorer. Replaces the CPU dequant+WHT+extract stage. Returns 0 on success,
// -1 on unsupported type (caller should fall back to the CPU path).
int tria_cuda_dequant_layer(const void * d_k, int ktype,
                            int phys_base, int n_new, int nkv,
                            int padded_hd, int fc, int rope_neox) {
    if (fc <= 0 || fc > TRIA_GPU_MAX_FC) return -1;
    if (padded_hd <= 0 || (padded_hd % 128) != 0) return -1;

    const int blocks_per_head = padded_hd / 128;
    const int blocks_per_row  = nkv * blocks_per_head;

    size_t row_size = 0;
    const dim3 grid(nkv, n_new);
    const int  threads = 128;
    const char * d_base = reinterpret_cast<const char *>(d_k);

    switch (ktype) {
        case GGML_TYPE_TURBO4_0:
            row_size = (size_t)blocks_per_row * sizeof(block_turbo4_0);
            tria_k_dequant<block_turbo4_0, turbo4_dequant_element><<<grid, threads>>>(
                d_base, phys_base, row_size, blocks_per_head, n_new, fc, rope_neox,
                g_ctx.d_kr, g_ctx.d_ki);
            break;
        case GGML_TYPE_TURBO3_0:
            row_size = (size_t)blocks_per_row * sizeof(block_turbo3_0);
            tria_k_dequant<block_turbo3_0, turbo3_dequant_element><<<grid, threads>>>(
                d_base, phys_base, row_size, blocks_per_head, n_new, fc, rope_neox,
                g_ctx.d_kr, g_ctx.d_ki);
            break;
        case GGML_TYPE_TURBO2_0:
            row_size = (size_t)blocks_per_row * sizeof(block_turbo2_0);
            tria_k_dequant<block_turbo2_0, turbo2_dequant_element><<<grid, threads>>>(
                d_base, phys_base, row_size, blocks_per_head, n_new, fc, rope_neox,
                g_ctx.d_kr, g_ctx.d_ki);
            break;
        default:
            return -1;  // unsupported K type — caller falls back to CPU
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tria_cuda_dequant_layer: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Score one attention layer whose d_kr/d_ki were already filled on-device by
// tria_cuda_dequant_layer (no host->device copy). Returns 0 on success.
int tria_cuda_score_layer_device(int n_new, int nkv, int fc, int gqa,
                                 int layer_idx, int num_heads,
                                 float max_beta, float layer_weight, int score_start) {
    if (fc <= 0 || fc > TRIA_GPU_MAX_FC) return -1;
    return tria_score_kernels(n_new, nkv, fc, gqa, layer_idx, num_heads,
                              max_beta, layer_weight, score_start);
}

int tria_cuda_global_download(float * dst_host, int n) {
    TRIA_CUDA_OK(cudaDeviceSynchronize());
    TRIA_CUDA_OK(cudaMemcpy(dst_host, g_ctx.d_global, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Begin a physical KV compaction pass: upload the kept physical row indices once
// (reused across all layers). num_kept rows will be gathered to [0, num_kept).
int tria_cuda_compact_begin(const int * keep_positions, int num_kept) {
    if (num_kept <= 0) return -1;

    // Entry fence: the caller has already issued llama_synchronize, but mirror the
    // scorer's bracketing so the gather is isolated from any ggml work still queued
    // on the non-blocking compute/copy streams (see tria_cuda_global_begin).
    TRIA_CUDA_OK(cudaDeviceSynchronize());

    if (ensure_capacity_i(&g_ctx.d_keep, &g_ctx.cap_keep, (size_t)num_kept)) return -1;
    TRIA_CUDA_OK(cudaMemcpy(g_ctx.d_keep, keep_positions, (size_t)num_kept * sizeof(int), cudaMemcpyHostToDevice));
    g_ctx.num_kept = num_kept;
    return 0;
}

// Compact one layer's K and V tensors in place: gather the kept rows into scratch
// then copy them back to the contiguous [0, num_kept) range. k_data/v_data are the
// CUDA device pointers (tensor->data); *_row_size is the per-row byte stride.
// All work runs on cudaStreamPerThread (the same stream ggml uses for D2D copies),
// so gather -> copy-back -> next gather are ordered without explicit per-op syncs;
// a single stream sync at the end guards the host return.
int tria_cuda_compact_layer(void * k_data, size_t k_row_size,
                            void * v_data, size_t v_row_size) {
    if (g_ctx.num_kept <= 0) return -1;
    if (!k_data || k_row_size == 0) return -1;

    const int num_kept = g_ctx.num_kept;
    const size_t max_row = k_row_size > v_row_size ? k_row_size : v_row_size;
    if (ensure_capacity_bytes(&g_ctx.d_scratch, &g_ctx.cap_scratch, (size_t)num_kept * max_row)) return -1;

    cudaStream_t stream = cudaStreamPerThread;
    const int    threads = 256;
    const int    grid_y  = num_kept < 65535 ? num_kept : 65535;

    // ---- K ----
    {
        const int grid_x = (int)((k_row_size + threads - 1) / threads);
        dim3 grid(grid_x > 0 ? grid_x : 1, grid_y);
        tria_gather_rows<<<grid, threads, 0, stream>>>(
            (char *)g_ctx.d_scratch, (const char *)k_data, g_ctx.d_keep, num_kept, k_row_size);
        TRIA_CUDA_OK(cudaMemcpyAsync(k_data, g_ctx.d_scratch,
            (size_t)num_kept * k_row_size, cudaMemcpyDeviceToDevice, stream));
    }

    // ---- V ----
    if (v_data && v_row_size > 0) {
        const int grid_x = (int)((v_row_size + threads - 1) / threads);
        dim3 grid(grid_x > 0 ? grid_x : 1, grid_y);
        tria_gather_rows<<<grid, threads, 0, stream>>>(
            (char *)g_ctx.d_scratch, (const char *)v_data, g_ctx.d_keep, num_kept, v_row_size);
        TRIA_CUDA_OK(cudaMemcpyAsync(v_data, g_ctx.d_scratch,
            (size_t)num_kept * v_row_size, cudaMemcpyDeviceToDevice, stream));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tria_cuda_compact_layer: launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    TRIA_CUDA_OK(cudaStreamSynchronize(stream));
    return 0;
}

void tria_cuda_cleanup(void) {
    cudaFree(g_ctx.d_head_stats); cudaFree(g_ctx.d_omega);
    cudaFree(g_ctx.d_kr); cudaFree(g_ctx.d_ki); cudaFree(g_ctx.d_key_pos);
    cudaFree(g_ctx.d_raw); cudaFree(g_ctx.d_lscores); cudaFree(g_ctx.d_global);
    cudaFree(g_ctx.d_keep); cudaFree(g_ctx.d_scratch);
    g_ctx = tria_cuda_ctx();
}

} // extern "C"
