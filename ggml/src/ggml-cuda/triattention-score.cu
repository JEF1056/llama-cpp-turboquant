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
        fprintf(stderr, "tria_cuda_score_layer: kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int tria_cuda_global_download(float * dst_host, int n) {
    TRIA_CUDA_OK(cudaDeviceSynchronize());
    TRIA_CUDA_OK(cudaMemcpy(dst_host, g_ctx.d_global, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

void tria_cuda_cleanup(void) {
    cudaFree(g_ctx.d_head_stats); cudaFree(g_ctx.d_omega);
    cudaFree(g_ctx.d_kr); cudaFree(g_ctx.d_ki); cudaFree(g_ctx.d_key_pos);
    cudaFree(g_ctx.d_raw); cudaFree(g_ctx.d_lscores); cudaFree(g_ctx.d_global);
    g_ctx = tria_cuda_ctx();
}

} // extern "C"
