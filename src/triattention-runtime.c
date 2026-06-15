/*
 * triattention-runtime.c — TriAttention runtime scoring
 */

#define _GNU_SOURCE
#include "triattention-runtime.h"
#include "triattention.h"
#include "triattention-backend.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <math.h>
#include <stdio.h>
#include "llama-log-c.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GGML_TRIA_CUDA
#include "triattention-cuda.h"
/* GPU turbo scoring config, read once from env. -1 = not yet read. */
static int   tria_gpu_turbo_mode     = -1;   /* TRIA_GPU_SCORING */
static int   tria_gpu_turbo_verify   = 0;    /* TRIA_GPU_VERIFY  */
static int   tria_gpu_stats_uploaded = 0;
static float tria_gpu_max_beta       = 0.0f; /* TRIA_MAX_BETA, default 0 */
static void tria_gpu_turbo_init_cfg(void) {
    if (tria_gpu_turbo_mode >= 0) return;
    const char * e = getenv("TRIA_GPU_SCORING");
    tria_gpu_turbo_mode   = (e && atoi(e) != 0) ? 1 : 0;
    const char * v = getenv("TRIA_GPU_VERIFY");
    tria_gpu_turbo_verify = (v && atoi(v) != 0) ? 1 : 0;
    const char * b = getenv("TRIA_MAX_BETA");
    if (b) { float f = strtof(b, NULL); if (isfinite(f) && f >= 0.0f && f <= 1.0f) tria_gpu_max_beta = f; }
    LLAMA_LOG_INFO("tria: GPU turbo scoring=%d verify=%d max_beta=%.2f\n",
                   tria_gpu_turbo_mode, tria_gpu_turbo_verify, tria_gpu_max_beta);
}
#endif

struct tria_runtime * g_tria_rt = NULL;

/* ---- Inverse Turbo WHT for scoring ---- */
/* Must match sign arrays in ggml-cpu/ops.cpp and ggml-turbo-quant.c */
static const float tria_wht_s1[128] = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static const float tria_wht_s2[128] = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

/* Apply inverse WHT to one group of 128 floats in-place.
 * Inverse = s2 → WHT butterfly → s1 → normalize (direction=1). */
static void tria_inverse_wht_group(float *x, int gs) {
    const float inv_sqrt = 1.0f / sqrtf((float)gs);
    /* Apply s2 first (inverse direction) */
    for (int i = 0; i < gs; i++) x[i] *= tria_wht_s2[i];
    /* WHT butterfly */
    for (int h = 1; h < gs; h *= 2) {
        for (int i = 0; i < gs; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    /* Normalize + apply s1 */
    for (int i = 0; i < gs; i++) x[i] *= inv_sqrt * tria_wht_s1[i];
}

/* Apply inverse WHT to a full dequantized row (multiple heads, each padded to padded_hd).
 * After inverse WHT, copy only the first hd elements per head into dst.
 * Processes per 128-element group — no full-head buffer needed. */
static void tria_inverse_wht_row(const float *src_phys, float *dst_logical,
                                  int nkv, int padded_hd, int hd) {
    const int gs = 128;
    float buf[128];
    for (int kvi = 0; kvi < nkv; kvi++) {
        const float *head_in = src_phys + kvi * padded_hd;
        float *head_out = dst_logical + kvi * hd;
        int groups = padded_hd / gs;
        for (int g = 0; g < groups; g++) {
            int src_off = g * gs;
            memcpy(buf, head_in + src_off, gs * sizeof(float));
            tria_inverse_wht_group(buf, gs);
            /* Copy only the portion that falls within logical hd */
            int dst_off = g * gs;
            if (dst_off >= hd) break;
            int copy_len = (dst_off + gs <= hd) ? gs : (hd - dst_off);
            memcpy(head_out + dst_off, buf, copy_len * sizeof(float));
        }
    }
}

struct llama_kv_layer {
    struct ggml_tensor * k;
    struct ggml_tensor * v;
};

extern struct ggml_tensor * tria_get_k_tensor(void * ctx, int layer_idx);
extern struct ggml_tensor * tria_get_v_tensor(void * ctx, int layer_idx);
extern int tria_get_n_kv(void * ctx);
extern int tria_get_used_n_kv(void * ctx);
extern int tria_get_n_ctx(void * ctx);
extern int tria_get_kv_positions(void * ctx, int * positions, int * phys_indices, int max_positions);
extern int tria_compact_kv(struct tria_runtime * rt, void * ctx);

struct tria_runtime * tria_runtime_init(
    struct tria_stats * stats,
    int budget_pct,
    int window,
    int interval,
    int sink
) {
    if (!stats || budget_pct <= 0 || budget_pct > 100) return NULL;
    if (stats->num_kv_heads == 0 || stats->num_layers == 0) return NULL;

    /* Overflow check (Codex review) */
    uint32_t n_pairs_u = (uint32_t)stats->num_layers * (uint32_t)stats->num_kv_heads;
    if (n_pairs_u / stats->num_layers != stats->num_kv_heads) return NULL;
    int n_pairs = (int)n_pairs_u;

    struct tria_runtime * rt = calloc(1, sizeof(*rt));
    if (!rt) return NULL;

    /* Initialize GPU backend (once) */
    static int backend_initialized = 0;
    if (!backend_initialized) {
        const int gpu_ok = tria_backend_init();
        (void) gpu_ok; /* logged inside tria_backend_init */
        backend_initialized = 1;
    }

    rt->stats      = stats;
    rt->budget_pct = budget_pct;
    rt->window     = window;
    rt->interval   = interval;
    rt->sink       = sink;
    rt->n_scored   = 0;

    rt->retained       = calloc(n_pairs, sizeof(int *));
    rt->retained_count = calloc(n_pairs, sizeof(int));
    if (!rt->retained || !rt->retained_count) {
        free(rt->retained); free(rt->retained_count); free(rt);
        return NULL;
    }

    /* Task 7: cache env-var config at init time */
    {
        const char *bt = getenv("TRIA_BUDGET_TOKENS");
        rt->cfg_budget_tokens = bt ? atoi(bt) : -1;

        const char *rsp = getenv("TRIA_RAMP_START_PCT");
        rt->cfg_ramp_start_pct = rsp ? atoi(rsp) : -1;

        rt->cfg_no_gpu_score = getenv("TRIA_NO_GPU_SCORE") ? 1 : 0;

        const char *sls = getenv("TRIA_SCORE_LAYER_STRIDE");
        rt->cfg_score_layer_stride = sls ? atoi(sls) : 1;
        if (rt->cfg_score_layer_stride < 1) rt->cfg_score_layer_stride = 1;

        const char *vlam = getenv("TRIA_V_LAMBDA");
        rt->cfg_v_lambda = vlam ? strtof(vlam, NULL) : 0.25f;

        const char *rnd = getenv("TRIA_RANDOM");
        rt->cfg_random = (rnd && rnd[0] == '1') ? 1 : 0;
        if (rt->cfg_random) LLAMA_LOG_INFO("tria: RANDOM EVICTION MODE (scoring disabled)\n");
    }
    /* Scratch buffers start NULL/0 — calloc already zeroes them */

    return rt;
}

void tria_runtime_free(struct tria_runtime * rt) {
    if (!rt) return;

    /* Clear global pointer to prevent UAF (Codex review) */
    if (g_tria_rt == rt) {
        g_tria_rt = NULL;
    }

    if (rt->retained) {
        int n_pairs = rt->stats->num_layers * rt->stats->num_kv_heads;
        for (int i = 0; i < n_pairs; i++) {
            free(rt->retained[i]);
        }
        free(rt->retained);
    }
    free(rt->retained_count);
    free(rt->global_scores);

    /* Free GPU scoring stats */
    if (g_tria_backend.stats_free) {
        g_tria_backend.stats_free(rt->gpu_omega, rt->gpu_q_mean_real, rt->gpu_q_mean_imag);
        if (rt->gpu_global_scores)
            g_tria_backend.stats_free(rt->gpu_global_scores, NULL, NULL);
    }

    /* Task 9+1: free persistent scratch buffers */
    free(rt->scratch_k_f32);
    free(rt->scratch_scores);
    free(rt->scratch_key_pos);
    free(rt->scratch_phys);
    free(rt->scratch_k_real);
    free(rt->scratch_k_imag);

    free(rt);
}

int tria_maybe_score(
    struct tria_runtime * rt,
    void * ctx
) {
    if (!rt || !rt->stats || !ctx) return 0;

    int n_kv = tria_get_n_kv(ctx);
    int n_used = tria_get_used_n_kv(ctx);
    if (n_kv <= 0 || n_used <= 0) return 0;

    /* Warn once if multi-slot detected — scoring is approximate */
    static int multi_slot_warned = 0;
    if (!multi_slot_warned && n_used > n_kv + 128) {
        LLAMA_LOG_WARN("tria: warning: n_used (%d) >> n_kv (%d), multi-slot detected. "
                "Scoring uses max-over-sequences which is approximate.\n", n_used, n_kv);
        multi_slot_warned = 1;
    }

    /* Reset if cache was cleared (perplexity resets between chunks) */
    if (n_kv < rt->n_scored) {
        rt->n_scored = 0;
        rt->compaction_active = 0;
        rt->score_pass = 0;
        rt->global_n = 0;
    }

    /* Check if we should score */
    if (n_kv - rt->n_scored < rt->interval) return 0;
    if (n_used <= rt->window) return 0;

    int nl  = rt->stats->num_layers;
    int nkv = rt->stats->num_kv_heads;
    int fc  = rt->stats->freq_count;
    int hd  = rt->stats->head_dim;

    int n_old = n_used - rt->window;
    if (n_old <= 0) return 0;

    /* Budget = pct% of n_ctx (total context), minus window (always kept).
     * This means budget=50% keeps ~50% of total context, not 50% of old tokens.
     * Previous behavior (pct of n_old) converged to ~250 tokens regardless of ctx. */
    int n_ctx = tria_get_n_ctx(ctx);
    if (n_ctx <= 0) n_ctx = 4096;
    int budget = (n_ctx * rt->budget_pct) / 100 - rt->window;
    if (budget > n_old) budget = n_old;  /* can't keep more than we have */
    int absolute_budget = 0;
    if (rt->cfg_budget_tokens >= 0) {
        budget = rt->cfg_budget_tokens;
        absolute_budget = 1;
    }
    if (budget <= 0) budget = 1;

    /* Exponential ramp eviction (opt-in via TRIA_RAMP_START_PCT).
     * When set: cap doubles each pass (e.g. 10->20->40->80->100%).
     * Prevents abrupt semantic shock after long prefill.
     * Disabled when TRIA_BUDGET_TOKENS is set (absolute budget). */
    if (!absolute_budget) {
        if (rt->cfg_ramp_start_pct >= 0) {
            int ramp_start = rt->cfg_ramp_start_pct;
            if (ramp_start < 1) ramp_start = 1;
            if (ramp_start > 100) ramp_start = 100;
            int cap_pct = ramp_start;
            for (int p = 0; p < rt->score_pass && cap_pct < 100; p++)
                cap_pct *= 2;
            if (cap_pct > 100) cap_pct = 100;
            int max_evict = n_old * cap_pct / 100;
            if (max_evict < 256) max_evict = 256;
            int ramp_budget = n_old - max_evict;
            if (ramp_budget > budget) budget = ramp_budget;
        }
    }

    /* No eviction possible this pass: the retained budget already covers every
     * old token (context still below triattention-budget% of n_ctx, or an
     * absolute budget larger than the cache).  Scoring would compute
     * global_scores that the eviction mask never consults, so skip the whole
     * pass — including the periodic full rescore and its value-energy read of
     * the entire V cache — and re-check after another `interval` tokens.  This
     * eliminates all TriAttention decode-loop stalls until the cache actually
     * exceeds budget.  It changes neither ctx-size nor triattention-budget; it
     * only avoids dead work below the eviction threshold. */
    if (budget >= n_old) {
        static int skip_logged = 0;
        if (!skip_logged) {
            LLAMA_LOG_INFO("tria_score: n_old=%d <= budget=%d — no eviction yet, "
                    "skipping scoring until cache exceeds budget\n", n_old, budget);
            skip_logged = 1;
        }
        rt->n_scored = n_kv;
        return 0;
    }

    /* Full rescore is required:
     *   - after physical compaction (tria_compact_kv reorders cells; cached scores
     *     map to the old ordering and would corrupt incremental scoring)
     *   - periodically (every TRIA_FULL_RESCORE_INTERVAL passes) to prevent score drift
     *   - on first use (no scores allocated yet)
     * All other passes use the incremental path: only new tokens are scored and
     * appended; old scores are preserved.  This avoids re-reading the full KV
     * cache from GPU on every decode step. */
    #define TRIA_FULL_RESCORE_INTERVAL 8
    int full_rescore = rt->compaction_active
                     || (rt->score_pass % TRIA_FULL_RESCORE_INTERVAL == 0)
                     || !rt->global_scores
                     || rt->global_n < 1;

    int n_prev = 0;
    int n_new  = n_old;
    int score_start = 0;

    if (!full_rescore && rt->global_scores && rt->global_n > 0) {
        n_prev = rt->global_n;
        if (n_prev > n_old) n_prev = n_old;
        score_start = n_prev;
        n_new = n_old - n_prev;
        if (n_new <= 0) {
            rt->global_budget = budget;
            rt->n_scored = n_kv;
            rt->score_pass++;
            return 0;
        }
    }

    LLAMA_LOG_INFO("tria_score: n_kv=%d n_old=%d budget=%d%s new=%d mode=%s (pass %d)\n",
            n_kv, n_old, budget,
            absolute_budget ? " (abs)" : " (pct)",
            n_new,
            full_rescore ? "full" : "incremental", rt->score_pass);

    /* TRIA_RANDOM=1: random scores for A/B testing against faithful scoring */
    int tria_random_mode = rt->cfg_random;

    if (!ctx) { rt->n_scored = n_kv; return 0; }

    /* Validate stats dimensions against actual KV tensor.
     * Turbo types pad head_dim to 128 multiples — checked per-layer below. */
    {
        struct ggml_tensor * k0 = tria_get_k_tensor(ctx, 0);
        if (k0) {
            int64_t actual_row = k0->ne[0];
            int expected_row = nkv * hd;
            int expected_padded = nkv * (((hd + 127) / 128) * 128);
            if (actual_row != expected_row && actual_row != expected_padded) {
                LLAMA_LOG_WARN("tria_score: TRIA stats mismatch: expected row %d (or padded %d), got %d — skipping\n",
                        expected_row, expected_padded, (int)actual_row);
                rt->n_scored = n_kv;
                return 0;
            }
        }
    }

    int n_embd_k_gqa = nkv * hd;
    /* Overflow check for element count (Codex review #2) */
    size_t n_elem = (size_t)n_new * (size_t)n_embd_k_gqa;
    if (n_elem / (size_t)n_new != (size_t)n_embd_k_gqa) {
        rt->n_scored = n_kv;
        return 0;
    }

    /* Tasks 9+1: resize persistent scratch buffers only when capacity is exceeded */
    if (n_old > rt->scratch_capacity) {
        free(rt->scratch_k_f32);
        free(rt->scratch_scores);
        free(rt->scratch_key_pos);
        free(rt->scratch_phys);
        rt->scratch_k_f32   = (float *)malloc((size_t)n_old * n_embd_k_gqa * sizeof(float));
        /* nkv slices: one private score buffer per KV head for the parallel loop. */
        rt->scratch_scores  = (float *)malloc((size_t)nkv * n_old * sizeof(float));
        rt->scratch_key_pos = (int   *)malloc((size_t)n_old * sizeof(int));
        rt->scratch_phys    = (int   *)malloc((size_t)n_old * sizeof(int));
        rt->scratch_capacity = n_old;
    }
    if (n_old > rt->scratch_fc_capacity) {
        free(rt->scratch_k_real);
        free(rt->scratch_k_imag);
        /* Sized nkv*n_old*fc so each KV head scores into a private slice in
         * parallel without locking (see the OpenMP kvi loop below). */
        rt->scratch_k_real = (float *)malloc((size_t)nkv * n_old * fc * sizeof(float));
        rt->scratch_k_imag = (float *)malloc((size_t)nkv * n_old * fc * sizeof(float));
        rt->scratch_fc_capacity = n_old;
    }
    float * k_f32   = rt->scratch_k_f32;
    float * scores  = rt->scratch_scores;
    int   * key_pos = rt->scratch_key_pos;
    int   * phys_idx = rt->scratch_phys;   /* Task 1: physical row indices */
    float * k_real  = rt->scratch_k_real;
    float * k_imag  = rt->scratch_k_imag;

    /* Resize global scores if needed */
    if (rt->global_n < n_old) {
        float * new_gs = (float *)malloc(n_old * sizeof(float));
        if (new_gs) {
            if (!full_rescore && rt->global_scores && rt->global_n > 0) {
                int copy_n = rt->global_n < n_old ? rt->global_n : n_old;
                memcpy(new_gs, rt->global_scores, copy_n * sizeof(float));
                for (int i = copy_n; i < n_old; i++) new_gs[i] = -1e30f;
            } else {
                for (int i = 0; i < n_old; i++) new_gs[i] = -1e30f;
            }
            free(rt->global_scores);
            rt->global_scores = new_gs;
            rt->global_n = n_old;
        } else {
            rt->n_scored = n_kv;
            return 0;
        }
    } else if (full_rescore) {
        for (int i = 0; i < n_old; i++) rt->global_scores[i] = -1e30f;
        rt->global_n = n_old;
    } else {
        for (int i = n_prev; i < n_old; i++) rt->global_scores[i] = -1e30f;
    }
    rt->global_budget = budget;
    rt->compaction_active = 0;

    if (!k_f32 || !scores || !key_pos || !phys_idx || !k_real || !k_imag || !rt->global_scores) {
        rt->n_scored = n_kv;
        return 0;
    }

    /* Task 1: pass phys_idx to get physical row indices alongside positions */
    if (tria_get_kv_positions(ctx, key_pos, phys_idx, n_old) != n_old) {
        rt->n_scored = n_kv;
        return 0;
    }

    int total_pruned = 0;

    /* Precompute mean layer weight for normalization */
    float layer_weight_mean = 0.0f;
    for (int l = 0; l < nl; l++) layer_weight_mean += rt->stats->layer_budget_scales[l];
    layer_weight_mean /= nl;
    if (layer_weight_mean <= 0.0f) layer_weight_mean = 1.0f;

    /* Detect K cache type and choose scoring path */
    int score_stride = rt->cfg_score_layer_stride;
    int use_gpu_scoring = 0;
    if (rt->cfg_no_gpu_score) use_gpu_scoring = -1; /* force CPU */
    {
        struct ggml_tensor * k0 = tria_get_k_tensor(ctx, 0);
        int is_q8_0 = k0 && k0->type == GGML_TYPE_Q8_0;

        /* GPU scoring path: use for q8_0 if GPU stats are available.
         * Disabled for GQA models (nh != nkv) — GPU kernel doesn't aggregate
         * across query heads correctly. Falls back to CPU path which does
         * proper per-query-head z-normalize + max aggregation (eq 12-13). */
        int nh = rt->stats->num_heads;
        if (is_q8_0 && nh == nkv && hd <= 128 && hd % 32 == 0 && use_gpu_scoring >= 0) {
            /* Lazy upload GPU stats on first use */
            if (!rt->gpu_omega || rt->gpu_q_mean_layers != nl || rt->gpu_q_mean_kv_heads != nkv) {
                /* Build flat q_mean_real/imag arrays [nl * nkv * fc] */
                float * qmr = (float *)malloc((size_t)nl * nkv * fc * sizeof(float));
                float * qmi = (float *)malloc((size_t)nl * nkv * fc * sizeof(float));
                if (qmr && qmi) {
                    for (int li = 0; li < nl; li++) {
                        for (int kvi = 0; kvi < nkv; kvi++) {
                            struct tria_head_stats * hs = &rt->stats->heads[li * nkv + kvi];
                            const size_t base = ((size_t)li * nkv + kvi) * fc;
                            for (int f = 0; f < fc; f++) {
                                qmr[base + f] = hs->q_mean_real[f];
                                qmi[base + f] = hs->q_mean_imag[f];
                            }
                        }
                    }
                    if (g_tria_backend.stats_free)
                        g_tria_backend.stats_free(rt->gpu_omega, rt->gpu_q_mean_real, rt->gpu_q_mean_imag);
                    rt->gpu_omega = NULL;
                    rt->gpu_q_mean_real = NULL;
                    rt->gpu_q_mean_imag = NULL;
                    rt->gpu_q_mean_layers = 0;
                    rt->gpu_q_mean_kv_heads = 0;
                    if (g_tria_backend.stats_upload &&
                        g_tria_backend.stats_upload(rt->stats->omega, fc, qmr, qmi, nl * nkv,
                                              &rt->gpu_omega, &rt->gpu_q_mean_real, &rt->gpu_q_mean_imag) == 0) {
                        rt->gpu_q_mean_layers = nl;
                        rt->gpu_q_mean_kv_heads = nkv;
                    }
                }
                free(qmr);
                free(qmi);
            }
            if (rt->gpu_omega) {
                use_gpu_scoring = 1;
                score_stride = 1; /* GPU can score all layers efficiently */
            } else {
                /* Fallback: CPU with stride (use 4 if not explicitly configured) */
                if (rt->cfg_score_layer_stride == 1) score_stride = 4;
            }
        }
    }

    /* Upload this pass's global_scores slice once; all GPU layers reuse it. */
    if (use_gpu_scoring) {
        if (rt->gpu_global_scores) {
            if (g_tria_backend.stats_free)
                g_tria_backend.stats_free(rt->gpu_global_scores, NULL, NULL);
            rt->gpu_global_scores = NULL;
            rt->gpu_global_scores_n = 0;
        }
        if (g_tria_backend.stats_upload &&
            g_tria_backend.stats_upload(rt->global_scores + score_start, n_new, NULL, NULL, 0,
                                  &rt->gpu_global_scores, NULL, NULL) == 0) {
            rt->gpu_global_scores_n = n_new;
        } else {
            use_gpu_scoring = 0;
        }
    }

    /* Precompute future offsets (same as CPU path) */
    int offsets[TRIA_N_OFFSETS];
    {
        int o = 1;
        for (int i = 0; i < TRIA_N_OFFSETS; i++) {
            offsets[i] = o;
            o *= 2;
        }
    }

    /* ---- GPU turbo/GQA scoring path (opt-in via TRIA_GPU_SCORING) ----
     * Offloads the expensive 17-offset trig + two-level GQA z-norm/max onto the
     * GPU while the CPU still dequantizes K + applies the inverse WHT (identical
     * k_real/k_imag feed both paths, so results are directly comparable).
     * Distinct from the q8_0 backend path above (which requires nh == nkv). */
#ifdef GGML_TRIA_CUDA
    int gpu_turbo = 0;
    {
        tria_gpu_turbo_init_cfg();
        int nh_all = rt->stats->num_heads;
        int gpu_nd = rt->stats->nonrot_dim;
        int gqa_all = (nkv > 0 && nh_all % nkv == 0) ? nh_all / nkv : 0;
        if (tria_gpu_turbo_mode == 1 && use_gpu_scoring != 1 && !tria_random_mode &&
            rt->cfg_no_gpu_score == 0 && gpu_nd == 0 && fc > 0 && fc <= 64 &&
            gqa_all > 0) {
            gpu_turbo = 1;
            if (!tria_gpu_stats_uploaded) {
                size_t blob_n = (size_t)nl * nh_all * 4 * fc;
                float * blob = (float *)malloc(blob_n * sizeof(float));
                if (blob) {
                    for (int li2 = 0; li2 < nl; li2++) {
                        for (int h = 0; h < nh_all; h++) {
                            struct tria_head_stats * hs = &rt->stats->heads[li2 * nh_all + h];
                            float * dst = blob + ((size_t)(li2 * nh_all + h)) * 4 * fc;
                            for (int f = 0; f < fc; f++) {
                                dst[0 * fc + f] = hs->q_mean_real[f];
                                dst[1 * fc + f] = hs->q_mean_imag[f];
                                dst[2 * fc + f] = hs->q_abs_mean[f];
                                dst[3 * fc + f] = hs->qma[f];
                            }
                        }
                    }
                    if (tria_cuda_upload_stats(rt->stats->omega, fc, blob, nl, nh_all) == 0) {
                        tria_gpu_stats_uploaded = 1;
                    } else {
                        gpu_turbo = 0;
                    }
                    free(blob);
                } else {
                    gpu_turbo = 0;
                }
            }
            if (gpu_turbo) {
                if (tria_cuda_global_begin(n_old, n_new, key_pos + score_start, n_kv,
                                           nkv, gqa_all, fc, rt->global_scores) != 0) {
                    gpu_turbo = 0;
                }
            }
        }
    }
#endif

    /* Random eviction mode: assign random scores, skip all scoring logic */
    if (tria_random_mode) {
        for (int s = 0; s < n_new; s++) {
            uint32_t h = (uint32_t)(score_start + s) * 2654435761u;
            rt->global_scores[score_start + s] = (float)(h & 0xFFFF) / 65536.0f;
        }
        goto scoring_done;
    }

    for (int li = 0; li < nl; li++) {
        if (li % score_stride != 0) continue;  /* sampled-layer skip */
        struct ggml_tensor * k_tensor = tria_get_k_tensor(ctx, li);
        if (!k_tensor) continue;

        /* GPU scoring path for q8_0: score directly on GPU, no CPU transfer */
        if (use_gpu_scoring && k_tensor->type == GGML_TYPE_Q8_0 && rt->gpu_omega) {
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;

            if (rt->gpu_global_scores) {
                g_tria_backend.score_q8_0(
                    k_tensor->data,
                    n_new, score_start, n_kv,
                    n_embd_k_gqa, nkv, hd, fc,
                    key_pos + score_start,
                    rt->gpu_omega,
                    rt->gpu_q_mean_real, rt->gpu_q_mean_imag,
                    li * nkv * fc,
                    layer_weight,
                    rt->gpu_global_scores,
                    TRIA_N_OFFSETS, offsets);
            }
            continue;  /* skip CPU path for this layer */
        }

#ifdef GGML_TRIA_CUDA
        /* GPU turbo path: dequantize K, apply the inverse WHT, extract the
         * post-RoPE complex pairs and score the whole layer on the GPU, reading
         * the quantized K straight from device memory (no CPU dequant/WHT/copy).
         * Falls through to the CPU path on any failure (unsupported geometry,
         * host-resident KV, or a CUDA error). */
        if (gpu_turbo && !tria_gpu_turbo_verify && k_tensor->buffer &&
            !ggml_backend_buffer_is_host(k_tensor->buffer) &&
            (k_tensor->type == GGML_TYPE_TURBO2_0 ||
             k_tensor->type == GGML_TYPE_TURBO3_0 ||
             k_tensor->type == GGML_TYPE_TURBO4_0)) {
            int nh_all  = rt->stats->num_heads;
            int gqa_all = nh_all / nkv;
            int phys_hd = hd;
            int64_t actual_row = k_tensor->ne[0];
            if (actual_row != n_embd_k_gqa) phys_hd = (int)(actual_row / nkv);
            int phys_base = (phys_idx && score_start < n_old) ? phys_idx[score_start] : (int)score_start;
            if ((phys_hd % 128) == 0) {
                float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
                if (layer_weight < 0.25f) layer_weight = 0.25f;
                if (layer_weight > 4.0f)  layer_weight = 4.0f;
                if (tria_cuda_dequant_layer(k_tensor->data, (int)k_tensor->type,
                                            phys_base, n_new, nkv, phys_hd, fc,
                                            rt->rope_neox) == 0 &&
                    tria_cuda_score_layer_device(n_new, nkv, fc, gqa_all, li, nh_all,
                                                 tria_gpu_max_beta, layer_weight,
                                                 score_start) == 0) {
                    continue;  /* layer fully dequantized + scored on GPU */
                }
            }
        }
#endif

        /* Compute per-layer physical row width (turbo types may pad to 128 multiples) */
        int layer_phys_hd = hd;
        int n_embd_k_phys = n_embd_k_gqa;
        {
            int64_t actual_row = k_tensor->ne[0];
            if (actual_row != n_embd_k_gqa) {
                layer_phys_hd = (int)(actual_row / nkv);
                n_embd_k_phys = (int)actual_row;
            }
        }
        size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_phys);
        /* Task 1: use physical row index of the first scored cell as base */
        int phys_base = (phys_idx && score_start < n_old) ? phys_idx[score_start] : (int)score_start;
        size_t read_offset = (size_t)phys_base * row_size;
        size_t read_bytes = (size_t)n_new * row_size;

        if (k_tensor->type == GGML_TYPE_F16) {
            uint16_t * k_f16 = (uint16_t *)malloc(read_bytes);
            if (!k_f16) continue;
            ggml_backend_tensor_get(k_tensor, k_f16, read_offset, read_bytes);
            for (size_t i = 0; i < n_elem; i++) {
                k_f32[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)k_f16)[i]);
            }
            free(k_f16);
        } else if (k_tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(k_tensor, k_f32, read_offset, n_elem * sizeof(float));
        } else if (k_tensor->type == GGML_TYPE_Q8_0) {
            /* Q8_0 block: [fp16 scale d][32 x int8 qs] — sizeof = 34 bytes.
             * Only reached when score_stride reduces layer count to manageable. */
            #define TRIA_QK8_0 32
            #define TRIA_Q8_0_BLOCK_SIZE (sizeof(ggml_fp16_t) + TRIA_QK8_0)
            if (n_embd_k_gqa % TRIA_QK8_0 != 0) { continue; }
            uint8_t * k_q8 = (uint8_t *)malloc(read_bytes);
            if (!k_q8) continue;
            ggml_backend_tensor_get(k_tensor, k_q8, read_offset, read_bytes);
            const int nb = n_embd_k_gqa / TRIA_QK8_0;
            for (int s = 0; s < n_new; s++) {
                const uint8_t * row_q8 = k_q8 + (size_t)s * row_size;
                float * dst = k_f32 + (size_t)s * n_embd_k_gqa;
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = row_q8 + b * TRIA_Q8_0_BLOCK_SIZE;
                    ggml_fp16_t d_fp16;
                    memcpy(&d_fp16, blk, sizeof(ggml_fp16_t));
                    float d = ggml_fp16_to_fp32(d_fp16);
                    const int8_t * qs = (const int8_t *)(blk + sizeof(ggml_fp16_t));
                    for (int j = 0; j < TRIA_QK8_0; j++)
                        dst[b * TRIA_QK8_0 + j] = d * qs[j];
                }
            }
            free(k_q8);
        } else {
            /* Generic dequant path — handles turbo2/3/4 and any future quantized KV type.
             * Turbo types store K in WHT-rotated domain; we must apply inverse WHT
             * to recover the original RoPE frequency-pair basis for scoring. */
            const struct ggml_type_traits * traits = ggml_get_type_traits(k_tensor->type);
            if (!traits || !traits->to_float) { continue; }
            uint8_t * k_raw = (uint8_t *)malloc(read_bytes);
            float * k_phys = (float *)malloc((size_t)n_embd_k_phys * sizeof(float));
            if (!k_raw || !k_phys) { free(k_raw); free(k_phys); continue; }
            ggml_backend_tensor_get(k_tensor, k_raw, read_offset, read_bytes);
            int is_turbo = (k_tensor->type == GGML_TYPE_TURBO2_0 ||
                            k_tensor->type == GGML_TYPE_TURBO3_0 ||
                            k_tensor->type == GGML_TYPE_TURBO4_0);
            for (int s = 0; s < n_new; s++) {
                traits->to_float(k_raw + (size_t)s * row_size, k_phys, n_embd_k_phys);
                if (is_turbo && layer_phys_hd != hd) {
                    /* Inverse WHT per head, then copy logical hd prefix */
                    tria_inverse_wht_row(k_phys, k_f32 + (size_t)s * n_embd_k_gqa,
                                         nkv, layer_phys_hd, hd);
                } else if (is_turbo) {
                    /* No padding but still WHT domain — inverse WHT in-place */
                    tria_inverse_wht_row(k_phys, k_f32 + (size_t)s * n_embd_k_gqa,
                                         nkv, hd, hd);
                } else {
                    memcpy(k_f32 + (size_t)s * n_embd_k_gqa, k_phys,
                           (size_t)n_embd_k_gqa * sizeof(float));
                }
            }
            free(k_raw);
            free(k_phys);
        }

        /* Score KV heads in parallel — heads are independent. Each head uses a
         * private nkv-indexed slice of the scratch buffers, so there are no races
         * during scoring; only the final max-merge into global_scores is guarded.
         * Max is commutative, so the result is bit-identical to the serial path
         * regardless of thread count/scheduling. Dequant above already ran serially
         * (the ggml backend is not thread-safe); only CPU math is parallel here. */
#ifdef GGML_TRIA_CUDA
        /* GPU turbo path (non-verify): extract complex pairs for all KV heads,
         * then run the whole layer's scoring on the GPU and skip the CPU loop. */
        if (gpu_turbo && !tria_gpu_turbo_verify) {
            int nh_all = rt->stats->num_heads;
            int gqa_all = nh_all / nkv;
            for (int kvi = 0; kvi < nkv; kvi++) {
                float * lk_real = k_real + (size_t)kvi * n_old * fc;
                float * lk_imag = k_imag + (size_t)kvi * n_old * fc;
                for (int s = 0; s < n_new; s++) {
                    float * row = k_f32 + (size_t)s * n_embd_k_gqa + kvi * hd;
                    if (rt->rope_neox) {
                        for (int f = 0; f < fc; f++) {
                            lk_real[s * fc + f] = row[f];
                            lk_imag[s * fc + f] = row[fc + f];
                        }
                    } else {
                        for (int f = 0; f < fc; f++) {
                            lk_real[s * fc + f] = row[2 * f + 0];
                            lk_imag[s * fc + f] = row[2 * f + 1];
                        }
                    }
                }
            }
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;
            tria_cuda_score_layer(k_real, k_imag, n_old, n_new, nkv, fc, gqa_all,
                                  li, nh_all, tria_gpu_max_beta, layer_weight, score_start);
            continue;  /* layer scored on GPU */
        }
#endif
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int kvi = 0; kvi < nkv; kvi++) {
            float * lk_real = k_real + (size_t)kvi * n_old * fc;
            float * lk_imag = k_imag + (size_t)kvi * n_old * fc;
            float * lscores = scores + (size_t)kvi * n_old;

            for (int s = 0; s < n_new; s++) {
                float * row = k_f32 + (size_t)s * n_embd_k_gqa + kvi * hd;
                /* Extract complex pairs from post-RoPE K.
                 * NEOX/IMROPE: split-half [r0..r_{fc-1}, i0..i_{fc-1}]
                 * NORMAL:      interleaved [r0, i0, r1, i1, ...] */
                if (rt->rope_neox) {
                    for (int f = 0; f < fc; f++) {
                        lk_real[s * fc + f] = row[f];
                        lk_imag[s * fc + f] = row[fc + f];
                    }
                } else {
                    for (int f = 0; f < fc; f++) {
                        lk_real[s * fc + f] = row[2*f + 0];
                        lk_imag[s * fc + f] = row[2*f + 1];
                    }
                }
            }

            /* Extract non-rotary K dims (suffix after rotary portion) */
            int nd = rt->stats->nonrot_dim;
            float *k_nr = NULL;
            if (nd > 0 && 2 * fc + nd <= hd) {
                k_nr = malloc((size_t)n_new * nd * sizeof(float));
                if (k_nr) {
                    for (int s = 0; s < n_new; s++) {
                        float *row = k_f32 + (size_t)s * n_embd_k_gqa + kvi * hd;
                        memcpy(k_nr + s * nd, row + 2 * fc, nd * sizeof(float));
                    }
                }
            }

            tria_score_kv_head(rt->stats, lk_real, lk_imag, k_nr,
                               key_pos + score_start,
                               n_kv, n_new, li, kvi, lscores);
            free(k_nr);

            float mean = 0, var = 0;
            for (int s = 0; s < n_new; s++) mean += lscores[s];
            mean /= n_new;
            for (int s = 0; s < n_new; s++) {
                float d = lscores[s] - mean;
                var += d * d;
            }
            float std = sqrtf(var / n_new + 1e-8f);

            /* Layer-aware max aggregation: weight positive z-scores by layer importance.
             * Diffuse early layers (high layer_budget_scale) contribute more.
             * Leave negative z-scores unweighted: scaling negative scores would invert the
             * intended retention bias by making important layers' below-mean tokens look worse.
             * Preserves max-aggregation semantics (same eviction rate as before)
             * while using calibration data to improve token selection quality. */
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;

#ifdef _OPENMP
            #pragma omp critical (tria_global_merge)
#endif
            for (int s = 0; s < n_new; s++) {
                float z = (lscores[s] - mean) / std;
                float wz = z > 0.0f ? z * layer_weight : z;
                if (wz > rt->global_scores[score_start + s])
                    rt->global_scores[score_start + s] = wz;
            }
        }
#ifdef GGML_TRIA_CUDA
        /* GPU verify: re-score the same layer on the GPU into the device global
         * buffer (reusing the k_real/k_imag the CPU loop just extracted). The
         * device result is compared against the CPU result after the layer loop. */
        if (gpu_turbo && tria_gpu_turbo_verify) {
            int nh_all = rt->stats->num_heads;
            int gqa_all = nh_all / nkv;
            float layer_weight = rt->stats->layer_budget_scales[li] / layer_weight_mean;
            if (layer_weight < 0.25f) layer_weight = 0.25f;
            if (layer_weight > 4.0f)  layer_weight = 4.0f;
            tria_cuda_score_layer(k_real, k_imag, n_old, n_new, nkv, fc, gqa_all,
                                  li, nh_all, tria_gpu_max_beta, layer_weight, score_start);
        }
#endif
    }

#ifdef GGML_TRIA_CUDA
    if (gpu_turbo) {
        if (!tria_gpu_turbo_verify) {
            /* Production GPU path: pull the merged scores back into global_scores. */
            tria_cuda_global_download(rt->global_scores, n_old);
        } else {
            /* Verify path: compare GPU result (device) vs CPU result (host). */
            float * gtmp = (float *)malloc((size_t)n_old * sizeof(float));
            if (gtmp && tria_cuda_global_download(gtmp, n_old) == 0) {
                double max_abs = 0.0, max_rel = 0.0;
                int worst = -1;
                for (int i = score_start; i < n_old; i++) {
                    float a = rt->global_scores[i];
                    float b = gtmp[i];
                    if (a < -1e29f && b < -1e29f) continue;  /* both unscored */
                    double d = fabs((double)a - (double)b);
                    if (d > max_abs) { max_abs = d; worst = i; }
                    double r = d / (fabs((double)a) + 1e-6);
                    if (r > max_rel) max_rel = r;
                }
                LLAMA_LOG_INFO("tria_gpu_verify: max_abs=%.6g max_rel=%.6g worst=%d n_new=%d\n",
                               max_abs, max_rel, worst, n_new);
            }
            free(gtmp);
        }
    }
#endif

    if (use_gpu_scoring && rt->gpu_global_scores) {
        g_tria_backend.scores_download(rt->global_scores + score_start, rt->gpu_global_scores, n_new);
    }

    total_pruned = (n_old - budget) * nl * nkv;

scoring_done:
    /* ---- Value-aware scoring boost (VATP/OBCache-inspired) ---- */
    /* Skip V-energy in random mode — random baseline must be pure random */
    if (tria_random_mode) goto scoring_final;
    /* Skip V-energy on incremental passes.  V-energy reads ALL n_old tokens and
     * adds a per-token boost to global_scores[0..n_old-1].  On an incremental pass
     * old tokens already carry the V-boost from the last full rescore; re-applying
     * it would double-count and inflate old-token scores relative to new ones. */
    if (!full_rescore) goto scoring_final;
    /* Compute per-token V energy across all layers, z-normalize,
       and add lambda * v_z to global_scores (bidirectional).
       Only works with non-transposed V (flash_attn mode).         */
    {
        const float lambda = rt->cfg_v_lambda;
        float * v_energy = (float *)calloc(n_old, sizeof(float));
        if (v_energy) {
            int v_layers = 0;
            for (int li = 0; li < nl; li++) {
                struct ggml_tensor * v_tensor = tria_get_v_tensor(ctx, li);
                if (!v_tensor) continue;
                /* Skip transposed V (nb[1] > nb[2]) — row reads would be wrong */
                if (v_tensor->nb[1] > v_tensor->nb[2]) continue;

                int n_embd_v = (int)v_tensor->ne[0];
                size_t v_row_size = ggml_row_size(v_tensor->type, n_embd_v);

                {
                    /* V energy: compute L2 norm per row */
                    const struct ggml_type_traits * vtraits = ggml_get_type_traits(v_tensor->type);
                    if (v_tensor->type == GGML_TYPE_F32) {
                        /* f32: bulk read all n_old rows in one GPU→CPU transfer */
                        size_t total_v_bytes = (size_t)n_old * v_row_size;
                        uint8_t * v_bulk = (uint8_t *)malloc(total_v_bytes);
                        if (v_bulk) {
                            ggml_backend_tensor_get(v_tensor, v_bulk, 0, total_v_bytes);
                            for (int s = 0; s < n_old; s++) {
                                const float * vf = (const float *)(v_bulk + (size_t)s * v_row_size);
                                float energy = 0.0f;
                                for (int d = 0; d < n_embd_v; d++)
                                    energy += vf[d] * vf[d];
                                v_energy[s] += energy;
                            }
                            free(v_bulk);
                            v_layers++;
                        }
                    } else if (vtraits && vtraits->to_float) {
                        /* Generic: bulk read all rows, then dequant row-by-row */
                        size_t total_v_bytes = (size_t)n_old * v_row_size;
                        uint8_t * v_bulk = (uint8_t *)malloc(total_v_bytes);
                        float   * v_f32  = (float *)malloc((size_t)n_embd_v * sizeof(float));
                        if (v_bulk && v_f32) {
                            ggml_backend_tensor_get(v_tensor, v_bulk, 0, total_v_bytes);
                            for (int s = 0; s < n_old; s++) {
                                vtraits->to_float(v_bulk + (size_t)s * v_row_size, v_f32, n_embd_v);
                                float energy = 0.0f;
                                for (int d = 0; d < n_embd_v; d++)
                                    energy += v_f32[d] * v_f32[d];
                                v_energy[s] += energy;
                            }
                            v_layers++;
                        }
                        free(v_bulk);
                        free(v_f32);
                    }
                }
            }

            if (v_layers > 0) {
                /* log1p + z-normalize */
                for (int s = 0; s < n_old; s++)
                    v_energy[s] = logf(1.0f + v_energy[s] / v_layers);

                float vmean = 0.0f;
                for (int s = 0; s < n_old; s++) vmean += v_energy[s];
                vmean /= n_old;

                float vvar = 0.0f;
                for (int s = 0; s < n_old; s++) {
                    float d = v_energy[s] - vmean;
                    vvar += d * d;
                }
                float vstd = sqrtf(vvar / n_old + 1e-8f);

                /* Guard: if variance collapsed (WHT-normalized turbo3), skip boost */
                if (vstd > 0.01f) {
                    int boosted = 0;
                    for (int s = 0; s < n_old; s++) {
                        float vz = (v_energy[s] - vmean) / vstd;
                        float boost = lambda * vz;
                        rt->global_scores[s] += boost;
                        if (boost > 0.01f) boosted++;
                    }
                    LLAMA_LOG_INFO("tria_score: value-aware boost applied to %d/%d tokens (lambda=%.2f, vstd=%.4f)\n",
                            boosted, n_old, lambda, vstd);
                } else {
                    LLAMA_LOG_INFO("tria_score: value-aware skipped (vstd=%.6f < 0.01, WHT-normalized)\n", vstd);
                }
            }
            free(v_energy);
        }
    }

scoring_final:
    if (total_pruned > 0) {
        LLAMA_LOG_INFO("tria_score: pruned %d tokens across %d×%d heads\n",
                total_pruned, nl, nkv);
    }

    {
        const int compacted = tria_compact_kv(rt, ctx);
        if (compacted > 0) {
            int new_n = tria_get_used_n_kv(ctx);
            LLAMA_LOG_INFO("tria: evicted %d cells → %d remaining (window=%d)\n",
                           compacted, new_n, rt->window);
            int new_old = new_n - rt->window;
            if (new_old > 0 && new_old <= rt->global_n) {
                rt->global_n = new_old;
            }
            rt->compaction_active = 1;
            rt->n_scored = tria_get_n_kv(ctx);
            rt->score_pass++;
            return compacted;
        }
    }

    /* Scores are fresh and no compaction happened this pass — clear the flag so
     * subsequent passes can use the cheaper incremental path. */
    rt->compaction_active = 0;
    rt->n_scored = n_kv;
    rt->score_pass++;
    return total_pruned;
}

/* Comparator for segment eviction sort — global scores ptr set before each use */
static const float * _g_evict_scores_ptr = NULL;

static int _evict_idx_cmp_asc(const void * a, const void * b) {
    float sa = _g_evict_scores_ptr[*(const int *)a];
    float sb = _g_evict_scores_ptr[*(const int *)b];
    return (sa > sb) - (sa < sb);  /* ascending: lowest score = most evictable */
}

int tria_get_evict_mask(
    const struct tria_runtime * rt,
    int n_kv,
    int8_t * evict_mask
) {
    if (!rt || rt->n_scored == 0 || !evict_mask) return 0;
    if (rt->compaction_active) return 0;
    if (!rt->global_scores || rt->global_budget <= 0) return 0;

    int n_old = rt->n_scored - rt->window;
    if (n_old <= 0) return 0;

    /* Clamp n_old to actual mask/scores bounds (Codex review: OOB fix) */
    if (n_old > n_kv) n_old = n_kv;
    if (n_old > rt->global_n) n_old = rt->global_n;

    int budget = rt->global_budget;
    if (budget >= n_old) {
        memset(evict_mask, 0, n_kv);
        return 1;
    }

    /* Quickselect for threshold — use local buffer, not static (Codex review: thread safety) */
    float * sorted = (float *)malloc(n_old * sizeof(float));
    if (!sorted) return 0;
    memcpy(sorted, rt->global_scores, n_old * sizeof(float));

    int lo = 0, hi = n_old - 1, target = budget - 1;
    while (lo < hi) {
        float pivot = sorted[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (sorted[i] > pivot) i++;
            while (sorted[j] < pivot) j--;
            if (i <= j) {
                float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
                i++; j--;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }
    /* threshold is unused by the segment-based eviction below,
       but kept for potential future use */
    (void)sorted[target];
    free(sorted);

    /* Build mask: V3 hybrid — prefix protection + per-segment eviction quota */
    memset(evict_mask, 0, n_kv);

    const int n_segments = 8;
    int prefix = rt->sink > 0 ? rt->sink : 128;
    if (prefix > n_old) prefix = n_old;

    int evictable = n_old - prefix;
    int n_to_evict = n_old - budget;
    if (evictable <= 0 || n_to_evict <= 0) return 1;

    int seg_size = evictable / n_segments;
    if (seg_size < 1) seg_size = 1;
    int actual_segs = (evictable + seg_size - 1) / seg_size;

    int total_evicted = 0;
    for (int s = 0; s < actual_segs && total_evicted < n_to_evict; s++) {
        int seg_start = prefix + s * seg_size;
        int seg_end   = seg_start + seg_size;
        if (seg_end > n_old) seg_end = n_old;
        int seg_len = seg_end - seg_start;
        if (seg_len <= 0) continue;

        int seg_evict = (n_to_evict * seg_len + evictable - 1) / evictable;
        int remaining = n_to_evict - total_evicted;
        if (seg_evict > remaining) seg_evict = remaining;
        if (seg_evict > seg_len)   seg_evict = seg_len;
        if (seg_evict <= 0) continue;

        int * idx = (int *)malloc(seg_len * sizeof(int));
        if (!idx) continue;
        for (int i = 0; i < seg_len; i++) idx[i] = seg_start + i;

        /* Sort ascending by score: lowest score = first to evict */
        _g_evict_scores_ptr = rt->global_scores;
        qsort(idx, (size_t)seg_len, sizeof(int), _evict_idx_cmp_asc);

        /* Mark the seg_evict lowest-scoring positions as evicted */
        for (int e = 0; e < seg_evict && total_evicted < n_to_evict; e++) {
            if (idx[e] >= 0 && idx[e] < n_kv) {
                evict_mask[idx[e]] = 1;
            }
            total_evicted++;
        }
        free(idx);
    }

    return 1;
}
