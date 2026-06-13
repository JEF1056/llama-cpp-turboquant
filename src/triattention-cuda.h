/*
 * triattention-cuda.h — host-side declarations for the GPU TriAttention scorer.
 *
 * Implemented in ggml/src/ggml-cuda/triattention-score.cu and linked into the
 * ggml-cuda backend shared library.  The `llama` target resolves these
 * default-visibility extern "C" symbols through the PUBLIC link chain
 * llama -> ggml -> ggml-cuda.  Only compiled into the call sites when the build
 * enables CUDA (GGML_TRIA_CUDA, set by src/CMakeLists.txt when GGML_CUDA=ON).
 */
#ifndef TRIATTENTION_CUDA_H
#define TRIATTENTION_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Upload per-head scoring stats once (idempotent for matching nl/nh/fc).
 * head_stats layout: [ (li*nh + h) * 4 * fc ] with per-head blocks
 *   q_mean_real[fc], q_mean_imag[fc], q_abs_mean[fc], qma[fc].
 * Returns 0 on success. */
int tria_cuda_upload_stats(const float * omega, int fc,
                           const float * head_stats, int nl, int nh);

/* Begin a scoring pass: (re)size device buffers, upload the scored-token
 * absolute positions and the initial global-score state. Returns 0 on success. */
int tria_cuda_global_begin(int n_old, int n_new, const int * key_pos_scored,
                           int cur_pos, int nkv, int gqa, int fc,
                           const float * global_init);

/* Score one attention layer. kr_host/ki_host are host buffers laid out as
 * [nkv][kvi_stride_rows][fc]; only the first n_new rows per KV head are read.
 * Max-merges layer-weighted z-scores into the persistent device global buffer. */
int tria_cuda_score_layer(const float * kr_host, const float * ki_host,
                          int kvi_stride_rows,
                          int n_new, int nkv, int fc, int gqa,
                          int layer_idx, int num_heads,
                          float max_beta, float layer_weight, int score_start);

/* Download the merged global scores (n elements) to host. Returns 0 on success. */
int tria_cuda_global_download(float * dst_host, int n);

/* Free all device buffers. */
void tria_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TRIATTENTION_CUDA_H */
