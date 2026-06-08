#pragma once

// Runtime sync-after-kernel for CUDA crash localisation.
// Set GGML_CUDA_SYNC_DEBUG=1 in the environment to enable.
// Zero overhead when unset — checked once and cached.
//
// Include this header (not set-rows.cuh) in any TU that needs the macro.

#include "common.cuh"
#include <cstdio>
#include <cstdlib>

static inline bool ggml_cuda_sync_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * v = getenv("GGML_CUDA_SYNC_DEBUG");
        enabled = (v && v[0] == '1') ? 1 : 0;
        if (enabled) {
            fprintf(stderr, "[ggml-cuda] GGML_CUDA_SYNC_DEBUG=1: synchronizing after each turbo set_rows / fattn kernel\n");
        }
    }
    return enabled == 1;
}

#define GGML_CUDA_SYNC_DEBUG_CHECK(stream) \
    do { if (ggml_cuda_sync_debug_enabled()) { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream)); } } while(0)
