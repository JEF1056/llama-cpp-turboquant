/*
 * llama-log-c.h — C-compatible LLAMA_LOG_* macros
 *
 * For C translation units that cannot include llama-impl.h (C++ only).
 * Declares llama_log_internal with extern "C" linkage and provides
 * the same LLAMA_LOG_* macros as llama-impl.h.
 */
#pragma once
#include "ggml.h"   /* ggml_log_level */

#ifdef __cplusplus
extern "C" {
#endif

void llama_log_internal(enum ggml_log_level level, const char * format, ...);

#ifdef __cplusplus
}
#endif

#define LLAMA_LOG_INFO(...)  llama_log_internal((enum ggml_log_level)GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal((enum ggml_log_level)GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal((enum ggml_log_level)GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal((enum ggml_log_level)GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal((enum ggml_log_level)GGML_LOG_LEVEL_CONT,  __VA_ARGS__)
