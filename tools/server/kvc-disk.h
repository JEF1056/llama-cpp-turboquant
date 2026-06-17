#pragma once

#include "llama.h"
#include "common.h"
#include "server-task.h"

#include <cstdint>
#include <string>
#include <vector>

/**
 * kvc-disk: KV-cache disk persistence for llama-server.
 *
 * Each cache file uses the "TKVD" format:
 *   [validation header][CRC-checked KV state blob][token array]
 *
 * Header layout (little-endian):
 *   u32  magic (0x44564B54 = "TKVD")
 *   u32  version
 *   u64  saved_unix_seconds — wall-clock time (time(2)) the file was last written.
 *        Placed immediately after the version so it can be read with a single
 *        16-byte prefix read (see kvc_disk_peek_timestamp) without touching the
 *        large KV blob — used to expire stale caches cheaply.
 *   u8[32] model_arch (null-padded)
 *   u64  model_size_bytes
 *   u32  n_layer, n_head, n_head_kv, n_ctx_train
 *   u32  type_k, type_v
 *   u32  n_ctx_orig_yarn
 *   f32  yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow
 *   f32  rope_freq_base, rope_freq_scale
 *   u32  n_tokens           — count of saved tokens (including LLAMA_TOKEN_NULL for images)
 *   u32  n_pos              — slot.prompt.tokens.pos_next() at save time (M-RoPE correct)
 *   u32  n_mtmd_chunks
 *   [per chunk: u32 seq_start, u32 seq_len, u32 pos_type, u32 id_len, char id[id_len]]
 *   u64  kv_data_size
 *   u32  kv_crc32           — CRC-32 over the KV blob only
 *   [kv_data_size bytes: raw llama_state_seq_get_data_ext output]
 *   [n_tokens × i32: raw token array including LLAMA_TOKEN_NULL (-1) for image positions]
 *
 * The token array is appended after the KV blob so the CRC region is unchanged.
 * Token array integrity is implicitly covered by the model fingerprint: a different
 * GGUF (different vocab / arch) would fail the fingerprint checks before tokens are used.
 *
 * On load every header field is validated against the running model/context.
 * Any mismatch causes the file to be skipped with a [KVC] log message — the
 * server never loads a cache that could corrupt attention scores.
 */

// Magic bytes for TKVD cache files ("TKVD" in little-endian u32)
static constexpr uint32_t KVC_DISK_MAGIC   = 0x44564B54u;
// Slot-cache format version. Bumped to 4 when the saved_unix_seconds field was
// added to the header; older files are treated as invalid and discarded.
static constexpr uint32_t KVC_DISK_VERSION = 4;

// Fingerprint of the model + context at save time.
// Stored in the file header so we can detect stale / incompatible caches.
uint32_t crc32_buf(const void * buf, size_t len);

struct kvc_disk_fingerprint {
    char     model_arch[32] = {};   // null-padded GGUF arch string
    uint64_t model_size     = 0;    // st_size of the .gguf file
    uint32_t n_layer        = 0;
    uint32_t n_head         = 0;
    uint32_t n_head_kv      = 0;
    uint32_t n_ctx_train    = 0;
    uint32_t type_k         = 0;    // ggml_type of K cache
    uint32_t type_v         = 0;    // ggml_type of V cache
    // RoPE / YaRN params that were active when the K-cache was written
    uint32_t n_ctx_orig_yarn    = 0;
    float    yarn_ext_factor    = 0.0f;
    float    yarn_attn_factor   = 0.0f;
    float    yarn_beta_fast     = 0.0f;
    float    yarn_beta_slow     = 0.0f;
    float    rope_freq_base     = 0.0f;
    float    rope_freq_scale    = 0.0f;
};

// Build a fingerprint from the currently loaded model and context.
kvc_disk_fingerprint kvc_disk_make_fingerprint(
    const llama_model   * model,
    llama_context       * ctx,
    const std::string   & model_path);

// Write a single slot's KV state to |path|.
// Prepends the TKVD v2 validation header (fingerprint + MTMD positions) and a
// CRC-32 over the raw KV blob. The token array is appended after the blob.
// Returns true on success.
bool kvc_disk_write(
    const std::string                    & path,
    const kvc_disk_fingerprint           & fp,
    const std::vector<mtmd_chunk_pos>    & mtmd_positions,
    llama_context                        * ctx,
    llama_seq_id                           seq_id,
    int32_t                                n_tokens,
    llama_pos                              n_pos,     // pos_next() at save time (M-RoPE correct)
    const llama_tokens                   & tokens);  // full token array incl. LLAMA_TOKEN_NULL

// Validate and load a TKVD v2 cache file into |ctx| at |seq_id|.
// On any mismatch or corruption the file is skipped (returns false) and a
// [KVC] log line explains why.
// On success |mtmd_positions_out|, |n_pos_out| and |tokens_out| are populated.
bool kvc_disk_read(
    const std::string              & path,
    const kvc_disk_fingerprint     & expected_fp,
    llama_context                  * ctx,
    llama_seq_id                     seq_id,
    std::vector<mtmd_chunk_pos>    & mtmd_positions_out,
    int32_t                        & n_tokens_out,
    llama_pos                      & n_pos_out,      // restored pos_next() value
    llama_tokens                   & tokens_out);    // restored token array

// Prompt-cache format version. Bumped to 5 when the saved_unix_seconds field was
// added to the header; older files are treated as invalid and discarded.
static constexpr uint32_t KVC_DISK_VERSION_V3 = 5;

// Quickly read just the last-updated timestamp from a TKVD cache file without
// loading the (potentially multi-GiB) KV blob. Reads only the fixed 16-byte
// prefix: magic(4) + version(4) + saved_unix_seconds(8).
// Returns true and sets |out_unix_seconds| only for a well-formed file with a
// recognised magic + version; returns false for an unreadable, truncated,
// wrong-magic or unknown-version file (i.e. "invalid" — safe to delete).
bool kvc_disk_peek_timestamp(const std::string & path, uint64_t & out_unix_seconds);

// Refresh a TKVD cache file's last-updated timestamp to the current wall-clock
// time by rewriting only the 8-byte field in place (no blob re-write). Used to
// keep a still-loaded idle slot's on-disk cache from being expired-deleted while
// its KV is live in RAM. Returns false for an unreadable / wrong-magic /
// unknown-version file.
bool kvc_disk_touch(const std::string & path);

// Write a server_prompt cache entry (including draft data and speculative checkpoints) to |path|.
// Uses the version-5 prompt format.
bool kvc_disk_write_prompt(
    const std::string                    & path,
    const kvc_disk_fingerprint           & fp,
    const server_prompt                  & prompt);

// Validate and load a version 3 prompt cache file from |path| into |prompt|.
bool kvc_disk_read_prompt(
    const std::string              & path,
    const kvc_disk_fingerprint     & expected_fp,
    server_prompt                  & prompt);

