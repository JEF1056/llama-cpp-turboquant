#include "kvc-disk.h"

#include "llama.h"
#include "log.h"
#include "mtmd.h"

// LLAMA_LOG_INFO / LLAMA_LOG_WARN / LLAMA_LOG_ERROR come from log.h (llama-server)
// which maps them to the common logging infrastructure.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// CRC-32 (ISO 3309 polynomial)
// ---------------------------------------------------------------------------

static uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t len) {
    static const uint32_t table[256] = {
        // Generated with polynomial 0xEDB88320 (reflected CRC-32)
        0x00000000,0x77073096,0xEE0E612C,0x990951BA,0x076DC419,0x706AF48F,
        0xE963A535,0x9E6495A3,0x0EDB8832,0x79DCB8A4,0xE0D5E91B,0x97D2D988,
        0x09B64C2B,0x7EB17CBF,0xE7B82D09,0x90BF1C9F,0x1DB71064,0x6AB020F2,
        0xF3B97148,0x84BE41DE,0x1ADAD47D,0x6DDDE4EB,0xF4D4B551,0x83D385C7,
        0x136C9856,0x646BA8C0,0xFD62F97A,0x8A65C9EC,0x14015C4F,0x63066CD9,
        0xFA0F3D63,0x8D080DF5,0x3B6E20C8,0x4C69105E,0xD56041E4,0xA2677172,
        0x3C03E4D1,0x4B04D447,0xD20D85FD,0xA50AB56B,0x35B5A8FA,0x42B2986C,
        0xDBBBC9D6,0xACBCF940,0x32D86CE3,0x45DF5C75,0xDCD60DCF,0xABD13D59,
        0x26D930AC,0x51DE003A,0xC8D75180,0xBFD06116,0x21B4F6B5,0x56B3C423,
        0xCFBA9599,0xB8BDA50F,0x2802B89E,0x5F058808,0xC60CD9B2,0xB10BE924,
        0x2F6F7C87,0x58684C11,0xC1611DAB,0xB6662D3D,0x76DC4190,0x01DB7106,
        0x98D220BC,0xEFD5102A,0x71B18589,0x06B6B51F,0x9FBFE4A5,0xE8B8D433,
        0x7807C9A2,0x0F00F934,0x9609A88E,0xE10E9818,0x7F6979BB,0x086D3D2D,
        0x91646C97,0xE6635C01,0x6B6B51F4,0x1C6C6162,0x856530D8,0xF262004E,
        0x6C0695ED,0x1B01A57B,0x8208F4C1,0xF50FC457,0x65B0D9C6,0x12B7E950,
        0x8BBEB8EA,0xFCB9887C,0x62DD1BDF,0x15DA2D49,0x8CD37CF3,0xFBD44C65,
        0x4DB26158,0x3AB551CE,0xA3BC0074,0xD4BB30E2,0x4ADFA541,0x3DD895D7,
        0xA4D1C46D,0xD3D6F4FB,0x4369E96A,0x346ED9FC,0xAD678846,0xDA60B8D0,
        0x44042D73,0x33031DE5,0xAA0A4C5F,0xDD0D7CC9,0x5005713C,0x270241AA,
        0xBE0B1010,0xC90C2086,0x5768B525,0x206F85B3,0xB966D409,0xCE61E49F,
        0x5EDEF90E,0x29D9C998,0xB0D09822,0xC7D7A8B4,0x59B33D17,0x2EB40D81,
        0xB7BD5C3B,0xC0BA6CAD,0xEDB88320,0x9ABFB3B6,0x03B6E20C,0x74B1D29A,
        0xEAD54739,0x9DD277AF,0x04DB2615,0x73DC1683,0xE3630B12,0x94643B84,
        0x0D6D6A3E,0x7A6A5AA8,0xE40ECF0B,0x9309FF9D,0x0A00AE27,0x7D079EB1,
        0xF00F9344,0x8708A3D2,0x1E01F268,0x6906C2FE,0xF762575D,0x806567CB,
        0x196C3671,0x6E6B06E7,0xFED41B76,0x89D32BE0,0x10DA7A5A,0x67DD4ACC,
        0xF9B9DF6F,0x8EBEEFF9,0x17B7BE43,0x60B08ED5,0xD6D6A3E8,0xA1D1937E,
        0x38D8C2C4,0x4FDFF252,0xD1BB67F1,0xA6BC5767,0x3FB506DD,0x48B2364B,
        0xD80D2BDA,0xAF0A1B4C,0x36034AF6,0x41047A60,0xDF60EFC3,0xA8670955,
        0x316658EF,0x466769AB,0xB40BBE37,0xC30C8EA1,0x5A05DF1B,0x2D02EF8D,
    };
    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc;
}

uint32_t crc32_buf(const void * buf, size_t len) {
    return crc32_update(0xFFFFFFFFu, (const uint8_t *) buf, len) ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Fingerprint helpers
// ---------------------------------------------------------------------------

kvc_disk_fingerprint kvc_disk_make_fingerprint(
        const llama_model   * model,
        llama_context       * ctx,
        const std::string   & model_path) {
    kvc_disk_fingerprint fp = {};

    // model arch string (null-padded, max 31 chars)
    char desc[128] = {};
    llama_model_desc(model, desc, sizeof(desc));
    // desc is like "LLaMA v3 8B Q4_0" – extract just the arch identifier
    // For fingerprinting we use the raw llama_model_meta_val_str for "general.architecture"
    char arch_buf[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) > 0) {
        std::strncpy(fp.model_arch, arch_buf, sizeof(fp.model_arch) - 1);
    } else {
        std::strncpy(fp.model_arch, desc, sizeof(fp.model_arch) - 1);
    }

    // model file size (0 if unavailable, e.g. in-memory model)
    if (!model_path.empty()) {
        std::error_code ec;
        fp.model_size = (uint64_t) std::filesystem::file_size(model_path, ec);
        if (ec) {
            fp.model_size = 0;
            LOG_WRN("[KVC] could not stat model file '%s': %s\n",
                model_path.c_str(), ec.message().c_str());
        }
    }

    fp.n_layer     = (uint32_t) llama_model_n_layer(model);
    fp.n_head      = (uint32_t) llama_model_n_head(model);
    fp.n_head_kv   = (uint32_t) llama_model_n_head_kv(model);
    fp.n_ctx_train = (uint32_t) llama_model_n_ctx_train(model);

    // KV cache types — retrieve via context params
    const llama_context_params cparams_dummy = llama_context_default_params();
    (void) cparams_dummy;
    // Use the public API to query what the context was built with.
    // llama_context_default_params gives defaults; the actual types were set at
    // context creation so we query the model directly for dtype defaults and
    // rely on the model arch.  For now we store GGML_TYPE_F16 as a fallback.
    // TODO: expose llama_context_get_type_k/v in the public API if needed.
    fp.type_k = (uint32_t) GGML_TYPE_F16;
    fp.type_v = (uint32_t) GGML_TYPE_F16;

    // RoPE / YaRN — read from context internal state via the public get_params API
    // We use llama_state_save_file output size as a proxy; actual params come from
    // the context's llama_cparams which are not directly exposed.
    // We populate these inside kvc_disk_write() where we have cparams.
    return fp;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

bool kvc_disk_write(
        const std::string                    & path,
        const kvc_disk_fingerprint           & fp,
        const std::vector<mtmd_chunk_pos>    & mtmd_positions,
        llama_context                        * ctx,
        llama_seq_id                           seq_id,
        int32_t                                n_tokens,
        llama_pos                              n_pos,
        const llama_tokens                   & tokens) {
    int64_t t0 = ggml_time_ms();

    // 1. Serialise the KV state to a buffer
    const size_t kv_size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (kv_size == 0) {
        LOG_INF("[KVC] slot seq=%d: KV state size is 0, skipping write\n", (int) seq_id);
        return false;
    }

    std::vector<uint8_t> kv_buf(kv_size);
    const size_t written = llama_state_seq_get_data_ext(ctx, kv_buf.data(), kv_size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (written == 0 || written > kv_size) {
        LOG_WRN("[KVC] slot seq=%d: llama_state_seq_get_data returned %zu (expected %zu)\n",
            (int) seq_id, written, kv_size);
        return false;
    }
    kv_buf.resize(written);

    // 2. Compute CRC-32 over the KV blob
    const uint32_t kv_crc = crc32_buf(kv_buf.data(), kv_buf.size());

    // 3. Open the output file
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_WRN("[KVC] slot seq=%d: cannot open '%s' for writing\n",
            (int) seq_id, path.c_str());
        return false;
    }

    auto write_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto write_u64 = [&](uint64_t v) { fwrite(&v, 8, 1, f); };
    auto write_f32 = [&](float  v)   { fwrite(&v, 4, 1, f); };

    // 4. Write TKVD header
    write_u32(KVC_DISK_MAGIC);
    write_u32(KVC_DISK_VERSION);

    // fingerprint
    fwrite(fp.model_arch, 1, sizeof(fp.model_arch), f);
    write_u64(fp.model_size);
    write_u32(fp.n_layer);
    write_u32(fp.n_head);
    write_u32(fp.n_head_kv);
    write_u32(fp.n_ctx_train);
    write_u32(fp.type_k);
    write_u32(fp.type_v);
    // RoPE/YaRN params (from fingerprint – caller must have filled these)
    write_u32(fp.n_ctx_orig_yarn);
    write_f32(fp.yarn_ext_factor);
    write_f32(fp.yarn_attn_factor);
    write_f32(fp.yarn_beta_fast);
    write_f32(fp.yarn_beta_slow);
    write_f32(fp.rope_freq_base);
    write_f32(fp.rope_freq_scale);

    // token count + KV position count (v2: n_pos enables M-RoPE-correct seq_rm on restore)
    write_u32((uint32_t) n_tokens);
    write_u32((uint32_t) n_pos);

    // MTMD chunk positions
    write_u32((uint32_t) mtmd_positions.size());
    for (const auto & cp : mtmd_positions) {
        write_u32((uint32_t) cp.seq_start);
        write_u32((uint32_t) cp.seq_len);
        write_u32((uint32_t) cp.pos_type);
        const uint32_t id_len = (uint32_t) cp.id.size();
        write_u32(id_len);
        if (id_len > 0) {
            fwrite(cp.id.data(), 1, id_len, f);
        }
    }

    // KV blob size (u64 to support large models) + CRC
    write_u64((uint64_t) kv_buf.size());
    write_u32(kv_crc);

    // 5. Write KV blob
    const size_t nw = fwrite(kv_buf.data(), 1, kv_buf.size(), f);
    if (nw != kv_buf.size()) {
        fclose(f);
        LOG_WRN("[KVC] slot seq=%d: short write to '%s' (%zu/%zu bytes)\n",
            (int) seq_id, path.c_str(), nw, kv_buf.size());
        // leave the partial file; next validate will reject it via CRC
        return false;
    }

    // 6. Write token array (v2: n_tokens × i32, including LLAMA_TOKEN_NULL for images)
    const size_t nt = fwrite(tokens.data(), sizeof(llama_token), tokens.size(), f);
    fclose(f);

    if (nt != tokens.size()) {
        LOG_WRN("[KVC] slot seq=%d: short write of token array to '%s' (%zu/%zu tokens)\n",
            (int) seq_id, path.c_str(), nt, tokens.size());
        return false;
    }

    const int64_t elapsed = ggml_time_ms() - t0;
    LOG_INF("[KVC] slot seq=%d: wrote '%s' "
        "(tokens=%d, n_pos=%d, kv=%.2f MiB, mtmd_chunks=%zu) in %" PRId64 " ms\n",
        (int) seq_id, path.c_str(),
        n_tokens, (int) n_pos, (double) kv_buf.size() / (1024.0 * 1024.0),
        mtmd_positions.size(), elapsed);
    return true;
}

// ---------------------------------------------------------------------------
// Read / validate
// ---------------------------------------------------------------------------

bool kvc_disk_read(
        const std::string              & path,
        const kvc_disk_fingerprint     & expected,
        llama_context                  * ctx,
        llama_seq_id                     seq_id,
        std::vector<mtmd_chunk_pos>    & mtmd_positions_out,
        int32_t                        & n_tokens_out,
        llama_pos                      & n_pos_out,
        llama_tokens                   & tokens_out) {
    int64_t t0 = ggml_time_ms();

    // helper to get basename for log messages
    const std::string fname = std::filesystem::path(path).filename().string();

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_WRN("[KVC] %s: cannot open for reading\n", fname.c_str());
        return false;
    }

    auto read_u32 = [&](uint32_t & v) { return fread(&v, 4, 1, f) == 1; };
    auto read_u64 = [&](uint64_t & v) { return fread(&v, 8, 1, f) == 1; };
    auto read_f32 = [&](float    & v) { return fread(&v, 4, 1, f) == 1; };

// Note: __VA_OPT__ would be cleaner but requires C++20; ## is a GNU extension that
// is supported by all compilers this project targets.
#define KVC_SKIP(fmt, ...) do { \
    LOG_WRN("[KVC] %s: " fmt " - skipped\n", fname.c_str(), ##__VA_ARGS__); \
    fclose(f); return false; } while(0)

    // --- magic + version
    uint32_t magic = 0, version = 0;
    if (!read_u32(magic) || !read_u32(version)) {
        KVC_SKIP("unexpected EOF reading header");
    }
    if (magic != KVC_DISK_MAGIC) {
        KVC_SKIP("not a TKVD cache file (magic=0x%08X)", magic);
    }
    if (version != KVC_DISK_VERSION) {
        KVC_SKIP("unsupported cache version %u (expected %u)", version, KVC_DISK_VERSION);
    }

    // --- fingerprint
    kvc_disk_fingerprint saved = {};
    if (fread(saved.model_arch, 1, sizeof(saved.model_arch), f) != sizeof(saved.model_arch)) {
        KVC_SKIP("unexpected EOF reading arch string");
    }
    if (!read_u64(saved.model_size)  ||
        !read_u32(saved.n_layer)     || !read_u32(saved.n_head) ||
        !read_u32(saved.n_head_kv)   || !read_u32(saved.n_ctx_train) ||
        !read_u32(saved.type_k)      || !read_u32(saved.type_v) ||
        !read_u32(saved.n_ctx_orig_yarn) ||
        !read_f32(saved.yarn_ext_factor) ||
        !read_f32(saved.yarn_attn_factor)||
        !read_f32(saved.yarn_beta_fast)  ||
        !read_f32(saved.yarn_beta_slow)  ||
        !read_f32(saved.rope_freq_base)  ||
        !read_f32(saved.rope_freq_scale)) {
        KVC_SKIP("unexpected EOF reading fingerprint");
    }

    // --- validate fingerprint
    if (std::strncmp(saved.model_arch, expected.model_arch, sizeof(expected.model_arch)) != 0) {
        KVC_SKIP("arch mismatch (saved='%.31s' loaded='%.31s')",
            saved.model_arch, expected.model_arch);
    }
    if (expected.model_size > 0 && saved.model_size != expected.model_size) {
        KVC_SKIP("model file changed (size %" PRIu64 " → %" PRIu64 ")",
            saved.model_size, expected.model_size);
    }
    if (saved.n_layer   != expected.n_layer   ||
        saved.n_head    != expected.n_head     ||
        saved.n_head_kv != expected.n_head_kv) {
        KVC_SKIP("incompatible model shape (layers %u/%u, heads %u/%u, kv-heads %u/%u)",
            saved.n_layer, expected.n_layer,
            saved.n_head,  expected.n_head,
            saved.n_head_kv, expected.n_head_kv);
    }
    if (saved.type_k != expected.type_k) {
        LOG_WRN("[KVC] %s: K-cache dtype changed (saved=%u loaded=%u)\n",
            fname.c_str(), saved.type_k, expected.type_k);
        // soft warn only — dtype changes could indicate a different quant choice
        // but the blob still encodes the right data type tag internally
    }
    // YaRN param validation (any mismatch → skip; wrong params = corrupted attention)
    bool yarn_ok = true;
    if (saved.n_ctx_orig_yarn != expected.n_ctx_orig_yarn) {
        LOG_WRN("[KVC] %s: n_ctx_orig_yarn changed (%u → %u)\n",
            fname.c_str(), saved.n_ctx_orig_yarn, expected.n_ctx_orig_yarn);
        yarn_ok = false;
    }
    if (saved.yarn_ext_factor != expected.yarn_ext_factor) {
        LOG_WRN("[KVC] %s: yarn_ext_factor changed (%.4f → %.4f)\n",
            fname.c_str(), saved.yarn_ext_factor, expected.yarn_ext_factor);
        yarn_ok = false;
    }
    if (saved.rope_freq_base != expected.rope_freq_base) {
        LOG_WRN("[KVC] %s: rope_freq_base changed (%.1f → %.1f)\n",
            fname.c_str(), saved.rope_freq_base, expected.rope_freq_base);
        yarn_ok = false;
    }
    if (!yarn_ok) {
        KVC_SKIP("RoPE/YaRN params changed — stale cache");
    }

    // --- token count + KV position count (v2)
    uint32_t saved_n_tokens = 0;
    if (!read_u32(saved_n_tokens)) {
        KVC_SKIP("unexpected EOF reading token count");
    }
    uint32_t saved_n_pos = 0;
    if (!read_u32(saved_n_pos)) {
        KVC_SKIP("unexpected EOF reading n_pos (v2 field)");
    }

    // --- MTMD positions
    uint32_t n_mtmd = 0;
    if (!read_u32(n_mtmd)) {
        KVC_SKIP("unexpected EOF reading mtmd chunk count");
    }
    std::vector<mtmd_chunk_pos> mtmd_positions;
    mtmd_positions.reserve(n_mtmd);
    for (uint32_t i = 0; i < n_mtmd; i++) {
        uint32_t seq_start = 0, seq_len = 0, pos_type = 0, id_len = 0;
        if (!read_u32(seq_start) || !read_u32(seq_len) ||
            !read_u32(pos_type)  || !read_u32(id_len)) {
            KVC_SKIP("unexpected EOF reading mtmd chunk %u", i);
        }
        mtmd_chunk_pos cp;
        cp.seq_start = (int32_t) seq_start;
        cp.seq_len   = (int32_t) seq_len;
        cp.pos_type  = (int)     pos_type;
        if (id_len > 0) {
            cp.id.resize(id_len);
            if (fread(cp.id.data(), 1, id_len, f) != id_len) {
                KVC_SKIP("unexpected EOF reading mtmd chunk %u id", i);
            }
        }
        mtmd_positions.push_back(std::move(cp));
    }

    // --- KV blob size (u64) + CRC
    uint64_t kv_blob_size = 0;
    uint32_t saved_crc    = 0;
    if (!read_u64(kv_blob_size) || !read_u32(saved_crc)) {
        KVC_SKIP("unexpected EOF reading KV blob header");
    }
    if (kv_blob_size == 0) {
        KVC_SKIP("KV blob size is zero");
    }

    // --- read KV blob
    std::vector<uint8_t> kv_buf(kv_blob_size);
    if (fread(kv_buf.data(), 1, kv_blob_size, f) != kv_blob_size) {
        KVC_SKIP("unexpected EOF reading KV blob (%u bytes)", kv_blob_size);
    }

    // --- read token array (v2: n_tokens × i32, appended after KV blob)
    // Must be read before fclose — file cursor is positioned right after the KV blob.
    llama_tokens tok_array;
    if (saved_n_tokens > 0) {
        tok_array.resize(saved_n_tokens);
        if (fread(tok_array.data(), sizeof(llama_token), saved_n_tokens, f) != saved_n_tokens) {
            KVC_SKIP("unexpected EOF reading token array (%u tokens)", saved_n_tokens);
        }
    }

    fclose(f);
    f = nullptr;

    // --- CRC check
    const uint32_t computed_crc = crc32_buf(kv_buf.data(), kv_buf.size());
    if (computed_crc != saved_crc) {
        LOG_ERR("[KVC] %s: CRC mismatch (expected=0x%08X got=0x%08X) — skipped\n",
            fname.c_str(), saved_crc, computed_crc);
        return false;
    }

    LOG_INF("[KVC] %s: arch=%.31s layers=%u type_k=%u yarn_ext=%.4f — OK\n",
        fname.c_str(), saved.model_arch, saved.n_layer,
        saved.type_k, saved.yarn_ext_factor);
    LOG_INF("[KVC] %s: CRC-32 verified (%.2f MiB)\n",
        fname.c_str(), (double) kv_buf.size() / (1024.0 * 1024.0));
    LOG_INF("[KVC] %s: restoring %u tokens to seq=%d (mtmd_chunks=%u)\n",
        fname.c_str(), saved_n_tokens, (int) seq_id, n_mtmd);

    // --- restore KV state
    const size_t nread = llama_state_seq_set_data_ext(ctx, kv_buf.data(), kv_buf.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (nread == 0) {
        LOG_ERR("[KVC] %s: llama_state_seq_set_data failed\n", fname.c_str());
        return false;
    }

    const int64_t elapsed = ggml_time_ms() - t0;
    LOG_INF("[KVC] %s: restored successfully in %" PRId64 " ms "
        "(tokens=%u, n_pos=%u, mtmd_chunks=%u)\n",
        fname.c_str(), elapsed, saved_n_tokens, saved_n_pos, n_mtmd);

    mtmd_positions_out = std::move(mtmd_positions);
    n_tokens_out       = (int32_t) saved_n_tokens;
    n_pos_out          = (llama_pos) saved_n_pos;
    tokens_out         = std::move(tok_array);

#undef KVC_SKIP
    return true;
}

bool kvc_disk_write_prompt(
        const std::string                    & path,
        const kvc_disk_fingerprint           & fp,
        const server_prompt                  & prompt) {
    int64_t t0 = ggml_time_ms();

    const size_t kv_main_size = prompt.data.main.size();
    const size_t kv_drft_size = prompt.data.drft.size();

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_WRN("[KVC] prompt_cache: cannot open '%s' for writing\n", path.c_str());
        return false;
    }

    auto write_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto write_u64 = [&](uint64_t v) { fwrite(&v, 8, 1, f); };
    auto write_f32 = [&](float  v)   { fwrite(&v, 4, 1, f); };

    // Write TKVD header with version 3
    write_u32(KVC_DISK_MAGIC);
    write_u32(KVC_DISK_VERSION_V3);

    // fingerprint
    fwrite(fp.model_arch, 1, sizeof(fp.model_arch), f);
    write_u64(fp.model_size);
    write_u32(fp.n_layer);
    write_u32(fp.n_head);
    write_u32(fp.n_head_kv);
    write_u32(fp.n_ctx_train);
    write_u32(fp.type_k);
    write_u32(fp.type_v);
    write_u32(fp.n_ctx_orig_yarn);
    write_f32(fp.yarn_ext_factor);
    write_f32(fp.yarn_attn_factor);
    write_f32(fp.yarn_beta_fast);
    write_f32(fp.yarn_beta_slow);
    write_f32(fp.rope_freq_base);
    write_f32(fp.rope_freq_scale);

    // token count + KV position count
    const llama_tokens tokens_all = prompt.tokens.get_tokens_all();
    write_u32((uint32_t) tokens_all.size());
    write_u32((uint32_t) prompt.tokens.pos_next());

    // MTMD chunk positions
    write_u32((uint32_t) prompt.mtmd_positions.size());
    for (const auto & cp : prompt.mtmd_positions) {
        write_u32((uint32_t) cp.seq_start);
        write_u32((uint32_t) cp.seq_len);
        write_u32((uint32_t) cp.pos_type);
        const uint32_t id_len = (uint32_t) cp.id.size();
        write_u32(id_len);
        if (id_len > 0) {
            fwrite(cp.id.data(), 1, id_len, f);
        }
    }

    // Main KV blob size + CRC
    write_u64(kv_main_size);
    uint32_t kv_main_crc = 0;
    if (kv_main_size > 0) {
        kv_main_crc = crc32_buf(prompt.data.main.data(), kv_main_size);
    }
    write_u32(kv_main_crc);

    // Draft KV blob size + CRC
    write_u64(kv_drft_size);
    uint32_t kv_drft_crc = 0;
    if (kv_drft_size > 0) {
        kv_drft_crc = crc32_buf(prompt.data.drft.data(), kv_drft_size);
    }
    write_u32(kv_drft_crc);

    // Write Main KV blob data
    if (kv_main_size > 0) {
        fwrite(prompt.data.main.data(), 1, kv_main_size, f);
    }

    // Write Draft KV blob data
    if (kv_drft_size > 0) {
        fwrite(prompt.data.drft.data(), 1, kv_drft_size, f);
    }

    // Write tokens array
    if (!tokens_all.empty()) {
        fwrite(tokens_all.data(), sizeof(llama_token), tokens_all.size(), f);
    }

    // Write speculative checkpoints
    write_u32((uint32_t) prompt.checkpoints.size());
    for (const auto & ckpt : prompt.checkpoints) {
        write_u64((uint64_t) ckpt.n_tokens);
        write_u32((uint32_t) ckpt.pos_min);
        write_u32((uint32_t) ckpt.pos_max);
        
        write_u64((uint64_t) ckpt.data_tgt.size());
        if (!ckpt.data_tgt.empty()) {
            fwrite(ckpt.data_tgt.data(), 1, ckpt.data_tgt.size(), f);
        }

        write_u64((uint64_t) ckpt.data_dft.size());
        if (!ckpt.data_dft.empty()) {
            fwrite(ckpt.data_dft.data(), 1, ckpt.data_dft.size(), f);
        }
    }

    fclose(f);

    const int64_t elapsed = ggml_time_ms() - t0;
    LOG_INF("[KVC] prompt_cache: saved prompt to %s in %" PRId64 " ms (tokens=%zu)\n",
        path.c_str(), elapsed, tokens_all.size());

    return true;
}

bool kvc_disk_read_prompt(
        const std::string              & path,
        const kvc_disk_fingerprint     & expected,
        server_prompt                  & prompt) {
    int64_t t0 = ggml_time_ms();

    std::filesystem::path fspath(path);
    std::string fname = fspath.filename().string();

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    auto read_u32 = [&]() -> uint32_t { uint32_t v = 0; fread(&v, 4, 1, f); return v; };
    auto read_u64 = [&]() -> uint64_t { uint64_t v = 0; fread(&v, 8, 1, f); return v; };
    auto read_f32 = [&]() -> float    { float  v = 0; fread(&v, 4, 1, f); return v; };

#define KVC_SKIP(fmt, ...) do { \
        LOG_WRN("[KVC] %s: " fmt " — skipped\n", fname.c_str(), ##__VA_ARGS__); \
        fclose(f); \
        return false; \
    } while (0)

    const uint32_t magic   = read_u32();
    const uint32_t version = read_u32();
    if (ferror(f) || feof(f)) {
        KVC_SKIP("unexpected EOF reading header");
    }
    if (magic != KVC_DISK_MAGIC) {
        KVC_SKIP("not a TKVD cache file (magic=0x%08X)", magic);
    }
    if (version != KVC_DISK_VERSION_V3) {
        KVC_SKIP("unsupported cache version %u (expected %u)", version, KVC_DISK_VERSION_V3);
    }

    kvc_disk_fingerprint saved = {};
    if (fread(saved.model_arch, 1, sizeof(saved.model_arch), f) != sizeof(saved.model_arch)) {
        KVC_SKIP("unexpected EOF reading arch string");
    }
    saved.model_size      = read_u64();
    saved.n_layer         = read_u32();
    saved.n_head          = read_u32();
    saved.n_head_kv       = read_u32();
    saved.n_ctx_train     = read_u32();
    saved.type_k          = read_u32();
    saved.type_v          = read_u32();
    saved.n_ctx_orig_yarn = read_u32();
    saved.yarn_ext_factor  = read_f32();
    saved.yarn_attn_factor = read_f32();
    saved.yarn_beta_fast   = read_f32();
    saved.yarn_beta_slow   = read_f32();
    saved.rope_freq_base   = read_f32();
    saved.rope_freq_scale  = read_f32();

    if (ferror(f) || feof(f)) {
        KVC_SKIP("unexpected EOF reading fingerprint");
    }

    // Validate fingerprint
    if (std::strncmp(saved.model_arch, expected.model_arch, 32) != 0) {
        KVC_SKIP("arch mismatch (saved='%.31s' loaded='%.31s')",
            saved.model_arch, expected.model_arch);
    }
    if (saved.model_size != expected.model_size) {
        KVC_SKIP("model file changed (size %" PRIu64 " → %" PRIu64 ")",
            saved.model_size, expected.model_size);
    }
    if (saved.n_layer   != expected.n_layer ||
        saved.n_head    != expected.n_head  ||
        saved.n_head_kv != expected.n_head_kv) {
        KVC_SKIP("incompatible model shape (layers %u/%u, heads %u/%u, kv-heads %u/%u)",
            saved.n_layer, expected.n_layer,
            saved.n_head, expected.n_head,
            saved.n_head_kv, expected.n_head_kv);
    }
    if (saved.type_k != expected.type_k || saved.type_v != expected.type_v) {
        KVC_SKIP("KV type mismatch (K:%u/%u V:%u/%u)",
            saved.type_k, expected.type_k, saved.type_v, expected.type_v);
    }
    if (saved.n_ctx_orig_yarn != expected.n_ctx_orig_yarn ||
        saved.yarn_ext_factor  != expected.yarn_ext_factor  ||
        saved.yarn_attn_factor != expected.yarn_attn_factor ||
        saved.yarn_beta_fast   != expected.yarn_beta_fast   ||
        saved.yarn_beta_slow   != expected.yarn_beta_slow   ||
        saved.rope_freq_base   != expected.rope_freq_base   ||
        saved.rope_freq_scale  != expected.rope_freq_scale) {
        KVC_SKIP("RoPE/YaRN params changed — stale cache");
    }

    const uint32_t saved_n_tokens = read_u32();
    const uint32_t saved_n_pos    = read_u32();
    if (ferror(f) || feof(f)) {
        KVC_SKIP("unexpected EOF reading token count");
    }

    // MTMD chunks
    const uint32_t n_mtmd = read_u32();
    if (ferror(f) || feof(f)) {
        KVC_SKIP("unexpected EOF reading mtmd chunk count");
    }
    std::vector<mtmd_chunk_pos> mtmd_positions(n_mtmd);
    for (uint32_t i = 0; i < n_mtmd; i++) {
        mtmd_positions[i].seq_start = (int32_t) read_u32();
        mtmd_positions[i].seq_len   = (int32_t) read_u32();
        mtmd_positions[i].pos_type  = (int)     read_u32();
        const uint32_t id_len = read_u32();
        if (ferror(f) || feof(f)) {
            KVC_SKIP("unexpected EOF reading mtmd chunk %u", i);
        }
        if (id_len > 0) {
            mtmd_positions[i].id.resize(id_len);
            if (fread(&mtmd_positions[i].id[0], 1, id_len, f) != id_len) {
                KVC_SKIP("unexpected EOF reading mtmd chunk %u id", i);
            }
        }
    }

    // KV blobs size + CRC
    const uint64_t kv_main_size = read_u64();
    const uint32_t saved_main_crc = read_u32();

    const uint64_t kv_drft_size = read_u64();
    const uint32_t saved_drft_crc = read_u32();

    if (ferror(f) || feof(f)) {
        KVC_SKIP("unexpected EOF reading KV blob header");
    }

    std::vector<uint8_t> main_buf(kv_main_size);
    if (kv_main_size > 0) {
        if (fread(main_buf.data(), 1, kv_main_size, f) != kv_main_size) {
            KVC_SKIP("unexpected EOF reading main KV blob (%zu bytes)", kv_main_size);
        }
    }

    std::vector<uint8_t> drft_buf(kv_drft_size);
    if (kv_drft_size > 0) {
        if (fread(drft_buf.data(), 1, kv_drft_size, f) != kv_drft_size) {
            KVC_SKIP("unexpected EOF reading draft KV blob (%zu bytes)", kv_drft_size);
        }
    }

    // Tokens
    llama_tokens tok_array;
    if (saved_n_tokens > 0) {
        tok_array.resize(saved_n_tokens);
        if (fread(tok_array.data(), sizeof(llama_token), saved_n_tokens, f) != saved_n_tokens) {
            KVC_SKIP("unexpected EOF reading token array (%u tokens)", saved_n_tokens);
        }
    }

    // Checkpoints
    uint32_t n_ckpts = read_u32();
    if (ferror(f)) {
        KVC_SKIP("unexpected EOF reading checkpoints count");
    }
    std::list<common_prompt_checkpoint> checkpoints;
    for (uint32_t i = 0; i < n_ckpts; i++) {
        common_prompt_checkpoint ckpt = {};
        ckpt.n_tokens = (int64_t) read_u64();
        ckpt.pos_min  = (llama_pos) read_u32();
        ckpt.pos_max  = (llama_pos) read_u32();

        const uint64_t tgt_size = read_u64();
        if (tgt_size > 0) {
            ckpt.data_tgt.resize(tgt_size);
            if (fread(ckpt.data_tgt.data(), 1, tgt_size, f) != tgt_size) {
                KVC_SKIP("unexpected EOF reading checkpoint %u tgt blob", i);
            }
        }

        const uint64_t dft_size = read_u64();
        if (dft_size > 0) {
            ckpt.data_dft.resize(dft_size);
            if (fread(ckpt.data_dft.data(), 1, dft_size, f) != dft_size) {
                KVC_SKIP("unexpected EOF reading checkpoint %u dft blob", i);
            }
        }
        checkpoints.push_back(std::move(ckpt));
    }

    fclose(f);
    f = nullptr;

    // Verify CRCs
    if (kv_main_size > 0) {
        const uint32_t computed_main_crc = crc32_buf(main_buf.data(), kv_main_size);
        if (computed_main_crc != saved_main_crc) {
            LOG_ERR("[KVC] %s: Main CRC mismatch (expected=0x%08X got=0x%08X) — skipped\n",
                fname.c_str(), saved_main_crc, computed_main_crc);
            return false;
        }
    }
    if (kv_drft_size > 0) {
        const uint32_t computed_drft_crc = crc32_buf(drft_buf.data(), kv_drft_size);
        if (computed_drft_crc != saved_drft_crc) {
            LOG_ERR("[KVC] %s: Draft CRC mismatch (expected=0x%08X got=0x%08X) — skipped\n",
                fname.c_str(), saved_drft_crc, computed_drft_crc);
            return false;
        }
    }

    // Reconstruct prompt.tokens
    prompt.tokens.clear();
    prompt.tokens.has_mtmd = n_mtmd > 0;
    prompt.mtmd_positions = std::move(mtmd_positions);

    size_t mtmd_idx = 0;
    for (size_t i = 0; i < tok_array.size(); ) {
        if (tok_array[i] == LLAMA_TOKEN_NULL) {
            size_t j = i;
            while (j < tok_array.size() && tok_array[j] == LLAMA_TOKEN_NULL) j++;
            const uint32_t n_null = (uint32_t)(j - i);

            if (mtmd_idx < prompt.mtmd_positions.size()) {
                const auto & cp = prompt.mtmd_positions[mtmd_idx++];
                auto * sentinel = mtmd_input_chunk_create_sentinel(
                    cp.id.c_str(), n_null, (llama_pos) cp.seq_len, cp.pos_type);
                prompt.tokens.push_back(sentinel);
                mtmd_input_chunk_free(sentinel);
            } else {
                LOG_WRN("[KVC] %s: NULL run at offset %zu but no mtmd_pos entry\n",
                    fname.c_str(), i);
                prompt.tokens.clear();
                return false;
            }
            i = j;
        } else {
            prompt.tokens.push_back(tok_array[i++]);
        }
    }

    prompt.data.main = std::move(main_buf);
    prompt.data.drft = std::move(drft_buf);
    prompt.checkpoints = std::move(checkpoints);

    const int64_t elapsed = ggml_time_ms() - t0;
    LOG_INF("[KVC] prompt_cache: loaded prompt from %s in %" PRId64 " ms (tokens=%zu)\n",
        fname.c_str(), elapsed, prompt.tokens.size());

#undef KVC_SKIP
    return true;
}

