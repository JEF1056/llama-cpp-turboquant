#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP);
                              // llama_dspark_meta / llama_model_dspark_get_meta / llama_model_dspark_get_markov (used by dspark)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <cinttypes>

#ifdef LLAMA_DSPARK_MARKOV_BLAS
#include <cblas.h>
#endif

#ifdef LLAMA_DSPARK_MARKOV_CUDA
#include "dspark-markov.h"
#endif

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    // draft-dspark requires the driver to engage multi-layer capture on the
    // target context (llama_set_capture_layers with the drafter's target layer
    // ids, plus logits requested on every row) before drafting -- see
    // need_embd_capture()/process(). The reference driver is
    // tests/test-dspark-real-eval.cpp; tools/server engages capture too (via
    // llama_model_target_layer_ids), but the generic CLI (--spec-type) does
    // not yet, so selecting draft-dspark there fails at the first draft round
    // with a clear error rather than running.
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    // reset per-seq state before a prompt is (re)processed, keeping the first
    // n_keep context rows already resident in the draft cache (0 for a full
    // reprocess). Default no-op; only dspark needs it, since its capture rows
    // are staged during prefill and must not be wiped at generation start.
    virtual void reset(llama_seq_id /*seq_id*/, llama_pos /*n_keep*/) {}

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }

    // true if this implementation requires the target's multi-layer tap capture
    // (see llama_set_capture_layers / llama_get_embeddings_capture_ith)
    virtual bool need_embd_capture() const { return false; }

    // TEST/DEBUG ONLY hook: lets a test harness inject target-tap context rows
    // directly (see common_speculative_dspark_stage_ctx_test in speculative.h),
    // bypassing the normal process()-driven capture path. No-op for every
    // implementation except dspark.
    virtual bool stage_test_ctx_feat(
            llama_seq_id /*seq_id*/,
            const float * /*feat*/,
            int64_t /*n_rows*/,
            int64_t /*n_embd_cap*/,
            const int32_t * /*pos*/) {
        return false;
    }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        __func__, rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        __func__, rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    bool is_mem_shared = false;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                }
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// dspark: EAGLE-style block-diffusion drafter (Phase 2 of the dspark port --
// see docs/dspark-scope.md). The forward graph itself
// (src/models/dspark.cpp) is Phase 1 and is not touched here; this class only
// drives the repeated draft/verify loop around it.
//
// Shape of one round, mirroring the Python reference implementation 1:1:
//   - L      = n_cache[seq]  : rows currently resident in ctx_dft's persistent
//                              KV cache for this seq (DynamicCache.get_seq_length()).
//   - start  = dp.n_past     : absolute position of the last committed token
//                              (dp.id_last) -- the reference's "start".
//   - ctx_len = start - L    : number of NEW target-tap context rows to feed
//                              this round (== previous round's n_accepted+1,
//                              or the whole prompt for the very first round).
//   - one llama_decode(ctx_dft) call over n_tokens = ctx_len + block_size:
//     ctx_len dummy-token rows (their real content comes from
//     llama_set_dspark_ctx, not batch.token) followed by the block_size draft
//     rows (block position 0 seeded with the REAL anchor token dp.id_last,
//     positions 1..block_size-1 seeded with mask_token_id).
//   - crop ctx_dft's cache back to `start` (DynamicCache.crop(start)):
//     the draft block's rows are always thrown away immediately regardless of
//     what the target ultimately accepts -- only accept() decides what
//     becomes real context for the NEXT round.
//   - sequential, host-side Markov resample over the block_size base logits
//     (see the class-level comment on the resample loop below).
//
// Context-row bookkeeping: llama_dspark_ctx (src/llama-graph.h) is a single
// staging slot on llama_context, not per-sequence, so -- like the Python reference
// reference itself (build_dspark_proposal asserts batch_size==1) -- this impl
// drafts sequences one at a time within draft(), not batched together in one
// llama_decode call. Fine for the single-stream case this Phase 2 gate
// targets; batching multiple concurrently-drafting seqs into one dspark call
// would need a per-seq-capable staging mechanism, out of scope here.
struct common_speculative_impl_draft_dspark : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    int64_t n_embd        = 0;
    int64_t n_vocab       = 0; // from token_embd's own shape; dspark has no tokenizer/vocab of its own
    int64_t n_capture     = 0; // target_layer_ids count
    int64_t n_embd_cap    = 0; // n_capture * n_embd (raw pre-fc tap width)
    int32_t block_size    = 0;
    int32_t mask_token_id = 0;

    // vanilla Markov head weights, host-resident (loaded once at construction
    // via llama_model_dspark_get_markov): [n_vocab * n_rank] row-major, rank
    // fastest-varying. See the resample loop in draft() for how these are used.
    std::vector<float> markov_w1;
    std::vector<float> markov_w2;
    std::vector<float> markov_bias;
    int64_t markov_rank = 0;
    bool    has_markov  = false;
    // in-graph markov resample fused into the drafter decode (src/models/dspark.cpp).
    // when on, the block draft ids come back via llama_get_dspark_draft_ids and the
    // drafter needs no host logits, so the block requests no output (skips the
    // per-round block_size*n_vocab logits D2H). Mirrors the graph-side env gate.
    bool    markov_ingraph = false;
#ifdef LLAMA_DSPARK_MARKOV_CUDA
    bool                        markov_use_cuda = false; // device-side resample (default when built with CUDA; LLAMA_DSPARK_MARKOV_CUDA=0 to disable)
    struct dspark_markov_cuda * markov_cuda     = nullptr;
#endif

    llama_batch batch; // ctx_dft batch; no embd channel -- context features are
                        // staged out-of-band via llama_set_dspark_ctx, not batch.embd

    // --- per-seq persistent state --------------------------------------
    // drafter KV-cache length ("L" above / DynamicCache.get_seq_length()):
    // number of context rows currently resident in ctx_dft's cache.
    std::vector<int64_t> n_cache;

    // growing buffer of not-yet-consumed target-tap context rows, accumulated
    // across process() calls since the last draft() call drained them. Rows
    // are contiguous and strictly increasing in position (asserted in draft()).
    std::vector<std::vector<float>>   ctx_feat; // [n_seq][rows * n_embd_cap]
    std::vector<std::vector<int32_t>> ctx_pos;  // [n_seq][rows]

    // how many of the currently-buffered rows were appended since the last
    // accept() call. accept() trims exactly this many down to n_accepted+1,
    // discarding the rejected tail, leaving any earlier
    // (already-accepted-but-not-yet-drained) rows untouched. This is what
    // lets dspark's context stay correct even on rounds where a DIFFERENT
    // implementation's draft is the one that gets verified: process() runs
    // (and accumulates) unconditionally for every registered impl, and
    // accept() runs on every impl too (is_other=true for the ones that didn't
    // draft), so dspark's own bookkeeping tracks the real generation stream
    // regardless of who proposed a given round's tokens.
    std::vector<int64_t> rows_since_accept;

    // true from reset() until the first draft()/accept() of a generation, i.e.
    // while a prompt is being prefilled. During that window every buffered row
    // is a committed prompt token, so maybe_flush_prefill_ctx() can safely drain
    // the front of ctx_feat into the drafter cache without racing the rollback
    // window accept() may trim (that window only exists once generation starts).
    std::vector<bool> prefilling;

    // process()'s per-seq contiguous-range bookkeeping (mirrors draft-mtp).
    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    common_speculative_impl_draft_dspark(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;
        GGML_ASSERT(ctx_dft && ctx_tgt && "dspark requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);

        llama_dspark_meta meta;
        if (!llama_model_dspark_get_meta(model_dft, &meta)) {
            throw std::runtime_error("dspark: ctx_dft's model does not look like a dspark drafter (missing dspark.*.block_size KV)");
        }

        n_embd        = meta.n_embd;
        n_vocab       = meta.n_vocab;
        n_capture     = meta.n_capture;
        n_embd_cap    = meta.n_embd_cap;
        block_size    = meta.block_size;
        mask_token_id = meta.mask_token_id;
        markov_rank   = meta.markov_rank;

        // Contract with the target model. The drafter consumes the target's
        // hidden states directly -- each captured layer row is exactly
        // target_hidden wide, and n_embd_cap == n_capture * n_embd is copied
        // verbatim out of llama_get_embeddings_capture_ith() in stage_ctx_feat()
        // -- and it resamples/argmaxes over the target's vocabulary. A drafter
        // trained against a differently-sized target would silently over-read the
        // capture rows or index the wrong vocab. Validate both here so an
        // incompatible pairing fails loudly at construction instead of corrupting
        // every round (mirrors draft-mtp's n_embd assert).
        {
            const llama_model * model_tgt = llama_get_model(ctx_tgt);
            const int64_t n_embd_tgt  = llama_model_n_embd(model_tgt);
            const int64_t n_vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            if (n_embd != n_embd_tgt) {
                LOG_ERR("%s: drafter tap width n_embd=%lld != target hidden size %lld\n",
                        __func__, (long long) n_embd, (long long) n_embd_tgt);
                throw std::runtime_error("dspark: drafter/target hidden-size mismatch "
                        "(the drafter was trained against a different target model)");
            }
            if (n_vocab != n_vocab_tgt) {
                LOG_ERR("%s: drafter vocab=%lld != target vocab=%lld\n",
                        __func__, (long long) n_vocab, (long long) n_vocab_tgt);
                throw std::runtime_error("dspark: drafter/target vocabulary mismatch "
                        "(the drafter must share the target's tokenizer)");
            }
        }

        has_markov = markov_rank > 0 && llama_model_dspark_get_markov(model_dft, markov_w1, markov_w2);
        if (n_vocab > std::numeric_limits<int>::max()) {
            throw std::runtime_error("dspark: vocab size exceeds cblas integer range");
        }
        markov_bias.resize((size_t) n_vocab);

        // in-graph resample default-on when the drafter carries a vanilla markov
        // head; disable with LLAMA_DSPARK_MARKOV_INGRAPH=0 (matches the graph gate
        // in src/models/dspark.cpp so the block-output decision stays consistent).
        markov_ingraph = has_markov;
        if (const char * e = getenv("LLAMA_DSPARK_MARKOV_INGRAPH")) {
            const char c   = e[0];
            markov_ingraph = markov_ingraph && !(c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F');
        }

#ifdef LLAMA_DSPARK_MARKOV_CUDA
        // Device path is the DEFAULT when built with CUDA and a real markov head:
        // upload the Markov factors once and run the whole sequential resample on
        // the GPU (functionally equivalent to the host scalar/BLAS path; not
        // bit-identical -- the warp reduction's accumulation order differs, see
        // common/dspark-markov.h). Opt out with LLAMA_DSPARK_MARKOV_CUDA=0 to
        // fall back to the host path.
        bool want_cuda_markov = has_markov;
        if (const char * e = getenv("LLAMA_DSPARK_MARKOV_CUDA")) {
            const char c     = e[0];
            want_cuda_markov = want_cuda_markov && !(c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F');
        }
        if (want_cuda_markov) {
            markov_cuda     = dspark_markov_cuda_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank);
            markov_use_cuda = markov_cuda != nullptr;
            LOG_INF("%s: - device markov resample (CUDA) %s\n", __func__,
                    markov_use_cuda ? "ENABLED (default; LLAMA_DSPARK_MARKOV_CUDA=0 to disable)" :
                                      "FAILED TO INIT (falling back to host path)");
        }
#endif

        LOG_INF("%s: adding speculative implementation 'draft-dspark'\n", __func__);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_capture=%lld, n_embd=%lld, n_vocab=%lld, markov_rank=%lld, has_markov=%d\n",
                __func__, block_size, mask_token_id, (long long) n_capture, (long long) n_embd, (long long) n_vocab,
                (long long) markov_rank, (int) has_markov);
        if (markov_rank > 0 && !has_markov) {
            LOG_WRN("%s: dspark model reports markov_rank=%lld but its markov head weights could not be read "
                    "(gated/rnn markov head type? only 'vanilla' is supported) -- "
                    "block logits will NOT be markov-corrected\n", __func__, (long long) markov_rank);
        }

        // dspark attention is fully non-causal within a call: the draft block
        // attends over the WHOLE persistent cache plus itself, with no
        // position-based masking (attention_mask=None, is_causal=False in the
        // reference) -- see src/models/dspark.cpp's header comment.
        llama_set_causal_attn(ctx_dft, false);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/* n_tokens = */ n_b, /* embd = */ 0, /* n_seq_max = */ 1);

        n_cache.assign(n_seq, 0);
        ctx_feat.assign(n_seq, {});
        ctx_pos.assign(n_seq, {});
        rows_since_accept.assign(n_seq, 0);
        prefilling.assign(n_seq, false);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);
    }

    ~common_speculative_impl_draft_dspark() override {
        llama_batch_free(batch);
#ifdef LLAMA_DSPARK_MARKOV_CUDA
        if (markov_cuda != nullptr) {
            dspark_markov_cuda_free(markov_cuda);
            markov_cuda = nullptr;
        }
#endif
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        // per-seq state is reset in reset() BEFORE the prompt is prefilled, so the
        // capture rows staged by process() during prefill survive into the first
        // draft (round 1's ctx_len covers the whole new prompt). Clearing here
        // would instead wipe that just-staged context, so this is a no-op.
        (void) seq_id;
    }

    void reset(llama_seq_id seq_id, llama_pos n_keep) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // drop stale per-seq staging left by a prior generation that reused this
        // slot, keeping the n_keep context rows the driver already has resident in
        // ctx_dft (the reused prompt-cache prefix; 0 for a full reprocess). Called
        // before the prompt is decoded, so process() then stages positions
        // [n_keep, N) cleanly and round 1 sees ctx_len == staged rows.
        n_cache[seq_id] = n_keep;
        ctx_feat[seq_id].clear();
        ctx_pos[seq_id].clear();
        rows_since_accept[seq_id] = 0;
        prefilling[seq_id] = true;

        llama_memory_seq_rm(llama_get_memory(params.ctx_dft), seq_id, n_keep, -1);
    }

    // Append `chunk` context rows (row-major tap features at feat_ptr, absolute
    // positions [cbeg, cbeg+chunk)) to the drafter KV cache, then crop away the
    // throwaway draft block the graph requires. Mirrors draft()'s per-chunk
    // context write; returns false if the decode or the tail crop fails.
    bool write_ctx_chunk(llama_seq_id seq_id, const float * feat_ptr, int64_t chunk, int64_t cbeg) {
        auto * ctx_dft = params.ctx_dft;
        const int64_t cend = cbeg + chunk;

        llama_set_dspark_ctx(ctx_dft, feat_ptr, chunk, n_embd_cap);

        common_batch_clear(batch);
        for (int64_t i = 0; i < chunk; ++i) {
            common_batch_add(batch, /* token = */ 0, (llama_pos)(cbeg + i), { seq_id }, /* logits = */ false);
        }
        // throwaway block: the graph needs >= 1 draft row, but it is cropped
        // right after and context rows never query it, so token ids and logits
        // are irrelevant.
        common_batch_add(batch, /* token = */ 0, (llama_pos) cend, { seq_id }, /* logits = */ false);
        for (int32_t k = 1; k < block_size; ++k) {
            common_batch_add(batch, mask_token_id, (llama_pos)(cend + k), { seq_id }, /* logits = */ false);
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        llama_set_dspark_ctx(ctx_dft, nullptr, 0, 0);
        if (rc != 0) {
            LOG_WRN("%s: llama_decode(ctx_dft) failed rc=%d for seq %d at ctx pos %lld\n",
                    __func__, rc, (int) seq_id, (long long) cbeg);
            return false;
        }
        return llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, (llama_pos) cend, -1);
    }

    // Bound host memory during a long prompt prefill: incrementally write
    // already-committed context rows into the drafter KV cache instead of
    // letting process() pile the whole prompt's tap features up in ctx_feat
    // (n_embd_cap floats per row) until the first draft() drains them -- that
    // buffer alone is ~n_embd_cap*4 bytes per prompt token and OOM-kills the
    // server well before a 100k+ context. The rows written here are identical
    // to what draft()'s non-final chunks write later (context rows issue no
    // query; their K/V is a pure function of their own tap features), so
    // draining them early is equivalent to one big prefill. Guarded by
    // prefilling[]: only runs before the first draft()/accept(), where every
    // buffered row is a committed prompt token and none can be rolled back.
    void maybe_flush_prefill_ctx(llama_seq_id seq_id) {
        if (!prefilling[seq_id]) {
            return;
        }

        auto & feat = ctx_feat[seq_id];
        auto & pos  = ctx_pos[seq_id];

        auto * ctx_dft = params.ctx_dft;
        const int64_t ctx_per_chunk = (int64_t) llama_n_ubatch(ctx_dft) - block_size;
        if (ctx_per_chunk < 1) {
            return; // draft() reports this fatal config; nothing to drain here
        }

        // keep the host buffer within ~2 chunks; leave a one-chunk tail so the
        // remaining rows still form a valid in-order range for the first draft().
        const int64_t trigger = 2 * ctx_per_chunk;
        if ((int64_t) pos.size() <= trigger) {
            return;
        }
        int64_t n_drain = (int64_t) pos.size() - ctx_per_chunk;
        n_drain -= n_drain % ctx_per_chunk; // whole chunks only
        if (n_drain <= 0) {
            return;
        }

        // throwaway blocks are cropped and never queried by context rows, so the
        // anchor is irrelevant; pin it to 0 for determinism.
        llama_set_dspark_anchor(ctx_dft, 0);
        for (int64_t off = 0; off < n_drain; off += ctx_per_chunk) {
            const int64_t chunk = std::min<int64_t>(ctx_per_chunk, n_drain - off);
            if (!write_ctx_chunk(seq_id, feat.data() + (size_t) off * (size_t) n_embd_cap,
                        chunk, n_cache[seq_id])) {
                // mirror draft()'s deterministic recovery: wipe the drafter
                // sequence and staged rows so generation still proceeds (dspark
                // just under-drafts this prompt) instead of desyncing the cache.
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);
                n_cache[seq_id] = 0;
                feat.clear();
                pos.clear();
                rows_since_accept[seq_id] = 0;
                prefilling[seq_id] = false;
                return;
            }
            n_cache[seq_id] += chunk;
        }

        // drop the drained front rows; the remaining tail keeps pos.front() ==
        // n_cache and pos.back() unchanged, exactly what draft() asserts.
        feat.erase(feat.begin(), feat.begin() + (size_t) n_drain * (size_t) n_embd_cap);
        pos.erase(pos.begin(), pos.begin() + (size_t) n_drain);
        rows_since_accept[seq_id] = std::max<int64_t>(0, rows_since_accept[seq_id] - n_drain);
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens? (mirrors draft-mtp)
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = params.ctx_tgt;

        // The row copy below reads n_embd_cap floats from each target capture
        // row, whose real width is n_capture_configured * n_embd. The ctor
        // validated n_embd/n_vocab against the target model, but capture layers
        // are engaged by the DRIVER after construction (llama_set_capture_layers),
        // so the configured layer count can only be checked here: a driver that
        // engaged fewer layers than the drafter was trained on would otherwise
        // over-read past the end of the capture row (and more layers would feed
        // misaligned features).
        {
            const uint32_t n_cap_cfg = llama_get_n_capture(ctx_tgt);
            if ((int64_t) n_cap_cfg != n_capture) {
                LOG_ERR("%s: target context has %u capture layers configured but the drafter "
                        "expects %lld -- the driver must pass the drafter's target layer list "
                        "to llama_set_capture_layers()\n",
                        __func__, n_cap_cfg, (long long) n_capture);
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t row0 = pos.size();
            feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
            pos.resize(row0 + (size_t) n_rows);

            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t k = i_batch_beg[seq_id] + i;

                // NOTE: capture rows always use the masked (output-row) layout
                // (see src/llama-context.cpp's get_embeddings_capture_ith) --
                // this requires the caller to have requested logits/output on
                // EVERY row it wants a capture row for (unlike the pre-norm
                // path MTP uses, which can force unmasked extraction). For a
                // long prompt this means every prefill row, not just the
                // last -- a caller-side (main/server driver loop) requirement
                // when a registered impl reports need_embd_capture(), exactly
                // analogous to draft-mtp's own begin()-time warning about
                // need_embd_nextn.
                const float * cap = llama_get_embeddings_capture_ith(ctx_tgt, k);
                if (cap == nullptr) {
                    LOG_ERR("%s: llama_get_embeddings_capture_ith(%d) returned null -- was "
                            "llama_set_capture_layers() engaged and logits requested for every "
                            "row this impl needs?\n", __func__, k);
                    return false;
                }

                std::memcpy(feat.data() + (row0 + (size_t) i) * (size_t) n_embd_cap, cap,
                        (size_t) n_embd_cap * sizeof(float));
                pos[row0 + i] = batch_in.pos[k];
            }

            rows_since_accept[seq_id] += n_rows;

            // drain committed prompt rows into the drafter cache as they arrive,
            // so the host ctx_feat buffer stays bounded on long contexts.
            maybe_flush_prefill_ctx(seq_id);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto * ctx_dft = params.ctx_dft;
        const int64_t n_ubatch_max = (int64_t) llama_n_ubatch(ctx_dft);

        // opt-in draft-internal attribution (LLAMA_DSPARK_PROFILE): splits the
        // per-draft cost into the ctx_dft decode (feature H2D + forward + fused
        // resample) vs the on-device draft-id drain sync, and tracks the ctx_len
        // shape spread (constant shape => the per-key CUDA graph cache reuses;
        // wide spread => padding to a constant n_tokens could help).
        static const bool prof = getenv("LLAMA_DSPARK_PROFILE") != nullptr;
        static int64_t prof_decode_us = 0, prof_drain_us = 0, prof_n = 0;
        static int64_t prof_ctxlen_sum = 0, prof_ctxlen_min = INT64_MAX, prof_ctxlen_max = 0;
        // per-decode time bucketed by batch n_tokens: reveals whether the ctx_dft
        // decode is overhead-bound (flat vs n_tokens => padding to a constant shape
        // for CUDA-graph reuse would help) or compute-bound (scales with n_tokens).
        static const int PROF_HIST_CAP = 64;
        static int64_t prof_hist_us[PROF_HIST_CAP] = {0};
        static int64_t prof_hist_n [PROF_HIST_CAP] = {0};

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            // generation has started for this seq: prompt prefill is over, so no
            // more prefill-time draining (its committed-rows-only invariant no
            // longer holds once accept() can roll back the verify tail).
            prefilling[seq_id] = false;

            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const int64_t L       = n_cache[seq_id];
            const int64_t start   = dp.n_past;
            const int64_t ctx_len = start - L;

            if (ctx_len <= 0) {
                LOG_WRN("%s: seq %d has no new context rows staged (n_past=%lld, cache=%lld) -- "
                        "skipping this round\n", __func__, (int) seq_id, (long long) start, (long long) L);
                continue;
            }
            if ((int64_t) pos.size() != ctx_len) {
                LOG_ERR("%s: seq %d staged context rows (%zu) != expected ctx_len (%lld) -- "
                        "n_past bookkeeping is out of sync with process()/accept(); "
                        "aborting draft for this seq this round\n",
                        __func__, (int) seq_id, pos.size(), (long long) ctx_len);
                continue;
            }
            GGML_ASSERT(pos.front() == (int32_t) L        && "dspark: staged rows do not start at the drafter's cache position");
            GGML_ASSERT(pos.back()  == (int32_t) start - 1 && "dspark: staged rows do not end just before the anchor position");

            // The chunk batch below writes context rows starting at position L,
            // which the batch allocator requires to be exactly (drafter cache max
            // pos for this seq) + 1. Prompt-cache reuse breaks that invariant: the
            // target can restore a cached prefix without re-decoding it, so the
            // tokens generated at the tail of the previous task were never fed
            // back through the target and process() never captured their tap
            // features -- the drafter cache stays behind n_cache with an
            // unfillable gap in between. When the cache and our bookkeeping
            // disagree, drop the stale drafter sequence so this round rebuilds
            // context from the freshly staged rows against an empty cache (the
            // allocator skips the continuity check for an empty sequence, and the
            // cache regrows as generation proceeds).
            const llama_pos cache_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
            if (cache_pos_max + 1 != (llama_pos) L) {
                LOG_WRN("%s: seq %d drafter cache ends at %d but staged context starts at %lld -- "
                        "dropping the stale drafter sequence to resync (prompt-reuse gap)\n",
                        __func__, (int) seq_id, (int) cache_pos_max, (long long) L);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);
            }

            // dspark conditions the draft block on the whole staged context in a
            // single non-causal decode, which llama.cpp bounds by n_ubatch. When
            // the context is larger than one ubatch, prefill it into the drafter
            // KV cache in n_ubatch-sized chunks: context rows issue no query and
            // their K/V is a pure function of their own tap features, so writing
            // them across several decodes is equivalent to one -- the same
            // growing-cache mechanism the multi-round loop already relies on to
            // attend to earlier committed context. Only the final chunk carries
            // the real draft block; earlier chunks append a throwaway block (the
            // graph requires >= 1 draft row) that is cropped away immediately.
            const int64_t ctx_per_chunk = n_ubatch_max - block_size;
            if (ctx_per_chunk < 1) {
                LOG_ERR("%s: seq %d n_ubatch=%lld cannot hold a draft block of %d plus context -- skipping\n",
                        __func__, (int) seq_id, (long long) n_ubatch_max, (int) block_size);
                continue;
            }

            bool round_failed = false;
            // stage the block anchor (dp.id_last -- the "prev" token for resample
            // position 0) for the in-graph markov resample. Constant across this
            // round's chunks, and ignored by graphs that don't fuse the resample.
            llama_set_dspark_anchor(ctx_dft, (int32_t) dp.id_last);
            for (int64_t off = 0; off < ctx_len; off += ctx_per_chunk) {
                const int64_t chunk = std::min<int64_t>(ctx_per_chunk, ctx_len - off);
                const bool    last  = (off + chunk == ctx_len);
                const int64_t cbeg  = L + off;      // first context position in this chunk
                const int64_t cend  = cbeg + chunk; // the block sits here; == start on the final chunk

                // stage only this chunk's tap features (feat is row-major
                // [ctx_len, n_embd_cap], ordered by position starting at L).
                llama_set_dspark_ctx(ctx_dft, feat.data() + (size_t) off * (size_t) n_embd_cap,
                        chunk, n_embd_cap);

                common_batch_clear(batch);
                for (int64_t i = 0; i < chunk; ++i) {
                    // dummy token id: this row's real content comes from the
                    // staged dspark ctx feature above, not the token embedding
                    // (see src/models/dspark.cpp -- these columns are sliced away
                    // before the residual stream even forms). logits=false: this
                    // impl never reads output for context rows.
                    common_batch_add(batch, /* token = */ 0, (llama_pos)(cbeg + i), { seq_id }, /* logits = */ false);
                }
                // block position 0 is seeded with the REAL last-accepted token
                // (the "anchor"), NOT mask_token_id -- matches the Python reference
                // reference's evaluator._propose (draft_input_ids[:,0] =
                // output_ids[:,start]). Positions 1..block_size-1 are masked. Only
                // the final chunk's block attends to the whole context, so only it
                // requests logits; earlier throwaway blocks do not. With the
                // in-graph resample the draft ids come back on-device, so the block
                // requests NO host logits at all (skips the block_size*n_vocab D2H);
                // the host/CUDA fallback still needs them, hence the guard.
                const bool block_logits = last && !markov_ingraph;
                common_batch_add(batch, dp.id_last, (llama_pos) cend, { seq_id }, /* logits = */ block_logits);
                for (int32_t k = 1; k < block_size; ++k) {
                    common_batch_add(batch, mask_token_id, (llama_pos)(cend + k), { seq_id }, /* logits = */ block_logits);
                }

                const int64_t prof_t_dec = prof ? ggml_time_us() : 0;
                const int32_t rc = llama_decode(ctx_dft, batch);
                if (prof) {
                    const int64_t dec_us = ggml_time_us() - prof_t_dec;
                    prof_decode_us += dec_us;
                    const int b = batch.n_tokens < PROF_HIST_CAP ? batch.n_tokens : PROF_HIST_CAP - 1;
                    prof_hist_us[b] += dec_us;
                    prof_hist_n [b] += 1;
                }

                // always clear the staged ctx immediately after use, success or not.
                llama_set_dspark_ctx(ctx_dft, nullptr, 0, 0);

                if (rc != 0) {
                    LOG_WRN("%s: llama_decode(ctx_dft) failed rc=%d for seq %d (context offset %lld)\n",
                            __func__, rc, (int) seq_id, (long long) off);
                    round_failed = true;
                    break;
                }

                // crop away the just-written draft block, keeping the committed
                // context rows [.., cend). On the final chunk cend == start, so
                // this mirrors past_key_values_draft.crop(start) in the Python
                // reference; on earlier chunks it drops the throwaway block before
                // the next chunk appends more context.
                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, (llama_pos) cend, -1)) {
                    // Could not crop just the speculative tail (e.g. the backend
                    // rejected the partial removal): the physical drafter cache still
                    // contains the draft rows, so advancing n_cache would desync
                    // bookkeeping from the cache and corrupt every later round.
                    // Recover deterministically by wiping the whole drafter sequence
                    // and resetting bookkeeping so the next round rebuilds its context
                    // from scratch (a full-sequence removal always succeeds).
                    LOG_ERR("%s: failed to crop drafter cache tail for seq %d at %lld -- "
                            "resetting the drafter sequence to recover\n",
                            __func__, (int) seq_id, (long long) cend);
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1);
                    n_cache[seq_id] = 0;
                    feat.clear();
                    pos.clear();
                    rows_since_accept[seq_id] = 0;
                    round_failed = true;
                    break;
                }
            }
            if (round_failed) {
                continue;
            }
            n_cache[seq_id] = start;

            feat.clear();
            pos.clear();
            rows_since_accept[seq_id] = 0; // this round's rows were just consumed

            if (prof) {
                prof_ctxlen_sum += ctx_len;
                if (ctx_len < prof_ctxlen_min) { prof_ctxlen_min = ctx_len; }
                if (ctx_len > prof_ctxlen_max) { prof_ctxlen_max = ctx_len; }
                if (++prof_n % 64 == 0) {
                    LOG_INF("[dspark-prof/draft] per draft avg: decode=%.2fms drain_sync=%.2fms | ctx_len avg=%.2f min=%lld max=%lld (n=%lld)\n",
                            (double) prof_decode_us / (double) prof_n / 1000.0,
                            (double) prof_drain_us  / (double) prof_n / 1000.0,
                            (double) prof_ctxlen_sum / (double) prof_n,
                            (long long) prof_ctxlen_min, (long long) prof_ctxlen_max, (long long) prof_n);

                    std::string hist;
                    char buf[64];
                    for (int b = 0; b < PROF_HIST_CAP; ++b) {
                        if (prof_hist_n[b] == 0) { continue; }
                        snprintf(buf, sizeof(buf), " n_tok=%d:%.2fms(x%lld)", b,
                                (double) prof_hist_us[b] / (double) prof_hist_n[b] / 1000.0,
                                (long long) prof_hist_n[b]);
                        hist += buf;
                    }
                    LOG_INF("[dspark-prof/draft] decode by batch n_tokens:%s\n", hist.c_str());
                }
            }

            // --- in-graph markov resample fast path -----------------------
            // When the drafter fused the whole block_size sequential resample
            // into its decode graph (vanilla markov head + the default
            // LLAMA_DSPARK_MARKOV_INGRAPH), the block_size argmax draft ids are
            // already computed on-device -- no logits device->host->device
            // round-trip and no extra host/CUDA resample. Consume them directly;
            // a null / short return means the graph produced none (target ctx,
            // no markov head, or the env override), so fall through to the
            // host/CUDA resample below.
            {
                int32_t n_draft_ids = 0;
                const int64_t prof_t_drn = prof ? ggml_time_us() : 0;
                const int32_t * draft_ids = llama_get_dspark_draft_ids(ctx_dft, &n_draft_ids);
                if (prof) { prof_drain_us += ggml_time_us() - prof_t_drn; }
                if (draft_ids != nullptr && n_draft_ids >= block_size) {
                    // A resample can legitimately emit mask_token_id (a real vocab
                    // id) at a chained position; that only yields a poor draft the
                    // target rejects, so warn once instead of aborting.
                    static bool warned_ingraph_mask = false;
                    for (int32_t k = 1; k < block_size && !warned_ingraph_mask; ++k) {
                        if (draft_ids[k - 1] == mask_token_id) {
                            LOG_WRN("%s: dspark in-graph markov resample produced mask_token_id at a "
                                    "chained draft position -- drafter emitted the mask sentinel; "
                                    "the target verify will reject it\n", __func__);
                            warned_ingraph_mask = true;
                        }
                    }

                    llama_tokens result(draft_ids, draft_ids + block_size);
                    if (result.size() < (size_t) params.n_min) {
                        continue; // dp.result stays empty: treated as a failed draft
                    }
                    *dp.result = std::move(result);
                    continue;
                }
            }

            // --- sequential Markov resample -------------------------------
            // step_logits[k] = base_logits[k] + markov_w2(markov_w1(prev_token)),
            // where prev_token is the block's own anchor token for k==0 and the
            // ACTUALLY SAMPLED token from step k-1 for k>0. This must never be
            // batched over mask_token_id for all block positions at once --
            // that exact bug class already hit the on-device (MLX/Swift) port.
            // The assert below makes the sequential dependency structural
            // rather than just a comment: it is unsatisfiable if this loop is
            // ever refactored to precompute prev_token_ids up front from
            // draft_input_ids instead of chaining the sampled result forward.
            llama_tokens result;
            result.reserve(block_size);

            // dense output buffer: only the block_size draft rows requested
            // logits this call, so llama_get_logits() is already exactly
            // block_size*n_vocab floats in row order -- no per-row index
            // resolution needed (mirrors tests/test-dspark-forward.cpp's
            // llama_get_logits(ctx) usage). llama_get_logits_ith(ctx, i)
            // would need i to be the RAW ubatch row (ctx_len + k here), since
            // it resolves through output_resolve_row() same as the
            // pre-norm/capture accessors -- the bulk buffer sidesteps that.
            const float * logits_base = llama_get_logits(ctx_dft);
            if (logits_base == nullptr) {
                LOG_ERR("%s: llama_get_logits(ctx_dft) returned null for seq %d\n", __func__, (int) seq_id);
                continue;
            }

            bool did_cuda = false;
#ifdef LLAMA_DSPARK_MARKOV_CUDA
            // Device path: one H2D of this round's base logits, then a
            // sequential per-position fused GEMV + add-base + argmax that
            // chains through a device-resident prev token. The chaining
            // invariant is structural -- position k's kernel reads the argmax
            // position k-1's kernel wrote to device memory, never a host id.
            if (markov_use_cuda && has_markov) {
                static_assert(sizeof(llama_token) == sizeof(int32_t),
                        "dspark cuda markov path assumes llama_token == int32_t");
                result.resize((size_t) block_size);
                if (dspark_markov_cuda_resample(markov_cuda, logits_base, (int32_t) dp.id_last,
                                                block_size, (int32_t *) result.data())) {
                    // The sequential chaining is guaranteed structurally on the
                    // device (position k reads position k-1's argmax from device
                    // memory, never a host-precomputed id). mask_token_id is a
                    // real vocabulary id, so a device-sampled token can legitimately
                    // equal it -- that only yields a low-quality draft the target
                    // will reject, not a violated invariant. Warn once instead of
                    // aborting a valid run.
                    static bool warned_cuda_mask = false;
                    for (int32_t k = 1; k < block_size && !warned_cuda_mask; ++k) {
                        if (result[(size_t) (k - 1)] == mask_token_id) {
                            LOG_WRN("%s: dspark cuda markov resample produced mask_token_id at a "
                                    "chained draft position -- drafter emitted the mask sentinel; "
                                    "the target verify will reject it\n", __func__);
                            warned_cuda_mask = true;
                        }
                    }
                    did_cuda = true;
                } else {
                    LOG_WRN("%s: cuda markov resample failed for seq %d -- falling back to the host path\n",
                            __func__, (int) seq_id);
                    markov_use_cuda = false;
                    result.clear();
                }
            }
#endif

            llama_token prev_token = dp.id_last;

            if (!did_cuda)
            for (int32_t k = 0; k < block_size; ++k) {
                // prev_token is the token SAMPLED at step k-1 (assigned from best_id
                // at the end of this loop), never a draft input id -- that is the
                // real structural guarantee that this resample chains forward rather
                // than being batched over the block. A sampled token can legitimately
                // equal mask_token_id (a real vocab id), which merely makes a poor
                // draft the target rejects, so warn once instead of aborting.
                if (k > 0 && prev_token == mask_token_id) {
                    static bool warned_host_mask = false;
                    if (!warned_host_mask) {
                        LOG_WRN("%s: dspark markov resample chained a mask_token_id prev at k=%d -- "
                                "drafter emitted the mask sentinel; the target verify will reject it\n",
                                __func__, k);
                        warned_host_mask = true;
                    }
                }

                const float * base_logits = logits_base + (size_t) k * n_vocab;

                llama_token best_id = 0;
                float       best_v  = -std::numeric_limits<float>::infinity();

                if (has_markov) {
                    const float * emb = markov_w1.data() + (size_t) prev_token * (size_t) markov_rank;
#ifdef LLAMA_DSPARK_MARKOV_BLAS
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                (int) n_vocab, (int) markov_rank,
                                1.0f,
                                markov_w2.data(), (int) markov_rank,
                                emb, 1,
                                0.0f,
                                markov_bias.data(), 1);

                    for (int64_t v = 0; v < n_vocab; ++v) {
                        const float logit = base_logits[v] + markov_bias[(size_t) v];
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#else
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        const float * w2row = markov_w2.data() + (size_t) v * (size_t) markov_rank;
                        float bias = 0.0f;
                        for (int64_t r = 0; r < markov_rank; ++r) {
                            bias += emb[r] * w2row[r];
                        }
                        const float logit = base_logits[v] + bias;
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#endif
                } else {
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (base_logits[v] > best_v) {
                            best_v  = base_logits[v];
                            best_id = (llama_token) v;
                        }
                    }
                }

                result.push_back(best_id);
                prev_token = best_id; // chain the SAMPLED token, never mask_token_id
            }

            if (result.size() < (size_t) params.n_min) {
                continue; // dp.result stays empty: treated as a failed draft this round
            }

            *dp.result = std::move(result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // an accept() means the target has verified a generation round: prefill
        // is definitively over, so prefill-time draining must stay disabled.
        prefilling[seq_id] = false;

        const int64_t n_round_rows = rows_since_accept[seq_id];
        rows_since_accept[seq_id] = 0;
        if (n_round_rows <= 0) {
            return;
        }

        // process() (or the test-only injection hook) unconditionally
        // captured tap features for the WHOLE verify batch, including any
        // positions past the accepted prefix; trim this round's
        // freshly-appended tail down to n_accepted+1 rows (the actually
        // committed context), discarding the rejected continuation. This
        // mirrors the Python reference implementation's evaluator._update():
        //   context.target_hidden_states = verified_target_hidden[:, :accepted_draft_tokens+1, :]
        // Runs the same way regardless of is_other: dspark's own context must
        // stay correct even on rounds where a different implementation's
        // draft is the one that gets verified.
        const int64_t keep = std::min<int64_t>(n_round_rows, (int64_t) n_accepted + 1);
        const int64_t drop = n_round_rows - keep;

        if (drop > 0) {
            auto & feat = ctx_feat[seq_id];
            auto & pos  = ctx_pos[seq_id];

            const size_t total_rows = pos.size();
            GGML_ASSERT((int64_t) total_rows >= drop);

            feat.resize((total_rows - (size_t) drop) * (size_t) n_embd_cap);
            pos.resize(total_rows - (size_t) drop);
        }
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return false;
    }

    bool need_embd_capture() const override {
        return true;
    }

    bool stage_test_ctx_feat(
            llama_seq_id seq_id,
            const float * feat_in,
            int64_t n_rows,
            int64_t n_embd_cap_in,
            const int32_t * pos_in) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (n_embd_cap_in != n_embd_cap) {
            LOG_ERR("%s: n_embd_cap mismatch: got %lld, expected %lld\n",
                    __func__, (long long) n_embd_cap_in, (long long) n_embd_cap);
            return false;
        }
        if (n_rows <= 0) {
            return true;
        }

        auto & feat = ctx_feat[seq_id];
        auto & pos  = ctx_pos[seq_id];

        const size_t row0 = pos.size();
        feat.resize((row0 + (size_t) n_rows) * (size_t) n_embd_cap);
        pos.resize(row0 + (size_t) n_rows);

        std::memcpy(feat.data() + row0 * (size_t) n_embd_cap, feat_in, (size_t) n_rows * (size_t) n_embd_cap * sizeof(float));
        std::memcpy(pos.data() + row0, pos_in, (size_t) n_rows * sizeof(int32_t));

        rows_since_accept[seq_id] += n_rows;
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                // dspark's real upper bound is the checkpoint's block_size
                // (typically 7), which isn't known until the GGUF is loaded --
                // this function only sees CLI params. Reuse the same
                // user-configurable draft.n_max bound the other draft-model
                // types use; callers enabling dspark should set --draft-max
                // to at least the checkpoint's block_size.
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) && params.draft.ctx_dft != nullptr;
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;
        bool has_dspark = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) && params.draft.ctx_dft != nullptr;



        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_dspark) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dspark>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id, llama_pos n_keep) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->reset(seq_id, n_keep);
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_capture(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_capture()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}

bool common_speculative_dspark_stage_ctx_test(
        common_speculative * spec,
        llama_seq_id seq_id,
        const float * feat,
        int64_t n_rows,
        int64_t n_embd_cap,
        const int32_t * pos) {
    if (spec == nullptr) {
        return false;
    }

    bool any = false;
    for (auto & impl : spec->impls) {
        if (impl->stage_test_ctx_feat(seq_id, feat, n_rows, n_embd_cap, pos)) {
            any = true;
        }
    }

    return any;
}
