#include "server-common.h"
#include "server-models.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "chat.h"
#include "gguf.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <sheredom/subprocess.h>

#include <functional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

// CMD defines moved to server-models.h

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

// Returns true if the preset specifies at least one model source (local file,
// URL, or HuggingFace repo).  Presets without any source cannot be loaded and
// should be excluded from the model mapping — this prevents ghost entries
// created by INI keys that appear before the first [section] header.
static bool preset_has_model_source(const common_preset & preset) {
    static const std::vector<std::string> model_keys = {
        "LLAMA_ARG_MODEL", "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_REPO_FILE",
        COMMON_ARG_PRESET_REMOTE_URL,
    };
    std::string val;
    for (const auto & key : model_keys) {
        if (preset.get_option(key, val) && !val.empty()) {
            return true;
        }
    }
    return false;
}

// Returns true if the preset specifies a local model source (file path, URL,
// or HuggingFace repo) — i.e. anything that requires a subprocess to load.
// This excludes the __PRESET_REMOTE_URL key which is used for remote backends.
static bool preset_has_local_model(const common_preset & preset) {
    static const std::vector<std::string> local_model_keys = {
        "LLAMA_ARG_MODEL", "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_REPO_FILE",
    };
    std::string val;
    for (const auto & key : local_model_keys) {
        if (preset.get_option(key, val) && !val.empty()) {
            return true;
        }
    }
    return false;
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);

    // Parse remote URL(s) — comma-separated list of URLs
    std::string remote_url_str;
    bool has_remote = preset.get_option(COMMON_ARG_PRESET_REMOTE_URL, remote_url_str) && !remote_url_str.empty();

    remote_urls.clear();
    if (has_remote) {
        for (auto & entry : string_split<std::string>(remote_url_str, ',')) {
            entry = string_strip(entry);
            if (!entry.empty()) {
                remote_urls.push_back(entry);
            }
        }
    }

    // Parse remote API key(s) — comma-separated list matching remote_urls order
    std::string remote_api_key_str;
    bool has_api_key = preset.get_option(COMMON_ARG_PRESET_REMOTE_API_KEY, remote_api_key_str) && !remote_api_key_str.empty();

    remote_api_keys.clear();
    if (has_api_key) {
        for (auto & entry : string_split<std::string>(remote_api_key_str, ',')) {
            entry = string_strip(entry);
            if (!entry.empty()) {
                remote_api_keys.push_back(entry);
            }
        }
    }

    // Parse remote max concurrency (per remote backend). 0 = unlimited.
    remote_max_concurrency = 0;
    std::string remote_max_conc_str;
    if (preset.get_option(COMMON_ARG_PRESET_REMOTE_MAX_CONCURRENCY, remote_max_conc_str) && !remote_max_conc_str.empty()) {
        try {
            remote_max_concurrency = std::max(0, std::stoi(string_strip(remote_max_conc_str)));
        } catch (...) {
            remote_max_concurrency = 0;
        }
    }

    // Parse remote fallback concurrency: effective cap when max_concurrency=0,
    // triggering a local switch once active_connections exceed this threshold.
    // 0 means never apply a fallback cap (remote is never skipped due to load).
    remote_fallback_concurrency = 0;
    std::string remote_fallback_conc_str;
    if (preset.get_option(COMMON_ARG_PRESET_REMOTE_FALLBACK_CONCURRENCY, remote_fallback_conc_str) && !remote_fallback_conc_str.empty()) {
        try {
            remote_fallback_concurrency = std::max(0, std::stoi(string_strip(remote_fallback_conc_str)));
        } catch (...) {
            remote_fallback_concurrency = 0;
        }
    }

    // Determine if there's a local model source (separate from remote URL presence)
    const bool has_local = preset_has_local_model(preset);
    is_remote = !has_local && !remote_urls.empty();

    // Set is_https if any remote URL uses https (regardless of local model presence)
    if (!remote_urls.empty()) {
        bool any_https = false;
        for (const auto & url : remote_urls) {
            bool this_https = url.rfind("https://", 0) == 0;
            any_https = any_https || this_https;
        }
        is_https = any_https;

        // Parse the first URL for host/port (display fallback)
        const auto & first_url = remote_urls[0];
        bool first_https = first_url.rfind("https://", 0) == 0;
        std::string address;
        if (first_https) {
            address = first_url.substr(8);
        } else if (first_url.rfind("http://", 0) == 0) {
            address = first_url.substr(7);
        } else {
            address = first_url;
        }
        size_t colon = address.rfind(':');
        if (colon != std::string::npos) {
            host = address.substr(0, colon);
            try {
                port = std::stoi(address.substr(colon + 1));
            } catch (...) {
                port = first_https ? 443 : 80;
            }
        } else {
            host = address;
            port = first_https ? 443 : 80;
        }
    }

    // Configure local backend if there's a local model source
    if (has_local) {
        preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
        preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    }
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
            "LLAMA_ARG_CHAT_TEMPLATE",
            "LLAMA_ARG_CHAT_TEMPLATE_FILE",
        });
        params.offline = true;
        // params.skip_download = true; // TODO: ideally, we should validate the model here, but it takes too much time
        common_params_handle_models(params, LLAMA_EXAMPLE_SERVER);
        if (params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }

        // Detect reasoning/thinking support offline so the web UI can show the
        // reasoning toggle for models that are not currently loaded. The child
        // server computes this the same way (see server-context.cpp), but only
        // while running, so we mirror it here from GGUF metadata.
        std::string tmpl_src = params.chat_template; // explicit override from preset, if any
        if (tmpl_src.empty() && !params.model.path.empty()) {
            struct gguf_init_params gguf_params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
            gguf_context * ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gguf_params);
            if (ctx_gguf) {
                int64_t key_id = gguf_find_key(ctx_gguf, "tokenizer.chat_template");
                if (key_id >= 0) {
                    const char * str = gguf_get_val_str(ctx_gguf, key_id);
                    if (str) {
                        tmpl_src = str;
                    }
                }
                gguf_free(ctx_gguf);
            }
        }
        chat_template = tmpl_src;
        if (!tmpl_src.empty()) {
            // Capability check only (mirrors the web UI's template heuristic, which is
            // independent of whether --jinja is enabled). The child server additionally
            // gates actual thinking on use_jinja at generation time.
            common_chat_templates_ptr tmpls = common_chat_templates_init(/*model =*/ nullptr, tmpl_src);
            supports_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
        } else {
            supports_thinking = false;
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
        chat_template = "";
        supports_thinking = false;
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
    // start the liveness watchdog after the initial mapping is populated
    watchdog_th = std::thread([this]() { watchdog_loop(); });
}

server_models::~server_models() {
    watchdog_stop.store(true);
    if (watchdog_th.joinable()) {
        watchdog_th.join();
    }
}

// Probe a child's /health endpoint. Any HTTP response (even 503 "loading") means
// the child's HTTP server is alive; only a transport-level failure (no response)
// indicates a wedged/dead backend. /health is unauthenticated in llama-server.
static bool probe_child_health(const std::string & host, int port, const std::string & api_key = "") {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(2, 0);
    cli.set_write_timeout(2, 0);

    if (!api_key.empty()) {
        httplib::Headers headers;
        headers.emplace("Authorization", "Bearer " + api_key);
        auto res = cli.Get("/health", headers);
        return static_cast<bool>(res);
    }

    auto res = cli.Get("/health");
    return static_cast<bool>(res);
}

std::string server_models::get_backend_api_key() {
    if (!base_params.remote_api_key.empty()) {
        return base_params.remote_api_key;
    }
    if (!base_params.api_keys.empty()) {
        return base_params.api_keys[0];
    }
    return "";
}

void server_models::mark_backend_dead(const std::string & name, subprocess_s * proc, int backend_idx) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && it->second.meta.is_running()) {
        // Find the backend matching the given proc and mark it dead.
        // If proc is nullptr and backend_idx < 0, mark all running backends
        // (e.g. from watchdog on a non-specific crash).
        // If backend_idx >= 0, mark only that specific backend (used for remote backends).
        bool marked = false;
        for (int bi = 0; bi < (int)it->second.backends.size(); ++bi) {
            auto & b = it->second.backends[bi];
            if (backend_idx >= 0 && (int)bi == backend_idx) {
                // Exact backend match (remote backend path).
                b.status = SERVER_MODEL_STATUS_UNLOADED;
                b.health_fail_count = 0;
                marked = true;
            } else if (backend_idx < 0 && (proc == nullptr || (b.subproc && b.subproc.get() == proc))) {
                b.status = SERVER_MODEL_STATUS_UNLOADED;
                b.health_fail_count = 0;
                marked = true;
            }
        }
        // Also update meta status, but only if ALL backends are now down.
        if (marked) {
            bool any_live = false;
            for (const auto & b : it->second.backends) {
                if (b.status != SERVER_MODEL_STATUS_UNLOADED) {
                    any_live = true;
                    break;
                }
            }
            if (!any_live) {
                it->second.meta.status    = SERVER_MODEL_STATUS_UNLOADED;
                it->second.meta.exit_code = 1;
            }
            cv.notify_all();
        }
    }
}

void server_models::watchdog_loop() {
    struct probe_t {
        std::string                   name;
        int                           backend_idx;
        std::shared_ptr<subprocess_s> subproc; // keep the struct alive while we probe it
        std::string                   host;
        int                           port;
        server_model_status           st;
        bool                          is_remote;
        std::string                   api_key;  // per-backend API key captured at snapshot time
    };
    while (!watchdog_stop.load()) {
        // 1) snapshot running instances under the lock (no blocking work here).
        //    hold a shared_ptr so a concurrent reap/replace can't free the subprocess
        //    out from under us between snapshot and probe.
        std::vector<probe_t> probes;
        {
            std::unique_lock<std::mutex> lk(mutex);
            std::lock_guard<std::mutex> lk2(stop_mutex); // lock order: mutex -> stop_mutex
            for (auto & [name, inst] : mapping) {
                // skip models we are intentionally stopping: their HTTP server is down
                // while the child flushes KV to disk, so /health would falsely fail and
                // a SIGKILL here would abort the graceful KVC save.
                if (stopping_models.find(name) != stopping_models.end()) {
                    continue;
                }
                if (inst.meta.status == SERVER_MODEL_STATUS_LOADED ||
                    inst.meta.status == SERVER_MODEL_STATUS_SLEEPING) {
                    // Probe each backend independently
                    for (int bi = 0; bi < (int)inst.backends.size(); ++bi) {
                        auto & b = inst.backends[bi];
                        probes.push_back({name, bi, b.subproc, b.host, b.port, b.status, b.is_remote, b.api_key});
                    }
                }
            }
        }
        // 2) probe OUTSIDE the lock (subprocess_alive + /health may block briefly)
        for (auto & p : probes) {
            if (watchdog_stop.load()) {
                break;
            }
            bool dead    = false;
            bool wedged = false;

            if (p.is_remote) {
                // Remote backends: no subprocess to check; rely on HTTP probe only.
                const bool healthy = probe_child_health(p.host, p.port, p.api_key);
                std::unique_lock<std::mutex> lk(mutex);
                auto it = mapping.find(p.name);
                if (it != mapping.end() && (size_t)p.backend_idx < it->second.backends.size()) {
                    if (healthy) {
                        it->second.backends[p.backend_idx].health_fail_count = 0;
                        if (it->second.backends[p.backend_idx].status == SERVER_MODEL_STATUS_UNLOADED) {
                            it->second.backends[p.backend_idx].status = SERVER_MODEL_STATUS_LOADED;
                        }
                    } else {
                        if (it->second.backends[p.backend_idx].health_fail_count < INT_MAX) {
                            it->second.backends[p.backend_idx].health_fail_count++;
                        }
                        wedged = (it->second.backends[p.backend_idx].health_fail_count >= 3);
                    }
                }
            } else {
                // Local backends: check subprocess alive first, then HTTP probe if loaded.
                subprocess_s * proc = p.subproc.get();
                dead = !subprocess_alive(proc);
                if (!dead && p.st == SERVER_MODEL_STATUS_LOADED) {
                    // /health is served on the child's HTTP thread, independent of the
                    // inference loop, so it answers even mid-generation. Require several
                    // consecutive failures before declaring the backend wedged.
                    const bool healthy = probe_child_health(p.host, p.port, p.api_key);
                    std::unique_lock<std::mutex> lk(mutex);
                    auto it = mapping.find(p.name);
                    if (it != mapping.end() && (size_t)p.backend_idx < it->second.backends.size()) {
                        if (it->second.backends[p.backend_idx].subproc.get() == proc) {
                            if (healthy) {
                                it->second.backends[p.backend_idx].health_fail_count = 0;
                                if (it->second.backends[p.backend_idx].status == SERVER_MODEL_STATUS_UNLOADED) {
                                    it->second.backends[p.backend_idx].status = SERVER_MODEL_STATUS_LOADED;
                                }
                            } else {
                                if (it->second.backends[p.backend_idx].health_fail_count < INT_MAX) {
                                    it->second.backends[p.backend_idx].health_fail_count++;
                                }
                                wedged = (it->second.backends[p.backend_idx].health_fail_count >= 3);
                            }
                        }
                    }
                }
            }

            if (dead || wedged) {
                // For local backends, verify the probed process is still the current
                // backend's process — a stale probe from before a respawn should not
                // trigger a second mark_dead or terminate.
                if (!p.is_remote) {
                    bool proc_stale = false;
                    {
                        std::unique_lock<std::mutex> lk(mutex);
                        auto it = mapping.find(p.name);
                        if (it == mapping.end()
                            || (size_t)p.backend_idx >= it->second.backends.size()
                            || it->second.backends[p.backend_idx].subproc.get() != p.subproc.get()) {
                            proc_stale = true;
                        }
                    }
                    if (proc_stale) {
                        continue; // backend was respawned; skip stale probe
                    }
                }

                SRV_WRN("watchdog: model %s backend %d — %s — marking UNLOADED for respawn\n",
                    p.name.c_str(), p.backend_idx, dead ? "exited" : "unresponsive");
                if (!p.is_remote) {
                    // force-kill so the child's stdout closes and the monitor thread
                    // unblocks to reap the process (clears the zombie).
                    subprocess_terminate(p.subproc.get());
                }
                if (p.is_remote) {
                    mark_backend_dead(p.name, nullptr, p.backend_idx);
                } else {
                    mark_backend_dead(p.name, p.subproc.get());
                }
            }
        }
        // ~2s poll, interruptible
        for (int i = 0; i < 20 && !watchdog_stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps();
    std::string name = meta.name;
    backend_t b;
    b.subproc = std::make_shared<subprocess_s>();
    mapping[name] = instance_t{
        /* th       */ std::thread(),
        /* meta     */ std::move(meta),
        /* backends */ {std::move(b)}
    };
}

void server_models::load_models() {
    // Phase 1: load presets from all sources — pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    for (const auto & [name, preset] : cached_models) final_presets[name] = preset;
    for (const auto & [name, preset] : local_models)  final_presets[name] = preset;
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
    }
    // server base preset from CLI args takes highest precedence
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    // Helpers that read `mapping` — must be called while holding the lock.
    std::unordered_set<std::string> custom_names;
    for (const auto & [name, preset] : custom_presets) custom_names.insert(name);
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            std::string info;
            if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally
    // (unload, load) or when joining threads (the monitoring thread calls update_status
    // which locks the mutex, so joining while holding it would deadlock).
    std::unique_lock<std::mutex> lk(mutex);
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            if (!preset_has_model_source(preset)) {
                SRV_WRN("skipping model preset '%s': no model source configured\n", name.c_str());
                continue;
            }
            server_model_meta meta{
              /* preset           */ preset,
                /* name             */ name,
                /* aliases          */ {},
                /* tags             */ {},
                /* port             */ 0,
                /* is_remote        */ false,
                /* is_https         */ false,
                /* host             */ CHILD_ADDR,
                /* remote_urls        */ {},
                /* remote_api_keys    */ {},
                /* remote_max_concurrency */ 0,
                /* remote_fallback_concurrency */ 0,
                /* status             */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used        */ 0,
                /* args             */ std::vector<std::string>(),
                /* loaded_info      */ {},
                /* exit_code        */ 0,
                /* stop_timeout     */ DEFAULT_STOP_TIMEOUT,
                /* multimodal       */ mtmd_caps{false, false},
                /* chat_template    */ {},
                /* supports_thinking */ false,
                /* need_download    */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        log_available_models();

        std::vector<std::string> models_to_load;
        for (const auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                models_to_load.push_back(name);
            }
        }
        if ((int)models_to_load.size() > base_params.models_max) {
            throw std::runtime_error(string_format(
                "number of models to load on startup (%zu) exceeds models_max (%d)",
                models_to_load.size(), base_params.models_max));
        }

        lk.unlock();
        for (const auto & name : models_to_load) {
            SRV_INF("(startup) loading model %s\n", name.c_str());
            load(name);
        }
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // collect all threads to join in one pass while the lock is held:
        // - monitoring threads from just-unloaded models (to_unload)
        // - threads of already-UNLOADED models that are being removed from source
        std::vector<std::thread> threads_to_join;
        for (const auto & name : to_unload) {
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.th.joinable()) {
                threads_to_join.push_back(std::move(it->second.th));
            }
        }
        for (auto & [name, inst] : mapping) {
            if (final_presets.find(name) == final_presets.end() && !inst.meta.is_running() && inst.th.joinable()) {
                threads_to_join.push_back(std::move(inst.th));
            }
        }

        // join outside the lock — monitoring thread calls update_status (needs lock)
        lk.unlock();
        for (auto & th : threads_to_join) th.join();
        lk.lock();

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                GGML_ASSERT(!it->second.th.joinable()); // must have been joined above
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps();
        }

        // add models that are new in this reload
        std::vector<std::string> newly_added;
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                if (!preset_has_model_source(preset)) {
                    SRV_WRN("(reload) skipping model preset '%s': no model source configured\n", name.c_str());
                    continue;
                }
 server_model_meta meta{
                  /* preset            */ preset,
                    /* name                */ name,
                    /* aliases             */ {},
                    /* tags                */ {},
                    /* port                */ 0,
                    /* is_remote           */ false,
                    /* is_https            */ false,
                    /* host                */ CHILD_ADDR,
 /* remote_urls        */ {},
                /* remote_api_keys    */ {},
                /* remote_max_concurrency */ 0,
                /* remote_fallback_concurrency */ 0,
                /* status             */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used        */ 0,
                /* args             */ std::vector<std::string>(),
                /* loaded_info      */ {},
                /* exit_code        */ 0,
                /* stop_timeout     */ DEFAULT_STOP_TIMEOUT,
                /* multimodal       */ mtmd_caps{false, false},
                /* chat_template    */ {},
                /* supports_thinking */ false,
                /* need_download    */ false,
            };
                add_model(std::move(meta));
                newly_added.push_back(name);
            }
        }

        apply_stop_timeout();

        // clear reload flag before unlocking for autoload — load() blocks on !is_reloading,
        // so clearing it here (while still locked) prevents a deadlock in the autoload calls below
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        // collect autoload candidates while still under the lock
        std::vector<std::string> to_autoload;
        for (const auto & name : newly_added) {
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                std::string val;
                if (it->second.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    to_autoload.push_back(name);
                }
            }
        }

        lk.unlock();
        for (const auto & name : to_autoload) {
            SRV_INF("(reload) loading new model %s\n", name.c_str());
            load(name);
        }
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    // "default" wildcard: if no model is explicitly named "default", resolve to
    // the most-recently-used model.  Prefer a currently loaded (ready) model so
    // requests are served immediately; fall back to the MRU unloaded model so
    // that autoload can restart a crashed child instead of returning "not found".
    if (name == COMMON_PRESET_DEFAULT_NAME || name.empty()) {
        const instance_t * best_loaded   = nullptr;
        const instance_t * best_unloaded = nullptr;
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.is_ready()) {
                if (!best_loaded || inst.meta.last_used > best_loaded->meta.last_used) {
                    best_loaded = &inst;
                }
            } else if (!inst.meta.is_running() && inst.meta.last_used > 0) {
                if (!best_unloaded || inst.meta.last_used > best_unloaded->meta.last_used) {
                    best_unloaded = &inst;
                }
            }
        }
        if (best_loaded)   return best_loaded->meta;
        if (best_unloaded) return best_unloaded->meta;
    }
    return std::nullopt;
}

static int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (getsockname(sock, (struct sockaddr*)&serv_addr, &namelen) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    int port = ntohs(serv_addr.sin_port);

    CLOSE_SOCKET(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    return port;
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    // Skip models that are currently loading or actively serving requests — evicting them
    // causes thrashing (loading model) or interrupts live generation (active model).
    // If every running model is busy (loading or has active connections), block until at
    // least one becomes idle before evicting it.
    while (true) {
        std::string lru_model_name = "";
        int64_t lru_last_used = ggml_time_ms();
        size_t count_active = 0;
        bool any_busy = false;
        {
            std::unique_lock<std::mutex> lk(mutex);
            for (const auto & m : mapping) {
                if (m.second.meta.is_running()) {
                    // Remote backends don't count toward local capacity limits
                    // and can't be evicted.
                    bool is_local = false;
                    for (const auto & b : m.second.backends) {
                        if (!b.is_remote) {
                            is_local = true;
                            break;
                        }
                    }
                    if (!is_local) {
                        continue; // skip remote-only models entirely
                    }

                    count_active++;
                    int total_active = 0;
                    for (const auto & b : m.second.backends) {
                        total_active += b.active_connections;
                    }
                    bool busy = m.second.meta.status == SERVER_MODEL_STATUS_LOADING
                               || total_active > 0;
                    if (busy) {
                        any_busy = true;
                    } else if (m.second.meta.last_used < lru_last_used) {
                        lru_model_name = m.first;
                        lru_last_used = m.second.meta.last_used;
                    }
                }
            }
        }

        if (count_active < (size_t)base_params.models_max) {
            return; // capacity available, nothing to evict
        }

        if (!lru_model_name.empty()) {
            SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
            unload(lru_model_name);
            // wait for unload to complete
            {
                std::unique_lock<std::mutex> lk(mutex);
                cv.wait(lk, [this, &lru_model_name]() {
                    return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
                });
            }
            return;
        }

        // All running models are busy (loading or actively serving); wait for any one to
        // become idle (loading finished or last connection closed) before retrying.
        if (any_busy) {
            SRV_INF("%s", "models_max limit reached, all models busy — waiting for one to become idle...\n");
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this]() {
                for (const auto & m : mapping) {
                    if (!m.second.meta.is_running()) {
                        continue;
                    }
                    // Skip remote-only models — they aren't evictable
                    bool is_local = false;
                    for (const auto & b : m.second.backends) {
                        if (!b.is_remote) {
                            is_local = true;
                            break;
                        }
                    }
                    if (!is_local) {
                        continue;
                    }
                    if (m.second.meta.status != SERVER_MODEL_STATUS_LOADING) {
                        int total_active = 0;
                        for (const auto & b : m.second.backends) {
                            total_active += b.active_connections;
                        }
                        if (total_active == 0) {
                            return true; // at least one evictable model available
                        }
                    }
                }
                return false;
            });
            // retry the scan
        } else {
            return; // no running models at all, nothing to evict
        }
    }
}

std::thread server_models::spawn_local_managing_thread(const std::string & name, std::shared_ptr<subprocess_s> child_proc, int port, int stop_timeout) {
    // captured variables are guaranteed to be destroyed only after the thread is joined
    return std::thread([this, name, child_proc, port, stop_timeout]() {
        FILE * stdin_file = subprocess_stdin(child_proc.get());
        FILE * stdout_file = subprocess_stdout(child_proc.get()); // combined stdout/stderr

        std::thread log_thread([&]() {
        // read stdout/stderr and forward to main server log
        // also handle status report from child process
        std::vector<char> vec_buf(128 * 1024); // large buffer for storing info
        char * buffer = vec_buf.data();
        if (stdout_file) {
            while (fgets(buffer, vec_buf.size(), stdout_file) != nullptr) {
                LOG("[%5d] %s", port, buffer);
                std::string str(buffer);
                if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_READY)) {
                    this->update_status(name, SERVER_MODEL_STATUS_LOADED, 0, child_proc.get());
                } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_ERROR)) {
                    SRV_ERR("model name=%s loading error: %s\n", name.c_str(), buffer);
                    this->update_status(name, SERVER_MODEL_STATUS_UNLOADED, 1, child_proc.get());
                    std::string err_msg(buffer);
                    size_t prefix_len = strlen(CMD_CHILD_TO_ROUTER_ERROR);
                    if (err_msg.size() > prefix_len) {
                        auto trimmed = err_msg.substr(prefix_len);
                        while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == '\r')) {
                            trimmed.pop_back();
                        }
                        this->update_last_error(name, trimmed);
                    }
                } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_INFO)) {
                    this->update_loaded_info(name, str);
                } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_SLEEP)) {
                    this->update_status(name, SERVER_MODEL_STATUS_SLEEPING, 0, child_proc.get());
                }
            }
            // EOF on stdout — child process exited (could be a crash).
            // Immediately mark UNLOADED so /v1/models stops advertising
            // this model as loaded.
            if (feof(stdout_file)) {
                this->update_status(name, SERVER_MODEL_STATUS_UNLOADED, 1, child_proc.get());
            }
        } else {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
        }
    });

    std::thread stopping_thread([&]() {
        // thread to monitor stopping signal OR child crash
        auto is_stopping = [this, &name]() {
            return this->stopping_models.find(name) != this->stopping_models.end();
        };
        auto should_wake = [&]() {
            return is_stopping() || !subprocess_alive(child_proc.get());
        };
        {
            std::unique_lock<std::mutex> lk(this->stop_mutex);
            this->cv_stop.wait(lk, should_wake);
        }
        // child may have already exited (e.g. crashed) — skip shutdown sequence
        if (!subprocess_alive(child_proc.get())) {
            return;
        }
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        // send interrupt to child process
        fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(stdin_file);
        // wait to stop gracefully or timeout
        int64_t start_time = ggml_time_ms();
        while (true) {
            std::unique_lock<std::mutex> lk(this->stop_mutex);
            if (!is_stopping()) {
                return; // already stopped
            }
            int64_t elapsed = ggml_time_ms() - start_time;
            if (elapsed >= stop_timeout * 1000) {
                // timeout, force kill
                SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                subprocess_terminate(child_proc.get());
                return;
            }
            this->cv_stop.wait_for(lk, std::chrono::seconds(1));
        }
    });

    // we reach here when the child process exits
    // note: we cannot join() prior to this point because it will close stdin_file
    if (log_thread.joinable()) {
        log_thread.join();
    }

    // The log thread may have detected EOF on stdout (child hung up)
    // without the child actually exiting — e.g. the client disconnected.
    // In that case the stopping thread is still waiting on cv_stop
    // because is_stopping() is false and subprocess_alive() is true.
    // Kill the child here so the stopping thread unblocks and cleanup
    // (subprocess_join/destroy) runs, freeing GPU memory.
    if (subprocess_alive(child_proc.get())) {
        SRV_WRN("model name=%s child still alive after log thread EOF, force-killing\n", name.c_str());
        subprocess_terminate(child_proc.get());
    }

    // stop the timeout monitoring thread
    {
        std::lock_guard<std::mutex> lk(this->stop_mutex);
        stopping_models.erase(name);
        cv_stop.notify_all();
    }
    if (stopping_thread.joinable()) {
        stopping_thread.join();
    }

    // get the exit code
    int exit_code = 0;
    subprocess_join(child_proc.get(), &exit_code);
    subprocess_destroy(child_proc.get());

    // update status and exit code — pass child_proc so stale updates from a replaced
    // instance don't overwrite the status of the new one (see update_status guard).
    this->update_status(name, SERVER_MODEL_STATUS_UNLOADED, exit_code, child_proc.get());
    SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });
}

void server_models::load(const std::string & name) {
    if (!has_model(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    // Decide up front whether this load will spawn a local child. A hybrid (local+remote)
    // model is served from its remote backend(s) and does NOT spawn a local child on load;
    // it only does so later via escalate_to_local() when no remote backend is usable. Like
    // remote-only models, hybrid models occupy no local capacity and must not evict others.
    bool will_spawn_local;
    {
        auto meta0 = get_meta(name);
        const bool has_local0 = meta0.has_value() && preset_has_local_model(meta0->preset);
        std::string rurl;
        const bool has_remote0 = meta0.has_value()
            && meta0->preset.get_option(COMMON_ARG_PRESET_REMOTE_URL, rurl) && !rurl.empty();
        will_spawn_local = has_local0 && !has_remote0;
    }

    if (will_spawn_local) {
        unload_lru();
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    if (will_spawn_local && base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                // Only count local models toward capacity
                bool is_local = false;
                for (const auto & b : m.second.backends) {
                    if (!b.is_remote) {
                        is_local = true;
                        break;
                    }
                }
                if (!is_local) {
                    continue;
                }
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }

    // prepare new instance info
    instance_t inst;
    inst.backends.clear(); // defensive: prevent unbounded backend accumulation on repeated loads
    inst.meta             = meta;
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

   // Render args first — update_args() parses remote_urls and determines has_local
    inst.meta.update_args(ctx_preset, bin_path);

    // Check if there's a local model to spawn
    const bool has_local = preset_has_local_model(inst.meta.preset);
    const bool has_remote = !inst.meta.remote_urls.empty();
    // Hybrid models (local + remote) defer their local child to escalate_to_local(); only a
    // local-only model spawns its child here at load time.
    const bool spawn_local_now = has_local && !has_remote;

    // Only allocate a local port for backends that spawn a local subprocess now
    if (spawn_local_now) {
        inst.meta.port = get_free_port();
        if (inst.meta.port <= 0) {
            throw std::runtime_error("failed to get a port number");
        }
        // Re-render args with the assigned port baked into the preset
        inst.meta.update_args(ctx_preset, bin_path);
    }

    // 1) Local backend: spawn subprocess if there's a local-only model source
    if (spawn_local_now) {
        backend_t b;
        b.is_remote = false;
        b.priority = 0;
        b.subproc = std::make_shared<subprocess_s>();
        b.host = CHILD_ADDR;
        b.port = inst.meta.port;
        b.status = SERVER_MODEL_STATUS_LOADING;
        inst.backends.push_back(std::move(b));

        {
            SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

            std::vector<std::string> child_args = inst.meta.args; // copy
            std::vector<std::string> child_env  = base_env; // copy
            child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

            SRV_INF("%s", "spawning server instance with args:\n");
            for (const auto & arg : child_args) {
                SRV_INF("  %s\n", arg.c_str());
            }
            inst.meta.args = child_args; // save for debugging

            std::vector<char *> argv = to_char_ptr_array(child_args);
            std::vector<char *> envp = to_char_ptr_array(child_env);

            // TODO @ngxson : maybe separate stdout and stderr in the future
            //                so that we can use stdout for commands and stderr for logging
            int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
            int result = subprocess_create_ex(argv.data(), options, envp.data(), inst.backends[0].subproc.get());
            if (result != 0) {
                throw std::runtime_error("failed to spawn server instance");
            }

            inst.backends[0].stdin_file = subprocess_stdin(inst.backends[0].subproc.get());
        }
    }

    // 2) Create remote backends for each remote URL
    for (const auto & url : inst.meta.remote_urls) {
        bool is_https_url = url.rfind("https://", 0) == 0;
        std::string address;
        if (is_https_url) {
            address = url.substr(8);
        } else if (url.rfind("http://", 0) == 0) {
            address = url.substr(7);
        } else {
            address = url;
        }
        size_t colon = address.rfind(':');
        std::string h;
        int p = is_https_url ? 443 : 80;
        if (colon != std::string::npos) {
            h = address.substr(0, colon);
            try {
                p = std::stoi(address.substr(colon + 1));
            } catch (...) {
                p = is_https_url ? 443 : 80;
            }
        } else {
            h = address;
        }

        backend_t b;
        b.is_remote = true;
        b.priority = 1;
        b.is_https = is_https_url;
        b.host = h;
        b.port = p;
        b.status = SERVER_MODEL_STATUS_LOADED;
        b.max_concurrency = inst.meta.remote_max_concurrency;
        b.fallback_concurrency = inst.meta.remote_fallback_concurrency;

        // Assign per-backend API key from config (falls back to global if not set)
        size_t backend_idx = inst.backends.size();
        if (backend_idx < inst.meta.remote_api_keys.size()) {
            b.api_key = inst.meta.remote_api_keys[backend_idx];
        } else {
            // Fallback to global backend API key
            b.api_key = get_backend_api_key();
        }

        inst.backends.push_back(std::move(b));
    }

    // Models with no local child to spawn now (remote-only, or hybrid deferring to remote)
    // are considered "loaded" as soon as their remote backend(s) are registered.
    if (!spawn_local_now && !inst.meta.remote_urls.empty()) {
        inst.meta.status = SERVER_MODEL_STATUS_LOADED;
    }

    // start a thread to manage the local child process (only if a local child was spawned).
    // Hybrid models defer their local child to escalate_to_local(), which creates its
    // managing thread on demand via spawn_local_managing_thread().
    if (spawn_local_now) {
        inst.th = spawn_local_managing_thread(name, inst.backends[0].subproc, inst.meta.port, inst.meta.stop_timeout);
    }

    // Drain / replace the old instance.
    // IMPORTANT: do NOT call th.join() while holding the mutex — the old monitoring
    // thread's final update_status() also acquires the mutex, causing a deadlock when
    // the child crashes while load() is executing (crash → CMD_ERROR sets UNLOADED →
    // new request immediately calls load() → holds mutex → joins old thread → deadlock).
    // Fix: move the old thread out of the mapping, replace the mapping entry (with the
    // new instance) and notify waiters, then release the mutex before joining.
    std::thread old_th;
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        // Check if old instance has a local backend with an alive subprocess
        bool old_has_local = false;
        int old_local_idx = -1;
        for (int i = 0; i < (int)old_instance.backends.size(); ++i) {
            if (!old_instance.backends[i].is_remote && old_instance.backends[i].subproc) {
                old_has_local = true;
                old_local_idx = i;
                break;
            }
        }
        if (old_has_local
            && old_instance.backends[old_local_idx].subproc
            && subprocess_alive(old_instance.backends[old_local_idx].subproc.get())) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            subprocess_terminate(old_instance.backends[old_local_idx].subproc.get()); // force kill
        }
        old_th = std::move(old_instance.th); // move out; join happens after lock release
    }

    mapping[name] = std::move(inst);
    cv.notify_all();
    lk.unlock(); // release before joining — old thread may need the mutex for update_status

    if (old_th.joinable()) {
        old_th.join();
    }
}

void server_models::unload(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.is_running()) {
            // Check if there are any local backends with subprocesses to manage
            bool has_local_backend = false;
            for (const auto & b : it->second.backends) {
                if (!b.is_remote && b.subproc) {
                    has_local_backend = true;
                    break;
                }
            }

            if (!has_local_backend) {
                // Remote-only backends have no subprocess to manage — just flip to UNLOADED.
                SRV_INF("unloading remote-only model name=%s\n", name.c_str());
                it->second.meta.status = SERVER_MODEL_STATUS_UNLOADED;
                for (auto & b : it->second.backends) {
                    b.status = SERVER_MODEL_STATUS_UNLOADED;
                }
                cv.notify_all();
                return;
            }

            SRV_INF("stopping model instance name=%s\n", name.c_str());
            {
                std::lock_guard<std::mutex> lk2(stop_mutex);
                stopping_models.insert(name);
                cv_stop.notify_all();
            }
            if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
                // special case: if model is in loading state, unloading means force-killing local backends
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                for (auto & b : it->second.backends) {
                    if (b.subproc) {
                        subprocess_terminate(b.subproc.get());
                    }
                }
            }
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        std::lock_guard<std::mutex> lk2(stop_mutex);
        for (auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;

            // Check if there are any local backends with subprocesses to manage
            bool has_local_backend = false;
            for (const auto & b : inst.backends) {
                if (!b.is_remote && b.subproc) {
                    has_local_backend = true;
                    break;
                }
            }

            if (!has_local_backend) {
                // Remote-only backends: no subprocess to manage, just mark UNLOADED.
                SRV_INF("unloading remote-only model name=%s\n", name.c_str());
                inst.meta.status = SERVER_MODEL_STATUS_UNLOADED;
                for (auto & b : inst.backends) {
                    b.status = SERVER_MODEL_STATUS_UNLOADED;
                }
                continue;
            }

            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);
            cv_stop.notify_all();
            // status change will be handled by the managing thread
            // Mark all backends UNLOADED so select_backend filters them while stopping
            for (auto & b : inst.backends) {
                b.status = SERVER_MODEL_STATUS_UNLOADED;
            }

            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, server_model_status status, int exit_code, subprocess_s * proc) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        // If a subprocess pointer is provided, guard against stale updates from a replaced
        // (crashed) child overwriting the status of a newly-spawned replacement instance.
        // Scan backends to find the matching one (supports multi-backend).
        if (proc != nullptr) {
            bool found = false;
            for (auto & b : it->second.backends) {
                if (b.subproc && b.subproc.get() == proc) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return; // stale update — belongs to a previous instance, ignore it
            }
        }
        auto & meta = it->second.meta;
        // Update per-backend status: if a specific proc is given, only update that backend;
        // otherwise propagate to all backends so select_backend can filter.
        for (auto & b : it->second.backends) {
            if (proc == nullptr || (b.subproc && b.subproc.get() == proc)) {
                b.status = status;
            }
        }
        if (proc == nullptr) {
            // Whole-model update (no specific backend): apply status directly.
            meta.status    = status;
            meta.exit_code = exit_code;
        } else {
            // Per-backend update: derive the model's aggregate status from its backends so a
            // hybrid model stays LOADED while any backend (e.g. its remote) is still serving —
            // a local child crash must not mark the whole model down while the remote is alive.
            server_model_status agg = SERVER_MODEL_STATUS_UNLOADED;
            bool any_loaded = false, any_loading = false, any_sleeping = false;
            for (const auto & b : it->second.backends) {
                switch (b.status) {
                    case SERVER_MODEL_STATUS_LOADED:   any_loaded   = true; break;
                    case SERVER_MODEL_STATUS_LOADING:  any_loading  = true; break;
                    case SERVER_MODEL_STATUS_SLEEPING: any_sleeping = true; break;
                    default: break;
                }
            }
            if (any_loaded)        agg = SERVER_MODEL_STATUS_LOADED;
            else if (any_loading)  agg = SERVER_MODEL_STATUS_LOADING;
            else if (any_sleeping) agg = SERVER_MODEL_STATUS_SLEEPING;
            meta.status = agg;
            // Only record a failure exit code when the whole model ends up down.
            if (agg == SERVER_MODEL_STATUS_UNLOADED) {
                meta.exit_code = exit_code;
            }
        }
    }
    cv.notify_all();
}

void server_models::update_loaded_info(const std::string & name, std::string & raw_info) {
    if (!string_starts_with(raw_info, CMD_CHILD_TO_ROUTER_INFO)) {
        SRV_WRN("invalid loaded info format from child for model name=%s: %s\n", name.c_str(), raw_info.c_str());
        return;
    }

    json info;
    try {
        info = json::parse(raw_info.substr(strlen(CMD_CHILD_TO_ROUTER_INFO)));
    } catch (const std::exception & e) {
        SRV_WRN("failed to parse loaded info from child for model name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.loaded_info = info;
    }
    cv.notify_all();
}

void server_models::update_last_error(const std::string & name, const std::string & error) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta.last_error = error;
    }
}

void server_models::wait_until_loading_finished(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this, &name]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return it->second.meta.status != SERVER_MODEL_STATUS_LOADING;
        }
        return false;
    });
}

bool server_models::ensure_model_ready(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    // wait for loading to complete (only needed if there's a local backend with subprocess)
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            bool has_local_backend = false;
            for (const auto & b : it->second.backends) {
                if (!b.is_remote && b.subproc) {
                    has_local_backend = true;
                    break;
                }
            }
            if (has_local_backend) {
                SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
                lk.unlock();
                wait_until_loading_finished(name);
            }
        }
    }

    // check final status
    meta = get_meta(name);
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    // For remote-only models, verify connectivity via /health probe.
    // For mixed models (local + remote), local backend is already probed
    // by wait_until_loading_finished; remote backends are trusted until
    // the watchdog marks them unhealthy.
    if (meta->is_remote && !meta->remote_urls.empty()) {
        const bool healthy = probe_child_health(meta->host, meta->port, get_backend_api_key());
        if (!healthy) {
            // Remote server is unreachable — mark it UNLOADED so we don't
            // keep advertising a broken backend.
            std::unique_lock<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                for (auto & b : it->second.backends) {
                    b.status = SERVER_MODEL_STATUS_UNLOADED;
                }
                it->second.meta.status = SERVER_MODEL_STATUS_UNLOADED;
            }
            throw std::runtime_error("remote model name=" + name + " is unreachable — /health probe failed");
        }
    }

    return true;
}

int server_models::select_backend(const std::string & name, bool ignore_capacity) {
    auto it = mapping.find(name);
    if (it == mapping.end()) {
        return -1;
    }
    instance_t & inst = it->second;
    int n = (int)inst.backends.size();
    if (n == 0) {
        return -1;
    }
    // Try priority-1 (remote) first, then priority-0 (local), each in round-robin order
    for (int tier = 1; tier >= 0; --tier) {
        for (int i = 0; i < n; ++i) {
            int idx = (inst.rr_index + i) % n;
            const backend_t & b = inst.backends[idx];
            if (b.priority != tier) continue;
            if (b.status != SERVER_MODEL_STATUS_LOADED || b.health_fail_count >= 3) continue;
            // Skip remote backends that are at their concurrency cap so the caller can
            // fall back to a local switch. Passing ignore_capacity=true disables this,
            // allowing the request to queue on the remote when no switch is possible.
            // When max_concurrency=0 (unlimited), use fallback_concurrency as an
            // effective cap to trigger a local switch once the remote is loaded.
            int effective_cap = b.max_concurrency;
            if (effective_cap == 0 && b.fallback_concurrency > 0) {
                effective_cap = b.fallback_concurrency;
            }
            if (!ignore_capacity && b.is_remote && effective_cap > 0 &&
                b.active_connections >= effective_cap) {
                continue;
            }
            // advance rr_index for next call
            inst.rr_index = (idx + 1) % n;
            return idx;
        }
    }
    return -1;
}

bool server_models::has_healthy_backend(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it == mapping.end()) return false;
    for (const auto & b : it->second.backends) {
        if (b.status == SERVER_MODEL_STATUS_LOADED && b.health_fail_count < 3) {
            return true;
        }
    }
    return false;
}

std::vector<backend_info> server_models::get_backends_info(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    std::vector<backend_info> result;
    if (it == mapping.end()) {
        return result;
    }
    const auto & backends = it->second.backends;
    result.reserve(backends.size());
    for (const auto & b : backends) {
        backend_info info;
        info.is_remote = b.is_remote;
        info.host = b.host;
        info.port = b.port;
        info.is_https = b.is_https;
        info.active_connections = b.active_connections;
        info.health_fail_count = b.health_fail_count;
        info.priority = b.priority;
        info.healthy = b.status == SERVER_MODEL_STATUS_LOADED && b.health_fail_count < 3;
        result.push_back(info);
    }
    return result;
}

bool server_models::escalate_to_local(const std::string & name) {
    // Called when no remote backend is usable (down or at its concurrency cap) for a hybrid
    // (local+remote) model. Spawns the model's local child, evicting an idle local-only LRU
    // model first if models_max requires it. Returns true if a local backend is now (being)
    // loaded; false if switching was unsafe (models_max reached and no idle local-only victim),
    // in which case the caller keeps routing to the remote (queueing).

    // Reap a previously-escalated local child that has since exited (a dead backend lingering
    // with an UNLOADED status). Its managing thread is finished but not yet joined — move it
    // out and join outside the lock to avoid a deadlock with its final update_status().
    std::thread stale_th;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end()) {
            return false;
        }
        bool has_live_local = false, has_dead_local = false;
        for (const auto & b : it->second.backends) {
            if (!b.is_remote && b.subproc) {
                (b.status == SERVER_MODEL_STATUS_UNLOADED ? has_dead_local : has_live_local) = true;
            }
        }
        if (has_live_local) {
            return true; // a local child is already loaded or loading
        }
        if (has_dead_local) {
            stale_th = std::move(it->second.th);
            auto & bks = it->second.backends;
            bks.erase(std::remove_if(bks.begin(), bks.end(), [](const backend_t & b) {
                return !b.is_remote && b.subproc && b.status == SERVER_MODEL_STATUS_UNLOADED;
            }), bks.end());
        }
    }
    if (stale_th.joinable()) {
        stale_th.join();
    }

    // 1) Make room if models_max requires it, but only by evicting an IDLE local-only model.
    //    If none is idle, do not switch (the other model is busy) — caller queues on remote.
    std::string victim;
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (base_params.models_max > 0) {
            size_t count_local = 0;
            int64_t lru_used = INT64_MAX;
            for (const auto & m : mapping) {
                bool spawned_local = false, has_remote = false;
                int active = 0;
                for (const auto & b : m.second.backends) {
                    if (!b.is_remote && b.subproc) spawned_local = true;
                    if (b.is_remote) has_remote = true;
                    active += b.active_connections;
                }
                if (!spawned_local || !m.second.meta.is_running()) {
                    continue;
                }
                count_local++;
                // Only local-only models are evictable victims: evicting a hybrid's local child
                // would need partial-unload handling we deliberately avoid here.
                bool busy = m.second.meta.status == SERVER_MODEL_STATUS_LOADING || active > 0;
                if (m.first != name && !has_remote && !busy && m.second.meta.last_used < lru_used) {
                    victim = m.first;
                    lru_used = m.second.meta.last_used;
                }
            }
            if (count_local >= (size_t) base_params.models_max) {
                if (victim.empty()) {
                    SRV_INF("escalate_to_local: models_max reached and no idle local-only model to evict; "
                            "keeping name=%s on remote\n", name.c_str());
                    return false; // cannot switch safely
                }
            } else {
                victim.clear(); // capacity available; no eviction needed
            }
        }
    }
    if (!victim.empty()) {
        SRV_INF("escalate_to_local: switching name=%s to local; evicting idle LRU name=%s\n", name.c_str(), victim.c_str());
        unload(victim);
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [this, &victim]() {
            auto it = mapping.find(victim);
            return it == mapping.end() || it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED;
        });
    }

    // 2) Spawn the local child and append it as a new backend on the existing instance.
    std::shared_ptr<subprocess_s> child_proc;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end()) {
            return false;
        }
        // Re-check nobody spawned a local child while we released the lock.
        for (const auto & b : it->second.backends) {
            if (!b.is_remote && b.subproc) {
                return true;
            }
        }
        auto & inst = it->second;
        inst.meta.port = get_free_port();
        if (inst.meta.port <= 0) {
            SRV_ERR("escalate_to_local: failed to get a port for name=%s\n", name.c_str());
            return false;
        }
        inst.meta.update_args(ctx_preset, bin_path);

        backend_t b;
        b.is_remote = false;
        b.priority  = 0;
        b.subproc   = std::make_shared<subprocess_s>();
        b.host      = CHILD_ADDR;
        b.port      = inst.meta.port;
        b.status    = SERVER_MODEL_STATUS_LOADING;

        SRV_INF("escalate_to_local: spawning local backend for name=%s on port %d\n", name.c_str(), inst.meta.port);
        std::vector<std::string> child_args = inst.meta.args;
        std::vector<std::string> child_env  = base_env;
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));
        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        if (subprocess_create_ex(argv.data(), options, envp.data(), b.subproc.get()) != 0) {
            SRV_ERR("escalate_to_local: failed to spawn local backend for name=%s\n", name.c_str());
            return false;
        }
        b.stdin_file = subprocess_stdin(b.subproc.get());
        child_proc   = b.subproc;
        int port         = inst.meta.port;
        int stop_timeout = inst.meta.stop_timeout;
        inst.backends.push_back(std::move(b));
        // The remote-registered instance had no managing thread; attach one for the child.
        inst.th = spawn_local_managing_thread(name, child_proc, port, stop_timeout);
    }
    cv.notify_all();

    // 3) Wait until the local child reports ready (LOADED) or fails (UNLOADED).
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [this, &name, &child_proc]() {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                return true;
            }
            for (const auto & b : it->second.backends) {
                if (!b.is_remote && b.subproc.get() == child_proc.get()) {
                    return b.status != SERVER_MODEL_STATUS_LOADING;
                }
            }
            return true; // backend vanished — stop waiting
        });
    }
    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    // Retry once if the backend dies during the connection phase (before any
    // response bytes reach the client). A respawned child restores conversation KV
    // from slot-save-path on startup, so the retried request resumes with context.
    constexpr int MAX_ATTEMPTS = 2;
    for (int attempt = 1; attempt <= MAX_ATTEMPTS; ++attempt) {
        auto meta = get_meta(name);
        if (!meta.has_value()) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        if (!meta->is_running()) {
            throw std::invalid_argument("model name=" + name + " is not running");
        }
        if (update_last_used) {
            std::unique_lock<std::mutex> lk(mutex);
            mapping[name].meta.last_used = ggml_time_ms();
        }
        subprocess_s * cur_proc = nullptr;
        int backend_idx = -1;
        std::string host;
        int port = 0;
        bool is_https = false;
        {
            std::unique_lock<std::mutex> lk(mutex);
            backend_idx = select_backend(name, /*ignore_capacity*/ false);
            if (backend_idx < 0) {
                // No remote backend is usable (down or at its concurrency cap). For a hybrid
                // model with no local child yet, try switching to local (evicting an idle
                // local-only model if models_max requires it).
                bool can_escalate = false;
                auto it = mapping.find(name);
                if (it != mapping.end()) {
                    bool has_remote = false, has_local_backend = false;
                    for (const auto & b : it->second.backends) {
                        if (b.is_remote) has_remote = true;
                        if (!b.is_remote && b.subproc) has_local_backend = true;
                    }
                    can_escalate = has_remote && !has_local_backend && preset_has_local_model(it->second.meta.preset);
                }
                if (can_escalate) {
                    lk.unlock();
                    escalate_to_local(name); // best-effort: no-op if switching is unsafe
                    lk.lock();
                    backend_idx = select_backend(name, false);
                }
                if (backend_idx < 0) {
                    // Switch was unsafe or the remote is merely saturated: queue on a healthy
                    // remote by ignoring its concurrency cap.
                    backend_idx = select_backend(name, /*ignore_capacity*/ true);
                }
                if (backend_idx < 0) {
                    throw std::runtime_error("no healthy backend available for model name=" + name);
                }
            }
            mapping[name].backends[backend_idx].active_connections++;
            cur_proc = mapping[name].backends[backend_idx].subproc.get();
            host = mapping[name].backends[backend_idx].host;
            port = mapping[name].backends[backend_idx].port;
            is_https = mapping[name].backends[backend_idx].is_https;
        }
        SRV_INF("proxying request to model %s on port %d\n", name.c_str(), port);
        std::string proxy_path = req.path;
        if (!req.query_string.empty()) {
            proxy_path += '?' + req.query_string;
        }

        // Build headers map — override Authorization for remote backends
        std::map<std::string, std::string> proxy_headers = req.headers;

        // For remote backends, inject the backend API key to authenticate with the remote server
        // This ensures the remote backend receives a valid API key even if the client didn't send one
        if (mapping[name].backends[backend_idx].is_remote) {
            const auto & b = mapping[name].backends[backend_idx];
            std::string api_key = !b.api_key.empty() ? b.api_key : get_backend_api_key();
            if (!api_key.empty()) {
                proxy_headers["Authorization"] = "Bearer " + api_key;
            }
        }

        auto proxy = std::make_unique<server_http_proxy>(
                method,
                is_https ? "https" : "http",
                host,
                port,
                proxy_path,
                proxy_headers,
                req.body,
                req.files,
                req.should_stop,
                base_params.timeout_read,
                base_params.timeout_write
                );
        proxy->cleanup = [this, name, backend_idx]() {
            std::unique_lock<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end() && (size_t)backend_idx < it->second.backends.size()) {
                if (it->second.backends[backend_idx].active_connections > 0) {
                    it->second.backends[backend_idx].active_connections--;
                }
            }
            cv.notify_all();
        };

        if (attempt < MAX_ATTEMPTS && proxy->is_backend_down()) {
            SRV_WRN("backend for model %s is down (attempt %d/%d) — respawning and retrying\n",
                name.c_str(), attempt, MAX_ATTEMPTS);
            proxy.reset(); // runs cleanup() → active_connections--
            if (cur_proc) {
                mark_backend_dead(name, cur_proc);
            } else {
                mark_backend_dead(name, nullptr, backend_idx);
            }
            ensure_model_ready(name); // respawn + block until ready
            continue;
        }
        return proxy;
    }
    // unreachable: the final attempt always returns its proxy
    throw std::runtime_error("proxy_request: exhausted retries for model " + name);
}

bool server_models::is_child_server() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

std::thread server_models::setup_child_server(const std::function<void(int)> & shutdown_handler, const json & model_info) {
    // send a notification to the router server that a model instance is ready
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s\n", CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
    fprintf(stdout, "%s%s\n", CMD_CHILD_TO_ROUTER_INFO, safe_json_to_str(model_info).c_str());
    fflush(stdout);
    common_log_resume(common_log_main());

    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_models::notify_router_sleeping_state(bool is_sleeping) {
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s\n", is_sleeping ? CMD_CHILD_TO_ROUTER_SLEEP : CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
    common_log_resume(common_log_main());
}


//
// server_models_routes
//

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (models_autoload) {
        models.ensure_model_ready(name);
    } else {
        if (!meta->is_running()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    // even if meta status is LOADED, verify at least one healthy backend exists
    if (!models.has_healthy_backend(name)) {
        res_err(res, format_error_response("no healthy backend available for model", ERROR_TYPE_UNAVAILABLE));
        return false;
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

void server_models_routes::init_routes() {
    this->get_router_health = [this](const server_http_req &) {
        // The router process itself is always alive here, so a plain "ok" would mask a
        // crashed backend. A model configured with load-on-startup=true is meant to be
        // running at all times; if its child died abnormally (is_failed() => UNLOADED
        // with a non-zero/negative exit code, e.g. an OOM kill or GGML_ABORT) report
        // UNHEALTHY (503) so Docker's healthcheck stops reporting the container as
        // healthy. Idle on-demand models, loading/loaded/sleeping models, and models
        // that were deliberately unloaded (graceful exit_code 0) all stay OK.
        auto res = std::make_unique<server_http_res>();
        for (const auto & meta : models.get_all_meta()) {
            std::string val;
            const bool load_on_startup =
                meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) &&
                common_arg_utils::is_truthy(val);
            if (load_on_startup && meta.is_failed()) {
                res_err(res, format_error_response(
                    string_format("model '%s' failed (exit status %d)", meta.name.c_str(), meta.exit_code),
                    ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }
        res_ok(res, {{"status", "ok"}});
        return res;
    };

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",          "router"},
                {"max_instances", params.models_max},
                {"models_autoload", params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",     ui_settings},
                // Deprecated: use ui_settings instead (kept for backward compat)
                {"webui_settings",  webui_settings},
                {"build_info",     std::string(llama_build_info())},
                {"cors_proxy_enabled", params.ui_mcp_proxy || params.webui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, true); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
                if (meta.is_signaled()) {
                    status["exit_signal"] = meta.exit_signal();
                }
            }
            if (!meta.last_error.empty()) {
                status["last_error"] = meta.last_error;
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            // per-backend information
            auto backends_info = models.get_backends_info(meta.name);
            json backends_json = json::array();
            bool any_local = false;
            bool any_remote = false;
            for (const auto & b : backends_info) {
                if (b.is_remote) {
                    any_remote = true;
                } else {
                    any_local = true;
                }
                backends_json.push_back(json {
                    {"type", b.is_remote ? "remote" : "local"},
                    {"host", b.host},
                    {"port", b.port},
                    {"is_https", b.is_https},
                    {"healthy", b.healthy},
                    {"health_fail_count", b.health_fail_count},
                    {"priority", b.priority},
                    {"active_connections", b.active_connections},
                });
            }
            std::string backend_type;
            if (any_local && any_remote) {
                backend_type = "mixed";
            } else if (any_remote) {
                backend_type = "remote";
            } else {
                backend_type = "local";
            }

            json model_info = json {
             {"id",                meta.name},
                {"aliases",         meta.aliases},
                {"tags",            meta.tags},
                {"object",          "model"},    // for OAI-compat
                {"owned_by",        "llamacpp"}, // for OAI-compat
                {"created",         t},          // for OAI-compat
                {"status",          status},
                {"architecture",    architecture},
                {"backend_type",    backend_type},
                {"backends",        backends_json},
                // reasoning/thinking support, computed offline so it is available
                // regardless of whether the model is currently loaded
                {"supports_thinking", meta.supports_thinking},
                {"need_download",   meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };
            if (!meta.chat_template.empty()) {
                model_info["chat_template"] = meta.chat_template;
            }

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running()) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };
}



//
// server_http_proxy
//

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse(body, nullptr, false);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                req.set_header(key, is_default_port ? host : host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->backend_down = std::make_shared<std::atomic<bool>>(false);
    auto backend_down = this->backend_down; // capture into the detached thread
    this->thread = std::thread([cli, pipe, req, backend_down]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            backend_down->store(true); // set BEFORE writing the header chunk
            pipe->write({{}, 502, "", ""}); // header (502 = upstream gone)
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
