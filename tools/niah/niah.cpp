#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "pflash.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#ifndef _WIN32
#include <unistd.h>
#endif

using json = nlohmann::json;

struct niah_params {
    std::string model;
    std::string fixture;
    int32_t max_gen = 64;
    int32_t n_repeat = 1;
    int32_t n_ctx = 0;
    int32_t n_batch = 2048;
    int32_t n_ubatch = 512;
    int32_t n_gpu_layers = -1;
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    bool flash_attn = true;
    bool warmup = true;
    bool verbose = false;
    bool chatml = true;
    bool no_think = true;

    // PFlash options
    std::string draft_model;
    float pflash_keep_ratio = 0.75f;
    int32_t pflash_block_size = 128;
    int32_t pflash_sink_tokens = 2048;
    int32_t pflash_recent_tokens = 4096;
    int32_t pflash_threshold = 8192;
    int32_t pflash_score_layer = -1;
    int32_t pflash_window_size = 4096;
    int32_t pflash_draft_gpu_layers = -1;
    std::string draft_cache_type_k = "f16";
    std::string draft_cache_type_v = "f16";
    bool pflash_bsa = false;
    int32_t pflash_bsa_auto_threshold = 0;
    bool pflash_keep_ratio_auto = false;
    int32_t pflash_min_scoring_budget = 0;
    int32_t pflash_coverage_zones = 0;
    int32_t pflash_min_blocks_per_file = 0;
    std::string pflash_score_method = "centrality"; // centrality | obs-attn
    int32_t pflash_obs_window   = 256;   // observation window tokens
    int32_t pflash_obs_pool      = 5;     // SnapKV pooling kernel size
    bool pflash_adaptive_anchors = false; // adaptive sink/recent sizing
    bool pflash_debug_scores     = false; // dump score histogram
};

struct niah_fixture {
    std::string id;
    std::string type;
    std::string filler_text;
    std::string question;
    std::vector<std::string> expected_substrings;
    size_t min_recovered = 1;
    int64_t context_tokens = 0;
};

struct niah_result {
    bool pass = false;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    int64_t first_token_us = 0;
    int64_t total_us = 0;
    int64_t n_prompt = 0;
    int64_t n_gen = 0;
    int64_t n_recovered = 0;
    std::string answer;
    std::string fixture_md5;

    bool pflash_bypassed = true;
    int64_t pflash_draft_us = 0;
    int64_t pflash_score_us = 0;
    int64_t pflash_select_us = 0;
    int64_t pflash_gather_us = 0;
    int32_t pflash_source_tokens = 0;
    int32_t pflash_kept_tokens = 0;
};

static void print_usage() {
    fprintf(stdout, "usage: llama-niah --model <model> --fixture <fixture.jsonl> [options]\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "options:\n");
    fprintf(stdout, "  -h, --help              show this help message and exit\n");
    fprintf(stdout, "  --model <path>          model file path (GGUF)\n");
    fprintf(stdout, "  --fixture <path>        NIAH fixture file (JSONL)\n");
    fprintf(stdout, "  --max-gen <n>           max tokens to generate (default: 64)\n");
    fprintf(stdout, "  --repeat <n>            repeat each fixture N times (default: 1)\n");
    fprintf(stdout, "  --ctx-size <n>          context size (default: auto from model)\n");
    fprintf(stdout, "  --batch-size <n>        batch size for prefill (default: 2048)\n");
    fprintf(stdout, "  --ubatch-size <n>       micro batch size (default: 512)\n");
    fprintf(stdout, "  --gpu-layers <n>        GPU layers to offload (-1 = all, default: -1)\n");
    fprintf(stdout, "  --cache-type-k <type>   KV cache type for K (default: f16)\n");
    fprintf(stdout, "  --cache-type-v <type>   KV cache type for V (default: f16)\n");
    fprintf(stdout, "  --no-flash-attn         disable flash attention\n");
    fprintf(stdout, "  --no-warmup             skip model warmup\n");
    fprintf(stdout, "  --no-chatml             use raw prompt without chat template\n");
    fprintf(stdout, "  --think                 allow <think> blocks (default: skip them)\n");
    fprintf(stdout, "  --verbose               verbose output\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "PFlash options:\n");
    fprintf(stdout, "  --draft <path>          draft model for PFlash compression\n");
    fprintf(stdout, "  --pflash-keep-ratio <f> keep ratio (default: 0.75)\n");
    fprintf(stdout, "  --pflash-block-size <n> scoring block size (default: 128)\n");
    fprintf(stdout, "  --pflash-sink <n>       sink tokens to keep (default: 2048)\n");
    fprintf(stdout, "  --pflash-recent <n>     recent tokens to keep (default: 4096)\n");
    fprintf(stdout, "  --pflash-threshold <n>  min tokens to apply PFlash (default: 8192)\n");
    fprintf(stdout, "  --pflash-layer <n>      scoring layer index (default: auto)\n");
    fprintf(stdout, "  --pflash-window <n>     chunk window size for drafter (default: 4096, 0=full)\n");
    fprintf(stdout, "  --pflash-draft-gpu-layers <n> GPU layers for draft model (-1 = all, default: -1)\n");
    fprintf(stdout, "  --pflash-bsa            use block-sparse attention in drafter\n");
    fprintf(stdout, "  --pflash-bsa-auto N     auto-select BSA single-pass for n_tokens <= N (default: 0=off)\n");
    fprintf(stdout, "  --pflash-keep-auto      adaptive keep ratio by context size\n");
    fprintf(stdout, "  --pflash-min-score-budget N  skip draft when scoring_budget < N (default: 0=off)\n");
    fprintf(stdout, "  --pflash-coverage-zones N   divide middle context into N equal zones (default: 0=off)\n");
    fprintf(stdout, "  --pflash-min-blocks-per-file N  keep >=N blocks per // ===== FILE: segment (default: 0=off)\n");
    fprintf(stdout, "  --pflash-score-method <m>  scoring method: centrality|obs-attn (default: centrality)\n");
    fprintf(stdout, "  --pflash-obs-window N      observation window tokens for obs-attn scorer (default: 256)\n");
    fprintf(stdout, "  --pflash-obs-pool N        SnapKV pooling kernel size (default: 5)\n");
    fprintf(stdout, "  --pflash-adaptive-anchors  scale sink/recent as fraction of context\n");
    fprintf(stdout, "  --pflash-debug-scores      dump score histogram and mid-zone range\n");
    fprintf(stdout, "  --draft-cache-k <type>  drafter K cache type (default: f16)\n");
    fprintf(stdout, "  --draft-cache-v <type>  drafter V cache type (default: f16)\n");
}

static std::string md5_hex(const std::string & str) {
    // Write content to a temp file then hash it, avoiding shell injection
    char tmp[] = "/tmp/pflash_niah_XXXXXX";
    int fd = mkstemp(tmp);
    if (fd < 0) return "";
    write(fd, str.data(), str.size());
    close(fd);
    std::string cmd = std::string("md5sum '") + tmp + "' 2>/dev/null | cut -d' ' -f1";
    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) { unlink(tmp); return ""; }
    char buf[33] = {0};
    if (fgets(buf, sizeof(buf), fp)) { buf[32] = 0; }
    pclose(fp);
    unlink(tmp);
    // Strip trailing newline if present
    std::string result(buf);
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

static std::vector<niah_fixture> load_fixtures(const std::string &path) {
    std::vector<niah_fixture> fixtures;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("failed to open fixture: %s", path.c_str());
        return fixtures;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            niah_fixture fix;
            fix.id = j.value("id", "");
            fix.type = j.value("type", "");
            fix.filler_text = j.value("filler_text", "");
            std::string ctx_path = j.value("context_file", "");
            if (!ctx_path.empty() && fix.filler_text.empty()) {
                std::ifstream ctx_f(ctx_path);
                if (ctx_f.is_open()) {
                    fix.filler_text = std::string(std::istreambuf_iterator<char>(ctx_f),
                                                  std::istreambuf_iterator<char>());
                } else {
                    LOG_WRN("context_file not found: %s\n", ctx_path.c_str());
                }
            }
            fix.question = j.value("question", "");
            fix.context_tokens = j.value("context_tokens", 0);
            if (j.contains("expected_answer_substrings")) {
                fix.expected_substrings = j["expected_answer_substrings"].get<std::vector<std::string>>();
                fix.min_recovered = j.value("min_recovered", (int64_t)fix.expected_substrings.size());
            } else {
                fix.expected_substrings.push_back(j.value("expected_answer_substring", ""));
                fix.min_recovered = 1;
            }
            if (j.contains("tokens")) {
                fix.filler_text = "";
            }
            fixtures.push_back(fix);
        } catch (const std::exception &e) {
            LOG_WRN("skipping malformed line: %s", e.what());
        }
    }
    return fixtures;
}

static std::vector<llama_token> tokenize_prompt(
    const struct llama_vocab *vocab,
    const std::string &filler,
    const std::string &question,
    bool chatml,
    bool no_think)
{
    std::string prompt;
    if (chatml) {
        prompt = "<|im_start|>user\n" + filler + "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        prompt = filler + "\n\n" + question;
    }

    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.data(), (int32_t)prompt.size(),
                           tokens.data(), (int32_t)tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, prompt.data(), (int32_t)prompt.size(),
                           tokens.data(), (int32_t)tokens.size(), true, false);
    }
    tokens.resize(std::max(0, n));

    if (chatml) {
        bool has_think = false;
        {
            const auto think_toks = common_tokenize(vocab, "</think>", true);
            if (!think_toks.empty()) {
                for (size_t i = 0; i + think_toks.size() <= tokens.size(); i++) {
                    bool match = true;
                    for (size_t j = 0; j < think_toks.size(); j++) {
                        if (tokens[i + j] != think_toks[j]) { match = false; break; }
                    }
                    if (match) { has_think = true; break; }
                }
            }
        }
        if (!has_think && !no_think) {
            auto think_close = common_tokenize(vocab, "</think>\n", false);
            tokens.insert(tokens.end(), think_close.begin(), think_close.end());
        }
    }

    return tokens;
}

static niah_result run_fixture(
    struct llama_context *ctx,
    const struct llama_vocab *vocab,
    struct llama_sampler *smpl,
    const niah_fixture &fix,
    const niah_params &params,
    struct llama_context *draft_ctx)
{
    niah_result res;

    auto tokens = tokenize_prompt(vocab, fix.filler_text, fix.question, params.chatml, params.no_think);
    if (tokens.empty()) {
        LOG_ERR("tokenization failed");
        return res;
    }
    res.n_prompt = (int64_t)tokens.size();

    if (params.verbose) {
        LOG_INF("prompt: %zu tokens", tokens.size());
    }

    // PFlash compression step
    std::vector<llama_token> prefill_tokens;
    if (draft_ctx && !params.draft_model.empty()) {
        pflash_params pparams;
        pparams.draft_model_path = params.draft_model;
        pparams.keep_ratio = params.pflash_keep_ratio;
        pparams.block_size = params.pflash_block_size;
        pparams.sink_tokens = params.pflash_sink_tokens;
        pparams.recent_tokens = params.pflash_recent_tokens;
        pparams.threshold_tokens = params.pflash_threshold;
        pparams.score_layer = params.pflash_score_layer;
        pparams.window_size = params.pflash_window_size;
        pparams.use_bsa = params.pflash_bsa;
        pparams.bsa_auto_threshold = params.pflash_bsa_auto_threshold;
        pparams.keep_ratio_auto = params.pflash_keep_ratio_auto;
        pparams.min_scoring_budget = params.pflash_min_scoring_budget;
        pparams.coverage_zones       = params.pflash_coverage_zones;
        pparams.min_blocks_per_file  = params.pflash_min_blocks_per_file;
        pparams.score_method         = (params.pflash_score_method == "obs-attn")
                                        ? PFLASH_SCORE_OBS_ATTN : PFLASH_SCORE_CENTRALITY;
        pparams.obs_window_tokens    = params.pflash_obs_window;
        pparams.obs_pool_kernel      = params.pflash_obs_pool;
        pparams.adaptive_anchors     = params.pflash_adaptive_anchors;
        pparams.debug_scores         = params.pflash_debug_scores;

        auto presult = pflash_compress(draft_ctx, tokens, pparams);
        res.pflash_bypassed = presult.bypassed;
        res.pflash_draft_us = presult.draft_us;
        res.pflash_score_us = presult.score_us;
        res.pflash_select_us = presult.select_us;
        res.pflash_gather_us = presult.gather_us;
        res.pflash_source_tokens = presult.source_count;
        res.pflash_kept_tokens = presult.kept_count;

        prefill_tokens = presult.tokens;

        if (params.verbose && !presult.bypassed) {
            LOG_INF("pflash: %d -> %d tokens (%.1f%% kept)",
                presult.source_count, presult.kept_count,
                100.0f * presult.kept_count / presult.source_count);
        }
    } else {
        prefill_tokens = tokens;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);

    if (params.warmup && res.n_prompt > 0) {
        int warmup_n = std::min((int)prefill_tokens.size(), params.n_batch);
        auto warmup_tokens = prefill_tokens;
        warmup_tokens.resize(warmup_n);
        auto batch = llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size());
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_synchronize(ctx);
    }

    int64_t t_start = ggml_time_us();

    int n_processed = 0;
    std::vector<llama_token> batch_tokens(params.n_batch);
    while (n_processed < (int)prefill_tokens.size()) {
        int n_tokens = std::min((int)prefill_tokens.size() - n_processed, params.n_batch);
        for (int i = 0; i < n_tokens; i++) {
            batch_tokens[i] = prefill_tokens[n_processed + i];
        }
        auto batch = llama_batch_get_one(batch_tokens.data(), n_tokens);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("llama_decode failed during prefill");
            return res;
        }
        n_processed += n_tokens;
    }
    llama_synchronize(ctx);
    int64_t t_prefill_end = ggml_time_us();

    llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

    int64_t t_first_token = ggml_time_us();

    res.first_token_us = t_first_token - t_start;
    res.prefill_us = t_prefill_end - t_start;

    std::vector<llama_token> generated;
    generated.push_back(new_token);

    for (int i = 1; i < params.max_gen; i++) {
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        auto batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
        llama_synchronize(ctx);
        new_token = llama_sampler_sample(smpl, ctx, -1);
        generated.push_back(new_token);
    }

    int64_t t_end = ggml_time_us();
    res.decode_us = t_end - t_prefill_end;
    res.total_us = t_end - t_start;
    res.n_gen = (int64_t)generated.size();

    res.answer = common_detokenize(ctx, generated, false);

    // Strip <think>...</think> blocks from answer when thinking is disabled
    if (params.no_think) {
        size_t think_end = res.answer.find("</think>");
        if (think_end != std::string::npos) {
            size_t think_start = res.answer.find("<think>");
            if (think_start != std::string::npos && think_start < think_end) {
                size_t start = think_end + 8; // len("</think>")
                while (start < res.answer.size() && res.answer[start] == '\n') start++;
                res.answer = res.answer.substr(start);
            }
        }
    }

    size_t recovered = 0;
    for (const auto &sub : fix.expected_substrings) {
        if (res.answer.find(sub) != std::string::npos) {
            recovered++;
        }
    }
    res.n_recovered = (int64_t)recovered;
    res.pass = (recovered >= fix.min_recovered);

    std::string raw = fix.filler_text + fix.question;
    for (auto &s : fix.expected_substrings) raw += s;
    res.fixture_md5 = md5_hex(raw);

    return res;
}

static void print_result_json(const niah_fixture &fix, const niah_result &res,
                               const niah_params & /*params*/, int run_idx)
{
    json j;
    j["id"] = fix.id;
    j["type"] = fix.type;
    j["context_tokens"] = fix.context_tokens;
    j["prompt_tokens"] = res.n_prompt;
    j["gen_tokens"] = res.n_gen;
    j["prefill_us"] = res.prefill_us;
    j["first_token_us"] = res.first_token_us;
    j["decode_us"] = res.decode_us;
    j["total_us"] = res.total_us;
    j["prefill_ms"] = res.prefill_us / 1000.0;
    j["first_token_ms"] = res.first_token_us / 1000.0;
    j["decode_ms"] = res.decode_us / 1000.0;
    j["ttft_ms"] = res.first_token_us / 1000.0;
    j["total_ms"] = res.total_us / 1000.0;
    j["pflash_bypassed"] = res.pflash_bypassed;
    j["pflash_source_tokens"] = res.pflash_source_tokens;
    j["pflash_kept_tokens"] = res.pflash_kept_tokens;
    j["pflash_draft_ms"] = res.pflash_draft_us / 1000.0;
    j["pflash_score_ms"] = res.pflash_score_us / 1000.0;
    j["pflash_select_ms"] = res.pflash_select_us / 1000.0;
    j["pflash_gather_ms"] = res.pflash_gather_us / 1000.0;
    j["pflash_total_ms"] = (res.pflash_draft_us + res.pflash_score_us + res.pflash_select_us + res.pflash_gather_us) / 1000.0;
    j["pass"] = res.pass;
    j["recovered"] = res.n_recovered;
    j["min_recovered"] = (int64_t)fix.min_recovered;
    j["expected"] = fix.expected_substrings;
    j["run"] = run_idx;
    j["answer"] = res.answer;
    if (res.pass) {
        j["status"] = "PASS";
    } else {
        j["status"] = "FAIL";
    }
    std::cout << j.dump() << std::endl;
}

static void print_result_human(const niah_fixture &fix, const niah_result &res, int /*run_idx*/)
{
    double prefill_ms = res.prefill_us / 1000.0;
    double first_ms = res.first_token_us / 1000.0;
    double decode_ms = res.decode_us / 1000.0;
    double total_ms = res.total_us / 1000.0;
    double prefill_tok_s = res.n_prompt > 0 && prefill_ms > 0
        ? (double)res.n_prompt / (prefill_ms / 1000.0) : 0.0;

    if (!res.pflash_bypassed && res.pflash_source_tokens > 0) {
        double draft_ms = res.pflash_draft_us / 1000.0;
        double score_ms = res.pflash_score_us / 1000.0;
        double pflash_total = draft_ms + score_ms + res.pflash_select_us / 1000.0 + res.pflash_gather_us / 1000.0;
        fprintf(stdout, "  ctx=%-5zu prompt=%-6zu->%-4zu gen=%-3zu | "
                        "draft=%-6.0fms score=%-5.0fms pflash=%-6.0fms | "
                        "prefill=%-8.0fms (%-5.0f t/s) ttft=%-8.0fms "
                        "decode=%-8.0fms total=%-8.0fms | "
                        "recv=%zu/%zu %s\n",
            fix.context_tokens, (size_t)res.n_prompt, (size_t)res.pflash_kept_tokens, (size_t)res.n_gen,
            draft_ms, score_ms, pflash_total,
            prefill_ms, prefill_tok_s,
            first_ms, decode_ms, total_ms,
            (size_t)res.n_recovered, fix.expected_substrings.size(),
            res.pass ? "PASS" : "FAIL");
    } else {
        fprintf(stdout, "  ctx=%-5zu prompt=%-6zu gen=%-3zu | "
                        "prefill=%-8.0fms (%-5.0f t/s) ttft=%-8.0fms "
                        "decode=%-8.0fms total=%-8.0fms | "
                        "recv=%zu/%zu %s\n",
            fix.context_tokens, (size_t)res.n_prompt, (size_t)res.n_gen,
            prefill_ms, prefill_tok_s,
            first_ms, decode_ms, total_ms,
            (size_t)res.n_recovered, fix.expected_substrings.size(),
            res.pass ? "PASS" : "FAIL");
    }
}

int main(int argc, char **argv) {
    niah_params params;
    std::string output_format = "human";

    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); i++) {
        const auto &arg = args[i];
        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < args.size()) {
            params.model = args[++i];
        } else if (arg == "--fixture" && i + 1 < args.size()) {
            params.fixture = args[++i];
        } else if (arg == "--max-gen" && i + 1 < args.size()) {
            params.max_gen = std::stoi(args[++i]);
        } else if (arg == "--repeat" && i + 1 < args.size()) {
            params.n_repeat = std::stoi(args[++i]);
        } else if (arg == "--ctx-size" && i + 1 < args.size()) {
            params.n_ctx = std::stoi(args[++i]);
        } else if (arg == "--batch-size" && i + 1 < args.size()) {
            params.n_batch = std::stoi(args[++i]);
        } else if (arg == "--ubatch-size" && i + 1 < args.size()) {
            params.n_ubatch = std::stoi(args[++i]);
        } else if (arg == "--gpu-layers" && i + 1 < args.size()) {
            params.n_gpu_layers = std::stoi(args[++i]);
        } else if (arg == "--cache-type-k" && i + 1 < args.size()) {
            params.cache_type_k = args[++i];
        } else if (arg == "--cache-type-v" && i + 1 < args.size()) {
            params.cache_type_v = args[++i];
        } else if (arg == "--no-flash-attn") {
            params.flash_attn = false;
        } else if (arg == "--no-warmup") {
            params.warmup = false;
        } else if (arg == "--no-chatml") {
            params.chatml = false;
        } else if (arg == "--think") {
            params.no_think = false;
        } else if (arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--draft" && i + 1 < args.size()) {
            params.draft_model = args[++i];
        } else if (arg == "--pflash-keep-ratio" && i + 1 < args.size()) {
            params.pflash_keep_ratio = std::stof(args[++i]);
        } else if (arg == "--pflash-block-size" && i + 1 < args.size()) {
            params.pflash_block_size = std::stoi(args[++i]);
        } else if (arg == "--pflash-sink" && i + 1 < args.size()) {
            params.pflash_sink_tokens = std::stoi(args[++i]);
        } else if (arg == "--pflash-recent" && i + 1 < args.size()) {
            params.pflash_recent_tokens = std::stoi(args[++i]);
        } else if (arg == "--pflash-threshold" && i + 1 < args.size()) {
            params.pflash_threshold = std::stoi(args[++i]);
        } else if (arg == "--pflash-layer" && i + 1 < args.size()) {
            params.pflash_score_layer = std::stoi(args[++i]);
        } else if (arg == "--pflash-window" && i + 1 < args.size()) {
            params.pflash_window_size = std::stoi(args[++i]);
        } else if (arg == "--pflash-draft-gpu-layers" && i + 1 < args.size()) {
            params.pflash_draft_gpu_layers = std::stoi(args[++i]);
        } else if (arg == "--pflash-bsa") {
            params.pflash_bsa = true;
        } else if (arg == "--pflash-bsa-auto" && i + 1 < args.size()) {
            params.pflash_bsa_auto_threshold = std::stoi(args[++i]);
        } else if (arg == "--pflash-keep-auto") {
            params.pflash_keep_ratio_auto = true;
        } else if (arg == "--pflash-min-score-budget" && i + 1 < args.size()) {
            params.pflash_min_scoring_budget = std::stoi(args[++i]);
        } else if (arg == "--pflash-coverage-zones" && i + 1 < args.size()) {
            params.pflash_coverage_zones = std::stoi(args[++i]);
        } else if (arg == "--pflash-min-blocks-per-file" && i + 1 < args.size()) {
            params.pflash_min_blocks_per_file = std::stoi(args[++i]);
        } else if (arg == "--pflash-score-method" && i + 1 < args.size()) {
            params.pflash_score_method = args[++i];
        } else if (arg == "--pflash-obs-window" && i + 1 < args.size()) {
            params.pflash_obs_window = std::stoi(args[++i]);
        } else if (arg == "--pflash-obs-pool" && i + 1 < args.size()) {
            params.pflash_obs_pool = std::stoi(args[++i]);
        } else if (arg == "--pflash-adaptive-anchors") {
            params.pflash_adaptive_anchors = true;
        } else if (arg == "--pflash-debug-scores") {
            params.pflash_debug_scores = true;
        } else if (arg == "--draft-cache-k" && i + 1 < args.size()) {
            params.draft_cache_type_k = args[++i];
        } else if (arg == "--draft-cache-v" && i + 1 < args.size()) {
            params.draft_cache_type_v = args[++i];
        } else if (arg == "--output" && i + 1 < args.size()) {
            output_format = args[++i];
        } else {
            LOG_ERR("unknown argument: %s", arg.c_str());
            print_usage();
            return 1;
        }
    }

    if (params.model.empty()) {
        LOG_ERR("--model is required");
        return 1;
    }
    if (params.fixture.empty()) {
        LOG_ERR("--fixture is required");
        return 1;
    }

    auto fixtures = load_fixtures(params.fixture);
    if (fixtures.empty()) {
        LOG_ERR("no valid fixtures found in %s", params.fixture.c_str());
        return 1;
    }
    LOG_INF("loaded %zu fixtures from %s", fixtures.size(), params.fixture.c_str());

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model: %s", params.model.c_str());
        llama_backend_free();
        return 1;
    }

    int32_t n_ctx_train = llama_model_n_ctx_train(model);
    int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : std::max(n_ctx_train, 32768);
    LOG_INF("model loaded: ctx_train=%d, using ctx=%d", n_ctx_train, n_ctx);

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = params.n_batch;
    cparams.n_ubatch = params.n_ubatch;
    cparams.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (params.cache_type_k == "turbo3") {
        cparams.type_k = GGML_TYPE_TURBO3_0;
    } else if (params.cache_type_k == "q8_0") {
        cparams.type_k = GGML_TYPE_Q8_0;
    } else {
        cparams.type_k = GGML_TYPE_F16;
    }
    if (params.cache_type_v == "turbo3") {
        cparams.type_v = GGML_TYPE_TURBO3_0;
    } else if (params.cache_type_v == "q8_0") {
        cparams.type_v = GGML_TYPE_Q8_0;
    } else {
        cparams.type_v = GGML_TYPE_F16;
    }

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("failed to create context");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    // Initialize draft model for PFlash if requested
    struct llama_context *draft_ctx = nullptr;
    if (!params.draft_model.empty()) {
        LOG_INF("loading draft model: %s", params.draft_model.c_str());
        draft_ctx = pflash_init_draft(
            params.draft_model, n_ctx,
            params.draft_cache_type_k, params.draft_cache_type_v,
            params.pflash_draft_gpu_layers,
            params.pflash_bsa);
        if (!draft_ctx) {
            LOG_ERR("failed to load draft model, continuing without PFlash");
        } else {
            LOG_INF("draft model loaded successfully");
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int total_pass = 0;
    int total_fail = 0;

    for (const auto &fix : fixtures) {
        if (params.verbose) {
            fprintf(stdout, "\n--- ctx=%zu q=\"%s\" ---\n",
                    fix.context_tokens,
                    fix.question.substr(0, 60).c_str());
        }

        for (int r = 0; r < params.n_repeat; r++) {
            auto res = run_fixture(ctx, vocab, smpl, fix, params, draft_ctx);

            if (output_format == "json") {
                print_result_json(fix, res, params, r);
            } else {
                print_result_human(fix, res, r);
            }

            if (res.pass) total_pass++; else total_fail++;
        }
    }

    fprintf(stdout, "\n--- SUMMARY ---\n");
    fprintf(stdout, "total: %d pass, %d fail, %d/%d\n",
            total_pass, total_fail, total_pass, total_pass + total_fail);

    if (draft_ctx) {
        const struct llama_model *draft_model = llama_get_model(draft_ctx);
        llama_free(draft_ctx);
        llama_model_free(const_cast<struct llama_model *>(draft_model));
    }
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return total_fail > 0 ? 1 : 0;
}
