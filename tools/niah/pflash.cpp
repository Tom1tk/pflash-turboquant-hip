#include "pflash.h"
#include "pflash-score.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

struct llama_context * pflash_init_draft(
    const std::string & model_path,
    int32_t n_ctx,
    const std::string & cache_type_k,
    const std::string & cache_type_v,
    int32_t n_gpu_layers,
    bool use_bsa)
{
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    struct llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("pflash: failed to load draft model: %s", model_path.c_str());
        return nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    cparams.offload_kqv = (n_gpu_layers != 0);
    cparams.use_pflash_bsa = use_bsa;

    if (cache_type_k == "q8_0") {
        cparams.type_k = GGML_TYPE_Q8_0;
    } else if (cache_type_k == "f16") {
        cparams.type_k = GGML_TYPE_F16;
    } else {
        cparams.type_k = GGML_TYPE_F32;
    }
    if (cache_type_v == "q8_0") {
        cparams.type_v = GGML_TYPE_Q8_0;
    } else if (cache_type_v == "f16") {
        cparams.type_v = GGML_TYPE_F16;
    } else {
        cparams.type_v = GGML_TYPE_F32;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("pflash: failed to create draft context");
        llama_model_free(model);
        return nullptr;
    }

    return ctx;
}

// Internal: run drafter forward on a single chunk of tokens and read K values.
// Uses explicit absolute positions so RoPE encodes global sequence position,
// making K vectors comparable across windows via cosine similarity.
static bool pflash_process_window(
    struct llama_context * draft_ctx,
    const std::vector<llama_token> & tokens,
    int32_t token_offset,
    int32_t n_window,
    int32_t score_layer,
    int32_t kv_dim,
    std::vector<float> & all_k,
    bool use_bsa,
    int32_t sink_tokens = 2048,
    int32_t recent_tokens = 4096)
{
    llama_memory_clear(llama_get_memory(draft_ctx), true);
    llama_synchronize(draft_ctx);

    int32_t end = std::min(token_offset + n_window, (int32_t)tokens.size());
    int32_t actual = end - token_offset;
    int32_t n_batch = 2048;

    // Compute and set BSA block mask before decode
    if (use_bsa) {
        const int BSA_BLOCK = 128;
        const int n_sink_b  = std::max(1, sink_tokens   / BSA_BLOCK);
        const int n_local_b = std::max(1, recent_tokens / BSA_BLOCK);

        int n_blocks = (actual + BSA_BLOCK - 1) / BSA_BLOCK;
        std::vector<int32_t> selected_blocks;
        selected_blocks.reserve(n_sink_b + n_local_b);

        // Sink blocks
        int sink_end = std::min(n_sink_b, n_blocks);
        for (int b = 0; b < sink_end; b++) {
            selected_blocks.push_back(b);
        }

        // Local window blocks (deduplicate with sink)
        int local_start = std::max(sink_end, n_blocks - n_local_b);
        for (int b = local_start; b < n_blocks; b++) {
            // Dedup: skip if already in sink range (happens when proc_n < actual, small windows)
            if (b >= sink_end) {
                selected_blocks.push_back(b);
            }
        }

        GGML_ASSERT(std::find(selected_blocks.begin(), selected_blocks.end(), 0) != selected_blocks.end());
        llama_set_pflash_bsa_mask(draft_ctx, selected_blocks.data(), (int32_t)selected_blocks.size());
    }

    for (int32_t i = 0; i < actual; i += n_batch) {
        int32_t n = std::min(actual - i, n_batch);
        struct llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int32_t j = 0; j < n; j++) {
            batch.token[j]     = tokens[token_offset + i + j];
            batch.pos[j]       = token_offset + i + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 0;
        }
        int ret = llama_decode(draft_ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) return false;
    }
    llama_synchronize(draft_ctx);

    // Bulk read all K values in one pass (O(n_cells) instead of O(n_cells * n_tokens))
    std::vector<int32_t> positions(actual);
    int32_t n_read = llama_kv_cache_read_k_bulk(draft_ctx, score_layer, 0, &all_k[(size_t)token_offset * kv_dim], positions.data(), actual);
    if (n_read != actual) return false;
    return true;
}

static int32_t compute_incr(
    int32_t b_start, int32_t b_end,
    const std::vector<pflash_span> & anchors,
    const std::vector<pflash_span> & selected)
{
    int32_t incr = b_end - b_start;
    for (const auto & a : anchors) {
        int32_t ov = std::min(b_end, a.end) - std::max(b_start, a.start);
        if (ov > 0) incr -= ov;
    }
    for (const auto & s : selected) {
        int32_t ov = std::min(b_end, s.end) - std::max(b_start, s.start);
        if (ov > 0) incr -= ov;
    }
    return incr;
}

std::vector<pflash_span> pflash_select(
    const std::vector<float> & scores,
    int32_t block_size,
    int32_t n_tokens,
    int32_t sink_tokens,
    int32_t recent_tokens,
    float keep_ratio,
    int32_t min_keep_tokens,
    int32_t coverage_zones,
    const std::vector<std::pair<int32_t,int32_t>> & file_block_ranges,
    int32_t min_blocks_per_file)
{
    std::vector<pflash_span> result;

    if (n_tokens <= 0) return result;

    // Compute target budget
    int32_t target_kept = std::max(min_keep_tokens, (int32_t)std::ceil(keep_ratio * n_tokens));
    target_kept = std::min(target_kept, n_tokens);

    // Early exit: keep everything
    if (target_kept >= n_tokens) {
        result.push_back({0, n_tokens});
        return result;
    }

    // Build mandatory anchors
    std::vector<pflash_span> anchors;

    // Sink tokens (prefix)
    int32_t sink_end = std::min(sink_tokens, n_tokens);
    if (sink_end > 0) {
        anchors.push_back({0, sink_end});
    }

    // Recent tokens (suffix)
    int32_t recent_start = std::max(0, n_tokens - recent_tokens);
    if (recent_start < n_tokens) {
        anchors.push_back({recent_start, n_tokens});
    }

    // Coalesce anchors (merge overlapping)
    if (anchors.size() >= 2) {
        std::sort(anchors.begin(), anchors.end(),
            [](const pflash_span & a, const pflash_span & b) { return a.start < b.start; });
        std::vector<pflash_span> merged;
        merged.push_back(anchors[0]);
        for (size_t i = 1; i < anchors.size(); i++) {
            if (anchors[i].start <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, anchors[i].end);
            } else {
                merged.push_back(anchors[i]);
            }
        }
        anchors = merged;
    }

    // Count anchored tokens
    int32_t anchored = 0;
    for (const auto & a : anchors) {
        anchored += (a.end - a.start);
    }

    // If anchors already meet budget, return them
    if (anchored >= target_kept) {
        return anchors;
    }

    // Middle budget: tokens to select from unanchored regions
    int32_t middle_budget = target_kept - anchored;

    // Build sorted list of blocks by score (descending)
    int32_t n_blocks = (int32_t)scores.size();
    std::vector<std::pair<int32_t, float>> ranked;
    for (int32_t b = 0; b < n_blocks; b++) {
        int32_t b_start = (int32_t)((int64_t)b * block_size);
        int32_t b_end = std::min(b_start + block_size, n_tokens);

        bool fully_covered = false;
        for (const auto & a : anchors) {
            if (b_start >= a.start && b_end <= a.end) {
                fully_covered = true;
                break;
            }
        }
        if (!fully_covered) {
            ranked.push_back({b, scores[b]});
        }
    }

    // Sort by score descending, then by block index ascending (tiebreaker)
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & a, const auto & b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

    // Select blocks greedily
    int32_t middle_kept = 0;
    std::vector<pflash_span> selected;

    // Build score lookup and forced set if any advanced mode is active
    std::unordered_map<int32_t, float> score_map;
    std::unordered_set<int32_t> forced_set;
    if (coverage_zones > 0 || min_blocks_per_file > 0) {
        score_map.reserve(ranked.size());
        for (const auto & r : ranked) score_map[r.first] = r.second;
    }

    // --- Coverage zones: force-select blocks from each zone ---
    if (coverage_zones > 0 && !ranked.empty()) {
        std::vector<int32_t> middle_idx;
        middle_idx.reserve(ranked.size());
        for (const auto & r : ranked) middle_idx.push_back(r.first);
        std::sort(middle_idx.begin(), middle_idx.end());

        int32_t n_mid = (int32_t)middle_idx.size();
        int32_t n_zones = std::min(coverage_zones, n_mid);
        if (n_zones < 1) n_zones = 1;
        int32_t zone_quota = (n_zones > 0) ? (middle_budget / n_zones) : 0;

        if (zone_quota > 0) {
            for (int32_t z = 0; z < n_zones; z++) {
                int32_t zone_start = (int64_t)z       * n_mid / n_zones;
                int32_t zone_end   = (int64_t)(z + 1) * n_mid / n_zones;
                if (z == n_zones - 1) zone_end = n_mid;

                std::vector<std::pair<float,int32_t>> zone_blocks;
                for (int32_t i = zone_start; i < zone_end; i++) {
                    int32_t b = middle_idx[i];
                    if (forced_set.count(b) == 0) {
                        zone_blocks.push_back({score_map[b], b});
                    }
                }
                std::sort(zone_blocks.begin(), zone_blocks.end(),
                    [](const auto & a, const auto & b) { return a.first > b.first; });

                int32_t taken = 0;
                for (const auto & zb : zone_blocks) {
                    if (taken >= zone_quota || middle_kept >= middle_budget) break;
                    int32_t bi = zb.second;
                    int32_t b_start = (int32_t)((int64_t)bi * block_size);
                    int32_t b_end   = std::min(b_start + block_size, n_tokens);
                    int32_t incr = compute_incr(b_start, b_end, anchors, selected);
                    if (incr > 0) {
                        selected.push_back({b_start, b_end});
                        middle_kept += incr;
                        forced_set.insert(bi);
                        taken++;
                    }
                }
            }
        }

        // Ensure score_map is built (needed by greedy loop for zones, and by file-aware below)
        if (score_map.empty()) {
            score_map.reserve(ranked.size());
            for (const auto & r : ranked) score_map[r.first] = r.second;
        }

        LOG_INF("pflash: coverage_zones=%d zone_quota=%d forced=%d/%d",
                n_zones, zone_quota, (int)forced_set.size(), (int)ranked.size());
    }

    // --- File-aware retention: force-select blocks from each file segment ---
    if (min_blocks_per_file > 0 && !file_block_ranges.empty()) {
        // score_map already built above (or in zone pass)
        if (score_map.empty()) {
            score_map.reserve(ranked.size());
            for (const auto & r : ranked) score_map[r.first] = r.second;
        }

        for (const auto & seg : file_block_ranges) {
            int32_t seg_blk_start = seg.first;
            int32_t seg_blk_end   = seg.second;

            std::vector<std::pair<float,int32_t>> seg_blocks;
            for (int32_t b = seg_blk_start; b < seg_blk_end && b < n_blocks; b++) {
                int32_t b_start = (int32_t)((int64_t)b * block_size);
                int32_t b_end   = std::min(b_start + block_size, n_tokens);
                bool in_anchor = false;
                for (const auto & a : anchors) {
                    if (b_start >= a.start && b_end <= a.end) { in_anchor = true; break; }
                }
                if (in_anchor) continue;
                if (forced_set.count(b)) continue;
                seg_blocks.push_back({score_map.count(b) ? score_map[b] : scores[b], b});
            }

            if (seg_blocks.empty()) continue;

            std::sort(seg_blocks.begin(), seg_blocks.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

            int32_t taken = 0;
            for (const auto & sb : seg_blocks) {
                if (taken >= min_blocks_per_file) break;
                int32_t bi = sb.second;
                int32_t b_start = (int32_t)((int64_t)bi * block_size);
                int32_t b_end   = std::min(b_start + block_size, n_tokens);
                int32_t incr = compute_incr(b_start, b_end, anchors, selected);
                if (incr > 0) {
                    selected.push_back({b_start, b_end});
                    middle_kept += incr;
                    forced_set.insert(bi);
                    taken++;
                }
            }
        }

        LOG_INF("pflash: file-aware: %d segs, middle_kept=%d/%d after file forcing",
                (int)file_block_ranges.size(), middle_kept, middle_budget);
    }

    // --- Greedy fill: top-scored remaining blocks ---
    for (const auto & r : ranked) {
        if (middle_kept >= middle_budget) break;

        int32_t b = r.first;
        int32_t b_start = (int32_t)((int64_t)b * block_size);
        int32_t b_end = std::min(b_start + block_size, n_tokens);

        int32_t incr = compute_incr(b_start, b_end, anchors, selected);

        if (incr > 0) {
            selected.push_back({b_start, b_end});
            middle_kept += incr;
        }
    }

    // Combine anchors and selected spans
    result = anchors;
    result.insert(result.end(), selected.begin(), selected.end());

    // Coalesce all spans
    std::sort(result.begin(), result.end(),
        [](const pflash_span & a, const pflash_span & b) { return a.start < b.start; });
    std::vector<pflash_span> coalesced;
    coalesced.push_back(result[0]);
    for (size_t i = 1; i < result.size(); i++) {
        if (result[i].start <= coalesced.back().end) {
            coalesced.back().end = std::max(coalesced.back().end, result[i].end);
        } else {
            coalesced.push_back(result[i]);
        }
    }

    return coalesced;
}

std::vector<llama_token> pflash_gather(
    const std::vector<llama_token> & source,
    const std::vector<pflash_span> & spans)
{
    std::vector<llama_token> result;
    result.reserve(source.size());

    for (const auto & span : spans) {
        for (int32_t i = span.start; i < span.end && i < (int32_t)source.size(); i++) {
            result.push_back(source[i]);
        }
    }

    return result;
}

// Returns (start_block, end_block_exclusive) pairs for each file segment.
// Detects file boundaries by searching for the token sequence encoding
// "// ===== FILE:" in the prompt tokens.
static std::vector<std::pair<int32_t,int32_t>> detect_file_block_ranges(
    const std::vector<llama_token> & tokens,
    const llama_vocab * vocab,
    int32_t block_size)
{
    const std::string MARKER = "// ===== FILE:";
    std::vector<llama_token> marker_toks(MARKER.size() + 4);
    int n = llama_tokenize(vocab, MARKER.c_str(), (int32_t)MARKER.size(),
                           marker_toks.data(), (int32_t)marker_toks.size(),
                           /*add_special=*/false, /*parse_special=*/false);
    if (n <= 0) {
        LOG_WRN("pflash: file-boundary marker failed to tokenize (n=%d)", n);
        return {};
    }
    marker_toks.resize(n);

    // Find all occurrences of marker_toks in tokens.
    std::vector<int32_t> boundary_positions;
    int32_t ntok = (int32_t)tokens.size();
    int32_t mlen = (int32_t)marker_toks.size();
    for (int32_t i = 0; i <= ntok - mlen; i++) {
        bool match = true;
        for (int32_t j = 0; j < mlen; j++) {
            if (tokens[i + j] != marker_toks[j]) { match = false; break; }
        }
        if (match) {
            boundary_positions.push_back(i);
            i += mlen - 1;
        }
    }

    if (boundary_positions.empty()) {
        return {};
    }

    std::vector<std::pair<int32_t,int32_t>> ranges;
    ranges.reserve(boundary_positions.size());
    for (int32_t i = 0; i < (int32_t)boundary_positions.size(); i++) {
        int32_t tok_start = boundary_positions[i];
        int32_t tok_end   = (i + 1 < (int32_t)boundary_positions.size())
                            ? boundary_positions[i + 1]
                            : ntok;
        int32_t blk_start = tok_start / block_size;
        int32_t blk_end   = (tok_end + block_size - 1) / block_size;
        ranges.push_back({blk_start, blk_end});
    }

    LOG_INF("pflash: detected %d file segments via // ===== FILE: marker",
            (int)ranges.size());
    return ranges;
}

pflash_result pflash_compress(
    struct llama_context * draft_ctx,
    const std::vector<llama_token> & tokens,
    const pflash_params & params)
{
    pflash_result res;
    res.source_count = (int32_t)tokens.size();

    // Check threshold
    if ((int32_t)tokens.size() < params.threshold_tokens) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    if (!draft_ctx) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    // Compute adaptive keep ratio
    float keep_ratio = params.keep_ratio;
    if (params.keep_ratio_auto) {
        int32_t n = (int32_t)tokens.size();
        if (n < 25600) {
            keep_ratio = 0.80f;
        } else if (n < 65536) {
            keep_ratio = 0.70f;
        } else {
            keep_ratio = 0.65f;
        }
    }

    // Compute effective window size for auto-mode
    int32_t effective_window = params.window_size;
    if (params.use_bsa && params.bsa_auto_threshold > 0) {
        if ((int32_t)tokens.size() <= params.bsa_auto_threshold) {
            effective_window = 0;  // BSA single-pass
        }
    }

    // Step 1: Run first window through drafter to populate KV cache for probing
    int64_t t0 = ggml_time_us();
    int32_t n_tokens = (int32_t)tokens.size();

    // Min-scoring-budget guard: skip draft when scoring adds little beyond anchors
    int32_t keep_budget = std::max(params.min_keep_tokens, (int32_t)std::ceil(keep_ratio * n_tokens));
    int32_t guard_sink   = params.adaptive_anchors
        ? std::min(std::max((int32_t)(0.10f * (float)n_tokens), 512), 2048) : params.sink_tokens;
    int32_t guard_recent = params.adaptive_anchors
        ? std::min(std::max((int32_t)(0.18f * (float)n_tokens), 1024), 4096) : params.recent_tokens;
    int32_t anchor_budget = guard_sink + guard_recent;
    int32_t scoring_budget = keep_budget - std::min(anchor_budget, keep_budget);
    if (params.min_scoring_budget > 0 && scoring_budget < params.min_scoring_budget) {
        LOG_INF("pflash: bypassed — scoring_budget=%d < min_scoring_budget=%d", scoring_budget, params.min_scoring_budget);
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }
    int32_t win = effective_window > 0 ? std::min(effective_window, n_tokens) : n_tokens;
    int32_t probe_n = std::min(win, n_tokens);
    llama_memory_clear(llama_get_memory(draft_ctx), true);
    llama_synchronize(draft_ctx);
    int32_t n_batch = 2048;

    // Set BSA block mask for initial probe window
    if (params.use_bsa) {
        const int BSA_BLOCK = 128;
        const int n_sink_b  = std::max(1, params.sink_tokens   / BSA_BLOCK);
        const int n_local_b = std::max(1, params.recent_tokens / BSA_BLOCK);
        int n_blocks = (probe_n + BSA_BLOCK - 1) / BSA_BLOCK;
        std::vector<int32_t> selected;
        selected.reserve(n_sink_b + n_local_b);
        int sink_end = std::min(n_sink_b, n_blocks);
        for (int b = 0; b < sink_end; b++) selected.push_back(b);
        int local_start = std::max(sink_end, n_blocks - n_local_b);
        for (int b = local_start; b < n_blocks; b++) selected.push_back(b);
        // Defensive: verify block 0 (sink) is always in the selected set
        // n_sink_b >= 1 guarantees this, but assert explicitly for future refactoring safety
        GGML_ASSERT(std::find(selected.begin(), selected.end(), 0) != selected.end());
        llama_set_pflash_bsa_mask(draft_ctx, selected.data(), (int32_t)selected.size());
    }

    for (int32_t i = 0; i < probe_n; i += n_batch) {
        int32_t n = std::min(probe_n - i, n_batch);
        auto batch = llama_batch_get_one(const_cast<llama_token *>(&tokens[i]), n);
        if (llama_decode(draft_ctx, batch) != 0) {
            LOG_ERR("pflash: draft decode failed at position %d", i);
            res.bypassed = true; res.tokens = tokens; return res;
        }
    }
    llama_synchronize(draft_ctx);

    // Step 2: Auto-select scoring layer from populated KV cache
    int32_t score_layer = params.score_layer;
    if (score_layer < 0) {
        score_layer = 0;
        const int32_t n_layer = (int32_t)llama_model_n_layer(llama_get_model(draft_ctx));
        std::vector<float> probe_buf(16384);
        for (int32_t il = 0; il < std::min(n_layer, 32); il++) {
            int32_t n_read = llama_kv_cache_read_k_for_pos(draft_ctx, il, 0, 0, probe_buf.data());
            if (n_read > 0) { score_layer = il; break; }
        }
        LOG_INF("pflash: auto-selected scoring layer %d", score_layer);
    }

    // Step 3: Determine K cache dimension
    std::vector<float> kv_probe(16384);
    int32_t kv_dim = llama_kv_cache_read_k_for_pos(draft_ctx, score_layer, 0, 0, kv_probe.data());
    if (kv_dim <= 0) {
        LOG_ERR("pflash: failed to read K cache at layer %d", score_layer);
        res.bypassed = true; res.tokens = tokens; return res;
    }

    // Step 4: Read K data for all tokens
    std::vector<float> all_k((size_t)n_tokens * kv_dim);

    // Bulk read K values from first window (single O(n_cells) pass)
    {
        std::vector<int32_t> positions(probe_n);
        int32_t n_read = llama_kv_cache_read_k_bulk(draft_ctx, score_layer, 0, all_k.data(), positions.data(), probe_n);
        if (n_read != probe_n) {
            LOG_ERR("pflash: bulk K cache read failed (got %d, expected %d)", n_read, probe_n);
            res.bypassed = true; res.tokens = tokens; return res;
        }

        // Sort by position: bulk read returns cell-index order, we need position order
        bool sorted = true;
        for (int32_t i = 1; i < probe_n; i++) {
            if (positions[i] < positions[i - 1]) { sorted = false; break; }
        }
        if (!sorted) {
            // Reorder all_k[0..probe_n-1] by positions[i]
            std::vector<float> temp((size_t)probe_n * kv_dim);
            for (int32_t i = 0; i < probe_n; i++) {
                int32_t pos = positions[i];
                if (pos >= 0 && pos < probe_n) {
                    memcpy(&temp[(size_t)pos * kv_dim], &all_k[(size_t)i * kv_dim], (size_t)kv_dim * sizeof(float));
                } else {
                    LOG_WRN("pflash: K-reorder row %d has unexpected position %d (expected [0, %d)) — row dropped", i, pos, probe_n);
                }
            }
            memcpy(all_k.data(), temp.data(), (size_t)probe_n * kv_dim * sizeof(float));
        }
    }

    // Process remaining windows
    const bool use_chunked = effective_window > 0 && n_tokens > effective_window;
    if (use_chunked) {
        for (int32_t offset = win; offset < n_tokens; offset += win) {
            int32_t n_win = std::min(win, n_tokens - offset);
            if (!pflash_process_window(draft_ctx, tokens, offset, n_win, score_layer, kv_dim, all_k, params.use_bsa, params.sink_tokens, params.recent_tokens)) {
                LOG_ERR("pflash: window at %d failed", offset);
                res.bypassed = true; res.tokens = tokens; return res;
            }
        }
    }
    // Non-chunked path: probe_n == n_tokens in this mode (win == n_tokens),
    // so all K data was already read above. No remainder branch needed.
    // Note: read_k_data_bulk always scans from cell 0 (no offset parameter),
    // so a second call would return duplicate rows, not [probe_n..n_tokens).
    // The single-bulk-read design avoids this by construction.
    res.draft_us = ggml_time_us() - t0;

    // Step 5: Score blocks (GPU when available and non-chunked; CPU fallback otherwise)
    // GPU scoring reads from cell tensor which is fully populated only for single-pass prefill.
    // In chunked mode the K tensor holds just the last window; fall back to all_k buffer.
    t0 = ggml_time_us();
    const int32_t n_blocks = (n_tokens + params.block_size - 1) / params.block_size;
    std::vector<float> scores(n_blocks);
    bool scored = false;

#if defined(GGML_USE_HIP)
    if (!use_chunked) {
        struct ggml_tensor * k_tensor = llama_kv_cache_get_k_tensor(draft_ctx, score_layer);
        if (k_tensor && k_tensor->data) {
            int32_t n_scored;
            if (params.score_method == PFLASH_SCORE_OBS_ATTN) {
                n_scored = pflash_score_gpu_obs_attn(
                    (const float *)k_tensor->data,
                    n_tokens, kv_dim,
                    params.block_size,
                    params.obs_window_tokens,
                    params.obs_pool_kernel,
                    scores.data());
            } else {
                n_scored = pflash_score_gpu(
                    (const float *)k_tensor->data,
                    n_tokens, kv_dim,
                    params.block_size,
                    scores.data());
            }
            if (n_scored == n_blocks) {
                scored = true;
            }
        }
    }
#endif

    if (!scored) {
        // CPU fallback: score from all_k buffer
        if (params.score_method == PFLASH_SCORE_OBS_ATTN) {
            // CPU obs-attn: proxy_q = mean(all_k[tail window]), then dot(proxy_q, all_k[i]), pool, block-max
            int obs_start = n_tokens - params.obs_window_tokens;
            if (obs_start < 0) obs_start = 0;
            int n_obs = n_tokens - obs_start;

            std::vector<double> proxy_q(kv_dim, 0.0);
            for (int p = obs_start; p < n_tokens; p++) {
                const float * kp = &all_k[(size_t)p * kv_dim];
                for (int d = 0; d < kv_dim; d++) proxy_q[d] += kp[d];
            }
            float inv_obs = 1.0f / (float)n_obs;
            for (int d = 0; d < kv_dim; d++) proxy_q[d] *= inv_obs;

            std::vector<float> tok_scores(n_tokens);
            for (int p = 0; p < n_tokens; p++) {
                const float * kp = &all_k[(size_t)p * kv_dim];
                float dot = 0.0f;
                for (int d = 0; d < kv_dim; d++) dot += (float)proxy_q[d] * kp[d];
                tok_scores[p] = dot;
            }

            int pool_k = params.obs_pool_kernel;
            int pool_half = pool_k / 2;
            for (int b = 0; b < n_blocks; b++) {
                int start = b * params.block_size;
                int end = std::min(start + params.block_size, n_tokens);
                float best = -1e30f;
                for (int p = start; p < end; p++) {
                    int p0 = p - pool_half;
                    int p1 = p + pool_half + 1;
                    if (p0 < 0) p0 = 0;
                    if (p1 > n_tokens) p1 = n_tokens;
                    float local_max = -1e30f;
                    for (int w = p0; w < p1; w++) {
                        if (tok_scores[w] > local_max) local_max = tok_scores[w];
                    }
                    if (local_max > best) best = local_max;
                }
                scores[b] = best;
            }
        } else {
            // CPU centrality fallback
        const float * last_k = &all_k[(size_t)(n_tokens - 1) * kv_dim];
        float last_len = 0.0f;
        for (int32_t i = 0; i < kv_dim; i++) last_len += last_k[i] * last_k[i];
        last_len = std::sqrt(std::max(last_len, 1e-12f));

        std::vector<double> mean_k_buf(kv_dim);
        for (int32_t b = 0; b < n_blocks; b++) {
            int32_t start = (int32_t)((int64_t)b * params.block_size);
            int32_t end = std::min(start + params.block_size, n_tokens);
            std::fill(mean_k_buf.begin(), mean_k_buf.end(), 0.0);
            for (int32_t p = start; p < end; p++) {
                const float * kp = &all_k[(size_t)p * kv_dim];
                for (int32_t i = 0; i < kv_dim; i++) mean_k_buf[i] += kp[i];
            }
            float inv_len = 1.0f / (float)(end - start);
            for (int32_t i = 0; i < kv_dim; i++) mean_k_buf[i] *= inv_len;
            float dot = 0.0f, ml = 0.0f;
            for (int32_t i = 0; i < kv_dim; i++) {
                dot += (float)mean_k_buf[i] * last_k[i];
                ml += (float)(mean_k_buf[i] * mean_k_buf[i]);
            }
            scores[b] = dot / (std::sqrt(std::max(ml, 1e-12f)) * last_len);
        }
        }
    }
    res.score_us = ggml_time_us() - t0;

    // Score-quality probe diagnostic (--pflash-debug-scores)
    if (params.debug_scores) {
        std::fprintf(stderr, "PFLASH-DEBUG: n_tokens=%d kr=%.2f score_method=%s\n",
                n_tokens, keep_ratio,
                params.score_method == PFLASH_SCORE_OBS_ATTN ? "obs-attn" : "centrality");
        // Top-5 blocks
        struct scored_block { int32_t idx; float score; int bucket; };
        std::vector<scored_block> tmp;
        tmp.reserve(n_blocks);
        for (int32_t b = 0; b < n_blocks; b++) {
            int b_start = b * params.block_size;
            int b_end = std::min(b_start + params.block_size, n_tokens);
            int bucket = 2; // mid default
            if (b_end <= std::min(params.sink_tokens, n_tokens)) bucket = 0; // sink
            else if (b_start >= std::max(0, n_tokens - params.recent_tokens)) bucket = 1; // recent
            tmp.push_back({b, scores[b], bucket});
        }
        std::sort(tmp.begin(), tmp.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
        std::fprintf(stderr, "PFLASH-DEBUG: top-5 blocks:\n");
        for (int k = 0; k < 5 && k < (int)tmp.size(); k++) {
            const char *bucket_name = tmp[k].bucket == 0 ? "sink" : (tmp[k].bucket == 1 ? "recent" : "mid");
            std::fprintf(stderr, "  #%d: block=%d score=%.4f zone=%s\n", k+1, tmp[k].idx, tmp[k].score, bucket_name);
        }
        // Score histogram (10 bins)
        if (!scores.empty()) {
            float s_min = scores[0], s_max = scores[0];
            for (auto s : scores) { if (s < s_min) s_min = s; if (s > s_max) s_max = s; }
            int hist[10] = {0};
            for (auto s : scores) {
                int bin = (int)((s - s_min) / std::max(s_max - s_min, 1e-9f) * 9.999f);
                if (bin < 0) bin = 0;
                if (bin > 9) bin = 9;
                hist[bin]++;
            }
            std::fprintf(stderr, "PFLASH-DEBUG: score histogram [%.4f .. %.4f]:\n", s_min, s_max);
            for (int b = 0; b < 10; b++) std::fprintf(stderr, "  bin%1d: %4d\n", b, hist[b]);
        }
        // Mid-zone score range vs anchor-zone score range
        float mid_min = 1e9f, mid_max = -1e9f;
        float anc_min = 1e9f, anc_max = -1e9f;
        for (int32_t b = 0; b < n_blocks; b++) {
            int b_start = b * params.block_size;
            int b_end = std::min(b_start + params.block_size, n_tokens);
            bool in_sink   = b_end <= std::min(params.sink_tokens, n_tokens);
            bool in_recent = b_start >= std::max(0, n_tokens - params.recent_tokens);
            if (in_sink || in_recent) {
                if (scores[b] < anc_min) anc_min = scores[b];
                if (scores[b] > anc_max) anc_max = scores[b];
            } else {
                if (scores[b] < mid_min) mid_min = scores[b];
                if (scores[b] > mid_max) mid_max = scores[b];
            }
        }
        std::fprintf(stderr, "PFLASH-DEBUG: anchor-zone score range [%.4f .. %.4f]\n", anc_min, anc_max);
        std::fprintf(stderr, "PFLASH-DEBUG: mid-zone   score range [%.4f .. %.4f]\n", mid_min, mid_max);
    }

    // Detect file segment block ranges for file-aware retention
    std::vector<std::pair<int32_t,int32_t>> file_block_ranges;
    if (params.min_blocks_per_file > 0) {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(draft_ctx));
        if (vocab) {
            file_block_ranges = detect_file_block_ranges(tokens, vocab, params.block_size);
            if (file_block_ranges.empty()) {
                LOG_WRN("pflash: min_blocks_per_file=%d but no file markers found in context",
                        params.min_blocks_per_file);
            }
        }
    }

    // Step 6: Select spans
    t0 = ggml_time_us();

    // Adaptive anchor sizing (P2): scale sink/recent as fractions of context
    int32_t eff_sink   = params.sink_tokens;
    int32_t eff_recent = params.recent_tokens;
    if (params.adaptive_anchors) {
        eff_sink   = (int32_t)(0.10f * (float)n_tokens);
        eff_sink   = std::min(std::max(eff_sink, 512), 2048);
        eff_recent = (int32_t)(0.18f * (float)n_tokens);
        eff_recent = std::min(std::max(eff_recent, 1024), 4096);
    }

    auto spans = pflash_select(
        scores, params.block_size, n_tokens,
        eff_sink, eff_recent,
        keep_ratio, params.min_keep_tokens,
        params.coverage_zones,
        file_block_ranges,
        params.min_blocks_per_file);
    res.select_us = ggml_time_us() - t0;

    // Step 7: Gather tokens
    t0 = ggml_time_us();
    res.tokens = pflash_gather(tokens, spans);
    res.gather_us = ggml_time_us() - t0;

    res.spans = spans;
    res.kept_count = (int32_t)res.tokens.size();

    // If compression didn't reduce, bypass
    if (res.kept_count >= res.source_count) {
        res.bypassed = true;
        res.tokens = tokens;
        return res;
    }

    fprintf(stderr,
            "PFLASH: source=%d kept=%d ratio=%.1f%% draft=%dms score=%dms select=%dms gather=%dms total=%dms\n",
            res.source_count, res.kept_count,
            100.0f * res.kept_count / res.source_count,
            (int)(res.draft_us / 1000), (int)(res.score_us / 1000),
            (int)(res.select_us / 1000), (int)(res.gather_us / 1000),
            (int)((res.draft_us + res.score_us + res.select_us + res.gather_us) / 1000));

    return res;
}
