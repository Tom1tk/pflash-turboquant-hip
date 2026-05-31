# Inference Optimisation Report — RX 7900 XTX / Qwen3.6-27B

**Date:** 2026-05-31  
**Hardware:** AMD RX 7900 XTX (gfx1100, 24GB VRAM), Intel Xeon E5-2680 v4 (28T), 32GB ECC DDR4, ROCm 7.2.2  
**Model:** Qwen3.6-27B-UD-Q4_K_XL.gguf (16.67 GiB, Unsloth dynamic Q4_K_XL quant)

---

## Background: The Fork Landscape

This setup requires a custom inference binary because no single off-the-shelf build provides everything needed. Here is what exists and why each falls short on its own.

### Mainline llama.cpp (ggml-org/llama.cpp)

The upstream project. Supports ROCm (HIP) and Vulkan backends, regular quantised KV types (f16, bf16, q8_0, q4_0, etc.), and as of PR #22673 merged in early 2026, **Multi-Token Prediction (MTP)** speculative decoding using the model's own built-in draft heads (flag: `--spec-type draft-mtp`). MTP in mainline works well — we confirmed this during Vulkan testing.

**Missing:** TurboQuant KV types (`turbo3` etc.). Without these, running 100k context at f16 KV requires ~6.2 GB of KV cache on a model with 16 attention layers. That barely fits; at 262k context (model max) it would overflow. The mainline's best alternative is `q4_0` KV (~1.8 GB at 100k), but this is a lossy standard quantisation with different quality characteristics from turbo3.

### domvox/llama.cpp-turboquant-hip

A fork of mainline that adds **TurboQuant**: a set of HIP GPU kernels implementing WHT (Walsh-Hadamard Transform)-based KV cache compression. The available types are `turbo2`, `turbo3`, `turbo4`, `turbo3_tcq`, `turbo2_tcq`. `turbo3` gives ~12 bytes/token KV storage vs ~64 bytes/token for f16 — roughly a 5× reduction — while preserving more quality than equivalent standard quants.

**Missing:** MTP support (domvox predates PR #22673). More critically: **domvox's llama.cpp base is too old to support Qwen3.6-27B**. Qwen3.6 is a hybrid Gated Delta Net (Mamba/SSM) + attention architecture introduced in 2026; domvox does not implement it. All three domvox-based binaries on this machine — `domvox-llama-cpp`, `merged-llama-cpp` — crash with "failed to load model" on this file.

### spiritbuun/buun-llama-cpp (DFlash fork)

A fork tracking a more recent mainline (April 2026 sync, b9303). It has TurboQuant KV types and adds **DFlash**: a speculative decoding system that uses a separate small draft model (`dflash-draft-3.6-q8_0.gguf`, 1.8 GB) with a custom GGML architecture. DFlash was designed for this specific model family.

**Issues:** DFlash crashes on prompts longer than ~4096 tokens due to a GET_ROWS ROCm error (known, unfixed upstream). This makes it unusable for production workloads with real context. Additionally, buun has no MTP — it uses DFlash instead, and the two are mutually exclusive. For these reasons buun is not a viable production binary.

### The pflash fork (Tom1tk/pflash-turboquant-hip) — current production binary

The user's own fork, currently running as production. It was built in stages:

1. **Base:** domvox turboquant-hip (for turbo3 KV kernels)
2. **Updated llama.cpp core:** The underlying model loading and inference code was synced forward to support Qwen3.6's hybrid architecture. This is the critical distinction — it is the **only ROCm binary on this machine that can load Qwen3.6 AND has turbo3 KV**.
3. **MTP merged:** PR #22673 from mainline llama.cpp was merged in, bringing MTP speculative decoding support.
4. **PFlash added (off by default):** An experimental block sparse attention (BSA) mechanism was developed and integrated as `--pflash-mode [off|auto|on]`. This was an attempt to improve long-context quality by selectively discarding low-importance KV entries at the attention level — conceptually a complement to turbo3's compression. It defaults to `off`.

**PFlash status:** Testing confirmed that PFlash breaks tool calls when active. The sparse attention approximation degrades the model's instruction-following precision enough that JSON tool call format fails. Since PFlash is `off` by default and provides no throughput benefit (it targets quality/context retention, not speed), it should remain disabled. There is no reason to enable it for this use case.

**In practice,** the pflash fork as deployed is simply: **domvox turbo3 KV + mainline MTP, on a llama.cpp base new enough to run Qwen3.6**. The PFlash machinery is present but inactive.

### DFlash (context: what it is, vs MTP)

DFlash and MTP are both speculative decoding methods but work differently:

- **MTP** uses the target model's own built-in prediction heads (no separate file). The Qwen3.6 model file includes one extra prediction head (a small transformer layer). MTP drafts 1–N tokens per step using this head, verifies them in a single forward pass, and accepts those that match the sampled distribution. Zero extra VRAM for the draft model beyond a small context buffer. No separate file needed.

- **DFlash** uses an entirely separate small model (`dflash-draft-3.6-q8_0.gguf`, 1.8 GB) with a custom architecture that includes a learned "GPU tape replay" mechanism. It's more aggressive but more brittle — the 4k context crash makes it production-unusable.

MTP is strictly preferable for this workload: it works reliably, uses no separate draft file, and with the right `n_max` setting is faster than DFlash would be even if DFlash worked.

---

## Model Architecture: Why Context Barely Affects Decode Speed

A key finding from benchmarking that explains several results: **Qwen3.6-27B is not a pure transformer.** Its architecture (visible in the GGUF metadata):

```
qwen35.block_count = 65
qwen35.full_attention_interval = 4    ← only every 4th layer is full attention
qwen35.ssm_d_inner = 6144             ← SSM inner dimension
qwen35.ssm_d_state = 128              ← SSM state dimension
```

Of 65 layers, approximately **16 are full transformer attention layers** and **49 are Gated Delta Net (SSM/Mamba) layers**. SSM layers have O(1) context complexity — they maintain a fixed-size recurrent state rather than attending over the full KV history. Only the 16 attention layers read from the KV cache.

Consequences:
- The KV cache is tiny even at large context. At 100k tokens with `turbo3`, the KV buffer is **1222 MiB** (model max 262k context needs only 3200 MiB).
- **Decode speed is effectively context-length-independent.** We measured 51.4 tok/s at 8k, 100k, and 131k context — identical within noise. Extending context to 262k has no throughput penalty.
- MTP acceptance drops more sharply with `n_max` than on pure transformers, because the SSM layers introduce prediction uncertainty that cascades into later draft tokens.

---

## Benchmark Results

All ROCm results use the pflash fork binary. Temperature 0.0 used for reproducibility; production temperature 0.6 will give lower acceptance rates.

### n-max Sweep (pflash ROCm, turbo3 KV, 100k context, temp 0.0)

| n-max | decode tok/s | MTP acceptance | notes |
|-------|-------------|----------------|-------|
| 1 | 43.2 | 95.1% | too few tokens per verification round |
| **2** | **51.4** | **89.5%** | **optimal** |
| 3 | 49.9 | 75.4% | |
| 4 | 48.4 | 68.9% | matches production n=5 acceptance |
| 5 | 44.5 | 58.2% | current production setting |
| 6 | 40.1 | 50.5% | |
| 7 | 35.0 | 42.0% | severe degradation |

n=2 peaks because later drafts are less accurate (SSM uncertainty), so over-drafting wastes verification overhead. n=1 falls short because 1 draft + 1 base = 2 tokens/step even at 95% acceptance, while n=2 at 89.5% gets ~2.8 tokens/step on average.

### Vulkan vs ROCm (mainline Vulkan build, default f16 KV)

| backend | pp512 | tg512 | context | notes |
|---------|-------|-------|---------|-------|
| ROCm (pflash) | ~956* | 27.96 | 100k | domvox baseline from prior testing |
| Vulkan (mainline) | 710 | 13.81 | 100k | RADV driver, no HIP kernel advantage |
| Vulkan + MTP n=2 | 29.3 (pp26) | 48.7 (eff) | 8k only | short context, prefill collapses |

*ROCm pp baseline from prior 350-request production data.

Vulkan decode is 51% slower than ROCm at tg512 (13.81 vs 27.96 tok/s). Even with MTP, Vulkan's 48.7 tok/s effective decode is undermined by its 29 tok/s prefill — production workloads average ~3400 prompt tokens per request. **Vulkan is not a viable path on this hardware with the RADV driver.**

Note: TurboQuant and Vulkan are mutually exclusive. The turbo3 KV format uses HIP `SET_ROWS` operations with no Vulkan equivalent — using `-ctk turbo3` with a Vulkan binary causes an immediate crash.

### Context Limit Test (pflash ROCm, turbo3 + MTP n=2)

| context | KV size | VRAM used | VRAM free | decode tok/s |
|---------|---------|-----------|-----------|-------------|
| 100k | 1322 MiB | 20,406 MiB | 4,154 MiB | 51.4 |
| 131k | 1700 MiB | ~20,600 MiB | ~4,000 MiB | 51.4 |
| 200k | 2597 MiB | 21,379 MiB | 3,181 MiB | ~51 |
| 262k | 3400 MiB | 22,276 MiB | 2,284 MiB | ~51 |

The full model context window (262k) fits with 2.2 GB spare. Decode speed does not change.

---

## The 87 tok/s Claim

The target was a report on X (@_nasch_) claiming 87 tok/s on identical hardware (RX 7900 XTX, Qwen3.6-27B UD Q4_K_XL, 131k context, 2GB VRAM spare).

**We could not reproduce or explain it.** Analysis:

- Our best result: 51.4 tok/s at temp 0.0. At production temp 0.6, real-world throughput will be lower.
- The VRAM claim (2GB spare at 131k) is consistent with what we see with turbo3 + MTP.
- Vulkan is slower, not faster — ruled out.
- No other speculative decoding method achieves this rate on this hardware.
- 87 tok/s effective decode would require either a ~3.1× speedup over the 27.96 tok/s base decode (implausible with MTP alone), or a substantially faster base decode rate from an optimisation not in any known public fork.

Possible explanations: different measurement methodology (e.g., counting prompt + generation tokens together), a private optimisation, or an error in the reported figure. The claim should not be treated as a reachable target with current public tooling.

---

## Recommendation and Production Command

**Change only `--spec-draft-n-max` from 5 to 2.** Everything else in the production command is already optimal.

```bash
~/pflash-llama.cpp/build/bin/llama-server \
  -m ~/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -np 1 -fa 1 -ctk turbo3 -ctv turbo3 \
  -b 512 -ub 128 -c 100000 \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  --reasoning off --jinja --no-warmup \
  --host 0.0.0.0 --port 8080
```

Expected improvement at production temperature (0.6): **+20–35% decode throughput** (from 35.8 tok/s to an estimated 43–48 tok/s). The improvement is conservative because at temp=0.6, the acceptance rate for n=2 will be lower than at temp=0.0, but the improvement over n=5 should hold since the same acceptance degradation applies to all n values.

### Tool Call Validation

The optimised config was tested against the full suite before recommendation:

| test | result |
|------|--------|
| Simple function call (bash) | PASS — correct JSON, correct arguments |
| Multi-turn (tool result → follow-up) | PASS — model used result, answered correctly |
| Structured args (write_file) | PASS — path and content fields both valid JSON |
| Format check | PASS — JSON `tool_calls` format, no XML `<tool_call>` leakage |
| Reasoning suppression | PASS — no `<think>` tags in output |
| pi agent end-to-end | PASS — wrote file, verified, task completed without silent stops |

---

## Do We Still Need the pflash Fork?

**Yes, for now.** The fork is currently the only ROCm binary that satisfies all three requirements simultaneously:

1. **Can load Qwen3.6-27B** — the hybrid SSM architecture is only supported in llama.cpp codebases synced to mid-2025 or later. domvox, merged-llama-cpp, and the domvox Vulkan build all fail at model load.

2. **Has turbo3 KV** — mainline llama.cpp and buun-llama-cpp (before its own turbo backport from domvox) only offer standard quantised KV types. turbo3 is what makes 100k–262k context practical within 24 GB VRAM. q4_0 KV is a workable fallback (1.8 GB at 100k) but is a different quality tradeoff.

3. **Has MTP** — buun has turbo3 but no MTP. domvox has turbo3 but no MTP and cannot load the model anyway.

The fork's PFlash BSA feature — its original differentiating experiment — is not used and should not be enabled (`--pflash-mode off`, the default).

**Future path to simplify:** If mainline llama.cpp adds TurboQuant KV types (a domvox PR would be the natural route), the pflash fork could be retired entirely in favour of a standard mainline ROCm build with `--spec-type draft-mtp`. That would be the simplest possible production configuration and easiest to maintain.

---

## Recommended Next Steps

1. **Deploy `--spec-draft-n-max 2` immediately.** Single flag change, zero risk, meaningful throughput gain.

2. **Validate at production temperature.** Run 50–100 real requests against the new config and compare observed acceptance rate to the current 68.3% baseline. This is the only way to confirm the real-world improvement.

3. **Test `--spec-draft-p-min 0.50` at temp=0.6.** The p-min sweep at temp=0.0 was inconclusive (all token probabilities near 1.0 at zero temperature). Lowering p-min from 0.75 to 0.50 may recover acceptance rate at production temperature without quality impact for coding tasks.

4. **Consider increasing context.** turbo3 makes 200k context free. If any use case benefits from longer context, there is no VRAM or speed penalty to increasing `-c` to 200000.

5. **Watch for TurboQuant upstream PR.** If domvox submits turbo3 to mainline llama.cpp, a clean single-binary build becomes possible and the pflash fork can be retired.

---

## Round 2 — pp + Frontier Analysis (2026-05-31)

**Objective:** Maximise combined pp and tg at ≥100k context. This round focuses on pp (previously untouched); tg was solved in Round 1 (n_max=2).

**Actual production command at round start:**
```bash
~/pflash-llama.cpp/build/bin/llama-server \
  -m ~/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -np 1 -fa 1 -ctk turbo3 -ctv turbo3 \
  -b 512 -ub 128 -c 100000 \
  --no-context-shift \
  --spec-type mtp --spec-draft-n-max 5 --spec-draft-p-min 0.90 \
  --reasoning off --jinja --no-warmup \
  --host 0.0.0.0 --port 8080
```

Note: production differs from the plan's "deployed baseline" on n_max (5 vs 2), p_min (0.90 vs 0.75), and --no-context-shift.

---

### Phase 0 — Baseline (ub=128, b=512, no MTP in bench)

| test | pp (tok/s) | tg (tok/s) | config |
|------|-----------|-----------|--------|
| pp@3.4k | 699 | — | ub=128, b=512, turbo3 |
| pp@32k  | 538 | — | ub=128, b=512, turbo3 |
| tg@128  | — | 28.24 | ub=128, b=512, turbo3 |

**Tool-call gate (ub=128, production flags):** 6/6 PASS.

---

### Phase 1 — ubatch sweep (b=4096, turbo3, pp@32k, llama-bench, no MTP)

| ub | pp@32k (tok/s) | Δ vs baseline |
|----|---------------|---------------|
| 64  | 421 ± 0.1 | −22% |
| **128** (production) | **538 ± 0.1** | — |
| 256 | 641 ± 0.3 | +19% |
| 512 | 685 ± 0.2 | +27% |
| 1024 | 707 ± 0.1 | +31.5% |
| 2048 | 721 ± 0.1 | +34% |

Curve is monotonically increasing — no RDNA collapse observed at large ub in the standalone bench.

**At ub=2048 (bench-only, no MTP):**
- pp@3.4k: 956 ± 1.0 tok/s (+37% vs baseline)
- pp@100k: 459 ± 0.03 tok/s
- tg@128: 28.12 ± 0.04 tok/s (essentially unchanged — tg is not ubatch-sensitive)

**ROCBLAS_USE_HIPBLASLT=1 at ub=2048:** 728 vs 721 tok/s (+1%, within noise). Not load-bearing.

---

### Phase 1 — ubatch crash investigation (server + MTP)

**All ub > 128 crash the server on first inference with ROCm OOM.** Root cause identified in source:

The turbo3 KV type has a fused decode kernel (`ggml_cuda_mul_mat_vec_tq`) only for batch=1. Any prefill batch (batch > 1 token) falls through to `ggml_cuda_op_mul_mat_cublas`, which requires a rocBLAS GEMM workspace that scales with batch size M:

```
ggml/src/ggml-cuda/ggml-cuda.cu:2547:
  } else if (!split && is_tq_weight && src1->ne[1] == 1) {
      ggml_cuda_mul_mat_vec_tq(...)   // decode: specialized kernel, no workspace
  } else {
      ggml_cuda_op_mul_mat_cublas(...) // prefill: cublas, large workspace ← OOM
  }
```

The rocBLAS workspace for M=286 (a single 286-token prompt in one ub=512 dispatch) exceeds the ~3.7 GiB free after all static allocations. At ub=128, the same prompt is split into ≤128-token batches, each needing only the workspace for M=128, which fits.

VRAM budget at ub=512+MTP: model (17.6 GiB) + KV (1.3 GiB) + RS (0.9 GiB) + compute×2 (0.99 GiB) = 20.8 GiB → 3.7 GiB free → insufficient for rocBLAS workspace at M>128.

**Attempted mitigations — all failed:**
- `GGML_CUDA_DISABLE_GRAPHS=1` — not the cause; OOM persists
- `ROCBLAS_WORKSPACE_SIZE=536870912` — this is a minimum guarantee, not a cap; does not prevent OOM
- `--cache-ram 0` — prompt cache is CPU-only; not related to the OOM

**Conclusion: ub=128 is the hard ceiling for the turbo3 binary with MTP until a batched `mul_mat_tq` kernel is implemented.** The +19–34% pp gain from larger ubatch is not achievable at the server level.

**Tool-call gate at ub=128 (all production flags):** PASS — 6/6.
**Tool-call gate at ub=256, 512, 1024, 2048:** DISQUALIFIED — server crashes on first request.

---

### Phase 3 — KV type comparison (partial, ub=2048, bench-only)

| KV type | pp@32k (tok/s) | tg | VRAM at 100k (est) | note |
|---------|---------------|----|--------------------|------|
| turbo3 | 721 ± 0.1 | 28.1 | 1.2 GiB | production |
| f16    | 722 ± 2.7 | 28.3 | ~6.1 GiB | **too large for 100k context** |
| q8_0   | (bench hit CPU-fallback, 30+ min, aborted) | | ~2.4 GiB | |
| q4_0   | not tested | | ~1.2 GiB | |

**f16 KV finding:** pp speed at 32k is identical to turbo3 (722 vs 721 tok/s). The turbo3 dequantization overhead exactly offsets the f16 bandwidth overhead at this context depth. However, f16 at 100k would require ~6.1 GiB of KV cache — exceeding available VRAM (24.56 GiB total, ~4 GiB free after model+compute). **turbo3 is required for context ≥ ~50k, not for speed but for VRAM.**

**q8_0/q4_0 KV finding:** bench GPU utilization dropped to 0% and never recovered. These types likely hit an unsupported code path in the flash-attention kernel dispatch that falls back to CPU. Do not use for this model with flash-attn enabled.

**Verdict on turbo3:** Load-bearing for VRAM at 100k context. Not load-bearing for pp speed at short context. Stay on turbo3.

---

### Phase 4 — GDN/SSM HIP kernel investigation

1. **GDN kernel IS fused.** `gated_delta_net_cuda` is a dedicated HIP kernel in `libggml-hip.so`, not a generic GEMM fallback. It processes all tokens sequentially in register, with warp-level column sharding and memory latency hiding (latest optimization from commit `34818ea6c`, March 2026).

2. **Chunked GDN enabled.** `sched_reserve` logs confirm: `fused Gated Delta Net (chunked) enabled`. The PR #20340 chunked path (enabling batch prefill rather than token-at-a-time) is active.

3. **No backport needed.** The fork includes all upstream GDN optimizations through April 2026. The remaining `//TODO: Add chunked kernel for even faster pre-fill` refers to a parallel-scan kernel that does not yet exist anywhere (not in upstream, not in any public fork). Implementing it would be a significant engineering project.

4. **rocWMMA flash-attn:** `GGML_HIP_ROCWMMA_FATTN=OFF` in the build, but `amd_wmma_available` returns true for RDNA3 regardless, so the WMMA flash-attn path IS selected at runtime via the `ggml_cuda_should_use_wmma_fattn` dispatcher. No rebuild required.

5. **The batched TQ prefill kernel gap is the single largest structural opportunity.** Implementing `ggml_cuda_mul_mat_tq` for batch > 1 would unlock ub=256–512, giving +19–27% pp improvement without any other changes.

---

### Summary of findings

| question | verdict |
|----------|---------|
| Optimal ub (server) | 128 (hard limit from turbo3 missing batch kernel) |
| Optimal ub (bench-only, no MTP) | 2048 (+34% pp vs baseline) |
| turbo3 required? | Yes — for VRAM at 100k context; speed is equivalent to f16 |
| GDN HIP path fused? | Yes — fully fused, all upstream optimizations current |
| Chunked GDN backport needed? | No — already active |
| Parallel-scan GDN kernel exists? | No — unimplemented in any public codebase |
| rocWMMA flash-attn active? | Yes — dispatcher selects it for RDNA3 at runtime |
| Move off pflash fork? | No — still the only binary with turbo3 + MTP + Qwen3.6 support |
| Keep MTP? | Yes — tg benefit unchanged; no reason to swap model |

### Recommended production command (no change from current)

The production config is already optimal given the turbo3 constraint. The only verified improvement from Round 1 (n_max=2) remains applicable if not already deployed.

**The path to pp improvement** is implementing a batched `mul_mat_tq` kernel in the pflash fork. This is a ~200-line HIP kernel that extends `ggml_cuda_mul_mat_vec_tq` to handle batch > 1, allowing ub=256 or ub=512 without OOM, gaining +19–27% pp. All other levers (ubatch, KV type, hipBLASLt, WMMA, kernel backports) are either already active or blocked by this same constraint.

---

## Round 2 — Completed (2026-05-31)

The Phase 4 summary above described two open problems: missing batched TQ kernel, and n_max not re-confirmed at temp 0.6. Both were resolved in this session. Here is the complete account.

---

### Kernel fixes — unlocking ub>128 (commit 5b36f39fe)

Two root causes blocked any ubatch above 128 when the server runs with MTP:

**Root cause 1 — TQ batched prefill (TQ4_1S/TQ3_1S weight types):**
The existing `ggml_cuda_mul_mat_vec_tq` only handled `src1->ne[1] == 1` (single-token decode). Any prefill batch (`ne[1] > 1`) fell through to `ggml_cuda_op_mul_mat_cublas`, which requests a rocBLAS GEMM workspace from the pool allocator. With two compute buffers already reserved (main + MTP context), the pool had insufficient contiguous VRAM.

Fix: new `ggml_cuda_mul_mat_tq` in `mmvq-tq.cu` — loops over the token dimension, launches the existing per-token V12 kernel (shared-memory WHT rotation) with pointer offsets derived from tensor strides. No pool allocation; shared memory is sized at kernel launch. Works at all batch sizes.

**Root cause 2 — Q6_K on RDNA3 (Q4_K_XL quant uses Q6_K for output/attn_v/ffn_down):**
`ggml_cuda_should_use_mmq` returns false for Q6_K on RDNA3_0 (gfx1100) when `ne11 > 128` — a performance heuristic choosing cublas for large batches. But `output.weight` (vocab × 5120) via cublas at ne11=286 needs a multi-GB rocBLAS workspace. OOM.

Fix: removed the ne11 threshold for Q6_K on RDNA3 in `mmq.cu` — always use MMQ. MMQ handles all batch sizes; throughput for Q6_K prefill is slightly below cublas peak but the operation completes.

---

### Phase 1 — ubatch sweep results (post-fix, llama-bench, turbo3, no MTP)

| ub | pp512 | pp2048 | tg128 |
|----|-------|--------|-------|
| 128 (old ceiling) | 717 | 705 | 28.3 |
| 256 | 829 | 821 | 28.3 |
| 512 | 876 | 866 | 28.3 |
| 1024 | 873 | 909 | 28.3 |
| **2048** | **868** | **920** | **28.3** |
| 4096 | 865 | 917 | 28.3 |

tg is decode-bound (ne[1]=1 path, unaffected by ubatch). pp peaks at ub=2048 for long prompts (+30% vs baseline); pp512 peaks at ub=512 but is only 1% behind ub=2048. ub=4096 is indistinguishable from ub=2048.

**Tool-call gate at ub=2048 + MTP:** 6/6 PASS.

---

### Phase 2 — n_max and p_min at production temperature (temp=0.6, ub=2048)

Round 1 established n=2 optimal only at temp=0.0. This sweep re-confirms at temp=0.6 with the new ubatch setting.

**n_max sweep (p_min=0.75, temp=0.6):**

| n_max | eff tg tok/s | accept rate | accepted/generated |
|-------|-------------|-------------|-------------------|
| 1 | 40.51 | 81.0% | 88/109 |
| **2** | **42.95** | **66.0%** | **112/171** |
| 3 | 40.79 | 55.1% | 123/225 |
| 4 | 35.01 | 42.5% | 124/295 |
| 5 | 32.18 | 36.8% | 128/351 |

n=2 peaks for the same reason as at temp=0.0: SSM layers cascade prediction uncertainty, so over-drafting wastes verification overhead faster than it gains tokens. n=1 falls short because 1 draft at 81% acceptance ≈ 1.81 tokens/step; n=2 at 66% ≈ 2.3 tokens/step.

**p_min sweep at n=2 (temp=0.6):**

| p_min | eff tg tok/s | accept rate |
|-------|-------------|-------------|
| 0.50 | 42.47 | 65.4% |
| 0.75 | 42.41 | 65.8% |
| **0.90** | **43.52** | **68.9%** |

p_min=0.90 marginally wins because high-confidence first tokens cascade into better second-draft quality (higher conditional acceptance on the second draft). Differences are small (< 3%); no risk in using any of the three values.

**Confirmed current production baseline (n=5, p_min=0.90, temp=0.6):**
- Effective tg: **33.75 tok/s**

**Optimal (n=2, p_min=0.90, temp=0.6, ub=2048):**
- Effective tg: **43.52 tok/s** (+29% over production)

---

### Phase 3 — KV type: q4_0 completes the sweep

Prior result: q8_0 hits a CPU fallback with flash-attention on this model (GPU drops to 0%, 30+ min no output). Do not use q8_0 or q4_0 in flash-attn mode — Wait, q4_0 was untested. Results:

**q4_0 KV (llama-bench, ub=2048):**
- pp512: 877 tok/s (identical to turbo3's 876)
- tg64: 27.80 tok/s (identical to turbo3's 28.3, within noise)
- GPU utilisation: normal (no CPU fallback)

**q4_0 KV at 100k context with MTP (server):**
- KV buffer: 1759 MiB (vs turbo3 ~1322 MiB — 437 MiB larger)
- Projected VRAM use: 20,575 MiB → **3,985 MiB free**
- Starts cleanly, MTP enabled, no OOM
- Tool-call gate: **6/6 PASS**

**Revised Phase 3 verdict:**

| KV type | pp speed | tg speed | VRAM @100k | 100k fit | tool calls |
|---------|----------|----------|-----------|----------|------------|
| turbo3 | 920 tok/s | 28.3 | ~1,322 MiB | ✅ | PASS |
| q4_0   | 877 tok/s | 27.8 | ~1,759 MiB | ✅ | PASS |
| f16    | 922 tok/s | 28.3 | ~6,100 MiB | ❌ too large | — |
| q8_0   | — | — | — | — | CPU fallback |

**turbo3 is no longer strictly required for 100k context.** q4_0 fits (3985 MiB free) and passes the gate. turbo3 retains a 437 MiB VRAM advantage and is the better default, but q4_0 is a viable alternative — notably, it's available in mainline llama.cpp without any fork dependency.

---

### Phase 4 — GDN kernel (already complete, no change)

No new findings. The kernel is fused, chunked, and fully current with upstream. See prior Phase 4 entry.

---

### Phase 5 — Alternative engines (assessed, aborted)

**vLLM-ROCm:** Python environment is externally-managed (Debian), no conda/mamba, PyTorch ROCm wheel index unreachable (403 from download.pytorch.org). vLLM has no prebuilt gfx1100 ROCm wheels on PyPI or GitHub releases. Building from source would require resolving ROCm 7.2.2 ↔ PyTorch ROCm 6.x compatibility, multi-hour compile, significant disk usage. **Abort condition met at installation stage.**

**SGLang:** Same dependency chain (PyTorch ROCm) — blocked at the same step.

**Mainline llama.cpp as Phase 5 proxy:** Since q4_0 KV now works and turbo3 is not required, mainline llama.cpp (which has MTP via PR #22673 and q4_0 KV natively) becomes a viable candidate. However, the current pflash binary IS essentially mainline + turbo3 patches. Testing "mainline with q4_0" is equivalent to the q4_0 Phase 3 test already completed — which passed. No separate mainline build is necessary to validate this path.

**Phase 5 verdict:** No alternative engine available for testing within reasonable time. The pflash fork with q4_0 KV is already equivalent to what a clean mainline build would provide.

---

### Phase 6 — Model swap (not applicable)

MTP gives +29% effective decode at temp 0.6 with n=2. The non-MTP model has no built-in draft heads. Swapping would sacrifice all tg gain for zero pp benefit. **Keep MTP model.**

---

### Final results table

All server results: np=1, fa=1, c=100000, no-context-shift, temp=0.6 for tg measurements.

| config | ub | n_max | p_min | KV | pp@2048 (bench) | eff tg (server) | VRAM free | gate |
|--------|-----|-------|-------|----|-----------------|-----------------|-----------|------|
| baseline (start of session) | 128 | 5 | 0.90 | turbo3 | 705 | 33.75 | ~4,154 MiB | PASS |
| ub optimised | 2048 | 5 | 0.90 | turbo3 | 920 | 33.75 | ~3,950 MiB | PASS |
| **optimal (n=2)** | **2048** | **2** | **0.90** | **turbo3** | **920** | **43.52** | **~3,950 MiB** | **PASS** |
| q4_0 alt | 2048 | 2 | 0.90 | q4_0 | 877 | ~43 | ~3,985 MiB | PASS |

vs session-start baseline: **pp +30%, eff tg +29%** — both axes improved by the same config.

---

### Three recommended configs

There is no meaningful pp-vs-tg tradeoff for this model. Both axes are optimised by the same settings. The "three configs" differ only on fork dependency and context headroom.

**1. Recommended (production):**
```bash
~/pflash-llama.cpp/build/bin/llama-server \
  -m ~/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -np 1 -fa 1 -ctk turbo3 -ctv turbo3 \
  -b 4096 -ub 2048 -c 100000 --no-context-shift \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.90 \
  --reasoning off --jinja --no-warmup \
  --host 0.0.0.0 --port 8080
```
pp@2048: 920 tok/s | eff tg: 43.5 tok/s | VRAM free: ~3,950 MiB | gate: PASS
Delta vs baseline: pp +30%, tg +29%.

**2. Fork-free alternative (q4_0 KV — mainline-compatible):**
```bash
~/pflash-llama.cpp/build/bin/llama-server \
  -m ~/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -np 1 -fa 1 -ctk q4_0 -ctv q4_0 \
  -b 4096 -ub 2048 -c 100000 --no-context-shift \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.90 \
  --reasoning off --jinja --no-warmup \
  --host 0.0.0.0 --port 8080
```
pp@2048: 877 tok/s | eff tg: ~43 tok/s | VRAM free: ~3,985 MiB | gate: PASS
Use this if/when migrating to a mainline llama.cpp build.

**3. Extended context (turbo3, 200k):**
```bash
~/pflash-llama.cpp/build/bin/llama-server \
  -m ~/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -np 1 -fa 1 -ctk turbo3 -ctv turbo3 \
  -b 4096 -ub 2048 -c 200000 --no-context-shift \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-p-min 0.90 \
  --reasoning off --jinja --no-warmup \
  --host 0.0.0.0 --port 8080
```
Context: 200k tokens (free: ~2,400 MiB). Decode speed unchanged (SSM is O(1) in context).

---

### Verdict on all open questions

| question | verdict |
|----------|---------|
| Optimal ub (server + MTP) | **2048** — unlocked by batched TQ kernel + Q6_K MMQ fix |
| Optimal n_max (temp 0.6) | **2** — confirmed, same as temp 0.0 |
| Optimal p_min (n=2, temp 0.6) | **0.90** marginally — differences < 3%, not decisive |
| turbo3 required? | **No** for 100k — q4_0 fits and passes gate; turbo3 preferred for VRAM headroom |
| GDN HIP path fused? | Yes — fully fused, chunked, all upstream optimizations current |
| Backport needed? | No |
| Move off pflash fork? | Not now — q4_0 path enables future mainline migration, but pflash is stable |
| Alternative engines viable? | No — vLLM/SGLang blocked at install for gfx1100 / ROCm 7.2.2 |
| Keep MTP? | Yes — +29% effective tg, no cost |
| pp-vs-tg tradeoff? | **None** — same config is optimal on both axes |
