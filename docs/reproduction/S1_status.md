# S1 reasoning + RL — what happened & where we are

> Historical run notes. For the revised paper interpretation and confirmed setup
> decisions, see [current reproduction docs](README.md) and [alignment audit](../paper-alignment.md).


> **TL;DR**
> - RL was broken (0 reward) → fixed: activation needs **5 epochs**, not 1; the model now actually reasons.
> - Dropping the altered temperature eval — it was only patching the broken model; **using the standard greedy eval** now.
> - Our before-RL is already strong (R@10 **0.137** vs paper's 0.039), so RL won't jump much.
> - RL now learns properly — we **stop at ~300–500 steps** (≈10–16k SBU) instead of the full run (~49k).

---

### The bug chain
1. **Ran 1-epoch activation** (to match the paper's "1 epoch is enough").
2. **GRPO reward = 0** — couldn't learn anything.
3. **Why:** 1 epoch didn't force the think format — the model never emitted `<think>…</think>`, so the reward parser (needs `</think>` + SID) returned 0 every time.
4. **Fix:** the original code already used **5 epochs** for activation (we'd cut it to 1). Restored it → model reasons. *(Also fixed a `load_best_model_at_end` setting that was saving the wrong, non-reasoning checkpoint as "final" — so even the 5-epoch run looked broken until we grabbed the right checkpoint.)*

| activation | P(`<think>`) as 1st token | emits reasoning? |
|---|---|---|
| 1 epoch | 0.001 | ❌ no |
| 5 epochs | 0.90 | ✅ yes |

### Eval — dropping the altered script
The fixed model is **much stronger than the paper before RL**, and **temperature makes no difference** — so the best-of-1 temperature eval is pointless now. **Going back to the standard greedy eval.**

| setting | R@10 | N@10 |
|---|---|---|
| greedy (temp 0) | 0.137 | 0.079 |
| temp 0.6 | 0.137 | 0.077 |
| temp 1.0 | 0.132 | 0.075 |
| **paper S1 before RL** | **0.039** | 0.020 |
| paper S1 after RL | 0.154 | — |

→ Our **before-RL ≈ paper's after-RL**, because we used 5 activation epochs vs their 1. So RL has little headroom — expect a small lift, not the paper's 4× jump.

### RL now actually learns (epoch-5 model)
Reward nonzero (`mean 0.069`, max 1.1), real gradients (`grad_norm 0.34`), short reasoning outputs (~197 tokens vs 1024 before). Pipeline works end-to-end.

### Cost (≈145 s/step, 808 SBU/hr)
Paper shows convergence by ~step 300–500, so we **stop early** instead of running all 1520:

| steps | time | cost (SBU) |
|---|---|---|
| 300 | ~12 h | ~9.8k |
| 500 | ~20 h | ~16k |
| 760 (5 ep) | ~31 h | ~25k |
| 1520 (full) | ~61 h | ~49k |

**Plan:** run, check reward/R@10 every 50 steps, stop at the plateau (~300–500 steps ≈ **10–16k SBU**).

### Cross-check vs authors' configs (from email)

Compared `config.json` / `generation_config.json` the authors shared against our setup:

| item | authors | ours | match |
|---|---|---|---|
| backbone | Qwen3-1.7B (hidden 2048, 28L, 16/8 heads, intermediate 6144) | same | ✅ |
| `tie_word_embeddings` / dtype | true / bfloat16 | true / bf16 | ✅ |
| `vocab_size` (base + SID tokens) | 152269 | 152269 | ✅ exact |
| `generation_config` | Qwen3 defaults (temp 0.6, top_p 0.95, top_k 20) | overridden by eval/RL | n/a |
| **transformers** | **4.57.0** | **4.51.3** | ⚠️ deviation |

**Only deviation: transformers 4.51.3 vs 4.57.0.** Left as-is — the stack is pinned around torch 2.6 / vllm
0.8.5.post1 / flash-attn-for-torch2.6, so bumping is risky. Plausible (minor) contributor to early `<think>`
chat-template differences, but the pipeline works on 4.51.3 and model + SID vocab are an exact match.
