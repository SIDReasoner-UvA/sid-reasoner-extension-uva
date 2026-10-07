# Component-Necessity Experiments

> **RQ2: Are RL and explicit reasoning strictly necessary?** Starting from the authors' reasoning-activation (RA)
> checkpoint, we test whether the post-RA gains require the GRPO reinforcement-learning stage
> specifically, or whether they can be recovered by more supervised training or stronger
> decoding; and we test whether the explicit `<think>…</think>` reasoning scaffold is actually
> responsible for the reported improvements.

This suite asks *which ingredients of the SIDReasoner recipe actually do the work*. The
GRPO suite ([`../grpo/`](../grpo/README.md)) asks how sensitive the RL stage is to its reward;
this suite asks the prior question of whether the RL stage and the reasoning scaffold are
needed at all. The two findings are complementary: components shows the RL stage is necessary
but the reasoning scaffold is only conditionally necessary; GRPO shows that *given* the RL
stage, reward edits have small top-10 effects but can improve rank-1 accuracy.

---

## Pipeline context

SIDReasoner trains in three stages:

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Teach the model to emit valid Semantic IDs (SIDs). | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Teach the model to wrap its answer in `<think>…</think>` before the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce the policy so reasoning leads to correct recommendations. | post-GRPO |

Every experiment here **branches from the authors' S2 / RA checkpoint** so that each branch
isolates one continuation choice. The RA checkpoint is the common ancestor; the questions are
"what should come *after* it?" and "does the reasoning behaviour S2 installed matter at
inference?".

Backbone: `Qwen/Qwen3-1.7B` (the original paper's setting). Datasets: `Video Games`,
`Office Products`, `Industrial & Scientific`.

---

## The two questions

| Axis | What it tests | Method |
|------|---------------|--------|
| **(A) Is RL necessary?** | Can the post-GRPO gain be recovered *without* RL — by more SFT, more RA, or a bigger inference search? | Train non-RL continuations from RA; compare to RA + GRPO. Also sweep decoding over the pre-RL checkpoint. |
| **(B) Is explicit reasoning necessary?** | Does removing / changing the `<think>` scaffold hurt? | Re-evaluate fixed checkpoints with the reasoning instruction and scaffold removed or reworded. |

---

## Experiments

### C1 — Can non-RL continuations replace GRPO? (RL necessity)

From the RA checkpoint, train two non-RL continuations and compare them to the actual
post-GRPO checkpoint, with deltas measured relative to RA on the same dataset:

- **Extra SFT** — continue supervised fine-tuning from RA for three epochs.
- **Extra RA** — continue reasoning-activation training from RA for ten epochs.
- **RA + RL** — the real GRPO continuation (the SIDReasoner recipe).

Both controls select the best epoch by validation loss, as confirmed by the authors.
The launcher sets `LOAD_BEST_MODEL_AT_END=True` for extra SFT and extra RA.
Extra SFT retains its epoch checkpoints for diagnostics; the main evaluation uses
its `final_checkpoint`, containing the restored best model. This selection applies to
continuations from an already activated model, not cold-start RA.

All continuations still produce **valid** SIDs, so any gap is not explained by malformed
output — it is an *objective-mismatch* effect: SFT and RA optimize imitation-style losses,
while GRPO directly optimizes a reward tied to recommendation correctness.

**Result (revised Table 2; relative percentage changes versus RA):**

| Dataset | Checkpoint | R@10 | NDCG@10 | ΔR@10 (%) | ΔNDCG@10 (%) |
|---|---|---|---|---|---|
| Games | RA | 0.0876 | 0.0474 | — | — |
| Games | Extra SFT | 0.0511 | 0.0294 | -41.7 | -38.0 |
| Games | Extra RA | 0.0440 | 0.0255 | -49.8 | -46.2 |
| Games | **RA + RL** | **0.1024** | **0.0546** | **+16.9** | **+15.2** |
| Office | RA | 0.1424 | 0.1023 | — | — |
| Office | Extra SFT | 0.1385 | 0.0995 | -2.7 | -2.7 |
| Office | Extra RA | 0.1424 | 0.1028 | +0.0 | +0.5 |
| Office | **RA + RL** | **0.1661** | **0.1202** | **+16.6** | **+17.5** |

**Finding.** GRPO is the **only** continuation that consistently improves over RA. On Games
the non-RL continuations degrade *below* RA; on Office they roughly hold RA but the post-GRPO
checkpoint is clearly best. The RL stage is **not** equivalent to "more training".

The non-RL continuation checkpoints (extra-SFT / extra-RA from RA) are on the Hugging Face Hub:

| Dataset | Extra SFT | Extra RA |
|---|---|---|
| Video Games | [`Video_Games_extra_sft_Qwen3-1.7B`](https://huggingface.co/edovergna/Video_Games_extra_sft_Qwen3-1.7B) | [`Video_Games_extra_ra_Qwen3-1.7B`](https://huggingface.co/edovergna/Video_Games_extra_ra_Qwen3-1.7B) |
| Office Products | [`Office_Products_extra_sft_Qwen3-1.7B`](https://huggingface.co/edovergna/Office_Products_extra_sft_Qwen3-1.7B) | [`Office_Products_extra_ra_Qwen3-1.7B`](https://huggingface.co/edovergna/Office_Products_extra_ra_Qwen3-1.7B) |

### C2 — Inference-time search over the pre-RL checkpoint (rules out a search explanation)

Could the post-GRPO gain just be a wider decoding budget that RL happens to approximate? We
evaluate larger beams and stochastic search over the **Video Games RA (pre-RL)** checkpoint:

- Best pre-RL top-10 remained plain `beam10`: R@10 = 0.0876, NDCG@10 = 0.0474.
- `beam20` $\to$ R@10 0.0842; `beam50` $\to$ R@10 0.0822 (wider beams *hurt*).
- Sampling variants also failed to close the gap; some beam-sampling configs produced many
  invalid predictions.
- Post-GRPO `beam10` stays ahead at R@10 = 0.1024, NDCG@10 = 0.0546.

**Finding.** RL is not merely substituting for a larger inference search budget.

### C3 — Removing the explicit-reasoning scaffold (reasoning necessity)

Strict **no-think** ablation: the evaluator removes both the instruction to reason and the
`<think>…</think>` scaffold, then re-runs the *same* checkpoints. Deltas are computed as
**no-think $-$ thinking** from the displayed scores, so a negative value means
thinking was better. This delta table is supplementary; the post-RL scores match Table 4.

| Dataset | Checkpoint | T R@10 | No-T R@10 | $\Delta$R@10 | T NDCG@10 | No-T NDCG@10 | $\Delta$NDCG@10 |
|---|---|---|---|---|---|---|---|
| Games | RA pre-RL | 0.0876 | 0.1001 | $+0.0125$ | 0.0474 | 0.0503 | $+0.0029$ |
| Games | RL post-GRPO | 0.1024 | 0.0980 | $-0.0044$ | 0.0546 | 0.0515 | $-0.0031$ |
| Office | RA pre-RL | 0.1424 | 0.1473 | $+0.0049$ | 0.1023 | 0.1028 | $+0.0005$ |
| Office | RL post-GRPO | 0.1661 | 0.1247 | $-0.0414$ | 0.1202 | 0.0974 | $-0.0228$ |
| Industrial | RL post-GRPO | 0.1474 | 0.1366 | $-0.0108$ | 0.1014 | 0.0999 | $-0.0015$ |

**Finding.** Reasoning is *strongly* beneficial only for **Office post-RL**, mildly beneficial
for **Industrial** at R@10, and marginal for **Games**. Crucially, **before RL** strict
no-think even *improves* R@10 on both Games and Office. So the scaffold's value is entangled
with the RL stage and the dataset — it is not a universal win.

### C4 — Prompt ablations (reasoning is sensitive to prompt *form*, not just presence)

On the final post-RL checkpoints, compare four prompt conditions under deterministic
constrained beam search (beam 10):

- **Original thinking** — paper-style "reason step by step, then output the SID".
- **Strict no-think** — scaffold removed, decode the SID directly from the first token.
- **No step-by-step** — keep the prompt but drop the explicit step-by-step instruction.
- **Preference summary** — briefly summarize the user's recent preference pattern, then the SID.

| Dataset | Prompt condition | R@10 | NDCG@10 |
|---|---|---|---|
| Video Games | Original thinking | 0.1024 | 0.0546 |
| Video Games | Strict no-think | 0.0980 | 0.0515 |
| Video Games | No step-by-step | 0.0864 | 0.0489 |
| Video Games | **Preference summary** | **0.1055** | **0.0666** |
| Office Products | **Original thinking** | **0.1661** | 0.1202 |
| Office Products | Strict no-think | 0.1247 | 0.0974 |
| Office Products | No step-by-step | 0.1603 | **0.1239** |
| Office Products | Preference summary | 0.1656 | 0.1207 |
| Industrial | Original thinking | 0.1474 | 0.1014 |
| Industrial | Strict no-think | 0.1366 | 0.0999 |
| Industrial | **No step-by-step** | **0.1489** | **0.1079** |
| Industrial | Preference summary | 0.1436 | 0.1024 |

**Finding.** Performance is sensitive to prompt *format*, not simply to the presence of
explicit reasoning: non-step-by-step or preference-summary prompts sometimes **match or beat**
the original reasoning prompt. This weakens any broad claim that step-by-step reasoning is what
produces the gains.

---

## Reproduce

The scripts are part of the SIDReasoner submodule (materialised by
`patches/common/sidreasoner-local.patch`). Run from the submodule root after applying
the patch; environment, data, and base checkpoints are assumed present (see the root
README and `scripts/common/data_prep/`).

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner

# C1: train non-RL continuations from RA, then compare RA / extra-SFT / extra-RA / RL
sbatch final_scripts/rq2_components/train_ra_continuations.sh
sbatch final_scripts/rq2_components/evaluate_nonrl_vs_rl.sh

# C2: supplementary pre-RL inference-search sweep
sbatch final_scripts/rq2_components/evaluate_inference_search.sh

# C3: thinking vs strict no-think
sbatch final_scripts/rq2_components/evaluate_reasoning_ablation.sh

# C4: prompt-form ablations (appendix)
sbatch final_scripts/rq2_components/evaluate_prompt_ablations.sh
```

Per-script options (datasets, checkpoints, decoders) and the assumed data/checkpoint
layout are documented in `final_scripts/rq2_components/README.md` inside the submodule.

---

## Bottom line

- **The RL stage is necessary.** Replacing GRPO with extra SFT, extra RA, or stronger
  inference-time search does **not** recover post-GRPO quality. The likely cause is
  objective mismatch, not malformed output (non-RL continuations still emit valid SIDs).
- **The explicit-reasoning scaffold is only conditionally necessary.** It helps mainly in
  specific dataset/checkpoint settings (notably Office post-RL); before RL it can even hurt,
  and alternative prompt forms can match it.
- **Practical implication.** Spend effort on RL and on initialization quality; treat explicit
  reasoning as a *conditional prompting choice* rather than a universally required component.

**Caveats.** Qwen3-1.7B only; deltas are vs. the RA checkpoint per dataset; the RL-replacement
table covers Games and Office; the reasoning ablations cover Games, Office, and Industrial with
varying coverage per cell. The scaffold ablation tests inference-time formatting, not
the value of reasoning supervision during training. C4 matches revised Table 4; C2 is
a supplementary repository diagnostic, not a table in the revised appendix.
