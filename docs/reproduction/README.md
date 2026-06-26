# Reproduction Experiments

> **RQ1: Can we reproduce the main performance trends reported in the original
> SIDReasoner paper?** We re-train Qwen3-1.7B **from scratch** on `Video Games` (S1
> data) through SFT → reasoning-activation and check that it recovers the paper's
> SFT → RA trend.

Result: the SFT → RA trend reproduces on Games — post-SFT `HR@10 0.0387` ≈ the paper's
`0.0414`, and 5-epoch RA lifts it to `0.0602`. Our RA *overshoots* the paper's
1-epoch RA (`0.0414`) because the paper specifies 1 activation epoch while the released
code (and us) run 5 — an epoch-count discrepancy, not a modelling difference. **Games
only**: Office S1 showed an unexplained ~3× inflation and is excluded. Full numbers in
[`../model_size/results_table.md`](../model_size/results_table.md); the debugging trail
(why RL was zero-reward, the `load_best_model_at_end` fix, eval-decoder choice, the
transformers-version deviation) is in [`S1_status.md`](S1_status.md).

---

## R1 — S1 SFT → reasoning-activation reproduction (Games)

Re-train Qwen3-1.7B from scratch on Games S1 data and check that the SFT → RA trend
matches the paper. HR@5/10, NDCG@5/10 — our greedy think-mode constrained beam-10 eval
(rows mirror [`../model_size/results_table.md`](../model_size/results_table.md)):

| Stage | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| SFT (raw align) | 0.0215 | 0.0387 | 0.0129 | 0.0184 |
| RA 5-epoch | 0.0321 | 0.0602 | 0.0187 | 0.0277 |

Paper (Qwen3-1.7B, Games S1), HR@10: SFT/RA `0.0414` → GRPO `0.0741`.

**Finding.** The **SFT → RA trend reproduces on Games**: post-SFT `HR@10 0.0387` ≈ the
paper's `0.0414`, and 5-epoch RA lifts it to `0.0602`. Our RA *overshoots* the paper's
1-epoch RA (`0.0414`) only because the paper specifies **1** activation epoch while the
released code (and us) run **5** — an epoch-count discrepancy, not a modelling difference.
Validated on **Games only**; Office S1 showed an unexplained ~3× inflation and is excluded
(see [`S1_status.md`](S1_status.md)).

---

## Where the code lives

The launch scripts are part of the **SIDReasoner submodule**, materialised by
`patches/common/sidreasoner-local.patch` (same pattern as `rq2_components` /
`rq5_diversity`). After applying the patch they sit under
`baselines/repos/SIDReasoner/final_scripts/rq1_reproduction/`. Run from the submodule
root with `$PYTHON_CMD`; source `final_scripts/snellius_env.sh` first.

| Stage | Script (`final_scripts/rq1_reproduction/`) |
|-------|--------------------------------------------|
| 1.7B S1 alignment SFT | `sft_Qwen3_S1_Games.sh` |
| 1.7B S1 reasoning activation (5-ep) | `sft_reasoning_activation_S1_Games.sh` |
| think-mode eval | `evaluate_Qwen3_think_S1.sh` (+ `_bestof1` for beam-1) |

> The reasoning-activation `eval_loss` **rises on purpose** during this stage (it is
> the direct-SID loss, which worsens as the model learns to think-then-answer). The
> shared `sft_reasoning_activation.py` defaults `load_best_model_at_end=False` so the
> activated final epoch is the checkpoint that is saved — do not "fix" the rising loss.

## Reproduce

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
source final_scripts/snellius_env.sh

sbatch final_scripts/rq1_reproduction/sft_Qwen3_S1_Games.sh                 # SFT  -> HR@10 0.0387
sbatch final_scripts/rq1_reproduction/sft_reasoning_activation_S1_Games.sh  # RA(5-ep) -> HR@10 0.0602
sbatch final_scripts/rq1_reproduction/evaluate_Qwen3_think_S1.sh            # standard greedy think eval
```

Use the **standard greedy think eval** (`evaluate_Qwen3_think_S1.sh`); the temperature
best-of-1 variant was only patching the previously-broken (1-epoch) model and is no
longer needed (see `S1_status.md`).

> **Version note for the write-up:** we ran on transformers 4.51.3 (paper used 4.57.0).
> The model + SID vocab (152269) are an exact match; the version gap is the main
> suspected contributor to the small residual difference from the paper's numbers.

---

## Bottom line

- **The paper's SFT → RA trend reproduces** on Games from scratch (`HR@10 0.0387 → 0.0602`),
  with post-SFT matching the paper (`0.0387` ≈ `0.0414`).
- The RA *overshoot* vs the paper is fully explained by the **1-vs-5 activation-epoch**
  discrepancy in the released code, not a modelling difference.
- This reproduction underpins the scaling comparison in
  [model_size](../model_size/README.md) (the 1.7B Games-S1 rows are shared).

## Caveats

- **Games only** — Office S1 showed an unexplained ~3× inflation vs the paper and is
  excluded.
- Single run; `Qwen3-1.7B`. We ran on **transformers 4.51.3** (paper: 4.57.0) — the model
  + SID vocab (152269) match exactly, so the version gap is the main suspected source of
  the small residual difference.
- The RA `eval_loss` **rises by design** (it is the direct-SID loss);
  `load_best_model_at_end=False` keeps the activated final epoch — do not "fix" it.

## Reference files

| Area | Path |
|------|------|
| Debugging trail (zero-reward RL, `load_best_model_at_end`, eval decoder, versions) | [`S1_status.md`](S1_status.md) |
| Full numbers (1.7B S1 rows) | [`../model_size/results_table.md`](../model_size/results_table.md) |
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq1_reproduction/` (via `patches/common/sidreasoner-local.patch`) |
| Jobs / scripts pointers | [`../../jobs/reproduction/`](../../jobs/reproduction/README.md) · [`../../scripts/reproduction/`](../../scripts/reproduction/README.md) |
