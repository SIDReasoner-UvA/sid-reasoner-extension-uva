# Beyond Accuracy in SIDReasoner: A Reproducibility Study of Reward, Scale, and Diversity

A reproducibility study and extension of **SIDReasoner** (*Reasoning over Semantic IDs Enhances
Generative Recommendation*). SIDReasoner casts sequential recommendation as **reasoning +
Semantic-ID (SID) generation**: a user's history is encoded as a prompt, the model *thinks*,
then emits a Semantic ID — three nested tokens `(a₁, a₂, a₃)`, coarse-to-fine — naming the next
item. It is trained in three stages and tuned with GRPO reinforcement learning.

We evaluate the authors' released checkpoints, independently retrain the early SFT and
reasoning-activation (RA) stages at 1.7B on Games, and run the full SFT → RA → GRPO
pipeline at 0.6B and 4B. Component and reward ablations branch from the authors' RA
checkpoints. Full independent retraining of the 1.7B pipeline is outside this study's budget.

## Setup

The experiments target a Slurm/GPU cluster (Snellius); the steps below are the order to run them
in. Run everything from the repo root.

**1. Restore the baseline submodules.** The two baselines
(`baselines/repos/SIDReasoner`, `baselines/repos/SASRec`) are public git submodules — clone with
`--recursive`, or fetch their contents after a plain clone / unzip:

```bash
git clone --recursive https://github.com/SIDReasoner-UvA/sid-reasoner-extension-uva.git
# or, if already cloned:
git submodule update --init --recursive
```

**2. Apply the working-tree patch.** Four of the five suites (`reproduction`, `components`,
`model_size`, `diversity`) run from scripts that live *inside* the SIDReasoner submodule, because
they call the submodule's own `split.py` / `merge.py` / `evaluate_Qwen3_think.py` directly. Rather
than committing our changes into the submodule (which would unpin it from the upstream commit),
we keep them as a single diff, `patches/common/sidreasoner-local.patch`, and lay them down into
the submodule's working tree:

```bash
bash scripts/common/apply_submodule_patch.sh
```

This is **idempotent** (safe to re-run) and creates, among other things, the per-suite launch
scripts under `baselines/repos/SIDReasoner/final_scripts/<rqN_*>/`. The submodule stays pinned to
its upstream commit (`.gitmodules` sets `ignore = dirty`); run this before sourcing the
submodule's `snellius_env.sh`. The patch must be re-applied after any fresh
`git submodule update`.

**3. Build the environment.** Creates the SIDReasoner `uv` env (Python 3.10) inside the
submodule and re-applies the patch first:

```bash
sbatch jobs/common/setup_env.sh            # add --sglang / --megatron / --all for extras
```

**4. Download data and checkpoints.** Amazon review datasets land under `data/raw/<slug>/`;
author checkpoints under `data/checkpoints/` (both are git-ignored and absent from the zip):

```bash
# Datasets (delegates to the SIDReasoner Makefile):
bash scripts/common/data_prep/download_amazon.sh Video_Games
bash scripts/common/data_prep/download_amazon.sh Office_Products
bash scripts/common/data_prep/download_amazon.sh Industrial_and_Scientific

# Author SFT/RA checkpoints from HuggingFace (heyingzhi/sidreasoner_*):
bash scripts/common/data_prep/download_checkpoints.sh

# Or as Slurm jobs (same steps, batched):
sbatch jobs/common/download_amazon.job
sbatch jobs/common/download_checkpoints.job
sbatch jobs/common/download_model.job        # Qwen3 backbone
```

Per-suite run commands (which launch script to `sbatch`, expected metrics) are in each suite's
doc — see the **Docs** column below.

## Training pipeline

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Emit valid Semantic IDs. | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Wrap the answer in `<think>…</think>` before the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce so reasoning leads to correct recommendations. | post-GRPO |

Backbone: `Qwen/Qwen3-1.7B`. Datasets: Amazon `Video Games`, `Office Products`,
`Industrial & Scientific` (5-core filtered, sliding window 10, chronological 8:1:1 split,
full-item ranking). Primary metrics: Recall@{5,10}, NDCG@{5,10}.

## Research questions / experiment suites

The study is organized into five suites, one per research question. The code tree and `docs/`
mirror these names (see [`CODE-STRUCTURE.md`](CODE-STRUCTURE.md)).

| Suite | RQ | Question | Headline finding | Docs |
|-------|----|----------|------------------|------|
| `reproduction` | RQ1 | Can released artifacts recover the reported results, and can partial pre-RL retraining recover the early-stage trends? | Released scores reproduce; early-stage retraining exposes an RA epoch/checkpoint-selection discrepancy. | [docs](docs/reproduction/README.md) |
| `components` | RQ2 | Are RL and explicit reasoning strictly necessary? | Extra supervised training does not recover RL gains; scaffold benefits depend on dataset and checkpoint. | [docs](docs/components/README.md) |
| `grpo` | RQ3 | Is the GRPO reward formulation a sensitive design choice? | Top-10 quality is stable; removing the format term or using linear credit improves rank-1 accuracy. | [docs](docs/grpo/README.md) |
| `model_size` | RQ4 | How does backbone scale influence behavior? | GRPO accuracy/CoT trends persist across sizes; 4B shows no advantage over 0.6B in these runs. | [docs](docs/model_size/README.md) |
| `diversity` | RQ5 | How do decoding strategies affect item exposure? | Beam search offers the strongest tested accuracy–exposure compromise; sampling broadens coverage at a large accuracy cost. | [docs](docs/diversity/README.md) |

RQ1–RQ2 are the reproduction analyses (§5); RQ3–RQ5 are extensions (§6).
The historical submodule directory `rq3_model_size/` implements **current RQ4**;
its name is retained for compatibility with existing jobs and artifacts.
See the [paper alignment audit](docs/paper-alignment.md) for confirmed setup decisions and artifact coverage.
`HR@K` in code/results is Recall@K here because each test case has one target item.
Pipeline stages 1–3 are distinct from the original paper's S1/S3 *data ablation* labels.

All five suites are written up — start with the **Docs** link in each row above, or see
[`docs/README.md`](docs/README.md) for the documentation index.

## Repository layout

The repo is organized **by experiment suite**. Each top-level work directory splits into a
`common/` folder (shared code) plus one folder per suite (`reproduction/ components/ grpo/
model_size/ diversity/`). `grpo` is wrapper-native (its configs/jobs/`src` live here); the other
four suites run from scripts inside the SIDReasoner submodule, materialised by the working-tree
patch (see [Setup](#setup) above).

| Directory | Contents |
|-----------|----------|
| `baselines/` | Baseline repos (submodules under `repos/`) + uniform `adapters/`. |
| `configs/` | YAML experiment configs (`configs/<suite>/<variant>/<dataset>.yaml`). |
| `data/` | `raw/ processed/ enriched/` datasets, SID mappings, checkpoints. |
| `scripts/` | Entry points — `common/` (eval, train, data prep) + per-suite. |
| `src/` | Importable library — `src.common.*`, `src.<suite>.*`. |
| `jobs/` | Slurm job files. |
| `results/` | Metrics + prediction files, mirroring `configs/`. |
| `reports/` | Written-up results. |
| `patches/` | Local submodule modifications. |
| `docs/` | Per-suite documentation (`docs/<suite>/`). |

See [`CODE-STRUCTURE.md`](CODE-STRUCTURE.md) for the full layout and conventions (depth-independent
repo-root resolution, the `model_size` underscore rule).
