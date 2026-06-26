# Documentation Index

Per-suite documentation for the SIDReasoner reproducibility & extension study. Each suite maps
to one research question from the paper and mirrors the code layout in [`CODE-STRUCTURE.md`](../CODE-STRUCTURE.md).

| Suite | Paper RQ (body) | Question | Status |
|-------|-----------------|----------|--------|
| [`reproduction/`](reproduction/README.md) | RQ1 | Can we reproduce the original SIDReasoner trends from released checkpoints? | **written** |
| [`components/`](components/README.md) | RQ2 | Are the RL stage and the explicit-reasoning scaffold necessary? | **written** |
| [`model_size/`](model_size/README.md) | RQ3 | How does backbone size affect performance? | **written** |
| [`grpo/`](grpo/README.md) | RQ4 | Is the GRPO reward / objective a sensitive design choice? | **written** |
| [`diversity/`](diversity/README.md) | RQ5 | Item-side exposure diversity (exposure concentration in the top-10). | **written** |

> **RQ numbering note.** The paper's *intro* lists four RQs, but the *body* renumbers them to
> five sections (RQ1 repro, RQ2 components, RQ3 model scale, RQ4 GRPO objective, RQ5 diversity).
> Suites follow the **body** numbering. `model_size` uses an underscore because `src/` imports
> it as a Python module.

---

## How to write a suite README (instructions for future passes)

All five suites are written, each as a single `docs/<suite>/README.md`. `grpo/` (the deepest)
and `components/` are the reference templates; keep new passes consistent with this recipe so
all suites read the same way. Note `diversity/` covers item-side exposure concentration (results
pending), and the 1.7B Games-S1 numbers are shared between `reproduction/` and
`model_size/results_table.md`.

### 1. Source of truth
The paper (`paper.txt`, LaTeX) is the source. Map suite $\to$ paper section:

| Suite | Paper sections to mine |
|-------|------------------------|
| `reproduction` | §RQ1 (`sec:rq1_reproduction`), Table `tab:sidreasoner_authors_checkpoint_eval`, App. `app:original_sidreasoner_results`, App. `app:sasrec_reproduction` |
| `model_size` | §"How does model scale affect…" (Qwen3-0.6B / 1.7B / 4B), the model-scale result table + GPU-hours table (both marked `[Insert …]` — fill from `results/`) |
| `diversity` | §RQ5 (`sec:rq4` label), Table `tab:diversity_accuracy_full_sweep_compact`, App. `app:diversity_accuracy_results` (Coverage/Entropy/Gini@10 exposure, beam-vs-sample deltas) |

When the paper has `[INSERT …]` / `[Insert … table]` placeholders, the numbers are not final —
pull them from `results/<suite>/` rather than inventing values, and flag anything still missing.

### 2. Required sections (match `grpo/README.md` and `components/README.md`)
1. **Title + RQ blockquote** — the verbatim research question as a `>` quote.
2. **One-paragraph framing** — what this suite asks and how it relates to the others.
3. **Pipeline context** — the S1 $\to$ S2 $\to$ S3 table, and which checkpoint each experiment branches from.
4. **Axes / experiment list** — one subsection per experiment (`X1`, `X2`, …) with: what it
   tests, the method, a results table copied from the paper, and a **bold one-line Finding**.
5. **Bottom line** — 2–4 bullets of takeaways + practical implication.
6. **Caveats** — backbone, seeds, datasets covered, single-run limitations.
7. **Reference files** — table linking `configs/ src/ scripts/ jobs/ tests/ results/ reports/`
   for that suite (use repo-relative `../../` links). If the suite is unpopulated, say so.

### 3. Conventions
- **Tables:** convert LaTeX tables to GitHub Markdown. Round consistently with the paper.
  Bold best / underline second-best as the paper does (Markdown can't underline — use bold for
  best and note second-best in prose, or **bold** both and label).
- **Deltas:** keep the paper's sign convention and state it (e.g. "no-think $-$ thinking, so
  negative = thinking better").
- **Links:** files use `[name](../../path)`; cross-suite refs use `[grpo](../grpo/README.md)`.
- **Style:** declarative, evidence-first. Every experiment ends in a Finding; the suite ends in
  a Bottom line + Caveats. No hedging beyond the stated caveats.
- After writing, update the **status** column in the table above and the **suite stub
  `README.md`s** (under `configs/ src/ scripts/ jobs/ tests/ results/ reports/ patches/`) only
  if they need more than the one-line "see docs" pointer.

### 4. The GRPO suite is the reference implementation
`grpo/README.md` is the deepest single-file suite write-up; its per-variant numbers live under
[`../reports/grpo/`](../reports/grpo/). A new suite README should match the *depth of the GRPO
`README.md`*. Split a suite into numbered files only if it grows large enough to warrant it.
