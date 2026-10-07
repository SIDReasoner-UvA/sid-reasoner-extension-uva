# Documentation Index

The revised paper organizes reproduction as RQ1–RQ2 (§5) and extensions as RQ3–RQ5 (§6).
Suite names mirror [CODE-STRUCTURE.md](../CODE-STRUCTURE.md).

| Suite | RQ | Research question | Paper location |
|---|---|---|---|
| [reproduction](reproduction/README.md) | RQ1 | Can released artifacts recover the reported results, and can partial pre-RL retraining recover the early-stage trends? | §5, Table 1, Appendix A |
| [components](components/README.md) | RQ2 | Are RL and explicit reasoning strictly necessary? | §5, Table 2, Appendix B / Table 4 |
| [grpo](grpo/README.md) | RQ3 | Is the GRPO reward formulation a sensitive design choice? | §6, Table 3, Appendix C |
| [model_size](model_size/README.md) | RQ4 | How does backbone scale influence behavior? | §6, Figure 2, Appendix D |
| [diversity](diversity/README.md) | RQ5 | How do decoding strategies affect item exposure? | §6, Figure 3 |

`rq3_model_size/` is a historical executable path for **current RQ4**, retained for
compatibility. The revised introduction and body now use the same five RQs.

The [alignment audit](paper-alignment.md) records changes and author-confirmed resolutions
between `ECIR_Paper_revised-3.pdf` and committed experiment artifacts. The PDF is not
vendored in this repository. Historical run notes retain their provenance; current
suite READMEs give the paper's interpretation and identify any unverified details.

When updating documentation, distinguish the original SIDReasoner paper from this
reproducibility paper. Use the revised paper as the authoritative source for reported numbers and its
printed precision. Label supplementary run diagnostics separately; do not rewrite raw
evaluation outputs or reward computation to force a match. Distinguish pipeline stages SFT / RA / GRPO from the
original S1/S3 data-ablation labels. `HR@K` equals Recall@K for the single-target test
protocol. Use validation data for checkpoint selection; reserve test data for reporting.
