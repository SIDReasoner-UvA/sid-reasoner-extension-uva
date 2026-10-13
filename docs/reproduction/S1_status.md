# Historical reasoning-activation diagnostics

These exploratory notes are separate from the final Games reproduction reported
in [RQ1](README.md). They are not evidence of full independent 1.7B GRPO retraining.
The associated raw logs and checkpoint configurations are not bundled.

The debugging runs encountered zero GRPO reward when outputs lacked the closing
`</think>` marker required by the reward parser. Two settings were investigated:
the RA epoch count and selection of the lowest direct-SID validation loss. The
cold-start RA implementation now saves the final epoch instead of restoring the
lowest-loss checkpoint. The continuation controls in RQ2 still use best-loss selection.

| Historical activation setting | Probability of `<think>` as first token |
|---|---|
| 1 epoch | 0.001 |
| 5 epochs | 0.90 |

Historical Office diagnostics recorded the following evaluation scores. These must
not be compared with the Games S1 reference or used as the final RQ1 result.

| Reasoning temperature | Recall@10 | NDCG@10 |
|---|---|---|
| 0 | 0.137 | 0.079 |
| 0.6 | 0.137 | 0.077 |
| 1.0 | 0.132 | 0.075 |

The notes also recorded mean reward 0.069, maximum reward 1.1, gradient norm 0.34,
and response length around 197 tokens after activation. These are debugging
observations, not final training or ranking measurements.

The local environment uses Transformers 4.51.3. The historical checkpoint
configuration comparison recorded 4.57.0 for the original environment; Appendix D
also reports this version difference. The model architecture and SID vocabulary
were reported as matching, but the configuration files are not included here.
