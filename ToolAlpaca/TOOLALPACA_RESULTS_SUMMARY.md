# ToolAlpaca Results Summary

| Evaluation Setting | Configuration | Forget Process | Retain Process | Forget Overall | Retain Overall | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Paper-aligned unlearning evaluation | Unlearned model alone (B0) | 33% | 53% | 28% | 50% | Baseline unlearned checkpoint. |
| Paper-aligned unlearning evaluation | Unlearned + ToolAlpaca-13B Thought harness (H1) | 53% | 62% | 47% | 54% | Retain process improves by 9 percentage points, but forgotten capability also recovers by 20 points. |
| Paper-aligned comparator | Unlearned + GPT-3.5 Thought (G35) | 52% | 42% | 47% | 40% | Restores forgotten capability but reduces retained performance. |
| Paper-aligned comparator | ToolAlpaca-13B alone (S13) | 82% | 76% | 76% | 72% | Strong intact-model result; this model has not been unlearned. |
| Paper-aligned comparator | GPT-3.5 alone | 28% | 23% | 14% | 12% | Low score mainly reflects incompatibility with the local ReAct parser. |
| Official ToolAlpaca-7B reproduction | Released repository result | 63% | — | 60% | — | Official released evaluation result. |
| Official ToolAlpaca-7B reproduction | Fresh all-GPT-4 evaluation | 56% | — | 54% | — | Close to the released result, but 6 percentage points lower overall. |

**Note:** Higher retain scores are better. Higher forget scores are worse because they indicate that the model has recovered a capability that was intended to be forgotten.
