The paper uses the fresh local A40 run in `local_gpu_verified/`.

- `original_per_task.csv` and `unlearned_per_task.csv`: 100 tasks each.
- `per_task_probabilities.csv`: both models, 200 rows.
- `summary.csv` / `summary.json`: arithmetic token means, then equal means over 100 tasks.
- `references.json`: exact prompts, reference thoughts/actions, and source indices.
- `verification.json`: numerical checks and artifact hashes.
- `../local_gpu_verified.log` relative to that directory: completed GPU run log.
- `task_success_verified.csv` in this directory: first-table counts rechecked against existing individual Luna judgments; these rollouts and judgments were not rerun.

The probability diagnostic uses the frozen 100 forget tasks. It scores the first gold thought (z), first gold action including arguments/markers conditioned on that thought (a), and their concatenation (y). It excludes prompt tokens and later rollout turns. Arithmetic token means are not sequence probabilities; per-task sequence log probabilities are also saved. Both models use identical tokens and contexts. Weights use FP16 and probability normalization uses FP32.

Reproduce from the workspace:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /users/2/jruan/miniconda3/envs/toolalpaca/bin/python -u score_gold_probabilities.py --output-dir probability_results/local_gpu_verified
```

Older results in this directory are retained. The manuscript cites the fresh subdirectory explicitly. Small numerical differences from the older run do not change the reported four-decimal values.

Sequence log probabilities are summarized by `python summarize_log_probabilities.py` in the workspace. The fresh-run directory now also contains `per_task_log_probabilities.csv` (200 rows), `log_probability_summary.csv`, and `log_probability_summary.json`. Natural logarithms are summed within tasks without length normalization; the paper reports the equal mean over 100 tasks. The summary CSV additionally includes the sum across all 100 tasks. The chain-rule residual is zero for every saved row. No additional GPU inference was needed: these sums were already computed from token log probabilities in the completed GPU run.
