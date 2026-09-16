#!/usr/bin/env bash
#SBATCH --job-name=gold_prob_100
#SBATCH --partition=mhong
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/users/2/jruan/6a99eb6ee4c4a85262c878e0/probability_results/scoring_%j.log

set -euo pipefail
cd /users/2/jruan/6a99eb6ee4c4a85262c878e0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/users/2/jruan/miniconda3/envs/toolalpaca/bin/python -u score_gold_probabilities.py
