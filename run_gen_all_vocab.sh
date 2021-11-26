#!/usr/bin/env bash
dataset_name=$1

# signature params
echo " signature params "
max_seq_length=50
masked_lm_prob=0.2
max_predictions_per_seq=10
prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

echo "${signature}"

python -u gen_all_vocab.py --dataset_name=${dataset_name} --signature=${signature}

