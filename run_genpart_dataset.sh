#!/usr/bin/env bash

dataset_name=$1
all_vocab_name=$2

max_seq_length=50
masked_lm_prob=0.2
max_predictions_per_seq=10
prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10
pool_size=16

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

python -u gen_data_partly.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \
    --all_vocab_file=./data/${all_vocab_name}${signature}.vocab  \
    --all_vocab=True
