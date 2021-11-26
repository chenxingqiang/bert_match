#!/usr/bin/env bash
CKPT_DIR="../RECALL_BERT"
conda activate tfg-1.15

dataset_name=$1
all_vocab_name=$2
train_multi=$3
eval_mode=$4

batch_size=64
dataset_name_start="20210419-user_iterms-refine-0-part-000"
max_seq_length=50
masked_lm_prob=0.2
max_predictions_per_seq=10
dim=64

num_train_steps=$((40000 * train_multi))
echo "num_train_steps: ${num_train_steps}"
num_eval_steps=$((40000 + 40000 * (train_multi -1)))
echo "num_eval_steps: ${num_eval_steps}"

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10
pool_size=16

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

if [[ $eval_mode -eq 1 ]] ;then
	python -u run.py \
    	--train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    	--test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    	--vocab_filename=./data/${all_vocab_name}${signature}.vocab \
    	--user_history_filename=./data/${dataset_name}${signature}.his \
    	--checkpointDir=${CKPT_DIR}/${dataset_name_start} \
    	--signature=${signature}-${dim} \
    	--do_train=True \
    	--do_eval=True \
    	--bert_config_file=./bert_train/bert_config_${all_vocab_name}_${dim}.json \
    	--batch_size=${batch_size} \
    	--max_seq_length=${max_seq_length} \
    	--max_predictions_per_seq=${max_predictions_per_seq} \
    	--num_train_steps=${num_eval_steps} \
    	--num_warmup_steps=100 \
    	--dataset_name=${dataset_name} \
    	--learning_rate=1e-4
else 
	echo "Did not do evaluaiton of latest model!"
fi

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${all_vocab_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name_start} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${all_vocab_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --dataset_name=${dataset_name} \
    --learning_rate=1e-4
