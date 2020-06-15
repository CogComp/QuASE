#!/usr/bin/env bash

for percent in 10
do
    echo "run setting percent:$percent% semanticbert"
    export SRL_DIR=/shared/hangfeng/code/SemanticBert/data/SRL
    python run_srl_single_V2.py \
    --data_dir $SRL_DIR \
    --task_name=SRL \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 40 \
    --max_seq_length 128 \
    --pretrain \
    --pretrained_model_file /shared/hangfeng/code/SemanticBert/data/QAMR/outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin \
    --output_dir $SRL_DIR/outputs-propbank-semanticbert-single-$percent%
done