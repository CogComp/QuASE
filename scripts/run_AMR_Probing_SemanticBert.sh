#!/usr/bin/env bash

# AMR probing on the s-QuASE embeddings
for percent in 100
do
    echo "run setting percent:$percent%"
    export AMR_DIR=/path/to/data/AMR
    python sources/probe_AMR_SemanticBert.py \
    --data_dir $AMR_DIR \
    --task_name=amr \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 100 \
    --eval_batch_size 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 40 \
    --max_seq_length 128 \
    --pretrain \
    --pretrained_model_file /path/to/models/QAMR/outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin \
    --output_dir $AMR_DIR/outputs-AMR-proxy-$percent%-SemanticBert-qamr
done