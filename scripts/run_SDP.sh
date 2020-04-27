#!/usr/bin/env bash

# BERT embeddings for SDP
for percent in 10 100
do
    echo "run setting percent:$percent% pretrain_bert:nopretrain"
    export SDP_DIR=/path/to/data/SDP
    python sources/run_sdp.py \
    --data_dir $SDP_DIR \
    --task_name=sdp \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 40 \
    --max_seq_length 128 \
    --output_dir $SDP_DIR/outputs-sdp-$percent%-nopretrain-weigthed-sum-24
done