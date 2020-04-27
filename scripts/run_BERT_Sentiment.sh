#!/usr/bin/env bash

# BERT model on Sentiment
export GLUE_DIR=/path/to/data/GLUE
for TASK_NAME in "SST-2"
do
    for epoch in 3
    do
        for train_batch_size in 32
        do
            for percent in 10 100
            do
                echo "run setting $TASK_NAME $epoch $train_batch_size $percent"
                python sources/run_classifier.py \
                --task_name $TASK_NAME \
                --do_train \
                --do_eval \
                --do_lower_case \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --bert_model bert-base-uncased \
                --max_seq_length 128 \
                --train_batch_size $train_batch_size \
                --eval_batch_size 64 \
                --learning_rate 2e-5 \
                --num_train_epochs $epoch \
                --percent $percent \
                --output_dir $GLUE_DIR/$TASK_NAME/outputs-$epoch-$train_batch_size-$percent%-2e-5
            done
        done
    done
done