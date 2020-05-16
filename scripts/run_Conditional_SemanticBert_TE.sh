#!/usr/bin/env bash

# BERT + p-QuASE on TE
export GLUE_DIR=/path/to/data/GLUE
for TASK_NAME in "MNLI"
do
    for epoch in 4
    do
        for train_batch_size in 32
        do
            for percent in 10 100
            do
                echo "run conditional semanticbert setting $TASK_NAME $epoch $train_batch_size $percent"
                python sources/run_classifier_conditional_semanticbert.py \
                --task_name $TASK_NAME \
                --do_train \
                --do_eval \
                --do_lower_case \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --bert_model bert-base-uncased \
                --max_seq_length 128 \
                --train_batch_size $train_batch_size \
                --eval_batch_size 64 \
                --learning_rate 5e-5 \
                --num_train_epochs $epoch \
                --percent $percent \
                --pretrain \
                --pretrained_model_file /path/to/models/QAMR/outputs-32-5-4/pytorch_model.bin \
                --output_dir $GLUE_DIR/$TASK_NAME/outputs-conditional-semanticbert-$epoch-$train_batch_size-$percent%-correct
            done
        done
    done
done
