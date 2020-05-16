#!/usr/bin/env bash

# Train the p-QuASE model with QAMR dataset
for train_batch_size in 32
do
    for lr in 5e-5
    do
        for epoch in 4
        do
            echo "run setting $train_batch_size $lr $epoch"
            export SQUAD_DIR=/path/to/data/QAMR
            python sources/run_squad.py \
            --bert_model bert-base-uncased \
            --do_train \
            --do_predict \
            --do_lower_case \
            --train_file $SQUAD_DIR/wiki.train.json \
            --predict_file $SQUAD_DIR/wiki.dev.json \
            --train_batch_size $train_batch_size \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --max_seq_length 128 \
            --doc_stride 128 \
            --output_dir $SQUAD_DIR/outputs-32-5-4
            python sources/evaluate-orig.py $SQUAD_DIR/wiki.dev.json $SQUAD_DIR/outputs-32-5-4/predictions.json
        done
    done
done
