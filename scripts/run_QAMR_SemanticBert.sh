#!/usr/bin/env bash

# Train the s-QuASE model with QAMR data
for epoch in 64
do
    for lr in 1e-4
    do
        for slr in 1e-4
        do
            echo "run setting $lr $slr $epoch"
            export SQUAD_DIR=/path/to/data/QAMR
            python sources/run_squad_qamr_probe_c2q_interaction_V3.py \
            --bert_model bert-base-uncased \
            --do_train \
            --do_predict \
            --do_lower_case \
            --train_file $SQUAD_DIR/wiki.train.json \
            --predict_file $SQUAD_DIR/wiki.dev.json \
            --train_batch_size 90 \
            --predict_batch_size 90 \
            --learning_rate $lr \
            --small_learning_rate $slr \
            --num_train_epochs $epoch \
            --max_seq_length 64 \
            --doc_stride 64 \
            --max_query_length 24 \
            --output_dir $SQUAD_DIR/outputs-semanticbert-probe-c2q-interaction-V3-all-$lr-$slr-$epoch
            python sources/evaluate-orig.py $SQUAD_DIR/wiki.dev.json $SQUAD_DIR/outputs-semanticbert-probe-c2q-interaction-V3-all-$lr-$slr-$epoch/predictions.json
        done
    done
done
