#!/usr/bin/env bash
for epoch in 32
do
    for lr in 1e-4
    do
        for slr in 1e-4
        do
            echo "run setting $lr $slr $epoch"
            export SQUAD_DIR=/scratch/hangfeng/code/SemanticBert/data/SQuAD
            python run_squad_qamr_probe_c2q_interaction_V3.py \
            --bert_model bert-base-uncased \
            --do_train \
            --do_predict \
            --do_lower_case \
            --train_file $SQUAD_DIR/train-v1.1.sample.51K.json \
            --predict_file $SQUAD_DIR/dev-v1.1.json \
            --train_batch_size 12 \
            --predict_batch_size 12 \
            --learning_rate $lr \
            --small_learning_rate $slr \
            --num_train_epochs $epoch \
            --max_seq_length 384 \
            --doc_stride 128 \
            --max_query_length 64 \
            --output_dir $SQUAD_DIR/outputs-semanticbert-probe-c2q-interaction-V3-squad-$lr-$slr-$epoch-51K
            python evaluate-orig.py $SQUAD_DIR/dev-v1.1.json $SQUAD_DIR/outputs-semanticbert-probe-c2q-interaction-V3-squad-$lr-$slr-$epoch-51K/predictions.json
        done
    done
done
