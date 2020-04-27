#!/usr/bin/env bash
for train_batch_size in 16
do
    for lr in 5e-5
    do
        for epoch in 4
        do
            for percent in 100
            do
                echo "run setting $train_batch_size $lr $epoch $percent"
                export SQUAD_DIR=/scratch/hangfeng/code/SemanticBert/data/SQuAD
                python run_squad.py \
                --bert_model bert-base-uncased \
                --do_train \
                --do_predict \
                --do_lower_case \
                --train_file $SQUAD_DIR/train-v1.1.sample.51K.json \
                --predict_file $SQUAD_DIR/dev-v1.1.json \
                --train_batch_size $train_batch_size \
                --learning_rate $lr \
                --num_train_epochs $epoch \
                --max_seq_length 384 \
                --doc_stride 128 \
                --percent $percent \
                --output_dir $SQUAD_DIR/outputs-$train_batch_size-$lr-$epoch-$percent-51K
                python evaluate-orig.py $SQUAD_DIR/dev-v1.1.json $SQUAD_DIR/outputs-$train_batch_size-$lr-$epoch-$percent-51K/predictions.json
            done
        done
    done
done
