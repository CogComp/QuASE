#!/usr/bin/env bash

# BERT embeddings for RE
for percent in 10
do
    echo "run setting percent:$percent% pretrain_bert:nopretrain"
    export RC_DIR=/path/to/data/RC
    python sources/run_relation_classification.py \
    --data_dir $RC_DIR \
    --task_name=rc \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 80 \
    --eval_batch_size 80 \
    --learning_rate 3e-3 \
    --num_train_epochs 40 \
    --max_seq_length 128 \
    --output_dir $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3
    perl $RC_DIR/semeval2010_task8_scorer-v1.2.pl $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3/pred.txt \
    $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3/gold.txt > $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3/results.txt
    perl $RC_DIR/semeval2010_task8_scorer-v1.2.pl $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3/pred.txt \
    $RC_DIR/outputs-RC-$percent%-nopretrain-attbilstm-3/gold.txt
done