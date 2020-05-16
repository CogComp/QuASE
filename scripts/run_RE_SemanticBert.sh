#!/usr/bin/env bash

# BERT embeddings + s-QuASE for RE
for percent in 10 100
do
    for seed in 666
    do
        echo "run setting semanticbert percent:$percent% pretrain_bert:qamr"
        export RC_DIR=/path/to/data/RC
        python sources/run_relation_classification_semanticbert.py \
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
        --num_train_epochs 80 \
        --max_seq_length 128 \
        --pretrain \
        --pretrained_model_file /path/to/models/QAMR/outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin \
        --output_dir $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80
        perl $RC_DIR/semeval2010_task8_scorer-v1.2.pl $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80/pred.txt \
        $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80/gold.txt > $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80/results.txt
        perl $RC_DIR/semeval2010_task8_scorer-v1.2.pl $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80/pred.txt \
        $RC_DIR/outputs-RC-$percent%-attbilstm-semanticbert-qamr-correct-3e-3-80/gold.txt
    done
done