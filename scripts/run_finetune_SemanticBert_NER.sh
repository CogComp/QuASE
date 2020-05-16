#!/usr/bin/env bash

# BERT + p-QuASE on uncased NER

for percent in 10 100
do
    echo "run setting finetune semanticbert percent:$percent% pretrain:QAMR"
    export NER_DIR=/path/to/data/NER
    python sources/run_finetune_semanticbert_ner.py \
    --data_dir $NER_DIR \
    --task_name=ner \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --max_seq_length 128 \
    --pretrain \
    --pretrained_model_file /path/to/models/QAMR/outputs-32-5-4/pytorch_model.bin \
    --output_dir $NER_DIR/outputs-ner-finetune-semanticbert-$percent-16-5-4-correct
done

