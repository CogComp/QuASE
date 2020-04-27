#!/usr/bin/env bash

# BERT model (fine-tuning) on uncased NER
for percent in 10 100
do
    echo "run setting finetune bert percent:$percent%"
    export NER_DIR=/path/to/data/NER
    python sources/run_finetune_bert_ner.py \
    --data_dir $NER_DIR \
    --task_name=ner \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --percent $percent \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --max_seq_length 128 \
    --output_dir $NER_DIR/outputs-ner-finetune-bert-$percent-32-5-4
done