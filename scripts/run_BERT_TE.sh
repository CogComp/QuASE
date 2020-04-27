#!/usr/bin/env bash

# BERT model on TE
export GLUE_DIR=/path/to/data/GLUE
export TASK_NAME=MNLI

python sources/run_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --percent 10 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --output_dir $GLUE_DIR/$TASK_NAME/outputs-32-5-4-10%

python sources/run_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --output_dir $GLUE_DIR/$TASK_NAME/outputs-32-5-4

