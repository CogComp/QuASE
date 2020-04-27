#!/usr/bin/env bash

##########################
# The script is used to evaluate the results of SRL model.
# Add semanticbert_token_embbedder.py into token_embedder/ and change the __init__.py

# Path (first argument): The path to the allennlp output directory for the model run.
# Device (second argument) The cuda device to use.
# Data path (third argument) The path to the _test_ data. The evaluation run uses the path in the allennlp config by default

# Test run.
python ../../allennlp/scripts/write_srl_predictions_to_conll_format.py --path=$1 --device=$2 --data=$3 --prefix=test

perl ../../allennlp/allennlp/tools/srl-eval.pl $1/test_gold.txt $1/test_predictions.txt | tee -a $1/test_results.txt

##########################
