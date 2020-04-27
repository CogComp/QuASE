# ISfromQA
This is the code repository for the ArXiv paper [Incidental Supervision from Question-Answering Signals](https://arxiv.org/pdf/1909.00333.pdf).
If you use this code for your work, please cite
```
@article{he2019incidental,
  title={Incidental Supervision from Question-Answering Signals},
  author={He, Hangfeng and Ning, Qiang and Roth, Dan},
  journal={arXiv preprint arXiv:1909.00333},
  year={2019}
}

```
## Play with our [online demo](http://dickens.seas.upenn.edu:4006).

## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
install pytorch\
pip install pytorch-pretrained-bert\
pip install allennlp

## The models
Our pre-trained models can be found in the [google drive](https://drive.google.com/drive/folders/1j6ufXtxFekPM9CfM5CxKfmwHsqLR8kNY?usp=sharing).

## Change the Dir Path
Change the /path/to/data (/path/to/models) to your data (models) dir. 

## Reproducing experiments

To reproduce our experiments based on the scripts:
```
sh scripts/run_script.sh
sh scripts/run_BERT_MRC.sh (an example)
```

To reproduce the experiments based on the allennlp (go into the allennlp-experiments dir):
```
allennlp train /path/to/model/configuration -s /path/to/serialization/dir --include-package models
allennlp train coref_bert.jsonnet -s coref-bert --include-package models (an example)
```

To reproduce the experiments based on the flair (go into the flair-experiments dir):
```
python ner_flair.py
python ner_flair_semanticbert.py
```

To train a conditional/standard SemanticBERT with your own QA data (SQuAD format):
```
sh scripts/run_BERT_squad_51K.sh
sh scripts/run_SemanticBERT_SQuAD.sh
```

SRL Evaluation:\
The SRL metric implementation (SpanF1Measure) does not exactly track the output of the official PERL script (is typically 1-1.5 F1 below), and reported results used the official evaluation script (allennlp-experiments/run_srl_evaluation.sh).
