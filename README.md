# QuASE
This is the code repository for the ACL paper [QuASE: Question-Answer Driven Sentence Encoding](https://cogcomp.seas.upenn.edu/page/publication_view/905).
If you use this code for your work, please cite
```
@inproceedings{HeNgRo20,
    author = {Hangfeng He and Qiang Ning and Dan Roth},
    title = {{QuASE: Question-Answer Driven Sentence Encoding}},
    booktitle = {Proc. of the Annual Meeting of the Association for Computational Linguistics (ACL)},
    year = {2020},
}

```
## Online Demo 
Play with our [online demo](https://cogcomp.seas.upenn.edu/page/demo_view/QuASE).

<!-- <img align="center" width="800" height="400" alt="image" src="https://user-images.githubusercontent.com/22654200/174347012-135d93e4-4541-4fcb-abde-33f55e42b600.png"> -->


## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6
```
conda create -n quase python=3.6 
pip install -r requirements.txt
```

## Models
Our pre-trained models can be found in the [google drive](https://drive.google.com/drive/folders/1j6ufXtxFekPM9CfM5CxKfmwHsqLR8kNY?usp=sharing).

## Change the Dir Path
Change the /path/to/data (/path/to/models) to your data (models) dir. 

## Reproducing Experiments

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

To train a s-QuASE/p-QuASE with your own QA data (SQuAD format):
```
sh scripts/run_SemanticBERT_SQuAD.sh
sh scripts/run_BERT_squad_51K.sh
```

## SRL Evaluation:
The SRL metric implementation (SpanF1Measure) does not exactly track the output of the official PERL script (is typically 1-1.5 F1 below), and reported results used the official evaluation script (allennlp-experiments/run_srl_evaluation.sh).
