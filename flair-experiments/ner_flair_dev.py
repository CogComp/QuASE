from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# Flair embeddings for NER on the small data setting

# 1. get the corpus
columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}
data_folder = '/path/to/data/NER'
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='valid.txt',
                              test_file='test.txt',
                              dev_file='test.txt')
# corpus: Corpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path='resources/tasks')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    PooledFlairEmbeddings('news-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('news-backward', pooling='min'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('/path/to/data/NER/ner-flair-dev',
              max_epochs=150)
