from abc import abstractmethod
from typing import List, Union, Dict
import copy
import flair
from flair.data import Sentence

import torch
from torch import nn
from pytorch_pretrained_bert import (
    BertTokenizer,
    BertModel,
)
from pytorch_pretrained_bert.modeling import BertLayer, BertPreTrainedModel
from allennlp.modules.scalar_mix import ScalarMix


class Interaction_2layer(nn.Module):
    def __init__(self, config):
        super(Interaction_2layer, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(2)])

    def forward(self, hidden_states, attention_mask, output_all_layers=True):
        all_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_layers.append(hidden_states)
        if output_all_layers:
            return all_layers
        else:
            return hidden_states


class SemanticBert(BertPreTrainedModel):
    """Standard QALM model for token-level classification.
    This module is composed of the Standard QALM model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(SemanticBert, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_ESRL = Interaction_2layer(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        all_encoder_layers, bert_list = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_ESRL_sequence_output = self.hidden_ESRL(sequence_output, extended_attention_mask)
        all_encoder_layers.extend(hidden_ESRL_sequence_output)
        return all_encoder_layers, bert_list


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


class SemanticBertEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        top_layer_only: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - bert_config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = self.load_semanticbert()
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.name = str(bert_model_or_path)
        self.static_embeddings = True
        if not top_layer_only:
            self._scalar_mix = ScalarMix(self.model.config.num_hidden_layers + 2,
                                         do_layer_norm=False)
        else:
            self._scalar_mix = None

    def load_semanticbert(self):
        pretrained_model = '/path/to/models/QAMR/' + \
                              'outputs-semanticbert-probe-c2q-interaction-V3-all-1e-4-1e-4-64/pytorch_model.bin'
        model = SemanticBert.from_pretrained("bert-base-uncased")
        print('load a pre-trained model from ' + pretrained_model)
        pretrained_state_dict = torch.load(pretrained_model)
        model_state_dict = model.state_dict()
        print('pretrained_state_dict', pretrained_state_dict.keys())
        print('model_state_dict', model_state_dict.keys())
        pretrained_state = {k: v for k, v in pretrained_state_dict.items() if
                            k in model_state_dict and v.size() == model_state_dict[k].size()}
        model_state_dict.update(pretrained_state)
        print('updated_state_dict', model_state_dict.keys())
        model.load_state_dict(model_state_dict)
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.hidden_ESRL.parameters():
            param.requires_grad = False
        return model

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[SemanticBertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0 : (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                SemanticBertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        # self.model.eval()
        all_encoder_layers, _ = self.model(
            all_input_ids, token_type_ids=None, attention_mask=all_input_masks
        )

        # with torch.no_grad():

        for sentence_index, sentence in enumerate(sentences):

            feature = features[sentence_index]

            # get aggregated embeddings for each BERT-subtoken in sentence
            subtoken_embeddings = []
            for token_index, _ in enumerate(feature.tokens):
                all_layers = []
                for layer_index in range(self.model.config.num_hidden_layers + 2):
                    layer_output = (
                        all_encoder_layers[int(layer_index)][sentence_index]
                    )
                    all_layers.append(layer_output[token_index])
                if self._scalar_mix is not None:
                    mix = self._scalar_mix(all_layers, feature.input_mask)
                else:
                    mix = all_layers[-1]
                subtoken_embeddings.append(mix)

            # get the current sentence object
            token_idx = 0
            for token in sentence:
                # add concatenated embedding to sentence
                token_idx += 1

                if self.pooling_operation == "first":
                    # use first subword embedding if pooling operation is 'first'
                    token.set_embedding(self.name, subtoken_embeddings[token_idx])
                else:
                    # otherwise, do a mean over all subwords in token
                    embeddings = subtoken_embeddings[
                        token_idx : token_idx
                        + feature.token_subtoken_count[token.idx]
                    ]
                    embeddings = [
                        embedding.unsqueeze(0) for embedding in embeddings
                    ]
                    mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                    token.set_embedding(self.name, mean)

                token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return self.model.config.hidden_size

