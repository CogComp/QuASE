import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


def separate_hyphens(og_sentence: List[str]):
    new_sentence = []
    new_indices = []
    i = 0
    for word in og_sentence:
        broken_h_indices = []
        h_idx = word.find('-')
        bslash_idx = word.find('/')
        h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
        prev_h_bs_idx = -1
        while h_bs_idx > 0:
            subsection = word[prev_h_bs_idx+1:h_bs_idx+1]
            broken_h_indices.append(i)
            new_sentence.append(subsection)
            prev_h_bs_idx = h_bs_idx
            h_idx = word.find('-', h_bs_idx+1)
            bslash_idx = word.find('/', h_bs_idx+1)
            h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
            i += 1
        subsection = word[prev_h_bs_idx+1:]
        new_sentence.append(subsection)
        broken_h_indices.append(i)
        i += 1
        new_indices.append(broken_h_indices)
    return new_sentence, new_indices


def get_bio_tags(args: List[str], new_indices: List[List[int]], new_sentence: List[str]):
    all_args_ordered = []
    for arg in args:
        arg = arg.replace('*', ',') # TODO understand diff between chains and split, for converting to BIO
        subargs = arg[:arg.find('-')]
        subargs = subargs.split(",")
        label = arg[arg.find('-')+1:]

        for arg in subargs:
            # If is a hyphenation for a span of words, include all inside up to edges' hyphens, if existant. Assumes continuity.
            start = arg[:arg.find(':')]
            sub_idx = start.find('_')
            if sub_idx < 0:
                new_start = new_indices[int(start)][0]
            else:
                if len(new_indices[int(start[:sub_idx])]) <= 1:
                    # Specified hyphenated arg, but start index not actually a hyphenation. Consider moving handling of this to the span.srl generating code.
                    new_start = new_indices[int(start[:sub_idx])][0]
                elif int(start[sub_idx+1:]) >= len(new_indices[int(start[:sub_idx])]):
                    print("Faulty data point with arg ", arg)
                    continue
                else:
                    new_start = new_indices[int(start[:sub_idx])][int(start[sub_idx+1:])]
            end = arg[arg.find(':')+1:]
            sub_idx = end.find('_')
            if sub_idx < 0:
                new_end = new_indices[int(end)][0]
            else:
                if len(new_indices[int(end[:sub_idx])]) <= 1:
                    new_end = new_indices[int(end[:sub_idx])][0]
                elif int(end[sub_idx+1:]) >= len(new_indices[int(end[:sub_idx])]):
                    print("Faulty data point with arg ", arg)
                    continue
                else:
                    new_end = new_indices[int(end[:sub_idx])][int(end[sub_idx+1:])]
            all_args_ordered.append((new_start, new_end, label))
    all_args_ordered = sorted(all_args_ordered, key=lambda x: x[0])

    bio_tags = ['O' for _ in range(len(new_sentence))]
    for arg in all_args_ordered:
        current_label_at_start = bio_tags[arg[0]]
        current_label_at_end = bio_tags[arg[1]]
        if current_label_at_start != 'O':
            if current_label_at_end != 'O':
                if current_label_at_start[2:] == current_label_at_end[2:]:
                    # This span is dominated, so we can just skip it.
                    continue

        bio_tags[arg[0]] = "B-{0}".format(arg[2])
        i = arg[0]+1
        while i <= arg[1]:
            bio_tags[i] = "I-{0}".format(arg[2])
            i += 1

    return bio_tags


def _convert_tags_to_wordpiece_tags(new_tags: List[int], end_offsets: List[int]):
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where apropriate to deal with words which
    are split into multiple wordpieced by the tokenizer.

    # Parameters

    new_tags: `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces. 
        Corresponds to hyphen-separated sentence, not original sentence.
    end_offsets: `List[int]`
        The wordpiece offsets.

    # Returns

    The new BIO tags.
    """
    wordpieced_tags = []
    j = 0
    for i, offset in enumerate(end_offsets):
        tag = new_tags[i]
        is_o = tag=="O"
        is_start = True
        while j < offset:
            if is_o:
                wordpieced_tags.append("O")
            elif tag.startswith("I"):
                wordpieced_tags.append(tag)
            elif is_start and tag.startswith("B"):
                wordpieced_tags.append(tag)
                is_start = False
            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                wordpieced_tags.append("I-" + label)
            j += 1
    return ["O"] + wordpieced_tags + ["O"]


def _convert_nom_indices_to_wordpiece_indices(nom_indices: List[int], end_offsets: List[int]):
    """
    Converts binary nom indicators to account for a wordpiece tokenizer.

    Parameters:
    
    nom_indices: `List[int]`
        The binary nom indicators, 0 for not the nom, 1 for the nom.
    end_offsets: `List[int]`
        The wordpiece end offsets, including for separated hyphenations.


    Returns:

    The new nom indices.
    """
    j = 0
    new_nom_indices = []
    for i, offset in enumerate(end_offsets): # For each word's offset (includes separated hyphenation)
        indicator = nom_indices[i] # 1 if word at i is nom, 0 if not.
        while j < offset:
            new_nom_indices.append(indicator) # Append indicator over length of wordpieces for word.
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_nom_indices + [0]


@DatasetReader.register("nom-srl-bio")
class SrlReader(DatasetReader):
    # Does this need to be called something else?
    """
    This DatasetReader is designed to read in the Nombank data that has been
    converted to self-defined "span" format. This dataset reader specifically
    will read the data into a BIO format, as a result throwing away some 
    information and providing functionality to break apart hyphenated words,
    in order to try to preserve as much information as possible. It returns
    a dataset of instances with the following fields:

    tokens: `TextField`
        The tokens in the sentence.
    (HOLD OFF ON THIS ONE FOR NOW.) hs_tokens: 'TextField'
        The tokens in the sentence, where hyphenated words are separated.
    nom_indicator: `SequenceLabelField`
        A sequence of binary indicators for whether the word is the nominal 
        predicate for this frame.
    tags: `SequenceLabelField`
        A sequence of Nombank tags for the given nominal in a BIO format.

    # Parameters

    token_indexers: `Dict[str, TokenIndexer]`, optional
        We use this for both the premise and hypothesis. 
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    bert_model_name: `Optional[str]`, (default=None)
        The BERT model to be wrapped. If you specify a bert_model here, the
        BERT model will be used throughout to expand tags and nom indicator.
        If not, tokens will be indexed regularly with token_indexers. 

    # Returns

    A `Dataset` of `Instances` for nominal Semantic ROle Labeling.
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            bert_model_name: str = None,
            lazy: bool = False,
        ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
            # TODO check about if we worry about this, bc nb specified something about capitalizations and NEs. Prob just need to choose a proper bert tokenizer.
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

    def _wordpiece_tokenize_input(
            self, tokens: List[str]
            ) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as
        adding BERT CLS and SEP tokens to the beginning and end of the 
        sentence. The offsets will also point to sub-words inside hyphenated
        tokens. 

        For example:
        `stalemate` will be bert tokenized as ["stale", "##mate"].
        `quick-stalemate` will be bert tokenized as ["quick", "##-", "##sta", "##lem", "##ate"]
        We will want the tags to be at the finst granularity specified, like
        [B-ARGM-MNR, I-ARGM-MNR, B-REL, I-REL, I-REL]. The offsets will 
        correspond to the first word out of each hyphen chunk, even if the
        entire initial token is one argument. In this example, offsets would
        be [0, 2]

        # Returns

        wordpieces: List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets: List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]` 
            results in the end wordpiece of each (separated) word chosen.
        start_offsets: List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets
        '''
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative+1)
            # +1 because we add the starting "[CLS]" token later.
            if token.upper() in ["--", "-LRB-", "-RRB-", "-LCB-", "-RCB-"]:
                # Also should do -LSB-, -RSB- apparently but it's not anywhere in wsj.
                h_indices = []
            else:
                h_indices = [i for i, x in enumerate(word_pieces) if x=='##-'] 
            for h_idx in h_indices:
                end_offsets.append(cumulative+h_idx+1)
                start_offsets.append(cumulative+h_idx+2)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets'''

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        # logger.info("Reading SRL instances from dataset file at %s", file_path)
        srl_data = self.read_nom_srl(file_path)
        for (og_sentence, og_nom_loc, new_indices, new_sentence, new_tags) in srl_data:
            og_tokens = [Token(t) for t in og_sentence]
            new_tokens = [Token(t) for t in new_sentence]
            new_pred_idx = new_indices[og_nom_loc[0]]
            if len(og_nom_loc) > 1:
                if og_nom_loc[1] >= len(new_pred_idx):
                    # Some datapoints are faulty, such as wsj_1495 10 47
                    print('Faulty data point. Trying to access hyphenation that does not exist.')
                    continue
                new_pred_idx = [new_pred_idx[og_nom_loc[1]]]
            # if isinstance(new_pred_idx, type([])):
                # Hacky thing where is the noun predicate is the entire hyphenated word, we'll need to point that out for the model. Is it ok that it's not one-hot? check. TODO
            nom_indicator = [1 if i in new_pred_idx else 0 for i in range(len(new_tokens))]
            yield self.text_to_instance(og_tokens, new_tokens, nom_indicator, new_tags)

    def read_nom_srl(self, filename):
        """
        This process reads in the nominal srl data in span format, and 
        converts it to BIO format. 

        example input line:
        Air & Water Technologies Corp. completed the acquisition of Falcon Associates Inc. , a Bristol , Pa. , asbestos-abatement concern , for $ 25 million of stock . ||| 18_1:18_1-rel 18_0:18_0-ARG1

        its output:
        (
            og_sentence = ['Air', '&', 'Water', 'Technologies' ... 'asbestos-abatement', 'concern', ',', 'for', '$', '25', 'million', 'of', 'stock', '.'],
            og_nom_loc = (18, 1),
            new_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, [18, 19], 20, 21, ... , 28],
            new_sentence = ['Air', '&', ... 'asbestos-', 'abatement', 'concern', ...],
            new_tags = ['O', ... 'O', 'B-ARG1', 'B-REL', 'O', ..., 'O']
            
        )
        """

        fin = open(filename, 'r')
        data = []
        for line in fin.readlines():
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            og_sentence = str_list[:separator_index]
            args = str_list[separator_index+1:]
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = args[0][:args[0].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = (int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx+1:]))
            # Get new indices and new sentence, hyphenations separated.
            new_sentence, new_indices = separate_hyphens(og_sentence)
            # Get BIO tags from argument spans. 
            new_tags = get_bio_tags(args[1:], new_indices, new_sentence)
            assert len(new_tags) == len(new_sentence)
            data.append((og_sentence, og_nom_loc, new_indices, new_sentence, new_tags))
        fin.close()
        return data

    def text_to_instance(
            self, og_tokens: List[Token], new_tokens: List[Token], nom_label: List[int], new_tags: List[str]=None
            ) -> Instance:
        """
        We take original sentence, `pre-tokenized` input as tokens here, as 
        well as the tokens and nominal indices corresponding to once the 
        is tokenized, the hyphenated subsections pulled out.
        The nom label is a [one-hot] binary vector, the same length as the 
        new_tokens, indicating the position to find arguments for. 
        The new_tags is the BIO labels for the new_tokens.

        let's have nom_label be corresponding to hyphenated, because sometimes nominal is just part of a word. TODO
        """

        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, end_offsets, start_offsets = self._wordpiece_tokenize_input(
                    [t.text for t in new_tokens]
                    )
            # end_offsets and start_offsets are computed to correspond to sentence with separated hyphens.
            new_nom = _convert_nom_indices_to_wordpiece_indices(nom_label, end_offsets)
            metadata_dict["offsets"] = start_offsets
            text_field = TextField(
                    [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                    token_indexers=self._token_indexers,
                    )
            nom_indicator = SequenceLabelField(new_nom, text_field)
        else: # Without a bert_tokenizer, just give it the new_tokens and corresponding information.
            text_field = TextField(new_tokens, token_indexers=self._token_indexers)
            nom_indicator = SequenceLabelField(nom_label, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = nom_indicator

        if all(x==0 for x in nom_label):
            nom = None
            nom_index = None
        else:
            nom_index = [i for i in range(len(nom_label)) if nom_label[i]==1] 
            nom = ''
            for n_idx in nom_index: # nom_index is indexed to words
                nom += new_tokens[n_idx].text
            # if includes mult tokens bc hyphenated separated

        metadata_dict["words"] = [x.text for x in new_tokens]
        metadata_dict["verb"] = nom
        metadata_dict["verb_index"] = nom_index

        if new_tags:
            if self.bert_tokenizer is not None:
                wordpieced_tags = _convert_tags_to_wordpiece_tags(new_tags, end_offsets)
                fields["tags"] = SequenceLabelField(wordpieced_tags, text_field)
            else:
                fields["tags"] = SequenceLabelField(new_tags, text_field)
            metadata_dict["gold_tags"] = new_tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)


