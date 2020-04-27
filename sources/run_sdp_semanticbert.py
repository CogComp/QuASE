from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig)
from apply_models import SemanticBertForSDP
from modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, relations=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
        """
        self.guid = guid
        self.text = text
        self.relations = relations
        # tuple(argument, predicate, label)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, segment_ids, dependencies, dependency_labels):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.dependencies = dependencies
        self.dependency_labels = dependency_labels


def readfile(filename):
    f = open(filename)
    data = []
    for line in f.readlines():
        example = json.loads(line)
        data.append(example)
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class SDPProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.examples.sdp.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.examples.sdp.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.examples.sdp.jsonl")), "test")

    def get_labels(self):
        return ['compound', 'ARG1', 'measure', 'loc', 'ARG2', 'BV', 'of', 'appos', 'ARG3', 'mwe', 'poss', '_nor_c',
                '_and_c', 'times', 'than', 'part', 'subord', 'conj', 'comp', 'neg', '_or_c', '_but_c', 'comp_so',
                'comp_less', 'plus', 'ARG4', '_rather+than_c', '_as+well+as_c', 'comp_enough', 'temp', 'discourse',
                'comp_too', 'parenthetical', '_but+not_c', '_and+so_c', '_but+also_c', '_not_c', '_then_c', 'manner',
                '_and+also_c', '_except_c', '_and+then_c', '_and+not_c', '_yet_c', '_versus_c', '_even_c',
                '_instead+of_c', '_plus_c', '_and+thus_c', '_minus_c', '_and+yet_c', '_after_c', '_if+not_c',
                '_not+to+mention_c', '_though_c', 'comp_not+so', '_much+less_c', '_formerly_c', '_except+that_c']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sent_id, sentence, relations) in enumerate(lines):
            guid = "%s-%s" % (set_type, sent_id)
            text = ' '.join(sentence)
            examples.append(InputExample(guid=guid, text=text, relations=relations))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text.split(' ')
        relationslist = example.relations
        # word level
        tokens = []
        orig_to_tok_index = {}
        for i, word in enumerate(textlist):
            orig_to_tok_index[i] = len(tokens)
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        relations = []
        for arg, pred, role in relationslist:
            relations.append((orig_to_tok_index[arg], orig_to_tok_index[pred], label_map[role]))
        # token level
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        relation_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        for arg, pred, role in relations:
            new_arg = arg + 1
            new_pred = pred + 1
            if new_arg < max_seq_length - 1 and new_pred < max_seq_length - 1:
                relation_ids.append((new_arg, new_pred, role))
        # input format
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        dependencies = [0] * (max_seq_length * max_seq_length)
        dependency_labels = [0] * (max_seq_length * max_seq_length)
        for arg, pred, role in relation_ids:
            dependencies[arg * max_seq_length + pred] = 1
            dependency_labels[arg * max_seq_length + pred] = role

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(tokens=ntokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          dependencies=dependencies,
                          dependency_labels=dependency_labels))
    return features


def get_results(y_true, y_pred):
    label_correct_num = 0
    correct_num = 0
    pred_num = 0
    gold_num = 0
    for index in range(len(y_true)):
        gold_relations = y_true[index]
        pred_relations = y_pred[index]
        for relation in gold_relations:
            if relation in pred_relations:
                label_correct_num += 1
        pred_num += len(pred_relations)
        gold_num += len(gold_relations)
        unlabeled_gold_relations = []
        for relation in gold_relations:
            unlabeled_gold_relations.append((relation[0], relation[1]))
        unlabeled_pred_relations = []
        for relation in pred_relations:
            unlabeled_pred_relations.append((relation[0], relation[1]))
        for relation in unlabeled_gold_relations:
            if relation in unlabeled_pred_relations:
                correct_num += 1
    if pred_num == 0:
        label_precision = 0
        precision = 0
    else:
        label_precision = label_correct_num / pred_num
        precision = correct_num / pred_num
    label_recall = label_correct_num / gold_num
    recall = correct_num / gold_num
    if label_correct_num == 0:
        label_F1 = 0.0
    else:
        label_F1 = 2 * label_precision * label_recall / (label_precision + label_recall)
    if correct_num == 0:
        F1 = 0.0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    print('pred_num', pred_num)
    print('gold_num', gold_num)
    print('correct_num', correct_num)
    print('label_correct_num', label_correct_num)
    return precision, recall, F1, label_precision, label_recall, label_F1


def write_predictions(examples, y_true, y_pred, output_file):
    fout = open(output_file, 'w')
    for index in range(len(examples)):
        text = examples[index].tokens
        gold_relations = y_true[index]
        pred_relations = y_pred[index]
        print(text)
        print(gold_relations)
        print(pred_relations)
        fout.write(json.dumps(text) + "\n")
        fout.write(json.dumps(gold_relations) + "\n")
        fout.write(json.dumps(pred_relations) + "\n")
    fout.close()


def get_tuples(dependencies, dependency_labels, input_mask):
    right_max = 0
    for x in range(len(input_mask)):
        if input_mask[x] == 0:
            right_max = x - 1
            break
    max_seq_length = len(input_mask)
    relations = []
    for i in range(right_max)[1:-1]:
        for j in range(right_max)[1:-1]:
            index = max_seq_length * i + j
            if dependencies[index] == 1:
                relations.append((int(i), int(j), int(dependency_labels[index])))
    return relations


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--percent",
                        default=100,
                        type=int,
                        help="The percentage of examples used in the training data.\n")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--pretrain',
                        action='store_true',
                        help="Whether to load a pre-trained model for continuing training")
    parser.add_argument('--pretrained_model_file',
                        type=str,
                        help="The path of the pretrained_model_file")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"sdp": SDPProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_examples = train_examples[:int(len(train_examples) * args.percent / 100)]
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = SemanticBertForSDP.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)

    # load pretrianed bert parameters to cls_bert
    model.cls_bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)

    if args.pretrain:
        # Load a pre-trained model
        print('load a pre-trained model from ' + args.pretrained_model_file)
        pretrained_state_dict = torch.load(args.pretrained_model_file)
        model_state_dict = model.state_dict()
        print('cls_bert', model_state_dict['cls_bert.encoder.layer.4.attention.self.value.weight'])
        print('bert', model_state_dict['bert.encoder.layer.4.attention.self.value.weight'])
        print('pretrained_state_dict', pretrained_state_dict.keys())
        print('model_state_dict', model_state_dict.keys())
        pretrained_state = {k: v for k, v in pretrained_state_dict.items() if
                            k in model_state_dict and v.size() == model_state_dict[k].size()}
        print('updated pretrained dict', pretrained_state.keys())
        model_state_dict.update(pretrained_state)
        print('updated_state_dict', model_state_dict.keys())
        model.load_state_dict(model_state_dict)
        model.to(device)
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.hidden_ESRL.parameters():
        param.requires_grad = False
    for param in model.cls_bert.parameters():
        param.requires_grad = False
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_dependencies = torch.tensor([f.dependencies for f in train_features], dtype=torch.long)
        all_dependency_labels = torch.tensor([f.dependency_labels for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_dependencies,
                                   all_dependency_labels)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, dependencies, dependency_labels = batch
                loss = model(input_ids, segment_ids, input_mask, dependencies, dependency_labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print('train loss', tr_loss)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i: label for i, label in enumerate(label_list, 0)}
        model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                        "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
        # Load a trained model and config that you have fine-tuned
    else:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        model_state_dict = torch.load(output_model_file)
        model = SemanticBertForSDP.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_dependencies = torch.tensor([f.dependencies for f in eval_features], dtype=torch.long)
        all_dependency_labels = torch.tensor([f.dependency_labels for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_dependencies,
                                  all_dependency_labels)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 0)}
        for input_ids, input_mask, segment_ids, dependencies, dependency_labels in tqdm(eval_dataloader,
                                                                                        desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            dependencies = dependencies.to(device)
            dependency_labels = dependency_labels.to(device)

            with torch.no_grad():
                dep_logits, label_logits = model(input_ids, segment_ids, input_mask)

            dependencies = dependencies.to('cpu').numpy()
            dependency_labels = dependency_labels.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                # gold_relations = get_tuples(dependencies[i], dependency_labels[i], input_mask[i])
                # pred_relations = get_tuples(dep_logits[i], label_logits[i], input_mask[i])
                # y_true.append(gold_relations)
                # y_pred.append(pred_relations)
                y_true.append((dependencies[i], dependency_labels[i], input_mask[i]))
                y_pred.append((dep_logits[i], label_logits[i], input_mask[i]))
        gold_relations = []
        pred_relations = []
        for index in range(len(y_true)):
            # print('feature index', index)
            # print(y_true[index][0].shape)
            # print(y_true[index][1].shape)
            # print(y_pred[index][0].shape)
            # print(y_pred[index][1].shape)
            gold_relations.append(get_tuples(y_true[index][0], y_true[index][1], y_true[index][2]))
            pred_relations.append(get_tuples(y_pred[index][0], y_pred[index][1], y_pred[index][2]))
        precision, recall, F1, label_precision, label_recall, label_F1 = get_results(gold_relations, pred_relations)
        print('precision', precision)
        print('recall', recall)
        print('F1', F1)
        print('label precision', label_precision)
        print('label recall', label_recall)
        print('label F1', label_F1)
        prediction_file = os.path.join(args.output_dir, 'predictions.txt')
        write_predictions(eval_features, gold_relations, pred_relations, prediction_file)


if __name__ == "__main__":
    main()