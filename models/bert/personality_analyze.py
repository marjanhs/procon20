import os
import random
import time

import numpy as np
import torch

from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.yelp2014_processor import Yelp2014Processor
from datasets.bert_processors.personality_processor import PersonalityProcessor
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.optimization import BertAdam
from utils.tokenization import BertTokenizer
from pathlib import Path

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss   Dev/F1Ma   Dev/Ham   Dev/Jac'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.split(','))


def do_personality_analysis(labels, set_name, path):
    print(labels[0].shape, labels[1].shape)
    print('files saved at ', path)
    np.save(path / f'predicted_{set_name}.npy', labels[0],)
    np.save(path / f'target_{set_name}.npy', labels[1])


def evaluate_split(model, processor, args, split='dev'):
    root = Path('out/bert')
    root.mkdir(exist_ok=True)
    evaluator = BertEvaluator(model, processor, args, split)
    start_time = time.time()
    accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc, predicted_labels, target_labels = evaluator.get_scores(silent=True)[0]
    '''print("Inference time", time.time() - start_time)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc))'''
    do_personality_analysis([predicted_labels, target_labels], split, root)


def do_main():
    # Set default configuration in args.py
    args = get_args()

    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('Distributed training:', bool(args.local_rank != -1))
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {

        'Personality': PersonalityProcessor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL

    if not args.trained_model:
        raise Exception('This method only works wit pre-trained models!')

    processor = dataset_map[args.dataset]()
    args.is_lowercase = 'uncased' in args.model
    args.is_hierarchical = False

    if args.trained_model:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

        #evaluate_split(model, processor, args, split='dev')
        #evaluate_split(model, processor, args, split='test')
        evaluate_split(model, processor, args, split=args.analyze_split)

