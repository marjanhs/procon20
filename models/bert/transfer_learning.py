import os
import random
import time
import math
import numpy as np
import torch

from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from datasets.bert_processors.news_art_processor import News_artProcessor
from datasets.bert_processors.news_processor import News_Processor
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.optimization import BertAdam
from utils.tokenization import BertTokenizer

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss   Dev/F1Ma   Dev/Ham   Dev/Jac'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.split(','))


def evaluate_split(model, processor, args, split='dev'):
    evaluator = BertEvaluator(model, processor, args, split)
    start_time = time.time()
    accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc, _ ,_  = evaluator.get_scores(silent=True)[0]
    print("Inference time", time.time() - start_time)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc))


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

        'News_art': News_artProcessor,
        'News': News_Processor
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
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    processor = dataset_map[args.dataset]()
    args.is_lowercase = 'uncased' in args.model
    args.is_hierarchical = False
    tokenizer = BertTokenizer.from_pretrained(args.model, is_lowercase=args.is_lowercase)

    train_examples = None
    num_train_optimization_steps = None
    if args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir, args.train_name)
        num_train_optimization_steps = int(
            math.ceil(len(train_examples) / args.batch_size) / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)  # creating news model!
    #model = BertForSequenceClassification.from_pretrained(args.model, cache_dir=cache_dir, num_labels=args.num_labels)

    if args.fp16:
        model.half()
    model.to(device)

    #model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)  # load personality model
    state={}
    for key in model_.state_dict().keys():
        new_key = key.replace("module.", "")
        state[new_key] = model_.state_dict()[key]

    del state['classifier.weight']  # removing  personality classifier!
    del state['classifier.bias']
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    print('t_total :', num_train_optimization_steps)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    args.freez_bert = False
    trainer = BertTrainer(model, optimizer, processor, args)

    trainer.train()
    model = torch.load(trainer.snapshot_path)

    evaluate_split(model, processor, args, split=args.dev_name)
    evaluate_split(model, processor, args, split=args.test_name)

