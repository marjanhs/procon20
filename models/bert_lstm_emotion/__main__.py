import os
import random
import time

import torch
from pathlib import Path
from common.evaluators.bert_emotion_evaluator import BertEvaluator
from common.trainers.bert_emotion_trainer import BertTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.procon_processor import ProconProcessor
from datasets.bert_processors.procondual_processor import ProconDualProcessor
from datasets.bert_processors.perspectrum_processor import PerspectrumProcessor
from models.bert_lstm_emotion.args import get_args
from models.bert_lstm_emotion.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.optimization import BertAdam
from utils.tokenization import BertTokenizer
from datasets.bert_processors.abstract_processor import convert_examples_to_features_with_emotion


# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss   Dev/F1Ma   Dev/Ham   Dev/Jac'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.split(','))
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(x, values):


    #x = x[:10]
    #values = values[:10]
    # sphinx_gallery_thumbnail_number = 2
    lbl_plot = x
    d = pd.DataFrame(np.array(values).reshape(1, -1))
    fig, axes = plt.subplots(nrows=10, figsize=(10, 20))
    ax = sns.heatmap(d, cmap="Blues", square=True, annot=True, xticklabels=x, yticklabels=False,
                     vmax=np.max(values), cbar=False)
    ax.set_xticklabels(rotation=40, labels=lbl_plot)
    fig.savefig('example.png')


def analyze(evaluator, input_indices, predicted_labels, target_labels):
    eval_features = convert_examples_to_features_with_emotion(
        evaluator.eval_examples, evaluator.args.max_seq_length, evaluator.tokenizer, evaluator.emotioner)

    all_indices = []
    for i, f in enumerate(eval_features):
        indices = np.array(f.input_ids)[input_indices[i]]
        all_indices.append(indices)

    all_indices = np.array(all_indices)

    tp = np.where(predicted_labels==target_labels) and np.where(target_labels)[0]
    tn = np.where(predicted_labels==target_labels) and np.where(target_labels==0)[0]

    cp, cn = Counter(all_indices[tp].flatten().tolist()), Counter(all_indices[tn].flatten().tolist())

    p_most_commons_ids = [k for (k,v) in cp.most_common(20)]
    n_most_commons_ids = [k for (k,v) in cn.most_common(20)]

    p_most_commons_tokens = evaluator.tokenizer.convert_ids_to_tokens(p_most_commons_ids)
    n_most_commons_tokens = evaluator.tokenizer.convert_ids_to_tokens(n_most_commons_ids)
    print('top 10 most common tokens in TP: ', p_most_commons_tokens)

    print('top 10 most common tokens in TN: ', n_most_commons_tokens)


def analyze_example(evaluator, input_indices, predicted_labels, target_labels, path):
    eval_features = convert_examples_to_features_with_emotion(
        evaluator.eval_examples, evaluator.args.max_seq_length, evaluator.tokenizer, evaluator.emotioner)

    all_indices = input_indices
    '''for i, f in enumerate(eval_features):
        locs = np.array(input_indices[i])
        all_indices.append(indices)'''

    all_indices = np.array(all_indices)
    tp = np.where(predicted_labels==target_labels) and np.where(target_labels==0)[0]
    #tn = np.where(predicted_labels==target_labels) and np.where(target_labels==0)[0]

    ls = []

    for j in range(len(tp)):

        id_p_example = tp[j]
        #id_n_example = tn[0]


        example_pos = all_indices[id_p_example].flatten().tolist()
        #example_neg = all_indices[id_n_example].flatten().tolist()


        cp = Counter(example_pos)

        p_ids = np.array(eval_features[id_p_example].input_ids)


        p_tokens = evaluator.tokenizer.convert_ids_to_tokens(p_ids)


        freqs = []
        for i in range(len(eval_features[id_p_example].input_ids)):
            freqs.append(cp[i])

        assert len(freqs) == len(p_tokens)
        l = [p_tokens, freqs]
        ls.append(l)
    df = pd.DataFrame(ls, columns=['tokens', 'freqs'])
    df.to_csv(path / 'neg_examples.tsv', sep='\t', index=None)
    #plot_heatmap(p_tokens, freqs)



'''def evaluate_split(model, processor, args, split='dev', return_indices=False):
    evaluator = BertEvaluator(model, processor, args, split)
    start_time = time.time()

    outs = evaluator.get_scores(silent=True, return_indices=return_indices)[0]
    accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc = outs[:8]
    if return_indices:
        indices = outs[-1]
        accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc, predicted_labels, target_labels, indice = outs
        analyze_example(evaluator, indices, predicted_labels, target_labels, Path('out'))

    print("Inference time", time.time() - start_time)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc))'''


def evaluate_split(model, processor, args, split='dev', return_indices=False):
    evaluator = BertEvaluator(model, processor, args, split)
    start_time = time.time()
    accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc, predicted_values, target_values = evaluator.get_scores(silent=True)[0]
    print("Inference time", time.time() - start_time)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss, f1_mac, hamming_loss, jacc))

    model_name = args.save_path.replace('model_checkpoints/', '')
    path = Path(args.save_path.replace('model_checkpoints', 'out'))
    path = path / args.dataset
    path.mkdir(parents=True, exist_ok=True)
    print('Saving prediction files in ', path)
    np.save(path / f'predicted_{model_name}_{split}.npy', predicted_values)
    np.save(path / f'target_{model_name}_{split}.npy', target_values)


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
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'IMDB': IMDBProcessor,
        'AAPD': AAPDProcessor,
        'AGNews': AGNewsProcessor,
        'Sogou': SogouProcessor,
        'Procon': ProconProcessor,
        'ProconDual': ProconDualProcessor,
        'Perspectrum': PerspectrumProcessor
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
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir, args.train_name)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.model, cache_dir=cache_dir, num_labels=args.num_labels,
                                                          no_bert_layers=args.no_bert_layers, pooling=args.pooling,
                                                          max_em_len=args.max_em_len
                                                          )


    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Install NVIDIA Apex to use distributed and FP16 training.")
        model = DDP(model)
    '''elif n_gpu > 1: changed by marjan

        model = torch.nn.DataParallel(model)'''

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install NVIDIA Apex for distributed and FP16 training")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    trainer = BertTrainer(model, optimizer, processor, args)

    if not args.trained_model:
        trainer.train()
        #print('last epoch')
        #evaluate_split(model, processor, args, split=args.test_name, return_indices=False)
        print('loading trained model ', trainer.snapshot_path)
        model = torch.load(trainer.snapshot_path)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels,
                                                              no_bert_layers=args.no_bert_layers, pooling=args.pooling,
                                                          max_em_len=args.max_em_len)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state={}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)
    print('best epoch')
    evaluate_split(model, processor, args, split=args.dev_name, return_indices=False)
    evaluate_split(model, processor, args, split=args.test_name, return_indices=False)


if __name__ == "__main__":
    do_main()