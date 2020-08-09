import os
import random
import time
import math
import numpy as np
import torch

from common.evaluators.bert_evaluator import BertEvaluator
from datasets.bert_processors.news_art_processor import News_artProcessor
from datasets.bert_processors.news_processor import News_Processor
from models.bert.args import get_args
from models.bert.model import BertForSequenceClassification
from utils.io import PYTORCH_PRETRAINED_BERT_CACHE
from utils.tokenization import BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f}'.split(','))

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', is_lowercase=True)

def get_feature_vector(evaluators, use_idf=False, ngram_range=(1, 1), max_seq_len=256):

    train_ev, dev_ev, test_ev = evaluators
    train_text, test_text, dev_text = [], [], []
    for i, x in enumerate(train_ev.eval_examples):
        #tokens_a = x.text_a.strip().split()
        tokens_a= [t for t in x.text_a.strip().split() if t not in ['', ' ']]
        #tokens_a = bert_tokenizer.tokenize(x.text_a)
        tokens_a = tokens_a[:max_seq_len]
        train_text.append(' '.join(tokens_a))

    for i, x in enumerate(test_ev.eval_examples):
        tokens_a = [t for t in x.text_a.strip().split() if t not in ['', ' ']]
        #tokens_a = bert_tokenizer.tokenize(x.text_a)
        tokens_a = tokens_a[:max_seq_len]
        test_text.append(' '.join(tokens_a))

    for i, x in enumerate(dev_ev.eval_examples):
        tokens_a = [t for t in x.text_a.strip().split() if t not in ['', ' ']]
        #tokens_a = bert_tokenizer.tokenize(x.text_a)
        tokens_a = tokens_a[:max_seq_len]
        dev_text.append(' '.join(tokens_a))


    tf_vect = TfidfVectorizer(use_idf=use_idf, ngram_range=ngram_range, binary=True) #change ! max_features=300 min_df=5
    train_xf = tf_vect.fit_transform(train_text)
    test_xf = tf_vect.transform(test_text)
    dev_xf = tf_vect.transform(dev_text)
    return train_xf, dev_xf, test_xf

def classification(train, test, cls='svc'):
    train_x, train_y = train
    test_x, test_y = test


    print('classifying ...')
    if cls == 'nb':
        clf = MultinomialNB().fit(train_x, train_y)
    elif cls == 'svc':
        clf = SVC(kernel='linear').fit(train_x, train_y)
    elif cls == 'lr':
        clf = LogisticRegression().fit(train_x, train_y)
    elif cls =='psvc':
        clf = Pipeline([('anova', SelectPercentile(chi2)),
                        ('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear'))])
        clf.set_params(anova__percentile=80)
        clf = clf.fit(train_x, train_y)
    else:
        raise Exception('Error in classifier name!')

    test_predicted = clf.predict(test_x)
    train_predicted = clf.predict(train_x)
    get_score(train_y, train_predicted)
    print(cls)
    return get_score(test_y, test_predicted)

def get_score(true_y, predicted_y, is_multilabel=False):
    avg = 'micro' if is_multilabel else 'binary'
    target_labels, predicted_labels = true_y, predicted_y
    accuracy = metrics.accuracy_score(target_labels, predicted_labels)
    precision = metrics.precision_score(target_labels, predicted_labels, average=avg)
    recall = metrics.recall_score(target_labels, predicted_labels, average=avg)
    f1 = metrics.f1_score(target_labels, predicted_labels, average=avg)
    d = {'acc': accuracy, 'pr': precision, 'rc': recall, 'f1': f1}
    return d


def evaluate(model, processor, args, last_bert_layers = -1, ngram_range=(1,1)):




    train_evaluator = BertEvaluator(model, processor, args, args.train_name)
    dev_evaluator = BertEvaluator(model, processor, args, args.dev_name)
    tst_evaluator = BertEvaluator(model, processor, args, args.test_name)

    start_time = time.time()
    train_layers, train_labels = train_evaluator.get_bert_layers(silent=True, last_bert_layers=last_bert_layers)
    dev_layers, dev_labels = dev_evaluator.get_bert_layers(silent=True, last_bert_layers=last_bert_layers)
    tst_layers, tst_labels = tst_evaluator.get_bert_layers(silent=True, last_bert_layers=last_bert_layers)

    train_xf, dev_xf, test_xf = get_feature_vector((train_evaluator, dev_evaluator, tst_evaluator), ngram_range=ngram_range, max_seq_len=args.max_seq_length)


    # train
    train_xf = train_xf.toarray()
    train_layers = train_layers.cpu().data.numpy()
    train_x = np.concatenate((train_layers, train_xf), axis=1)

    #dev
    dev_xf = dev_xf.toarray()
    dev_layers = dev_layers.cpu().data.numpy()
    dev_x = np.concatenate((dev_layers, dev_xf), axis=1)

    #test
    test_xf = test_xf.toarray()
    tst_layers = tst_layers.cpu().data.numpy()
    test_x = np.concatenate((tst_layers, test_xf), axis=1)


    #train, tst, dev = (train_x, train_labels), (test_x, tst_labels), (dev_x, dev_labels)
    train, tst, dev = (train_xf, train_labels), (test_xf, tst_labels), (dev_xf, dev_labels)
    #print('train labels length', len(train_labels), train_labels[0])
    #train, tst, dev = (train_layers, train_labels), (tst_layers, tst_labels), (dev_layers, dev_labels)
    scatter_plot(train, dev, tst)
    #train, tst, dev = (train_both_models, train_labels), (tst_both_models, tst_labels), (dev_both_models, dev_labels)


    print('train, test shape : ', train[0].shape, tst[0].shape)
    print("Inference time", time.time() - start_time)
    r_test = classification(train, tst)
    r_dev = classification(train, dev)
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format("DEV", r_dev['acc'],r_dev['pr'] , r_dev['rc'], r_dev['f1']))
    print(LOG_TEMPLATE.format("TEST", r_test['acc'], r_test['pr'], r_test['rc'], r_test['f1']))


def scatter_plot(train, dev, tst):

    '''print(train[0].shape)
    x = np.concatenate((train[0] , dev[0] , tst[0]))
    y = train[1] + dev[1] + tst[1]'''

    x_train, labels_train = train
    x_test , labels_test = tst

    x = np.concatenate((x_train, x_test))



    x = PCA(n_components=2).fit_transform(x)
    #x = TSNE(n_components=2).fit_transform(x)
    pos_train = np.array([t[0] for t in zip(x[:len(x_train),], labels_train) if t[1]==1])
    neg_train = np.array([t[0] for t in zip(x[:len(x_train),], labels_train)  if t[1] == 0])

    pos_test = np.array([t[0] for t in zip(x[len(x_train):,], labels_test) if t[1]==1])
    neg_test = np.array([t[0] for t in zip(x[len(x_train):,], labels_test)  if t[1] == 0])

    #plt.scatter(x[:, 0],x[:, 1], c=colors, alpha=0.5)
    plt.scatter(pos_train[:,0], pos_train[:, 1], c='k', marker='+', alpha=1)
    plt.scatter(neg_train[:, 0], neg_train[:, 1], c='b', marker='o', alpha=0.5)

    plt.scatter(pos_test[:,0], pos_test[:, 1], c='r', marker='+', alpha=1)
    plt.scatter(neg_test[:, 0], neg_test[:, 1], c='c', marker='o', alpha=0.5)
    plt.savefig('news_art_pers.png')



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

    num_train_optimization_steps = None
    if args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir, args.train_name)
        num_train_optimization_steps = int(
            math.ceil(len(train_examples) / args.batch_size) / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model_bert = BertForSequenceClassification.from_pretrained(args.model, num_labels=4)


    model_bert.to(device)

    if args.trained_model:
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)  # load personality model
        state={}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]

        del state['classifier.weight']  # removing  personality classifier!
        del state['classifier.bias']
        model_bert.load_state_dict(state, strict=False)
        model_bert = model_bert.to(device)
    args.freez_bert = False
    evaluate(model_bert, processor, args, last_bert_layers=-1, ngram_range=(1,1))


