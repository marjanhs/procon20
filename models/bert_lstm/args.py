import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--dataset', type=str, default='SST-2', choices=['SST-2', 'AGNews', 'Reuters', 'AAPD', 'IMDB',
                                                                         'Yelp2014',
                                                                         'Procon', 'ProconDual'])
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'bert_lstm'))
    parser.add_argument('--cache-dir', default='cache', type=str)
    parser.add_argument('--trained-model', default=None, type=str)
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--fp16', action='store_true', help='use 16-bit floating point precision')

    parser.add_argument('--max-seq-length',
                        default=128,
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')

    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')

    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument('--loss-scale',
                        type=float,
                        default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')

    parser.add_argument("--train-name", default=None, type=str)
    parser.add_argument("--dev-name", default='dev.tsv', type=str)
    parser.add_argument("--test-name", default='test.tsv', type=str)
    parser.add_argument("--pooling", action='store_true')


    args = parser.parse_args()
    return args
