# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)
from models.bert.model import BertPreTrainedModel, BertModel

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

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
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, max_em_len, no_bert_layers=1, pooling=False):
        #print('before init')
        #print(config)
        super(BertForSequenceClassification, self).__init__(config)
        #print('after init')
        #print(config)
        self.num_labels = num_labels
        self.pooling = pooling
        self.bert = BertModel(config)
        self.no_bert_layers = no_bert_layers
        self.hidden_size = config.hidden_size
        self.emotion_embeddings = nn.Embedding(max_em_len, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_pool = nn.Dropout(0.1)
        if self.pooling:
            self.classifier = nn.Linear(1 * 3 * config.hidden_size, num_labels) # avd_pool max_pool, last hidden state
        else:
            self.classifier = nn.Linear(1 * config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

        # wo att
        self.lstm_final = nn.GRU(config.hidden_size * (abs(self.no_bert_layers) + 1), config.hidden_size, bidirectional=False)

        # w/ att
        '''self.lstm_final = nn.GRU(2 * config.hidden_size * (abs(self.no_bert_layers)), config.hidden_size,
                                 bidirectional=True)'''

        self.att_lin = nn.Linear(config.hidden_size * (abs(self.no_bert_layers) + 1), config.hidden_size * (abs(self.no_bert_layers)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, bs):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(1, bs, self.hidden_size).to(self.device)

    def get_att(self, hiddes, emotion_embd):

        concat = torch.cat([hiddes, emotion_embd], -1)
        g = self.att_lin(concat)
        alpha = F.softmax(g, dim=0)
        att_hidden = alpha * hiddes
        return att_hidden

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, emotion_ids=None, return_indices=False):

        bert_encoded_layers_raw, _ = self.bert(input_ids, token_type_ids, attention_mask) # 12 layers each (batch_size, seg_lengh, hidden_size)

        # disconnect the computational graph
        bert_encoded_layers_raw = [l.clone().detach() for l in bert_encoded_layers_raw]

        bert_encoded_layers = self.dropout(torch.cat(bert_encoded_layers_raw[-self.no_bert_layers:], dim=-1))
        emotion_embeddings = self.emotion_embeddings(emotion_ids).mean(dim=2)

        bs,_,_ = bert_encoded_layers.shape


        #concat = encoded_layers + emotion_embeddings
        #concat = encoded_layers


        #concat = torch.cat([bert_encoded_layers, sent_embeddings], -1)
        #concat = self.dropout(concat)
        #concat = concat.permute(1, 0, 2)  # convert it to (seq_len, batch, input_size)
        #out_score, _ = self.lstm(concat)  # out_score.shape = (seq_len, batch, input_size)

        # w/ att
        '''out_att = self.get_att(bert_encoded_layers, emotion_embeddings)
        concat = torch.cat([out_att, bert_encoded_layers], dim=-1).permute(1, 0, 2) #(bs, sl, hd_sz) => (sl, bs, hd_sz)'''

        # wo att
        concat = torch.cat([bert_encoded_layers, emotion_embeddings], -1).permute(1, 0, 2)

        hidden = self.init_hidden(bs)
        out_score, _ = self.lstm_final(concat, hidden)
        out_score = self.dropout(out_score) # added

        if self.pooling:
            _, bs, _ = out_score.shape
            if return_indices:

                mx_pool, mx_pool_idx = F.adaptive_max_pool1d_with_indices(out_score.permute(1, 2, 0), (1,))
            else:
                mx_pool = F.adaptive_max_pool1d(out_score.permute(1, 2, 0), (1,))
            avg_pool = F.adaptive_avg_pool1d(out_score.permute(1, 2, 0), (1,)) # (bs, input_size, seq_len)
            out_score = torch.cat([mx_pool.view(bs, -1), avg_pool.view(bs, -1), out_score[-1]], -1)
            out_score = self.dropout_pool(out_score)


        else:
            out_score = out_score[-1]

        logits = self.classifier(out_score)
        if return_indices and self.pooling:
            return logits, mx_pool_idx

        return logits




