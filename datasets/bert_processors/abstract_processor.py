import csv
import sys
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize
import nltk.tokenize as nltk_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


def split_sents_Overal_Score(string):
    r'''
    split sentences and assign the overall score to all tokens
    :param string:
    :return:
    '''

    sentences = nltk_tokenize.sent_tokenize(string)
    scores = []
    for st in sentences:
        score = analyzer.polarity_scores(st)['compound']
        # neutral: 0, positive: 1, negative : 2
        if score >= 0.05:
            score = 1
        elif score <= -0.05:
            score = -1
        else:
            score = 0
        scores.append(score)
    t_score = sum(scores)
    if t_score >0 :
        scores = [1] * len(sentences)
    elif t_score <0:
        scores = [2] * len(sentences)
    else :
        scores = [0] * len(sentences)

    return sentences, scores


def split_sents_Sent_Score(string):
    sentences = nltk_tokenize.sent_tokenize(string)
    scores = []
    for st in sentences:
        score = get_sent_score(st)
        scores.append(score)
    return sentences, scores


def get_sent_score(sentence):
    score = analyzer.polarity_scores(sentence)['compound']
    # neutral: 0, positive: 1, negative : 2
    if score >= 0.05:
        score = 1
    elif score <= -0.05:
        score = 2
    else:
        score = 0
    return score


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentiment_scores=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sentiment_scores = sentiment_scores




class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def convert_examples_to_stancy_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens_single = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_single = [0] * len(tokens_single)
        #tokens_double = tokens_single + tokens_b + ["[SEP]"]
        tokens_double = ["[CLS]"] + tokens_b + ["[SEP]"] + tokens_a + ["[SEP]"]
        segment_ids_double =  [1] * (len(tokens_b) + 1) + segment_ids_single

        input_ids_single = tokenizer.convert_tokens_to_ids(tokens_single)
        input_ids_double = tokenizer.convert_tokens_to_ids(tokens_double)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_single = [1] * len(input_ids_single)
        input_mask_double = [1] * len(input_ids_double)

        max_seq_length_single = max_seq_length

        # Zero-pad up to the sequence length.
        padding_single = [0] * (max_seq_length_single - len(input_ids_single))
        input_ids_single += padding_single
        input_mask_single += padding_single
        segment_ids_single += padding_single

        assert len(input_ids_single) == max_seq_length_single
        assert len(input_mask_single) == max_seq_length_single
        assert len(segment_ids_single) == max_seq_length_single

        padding_double = [0] * (max_seq_length - len(input_ids_double))
        input_ids_double += padding_double
        input_mask_double += padding_double
        segment_ids_double += padding_double

        assert len(input_ids_double) == max_seq_length
        assert len(input_mask_double) == max_seq_length
        assert len(segment_ids_double) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("single:")
            print("tokens: %s" % " ".join([str(x) for x in tokens_single]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids_single]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask_single]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids_single]))

            print('double:')
            print("tokens: %s" % " ".join([str(x) for x in tokens_double]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids_double]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask_double]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids_double]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids={"single":input_ids_single, "double": input_ids_double},
                                      input_mask={"single":input_mask_single, "double": input_mask_double},
                                      segment_ids={"single":segment_ids_single, "double": segment_ids_double},
                                      label_id=label_id))

    return features


def convert_examples_to_features_with_sentiment(examples, max_seq_length, tokenizer, print_examples=False, overal_sent=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :param overal_sent: whether choose overall sentiment of a sentence for tokens or not
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if overal_sent:
            tmp_scores_a = [get_sent_score(example.text_a)]
            tmp_sntcs_a = [example.text_a]
        else:
            tmp_sntcs_a, tmp_scores_a = split_sents_Sent_Score(example.text_a)
        tokens_a, scores_a = [], []
        for st, score in zip(tmp_sntcs_a, tmp_scores_a):
            tok_st = tokenizer.tokenize(st)
            score_st = [score] * len(tok_st)
            tokens_a += tok_st
            scores_a += score_st

        if example.text_b:
            #tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            if overal_sent:
                tmp_scores_b = [get_sent_score(example.text_b)]
                tmp_sntcs_b = [example.text_b]
            else:
                tmp_sntcs_b, tmp_scores_b = split_sents_Sent_Score(example.text_b)
            tokens_b, scores_b = [], []
            for st, score in zip(tmp_sntcs_b, tmp_scores_b):
                tok_st = tokenizer.tokenize(st)
                score_st = [score] * len(tok_st)
                tokens_b += tok_st
                scores_b += score_st
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_pair(scores_a, scores_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            scores = [0] + scores_a + [0] + scores_b + [0]
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                scores_a = scores_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            scores = [0] + scores_a + [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        scores += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(scores) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("sentiment_ids: %s" % " ".join([str(x) for x in scores]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      sentiment_scores=scores,
                                      label_id=label_id,
                                      ))
    return features


def convert_examples_to_features_with_emotion(examples, max_seq_length, tokenizer,  emotioner, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param emotioner: Emotion object to convert emotions to ids
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []

    for (ex_index, example) in enumerate(examples):

        emotion_ids_a = emotioner.get_padded_ids(example.text_a)
        tokens_a, scores_a = [], []
        for st, score in zip(example.text_a.split(" "), emotion_ids_a):
            tok_st = tokenizer.tokenize(st)
            score_st = [score] * len(tok_st)
            tokens_a += tok_st
            scores_a += score_st

        if example.text_b:
            #tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            emotion_ids_b = emotioner.get_padded_ids(example.text_b)
            tokens_b, scores_b = [], []
            for st, score in zip(example.text_b.split(" "), emotion_ids_b):
                tok_st = tokenizer.tokenize(st)
                score_st = [score] * len(tok_st)
                tokens_b += tok_st
                scores_b += score_st
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_pair(scores_a, scores_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            pad = [[0] * emotioner.max_em_len]
            scores = pad + scores_a + pad + scores_b + pad
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                scores_a = scores_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            pad = [[0] * emotioner.max_em_len]
            scores = pad + scores_a + pad
            #print('len tokens a', len(tokens), 'len scores a', len(scores))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_scores = [[0] * emotioner.max_em_len] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        scores += padding_scores

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(scores) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("sentiment_ids: %s" % " ".join([str(x) for x in scores]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      sentiment_scores=scores,
                                      label_id=label_id,
                                      ))
    return features


def convert_examples_to_stancy_features_with_emotion(examples, max_seq_length, tokenizer,  emotioner, print_examples=False):
    """
    TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1COMPLETE THIS FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param emotioner: Emotion object to convert emotions to ids
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []

    for (ex_index, example) in enumerate(examples):

        emotion_ids_a = emotioner.get_padded_ids(example.text_a)
        tokens_a, scores_a = [], []
        for st, score in zip(example.text_a.split(" "), emotion_ids_a):
            tok_st = tokenizer.tokenize(st)
            score_st = [score] * len(tok_st)
            tokens_a += tok_st
            scores_a += score_st

        if example.text_b:
            #tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            emotion_ids_b = emotioner.get_padded_ids(example.text_b)
            tokens_b, scores_b = [], []
            for st, score in zip(example.text_b.split(" "), emotion_ids_b):
                tok_st = tokenizer.tokenize(st)
                score_st = [score] * len(tok_st)
                tokens_b += tok_st
                scores_b += score_st
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_pair(scores_a, scores_b, max_seq_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            pad = [[0] * emotioner.max_em_len]
            scores = pad + scores_a + pad + scores_b + pad
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                scores_a = scores_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            pad = [[0] * emotioner.max_em_len]
            scores = pad + scores_a + pad
            #print('len tokens a', len(tokens), 'len scores a', len(scores))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_scores = [[0] * emotioner.max_em_len] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        scores += padding_scores

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(scores) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("sentiment_ids: %s" % " ".join([str(x) for x in scores]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      sentiment_scores=scores,
                                      label_id=label_id,
                                      ))
    return features


def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_a)]
        tokens_b = None

        if example.text_b:
            tokens_b = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP]
            for i0 in range(len(tokens_a)):
                if len(tokens_a[i0]) > max_seq_length - 2:
                    tokens_a[i0] = tokens_a[i0][:(max_seq_length - 2)]

        tokens = [["[CLS]"] + line + ["[SEP]"] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = list()
        for line in tokens:
            input_ids.append(tokenizer.convert_tokens_to_ids(line))

        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
