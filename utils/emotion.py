
import logging
import collections

logger = logging.getLogger(__name__)


class Emotion:
    def __init__(self, nrc_path, max_em_len, emotion_filters):
        self.nrc_path = nrc_path
        self.word_to_em, self.em_to_id = Emotion.load_nrc_emotion(self.nrc_path, emotion_filters)
        self.max_len = len(self.word_to_em) + 1
        self.max_em_len = 1

    @staticmethod
    def load_nrc_emotion(input_file, emotion_filters=None):
        '''
        load NRC file and load it to a dictionary!
        :param input_file:
        :return:
        '''
        ss = 0
        if emotion_filters:
            emotion_filters = [em.strip() for em in emotion_filters.split(',')]
        #print('Filtering emotions', emotion_filters)
        words_to_em = collections.defaultdict(lambda: ["UNKE"])
        emotion_to_id = dict()
        emotion_to_id["UNKE"] = 0
        idx = 1
        with open(input_file, 'r', encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if line != "":
                    if len(line.split('\t')) != 3:
                        print('Skipping line in lemotion lexicon!', line)
                    else:
                        word, emotion, val = line.split('\t')
                        if emotion_filters is None or (emotion_filters is not None and emotion not in emotion_filters):
                            if emotion not in emotion_to_id:
                                emotion_to_id[emotion] = idx
                                idx += 1
                            if int(val) == 1:
                                if word not in words_to_em:
                                    words_to_em[word] = [emotion]
                                else:
                                    words_to_em[word].append(emotion)
                        if emotion_filters is not None and emotion not in emotion_filters:
                            ss +=1
        #print('total number of emotion-filtered words', ss)
        #print('tottal number of emotion-unfiltered words', len(words_to_em))
        return words_to_em, emotion_to_id

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append([self.em_to_id[em] for em in self.word_to_em[token]])

        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def get_padded_ids(self, sequence):
        '''
        return emotion ids of the tokens in given sequence
        :param sequence:
        :param max_emotion: keep the total emotions for all tokens equal! padding 0 will be added if necessary
        :return:
        '''
        ids = self.convert_tokens_to_ids(sequence.split(" "))
        padded_ids = list()
        for gid in ids:
            gid = gid[: self.max_em_len]
            gid = [0] * (self.max_em_len - len(gid)) + gid
            padded_ids.append(gid)
        return padded_ids



