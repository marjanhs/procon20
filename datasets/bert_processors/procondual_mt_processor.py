import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class ProconDual_mtProcessor(BertProcessor):
    NAME = 'ProconDual_mt'
    NUM_CLASSES = 2 + 4
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir, name=None):
        name = 'train.tsv' if not name else name
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ProconDual_mt', name)), 'train')

    def get_dev_examples(self, data_dir, name=None):
        name = 'dev.tsv' if not name else name
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ProconDual_mt', name)), 'dev')

    def get_test_examples(self, data_dir, name=None):
        name = 'test.tsv' if not name else name
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ProconDual_mt', name)), 'test')

    def get_any_examples(self, data_dir, split):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'ProconDual_mt', split)), split)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]  #question
            label = line[0]
            text_b = line[2]  #opinion
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples