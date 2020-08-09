import datetime
import os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from common.evaluators.bert_emotion_evaluator import BertEvaluator
from datasets.bert_processors.abstract_processor import convert_examples_to_features_with_emotion
from datasets.bert_processors.abstract_processor import convert_examples_to_hierarchical_features
from utils.optimization import warmup_linear
from utils.preprocessing import pad_input_matrix
from utils.tokenization import BertTokenizer
from utils.emotion import Emotion


class BertTrainer(object):

    def __init__(self, model, optimizer, processor, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.train_examples = self.processor.get_train_examples(args.data_dir, args.train_name)
        self.tokenizer = BertTokenizer.from_pretrained(args.model, is_lowercase=args.is_lowercase)
        self.emotioner = Emotion(args.nrc_path, args.max_em_len, args.emotion_filters)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, '%s.pt' % timestamp)
        print('initial snapshot path', self.snapshot_path)

        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            self.num_train_optimization_steps = args.num_train_optimization_steps // torch.distributed.get_world_size()

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss   Dev/F1ma, Dev/HLoss, Dev/Jacc, Train/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.split(','))

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_measure, self.unimproved_iters = 0, 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        loss_epoch = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, emotion_ids, label_ids = batch

            logits = self.model(input_ids, segment_ids, input_mask, emotion_ids=emotion_ids)
            loss_extra = 0
            if isinstance(logits, tuple):
                logits , (first_SEP, second_SEP) = logits
                cos_simi = F.cosine_similarity(first_SEP, second_SEP)
                for i in range(len(label_ids)):
                    if torch.eq(label_ids[i], torch.Tensor([0,1]).long().cuda()).all():
                        loss_extra += 1 - cos_simi[i]
                    elif torch.eq(label_ids[i], torch.Tensor([1, 0]).long().cuda()).all():
                        loss_extra += max(0, cos_simi[i])
                    else:
                        print('Invalid label value ERROR', label_ids[i])
                        exit(1)

            if self.args.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())
            else:
                loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))
                loss += loss_extra

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()

            self.tr_loss += loss.item()
            loss_epoch += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    lr_this_step = self.args.learning_rate * warmup_linear(self.iterations / self.num_train_optimization_steps, self.args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.iterations += 1
        return loss_epoch / (step+1)

    def train(self):

        if self.args.is_hierarchical:
            train_features = convert_examples_to_hierarchical_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)
        else:
            train_features = convert_examples_to_features_with_emotion(
                self.train_examples, self.args.max_seq_length, self.tokenizer, self.emotioner)

        unpadded_input_ids = [f.input_ids for f in train_features]
        unpadded_input_mask = [f.input_mask for f in train_features]
        unpadded_segment_ids = [f.segment_ids for f in train_features]
        unpadded_emotion_scores = [f.sentiment_scores for f in train_features] # sentiment_scores are emotion_ids!

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        padded_emotion_ids = torch.tensor(unpadded_emotion_scores, dtype=torch.long)

        train_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, padded_emotion_ids, label_ids)

        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            loss_epoch = self.train_epoch(train_dataloader)
            dev_evaluator = BertEvaluator(self.model, self.processor, self.args, split='dev')
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, dev_f1_macro, dev_hamming_loss, dev_jaccard_score, dev_predicted_labels, dev_target_labels = dev_evaluator.get_scores()[0]

            # Print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, dev_f1_macro, dev_hamming_loss, dev_jaccard_score, loss_epoch))

            if self.args.early_on_f1:
                dev_measure = dev_f1
                measure_name = 'F1'
            else:
                dev_measure = dev_acc
                measure_name = 'Balanced Acc'

            # Update validation results
            if dev_measure > self.best_dev_measure:
                self.unimproved_iters = 0
                self.best_dev_measure = dev_measure
                torch.save(self.model, self.snapshot_path)

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    print("Early Stopping. Epoch: {}, Best {}: {}".format(epoch, measure_name, self.best_dev_measure))
                    break



