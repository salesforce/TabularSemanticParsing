"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Check if an input question can be translated to SQL and extract the span that causes the question to be untranslatable.
"""

import json
import numpy as np
import os
import random
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

import src.common.lr_scheduler as lrs
from src.common.nn_modules import *
import src.common.ops as ops
from src.data_processor.processor_utils import get_table_aware_transformer_encoder_inputs
from src.data_processor.schema_loader import load_schema_graphs
import src.data_processor.tokenizers as tok
from src.trans_checker.args import args
from src.utils.trans import bert_utils as bu
import src.utils.utils as utils

torch.cuda.set_device('cuda:{}'.format(args.gpu))
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpanExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.start_pred = Linear(input_dim, 1)
        self.end_pred = Linear(input_dim, 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_hiddens, text_masks):
        """
        :param encoder_hiddens: [batch_size, encoder_seq_len, hidden_dim]
        :param text_masks: [batch_size, text_len + text_start_offset]
        """
        # [batch_size, encoder_seq_len]
        start_potential = self.start_pred(encoder_hiddens).squeeze(2)
        end_potential = self.end_pred(encoder_hiddens).squeeze(2)
        start_logit = self.log_softmax(start_potential - text_masks * ops.HUGE_INT)
        end_logit = self.log_softmax(end_potential - text_masks * ops.HUGE_INT)
        return torch.cat([start_logit.unsqueeze(1), end_logit.unsqueeze(1)], dim=1)


class TranslatabilityChecker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrained_transformer = args.pretrained_transformer
        self.pretrained_lm_dropout = args.pretrained_lm_dropout_rate
        self.transformer_encoder = TransformerHiddens(self.pretrained_transformer, dropout=self.pretrained_lm_dropout,
                                                      requires_grad=True)
        self.encoder_hidden_dim = args.encoder_hidden_dim if args.encoder_hidden_dim > 0 else args.encoder_input_dim
        self.translatability_pred = nn.Sequential(nn.Linear(self.encoder_hidden_dim, 1), nn.Sigmoid())
        self.span_extractor = SpanExtractor(self.encoder_hidden_dim)

    def forward(self, input_ids, text_masks):
        inputs, input_masks = input_ids
        batch_size = len(inputs)
        segment_ids, position_ids = self.get_segment_and_position_ids(inputs)
        inputs_embedded, _ = self.transformer_encoder(inputs, input_masks,
                                                      segments=segment_ids, position_ids=position_ids)
        text_masks = torch.cat([ops.zeros_var_cuda([batch_size, 1]), text_masks.float()], dim=1)
        text_embedded = inputs_embedded[:, :text_masks.size(1), :]
        output = self.translatability_pred(text_embedded[:, 0, :])
        span_extractor_output = self.span_extractor(text_embedded, text_masks)
        return output, span_extractor_output

    def inference(self, dev_data):
        self.eval()
        batch_size = min(len(dev_data), 16)

        outputs, output_spans = [], []
        for batch_start_id in tqdm(range(0, len(dev_data), batch_size)):
            mini_batch = dev_data[batch_start_id: batch_start_id + batch_size]
            _, text_masks = ops.pad_batch([exp.text_ids for exp in mini_batch], bu.pad_id)
            encoder_input_ids = ops.pad_batch([exp.ptr_input_ids for exp in mini_batch], bu.pad_id)
            # [batch_size, 2, encoder_seq_len]
            output, span_extract_output = self.forward(encoder_input_ids, text_masks)
            outputs.append(output)
            encoder_seq_len = span_extract_output.size(2)
            # [batch_size, encoder_seq_len]
            start_logit = span_extract_output[:, 0, :]
            end_logit = span_extract_output[:, 1, :]
            # [batch_size, encoder_seq_len, encoder_seq_len]
            span_logit = start_logit.unsqueeze(2) + end_logit.unsqueeze(1)
            valid_span_pos = ops.ones_var_cuda([len(span_logit), encoder_seq_len, encoder_seq_len]).triu()
            span_logit = span_logit - (1 - valid_span_pos) * ops.HUGE_INT

            for i in range(len(mini_batch)):
                span_pos = span_logit[i].argmax()
                start = int(span_pos / encoder_seq_len)
                end = int(span_pos % encoder_seq_len)
                output_spans.append((start, end))
        return torch.cat(outputs), output_spans

    def get_segment_and_position_ids(self, encoder_input_ids):
        batch_size, input_size = encoder_input_ids.size()
        position_ids = ops.arange_cuda(input_size).unsqueeze(0).expand_as(encoder_input_ids)
        # [CLS] t1 t2 ... [SEP] ...
        # 0     0  0  ...  0    ...
        seg1_sizes = torch.nonzero(encoder_input_ids == bu.sep_id)[:, 1].view(batch_size, 2)[:, 0] + 1
        segment_ids = (position_ids >= seg1_sizes.unsqueeze(1)).long()
        # position_ids = position_ids * (1 - segment_ids) + (self.mdl.max_in_seq_len - 1) * segment_ids
        position_ids = None
        return segment_ids, position_ids

    def load_checkpoint(self, in_tar):
        if os.path.isfile(in_tar):
            print('=> loading checkpoint \'{}\''.format(in_tar))
            checkpoint = torch.load(in_tar)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('=> no checkpoint found at \'{}\''.format(in_tar))

    def save_checkpoint(self, optimizer, lrs, out_tar):
        ckpt = dict()
        ckpt['model_state_dict'] = self.state_dict()
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
        ckpt['lr_scheduler_dict'] = lrs.state_dict()
        torch.save(ckpt, out_tar)
        print('Checkpoint saved to {}'.format(out_tar))


class Example(object):
    def __init__(self, text, schema):
        self.text = text
        self.schema = schema
        self.text_tokens = []
        self.input_tokens = []
        self.text_ids = []
        self.ptr_input_ids = []
        self.span_ids = []


def run_train():
    dataset = load_data(args)
    train_data = dataset['train']
    dev_data = dataset['dev']

    with torch.set_grad_enabled(True):
        train(train_data, dev_data)


def train(train_data, dev_data):
    # Model
    model_dir = get_model_dir(args)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    trans_checker = TranslatabilityChecker(args)
    trans_checker.cuda()
    ops.initialize_module(trans_checker, 'xavier')

    wandb.init(project='translatability-prediction', name=get_wandb_tag(args))
    wandb.watch(trans_checker)

    # Hyperparameters
    batch_size = min(len(train_data), 12)
    num_peek_epochs = 1

    # Loss function
    loss_fun = nn.BCELoss()
    span_extract_pad_id = -100
    span_extract_loss_fun = MaskedCrossEntropyLoss(span_extract_pad_id)

    # Optimizer
    optimizer = optim.Adam(
        [{'params': [p for n, p in trans_checker.named_parameters() if not 'trans_parameters' in n and p.requires_grad]},
         {'params': [p for n, p in trans_checker.named_parameters() if 'trans_parameters' in n and p.requires_grad],
          'lr': args.bert_finetune_rate}],
        lr=args.learning_rate)
    lr_scheduler = lrs.LinearScheduler(
        optimizer, [args.warmup_init_lr, args.warmup_init_ft_lr], [args.num_warmup_steps, args.num_warmup_steps],
        args.num_steps)

    best_dev_metrics = 0
    for epoch_id in range(args.num_epochs):
        random.shuffle(train_data)
        trans_checker.train()
        optimizer.zero_grad()

        epoch_losses = []

        for i in tqdm(range(0, len(train_data), batch_size)):
            wandb.log({'learning_rate/{}'.format(args.dataset_name): optimizer.param_groups[0]['lr']})
            wandb.log({'fine_tuning_rate/{}'.format(args.dataset_name): optimizer.param_groups[1]['lr']})
            mini_batch = train_data[i : i + batch_size]
            _, text_masks = ops.pad_batch([exp.text_ids for exp in mini_batch], bu.pad_id)
            encoder_input_ids = ops.pad_batch([exp.ptr_input_ids for exp in mini_batch], bu.pad_id)
            target_ids = ops.int_var_cuda([1 if exp.span_ids[0] == 0 else 0 for exp in mini_batch])
            target_span_ids, _ = ops.pad_batch([exp.span_ids for exp in mini_batch], bu.pad_id)
            target_span_ids = target_span_ids * (1 - target_ids.unsqueeze(1)) + \
                              target_ids.unsqueeze(1).expand_as(target_span_ids) * span_extract_pad_id
            output, span_extract_output = trans_checker(encoder_input_ids, text_masks)
            loss = loss_fun(output, target_ids.unsqueeze(1).float())
            span_extract_loss = span_extract_loss_fun(span_extract_output, target_span_ids)
            loss += span_extract_loss
            loss.backward()
            epoch_losses.append(float(loss))

            if args.grad_norm > 0:
                nn.utils.clip_grad_norm_(trans_checker.parameters(), args.grad_norm)
            lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            if args.num_epochs % num_peek_epochs == 0:
                stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(epoch_losses))
                print(stdout_msg)
                wandb.log({'cross_entropy_loss/{}'.format(args.dataset_name): np.mean(epoch_losses)})
                pred_trans, pred_spans = trans_checker.inference(dev_data)
                targets = [1 if exp.span_ids[0] == 0 else 0 for exp in dev_data]
                target_spans = [exp.span_ids for exp in dev_data]
                trans_acc = translatablity_eval(pred_trans, targets)
                print('Dev translatability accuracy = {}'.format(trans_acc))
                if trans_acc > best_dev_metrics:
                    model_path = os.path.join(model_dir, 'model-best.tar')
                    trans_checker.save_checkpoint(optimizer, lr_scheduler, model_path)
                    best_dev_metrics = trans_acc
                span_acc, prec, recall, f1 = span_eval(pred_spans, target_spans)
                print('Dev span accuracy = {}'.format(span_acc))
                print('Dev span precision = {}'.format(prec))
                print('Dev span recall = {}'.format(recall))
                print('Dev span F1 = {}'.format(f1))
                wandb.log({'translatability_accuracy/{}'.format(args.dataset_name): trans_acc})
                wandb.log({'span_accuracy/{}'.format(args.dataset_name): span_acc})
                wandb.log({'span_f1/{}'.format(args.dataset_name): f1})


def run_inference():
    dataset = load_data(args)
    dev_data = dataset['dev']

    # Model
    model_dir = get_model_dir(args)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, 'model-best.tar')
    trans_checker = TranslatabilityChecker(args)
    trans_checker.load_checkpoint(model_path)
    trans_checker.cuda()
    trans_checker.eval()

    with torch.no_grad():
        pred_trans, pred_spans = trans_checker.inference(dev_data)
        targets = [1 if exp.span_ids[0] == 0 else 0 for exp in dev_data]
        target_spans = [exp.span_ids for exp in dev_data]
        trans_acc = translatablity_eval(pred_trans, targets)
        print('Dev translatability accuracy = {}'.format(trans_acc))
        span_acc, prec, recall, f1 = span_eval(pred_spans, target_spans)
        print('Dev span accuracy = {}'.format(span_acc))
        print('Dev span precision = {}'.format(prec))
        print('Dev span recall = {}'.format(recall))
        print('Dev span F1 = {}'.format(f1))


def load_data(args):
    def load_split(in_json):
        examples = []
        with open(in_json) as f:
            content = json.load(f)
        for exp in tqdm(content):
            question = exp['question']
            question_tokens = exp['question_toks']
            db_name = exp['db_id']
            schema = schema_graphs[db_name]
            example = Example(question, schema)
            text_tokens = bu.tokenizer.tokenize(question)
            example.text_tokens = text_tokens
            example.text_ids = bu.tokenizer.convert_tokens_to_ids(example.text_tokens)
            schema_features, _ = schema.get_serialization(bu, flatten_features=True,
                                                          question_encoding=question,
                                                          top_k_matches=args.top_k_picklist_matches)
            example.input_tokens, _, _, _ = get_table_aware_transformer_encoder_inputs(
                text_tokens, text_tokens, schema_features, bu)
            example.ptr_input_ids = bu.tokenizer.convert_tokens_to_ids(example.input_tokens)
            if exp['untranslatable']:
                modify_span = exp['modify_span']
                if modify_span[0] == -1:
                    example.span_ids = [1, len(text_tokens)]
                else:
                    assert(modify_span[0] >= 0 and modify_span[1] >= 0)
                    span_ids = utils.get_sub_token_ids(question_tokens, modify_span, bu)
                    if span_ids[0] >= len(text_tokens) or span_ids[1] > len(text_tokens):
                        a, b = span_ids
                        while(a >= len(text_tokens)):
                            a -= 1
                        while(b > len(text_tokens)):
                            b -= 1
                        span_ids = (a, b)
                    example.span_ids = [span_ids[0] + 1, span_ids[1]]
            else:
                example.span_ids = [0, 0]
            examples.append(example)
        print('{} examples loaded from {}'.format(len(examples), in_json))
        return examples

    data_dir = args.data_dir
    train_json = os.path.join(data_dir, 'train_ut.json')
    dev_json = os.path.join(data_dir, 'dev_ut.json')
    text_tokenize, _, _, _ = tok.get_tokenizers(args)

    schema_graphs = load_schema_graphs(args)
    schema_graphs.lexicalize_graphs(tokenize=text_tokenize, normalized=True)
    if args.train:
        train_data = load_split(train_json)
    else:
        train_data = None
    dev_data = load_split(dev_json)
    dataset = dict()
    dataset['train'] = train_data
    dataset['dev'] = dev_data
    dataset['schema'] = schema_graphs
    return dataset


def translatablity_eval(pred_trans, targets):
    thresh = 0.5
    assert(len(pred_trans) == len(targets))
    num_correct = 0
    for i, (pred_tran, target) in enumerate(zip(pred_trans, targets)):
        pred_tran_bi = int(pred_tran > thresh)
        if pred_tran_bi == target:
            num_correct += 1
    return num_correct / len(pred_trans)


def span_eval(pred_spans, gt_spans):
    assert(len(pred_spans) == len(gt_spans))
    num_correct = 0
    precs, recalls, f1s = [], [], []
    for i, (pred_span, gt_span) in enumerate(zip(pred_spans, gt_spans)):
        if tuple(pred_span) == tuple(gt_span):
            num_correct += 1
        st = max(pred_span[0], gt_span[0])
        ed = min(pred_span[1], gt_span[1])
        overlapped = (ed + 1) - st
        if overlapped < 0:
            overlapped = 0
        p = overlapped / (pred_span[1] + 1 - pred_span[0])
        r = overlapped / (gt_span[1] + 1 - gt_span[0])
        if (p + r) == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        precs.append(p)
        recalls.append(r)
        f1s.append(f1)
    acc = num_correct / len(pred_spans)
    prec = np.mean(precs)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    return acc, prec, recall, f1


def get_model_dir(args):
    model_root_dir = args.model_dir
    model_sub_dir = 'trans_check_{}_{}_{}'.format(args.learning_rate, args.bert_finetune_rate, args.num_warmup_steps)
    return os.path.join(model_root_dir, model_sub_dir)


def get_wandb_tag(args):
    return '{}-{}'.format(args.learning_rate, args.bert_finetune_rate)


if __name__ == '__main__':
    if args.train:
        run_train()
    elif args.inference:
        run_inference()
    else:
        raise NotImplementedError



