"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Base learning framework.
"""

import copy
import functools
import numpy as np
import os
import pickle
import random
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import src.common.lr_scheduler as lrs
from src.common.nn_visualizer import LayerVisualizationDataWriter
from src.data_processor.processor_utils import WIKISQL, SPIDER
from src.data_processor.path_utils import get_wandb_group, get_wandb_tag, get_no_join_tag
import src.data_processor.tokenizers as tok
import src.eval.eval_tools as eval_tools
import src.eval.eval_utils as eval_utils
from src.eval.wikisql.lib.dbengine import DBEngine
import src.utils.utils as utils


LR_STEP = 0
LR_LINEAR = 1
LR_INVERSE_SQUARE = 2
LR_INVERSE_POWER = 3
LR_PLATEAU = 4


learning_rate_scheduler_sigs = {
    'step': LR_STEP,
    'linear': LR_LINEAR,
    'inverse-square': LR_INVERSE_SQUARE,
    'inverse-power': LR_INVERSE_POWER,
    'plateau': LR_PLATEAU
}


class LFramework(nn.Module):
    """
    Learning framework interface.
    """
    def __init__(self, args):
        super(LFramework, self).__init__()
        self.model = args.model
        self.model_id = args.model_id

        self.tu = utils.get_trans_utils(args)
        self.schema_graphs = None

        # Training hyperparameters
        self.args = args
        _, _, _, self.tu = tok.get_tokenizers(args)
        self.dataset = args.dataset_name
        self.model_dir = args.model_dir
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size

        self.start_step = args.start_step
        self.num_steps = args.num_steps
        self.num_peek_steps = args.num_peek_steps
        self.num_log_steps = args.num_log_steps
        self.num_accumulation_steps = args.num_accumulation_steps
        self.save_best_model_only = args.save_best_model_only

        self.optimizer = args.optimizer
        self.bert_finetune_rate = args.bert_finetune_rate
        self.learning_rate = args.learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler_sigs[args.learning_rate_scheduler]
        self.ft_learning_rate_scheduler = learning_rate_scheduler_sigs[args.trans_learning_rate_scheduler]
        self.warmup_init_lr = args.warmup_init_lr
        self.warmup_init_ft_lr = args.warmup_init_ft_lr
        self.num_warmup_steps = args.num_warmup_steps
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None
        self.lr_scheduler = None

        self.decoding_algorithm = args.decoding_algorithm
        self.beam_size = args.beam_size

        self.save_all_checkpoints = args.save_all_checkpoints

        # Visualization saver
        self.vis_writer = LayerVisualizationDataWriter(log_dir=args.viz_dir)

    def loss(self, formatted_batch):
        """
        Interface.
        """
        return 0

    def run_train(self, train_data, dev_data):
        self.print_model_parameters()

        import wandb
        wandb.init(project='smore-{}-group-{}-final'.format(self.args.dataset_name,
                                                     get_no_join_tag(self.args, separator_in_front=True)),
                                                     group=get_wandb_group(self.args),
                                                     name=get_wandb_tag(self.args))
        os.environ["WANDB_RUN_GROUP"] = get_wandb_group(self.args)
        wandb.watch(self)

        if self.args.augment_with_wikisql:
            train_data_, train_data_augment = [], []
            for example in train_data:
                if example.dataset_id == WIKISQL:
                    train_data_augment.append(example)
                else:
                    train_data_.append(example)
            train_data = train_data_
            train_batch_size = round(self.train_batch_size * 0.7)
            train_augment_batch_size = self.train_batch_size - train_batch_size

            dev_data_, dev_data_augment = [], []
            for example in dev_data:
                if example.dataset_id == WIKISQL:
                    dev_data_augment.append(example)
                else:
                    dev_data_.append(example)
                dev_data = dev_data_
            print('**************************')
            print('{} training examples'.format(len(train_data)))
            print('{} augmented training examples'.format(len(train_data_augment)))
            print('train batch size = {}'.format(train_batch_size))
            print('train augment batch size = {}'.format(train_augment_batch_size))
            print('{} dev examples'.format(len(dev_data)))
            print('{} augmented dev examples'.format(len(dev_data_augment)))
            print('**************************')
        else:
            train_batch_size = self.train_batch_size
            train_augment_batch_size = 0

        # Track training losses dev metrics changes
        ############################
        epoch_losses = []
        best_dev_metrics = 0
        dev_metrics_history = []
        ############################

        all_train_data = copy.deepcopy(train_data)
        # Curriculum learning (start from easy category)
        if self.args.curriculum_interval > 0:
            # assert(self.args.curriculum_interval % self.args.num_peek_steps == 0)
            train_data = [exp for exp in all_train_data if exp.hardness in ['easy', 'medium']]
            print('Curriculumn: [easy, medium] ({}) ------'.format(len(train_data)))

        num_steps = self.num_steps * self.num_accumulation_steps
        num_peek_steps = self.num_peek_steps * self.num_accumulation_steps
        curriculum_interval = self.args.curriculum_interval * self.num_accumulation_steps

        random.shuffle(train_data)
        if self.args.augment_with_wikisql:
            random.shuffle(train_data_augment)
            augment_example_id = 0
        step_id, example_id = 0, 0

        self.optim.zero_grad()
        self.train()

        for interval_step_id in range(self.start_step, num_steps, num_peek_steps):
            # Update model parameters
            self.train()

            for s_id in tqdm(range(num_peek_steps)):
                step_id = interval_step_id + s_id
                if self.log_in_wandb(step_id / self.num_accumulation_steps):
                    wandb.log({'learning_rate/{}'.format(self.dataset): self.optim.param_groups[0]['lr']})
                    wandb.log({'fine_tuning_rate/{}'.format(self.dataset): self.optim.param_groups[1]['lr']})

                batch_end = example_id + train_batch_size
                if curriculum_interval > 0 and step_id % curriculum_interval == 0 and \
                        0 < step_id / curriculum_interval <= 2:
                    if float(step_id) / curriculum_interval == 1:
                        train_data = [exp for exp in all_train_data if exp.hardness in ['easy', 'medium', 'hard']]
                        print('Curriculumn: [easy, medium, hard] ({}) ------'.format(len(train_data)))
                    elif float(step_id) / curriculum_interval == 2:
                        train_data = all_train_data
                        print('Curriculumn: [easy, medium, hard, extra] ({}) ------'.format(len(train_data)))
                    random.shuffle(train_data)
                    example_id, batch_end = 0, train_batch_size
                if batch_end > len(train_data):
                    random.shuffle(train_data)
                    example_id, batch_end = 0, train_batch_size
                mini_batch = train_data[example_id:batch_end]
                example_id = batch_end
                if self.args.augment_with_wikisql:
                    augment_batch_end = augment_example_id + train_augment_batch_size
                    if augment_batch_end > len(train_data_augment):
                        random.shuffle(train_data_augment)
                        augment_example_id, augment_batch_end = 0, train_augment_batch_size
                    mini_batch += train_data_augment[augment_example_id:augment_batch_end]
                    augment_example_id = augment_batch_end

                formatted_batch = self.format_batch(mini_batch)
                loss = self.loss(formatted_batch)
                loss.backward()
                epoch_losses.append(float(loss) * self.num_accumulation_steps)

                if (step_id + 1) % self.num_accumulation_steps == 0:
                    # Gradient clipping
                    if self.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                    # Update learning rate scheduler
                    self.lr_scheduler.step()
                    # Update parameters
                    self.optim.step()
                    self.optim.zero_grad()

            # Check training statistics
            if step_id > 0 and (step_id + 1) % num_peek_steps == 0:
                stdout_msg = 'Step {}: average training loss = {}'.format(
                    step_id / self.num_accumulation_steps, np.mean(epoch_losses))
                print(stdout_msg)
                wandb.log({'cross_entropy_loss/{}'.format(self.dataset): np.mean(epoch_losses)})
                epoch_losses = []

            # Check model performance
            if step_id > 0 and (step_id + 1) % num_peek_steps == 0:
                self.eval()
                if self.args.process_sql_in_execution_order:
                    pred_restored_cache = self.load_pred_restored_cache()
                    pred_restored_cache_size = sum(len(v) for v in pred_restored_cache.values())
                else:
                    pred_restored_cache = None
                engine_path = os.path.join(self.args.data_dir, 'dev.db') if self.args.dataset_name == 'wikisql' else None
                engine = DBEngine(engine_path) if engine_path else None

                output_dict = self.inference(dev_data, restore_clause_order=self.args.process_sql_in_execution_order,
                                             pred_restored_cache=pred_restored_cache,
                                             check_schema_consistency_=self.args.sql_consistency_check,
                                             engine=engine, inline_eval=True, verbose=False)
                metrics = eval_tools.get_exact_match_metrics(dev_data, output_dict['pred_decoded'], engine=engine)
                dev_metrics_history.append(metrics)

                eval_metrics = metrics['top_1_ex'] if self.args.dataset_name == 'wikisql' else metrics['top_1_em']
                wandb.log({'dev_exact_match/{}'.format(self.dataset): eval_metrics})

                print('Dev set performance:')
                print('Top-1 exact match: {}'.format(metrics['top_1_em']))
                print('Top-3 exact match: {}'.format(metrics['top_3_em']))
                if self.args.dataset_name == 'wikisql':
                    print('Top-1 exe acc: {}'.format(metrics['top_1_ex']))
                    print('Top-3 exe acc: {}'.format(metrics['top_3_ex']))

                if eval_metrics >= best_dev_metrics:
                    best_dev_metrics = eval_metrics
                    self.save_checkpoint(step_id, step_id / num_peek_steps, output_dict['pred_decoded'], is_best=True)
                if self.args.augment_with_wikisql and (step_id + 1) % (num_peek_steps * 3) == 0:
                    wikisql_output_dict = self.inference(dev_data_augment, inline_eval=True, verbose=False)
                    wikisql_metrics = eval_tools.get_exact_match_metrics(dev_data_augment, wikisql_output_dict['pred_decoded'])
                    wandb.log({'wikisql_dev_exact_match/{}'.format(self.dataset): wikisql_metrics['top_1_em']})
                    print('WikiSQL dev set performance:')
                    print('Top-1 exact match: {}'.format(wikisql_metrics['top_1_em']))
                    print('Top-3 exact match: {}'.format(wikisql_metrics['top_3_em']))
                if self.args.process_sql_in_execution_order:
                    new_pred_restored_cache_size = sum(len(v) for v in output_dict['pred_restored_cache'].values())
                    newly_cached_size = new_pred_restored_cache_size - pred_restored_cache_size
                    if newly_cached_size > 0:
                        self.save_pred_restored_cache(output_dict['pred_restored_cache'], newly_cached_size)

    def forward(self, *args, **kwargs):
        """
        Interface.
        """
        return

    def inference(self, *args, **kwargs):
        """
        Interface.
        """
        return

    def print_prediction(self, *args, **kwargs):
        """
        Interface.
        """
        return

    def get_readable_data(self, *args, **kwargs):
        """
        Interface.
        """
        return

    def de_vectorize(self, *args, **kwargs):
        """
        Interface.
        """
        return

    def format_batch(self, mini_batch):
        if self.training and self.args.enumerate_ground_truth:
            for example in mini_batch:
                example.set_p_idx()

    def define_lr_scheduler(self):
        if self.learning_rate_scheduler == self.ft_learning_rate_scheduler:
            if self.learning_rate_scheduler == LR_LINEAR:
                self.lr_scheduler = lrs.LinearScheduler(
                    self.optim, [self.warmup_init_lr, self.warmup_init_ft_lr], [self.num_warmup_steps, self.num_warmup_steps],
                    self.num_steps)
            elif self.learning_rate_scheduler == LR_INVERSE_SQUARE:
                self.lr_scheduler = lrs.InverseSquareRootScheduler(
                    self.optim, [self.warmup_init_lr, self.warmup_init_ft_lr], [self.num_warmup_steps, self.num_warmup_steps],
                    self.num_steps)
            elif self.learning_rate_scheduler == LR_INVERSE_POWER:
                self.lr_scheduler = lrs.InversePowerScheduler(
                    self.optim, 1.0, [self.warmup_init_lr, self.warmup_init_ft_lr], [self.num_warmup_steps, self.num_warmup_steps])
            elif self.learning_rate_scheduler == LR_PLATEAU:
                self.lr_scheduler = lrs.ReduceLROnPlateau(
                    self.optim, factor=0.5, patience=5, min_lr=1e-5, verbose=True)
            else:
                raise NotImplementedError
        else:
            assert(self.learning_rate_scheduler == LR_LINEAR and self.ft_learning_rate_scheduler == LR_INVERSE_SQUARE)
            self.lr_scheduler = lrs.HybridScheduler(self.optim,
                                                    [self.learning_rate_scheduler, self.ft_learning_rate_scheduler],
                                                    [self.warmup_init_lr, self.warmup_init_ft_lr],
                                                    [self.num_warmup_steps, self.num_warmup_steps],
                                                    self.num_steps)

    def define_optimizer(self):
        if self.optimizer == 'adam':
            self.optim = optim.Adam(
            [
                {'params': [p for n, p in self.named_parameters() if not 'trans_parameters' in n and p.requires_grad]},
                {'params': [p for n, p in self.named_parameters() if 'trans_parameters' in n and p.requires_grad],
                 'lr': self.bert_finetune_rate}
            ], lr=self.learning_rate)
        else:
            raise NotImplementedError

    # --- Data I/O --- #
    def load_pred_restored_cache(self):
        split = 'test' if self.args.test else 'dev'
        pred_restored_cache_path = os.path.join(
            self.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if os.path.exists(pred_restored_cache_path):
            with open(pred_restored_cache_path, 'rb') as f:
                pred_restored_cache = pickle.load(f)
                pred_restored_cache_size = sum([len(pred_restored_cache[k]) for k in pred_restored_cache])
                print('{} pre-computed prediction order reconstruction cached'.format(pred_restored_cache_size))
            return pred_restored_cache
        else:
            return dict()

    def save_pred_restored_cache(self, pred_restored_cache, newly_cached_size):
        split = 'test' if self.args.test else 'dev'
        pred_restored_cache_path = os.path.join(
            self.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if os.path.exists(pred_restored_cache_path):
            shutil.copyfile(pred_restored_cache_path, pred_restored_cache_path + '.copy')
        with open(pred_restored_cache_path, 'wb') as o_f:
            pickle.dump(pred_restored_cache, o_f)
            print('{} sql order restoration newly cached'.format(newly_cached_size))

    # --- Model checkpoints --- #

    def save_checkpoint(self, checkpoint_id, interval_step_id, predictions, loss=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param predictions: List of predicted strings.
        :param step_id: Training interval step id.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['model_state_dict'] = self.state_dict()
        if self.optim:
            checkpoint_dict['optimizer_state_dict'] = self.optim.state_dict()
        if self.lr_scheduler:
            checkpoint_dict['lr_scheduler_dict'] = self.lr_scheduler.state_dict()
        checkpoint_dict['interval_step_id'] = interval_step_id
        checkpoint_dict['loss'] = loss

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model-best.{}.tar'.format(self.beam_size))
            if os.path.exists(out_tar):
                shutil.copyfile(out_tar, best_path)
            else:
                torch.save(checkpoint_dict, best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            print('=> saving checkpoint to \'{}\''.format(out_tar))

        with open(os.path.join(self.model_dir, 'best_dev_iteration.{}.dat'.format(self.beam_size)), 'w') as o_f:
            o_f.write('{}'.format(checkpoint_id))
        out_txt = os.path.join(self.model_dir, 'predictions.{}.txt'.format(self.beam_size))
        with open(out_txt, 'w') as o_f:
            for pred_sql in predictions:
                o_f.write('{}\n'.format(pred_sql[0]))
            print('=> Model predictions saved to {}'.format(out_txt))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file)
            self.load_state_dict(checkpoint['model_state_dict'])
            if self.args.train:
                self.start_step = checkpoint['interval_step_id'] + 1
                if 'optimizer_state_dict' in checkpoint:
                    self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'lr_scheduler_dict' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_dict'])
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    # --- Train flow control functions --- #

    def log_in_wandb(self, step_id):
        return step_id % self.num_log_steps == 0

    @property
    def save_vis(self):
        return not self.training and self.args.save_nn_weights_for_visualizations

    # --- Printing functions --- #

    def print_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        if self.args.share_vocab:
            print('encoder and decoder share vocabulary')
        num_display = 500
        count = 0
        for name, param in self.named_parameters():
            if 'trans_parameters' in name:
                continue
            if count < num_display:
                print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
            count += 1
        if count >= num_display:
            print('...')
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def print_model_parameter_values(self):
        print('\nModel Parameter Values') 
        print('--------------------------')
        if self.args.share_vocab:
            print('encoder and decoder share vocabulary')
        num_display = 500
        count = 0
        for name, param in self.named_parameters():
            if count < num_display:
                print(name, param, 'requires_grad={}'.format(param.requires_grad))
            count += 1
        if count >= num_display:
            print('...')
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def print_model_parameter_grads(self):
        print('\nModel Parameter Gradients')
        print('--------------------------')
        if self.args.share_vocab:
            print('encoder and decoder share vocabulary')
        num_display = 500
        count = 0
        for name, param in self.named_parameters():
            if count < num_display:
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(name, param.grad, 'requires_grad={}'.format(param.requires_grad))
                    # import pdb
                    # pdb.set_trace()
            count += 1
        if count >= num_display:
            print('...')
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
