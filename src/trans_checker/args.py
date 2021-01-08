"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import argparse
import os


code_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Question Translatability Checker')

parser.add_argument('--train', action='store_true',
                    help='run model training (default: False)')
parser.add_argument('--inference', action='store_true',
                    help='run inference (default: False)')

parser.add_argument('--dataset_name', type=str, default='spider_ut',
                    help='name of dataset (default: spider_ut)')
parser.add_argument('--data_dir', type=str, default=os.path.join(code_base_dir, 'data/spider_ut'),
                    help='directory where the data is stored (default: None)')
parser.add_argument('--db_dir', type=str, default=os.path.join(code_base_dir, 'data/spider_ut/database'),
                    help='directory where the database files are stored (default: None)')
parser.add_argument('--checkpoint_path', type=str, default=os.path.join(code_base_dir, 'model/trans_check_0.0001_3e-05_800/model-best.tar'),
                    help='Path to the translatability checker model checkpoint (default: None)')
parser.add_argument('--random_field_order', action='store_true',
                    help='If set, use random field order in augmented table schema (default: False).')
parser.add_argument('--augment_with_wikisql', action='store_true',
                    help='If set, augment training data with WikiSQL (default: False)')
parser.add_argument('--model_dir', type=str, default=os.path.join(code_base_dir, 'model'),
                    help='directory where the model parameters are stored (default: None)')
parser.add_argument('--pretrained_transformer', type=str, default='bert-large-uncased',
                    help='Specify pretrained transformer model to use.')
parser.add_argument('--pretrained_lm_dropout_rate', type=float, default=0.3,
                    help='Pretrained LM features dropout rate (default: 0.0)')
parser.add_argument('--encoder_input_dim', type=int, default=1024,
                    help='Encoder input dimension (default: 200)')
parser.add_argument('--encoder_hidden_dim', type=int, default=-1,
                    help='Encoder hidden dimension (default to encoder input dimension)')

parser.add_argument('--top_k_picklist_matches', type=int, default=1,
                    help='Maximum number of values that matches the input to select from a picklist (default: 1)')

parser.add_argument('--bert_finetune_rate', type=float, default=0.00003,
                    help='BERT finetune rate (default: 0.00003)')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--learning_rate_scheduler', type=str, default='linear',
                    help='learning rate scheduler (default: linear)')
parser.add_argument('--warmup_init_lr', type=float, default=0.0001,
                    help='learning rate at the beginning of the warmup procedure (default: 1e-4)')
parser.add_argument('--warmup_init_ft_lr', type=float, default=0.00003,
                    help='fine tuning rate at the beginning of the warmup procedure (default: 3e-5)')
parser.add_argument('--num_warmup_steps', type=int, default=800,
                    help='# warmup steps to do in the warmup procedure (default: 800)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='maximum number of training epochs (default: 100)')
parser.add_argument('--num_steps', type=int, default=84000,
                    help='maximum number of training steps (default: 84000)')
parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
parser.add_argument('--adam_eps', type=float, default=1e-8,
                    help='Adam: denominator bias to improve numerical stability (default: 1e-8)')
parser.add_argument('--grad_norm', type=float, default=0.3,
                    help='norm threshold for gradient clipping (default: 0.3)')
parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')

parser.add_argument('--gpu', type=int, default=0, help='gpu device (default: 0)')
parser.add_argument('--seed', type=int, default=543, metavar='S', help='random seed (default: 543)')

args = ['--gpu', '0']
args = parser.parse_args(args)