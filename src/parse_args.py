"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Experiment Hyperparameters.
"""

import argparse
import os


parser = argparse.ArgumentParser(description='Neural Semantic Parsing with Transformer-Pointer Network')

# Experiment control
parser.add_argument('--process_data', action='store_true',
                    help='data preprocessing and numericalization (default: False)')
parser.add_argument('--process_new_data_split', action='store_true',
                    help='preprocess a specified data split and add it to existing processed data (default: False)')
parser.add_argument('--train', action='store_true',
                    help='run model training (default: False)')
parser.add_argument('--inference', action='store_true',
                    help='run inference (default: False)')
parser.add_argument('--ensemble_inference', action='store_true',
                    help='run inference with the ensemble of multiple models specified ')
parser.add_argument('--predict_tables', action='store_true',
                    help='run table prediction experiments (default: False)')
parser.add_argument('--no_join_condition', action='store_true',
                    help='do not predict join condition (default: False)')
parser.add_argument('--test', action='store_true',
                    help='perform inference on the test set (default: False)')
parser.add_argument('--fine_tune', action='store_true',
                    help='fine tuning model on a given dataset (default: False)')
parser.add_argument('--demo', action='store_true',
                    help='run interactive commandline demo (default: False)')
parser.add_argument('--demo_db', type=str, default=None,
                    help='the database used in the interactive demo')
parser.add_argument('--data_statistics', action='store_true',
                    help='print dataset statistics (default: False)')
parser.add_argument('--search_random_seed', action='store_true',
                    help='run experiments with multiple random initializations and compute the result '
                         'statistics (default: False)')
parser.add_argument('--eval', action='store_true',
                    help='compute evaluation metrics (default: False)')
parser.add_argument('--eval_by_relation_type', action='store_true',
                    help='compute evaluation metrics for to-M and to-1 relations separately (default: False)')
parser.add_argument('--error_analysis', action='store_true',
                    help='run error analysis (default: False)')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the data is stored (default: None)')
parser.add_argument('--db_dir', type=str, default=None,
                    help='directory where the database files are stored (default: None)')
parser.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='root directory where the model parameters are stored (default: None)')
parser.add_argument('--viz_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viz'),
                    help='root directory where the network visualizations are stored (default: None)')
parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the model parameters are stored (default: None)')
parser.add_argument('--viz_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viz'),
                    help='directory where the network visualizations are stored (default: None)')
parser.add_argument('--save_all_checkpoints', action='store_true',
                    help='If set, save all checkpoints during training; otherwise, save the checkpoints w/ the best '
                         'dev performance only. (default: False)')
parser.add_argument('--gpu', type=int, default=0, help='gpu device (default: 0)')

# Leaderboard submission
parser.add_argument('--leaderboard_submission', action='store_true',
                    help='If set, switch to leaderboard submission mode.')
parser.add_argument('--codalab_data_dir', type=str, default=None,
                    help='Data directory on Codalab.')
parser.add_argument('--codalab_db_dir', type=str, default=None,
                    help='Database directory on Codalab.')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to a pretrained checkpoint (default: None)')
parser.add_argument('--prediction_path', type=str, default=None,
                    help='path to which the model prediction is saved (default: None)')

# Data
parser.add_argument('--use_pred_tables', action='store_true',
                    help='If set, use automatically predicted tables.')
parser.add_argument('--use_graph_encoding', action='store_true',
                    help='If set, use graph encoding as input.')
parser.add_argument('--use_picklist', action='store_true',
                    help='If set, use field value pick list (default: False)')
parser.add_argument('--read_picklist', action='store_true',
                    help='If set, read field values from pick list (default: False)')
parser.add_argument('--top_k_picklist_matches', type=int, default=1,
                    help='Maximum number of values that matches the input to select from a picklist (default: 1)')
parser.add_argument('--num_values_per_field', type=int, default=0,
                    help='Number of sample values to include in a field representation')
parser.add_argument('--num_random_tables_added', type=int, default=0,
                    help='Number of random tables added in addition to groundtruth during stage-2 training')
parser.add_argument('--table_shuffling', action='store_true',
                    help='If set, shuffle table order (default: False)')
parser.add_argument('--anchor_text_match_threshold', type=float, default=0.85,
                    help='Score threshold above which an anchor text match is considered (default: 0.85)')
parser.add_argument('--no_anchor_text', action='store_true',
                    help='If add, add only the special value token but not the value text to the schema representation')
parser.add_argument('--question_split', action='store_true',
                    help='split dataset based on questions (default: False)')
parser.add_argument('--query_split', action='store_true',
                    help='split dataset based on queries (default: False)')
parser.add_argument('--question_only', action='store_true',
                    help='Take only the natural language questions as input (default: False)')
parser.add_argument('--pretrained_transformer', type=str, default='',
                    help='Specify pretrained transformer model to use.')
parser.add_argument('--fix_pretrained_transformer_parameters', action='store_true',
                    help='If set, no finetuning is performed on the pretrained BERT embeddings (default: False).')
parser.add_argument('--use_typed_field_markers', action='store_true',
                    help='If set, use typed column special tokens to feed into the BERT layer (default: False).')
parser.add_argument('--vocab_min_freq', type=int, default=1,
                    help='Minimum token frequency in shared vocabulary (default: 1)')
parser.add_argument('--share_vocab', action='store_true',
                    help='Share input and output vocabulary (default: False)')
parser.add_argument('--text_vocab_min_freq', type=int, default=1,
                    help='Minimum word frequency in natural language vocabulary (default: 1)')
parser.add_argument('--program_vocab_min_freq', type=int, default=1,
                    help='Minimum token frequency in program vocabulary (default: 1)')
parser.add_argument('--enumerate_ground_truth', action='store_true',
                    help='Sample ground truth queries during training when there are multiple correct ones '
                         '(default: False)')
parser.add_argument('--save_nn_weights_for_visualizations', action='store_true',
                    help='Save visualizations of neural network layers. (default: False)')
parser.add_argument('--dataset_name', type=str, default='wikisql',
                    help='name of dataset (default: wikisql)')
parser.add_argument('--normalize_variables', action='store_true',
                    help='replace constant values in text and programs with place holders (default: False)')
parser.add_argument('--denormalize_sql', action='store_true',
                    help='Denormalize SQL queries (default: False)')
parser.add_argument('--omit_from_clause', action='store_true',
                    help='Remove FROM clause from SQL query in data preprocessing (default: False)')
parser.add_argument('--use_oracle_tables', action='store_true',
                    help='If set, use only ground truth tables in the database schema representation (default: False)')
parser.add_argument('--atomic_value', action='store_true',
                    help='If set, make value copying an atomic action (default: False).')
parser.add_argument('--data_augmentation_factor', type=int, default=1,
                    help='Data augmentation factor (default: 1)')
parser.add_argument('--schema_augmentation_factor', type=int, default=1,
                    help='Schema augmentation factor (default: 1)')
parser.add_argument('--random_field_order', action='store_true',
                    help='If set, use random field order in augmented table schema (default: False).')
parser.add_argument('--augment_with_wikisql', action='store_true',
                    help='If set, augment training data with WikiSQL (default: False)')
parser.add_argument('--process_sql_in_execution_order', action='store_true',
                    help='Generate and process SQL clauses in execution order (default: False)')
parser.add_argument('--sql_consistency_check', action='store_true',
                    help='Check consistency of a genereated SQL query (default: False)')

parser.add_argument('--data_parallel', action='store_true',
                    help='If set, use data parallelization. (default: False)')

# Encoder-decoder model
parser.add_argument('--model', type=str, default='bridge',
                    help='semantic parsing model (default: bridge)')
parser.add_argument('--model_id', type=int, default=None,
                    help='semantic parsing model ID (default: None)')
parser.add_argument('--loss', type=str, default='cross_entropy',
                    help='sequence training loss (default: cross_entropy)')
parser.add_argument('--encoder_input_dim', type=int, default=200,
                    help='Encoder input dimension (default: 200)')
parser.add_argument('--decoder_input_dim', type=int, default=200,
                    help='Decoder input dimension (default: 200)')
parser.add_argument('--emb_dropout_rate', type=float, default=0.0,
                    help='Embedding dropout rate (default: 0.0)')
parser.add_argument('--pretrained_lm_dropout_rate', type=float, default=0.0,
                    help='Pretrained LM features dropout rate (default: 0.0)')
parser.add_argument('--res_input_dropout_rate', type=float, default=0.0,
                    help='Residual input dropout rate (default: 0.0)')
parser.add_argument('--res_layer_dropout_rate', type=float, default=0.0,
                    help='Residual layer dropout rate (default: 0.0)')
parser.add_argument('--cross_attn_num_heads', type=int, default=1,
                    help='Encoder-decoder attention # heads (default: 1)')
parser.add_argument('--cross_attn_dropout_rate', type=float, default=0.0,
                    help='Encoder-decoder attention dropout rate (default: 0.0)')
parser.add_argument('--max_in_seq_len', type=int, default=120,
                    help='Maximum input length (default: 120)')
parser.add_argument('--max_out_seq_len', type=int, default=120,
                    help='Maximum output length (default: 120)')
parser.add_argument('--use_lstm_encoder', action='store_true',
                    help='If set, use LSTM encoder on top of pretrained transformer encoders (default: False)')
parser.add_argument('--use_meta_data_encoding', action='store_true',
                    help='If set, encode meta data features in the DB schema (default: False)')

# RNNs
parser.add_argument('--encoder_hidden_dim', type=int, default=-1,
                    help='Encoder hidden dimension (default to encoder input dimension)')
parser.add_argument('--decoder_hidden_dim', type=int, default=-1,
                    help='Decoder hidden dimension (default to decoder input dimension)')
parser.add_argument('--rnn_input_dropout_rate', type=float, default=0.0,
                    help='RNN input layer dropout rate (default: 0.0)')
parser.add_argument('--rnn_layer_dropout_rate', type=float, default=0.0,
                    help='Dropout rate between RNN layers (default: 0.0)')
parser.add_argument('--rnn_weight_dropout_rate', type=float, default=0.0,
                    help='RNN hidden weights dropout rate (default: 0.0)')
parser.add_argument('--num_rnn_layers', type=int, default=2,
                    help='# RNN layers (default: 2)')

# Schema Encoder
parser.add_argument('--schema_hidden_dim', type=int, default=200,
                    help='Dimension of the schema encoding')
parser.add_argument('--schema_dropout_rate', type=float, default=0.0,
                    help='Dropout rate of the schema encoder')
parser.add_argument('--schema_rnn_num_layers', type=int, default=1,
                    help='Number of layers in the rnn which encodes the lexical features of the schema')
parser.add_argument('--schema_rnn_input_dropout_rate', type=float, default=0.0)
parser.add_argument('--schema_rnn_layer_dropout_rate', type=float, default=0.0)
parser.add_argument('--schema_rnn_weight_dropout_rate', type=float, default=0.0)
parser.add_argument('--use_additive_features', action='store_true',
                    help='Add database key and field type features to column encodings (default: False)')
parser.add_argument('--num_const_attn_layers', type=int, default=0,
                    help='Number of self attention layers to use to encode the constants (default: 0)')

# Transformer
parser.add_argument('--encoder_tf_hidden_dim', type=int, default=-1,
                    help='Transformer encoder hidden dimension (default to encoder input dimension)')
parser.add_argument('--decoder_tf_hidden_dim', type=int, default=-1,
                    help='Transformer decoder hidden dimension (default to encoder input dimension)')
parser.add_argument('--sa_num_layers', type=int, default=2,
                    help='# self-attention layers (default: 2)')
parser.add_argument('--sa_num_heads', type=int, default=1,
                    help='# self attention heads (default: 1)')
parser.add_argument('--sa_input_dropout_rate', type=float, default=0.0,
                    help='Self attention input dropout rate (default: 0.0)')
parser.add_argument('--sa_dropout_rate', type=float, default=0.0,
                    help='Self attention dropout rate (default: 0.0)')

# Feedforward Networks
parser.add_argument('--ff_input_dropout_rate', type=float, default=0.0,
                    help='Transformer feed-forward input dropout rate (default: 0.0)')
parser.add_argument('--ff_hidden_dropout_rate', type=float, default=0.0,
                    help='Transformer feed-forward hidden layer dropout rate (default: 0.0)')
parser.add_argument('--ff_output_dropout_rate', type=float, default=0.0,
                    help='Transformer feed-forward output layer dropout rate (default: 0.0)')

# Optimization
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
parser.add_argument('--num_wait_epochs', type=int, default=200,
                    help='number of epochs to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_epochs', type=int, default=2,
                    help='number of epochs to wait for next dev set result check (default: 2)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
parser.add_argument('--num_steps', type=int, default=20000,
                    help='maximum number of training steps (default: 20000)')
parser.add_argument('--save_best_model_only', action='store_true',
                    help='If set, only save the model that achieves best performance on the validation set. (default: False)')
parser.add_argument('--num_wait_steps', type=int, default=20000,
                    help='number of steps to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_steps', type=int, default=400,
                    help='number of steps to wait for next dev set result check (default: 400)')
parser.add_argument('--num_accumulation_steps', type=int, default=1,
                    help='number of steps to wait before running an optimizer step (default: 1)')
parser.add_argument('--num_log_steps', type=int, default=500,
                    help='number of steps to wait for next wandb log save (default: 500)')
parser.add_argument('--start_step', type=int, default=0,
                    help='step from which the training should start (default: 0)')
parser.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
parser.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
parser.add_argument('--margin', type=float, default=0,
                    help='margin used for base MAMES training (default: 0)')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--bert_finetune_rate', type=float, default=0,
                    help='BERT finetune rate (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--learning_rate_scheduler', type=str, default='step',
                    help='learning rate scheduler (default: step)')
parser.add_argument('--trans_learning_rate_scheduler', type=str, default='step',
                    help='learning rate scheduler for fine-tuning pre-trained transformers (default: step)')
parser.add_argument('--warmup_init_lr', type=float, default=0.0005,
                    help='learning rate at the beginning of the warmup procedure (default: 5e-4)')
parser.add_argument('--warmup_init_ft_lr', type=float, default=0.00005,
                    help='fine tuning rate at the beginning of the warmup procedure (default: 5e-5)')
parser.add_argument('--curriculum_interval', type=int, default=0,
                    help='# number of steps for fitting a hardness level during curriculumn learning (default: 0)')
parser.add_argument('--num_warmup_steps', type=int, default=4000,
                    help='# warmup steps to do in the warmup procedure (default: 4000)')
parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
parser.add_argument('--adam_eps', type=float, default=1e-8,
                    help='Adam: denominator bias to improve numerical stability (default: 1e-8)')
parser.add_argument('--grad_norm', type=float, default=0,
                    help='norm threshold for gradient clipping (default 0, no gradient normalization is used)')
parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')
parser.add_argument('--random_parameters', type=bool, default=False,
                    help='Inference with random parameters (default: False)')

# Search Decoding
parser.add_argument('--decoding_algorithm', type=str, default='beam-search',
                    help='decoding algorithm (default: "beam-search")')
parser.add_argument('--beam_size', type=int, default=100,
                    help='size of beam used in beam search inference (default: 100))')
parser.add_argument('--bs_alpha', type=float, default=1,
                    help='bea, search length normalization coefficient')
parser.add_argument('--execution_guided_decoding', action='store_true',
                    help='If set, use execution guided decoding to prune decoded queries.')

# Hyperparameter Search
parser.add_argument('--tune', type=str, default='',
                    help='Specify the hyperparameters to tune during the search, separated by commas (default: None)')
parser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')


args = parser.parse_args()
