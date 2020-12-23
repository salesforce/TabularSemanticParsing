#!/usr/bin/env bash

data_dir="data/wikisql1.1/"
db_dir="data/wikisql1.1/"
dataset_name="wikisql"
model="bridge"
question_split="True"
query_split="False"
question_only="True"
normalize_variables="False"
denormalize_sql="True"
omit_from_clause="True"
no_join_condition="False"
table_shuffling="True"
use_graph_encoding="False"
use_typed_field_markers="False"
use_lstm_encoder="True"
use_meta_data_encoding="True"
use_picklist="True"
no_anchor_text="False"
anchor_text_match_threshold=0.85
top_k_picklist_matches=2
atomic_value_copy="False"
process_sql_in_execution_order="False"
sql_consistency_check="False"
share_vocab="False"
sample_ground_truth="False"
save_nn_weights_for_visualizations="False"
vocab_min_freq=0
text_vocab_min_freq=0
program_vocab_min_freq=0
max_in_seq_len=512
max_out_seq_len=60

num_steps=30000
curriculum_interval=0
num_peek_steps=400
num_accumulation_steps=3
save_best_model_only="True"
train_batch_size=16
dev_batch_size=24
encoder_input_dim=1024
encoder_hidden_dim=512
decoder_input_dim=512
num_rnn_layers=1
num_const_attn_layers=0

use_oracle_tables="False"
num_random_tables_added=0

use_additive_features="False"

schema_augmentation_factor=1
random_field_order="False"
data_augmentation_factor=1
augment_with_wikisql="False"
num_values_per_field=0
pretrained_transformer="bert-large-uncased"
fix_pretrained_transformer_parameters="False"
bert_finetune_rate=0.00005
learning_rate=0.0003
learning_rate_scheduler="inverse-square"
trans_learning_rate_scheduler="inverse-square"
warmup_init_lr=0.0003
warmup_init_ft_lr=0
num_warmup_steps=3000
emb_dropout_rate=0.3
pretrained_lm_dropout_rate=0
rnn_layer_dropout_rate=0.1
rnn_weight_dropout_rate=0
cross_attn_dropout_rate=0
cross_attn_num_heads=8
res_input_dropout_rate=0.2
res_layer_dropout_rate=0
ff_input_dropout_rate=0.4
ff_hidden_dropout_rate=0.0

grad_norm=0.3
decoding_algorithm="beam-search"
beam_size=64
bs_alpha=1.0

data_parallel="False"
