"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Utilities for tracking data and model checkpoints for different experiments.
"""
import os
import sys
from src.utils.utils import *


# --- model options --- #
def get_model_tag(args, no_subtask=False):
    if no_subtask:
        if args.model.endswith('.pt'):
            return args.model[:-3]
    return args.model


def get_schema_feature_tag(args):
    feat_tag = 'feat.'
    return feat_tag


def get_sample_gt_tag(args):
    sample_gt_tag = '-sgt' if args.enumerate_ground_truth else ''
    return sample_gt_tag


# --- data options --- #
def get_picklist_tag(args):
    pl_tag = ''
    if args.use_picklist:
        pl_tag = 'ppl.'
        if args.read_picklist:
            pl_tag = 'r' + pl_tag
    if pl_tag:
        if args.no_anchor_text:
            pl_tag = pl_tag[:-1] + '-nat.'
        else:
            pl_tag = pl_tag[:-1] + '-{}.'.format(args.anchor_text_match_threshold)
        if args.top_k_picklist_matches > 1:
            pl_tag += '{}.'.format(args.top_k_picklist_matches)
    return pl_tag


def get_lstm_encoding_tag(args):
    if args.use_lstm_encoder:
        return 'lstm.'
    else:
        return ''


def get_meta_encoding_tag(args):
    if args.use_meta_data_encoding:
        return 'meta.'
    else:
        return ''


def get_graph_encoding_tag(args):
    if args.use_graph_encoding:
        return 'ge.'
    else:
        return ''


def get_value_tag(args):
    if args.num_values_per_field > 0:
        return 'v{}.'.format(args.num_values_per_field)
    else:
        return ''


def get_table_shuffle_tag(args):
    if args.table_shuffling:
        return 'ts.'
    else:
        return ''


def get_random_table_tag(args):
    if args.num_random_tables_added > 0:
        return 'rt{}.'.format(args.num_random_tables_added)
    else:
        return ''


def get_random_field_order_tag(args):
    if args.random_field_order:
        return 'rfo.'
    else:
        return ''


def get_norm_tag(args):
    norm_tag = 'norm.' if args.normalize_variables else ''
    return norm_tag


def get_denorm_tag(args):
    denorm_tag = 'dn.' if args.denormalize_sql else ''
    return denorm_tag


def get_from_clause_tag(args):
    from_clause_tag = 'no_from.' if args.omit_from_clause else ''
    return from_clause_tag


def get_typed_token_tag(args):
    typed_token_tag = 'ts.' if args.use_typed_field_markers else ''
    return typed_token_tag


def get_execution_order_tag(args):
    eo_tag = 'eo.' if args.process_sql_in_execution_order else ''
    return eo_tag


def get_data_augmentation_tag(args):
    aug_tag = 'aug-{}.'.format(args.data_augmentation_factor) if args.data_augmentation_factor > 1 else ''
    return aug_tag


def get_data_augmentation_with_wikisql_tag(args):
    aug_wikisql_tag = 'wikisql.' if args.augment_with_wikisql else ''
    return aug_wikisql_tag


def get_oracle_table_tag(args):
    ot_tag = 'ot.' if args.use_oracle_tables else ''
    return ot_tag


def get_no_join_tag(args, separator_in_front=False):
    if args.no_join_condition:
        nj_tag = '.nj' if separator_in_front else 'nj.'
    else:
        nj_tag =  ''
    return nj_tag


def get_atomic_value_tag(args):
    avc_tag = 'avc.' if args.atomic_value else ''
    return avc_tag


def get_tokenizer_tag(args):
    if args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased'):
        return 'bert.'
    elif args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-cased'):
        return 'bert.cased.'
    elif args.pretrained_transformer.startswith('roberta'):
        return 'roberta.'
    elif args.pretrained_transformer == 'table-bert':
        return 'table-bert.'
    elif args.pretrained_transformer == 'null':
        return 'revtok.'
    else:
        raise NotImplementedError


def get_wandb_group(args):
    pl_tag = get_picklist_tag(args)
    if args.read_picklist and args.num_const_attn_layers > 0:
        pl_tag += '{}.'.format(args.num_const_attn_layers)
    le_tag = get_lstm_encoding_tag(args)
    me_tag = get_meta_encoding_tag(args)
    ge_tag = get_graph_encoding_tag(args)
    ts_tag = get_table_shuffle_tag(args)
    rfo_tag = get_random_field_order_tag(args)
    return '{}{}{}{}{}{}{}-norm-digit-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        pl_tag, le_tag, me_tag, ge_tag, ts_tag, rfo_tag, args.pretrained_transformer, args.encoder_hidden_dim,
        args.curriculum_interval, args.pretrained_lm_dropout_rate, args.learning_rate, args.learning_rate_scheduler,
        args.trans_learning_rate_scheduler, args.num_steps, args.num_warmup_steps)


def get_wandb_tag(args):
    return get_model_subdir(args)


def get_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path
    # checkpoint_path = os.path.join(args.model_dir, 'model-best.{}.tar'.format(args.beam_size))
    checkpoint_path = os.path.join(args.model_dir, 'model-best.tar')
    try:
        assert(os.path.exists(checkpoint_path))
    except AssertionError:
        print('Checkpoint not found: {}'.format(checkpoint_path))
        sys.exit(0)
    return checkpoint_path


def get_model_subdir(args, with_time_stamp=True):
    dataset = os.path.basename(args.dataset_name)

    initialization_tag = '{}.'.format(args.pretrained_transformer)

    if args.xavier_initialization:
        initialization_tag += 'xavier'
    else:
        initialization_tag = ''

    if args.num_accumulation_steps > 1:
        hyperparameter_sig = '-'.join([str(x) for x in [
            args.encoder_input_dim,
            args.encoder_hidden_dim,
            args.decoder_input_dim,
            args.train_batch_size,
            args.num_accumulation_steps,
            args.learning_rate
        ]])
    else:
        hyperparameter_sig = '-'.join([str(x) for x in [
            args.encoder_input_dim,
            args.encoder_hidden_dim,
            args.decoder_input_dim,
            args.train_batch_size,
            args.learning_rate
        ]])
    if args.curriculum_interval > 0:
        hyperparameter_sig += '-curr-{}'.format(args.curriculum_interval)
    if args.learning_rate_scheduler == 'inverse-square':
        hyperparameter_sig += '-inv-sqr-{}'.format(args.warmup_init_lr)
        hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
    elif args.learning_rate_scheduler == 'inverse-power':
        hyperparameter_sig += '-inv-pow-{}'.format(args.warmup_init_lr)
        hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
    elif args.learning_rate_scheduler == 'linear':
        hyperparameter_sig += '-linear-{}'.format(args.warmup_init_lr)
        hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
    # elif args.learning_rate_scheduler == 'step':
    #     hyperparameter_sig += '-step-{}-{}'.format(args.step_size, args.gamma)
    if args.pretrained_transformer and not args.fix_pretrained_transformer_parameters:
        hyperparameter_sig += '-{}'.format(args.bert_finetune_rate)
        if args.trans_learning_rate_scheduler == 'inverse-square':
            hyperparameter_sig += '-inv-sqr-{}'.format(args.warmup_init_ft_lr)
            hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
        elif args.trans_learning_rate_scheduler == 'inverse-power':
            hyperparameter_sig += '-inv-pow-{}'.format(args.warmup_init_ft_lr)
            hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
        elif args.trans_learning_rate_scheduler == 'linear':
            hyperparameter_sig += '-linear-{}'.format(args.warmup_init_ft_lr)
            hyperparameter_sig += '-{}'.format(args.num_warmup_steps)
    hyperparameter_sig += ('-' + '-'.join([str(x) for x in [
        args.grad_norm,
        args.emb_dropout_rate,
        args.pretrained_lm_dropout_rate,
        args.cross_attn_dropout_rate
    ]]))

    res_sig = 'res-{}-{}'.format(args.res_input_dropout_rate, args.res_layer_dropout_rate)
    ff_sig = 'ff-{}-{}'.format(args.ff_input_dropout_rate, args.ff_hidden_dropout_rate,)
    if args.model_id in [BRIDGE, SEQ2SEQ, SEQ2SEQ_PG]:
        hyperparameter_sig += ('-' + '-'.join([str(x) for x in [
            args.num_rnn_layers,
            args.cross_attn_num_heads,
            args.rnn_layer_dropout_rate,
            args.rnn_weight_dropout_rate,
            res_sig,
            ff_sig
        ]]))
    else:
        raise NotImplementedError

    pl_tag = get_picklist_tag(args)
    if args.read_picklist and args.num_const_attn_layers > 0:
        pl_tag += '{}.'.format(args.num_const_attn_layers)
    model_sub_dir = '{}.{}.{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}-{}'.format(
        dataset,
        get_model_tag(args),
        get_lstm_encoding_tag(args),
        get_meta_encoding_tag(args),
        get_graph_encoding_tag(args),
        get_table_shuffle_tag(args),
        pl_tag,
        get_value_tag(args),
        get_norm_tag(args),
        get_denorm_tag(args),
        get_no_join_tag(args),
        get_typed_token_tag(args),
        get_from_clause_tag(args),
        get_execution_order_tag(args),
        get_oracle_table_tag(args),
        get_random_table_tag(args),
        get_atomic_value_tag(args),
        get_random_field_order_tag(args),
        get_data_augmentation_tag(args),
        get_data_augmentation_with_wikisql_tag(args),
        get_schema_feature_tag(args),
        get_sample_gt_tag(args),
        initialization_tag,
        hyperparameter_sig
    )

    if args.test:
        model_sub_dir += '.test'

    if args.train and with_time_stamp:
        model_sub_dir += '.{}.{}'.format(get_time_tag(), get_random_tag(4))

    return model_sub_dir


def get_model_dir(args):
    # add model parameter info to model directory names
    model_subdir = get_model_subdir(args)
    model_dir = os.path.join(args.model_root_dir, model_subdir)
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    viz_dir = os.path.join(args.viz_root_dir, model_subdir)
    args.viz_dir = viz_dir
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print('Visualization directory created: {}'.format(viz_dir))
    else:
        print('Visualization directory exists: {}'.format(viz_dir))


def get_data_signature(args):
    data_split = 'question' if args.question_split else "query"
    model_tag = get_model_tag(args, no_subtask=True)
    ge_tag = get_graph_encoding_tag(args)
    pl_tag = get_picklist_tag(args)
    norm_tag = get_norm_tag(args)
    denorm_tag = get_denorm_tag(args)
    nj_tag = get_no_join_tag(args)
    ts_tag = get_typed_token_tag(args)
    from_tag = get_from_clause_tag(args)
    eo_tag = get_execution_order_tag(args)
    avc_tag = get_atomic_value_tag(args)
    aug_tag = get_data_augmentation_tag(args)
    aug_wikisql_tag = get_data_augmentation_with_wikisql_tag(args)
    ot_tag = get_oracle_table_tag(args)
    tokenizer_tag = get_tokenizer_tag(args)

    return '{}.{}.{}-split.{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(
        args.dataset_name,
        model_tag,
        data_split,
        ge_tag,
        pl_tag,
        norm_tag,
        denorm_tag,
        nj_tag,
        ts_tag,
        from_tag,
        eo_tag,
        avc_tag,
        aug_tag,
        aug_wikisql_tag,
        ot_tag,
        tokenizer_tag)


def get_processed_data_path(args):
    data_sig = get_data_signature(args)
    return os.path.join(args.data_dir, '{}pkl'.format(data_sig))


def get_vocab_path(args, vocab_tag):
    data_split = 'question' if args.question_split else "query"
    model_tag = get_model_tag(args, no_subtask=True)
    norm_tag = get_norm_tag(args)
    denorm_tag = get_denorm_tag(args)
    from_tag = get_from_clause_tag(args)
    avc_tag = get_atomic_value_tag(args)
    aug_tag = get_data_augmentation_tag(args)
    aug_wikisql_tag = get_data_augmentation_with_wikisql_tag(args)
    tokenizer_tag = get_tokenizer_tag(args)

    return os.path.join(args.data_dir, '{}.{}.{}-split.{}{}{}{}{}{}{}{}.vocab'.format(
        args.dataset_name,
        model_tag,
        data_split,
        norm_tag,
        denorm_tag,
        from_tag,
        avc_tag,
        aug_tag,
        aug_wikisql_tag,
        tokenizer_tag,
        vocab_tag.lower()))


# --- file system operations --- #
def safe_mkdir(path):
    try:
        os.mkdir(path)
        print('{} created'.format(path))
    except FileExistsError as e:
        pass


def safe_mkdir_hier(base_dir, nested_dir):
    dir_ = base_dir
    for subdir in nested_dir.split('/'):
        safe_mkdir(os.path.join(dir_, subdir))
        dir_ = os.path.join(dir_, subdir)
    return dir_
