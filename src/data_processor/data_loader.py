"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Load raw or processed data.
"""
import collections
import json
import os
import pickle
import shutil

from src.data_processor.processor_utils import WIKISQL, SPIDER, OTHERS
from src.data_processor.processor_utils import Text2SQLExample, AugmentedText2SQLExample
from src.data_processor.path_utils import get_norm_tag, get_data_augmentation_tag
from src.data_processor.path_utils import get_processed_data_path, get_vocab_path
from src.data_processor.schema_loader import load_schema_graphs_spider, load_schema_graphs_wikisql
from src.data_processor.sql.sql_reserved_tokens import sql_reserved_tokens, sql_reserved_tokens_revtok
from src.data_processor.vocab_utils import is_functional_token, Vocabulary, value_vocab
import src.utils.utils as utils


def load_processed_data(args):
    """
    Load preprocessed data file.
    """
    if args.process_sql_in_execution_order:
        split = 'test' if args.test else 'dev'
        pred_restored_cache_path = os.path.join(
            args.model_dir, '{}.eo.pred.restored.pkl'.format(split))
        if not os.path.exists(pred_restored_cache_path):
            cache_path = os.path.join(args.data_dir, '{}.eo.pred.restored.pkl'.format(split))
            if not os.path.exists(cache_path):
                pred_restored_cache = collections.defaultdict(dict)
                with open(cache_path, 'wb') as o_f:
                    pickle.dump(pred_restored_cache, o_f)
            shutil.copyfile(cache_path, pred_restored_cache_path)
            print('execution order restoration cache copied')
            print('source: {}'.format(cache_path))
            print('dest: {}'.format(pred_restored_cache_path))
            print()

    in_pkl = get_processed_data_path(args)
    print('loading preprocessed data: {}'.format(in_pkl))
    with open(in_pkl, 'rb') as f:
        return pickle.load(f)


def load_data_by_split(args):
    """
    Load text-to-SQL dataset released by Finegan-Dollak et. al. 2018.

    The dataset adopts two different types of splits (split by question or by query type).
    """

    def fill_in_variables(s, s_vars, variables, target):
        var_list = {}
        for v_key, v_val in s_vars.items():
            if len(v_val) == 0:
                for var in variables:
                    if var['name'] == v_key:
                        v_val = var['example']
            s = s.replace(v_key, v_val)
            var_list[v_key] = v_val
        for var in variables:
            if not var['name'] in s_vars:
                v_loc = var['location']
                if target == 'program' and (v_loc == 'sql-only' or v_loc == 'both'):
                    v_key = var['name']
                    v_val = var['example']
                    s = s.replace(v_key, v_val)
                    var_list[v_key] = v_val
        return s, var_list

    dataset = dict()
    in_json = os.path.join(args.data_dir, '{}.json'.format(args.dataset_name))
    with open(in_json) as f:
        content = json.load(f)
        for example in content:
            programs = example['sql']
            query_split = example['query-split']
            variables = example['variables']
            for sentence in example['sentences']:
                question_split = sentence['question-split']
                nl = sentence['text']
                s_variables = sentence['variables']
                exp = Text2SQLExample(OTHERS, args.dataset_name, 0)
                if not args.normalize_variables:
                    nl, var_list = fill_in_variables(nl, s_variables, variables, target='text')
                    exp.variables = var_list
                exp.text = nl
                for i, program in enumerate(programs):
                    if not args.normalize_variables:
                        program, _ = fill_in_variables(program, s_variables, variables, target='program')
                    if program.endswith(';'):
                        program = program[:-1].rstrip()
                    exp.add_program(program)
                split = question_split if args.question_split else query_split
                if not split in dataset:
                    dataset[split] = []
                dataset[split].append(exp)
    return dataset


def load_data_wikisql(args):
    """
    Load the WikiSQL dataset released by Zhong et. al. 2018, assuming that the data format has been
    changed by the script `data_processor_wikisql.py`.
    """
    in_dir = args.data_dir
    splits = ['train', 'dev', 'test']
    schema_graphs = load_schema_graphs_wikisql(in_dir, splits=splits)

    dataset = dict()
    for split in splits:
        dataset[split] = load_data_split_wikisql(in_dir, split, schema_graphs)
    dataset['schema'] = schema_graphs
    return dataset


def load_data_split_wikisql(in_dir, split, schema_graphs):
    in_jsonl = os.path.join(in_dir, '{}.jsonl'.format(split))
    data_split = []
    with open(in_jsonl) as f:
        for line in f:
            example = json.loads(line.strip())
            db_name = example['table_id']
            text = example['question']
            exp = Text2SQLExample(WIKISQL, db_name, db_id=schema_graphs.get_db_id(db_name))
            exp.text = text
            # program = example['query']
            # if program.endswith(';'):
            #     program = program[:-1].rstrip()
            program_ast = example['sql']
            exp.add_program_official('', program_ast)
            schema_graph = schema_graphs[db_name]
            gt_tables = [0]
            gt_table_names = [schema_graph.get_table(0).name]
            exp.add_gt_tables(gt_tables, gt_table_names)
            data_split.append(exp)
    return data_split


def load_data_spider(args):
    """
    Load the Spider dataset released by Yu et. al. 2018.
    """
    in_dir = args.data_dir
    dataset = dict()
    schema_graphs = load_schema_graphs_spider(in_dir, 'spider', augment_with_wikisql=args.augment_with_wikisql,
                                              db_dir=args.db_dir)
    dataset['train'] = load_data_split_spider(in_dir, 'train', schema_graphs, get_data_augmentation_tag(args),
                                              augment_with_wikisql=args.augment_with_wikisql)
    dataset['dev'] = load_data_split_spider(in_dir, 'dev', schema_graphs,
                                            augment_with_wikisql=args.augment_with_wikisql)
    dataset['schema'] = schema_graphs

    fine_tune_set = load_data_split_spider(in_dir, 'fine-tune', schema_graphs,
                                           augment_with_wikisql=args.augment_with_wikisql)
    if fine_tune_set:
        dataset['fine-tune'] = fine_tune_set
    return dataset


def load_data_split_spider(in_dir, split, schema_graphs, aug_tag='', augment_with_wikisql=False):
    if split == 'train':
        in_json = os.path.join(in_dir, '{}.{}json'.format(split, aug_tag))
    else:
        in_json = os.path.join(in_dir, '{}.json'.format(split))
    if not os.path.exists(in_json):
        print('Warning: file {} not found.'.format(in_json))
        return None
    data_split = []
    num_train_exps_by_db = collections.defaultdict(int)
    with open(in_json) as f:
        content = json.load(f)
        for example in content:
            db_name = example['db_id']
            if split == 'train':
                num_train_exps_by_db[db_name] += 1
            exp = Text2SQLExample(SPIDER, db_name, db_id=schema_graphs.get_db_id(db_name))
            text = example['question'].replace('’', '\'')
            program = example['query']
            if program.endswith(';'):
                program = program[:-1].rstrip()
            exp.text = text
            if 'question_toks' in example:
                text_tokens = example['question_toks']
                exp.text_tokens = [t.lower() for t in text_tokens]
                exp.text_ptr_values = text_tokens
            program_ast = example['sql'] if 'sql' in example else None
            program_tokens = example['query_toks'] if 'query_toks' in example else None
            if program_tokens and program_tokens[-1] == ';':
                program_tokens = program_tokens[:len(program_tokens)-1]
            exp.add_program_official(program, program_ast, program_tokens)
            if 'tables' in example:
                gt_tables = example['tables']
                gt_table_names = example['table_names']
                exp.add_gt_tables(gt_tables, gt_table_names)
            if 'hardness' in example:
                exp.hardness = example['hardness']
            data_split.append(exp)

    print('{} {} examples loaded'.format(len(data_split), split))

    if split in ['train', 'dev'] and augment_with_wikisql:
        data_dir = os.path.dirname(in_dir)
        wikisql_dir = os.path.join(data_dir, 'wikisql1.1')
        wikisql_split = load_data_split_wikisql(wikisql_dir, split, schema_graphs)
        data_split += wikisql_split
        print('{} {} examples loaded (+wikisql)'.format(len(data_split), split))

    return data_split


def load_parsed_sqls(args, augment_with_wikisql=False):
    data_dir = args.data_dir
    dataset = args.dataset_name
    norm_tag = get_norm_tag(args)
    in_json = os.path.join(data_dir, '{}.{}parsed.json'.format(dataset, norm_tag))
    if not os.path.exists(in_json):
        print('Warning: parsed SQL files not found!')
        return dict()
    with open(in_json) as f:
        parsed_sqls = json.load(f)
        print('{} parsed SQL queries loaded'.format(len(parsed_sqls)))

    if augment_with_wikisql:
        parent_dir = os.path.dirname(data_dir)
        wikisql_dir = os.path.join(parent_dir, 'wikisql1.1')
        wikisql_parsed_json = os.path.join(wikisql_dir, 'wikisql.parsed.json')
        with open(wikisql_parsed_json) as f:
            wikisql_parsed_sqls = json.load(f)
            print('{} parsed wikisql SQL queries loaded'.format(len(wikisql_parsed_sqls)))
        parsed_sqls.update(wikisql_parsed_sqls)
        print('{} parsed SQL queries loaded (+wikisql)'.format(len(parsed_sqls)))

    return parsed_sqls


def save_parsed_sqls(args, parsed_sqls):
    data_dir = args.data_dir
    dataset = args.dataset_name
    norm_tag = get_norm_tag(args)
    out_json = os.path.join(data_dir, '{}.{}parsed.json'.format(dataset, norm_tag))
    # save a copy of the parsed file before directly modifying it
    if os.path.exists(out_json):
        shutil.copyfile(out_json, os.path.join('/tmp', '{}.{}parsed.json'.format(dataset, norm_tag)))
    with open(out_json, 'w') as o_f:
        json.dump(parsed_sqls, o_f, indent=4)
        print('parsed SQL queries dumped to {}'.format(out_json))


def load_vocabs(args):
    """
    :return text_vocab: tokens appeared in the natural language query and schema
    :return program_vocab: tokens appeared in the program used for program generation
    :return world_vocab: tokens in the program that does not come from the input natural language query nor the schema
            (which likely needed to be inferred from world knowledge)
    """
    if args.model == 'seq2seq':
        return load_vocabs_seq2seq(args)
    elif args.model in ['seq2seq.pg', 'bridge', 'bridge.pt']:
        return load_vocabs_seq2seq_ptr(args)
    else:
        raise NotImplementedError


def load_vocabs_seq2seq(args):
    if args.share_vocab:
        vocab_path = get_vocab_path(args, 'full')
        vocab = load_vocab(vocab_path, args.vocab_min_freq,  tu=utils.get_trans_utils(args))
        text_vocab, program_vocab = vocab, vocab
    else:
        text_vocab_path = get_vocab_path(args, 'nl')
        text_vocab = load_vocab(text_vocab_path, args.text_vocab_min_freq, tu=utils.get_trans_utils(args))
        program_vocab_path = get_vocab_path(args, 'cm')
        program_vocab = load_vocab(program_vocab_path, args.program_vocab_min_freq)

    print('* text vocab size = {}'.format(text_vocab.size))
    print('* program vocab size = {}'.format(program_vocab.size))
    vocabs = {
        'text': text_vocab,
        'program': program_vocab
    }
    return vocabs


def load_vocabs_seq2seq_ptr(args):
    if args.pretrained_transformer:
        tu = utils.get_trans_utils(args)
        text_vocab = Vocabulary(tag='text', func_token_index=None, tu=tu)
        for v in tu.tokenizer.vocab:
            text_vocab.index_token(v, in_vocab=True, check_for_seen_vocab=True)
    else:
        text_vocab_path = get_vocab_path(args, 'nl')
        text_vocab = load_vocab(text_vocab_path, args.text_vocab_min_freq, tu=utils.get_trans_utils(args))

    program_vocab = sql_reserved_tokens if args.pretrained_transformer else sql_reserved_tokens_revtok

    print('* text vocab size = {}'.format(text_vocab.size))
    print('* program vocab size = {}'.format(program_vocab.size))
    print()
    vocabs = {
        'text': text_vocab,
        'program': program_vocab
    }
    return vocabs


def load_vocab(vocab_path, min_freq, tag='', func_token_index=None, tu=None):
    """
    :param vocab_path: path to vocabulary file.
    :param min_freq: minimum frequency of known vocabulary (does not apply to meta data tokens).
    :param tag: a tag to mark the purpose of the vocabulary
    :param functional_tokens: funtional tokens prepended to the vocabulary.
    :param tu: pre-trained transformer utility object to use.
    :return: token to id mapping and the reverse mapping.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = Vocabulary(tag=tag, func_token_index=func_token_index, tu=tu)
        for line in f.readlines():
            line = line.rstrip()
            v, freq = line.rsplit('\t', 1)
            freq = int(freq)
            in_vocab = is_functional_token(v) or freq < 0 or freq >= min_freq
            vocab.index_token(v, in_vocab, check_for_seen_vocab=True)
        print('vocab size = {}, loaded from {} with frequency threshold {}'.format(vocab.size, vocab_path, min_freq))
    return vocab
