"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Preprocessing text-to-SQL dataset.
"""
import collections
import pickle
from tqdm import tqdm

import moz_sp.sql_tokenizer as sql_tokenizer
from src.data_processor.data_loader import load_parsed_sqls, save_parsed_sqls
from src.data_processor.data_loader import load_vocabs
from src.data_processor.data_stats import DatasetStatistics
import src.data_processor.processors.data_processor_spider as data_processor_spider
import src.data_processor.processors.data_processor_wikisql as data_processor_wikisql
from src.data_processor.processor_utils import Text2SQLExample, AugmentedText2SQLExample, WIKISQL
from src.data_processor.path_utils import get_processed_data_path
import src.data_processor.tokenizers as tok
import src.data_processor.vectorizers as vec
from src.eval.eval_constant_extraction import SchemaLinkingEvaluator
from src.utils.utils import SEQ2SEQ, SEQ2SEQ_PG, BRIDGE
import src.utils.utils as utils


spider_dev_dbs = {
    'employee_hire_evaluation',
    'battle_death',
    'student_transcripts_tracking',
    'poker_player',
    'wta_1',
    'world_1',
    'dog_kennels',
    'tvshow',
    'museum_visit',
    'voter_1',
    'singer',
    'pets_1',
    'concert_singer',
    'real_estate_properties',
    'orchestra',
    'course_teach',
    'cre_Doc_Template_Mgt',
    'network_1',
    'flight_2',
    'car_1'
}


spider_empty_dbs = {
    'music_2',
    'scholar',
    'sakila_1',
    'yelp',
    'geo',
    'academic',
    'formula_1',
    'restaurants',
    'imdb'
}


def preprocess(args, dataset, process_splits=('train', 'dev', 'test'), print_aggregated_stats=False, verbose=False,
               save_processed_data=True):
    """
    Data pre-processing for baselines that does only shallow processing on the schema.
    """
    text_tokenize, program_tokenize, post_process, trans_utils = tok.get_tokenizers(args)
    parsed_programs = load_parsed_sqls(args, augment_with_wikisql=args.augment_with_wikisql)
    num_parsed_programs = len(parsed_programs)

    vocabs = load_vocabs(args)

    schema_graphs = dataset['schema']
    schema_graphs.lexicalize_graphs(tokenize=text_tokenize, normalized=(args.model_id in [BRIDGE]))

    ############################
    # data statistics
    ds = DatasetStatistics()
    ############################

    # parallel data
    for split in process_splits:
        if split not in dataset:
            continue
        ds_split, sl_split = preprocess_split(dataset, split, args, parsed_programs,
                                              text_tokenize, program_tokenize, post_process, trans_utils,
                                              schema_graphs, vocabs, verbose=verbose)
        ds_split.print(split)
        sl_split.print()
        ############################
        # update data statistics
        ds.accumulate(ds_split)
        ############################

    if len(parsed_programs) > num_parsed_programs:
        save_parsed_sqls(args, parsed_programs)

    if print_aggregated_stats:
        ds.print()

    if save_processed_data:
        out_pkl = get_processed_data_path(args)
        with open(out_pkl, 'wb') as o_f:
            pickle.dump(dataset, o_f)
            print('Processed data dumped to {}'.format(out_pkl))


def preprocess_split(dataset, split, args, parsed_programs, text_tokenize, program_tokenize, post_process, trans_utils,
                     schema_graphs, vocabs, cache_examples=False, verbose=False):
    data_split = dataset[split]
    print('processing {} examples from {}...'.format(len(data_split), split))

    ############################
    # data statistics
    ds = DatasetStatistics()
    sl = SchemaLinkingEvaluator()
    ############################

    if args.dataset_name == 'wikisql':
        preprocess_example = data_processor_wikisql.preprocess_example
    elif args.dataset_name == 'spider':
        preprocess_example = data_processor_spider.preprocess_example
    else:
        raise NotImplementedError

    START_PROCESS = False
    for i, example in enumerate(tqdm(data_split)):
        # if example.db_name != 'assets_maintenance':
        #     continue
        # if 'Glenn' in example.text:
        #     START_PROCESS = True
        # if not START_PROCESS:
        #     continue
        # print(example.text)
        schema_graph = schema_graphs.get_schema(example.db_id)
        query_oov, denormalized, schema_truncated, token_restored = \
            preprocess_example(split, example, args,
                               parsed_programs,
                               text_tokenize,
                               program_tokenize,
                               post_process,
                               trans_utils,
                               schema_graph,
                               vocabs,
                               verbose=verbose)

        # evaluate value extraction
        sl.eval_const_f1(example.values, [example.matched_values[pos] for pos in example.matched_values],
                         eval_field=True)
        if sl.f1s[-1] < 1:
            print('--------------------')
            print('text: ', example.text)
            print('sql: ', example.program)
            print('ground truth values: ', example.values)
            print('matched values', example.matched_values)
            print('--------------------')

        # update data statistics
        ############################
        ds.num_examples += 1
        if query_oov:
            ds.num_oov += 1
        if not denormalized:
            ds.num_denormalization_failed += 1
        if schema_truncated:
            ds.num_schema_truncated += 1
        if token_restored:
            ds.num_token_restored += 1
        ############################

        if split == 'train' and isinstance(example, Text2SQLExample):
            for _, var_val in example.values:
                var_tokens = text_tokenize(var_val)
                if len(var_tokens) > ds.max_ptr_span_size:
                    ds.max_ptr_span_size = len(var_tokens)
        ds.num_text_tokens.append(example.num_text_tokens)
        ds.num_input_tokens.append(example.num_input_tokens)
        ds.num_cm_tokens.append(example.num_program_tokens)
        ds.num_cm_whole_field_tokens.append(len(example.program_singleton_field_input_ids))

        if i > 0 and i % 100000 == 0:
            print('{} examples processed'.format(i))
            if cache_examples:
                with open('temp_{}.pkl'.format(i), 'wb') as o_f:
                    pickle.dump(data_split, o_f)

    return ds, sl


def extract_value_spans(program_tokens, program_token_types, tu):
    values = []
    value, is_value = [], False
    for t, t_type in zip(program_tokens, program_token_types):
        if t_type == sql_tokenizer.VALUE:
            value.append(t)
        else:
            if value:
                value_str = tu.tokenizer.convert_tokens_to_string(value)
                value_str = value_str.replace(' . ', '.')
                value_str = value_str.replace(' @ ', '@')
                value_str = value_str.replace(' - ', '-')
                if not utils.is_number(value_str):
                    values.append(value_str)
                value = []
    return values
