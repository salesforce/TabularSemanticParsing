"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
"""
Test SQL parser extensions at scale.
"""

import collections
import json
import pyparsing
from pyparsing import ParseException
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
np.random.seed(100)
import random

from moz_sp import parse
from moz_sp import format
from moz_sp import check_schema_consistency
from moz_sp import convert_to_execution_order
from moz_sp import restore_clause_order
from moz_sp import extract_foreign_keys
from moz_sp import denormalize
from moz_sp import shallow_normalize
from moz_sp import tokenize
from moz_sp.debugs import DEBUG
from moz_sp.tests.unit_tests import complex_queries, denormalizer_unit_test

from src.data_processor.tokenizers import revtok_sql_tokenize, revtok_de_tokenize
from src.data_processor.data_loader import load_data_split_spider
from src.data_processor.schema_loader import load_schema_graphs_spider
from src.data_processor.tokenizers import sql_tokenize
import src.utils.trans.bert_utils as bu

dbs = [
    'geography',
    'imdb',
    'scholar',
    'restaurants',
    'yelp',
    'academic',
    'advising',
    'atis',
    'spider',
    'wikisql'
]


unparsable_queries = {
    'academic': {},
    'advising': {},
    'atis': {},
    'geography': {},
    'imdb': {},
    'scholar': {},
    'restaurants': {},
    'yelp': {},
    'spider': {},
    'wikisql': {}
}


def load_content(in_json):
    with open(in_json) as f:
        return json.load(f)


def load_sqls(in_json, normalize_variables):

    def fill_in_variables(s, s_vars, variables):
        var_list = {}
        for v_key, v_val in s_vars.items():
            if len(v_val) == 0:
                for var in variables:
                    if var['name'] == v_key:
                        v_val = var['example']
            s = s.replace(v_key, v_val)
            var_list[v_val] = v_key
        return s

    sqls = set()
    with open(in_json) as f:
        dataset = json.load(f)
        for example in dataset:
            variables = example['variables']
            for sentence in example['sentences']:
                s_variables = sentence['variables']
                for sql in example['sql']:
                    sql = sql.strip()
                    if sql.endswith(';'):
                        sql = sql[:-1].strip()
                    if not normalize_variables:
                        sql = fill_in_variables(sql, s_variables, variables)
                    sqls.add(sql)

    print('{} SQL queries loaded'.format(len(sqls)))

    return list(sqls)


def sample_queries(data_dir, db_name, out_dir):
    in_json = os.path.join(data_dir, '{}.json'.format(db_name))
    sqls = load_sqls(in_json, normalize_variables=True)

    num_samples = 3
    count = 0
    for idx in np.random.randint(0, len(sqls), num_samples):
        out_txt = os.path.join(out_dir, '{}-{}.txt'.format(db_name, count))
        with open(out_txt, 'w') as o_f:
            o_f.write(sqls[idx])
        print('{} SQL query saved to {}'.format(db_name, out_txt))
        count += 1


def batch_extract_foreign_keys(data_dir, dataset):

    def parse_column_name(cn):
        table_alias, column_name = cn.split('.')
        table_name, _ = table_alias.split('alias')
        return table_name, column_name

    references = {}

    num_parsed = 0
    num_queries = 0
    in_json = os.path.join(data_dir, '{}.json'.format(dataset))
    with open(in_json) as f:
        data = json.loads(f.read())
        for example in data:
            for sql_query in example['sql']:
                num_queries += 1
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1]
                try:
                    sql_query_pt = parse(sql_query)
                    cr = extract_foreign_keys(sql_query_pt)
                    for cn1 in cr:
                        t1, c1 = parse_column_name(cn1)
                        for cn2 in cr[cn1]:
                            t2, c2 = parse_column_name(cn2)
                            t_key = '{} -> {}'.format(t1, t2)
                            c_key = '{} -> {}'.format(c1, c2)
                            if not t_key in references:
                                references[t_key] = collections.defaultdict(int)
                            references[t_key][c_key] += 1
                    num_parsed += 1
                except ParseException as e:
                    print(e)
                except RecursionError as e:
                    print("RecursionError")
    print('{}/{} SQL queries parsed'.format(num_parsed, num_queries))

    out_cr_json = os.path.join(data_dir, '{}-column-references.json'.format(dataset))
    with open(out_cr_json, 'w') as o_f:
        json.dump(references, o_f, indent=4)
        print('Column references saved to {}'.format(out_cr_json))


def batch_parse(data_dir, db_name, normalize_variables, start_id=0, parse_all=False):
    in_json = os.path.join(data_dir, '{}.json'.format(db_name))
    if db_name in ['spider', 'wikisql']:
        sqls = set()
        with open(in_json) as f:
            content = json.load(f)
            for example in content:
                program = example['query']
                if program.endswith(';'):
                    program = program[:-1].rstrip()
                sqls.add(program)
        sqls = sorted(list(sqls))
    else:
        sqls = load_sqls(in_json, normalize_variables)
    sqls = sqls[start_id:]

    print('Parsing {} queries in {}...'.format(len(sqls), db_name))
    start_id_tag = '' if start_id == 0 else '.{}'.format(start_id)
    if normalize_variables:
        out_json = os.path.join(data_dir, '{}.norm.parsed.json'.format(db_name, start_id))
    else:
        out_json = os.path.join(data_dir, '{}.parsed.json'.format(db_name, start_id))
    out_err = os.path.join(data_dir, '{}{}.parsed.err.txt'.format(db_name, start_id))
    if os.path.exists(out_json):
        parse_trees = load_content(out_json)
        print('{} parsed queries loaded'.format(len(parse_trees)))
    else:
        parse_trees = {}

    with open(out_err, 'a') as e_o_f:
        num_errors = 0
        random.shuffle(sqls)
        for idx, sql in enumerate(sqls):
            try: 
                if idx in unparsable_queries[db_name]:
                    raise ValueError
                if parse_all or not sql in parse_trees:
                    sql_pt = parse(sql)
                    parse_trees[sql] = sql_pt
                    print(json.dumps(sql_pt, indent=4))
                    import pdb
                    pdb.set_trace()
                    # print('parsed')
                # else:
                #     print('parse tree cached')
            except Exception as e:
                e_o_f.write('error parsing: {}\n'.format(sql))
                e_o_f.write('{}\n\n'.format(e))
                num_errors += 1
            if idx > 0 and idx % 600 == 0:
                e_o_f.close()
                e_o_f = open(out_err, 'a')
                with open(out_json, 'w') as o_f:
                    json.dump(parse_trees, o_f, indent=4)
                    print('{} parsed SQL queries dumped to {}'.format(idx, out_json))

    with open(out_json, 'w') as o_f:
        json.dump(parse_trees, o_f, indent=4)
        print('parsed SQL queries dumped to {}'.format(out_json))
        print('{} parsing error cases dumped to {}'.format(num_errors, out_err))


def batch_format(data_dir, db_name, start_id=0):
    print('Testing {}...'.format(db_name))
    in_json = os.path.join(data_dir, '{}.json'.format(db_name))
    out_json = os.path.join(data_dir, '{}.formatted.json'.format(db_name))
    out_err = os.path.join(data_dir, '{}.formatted.err.txt'.format(db_name))

    sqls = load_sqls(in_json, normalize_variables=False)
    if os.path.exists(out_json):
        formatted_sqls = load_content(out_json)
    else:
        formatted_sqls = {}
    num_formatted = max([int(x) for x in list(formatted_sqls.keys())]) + 1 if formatted_sqls else 0
    assert(num_formatted == start_id or start_id == 0)
    if start_id == 0:
        formatted_sqls = {}

    with open(out_err, 'a') as e_o_f:
        for idx, sql in enumerate(sqls):
            if idx < start_id:
                continue
            sql = sqls[idx]
            try:
                sql_pt = parse(sql)
                print('{}. {}'.format(idx, sql))
                formatted_sql = format(sql_pt)
                formatted_sqls[idx] = formatted_sql
            except Exception as e:
                e_o_f.write('error parsing: {}\n'.format(sql))
                e_o_f.write('{}\n\n'.format(e))

    with open(out_json, 'w') as o_f:
        json.dump(formatted_sqls, o_f, indent=4)
        print('formatted SQL queries dumped to {}'.format(out_json))


def batch_tokenize(data_dir, db_name, denormalization=False):
    print('Tokenizing {}'.format(db_name))
    num_tokens_list = []
    if db_name == 'spider':
        schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
        in_json = os.path.join(data_dir, 'dev.json')
        with open(in_json) as f:
            content = json.load(f)
            for i, example in enumerate(content):
                sql = example['query']
                if sql.endswith(';'):
                    sql = sql[:-1]
                db_name = example['db_id']
                schema = schema_graphs[db_name]
                ast = parse(sql)
                denormalized_ast, _ = denormalize(ast, schema, return_parse_tree=True)
                tokens, token_types, constants = \
                    sql_tokenize(denormalized_ast, bu.tokenizer.tokenize,
                                 return_token_types=True, schema=schema,
                                 keep_singleton_fields=True, atomic_value=True,
                                 num_token=' <NUM> ', str_token=' <STRING> ')
                num_tokens_list.append(len(tokens))
    else:
        return NotImplementedError
    print('{}: avg # tokens = {}'.format(db_name, np.mean(num_tokens_list)))


def batch_denormalize():
    data_dir = sys.argv[1]
    schema_graphs = load_schema_graphs_spider(data_dir, 'spider')
    train_data = load_data_split_spider(data_dir, 'train', schema_graphs)
    # random.shuffle(train_data)
    for i, example in enumerate(train_data):
        if not example.program in complex_queries:
            continue
        schema_graph = schema_graphs.get_schema(example.db_id)
        if DEBUG:
            denormalizer_unit_test(example.program, schema_graph)
        else:
            try:
                denormalizer_unit_test(example.program, schema_graph)
            except KeyError as e:
                example.pretty_print(schema_graph)
                print(str(e))
                import pdb
                pdb.set_trace()
            except AssertionError as e:
                example.pretty_print(schema_graph)
                print(str(e))
                import pdb
                pdb.set_trace()
            except pyparsing.ParseException as e:
                example.pretty_print(schema_graph)
                print(str(e))
                import pdb
                pdb.set_trace()
        if i > 0 and i % 500 == 0:
            print('{} examples processed'.format(i))


def parse_queries():
    data_dir = sys.argv[1]
    start_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for db_name in dbs:
        if db_name != 'spider':
            continue
        batch_parse(data_dir, db_name, normalize_variables=False, start_id=start_id)


def test_tokenizer():
    data_dir = sys.argv[1]
    db_name = sys.argv[2]
    if db_name == 'spider':
        batch_tokenize(data_dir, db_name, denormalization=True)


def test_restore_clause_order():
    in_json = sys.argv[1]
    data_dir = sys.argv[2]
    schema_graphs = load_schema_graphs_spider(data_dir)

    with open(in_json) as f:
        content = json.load(f)
        num_errors = 0
        for i, example in enumerate(content):
            sql = example['query']
            print('Orig SQL: {}'.format(sql))
            db_name = example['db_id']
            schema = schema_graphs[db_name]
            # sn_sql = shallow_normalize(sql, schema)
            dn_sql, _ = denormalize(sql, schema)
            print('DN SQL:\t\t{}'.format(dn_sql))
            eo_sql = convert_to_execution_order(dn_sql, schema)
            restored_sql = restore_clause_order(eo_sql, schema)
            print('Restored SQL:\t{}'.format(restored_sql))
            # print('EO Pred SQL: {}'.format(eo_sql))
            # print('SN SQL:\t\t{}'.format(sn_sql))
            print()
            if dn_sql != restored_sql:
                num_errors += 1
                import pdb
                pdb.set_trace()
        print('{}/{} errors detected'.format(num_errors, len(content)))


if __name__ == '__main__':
    # parse_queries()
    test_tokenizer()
    # test_restore_clause_order()
