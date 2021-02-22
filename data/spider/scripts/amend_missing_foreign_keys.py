"""
Check and correct errors in existing datasets.
"""

import collections
import itertools
import json
import os
import shutil
import sys

from moz_sp import parse, denormalize, extract_foreign_keys
from src.data_processor.data_loader import load_data_split_spider
from src.data_processor.processor_utils import get_ast
from src.data_processor.schema_loader import load_schema_graphs_spider


def load_spider_tables(in_json):
    with open(in_json) as f:
        tables = json.load(f)
    return tables


def amend_missing_foreign_keys():
    data_dir = sys.argv[1]
    table_path = os.path.join(data_dir, 'tables.json')
    tables = load_spider_tables(table_path)

    num_foreign_keys_added = 0
    for table in tables:
        c_dict = collections.defaultdict(list)
        for i, c in enumerate(table['column_names_original']):
            c_name = c[1].lower()
            c_dict[c_name].append(i)
        primary_keys = table['primary_keys']
        # print(primary_keys)
        foreign_keys = set([tuple(sorted(x)) for x in table['foreign_keys']])
        for c_name in c_dict:
            if c_name in ['name', 'id', 'code']:
                continue
            if len(c_dict[c_name]) > 1:
                for p, q in itertools.combinations(c_dict[c_name], 2):
                    if p in primary_keys or q in primary_keys:
                        if not (p, q) in foreign_keys:
                            foreign_keys.add((p, q))
                            print('added: {}-{}, {}-{}'.format(p, table['column_names_original'][p],
                                                               q, table['column_names_original'][q]))
                            num_foreign_keys_added += 1
                            # if num_foreign_keys_added % 10 == 0:
                            #     import pdb
                            #     pdb.set_trace()
        foreign_keys = sorted(list(foreign_keys), key=lambda x: x[0])
        table['foreign_keys'] = foreign_keys

    print('{} foreign key pairs added'.format(num_foreign_keys_added))
    shutil.copyfile(table_path, os.path.splitext(table_path)[0] + '.original.json')
    with open(table_path, 'w') as o_f:
        json.dump(tables, o_f, indent=4)


def check_foreign_keys_in_queries():
    data_dir = sys.argv[1]
    dataset = sys.argv[2]
    schema_graphs = load_schema_graphs_spider(data_dir)
    train_data = load_data_split_spider(data_dir, 'train', schema_graphs)
    dev_data = load_data_split_spider(data_dir, 'dev', schema_graphs)
    tables = load_spider_tables(os.path.join(data_dir, 'tables.json'))
    table_dict = dict()
    for table in tables:
        table_dict[table['db_id']] = table
    in_json = os.path.join(data_dir, '{}.parsed.json'.format(dataset))
    with open(in_json) as f:
        parsed_sqls = json.load(f)

    for i, example in enumerate(train_data + dev_data):
        schema_graph = schema_graphs.get_schema(example.db_id)
        ast, _ = get_ast(example.program, parsed_sqls, denormalize_sql=True, schema_graph=schema_graph)
        foreign_keys_readable, foreign_keys = extract_foreign_keys(ast, schema_graph)
        for f_key in foreign_keys:
            if not tuple(sorted(f_key)) in schema_graph.foreign_keys:
                print(example.program)
                print(json.dumps(ast, indent=4))
                print('Missing foreign key detected:')
                print('- {}'.format(schema_graph.get_field_signature(f_key[0])))
                print('- {}'.format(schema_graph.get_field_signature(f_key[1])))
                import pdb
                pdb.set_trace()


if __name__ == '__main__':
    amend_missing_foreign_keys()
    # check_foreign_keys_in_queries()
