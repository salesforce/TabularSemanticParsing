#!/usr/bin/env python3

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Load database schema.
"""

import collections
import csv
import json
import sqlite3
import os
import re
import sys

from src.data_processor.schema_graph import SchemaGraph, WikiSQLSchemaGraph, SchemaGraphs


def get_field_id(table_name, field_name):
    return '.'.join([table_name, field_name])


def load_csv_schema(data_dir, dataset):
    """
    Load table schema and foreign key references.
    """
    in_csv = os.path.join(data_dir, '{}-schema.csv'.format(dataset))
    in_cr_json = os.path.join(data_dir, '{}-column-references.json'.format(dataset))

    schema = {
        "name": dataset,
        "children": []
    }
    tables = schema['children']
    join_anchors = []
    field_index = {}
    field_ids = set()

    with open(in_csv) as f:
        reader = csv.DictReader(f)
        current_tn = ''
        current_table = None
        for row in reader:
            if row['Table Name'] == '-':
                continue
            table_name = row['Table Name'].strip().upper()
            field_name = row[' Field Name'].strip().upper()
            is_primary_key = int(row[' Is Primary Key'].strip().lower() in ['y', 'yes', 'pri'])
            is_foreign_key = int(row[' Is Foreign Key'].strip().lower() in ['y', 'yes', 'pri'])
            type = row[' Type']
            if table_name != current_tn:
                if current_table:
                    tables.append(current_table)
                current_table = {
                    'id': table_name,
                    'name': table_name,
                    'children': [],
                    'is_primary_key': -1,
                    'is_foreign_key': -1,
                    'type': None
                }
                current_tn = table_name

            # add to database schema
            current_table['children'].append({
                'id': get_field_id(table_name, field_name),
                'name': field_name,
                'is_primary_key': is_primary_key,
                'is_foreign_key': is_foreign_key,
                'type': type
            })
            field_ids.add(get_field_id(table_name, field_name))

            # coarse extraction of join anchors based on column name matching
            # Note: skip primary key and foreign key filtering since the DB schema contain noise
            # if is_primary_key or is_foreign_key:
            if not field_name in field_index:
                field_index[field_name] = [table_name]
            else:
                for table_name0 in field_index[field_name]:
                    join_anchors.append({
                        'source': (table_name0, field_name),
                        'target': (table_name, field_name)
                    })
                field_index[field_name].append(table_name)

        if current_table:
            tables.append(current_table)

    with open(in_cr_json) as f:
        references = json.load(f)

        foreign_keys = set()
        foreign_key_names = set()
        for t_key in references:
            t1, t2 = t_key.split(' -> ')
            for c_key in references[t_key]:
                c1, c2 = c_key.split(' -> ')
                foreign_keys.add((t1, c1))
                foreign_keys.add((t2, c2))
                foreign_key_names.add(c1)
                foreign_key_names.add(c2)

        for join_anchor in join_anchors:
            if join_anchor['source'][1] in foreign_key_names and \
                    join_anchor['target'][1] in foreign_key_names:
                t1, c1 = join_anchor['source']
                t2, c2 = join_anchor['target']
                t_key = '{} -> {}'.format(t1, t2)
                c_key = '{} -> {}'.format(c1, c2)
                if not t_key in references:
                    references[t_key] = collections.defaultdict(int)
                if not c_key in references[t_key]:
                    references[t_key][c_key] = 0
                references[t_key][c_key] += 1

        _join_anchors = []
        for t_key in references:
            t1, t2 = t_key.split(' -> ')
            for c_key in references[t_key]:
                c1, c2 = c_key.split(' -> ')
                source_field_id = get_field_id(t1, c1)
                target_field_id = get_field_id(t2, c2)
                assert(source_field_id in field_ids and target_field_id in field_ids)
                _join_anchors.append({
                    'source': source_field_id,
                    'target': target_field_id
                })

    return {
            "schema": schema,
            "join_anchors": _join_anchors
        }


def load_schema_graphs(args):
    """
    Load database schema as a graph.
    """
    dataset_name = args.dataset_name
    if dataset_name in ['spider', 'spider_ut']:
        return load_schema_graphs_spider(args.data_dir, dataset_name, db_dir=args.db_dir,
                                         augment_with_wikisql=args.augment_with_wikisql)
    if dataset_name == 'wikisql':
        return load_schema_graphs_wikisql(args.data_dir)

    in_csv = os.path.join(args.data_dir, '{}-schema.csv'.format(dataset_name))
    schema_graphs = SchemaGraphs()
    schema_graph = SchemaGraph(dataset_name)
    schema_graph.load_data_from_finegan_dollak_csv_file(in_csv)
    schema_graphs.index_schema_graph(schema_graph)
    return schema_graphs


def load_schema_graphs_spider(data_dir, dataset_name, db_dir=None, augment_with_wikisql=False):
    """
    Load indexed database schema.
    """
    in_json = os.path.join(data_dir, 'tables.json')
    schema_graphs = SchemaGraphs()

    with open(in_json) as f:
        content = json.load(f)
        for db_content in content:
            db_id = db_content['db_id']
            if dataset_name == 'spider':
                db_path = os.path.join(db_dir, db_id, '{}.sqlite'.format(db_id)) if db_dir else None
            else:
                db_id_parts = db_id.rsplit('_', 1)
                if len(db_id_parts) > 1:
                    m_suffix_pattern = re.compile('m\d+')
                    m_suffix = db_id_parts[1]
                    if re.fullmatch(m_suffix_pattern, m_suffix):
                        db_base_id = db_id_parts[0]
                    else:
                        db_base_id = db_id
                else:
                    db_base_id = db_id_parts[0]
                db_path = os.path.join(db_dir, db_base_id, '{}.sqlite'.format(db_base_id)) if db_dir else None
            schema_graph = SchemaGraph(db_id, db_path)
            if db_dir is not None:
                schema_graph.compute_field_picklist()
            schema_graph.load_data_from_spider_json(db_content)
            schema_graphs.index_schema_graph(schema_graph)
        print('{} schema graphs loaded'.format(schema_graphs.size))

    if augment_with_wikisql:
        parent_dir = os.path.dirname(data_dir)
        wikisql_dir = os.path.join(parent_dir, 'wikisql1.1')
        wikisql_schema_graphs = load_schema_graphs_wikisql(wikisql_dir)
        for db_id in range(wikisql_schema_graphs.size):
            schema_graph = wikisql_schema_graphs.get_schema(db_id)
            schema_graphs.index_schema_graph(schema_graph)
        print('{} schema graphs loaded (+wikisql)'.format(schema_graphs.size))

    return schema_graphs


def load_schema_graphs_wikisql(data_dir, splits=['train', 'dev', 'test']):
    schema_graphs = SchemaGraphs()

    for split in splits:
        in_jsonl = os.path.join(data_dir, '{}.tables.jsonl'.format(split))
        db_count = 0
        with open(in_jsonl) as f:
            for line in f:
                table = json.loads(line.strip())
                db_name = table['id']
                schema_graph = WikiSQLSchemaGraph(db_name, table, caseless=False)
                schema_graph.id = table['id']
                schema_graph.load_data_from_wikisql_json(table)
                schema_graph.compute_field_picklist(table)
                schema_graphs.index_schema_graph(schema_graph)
                db_count += 1
        print('{} databases in {}'.format(db_count, split))
    print('{} databases loaded in total'.format(schema_graphs.size))

    return schema_graphs


def load_schema_graphs_ask_data(data_dir):
    datasets = [
        'airbnb_san_francisco',
        'airbnb_seattle',
        'sports_salaries',
        'wines'
    ]

    schema_graphs = SchemaGraphs()
    for dataset in datasets:
        in_csv = os.path.join(data_dir, 'raw/{}.csv'.format(dataset))
        schema_graph = SchemaGraph(dataset)
        schema_graph.load_data_from_csv_file(in_csv)
        schema_graphs.index_schema_graph(schema_graph)
        schema_graph.pretty_print()
    print('{} schema graphs loaded'.format(schema_graphs.size))


if __name__ == '__main__':
    target_db_name = sys.argv[1]
    data_dir = sys.argv[2]
