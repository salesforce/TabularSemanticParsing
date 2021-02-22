#  -*- coding: utf-8 -*-

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 We represent each database schema using a graph structure.
"""

from asciitree import LeftAligned
import collections
import csv
import functools
from mo_future import string_types
import numpy as np
np.random.seed(100)
import random
import scipy.sparse as ssp
import sqlalchemy
import sqlite3

import src.common.ops as ops
import src.common.content_encoder as ce
from src.data_processor.sql.sql_operators import field_types
from src.data_processor.vocab_utils import Vocabulary
from src.utils.utils import deprecated
import src.utils.utils as utils


TABLE = 0
FIELD = 1
DATA_TYPE = 2

DUMMY_REL = 0
SELF = 1
TABLE_FIELD_REF = 2
FIELD_TABLE_REF = 3
TABLE_FIELD_PRI = 4
FIELD_TABLE_PRI = 5
FOREIGN_PRI = 6
FOREIGN_FOR = 7
SAME_TABLE = 8
FOREIGN_TAB_F = 9
FOREIGN_TAB_R = 10
FOREIGN_TAB_B = 11

graph_relations = [
    DUMMY_REL,
    SELF,
    TABLE_FIELD_REF,
    FIELD_TABLE_REF,
    TABLE_FIELD_PRI,
    FIELD_TABLE_PRI,
    FOREIGN_PRI,
    FOREIGN_FOR,
    SAME_TABLE,
    FOREIGN_TAB_F,
    FOREIGN_TAB_R,
    FOREIGN_TAB_B
]


class Node(object):
    def __init__(self, node_type, name, n_name=None, caseless=True):
        self.node_type = node_type
        self.name = name
        self.normalized_name = n_name if n_name else name
        self.indexable_name = utils.to_indexable(name, caseless)
        self.lexical_features = None

    def compute_lexical_features(self, tokenize=None, normalized=False):
        name = self.normalized_name if normalized else self.name
        if tokenize is None:
            self.lexical_features = name.split(' ')
        else:
            self.lexical_features = tokenize(name)

    @property
    def signature(self):
        return self.name

    @property
    def indexable_signature(self):
        return self.indexable_name

    @property
    def printable_name(self):
        return '{} ({})'.format(self.name, self.normalized_name)


class DataType(Node):
    def __init__(self, name, n_name=None):
        super().__init__(DATA_TYPE, name, n_name)


class Table(Node):
    def __init__(self, name, n_name=None, caseless=True):
        super().__init__(TABLE, name, n_name, caseless)
        # TODO: When we scramble the table/field order, the two attributes below are invalid.
        # self.table_id = None
        # self.node_id = None
        self.fields = []
        self.num_rows = None

    @property
    def num_fields(self):
        return len(self.fields)


class Field(Node):
    def __init__(self, table, name, n_name=None, caseless=True, data_type='text', is_primary_key=False,
                 is_foreign_key=False):
        super().__init__(FIELD, name, n_name, caseless)
        table.fields.append(self)
        # TODO: When we scramble the table/field order, the two attributes below are invalid.
        # self.field_id = None
        # self.node_id = None
        self.table = table
        self.data_type = data_type
        self.is_primary_key = is_primary_key
        self.is_foreign_key = is_foreign_key

    def get_serialization(self, tu, with_table=True):
        features = []
        if with_table:
            features.append(tu.foreign_key_ref_table_marker)
            features.extend(self.table.lexical_features)
        features.append(tu.foreign_key_ref_field_marker)
        features.extend(self.lexical_features)
        return features

    @property
    def signature(self):
        return self.table.name + '.' + self.name

    @property
    def indexable_signature(self):
        return self.table.indexable_name + '.' + self.indexable_name

    @property
    def printable_name(self):
        p_badge = ' [PRIMARY]' if self.is_primary_key else ''
        f_badge = ' [FOREIGN]' if self.is_foreign_key else ''
        return '{} ({}){}{}'.format(self.name, self.normalized_name, p_badge, f_badge)

    @property
    def is_numeric(self):
        return self.data_type == 'number'


class SchemaGraphs(object):
    def __init__(self):
        self.db_index, self.db_rev_index = dict(), dict()
        self.lexicalized = False

    def get_db_id(self, db_name):
        return self.db_index[db_name]

    def get_current_schema_layout(self, pad_id=None, add_paddings=False):
        F = []
        for db_id in range(self.size):
            schema_graph = self.db_rev_index[db_id]
            F.append(schema_graph.get_current_schema_layout())
        if add_paddings:
            return ops.pad_and_cat_2d(F, pad_id)
        else:
            return F

    def get_schema(self, db_id):
        return self.db_rev_index[db_id]

    def get_lexical_vocab(self):
        vocab = Vocabulary('schema')
        for db_name in self.db_index:
            db_id = self.db_index[db_name]
            schema_graph = self.db_rev_index[db_id]
            vocab.merge_with(schema_graph.get_lexical_vocab())
        return vocab

    def index_schema_graph(self, schema_graph):
        db_id = len(self.db_index)
        assert(schema_graph.name not in self.db_index)
        self.db_index[schema_graph.name] = db_id
        self.db_rev_index[db_id] = schema_graph

    def lexicalize_graphs(self, tokenize=None, normalized=False):
        for db_name in self.db_index:
            schema_graph = self.__getitem__(db_name)
            schema_graph.lexicalize_graph(tokenize=tokenize, normalized=normalized)
        self.lexicalized = False

    def __getitem__(self, db_name):
        db_id = self.get_db_id(db_name)
        return self.db_rev_index[db_id]

    @property
    def size(self):
        return len(self.db_index)


class SchemaGraph(object):
    """
    Schema Graph Representation.

    We maintain three types of nodes in a schema graph:
        1. tables
        2. fields
        3. data types.
    The nodes are connected with five types of relations:
        1. TABLE_REF -- connects tables in the same database
        2. TABLE_FIELD_PRI (and reverse) -- connects a table with its primary key(s)
        3. TABLE_FIELD_FOR (and reverse) -- connects a table with its foreign key(s)
        4. TABLE_FIELD_REF (and reverse) -- connects a table with its fields
        5. FIELD_TYPE_REF (and reverse) -- connects a field with its data type.
    """

    def __init__(self, name, db_path=None, caseless=True):
        self.id = None
        self.name = name
        self.db_path = db_path

        self.caseless = caseless

        self.table_index, self.table_rev_index = dict(), dict()
        self.field_index, self.field_rev_index = dict(), dict()
        self.node_index, self.node_rev_index = dict(), dict()
        self.table_names = set()
        self.field_names = set()
        self.printable = {self.name: collections.OrderedDict()}

        self.adj_matrix = None
        self.lexicalized = False

        self.bert_feature_idx = dict()
        self.bert_feature_idx_rev = dict()

        self.foreign_key_index = dict()
        self.foreign_key_pairs = []
        self.field_id_to_official, self.official_to_field_id = dict(), dict()

        self.picklists = dict()

        self.question_field_match_cache = dict()


    def get_table_id(self, signature):
        signature = utils.to_indexable(signature, self.caseless)
        return self.table_index.get(signature, None)

    def get_field_id(self, signature):
        signature = utils.to_indexable(signature, self.caseless)
        return self.field_index.get(signature, None)

    def get_table(self, table_id):
        return self.table_rev_index[table_id]

    def get_field(self, field_id):
        return self.field_rev_index[field_id]

    def get_table_by_name(self, table_name):
        table_id = self.get_table_id(table_name)
        return self.get_table(table_id)

    def get_field_by_name(self, field_name):
        field_id = self.get_field_id(field_name)
        return self.get_field(field_id)

    def get_field_signature(self, field_id):
        return self.field_rev_index[field_id].signature

    def is_table_name(self, name):
        return utils.to_indexable(name, self.caseless) in self.table_names

    def is_field_name(self, name):
        return utils.to_indexable(name, self.caseless) in self.field_names

    def field_in_table(self, f_name, t_name):
        indexable_signature = '{}.{}'.format(utils.to_indexable(t_name, self.caseless),
                                             utils.to_indexable(f_name, self.caseless))
        return indexable_signature in self.field_index

    def get_foreign_keys_between_tables(self, tn1, tn2):
        table_id1 = self.get_table_id(tn1)
        table_id2 = self.get_table_id(tn2)
        key = (table_id1, table_id2)
        return self.foreign_key_index.get(key, None)

    # --- serialized representation --- #

    def get_schema_perceived_order(self, tables=None, random_table_order=False, random_field_order=False):
        if tables is None:
            tables = list(range(self.num_tables))
        table_perceived_order = random.sample(tables, k=len(tables)) if random_table_order else tables
        field_perceived_order = dict()
        for table_id in tables:
            table = self.get_table(table_id)
            field_order = list(range(table.num_fields))
            if random_field_order:
                primary_keys, other_fields = [], []
                for field_id in field_order:
                    if table.fields[field_id].is_primary_key:
                        primary_keys.append(field_id)
                    else:
                        other_fields.append(field_id)
                field_perceived_order[table_id] = primary_keys + random.sample(other_fields, k=len(other_fields))
            else:
                field_perceived_order[table_id] = field_order
        return table_perceived_order, field_perceived_order

    def get_schema_pos(self, signature):
        if signature == '*':
            return 0
        signature = utils.to_indexable(signature, self.caseless)
        # print(signature, self.bert_feature_idx.get(signature, None))
        return self.bert_feature_idx.get(signature, None)

    def get_signature_by_schema_pos(self, schema_pos, table_po=None, field_po=None):
        """
        Return signature of schema component based on its position in the serialization.
        """
        if schema_pos == 0:
            return '*'
        if table_po is None:
            if not schema_pos in self.bert_feature_idx_rev:
                return 'Unknown_Schema_Component'
            return self.bert_feature_idx_rev[schema_pos].signature
        else:
            assert(field_po is not None)

        sp = 1
        for table_id in table_po:
            table = self.table_rev_index[table_id]
            if sp == schema_pos:
                return table.signature
            sp += 1
            for field_id in field_po[table_id]:
                field = table.fields[field_id]
                if sp == schema_pos:
                    return field.signature
                sp += 1

    def get_serialization(self, tu, table_po=None, field_po=None,
                          asterisk_marker=None, table_marker=None, field_marker=None,
                          flatten_features=False, use_typed_field_markers=False,
                          use_graph_encoding=False, question_encoding=None,
                          top_k_matches=1, match_threshold=0.85,
                          num_values_per_field=0, no_anchor_text=False, verbose=True):
        use_picklist = question_encoding is not None
        if asterisk_marker is None:
            asterisk_marker = tu.asterisk_marker
        if table_marker is None:
            table_marker = tu.table_marker
        if field_marker is None:
            field_marker = tu.field_marker
        self.bert_feature_idx = dict()
        self.bert_feature_idx_rev = dict()
        matched_values = collections.OrderedDict()

        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()

        bert_features = [[asterisk_marker]]
        schema_pos = len(bert_features)
        for table_id in table_po:
            table_features = [table_marker]
            table_node = self.table_rev_index[table_id]
            if num_values_per_field > 0:
                if self.num_rows(table_id) >= num_values_per_field:
                    row_ids = random.sample(range(self.num_rows(table_id)), k=num_values_per_field)
                else:
                    row_ids = range(self.num_rows(table_id))
                    np.random.shuffle(row_ids)
                row_values = [self.get_row(table_id, row_id=row_id, mask_fill=True) for row_id in row_ids]
                while len(row_values) < num_values_per_field:
                    row_values.append([tu.tokenizer.mask_token for _ in range(self.get_table(table_id).num_fields)])
            else:
                row_values = None
            self.bert_feature_idx[table_node.indexable_signature] = schema_pos
            self.bert_feature_idx_rev[schema_pos] = table_node
            schema_pos += 1
            table_features.extend(table_node.lexical_features)
            for i in field_po[table_id]:
                field_node = table_node.fields[i]
                if use_typed_field_markers:
                    if field_node.data_type == 'text':
                        table_features.append(tu.text_field_marker)
                    elif field_node.data_type == 'number':
                        table_features.append(tu.number_field_marker)
                    elif field_node.data_type == 'time':
                        table_features.append(tu.time_field_marker)
                    elif field_node.data_type == 'boolean':
                        table_features.append(tu.boolean_field_marker)
                    else:
                        table_features.append(tu.other_field_marker)
                else:
                    table_features.append(field_marker)
                self.bert_feature_idx[field_node.indexable_signature] = schema_pos
                self.bert_feature_idx_rev[schema_pos] = field_node
                schema_pos += 1
                table_features.extend(field_node.lexical_features)
                field_id = self.get_field_id(field_node.signature)
                if use_graph_encoding and field_node.is_foreign_key:
                    foreign_key_refs = self.foreign_key_index[field_id]
                    for ref_id in foreign_key_refs:
                        ref_node = self.get_field(ref_id)
                        if ref_node.is_primary_key:
                            table_features.extend(ref_node.get_serialization(tu, with_table=True))
                if use_picklist:
                    picklist = self.get_field_picklist(field_id)
                    if picklist and isinstance(picklist[0], string_types):
                        key = (question_encoding, table_node.name, field_node.name)
                        if key in self.question_field_match_cache:
                            matches = self.question_field_match_cache[key]
                        else:
                            matches = ce.get_matched_entries(
                                question_encoding, picklist, m_theta=match_threshold, s_theta=match_threshold)
                            self.question_field_match_cache[key] = matches
                        if matches:
                            num_values_inserted = 0
                            for match_str, (field_value, s_match_str, match_score, s_match_score, match_size) in matches:
                                if 'name' in field_node.normalized_name and match_score * s_match_score < 1:
                                    continue
                                if table_node.name != 'sqlite_sequence':        # Spider database artifact
                                    table_features.append(tu.value_marker)
                                    value_start_pos = sum([len(x) for x in bert_features]) + len(table_features)
                                    matched_values[value_start_pos] = (field_node.signature, field_value)
                                    if not no_anchor_text:
                                        table_features.extend(tu.tokenizer.tokenize(field_value))
                                    if verbose:
                                        print('Picklist: {}, {}, {}, [{}]'.format(
                                            question_encoding, table_node.name, field_node.name, field_value))
                                    num_values_inserted += 1
                                    if num_values_inserted >= top_k_matches:
                                        break
                if num_values_per_field > 0:
                    assert(len(row_values) == num_values_per_field)
                    for j in range(num_values_per_field):
                        table_features.append(tu.value_marker)
                        row_value = utils.to_string(row_values[j][i])
                        if not row_value:
                            print(row_values)
                        table_features.extend(tu.tokenizer.tokenize(row_value))
            bert_features.append(table_features)
        if flatten_features:
            bert_features = [x for table_features in bert_features for x in table_features]
        return bert_features, matched_values

    def get_primary_key_ids(self, num_included_nodes, table_po=None, field_po=None):
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        output = [0]
        include_more_nodes = True
        for table_id in table_po:
            if not include_more_nodes:
                break
            output.append(0)
            if len(output) >= num_included_nodes:
                break
            table = self.get_table(table_id)
            for field_id in field_po[table_id]:
                field = table.fields[field_id]
                output.append(int(field.is_primary_key))
                if len(output) >= num_included_nodes:
                    include_more_nodes = False
                    break
        return output

    def get_foreign_key_ids(self, num_included_nodes, table_po=None, field_po=None):
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        output = [0]
        include_more_nodes = True
        for table_id in table_po:
            if not include_more_nodes:
                break
            output.append(0)
            if len(output) >= num_included_nodes:
                break
            table = self.get_table(table_id)
            for field_id in field_po[table_id]:
                field = table.fields[field_id]
                output.append(int(field.is_foreign_key))
                if len(output) >= num_included_nodes:
                    include_more_nodes = False
                    break
        return output

    def get_field_type_ids(self, num_included_nodes, table_po=None, field_po=None):
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        output = [field_types.to_idx('not_a_field')]
        include_more_nodes = True
        for table_id in table_po:
            if not include_more_nodes:
                break
            output.append(field_types.to_idx('not_a_field'))
            if len(output) >= num_included_nodes:
                break
            table = self.get_table(table_id)
            for field_id in field_po[table_id]:
                field = table.fields[field_id]
                output.append(field_types.to_idx(field.data_type))
                if len(output) >= num_included_nodes:
                    include_more_nodes = False
                    break
        return output

    def get_table_masks(self, num_included_nodes, table_po=None, field_po=None):
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        output = [0]
        for table_id in table_po:
            output.append(1)
            if len(output) >= num_included_nodes:
                break
            table = self.get_table(table_id)
            for _ in table.fields:
                output.append(0)
                if len(output) >= num_included_nodes:
                    break
            if len(output) >= num_included_nodes:
                break
        return output

    def get_table_scopes(self, num_include_nodes, table_po=None, field_po=None):
        """
        :return pos: [t1_pos_in_serial, t2_pos_in_serial, ...]
        :return field_scopes:
            [
                [f11_pis, f12_pis, ...],
                [f21_pis, f22_pis, ...],
                ...
            ]
        """
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        pos, field_scopes = [], []
        num_scanned_nodes = 0
        for table_id in table_po:
            num_scanned_nodes += 1
            pos.append(num_scanned_nodes)
            if num_scanned_nodes >= num_include_nodes:
                break
            table = self.get_table(table_id)
            field_scope = []
            for _ in table.fields:
                num_scanned_nodes += 1
                field_scope.append(num_scanned_nodes)
                if num_scanned_nodes >= num_include_nodes:
                    break
            field_scopes.append(field_scope)
            if num_scanned_nodes >= num_include_nodes:
                break
        return pos, field_scopes

    def get_field_table_pos(self, num_included_nodes, table_po=None, field_po=None):
        """
        :return output: [0, t1_pis, t1_pis, ..., 0, t2_pis, t2_pis, ...]
        """
        if table_po is None:
            table_po, field_po = self.get_schema_perceived_order()
        output = [0]
        for table_id in table_po:
            output.append(0)
            table_pos = len(output) - 1
            if len(output) >= num_included_nodes:
                break
            table = self.get_table(table_id)
            for _ in table.fields:
                output.append(table_pos)
                if len(output) >= num_included_nodes:
                    break
            if len(output) >= num_included_nodes:
                break
        return output

    # --- Lexical features --- #

    def get_lexical_vocab(self):
        """
        Compute the constant vocabulary in the schema.
        """
        vocab = Vocabulary('schema-{}'.format(self.name))
        for t_s in self.table_index:
            table_id = self.table_index[t_s]
            table_node = self.table_rev_index[table_id]
            for token in table_node.lexical_features:
                vocab.index_token(token, in_vocab=True)
        for f_s in self.field_index:
            field_id = self.field_index[f_s]
            field_node = self.field_rev_index[field_id]
            for token in field_node.lexical_features:
                vocab.index_token(token, in_vocab=True)
        return vocab

    def lexicalize_graph(self, tokenize=None, normalized=False):
        for i in self.table_rev_index:
            self.table_rev_index[i].compute_lexical_features(tokenize=tokenize, normalized=normalized)
        for i in self.field_rev_index:
            self.field_rev_index[i].compute_lexical_features(tokenize=tokenize, normalized=normalized)
        self.lexicalized = True

    # --- DB access --- #

    def compute_field_picklist(self):
        for field_id in self.field_rev_index:
            self.get_field_picklist(field_id)

    def get_field_picklist(self, field_id):
        if field_id not in self.picklists:
            field_node = self.get_field(field_id)
            field_name = field_node.name
            table_name = field_node.table.name
            fetch_sql = 'SELECT `{}` FROM `{}`'.format(field_name, table_name)
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = bytes
            c = conn.cursor()
            c.execute(fetch_sql)
            picklist = set()
            for x in c.fetchall():
                if isinstance(x[0], str):
                    picklist.add(x[0].encode('utf-8'))
                elif isinstance(x[0], bytes):
                    try:
                        picklist.add(x[0].decode('utf-8'))
                    except UnicodeDecodeError:
                        picklist.add(x[0].decode('latin-1'))
                else:
                    picklist.add(x[0])
            self.picklists[field_id] = list(picklist)
            conn.close()
        return self.picklists[field_id]

    def get_row(self, table_id, row_id=None, mask_fill=None):
        table_node = self.get_table(table_id)
        table_name = table_node.name
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = bytes
        c = conn.cursor()
        if row_id is None:
            # return a random row
            c.execute('SELECT * from {} ORDER BY RANDOM() LIMIT 1'.format(table_name))
        else:
            # return ith row
            c.execute('SELECT * from {} LIMIT 1 OFFSET {}'.format(table_name, row_id))
        row = c.fetchall()
        conn.close()
        if row:
            if mask_fill:
                def replace_empty_with_mask(x):
                    if x:
                        return x
                    else:
                        return mask_fill

                return [replace_empty_with_mask(x) for x in row[0]]
            else:
                return row[0]
        else:
            return None

    def num_rows(self, table_id):
        table_node = self.get_table(table_id)
        if table_node.num_rows is None:
            table_name = table_node.name
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = bytes
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM {}'.format(table_name))
            table_node.num_rows = c.fetchall()[0][0]
            conn.close()
        return table_node.num_rows

    # --- Loaders --- #

    def load_data_from_spider_json(self, in_json):
        """
        Load graph data from json object (as released in Spider by Yu et. al. 2018) and create adjacency matrix.
        """
        def index_fields(table_fields):
            # order primary keys in front of other keys
            for key in sorted(table_fields.keys()):
                for field_node, o_field_id in table_fields[key]:
                    field_id, _ = self.index_field(field_node)
                    self.official_to_field_id[o_field_id] = field_id
                    self.field_id_to_official[field_id] = o_field_id

        table_names = in_json['table_names_original']
        field_names = in_json['column_names_original']
        table_normalized_names = in_json['table_names']
        field_normalized_names = in_json['column_names']
        field_types = in_json['column_types']
        assert (len(field_names) == len(field_types))
        primary_keys = set(in_json['primary_keys'])
        if in_json['foreign_keys']:
            foreign_keys = set(functools.reduce(lambda x,y: x+y, [list(t) for t in in_json['foreign_keys']]))
        else:
            foreign_keys = []

        for _, (table_name, table_normalized_name) in enumerate(zip(table_names, table_normalized_names)):
            table_node = Table(table_name, table_normalized_name, caseless=self.caseless)
            self.index_table(table_node)

        last_table_id = -1
        table_fields = collections.defaultdict(list)
        for i, ((table_id, field_name), (table_id2, field_normalized_name)) in \
                enumerate(zip(field_names, field_normalized_names)):
            if table_id >= 0:
                if table_id != last_table_id:
                    # new table starts
                    if table_fields:
                        index_fields(table_fields)
                    table_fields = collections.defaultdict(list)
                table_node = self.table_rev_index[table_id]
                data_type = field_types[i]
                is_primary_key = (i in primary_keys)
                is_foreign_key = (i in foreign_keys)
                field_node = Field(table_node, field_name, field_normalized_name,
                                   caseless=self.caseless,
                                   data_type=data_type,
                                   is_primary_key=is_primary_key,
                                   is_foreign_key=is_foreign_key)
                o_field_id = i
                order_token = 0 if is_primary_key else 1
                table_fields[order_token].append((field_node, o_field_id))
                last_table_id = table_id
        if table_fields:
            index_fields(table_fields)
        self.foreign_key_pairs = [(self.official_to_field_id[x], self.official_to_field_id[y])
                                  for x, y in in_json['foreign_keys']]
        self.index_foreign_keys()
        self.create_adjacency_matrix()

    def index_foreign_keys(self):
        for x, y in self.foreign_key_pairs:
            if x not in self.foreign_key_index:
                self.foreign_key_index[x] = []
            if y not in self.foreign_key_index:
                self.foreign_key_index[y] = []
            self.foreign_key_index[x].append(y)
            self.foreign_key_index[y].append(x)

    def load_data_from_finegan_dollak_csv_file(self, in_csv):
        """
        Load graph data from .csv file (as released by Finegan-Dollak et. al. 2018) and create adjacency matrix.
        """
        with open(in_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            current_table = None
            for row in reader:
                if row['Table Name'] == '-':
                    continue
                table_name = row['Table Name'].strip().upper()
                field_name = row[' Field Name'].strip().upper()
                is_primary_key = row[' Is Primary Key'].strip().lower() in ['y', 'yes', 'pri']
                is_foreign_key = row[' Is Foreign Key'].strip().lower() in ['y', 'yes']
                data_type = row[' Type'].strip().lower()
                if current_table is None or table_name != current_table.name:
                    table_node = Table(table_name, caseless=self.caseless)
                    self.index_table(table_node)
                    current_table = table_node
                field_node = Field(current_table, field_name,
                                   caseless=self.caseless,
                                   data_type=data_type,
                                   is_primary_key=is_primary_key,
                                   is_foreign_key=is_foreign_key)
                self.index_field(field_node)

        self.create_adjacency_matrix()

    def load_data_from_csv_file(self, in_csv, field_type_file=None):
        data_types = None
        if field_type_file is not None:
            with open(field_type_file) as f:
                data_types = [x.strip() for x in f.readlines()]
        with open(in_csv) as f:
            reader = csv.reader(f)
            table_header = next(reader)
            table_header = [x.replace(' (SF LISTINGS.csv)', '') for x in table_header]
            assert(data_types is None or len(table_header) == len(data_types))
            table_name = self.name
            table_node = Table(table_name, ' '.join(table_name.lower().split('_')), caseless=self.caseless)
            self.index_table(table_node)
            if data_types:
                for field_name, data_type in zip(table_header, data_types):
                    field_node = Field(table_node, field_name, ' '.join(field_name.split('_')),
                                       caseless=self.caseless, data_type=data_type)
                    self.index_field(field_node)
            else:
                for field_name in table_header:
                    field_node = Field(table_node, field_name, ' '.join(field_name.split('_')),
                                       caseless=self.caseless)
                    self.index_field(field_node)
        self.create_adjacency_matrix()

    def load_data_from_sqlalchemy_metadata(self, metadata):
        foreign_keys = []
        for table in metadata:
            table_name = table.name
            table_normalized_name = get_normalized_name(table.name)
            table_node = Table(table_name, table_normalized_name, caseless=self.caseless)
            self.index_table(table_node)

            table_fields = collections.defaultdict(list)
            for column in table.columns:
                field_name = column.name
                field_normalized_name = get_normalized_name(field_name)
                is_primary_key = column.primary_key
                if column.foreign_keys:
                    is_foreign_key = True
                    foreign_keys.extend([foreign_key for foreign_key in column.foreign_keys])
                else:
                    is_foreign_key = False
                data_type = sqlalchemy_type_to_string(column.type)
                field_node = Field(table_node, field_name, field_normalized_name,
                                   caseless=self.caseless,
                                   data_type=data_type,
                                   is_primary_key=is_primary_key,
                                   is_foreign_key=is_foreign_key)
                order_token = 0 if is_primary_key else 1
                table_fields[order_token].append(field_node)
            # order primary keys in front of other keys
            if table_fields:
                for key in sorted(table_fields.keys()):
                    for field_node in table_fields[key]:
                        field_id, _ = self.index_field(field_node)
        self.foreign_key_pairs = []
        for foreign_key in foreign_keys:
            f1 = foreign_key.column
            f2 = foreign_key.parent
            f1_signature = '{}.{}'.format(f1.table.name, f1.name)
            f2_signature = '{}.{}'.format(f2.table.name, f2.name)
            # print(f1_signature, f2_signature)
            self.foreign_key_pairs.append((self.get_field_id(f1_signature), self.get_field_id(f2_signature)))
        self.index_foreign_keys()
        self.create_adjacency_matrix()

    def load_data_from_2d_array(self, array):
        """
        A sample array:
        [
            ['name', 'age', 'gender'],
            ['John', 18, 'male'],
            ['Kate', 19, 'female']
        ]
        """
        table_name = self.name
        table_node = Table(table_name, None, self.caseless)
        self.index_table(table_node)
        table_header = array[0]
        for i, field_name in enumerate(table_header):
            is_primary_key = (i == 0)
            is_foreign_key = False
            data_type = 'text'
            field_node = Field(table_node, field_name, caseless=self.caseless, data_type=data_type,
                               is_primary_key=is_primary_key, is_foreign_key=is_foreign_key)
            self.index_field(field_node)

        self.create_adjacency_matrix()

    def indexed_table(self, signature):
        return utils.to_indexable(signature, self.caseless) in self.table_index

    def indexed_field(self, signature):
        return utils.to_indexable(signature, self.caseless) in self.field_index

    def index_table(self, node):
        t_id = len(self.table_index)
        n_id = len(self.node_index)
        self.table_index[node.indexable_signature] = t_id
        self.table_rev_index[t_id] = node
        self.node_index[node.indexable_signature] = n_id
        self.node_rev_index[n_id] = node
        self.table_names.add(node.indexable_name)
        self.printable[self.name][node.printable_name] = collections.OrderedDict()
        return t_id, n_id

    def index_field(self, node):
        f_id = len(self.field_index)
        n_id = len(self.node_index)
        self.field_index[node.indexable_signature] = f_id
        self.field_rev_index[f_id] = node
        self.node_index[node.indexable_signature] = n_id
        self.node_rev_index[n_id] = node
        self.field_names.add(node.indexable_name)
        self.printable[self.name][node.table.printable_name][node.printable_name] = {}
        return f_id, n_id

    def create_adjacency_matrix(self):

        def get_current_schema_layout(tables=None, return_dict=False):
            """
            Return the field ids of all fields in all tables as list of lists.
            """
            field_ids = collections.OrderedDict()
            if tables is None:
                tables = range(self.num_tables)

            for table_id in tables:
                table = self.get_table(table_id)
                field_ids[table_id] = [self.get_field_id(field.indexable_signature) for field in table.fields]

            if return_dict:
                return field_ids
            else:
                return [field_ids[i] for i in field_ids.keys()]

        self.bert_feature_idx = dict()
        self.bert_feature_idx_rev = dict()
        table_field_ids = get_current_schema_layout(return_dict=True)
        schema_pos = 1
        for table_id in range(self.num_tables):
            table_node = self.table_rev_index[table_id]
            self.bert_feature_idx[table_node.indexable_signature] = schema_pos
            self.bert_feature_idx_rev[schema_pos] = table_node
            schema_pos += 1
            for field_id in table_field_ids[table_id]:
                field_node = self.field_rev_index[field_id]
                self.bert_feature_idx[field_node.indexable_signature] = schema_pos
                self.bert_feature_idx_rev[schema_pos] = field_node
                schema_pos += 1

        M = ssp.lil_matrix((self.num_nodes + 1, self.num_nodes + 1), dtype=np.int)
        table_field_ids = [table_field_ids[i] for i in table_field_ids.keys()]
        for t_idx in range(len(table_field_ids)):
            t_schema_pos = self.get_schema_pos(self.table_rev_index[t_idx].signature)
            for f_idx in table_field_ids[t_idx]:
                f_node = self.field_rev_index[f_idx]
                f_schema_pos = self.get_schema_pos(f_node.signature)
                if f_node.is_primary_key:
                    M[t_schema_pos, f_schema_pos] = TABLE_FIELD_PRI
                    M[f_schema_pos, t_schema_pos] = FIELD_TABLE_PRI
                else:
                    M[t_schema_pos, f_schema_pos] = TABLE_FIELD_REF
                    M[f_schema_pos, t_schema_pos] = FIELD_TABLE_REF
                for f2_idx in table_field_ids[t_idx]:
                    if f2_idx != f_idx:
                        f2_node = self.field_rev_index[f2_idx]
                        f2_schema_pos = self.get_schema_pos(f2_node.signature)
                        # try:
                        M[f_schema_pos, f2_schema_pos] = SAME_TABLE
                        M[f2_schema_pos, f_schema_pos] = SAME_TABLE
                        # except Exception:
                        #     print(f_schema_pos, f2_schema_pos)
                        #     import pdb
                        #     pdb.set_trace()

        for f1_idx, f2_idx in self.foreign_key_pairs:
            f1_node = self.field_rev_index[f1_idx]
            f2_node = self.field_rev_index[f2_idx]
            t1_node = f1_node.table
            t2_node = f2_node.table
            f1_schema_pos = self.get_schema_pos(f1_node.signature)
            f2_schema_pos = self.get_schema_pos(f2_node.signature)
            t1_schema_pos = self.get_schema_pos(t1_node.signature)
            t2_schema_pos = self.get_schema_pos(t2_node.signature)
            if f1_node.is_primary_key:
                M[f1_schema_pos, f2_schema_pos] = FOREIGN_PRI
                M[f2_schema_pos, f1_schema_pos] = FOREIGN_FOR
                if M[t1_schema_pos, t2_schema_pos] == FOREIGN_TAB_R:
                    M[t1_schema_pos, t2_schema_pos] = FOREIGN_TAB_B
                    M[t2_schema_pos, t1_schema_pos] = FOREIGN_TAB_B
                else:
                    M[t1_schema_pos, t2_schema_pos] = FOREIGN_TAB_F
                    M[t2_schema_pos, t1_schema_pos] = FOREIGN_TAB_R
            else:
                M[f1_schema_pos, f2_schema_pos] = FOREIGN_FOR
                M[f2_schema_pos, f1_schema_pos] = FOREIGN_PRI
                if M[t1_schema_pos, t2_schema_pos] == FOREIGN_TAB_F:
                    M[t1_schema_pos, t2_schema_pos] = FOREIGN_TAB_B
                    M[t2_schema_pos, t1_schema_pos] = FOREIGN_TAB_B
                else:
                    M[t1_schema_pos, t2_schema_pos] = FOREIGN_TAB_R
                    M[t2_schema_pos, t1_schema_pos] = FOREIGN_TAB_F

        for i in range(self.num_nodes + 1):
            M[i, i] = SELF
        self.adj_matrix = M

    def get_adj_matrix(self, tables=None):
        if tables is None:
            return self.adj_matrix

    def pretty_print(self):
        tr = LeftAligned()
        print(tr(self.printable))

    @property
    def base_name(self):
        return self.name.split('-')[0]

    @property
    def num_tables(self):
        return len(self.table_index)

    @property
    def num_fields(self):
        return len(self.field_index)

    @property
    def num_nodes(self):
        return len(self.node_index)

    def get_num_perceived_nodes(self, tables=None):
        if tables is None:
            return self.num_nodes
        if not tables:
            return 0
        return len(tables) + sum([self.get_table(table_id).num_fields for table_id in tables])

    @deprecated
    def index_foreign_keys_between_tables(self):
        for x, y in self.foreign_key_pairs:
            t_id1 = self.get_table_id(self.get_field(x).table.name)
            t_id2 = self.get_table_id(self.get_field(y).table.name)
            key1 = (t_id1, t_id2)
            key2 = (t_id2, t_id1)
            if key1 not in self.foreign_key_index:
                self.foreign_key_index[key1] = []
            if key2 not in self.foreign_key_index:
                self.foreign_key_index[key2] = []
            self.foreign_key_index[key1].append((x, y))
            self.foreign_key_index[key2].append((y, x))


class WikiSQLSchemaGraph(SchemaGraph):
    """
    Table Representation for WikiSQL dataset.
    """
    def __init__(self, name, table, caseless=False):
        super().__init__(name, caseless=caseless)
        self.table = table

    def compute_field_picklist(self, table):
        for field_id in self.field_rev_index:
            self.get_field_picklist(field_id)

    def get_field_picklist(self, field_id):
        if field_id not in self.picklists:
            picklist = set()
            for row in self.table['rows']:
                picklist.add(row[field_id])
            self.picklists[field_id] = list(picklist)
        return self.picklists[field_id]

    def load_data_from_wikisql_json(self, in_json):
        """
        Load graph data from json object (as release in WikiSQL by Zhong et. al. 2017) and create adjacency list.
        """

        def get_table_name(json):
            if 'caption' in json and len(json['caption']) > 0:
                table_name = utils.remove_parentheses_str(json['caption'])
            elif 'section_title' in json and len(json['section_title']) > 0:
                table_name = utils.remove_parentheses_str(json['section_title'])
            elif 'page_title' in json and len(json['page_title']) > 0:
                table_name = utils.remove_parentheses_str(json['page_title'])
            else:
                table_name = utils.remove_parentheses_str(json['header'][0])
            if not table_name:
                table_name = 'table'
            # escape "'"
            table_name = table_name.replace('\'', '\'\'')
            return table_name

        table_name = get_table_name(in_json)
        table_node = Table(table_name, caseless=self.caseless)
        self.index_table(table_node)

        for i, (field_name, field_type) in enumerate(zip(in_json['header'], in_json['types'])):
            field_name = ' '.join(field_name.split())
            # escape "'"
            field_name = field_name.replace('\'', '\'\'')
            if field_type == 'real':
                field_type = 'number'
            field_node = Field(table_node, field_name, caseless=self.caseless, data_type=field_type)
            self.index_field(field_node)

        self.create_adjacency_matrix()


def get_normalized_name(s):
    return s.lower().replace('_', ' ')


def sqlalchemy_type_to_string(t):
    if isinstance(t, sqlalchemy.Text):
        return 'text'
    elif isinstance(t, sqlalchemy.Integer) or isinstance(t, sqlalchemy.Float):
        return 'number'
    elif isinstance(t, sqlalchemy.DateTime):
        return 'time'
    elif isinstance(t, sqlalchemy.Boolean):
        return 'boolean'
    else:
        return 'others'


