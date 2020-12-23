# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

Processing the SQL parse tree structures released by Yu et. al. 2018.

Code adapted from:
https://github.com/taoyds/syntaxSQL/blob/master/preprocess_train_dev_data.py
https://github.com/taoyds/spider/blob/master/preprocess/parse_raw_json.py

  Assumptions:
    1. sql is correct
    2. only table name has alias
    3. only one intersect/union/except

  val: number(float)/string(str)/sql(dict)
  col_unit: (agg_id, col_id, isDistinct(bool))
  val_unit: (unit_op, col_unit1, col_unit2)
  table_unit: (table_type, col_unit/sql)
  cond_unit: (not_op, op_id, val_unit, val1, val2)
  condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
  sql {
    'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
    'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
    'where': condition
    'groupBy': [col_unit1, col_unit2, ...]
    'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
    'having': condition
    'limit': None/limit value
    'intersect': None/sql
    'except': None/sql
    'union': None/sql
"""

import copy
import json
import os
import random
import sys
from src.utils.utils import deprecated


compound_sql_ops = {
    'union': 0,
    'except': 1,
    'intersect': 2
}


# in standard SQL excution order
#   https://www.designcise.com/web/tutorial/what-is-the-order-of-execution-of-an-sql-query
sql_clauses = [
    'from',
    'where',
    'groupBy',
    'having',
    'select',
    'orderBy',
    'limit'
]


def is_query(json):
    return type(json) is dict and 'select' in json


class SQLASTIndexer(object):
    """
    In-place conversion of all components in a SQL parse tree to indices.
    """
    def column_unit(self, json):
        if json:
            json[2] = 1 if json[2] else 0

    def condition_clause(self, json):
        for i in range(len(json)):
            if i % 2 == 0:
                # condition unit
                assert(type(json[i]) is list)
                self.condition_unit(json[i])
            else:
                # condition op
                assert(type(json[i]) is str)
                json[i] = 1 if json[i] == 'OR' else 0

    def condition_unit(self, json):
        json[0] = 1 if json[0] else 0
        self.value_unit(json[2])
        self.value(json[3])
        self.value(json[4])

    def value(self, json):
        if is_query(json):
            self.query(json)
        if type(json) is list and len(json) == 3 and type(json[2]) is bool:
            self.column_unit(json)

    def value_unit(self, json):
        self.column_unit(json[1])
        self.column_unit(json[2])

    def query(self, json):
        assert(is_query(json))
        for key in json:
            # ignore empty sql clauses
            if json[key]:
                clause = json[key]
                if key in compound_sql_ops:
                    self.query(clause)
                if key == 'select':
                    clause[0] = 1 if clause[0] else 0
                    for i in range(len(clause[1])):
                        self.value_unit(clause[1][i][1])
                if key == 'groupBy':
                    for i in range(len(clause)):
                        self.column_unit(clause[i])
                if key == 'orderBy':
                    clause[0] = 1 if clause[0] == 'asc' else 0
                    for i in range(len(clause[1])):
                        self.value_unit(clause[1][i])
                if key == 'from':
                    if 'conds' in clause:
                        self.condition_clause(clause['conds'])
                if key == 'where':
                    self.condition_clause(clause)
                if key == 'having':
                    self.condition_clause(clause)


class SQLVectorizer(object):
    """
    Convert indexed elements in a SQL parse tree to the vector format predicted by the NN model.
    """
    def query(self, json):
        v_query = VQuery()

        v_subquery = VSubQuery()
        # A. Processing current subquery
        for key in sql_clauses:
            # ignore empty sql clauses
            if json[key]:
                if key == 'select':
                    pass

                if key == 'groupBy':
                    pass

                if key == 'orderBy':
                    pass

                if key == 'where':
                    pass

                if key == 'having':
                    pass

                if key == 'limit':
                    pass

                if key == 'from':
                    pass

        v_query.sub_queries.append(v_subquery)

        # B. Processing siblng queries
        queries, compound_ops = [], []
        for key in compound_sql_ops:
            if json[key]:
                compound_ops.append(compound_sql_ops[key])
                queries_, compound_ops_ = self.query(json[key])
                queries.extend(queries_)
                compound_ops.extend(compound_ops_)

        return queries, compound_ops


class VQuery(object):
    """
    Store indexed elements in a SQL query in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.sub_queries = []
        self.compount_ops = []


class VSubQuery(object):
    """
    Store indexed elements in a SQL subquery in the vector format predicted by the NN model.

    Note: a subquery query contains no compound operators including "intersect", "except" and "union".
    """
    def __init__(self):
        self.slt_clause = VSelectClause()
        self.grp_clause = VGroupByClause()
        self.ord_clause = VOrderByClause()
        self.frm_clause = VFromClause()
        self.whr_clause = VWhereClause()
        self.hav_clause = VHavingClause()
        self.lmt_clause = VLimitClause


class VSelectClause(object):
    """
    Store indexed elements in a SELECT clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields = None
        self.field_aggregation_ops = None
        self.field_distincts = None
        self.field_arithmetic_ops = None
        self.expression_aggs = None
        self.distinct = None


class VGroupByClause(object):
    """
    Store indexed elements in a GROUP_BY clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields = None
        self.field_aggregation_ops = None
        self.field_distincts = None
        self.field_arithmetic_ops = None


class VOrderByClause(object):
    """
    Store indexed elements in a ORDER_BY clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields = None
        self.field_aggregation_ops = None
        self.field_distincts = None
        self.field_arithmentic_ops = None
        self.ascs = None


class VWhereClause(object):
    """
    Store indexed elements in a WHERE clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields = None
        self.field_aggregation_ops = None
        self.field_distincts = None
        self.condition_ops = None
        self.condition_vals = None


class VHavingClause(object):
    """
    Store indexed elements in a HAVING clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields = None
        self.field_aggregation_ops = None
        self.field_distincts = None
        self.condition_ops = None
        self.condition_vals = None


class VLimitClause(object):
    """
    Store indexed elements in a LIMIT clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.value = None


class VFromClause(object):
    """
    Store indexed elements in a FROM clause in the vector format predicted by the NN model.
    """
    def __init__(self):
        self.fields_a = None
        self.fields_b = None


class SQLSerializer(object):
    """
    Convert a SQL parse tree to a valid SQL query.
    """
    def __init__(self,
                 schema_graph, field_vocab, aggregation_ops, arithmetic_ops, condition_ops, logical_ops, value_vocab):
        self.schema_graph = schema_graph
        self.field_vocab = field_vocab
        self.aggregation_ops = aggregation_ops
        self.arithmetic_ops = arithmetic_ops
        self.condition_ops = condition_ops
        self.logical_ops = logical_ops
        self.value_vocab = value_vocab

    def serialize_field_expression(self, ast, ops):
        if type(ast) is tuple:
            assert(len(ast) == 3)
            op = ops.to_token(ast[0])
            l_field = self.schema_graph.get_field(ast[1])
            return '{} {} {}'.format(l_field, op, self.serialize_field_expression(ast[2], ops))
        else:
            field_id = ast[0]
            return self.schema_graph.get_field(field_id)

    def serialize_field(self, ast):
        agg, field, distinct = ast
        field_ = '{}({})'.format(self.aggregation_ops.to_token(agg), self.schema_graph.get_field(field))
        if distinct:
            return 'DISTINCT {}'.format(field_)
        else:
            return field_

    def serialize_select(self, ast):
        cla_distinct = ast[0]
        field_expression_list = []
        for vu_agg, field_exp in ast[1]:
            field_expression_list.append('{}({})'.format(
                self.aggregation_ops.to_token(vu_agg),
                self.serialize_field_expression(field_exp, self.arithmetic_ops)))
        field_expression = ', '.join(field_expression_list)
        if cla_distinct:
            return 'SELECT DISTINCT {}'.format(field_expression)
        else:
            return 'SELECT {}'.format(field_expression)

    def serialize_group_by(self, ast):
        field_expression_list = []
        for field_exp in ast:
            field_expression_list.append(self.serialize_field_expression(field_exp, self.arithmetic_ops))
        return 'GROUP BY {}'.format(', '.join(field_expression_list))

    def serialize_order_by(self, ast):
        asc = ast[0].upper()
        field_expression_list = []
        for field_exp in ast[1]:
            field_expression_list.append(self.serialize_field_expression(field_exp, self.arithmetic_ops))
        return 'ORDER BY {} {}'.format(', '.join(field_expression_list), asc)


@deprecated
class FieldExtractor(object):
    """
    Extract the fields appeared in each SQL components.

    TODO: this implementation does not handle
        1. nested SQL queries
        2. field expressions that involve arithmetic calculations, i.e., we convert every val_unit to a col_unit.
    """
    def __init__(self, schema):
        self.schema = schema
        self.slt_fields = []
        self.grp_fields = []
        self.ord_fields = []
        self.whr_fields = []
        self.hav_fields = []
        self.frm_fields = []

    def extract_field_ids(self, sql):
        for key in sql.keys():
            if key == 'groupBy' and sql[key]:
                sql_cols = sql[key]
                for col in sql_cols:
                    self.grp_fields.append(col)
            if key == 'orderBy' and sql[key]:
                sql_vals = sql[key][1]
                for val_unit in sql_vals:
                    # TODO: we assumed that the orderBy expression contains only 1 col_unit
                    self.ord_fields.append(val_unit[1])
            if key == 'select' and sql[key]:
                sql_cols = sql[key][1]
                for agg, val_unit in sql_cols:
                    self.slt_fields.append(val_unit[1])
            if key == 'where' and sql[key]:
                cond = sql[key]
                for i in range(len(cond)):
                    if i % 2 == 0:
                        cond_unit = cond[i]
                        val_unit = cond_unit[2]
                        self.whr_fields.append(val_unit[1])
                        # val1 = cond_unit[3]
                        # val2 = cond_unit[4]
                        # if isinstance(val1, dict):
                        #     self.extract_field_ids(val1)
                        # if isinstance(val2, dict):
                        #     self.extract_field_ids(val2)
            if key == 'having' and sql[key]:
                cond = sql[key]
                for i in range(len(cond)):
                    if i % 2 == 0:
                        cond_unit = cond[i]
                        val_unit = cond_unit[2]
                        self.hav_fields.append(val_unit[1])
                        # val1 = cond_unit[3]
                        # val2 = cond_unit[4]
                        # if isinstance(val1, dict):
                        #     self.extract_field_ids(val1)
                        # if isinstance(val2, dict):
                        #     self.extract_field_ids(val2)
            if key == 'from' and sql[key]:
                for table_unit in sql[key]['table_units']:
                    pass
                    # if isinstance(table_unit, dict):
                    #     self.extract_field_ids(table_unit)
                conds = sql[key]['conds']
                for i in range(len(conds)):
                    if i % 2 == 0:
                        cond_unit = conds[i]
                        val1_unit = cond_unit[2]
                        self.frm_fields.append(val1_unit[1])
                        col2_unit = cond_unit[3]
                        assert(col2_unit)
                        self.frm_fields.append(col2_unit)
            if key in compound_sql_ops and sql[key]:
                # self.extract_field_ids(sql[key])
                pass

        all_fields = self.slt_fields + self.grp_fields + self.ord_fields + self.hav_fields + self.whr_fields + \
                     self.frm_fields

        fields = {
            'select': self.slt_fields,
            'groupBy': self.grp_fields,
            'orderBy': self.ord_fields,
            'where': self.whr_fields,
            'having': self.hav_fields,
            'from': self.frm_fields,
            'all': all_fields
        }
        return fields


#############
# Unit tests
#############

def test_ast_vectorizer():
    in_json = sys.argv[1]
    with open(in_json) as f:
        content = json.load(f)

    ast_indexer = SQLASTIndexer()

    random.shuffle(content)
    for example in content:
        ast = example['sql']
        ast2 = copy.deepcopy(ast)
        print(json.dumps(ast, indent=4))
        print()
        ast_indexer.query(ast2)
        print(json.dumps(ast2, indent=4))
        import pdb
        pdb.set_trace()


def test_field_extractor():
    data_dir = sys.argv[1]
    in_table_json = os.path.join(data_dir, 'tables.json')
    db_index = dict()
    with open(in_table_json) as f:
        content = json.load(f)
        for db in content:
            db_index[db['db_id']] = db
    in_data_json = os.path.join(data_dir, 'train.json')
    sql_asts = dict()
    with open(in_data_json) as f:
        content = json.load(f)
        for example in content:
            sql_asts[example['query']] = example['sql']

    in_sql = sys.argv[2]
    with open(in_sql) as f:
        sql = f.read().strip()
    ast = sql_asts[sql]

    db_id = sys.argv[3]
    tables = db_index[db_id]

    f_extractor = FieldExtractor(tables)
    fields = f_extractor.extract_field_ids(ast)
    print(ast)
    print(fields)


if __name__ == '__main__':
    test_ast_vectorizer()
