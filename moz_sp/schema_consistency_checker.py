"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import collections

from mo_future import string_types, text

from moz_sp.debugs import debug_wrapper
from moz_sp.keywords import join_keywords
from moz_sp.sql_normalizer import SchemaGroundedTraverser
import moz_sp.utils as utils


def Operator(op):
    def func(self, json):
        for v in json:
            if isinstance(v, dict) and utils.is_subquery(v):
                self.dispatch(v)
            else:
                self.dispatch(v)
    return func


class SchemaConsistencyChecker(SchemaGroundedTraverser):

    clauses = [
        'from_',
        'where',
        'groupby',
        'having',
        'select',
        'orderby',
        'limit',
        'offset'
    ]

    # simple operators
    _concat = Operator('||')
    _mul = Operator('*')
    _div = Operator('/')
    _add = Operator('+')
    _sub = Operator('-')
    _neq = Operator('!=')
    _gt = Operator('>')
    _lt = Operator('<')
    _gte = Operator('>=')
    _lte = Operator('<=')
    _eq = Operator('=')
    _or = Operator('OR')
    _and = Operator('AND')

    def __init__(self, schema, verbose=False):
        super().__init__(schema, verbose)
        self.table_coverage = None

    @debug_wrapper
    def check_table(self, s):
        if not self.is_table(s):
            raise ValueError('ERROR: "{}" appeared in FROM clause and is not a table'.format(s))

    @debug_wrapper
    def check_field(self, s):
        if not '.' in s:
            return
        table_name, field_name = s.split('.')
        if not self.is_table(table_name):
            raise ValueError('ERROR: "{}" appeared in field expression {} is not a table'.format(table_name, s))
        if table_name not in self.table2alias:
            raise ValueError('ERROR: table "{}" in field expression {} is not in the current scope'.format(
                table_name, s))
        self.table_coverage[table_name] = True

    @debug_wrapper
    def check_numeric_field(self, key, s):
        if not self.is_numeric_field(s):
            raise ValueError('ERROR: "{}" attached to non-numeric field {}'.format(key, s))

    @debug_wrapper
    def dispatch(self, json, is_table=False):
        if isinstance(json, list):
            return self.delimited_list(json)

        if isinstance(json, dict):
            if len(json) == 0:
                return
            elif 'value' in json:
                return self.value(json)
            elif 'from' in json:
                # Nested query 'from'
                self.check(json)
            elif 'query' in json:
                # Nested query 'query'
                self.check(json['query'])
            elif 'union' in json:
                # Nested query 'union'
                self.union(json['union'])
            elif 'intersect' in json:
                self.intersect(json['intersect'])
            elif 'except' in json:
                self.except_(json['except'])
            else:
                return self.op(json)
        if isinstance(json, string_types):
            if json == '*':
                for table in self.table_coverage:
                    self.table_coverage[table] = True
            if self.is_field(json):
                self.check_field(json)

    @debug_wrapper
    def check(self, json):
        if 'union' in json:
            self.union(json['union'])
        elif 'intersect' in json:
            self.intersect(json['intersect'])
        elif 'except' in json:
            self.except_(json['except'])
        else:
            self.query(json)

    @debug_wrapper
    def union(self, json):
        for query in json:
            self.check(query)

    @debug_wrapper
    def intersect(self, json):
        for query in json:
            self.check(query)

    @debug_wrapper
    def except_(self, json):
        for query in json:
            self.check(query)

    @debug_wrapper
    def get_alias_table_map(self, json):
        alias2table, table2alias, table_coverage  = dict(), collections.defaultdict(list), dict()
        assert ('from' in json)
        from_ = json['from']

        if not isinstance(from_, list):
            from_ = [from_]

        for item in from_:
            if isinstance(item, dict) and ('name' in item or 'value' in item):
                self.process_table_item(item, alias2table, table2alias)
            elif 'join' in item:
                join_item = item['join']
                if isinstance(join_item, string_types):
                    table_name = join_item
                    table2alias[table_name].append(None)
                else:
                    self.process_table_item(join_item, alias2table, table2alias)
            elif isinstance(item, string_types):
                table_name = item
                table2alias[table_name].append(None)
            else:
                raise ValueError

        for table in table2alias:
            table_coverage[table] = False
        self.table_alias_stack.append((alias2table, table2alias, table_coverage))
        self.alias2table, self.table2alias, self.table_coverage = self.table_alias_stack[-1]

    @debug_wrapper
    def pop_table_alias_stack(self):
        self.table_alias_stack.pop()
        if self.table_alias_stack:
            self.alias2table, self.table2alias, self.table_coverage = self.table_alias_stack[-1]
        else:
            self.alias2table, self.table2alias, self.table_coverage = None, None, None

    @debug_wrapper
    def query(self, json):
        self.get_alias_table_map(json)
        for clause in self.clauses:
            getattr(self, clause)(json)
        # TODO: table coverage check is not a valid heuristic
        # for table in self.table_coverage:
        #     if not self.table_coverage[table]:
        #         raise ValueError('ERROR: table "{}" does not appear in any SQL clause'.format(table))
        self.pop_table_alias_stack()

    @debug_wrapper
    def from_(self, json):

        def extract_join_conds(json):
            join_conds = []
            for key in json:
                if key == 'eq':
                    join_conds.append(json[key])
                else:
                    assert(isinstance(json[key], list))
                    for item in json[key]:
                        join_conds.extend(extract_join_conds(item))
            return join_conds

        if 'from' in json:
            from_ = json['from']
            if isinstance(from_, dict):
                return self.dispatch(from_)
            if not isinstance(from_, list):
                from_ = [from_]
            table_list = []
            join_cond_dict = collections.defaultdict(list)
            for token in from_:
                join_keyword = None
                for join_kw in join_keywords:
                    if join_kw in token:
                        join_keyword = join_kw
                        break
                if join_keyword:
                    table_name = token[join_keyword]
                    self.check_table(table_name)
                    table_list.append(table_name)
                    join_conds = extract_join_conds(token['on'])
                    for join_cond in join_conds:
                        if '.' not in join_cond[0]:
                            raise ValueError('ERROR: invalid JOIN field "{}": {}'.format(join_cond[0], from_))
                        if '.' not in join_cond[1]:
                            raise ValueError('ERROR: invalid JOIN field "{}": {}'.format(join_cond[1], from_))
                        table1, field1 = join_cond[0].split('.')
                        table2, field2 = join_cond[1].split('.')
                        if table1 not in table_list or table2 not in table_list:
                            raise ValueError('ERROR: Join condition does not match with table: {}'.format(from_))
                        if table_name not in [table1, table2]:
                            raise ValueError('ERROR: Join condition does not match with table: {}'.format(from_))
                        if len(table_list) == 2 and table_list[0] not in [table1, table2]:
                            raise ValueError('ERROR: Join condition does not match with table: {}'.format(from_))
                        join_cond_dict[table1].append(table2)
                        join_cond_dict[table2].append(table1)
                else:
                    if isinstance(token, str):
                        table_name = token
                        self.check_table(table_name)
                        table_list.append(table_name)
            for table in join_cond_dict:
                if len(join_cond_dict[table]) > 1:
                    self.table_coverage[table] = True

    @debug_wrapper
    def where(self, json):
        if 'where' in json:
            self.dispatch(json['where'])

    @debug_wrapper
    def groupby(self, json):
        if 'groupby' in json:
            self.dispatch(json['groupby'])
            if isinstance(json['groupby'], list):
                unique_group_by = []
                for x in json['groupby']:
                    if not x in unique_group_by:
                        unique_group_by.append(x)
                json['groupby'] = unique_group_by

    @debug_wrapper
    def having(self, json):
        if 'having' in json:
            self.dispatch(json['having'])

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            self.dispatch(json['select'])
            if isinstance(json['select'], list):
                unique_select = []
                for x in json['select']:
                    if not x in unique_select:
                        unique_select.append(x)
                json['select'] = unique_select

    @debug_wrapper
    def orderby(self, json):
        if 'orderby' in json:
            orderby = json['orderby']
            if isinstance(orderby, dict):
                orderby = [orderby]
            for o in orderby:
                self.dispatch(o)

    @debug_wrapper
    def limit(self, json):
        if 'limit' in json:
            return

    @debug_wrapper
    def offset(self, json):
        if 'offset' in json:
            return

    @debug_wrapper
    def delimited_list(self, json):
        for element in json:
            self.dispatch(element)

    @debug_wrapper
    def value(self, json):
        return self.dispatch(json['value'], is_table=('is_table' in json))

    @debug_wrapper
    def op(self, json):
        if 'on' in json:
            return self._on(json)

        if len(json) > 1:
            raise Exception('Operators should have only one key!')
        key, value = list(json.items())[0]

        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        attr = '_{0}'.format(key)
        if hasattr(self, attr) and not key.startswith('_'):
            method = getattr(self, attr)
            return method(value)

        # treat as regular function call
        if isinstance(value, dict) and len(value) == 0:
            return True  # NOT SURE IF AN EMPTY dict SHOULD BE DELT WITH HERE, OR IN self.dispatch()
        else:
            # TODO: aggregation operator consistency check is not being done now because ground truth data type
            #  contains errors
            # if key in ['avg', 'sum'] and isinstance(value, string_types):
            #     self.check_numeric_field(key, value)
            return self.dispatch(value)

    @debug_wrapper
    def _exists(self, value):
        return self.dispatch(value)

    @debug_wrapper
    def _missing(self, value):
        return self.dispatch(value)

    @debug_wrapper
    def _between(self, triple):
        self.dispatch(triple[0])
        self.dispatch(triple[1])
        self.dispatch(triple[2])

    @debug_wrapper
    def _like(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _nlike(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _is(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _in(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _nin(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _case(self, checks):
        for check in checks:
            if isinstance(check, dict):
                self.dispatch(check['when'])
                self.dispatch(check['then'])
            else:
                self.dispatch(check)

    @debug_wrapper
    def _literal(self, json):
        if isinstance(json, list):
            for v in json:
                self._literal(v)

    @debug_wrapper
    def _on(self, json):
        for join_key in join_keywords:
            if join_key in json:
                self.dispatch(json[join_key], is_table=isinstance(json[join_key], string_types))
                self.dispatch(json['on'])
