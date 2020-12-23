"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Traverse a SQL AST and perform common operations.
"""

import collections
import json
import re

from mo_future import string_types
from moz_sp.debugs import debug_wrapper
from moz_sp.keywords import join_keywords
import moz_sp.utils as utils
from src.utils.utils import is_number


class SchemaGroundedTraverser(object):

    compound_sql_ops = {
        'intersect',
        'union',
        'except'
    }

    clauses = [
        'select',
        'from_',
        'where',
        'groupby',
        'having',
        'orderby',
        'limit',
        'offset'
    ]

    def __init__(self, schema, verbose=False):
        self.schema = schema
        self.table_alias_stack = []
        self.alias2table, self.table2alias = None, None
        self.current_clause = None
        self.verbose = verbose
        self.noisy_schema_component_names = False

    @debug_wrapper
    def is_field(self, s):
        if not isinstance(s, string_types):
            return False
        if is_number(s):
            return False
        if re.fullmatch(utils.field_pattern, s):
            table_name, field_name = s.split('.')
            if re.fullmatch(utils.alias_pattern, table_name):
                table_name = self.get_table_name_by_alias(table_name)
            return self.schema.is_table_name(table_name) and self.schema.is_field_name(field_name)
        else:
            return self.schema.is_field_name(s)

    @debug_wrapper
    def is_numeric_field(self, s):
        assert(isinstance(s, string_types))
        if is_number(s):
            return False
        assert(re.fullmatch(utils.field_pattern, s))
        field_id = self.schema.get_field_id(s)
        field_node = self.schema.get_field(field_id)
        return field_node.is_numeric

    @debug_wrapper
    def is_table(self, s):
        if not isinstance(s, string_types):
            return False
        return self.schema.is_table_name(s)

    @debug_wrapper
    def known_table(self, name):
        for ptr in range(len(self.table_alias_stack) - 1, -1, -1):
            table2alias = self.table_alias_stack[ptr][1]
            if name in table2alias or name.lower() in table2alias or name.upper() in table2alias:
                return True
        return False

    @debug_wrapper
    def get_alias_table_map(self, json):
        alias2table, table2alias = dict(), collections.defaultdict(list)
        assert('from' in json)
        from_ = json['from']

        if not isinstance(from_, list):
            from_ = [from_]

        for item in from_:
            if any([co in item for co in self.compound_sql_ops]):
                for co in self.compound_sql_ops:
                    if co in item:
                        break
                # self.dispatch(item[co])
            else:
                if isinstance(item, dict) and ('name' in item or 'value' in item):
                    self.process_table_item(item, alias2table, table2alias)
                elif any([jk in item for jk in join_keywords]):
                    for jk in join_keywords:
                        if jk in item:
                            break
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
                self.table_alias_stack.append((alias2table, table2alias))
                self.alias2table, self.table2alias = self.table_alias_stack[-1]

    @debug_wrapper
    def process_table_item(self, item, alias2table, table2alias):
        assert (isinstance(item, dict))
        if 'value' in item:
            value = item['value']
            table_alias = item.get('name', None)
            if utils.is_subquery(value):
                alias2table[table_alias] = value
            else:
                table_name = value
                item['is_table'] = True
                assert (not table_name in alias2table)
                table2alias[table_name].append(table_alias)
                if table_alias:
                    alias2table[table_alias] = table_name

    @debug_wrapper
    def pop_table_alias_stack(self):
        self.table_alias_stack.pop()
        if self.table_alias_stack:
            self.alias2table, self.table2alias = self.table_alias_stack[-1]
        else:
            self.alias2table, self.table2alias = None, None

    @debug_wrapper
    def get_table_name_by_alias(self, alias):
        for ptr in range(len(self.table_alias_stack) - 1, -1, -1):
            alias2table = self.table_alias_stack[ptr][0]
            if alias in alias2table:
                return alias2table[alias]
            if alias.lower() in alias2table:
                return alias2table[alias.lower()]
            if alias.upper() in alias2table:
                return alias2table[alias.upper()]
        return None

    @debug_wrapper
    def delimited_list(self, json):
        """
        This is a dummy function.
        """
        return

    @debug_wrapper
    def dispatch(self, json):
        """
        This function is a dummy.
        """
        return

    @debug_wrapper
    def op(self, json):
        """
        This is a dummy function.
        """
        return

    @debug_wrapper
    def root(self, json):
        """
        Normalized AST -> denormalized AST.
        Remove all removable aliases.
        """
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
        for i, query in enumerate(json):
            self.root(query)

    @debug_wrapper
    def intersect(self, json):
        for i, query in enumerate(json):
            self.root(query)

    @debug_wrapper
    def except_(self, json):
        for i, query in enumerate(json):
            self.root(query)

    @debug_wrapper
    def query(self, json):
        self.get_alias_table_map(json)
        for clause in self.clauses:
            self.current_clause = clause
            getattr(self, clause)(json)
        if self.table_alias_stack:
            self.pop_table_alias_stack()

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            self.dispatch(json['select'])

    @debug_wrapper
    def from_(self, json):
        """
        This is a dummy function.
        """
        return

    @debug_wrapper
    def value(self, json):
        """
        This function is a dummy.
        """
        return

    @debug_wrapper
    def where(self, json):
        if 'where' in json:
            self.dispatch(json['where'])

    @debug_wrapper
    def groupby(self, json):
        if 'groupby' in json:
            self.dispatch(json['groupby'])

    @debug_wrapper
    def having(self, json):
        if 'having' in json:
            self.dispatch(json['having'])

    @debug_wrapper
    def orderby(self, json):
        if 'orderby' in json:
            self.dispatch(json['orderby'])

    @debug_wrapper
    def limit(self, json):
        if 'limit' in json:
            self.dispatch(json['limit'])

    @debug_wrapper
    def offset(self, json):
        if 'offset' in json:
            self.dispatch(json['offset'])

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
                self.dispatch(json[join_key])
                self.dispatch(json['on'])
                return
        raise AttributeError('Unrecognized JOIN keywords: {}'.format(json))

