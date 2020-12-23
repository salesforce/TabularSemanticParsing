# encoding: utf-8

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Traverse a SQL AST and extract condition expressions from it.
"""
from mo_future import string_types
from moz_sp.debugs import debug_wrapper
from moz_sp.keywords import join_keywords
from moz_sp.sql_parser import DEBUG
from moz_sp.traverser import SchemaGroundedTraverser
from src.utils.utils import is_number


class ValueExtractor(SchemaGroundedTraverser):

    def Operator(op):
        def func(self, json):
            if op in ['<>', '>', '<', '>=', '<=', '=', '!='] and \
                    isinstance(json[0], string_types) and \
                    (isinstance(json[1], string_types) or (isinstance(json[1], dict) and 'literal' in json[1])):
                assert (len(json) == 2 and isinstance(json, list))
                v1, v2 = json
                if isinstance(v2, dict):
                    v2 = v2['literal']
                if is_number(v2):
                    return
                if v1 != v2:
                    if self.is_field(v1) and not self.is_field(v2):
                        v1_id = self.schema.get_field_id(v1)
                        v1 = self.schema.get_field_signature(v1_id)
                        self.values.append((v1, v2))
            else:
                for v in json:
                    self.dispatch(v)

        return func

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
        self.values = []

    @debug_wrapper
    def extract(self, json):
        self.root(json)

    @debug_wrapper
    def delimited_list(self, json):
        for element in json:
            self.dispatch(element)

    @debug_wrapper
    def dispatch(self, json):
        if isinstance(json, list):
            self.delimited_list(json)
        if isinstance(json, dict):
            if len(json) == 0:
                return
            elif 'value' in json:
                self.value(json)
            elif 'from' in json:
                # Nested query 'from'
                self.extract(json)
            elif 'query' in json:
                # Nested queries 'query'
                self.extract(json['query'])
            elif 'union' in json:
                # Nested queries 'union'
                self.union(json['union'])
            elif 'intersect' in json:
                return self.intersect(json['intersect'])
            elif 'except' in json:
                return self.except_(json['except'])
            else:
                self.op(json)

    @debug_wrapper
    def from_(self, json):
        if 'from' in json:
            from_ = json['from']

            if isinstance(from_, dict):
                return self.dispatch(from_)

            if not isinstance(from_, list):
                from_ = [from_]
            for token in from_:
                self.dispatch(token)

    @debug_wrapper
    def groupby(self, json):
        if 'groupby' in json:
            self.dispatch(json['groupby'])

    @debug_wrapper
    def having(self, json):
        if 'having' in json:
            self.dispatch(json['having'])

    @debug_wrapper
    def limit(self, json):
        if 'limit' in json:
            self.dispatch(json['limit'])

    @debug_wrapper
    def offset(self, json):
        if 'offset' in json:
            self.dispatch(json['offset'])

    @debug_wrapper
    def op(self, json):
        if 'on' in json:
            self._on(json)
            return

        if len(json) > 1:
            raise Exception('Operators should have only one key!')

        key, value = list(json.items())[0]
        if DEBUG:
            print(key)
        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        attr = '_{0}'.format(key)
        if hasattr(self, attr) and not key.startswith('_'):
            getattr(self, attr)(value)
            return

        # treat as regular function call
        if isinstance(value, dict) and len(value) == 0:
            return
        else:
            self.dispatch(value)
            return

    @debug_wrapper
    def orderby(self, json):
        if 'orderby' in json:
            self.dispatch(json['orderby'])

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            self.dispatch(json['select'])

    @debug_wrapper
    def union(self, json):
        for query in json:
            self.extract(query)

    @debug_wrapper
    def intersect(self, json):
        for i, query in enumerate(json):
            self.extract(query)

    @debug_wrapper
    def except_(self, json):
        for i, query in enumerate(json):
            self.extract(query)

    @debug_wrapper
    def query(self, json):
        self.get_alias_table_map(json)
        for clause in self.clauses:
            getattr(self, clause)(json)
        if self.table_alias_stack:
            self.pop_table_alias_stack()

    @debug_wrapper
    def root(self, json):
        if 'union' in json:
            self.union(json['union'])
        elif 'intersect' in json:
            self.intersect(json['intersect'])
        elif 'except' in json:
            self.except_(json['except'])
        else:
            self.query(json)

    @debug_wrapper
    def value(self, json):
        self.dispatch(json['value'])

    @debug_wrapper
    def where(self, json):
        if 'where' in json:
            self.dispatch(json['where'])

    @debug_wrapper
    def _case(self, checks):
        for check in checks:
            if isinstance(check, dict):
                self.dispatch(check['when'])
                self.dispatch(check['then'])
            else:
                self.dispatch(check)

    @debug_wrapper
    def _exists(self, value):
        self.dispatch(value)

    @debug_wrapper
    def _in(self, json):
        self.dispatch(json[1])

    @debug_wrapper
    def _nin(self, json):
        self.dispatch(json[1])

    @debug_wrapper
    def _is(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _like(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _nlike(self, pair):
        self.dispatch(pair[0])
        self.dispatch(pair[1])

    @debug_wrapper
    def _literal(self, json):
        if isinstance(json, list):
            for v in json:
                self._literal(v)

    @debug_wrapper
    def _missing(self, value):
        self.dispatch(value)

    @debug_wrapper
    def _on(self, json):
        for key in join_keywords:
            if key in json:
                self.dispatch(json[key])
                self.dispatch(json['on'])