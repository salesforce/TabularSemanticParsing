"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Normalizing a SQL AST by adding aliases; denormalizing by removing all removbale aliases.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import collections
from mo_future import string_types
from moz_sp.debugs import debug_wrapper
from moz_sp.traverser import SchemaGroundedTraverser
import moz_sp.utils as utils


class Denormalizer(SchemaGroundedTraverser):
    """
    Perform in-place denormalization of a SQL AST by removing all removable aliases.
    """

    def __init__(self, schema, verbose=False):
        super(Denormalizer, self).__init__(schema, verbose)
        self.contains_self_join = False

    @debug_wrapper
    def denormalize(self, json):
        self.root(json)

    @debug_wrapper
    def dispatch(self, json):
        if isinstance(json, list):
            self.delimited_list(json)
        elif isinstance(json, dict):
            if len(json) > 0:
                if 'value' in json:
                    self.value(json)
                elif 'from' in json:
                    self.root(json)
                elif 'select' in json:
                    self.root(json)
                elif 'query' in json:
                    self.root(json['query'])
                elif 'union' in json:
                    self.union(json['union'])
                elif 'intersect' in json:
                    self.intersect(json['intersect'])
                elif 'except' in json:
                    self.except_(json['except'])
                else:
                    self.op(json)
        elif isinstance(json, string_types):
            return self.remove_alias(json, self.is_field(json))

    @debug_wrapper
    def delimited_list(self, json):
        for i, element in enumerate(json):
            if isinstance(element, string_types):
                json[i] = self.dispatch(element)
            else:
                self.dispatch(element)

    @debug_wrapper
    def op(self, json):
        if 'on' in json:
            self._on(json)
            return

        if len(json) > 1:
            if 'when' in json and len(json) == 2:
                self._case(json)
            else:
                raise Exception('Operators should have only one key!')
        key, value = list(json.items())[0]

        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        if not key.startswith('_'):
            if isinstance(value, string_types):
                json[key] = self.dispatch(value)
            else:
                if isinstance(value, dict) and len(value) == 0:
                    return
                else:
                    self.dispatch(value)

    @debug_wrapper
    def from_(self, json):
        if 'from' in json:
            from_ = json['from']
            if isinstance(from_, list):
                for token in from_:
                    if isinstance(token, dict) and 'value' in token:
                        self.dispatch(token)
                    else:
                        self.dispatch(token)
            elif isinstance(from_, dict):
                self.dispatch(from_)
            elif not isinstance(from_, string_types):
                raise ValueError('Unrecognized from clause: {}'.format(from_))

    @debug_wrapper
    def value(self, json):
        if isinstance(json['value'], string_types):
            self.remove_alias_and_name(json, self.is_field(json['value']))
        else:
            self.dispatch(json['value'])
            if 'name' in json:
                self.dispatch(json['name'])

    @debug_wrapper
    def remove_alias(self, s, is_field=False):
        if not utils.is_derived(s) and s != '*':
            if utils.field_pattern.fullmatch(s):
                alias_str, field_name = s.split('.', 1)
                table_name = self.get_table_name_by_alias(alias_str)
                if table_name is not None:
                    if not self.schema.field_in_table(field_name, table_name):
                        raise AssertionError('{} is not part of table {}'.format(field_name, table_name))
                    return table_name + '.' + field_name
                else:
                    if self.table_alias_stack and not self.known_table(alias_str):
                        raise AssertionError('Unrecognized alias_str: "{}" ({})'.format(alias_str, self.table2alias))
            elif is_field: # and len(self.table2alias) == 1
                for table_name in self.table2alias:
                    if self.schema.field_in_table(s, table_name):
                        if len(self.table2alias) > 1:
                            if self.verbose:
                                print('Warning: isolated field name matched to table: {} -> {}'.format(s, table_name))
                        return table_name + '.' + s
                if self.verbose:
                    print('WARNING: Isolated field name has not been matched to a table: {} ({})'.format(
                        s, self.table2alias))
        return s

    @debug_wrapper
    def remove_alias_and_name(self, json, is_field=False):
        s = json['value']
        if not utils.is_derived(s):
            if 'is_table' not in json:
                json['value'] = self.remove_alias(s, is_field=is_field)
            if 'name' in json:
                n = json['name']
                if not utils.is_derived(n):
                    del json['name']


class Normalizer(SchemaGroundedTraverser):
    """
    Perform normalization on a SQL AST by
        1. giving each table an alias indexed by the order of occurrences;
        2. adding table alias prefixes to all column names.

    TODO: unit test.
    (Finegan-Dollak et. al. 2018) implemented a SQL query normalization procedures with sophisticated heuristics
    and we decided to adopt it in our project:
        https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/canonicaliser.py
    """

    def Operator(op):
        def func(self, json):
            for i, v in enumerate(json):
                if isinstance(v, string_types):
                    json[i] = self.add_alias(v)
                else:
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
        super(Normalizer, self).__init__(schema, verbose)
        self.table2alias = collections.defaultdict(int)
        self.field2alias = collections.defaultdict(int)

    @debug_wrapper
    def normalize(self, json):
        self.root(json)

    @debug_wrapper
    def dispatch(self, json):
        if isinstance(json, list):
            return self.delimited_list(list)
        if isinstance(json, dict):
            if len(json) > 0:
                if 'from' in json:
                    return self.root(json)
                elif 'select' in json:
                    return self.root(json)
                elif 'query' in json:
                    self.query(json['query'])
                    table_name = utils.DERIVED_TABLE_PREFIX
                    alias_str = utils.DERIVED_TABLE_PREFIX
                    normalized_alias = self.record_table(table_name, alias_str)
                    json['name'] = normalized_alias
                elif 'union' in json:
                    return self.union(json['union'])
                elif 'intersect' in json:
                    return self.intersect(json['intersect'])
                elif 'except' in json:
                    return self.except_(json['except'])
                else:
                    return self.op(json)

    @debug_wrapper
    def from_(self, json):
        if 'from' in json:
            from_ = json['from']
            if isinstance(from_, list):
                for token in from_:
                    table_name = token['value']
                    alias_str = token['name'] if 'name' in token else None
                    normalized_alias = self.record_table(table_name, alias_str)
                    token['name'] = normalized_alias
            elif isinstance(from_, dict):
                self.dispatch(from_)
            else:
                raise ValueError('Unrecognized from clause type')

    @debug_wrapper
    def op(self, json):
        if 'on' in json:
            self._on(json)
            return

        if len(json) > 1:
            raise Exception('Operators should have only one key!')
        key, value = list(json.items())[0]

        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        attr = '_{0}'.format(key)
        if not key.startswith('_') and hasattr(self, attr):
            method = getattr(self, attr)
            return method(value)

        # treat as regular function call
        if isinstance(value, dict) and len(value) == 0:
            return
        elif isinstance(value, string_types):
            json[key] = self.normalize_field_mention(value)
        else:
            raise ValueError('Unrecognized value type: {}'.format(value))

    @debug_wrapper
    def query(self, json):
        # index all tables in the query
        for clause in self.clauses:
            getattr(self, clause)(json)

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            select_ = json['select']
            if 'value' in select_:
                return self.value(select_)

    @debug_wrapper
    def value(self, json):
        value_ = json['value']
        if isinstance(value_, string_types):
            json['value'] = self.normalize_field_mention(value_)
        elif len(self.table2alias) == 1:
            self.dispatch(json)
        normalized_alias = self.record_field()
        json['name'] = normalized_alias

    @debug_wrapper
    def normalize_field_mention(self, fm):
        fm = fm.lower()
        if '.' in fm:
            table_key, field_name = fm.split('.')
        else:
            table_key = self.inv_schema[fm]
            field_name = fm
        _, table2alias = self.get_alias_table_map()
        table_alias = table2alias[table_key]
        return '{}.{}'.format(table_alias, field_name)

    @debug_wrapper
    def record_table(self, table_name, alias_str):
        mention_id = self.table2alias[table_name.lower()]
        _, table2alias = self.get_alias_table_map()
        table_key = alias_str if alias_str is not None else table_name
        normalized_alias = '{}alias{}'.format(table_name.upper(), mention_id)
        table2alias[table_key] = normalized_alias
        self.table2alias[table_name.lower()] += 1
        return normalized_alias

    @debug_wrapper
    def record_field(self):
        field_name = utils.DERIVED_FIELD_PREFIX
        mention_id = self.field2alias[field_name]
        normalized_alias = '{}alias{}'.format(field_name.upper(), mention_id)
        self.field2alias[field_name] += 1
        return normalized_alias
