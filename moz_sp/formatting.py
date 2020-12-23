# encoding: utf-8

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Beto Dealmeida (beto@dealmeida.net)

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import re

from mo_future import text, string_types

from moz_sp.debugs import debug_wrapper
from moz_sp.traverser import SchemaGroundedTraverser
from moz_sp.keywords import RESERVED, join_keywords
from moz_sp.utils import alias_pattern, field_pattern, number_pattern, datetime_pattern, table_pattern
import moz_sp.utils as utils


def add_parentheses(x):
    if x.startswith('(') and x.endswith(')'):
        return x
    return '({0})'.format(x)


def always(identifier):
    return True


def not_number_date_field(identifier):
    return (identifier != '*'
            and not re.fullmatch(number_pattern, identifier)
            and not re.fullmatch(datetime_pattern, identifier)
            and not re.fullmatch(field_pattern, identifier))


def not_number_date_field_table(identifier):
    return (identifier != '*'
            and not re.fullmatch(number_pattern, identifier)
            and not re.fullmatch(datetime_pattern, identifier)
            and not re.fullmatch(field_pattern, identifier)
            and not re.fullmatch(table_pattern, identifier)
            and not re.fullmatch(alias_pattern, identifier))


def never(identifier):
    return False


def should_quote(identifier):
    """
    Return true if a given identifier should be quoted.

    This is usually true when the identifier:

      - is a reserved word
      - contain spaces
      - does not match the regex `[a-zA-Z_]\w*`

    """
    VALID = re.compile(r'[a-zA-Z_]\w*')
    return (identifier != '*' and
            (not identifier in RESERVED) and
            (not re.fullmatch(number_pattern, identifier)))
    # return (
    #     identifier != '*' and
    #     (not VALID.match(identifier) or identifier in RESERVED) and
    #     not re.fullmatch(number_pattern, identifier))


def escape(identifier, ansi_quotes, should_quote):
    """
    Escape identifiers.

    ANSI uses single quotes, but many databases use back quotes.

    """
    if not should_quote(identifier):
        return identifier

    quote = '"' if ansi_quotes else '`'
    identifier = identifier.replace(quote, 2*quote)
    return '{0}{1}{2}'.format(quote, identifier, quote)


def Operator(op, parentheses=False):
    op = ' {0} '.format(op)
    def func(self, json):
        arguments = []
        for v in json:
            if isinstance(v, dict) and utils.is_subquery(v):
                arguments.append(add_parentheses(self.dispatch(v)))
            else:
                arguments.append(self.dispatch(v))
        out = op.join(arguments)
        if parentheses:
            out = add_parentheses(out)
        return out
    return func


class Formatter(SchemaGroundedTraverser):

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

    clauses_in_execution_order = [
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
    _div = Operator('/', parentheses=True)
    _add = Operator('+')
    _sub = Operator('-', parentheses=True)
    _neq = Operator('!=')
    _gt = Operator('>')
    _lt = Operator('<')
    _gte = Operator('>=')
    _lte = Operator('<=')
    _eq = Operator('=')
    _or = Operator('OR')
    _and = Operator('AND')

    def __init__(self, schema, ansi_quotes=True, quote_values=not_number_date_field, should_quote=should_quote,
                 in_execution_order=False, verbose=False):
        super().__init__(schema, verbose)
        self.ansi_quotes = ansi_quotes
        self.quote_values = quote_values
        self.should_quote = should_quote
        self.in_execution_order = in_execution_order

    @debug_wrapper
    def format(self, json):
        if 'union' in json:
            return self.union(json['union'])
        elif 'intersect' in json:
            return self.intersect(json['intersect'])
        elif 'except' in json:
            return self.except_(json['except'])
        else:
            return self.query(json)

    @debug_wrapper
    def dispatch(self, json, is_table=False, should_quote=should_quote):
        if isinstance(json, list):
            return self.delimited_list(json)

        if isinstance(json, dict):
            if len(json) == 0:
                return ''
            elif 'value' in json:
                return self.value(json)
            elif 'from' in json:
                # Nested query 'from'
                return add_parentheses(self.format(json))
            elif 'query' in json:
                # Nested query 'query'
                nested_query = self.format(json['query'])
                if 'name' in json:
                    return '{0} AS {1}'.format(add_parentheses(nested_query), json['name'])
                else:
                    return add_parentheses(nested_query)
            elif 'union' in json:
                # Nested query 'union'
                return add_parentheses(self.union(json['union']))
            elif 'intersect' in json:
                return add_parentheses(self.intersect(json['intersect']))
            elif 'except' in json:
                return add_parentheses(self.except_(json['except']))
            else:
                return self.op(json)
        if isinstance(json, text):
            if is_table or self.is_field(json):
                return json
            else:
                return escape(json, self.ansi_quotes, should_quote)
        
        return text(json)

    @debug_wrapper
    def delimited_list(self, json):
        return ', '.join(self.dispatch(element) for element in json)

    @debug_wrapper
    def value(self, json):
        parts = [self.dispatch(json['value'], is_table=('is_table' in json))]
        if 'name' in json:
            parts.extend(['AS', self.dispatch(json['name'], should_quote=never)])
        return ' '.join(parts)

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
            return key.upper() + "()"  # NOT SURE IF AN EMPTY dict SHOULD BE DELT WITH HERE, OR IN self.dispatch()
        else:
            key = key.upper()
            if key in ['DISTINCT', 'ALL']:
                return '{0} {1}'.format(key, self.dispatch(value))
            else:
                return '{0}{1}'.format(key, add_parentheses(self.dispatch(value)))

    @debug_wrapper
    def _exists(self, value):
        return '{0} IS NOT NULL'.format(self.dispatch(value))

    @debug_wrapper
    def _missing(self, value):
        return '{0} IS NULL'.format(self.dispatch(value))

    @debug_wrapper
    def _between(self, triple):
        return '{0} BETWEEN {1} AND {2}'.format(self.dispatch(triple[0]), self.dispatch(triple[1]), self.dispatch(triple[2]))

    @debug_wrapper
    def _like(self, pair):
        return '{0} LIKE {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    @debug_wrapper
    def _nlike(self, pair):
        return '{0} NOT LIKE {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    @debug_wrapper
    def _is(self, pair):
        return '{0} IS {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    @debug_wrapper
    def _in(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = add_parentheses(valid)

        return '{0} IN {1}'.format(json[0], valid)

    @debug_wrapper
    def _nin(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = add_parentheses(valid)

        return '{0} NOT IN {1}'.format(json[0], valid)

    @debug_wrapper
    def _case(self, checks):
        parts = ['CASE']
        for check in checks:
            if isinstance(check, dict):
                parts.extend(['WHEN', self.dispatch(check['when'])])
                parts.extend(['THEN', self.dispatch(check['then'])])
            else:
                parts.extend(['ELSE', self.dispatch(check)])
        parts.append('END')
        return ' '.join(parts)

    @debug_wrapper
    def _literal(self, json):
        if isinstance(json, list):
            return add_parentheses(', '.join(self._literal(v) for v in json))
        elif isinstance(json, text):
            return "'{0}'".format(json.replace("'", '"'))
        else:
            return str(json)

    @debug_wrapper
    def _on(self, json):
        for join_key in join_keywords:
            if join_key in json:
                return '{0} {1} ON {2}'.format(
                    join_key.upper(), self.dispatch(json[join_key], is_table=isinstance(json[join_key], text)),
        self.dispatch(json['on']))
        raise AttributeError('Unrecognized JOIN keywords: {}'.format(json))

    @debug_wrapper
    def union(self, json):
        return ' UNION '.join(self.format(query) for query in json)

    @debug_wrapper
    def intersect(self, json):
        return ' INTERSECT '.join(self.format(query) for query in json)

    @debug_wrapper
    def except_(self, json):
        return ' EXCEPT '.join(self.format(query) for query in json)

    @debug_wrapper
    def query(self, json):
        clauses = self.clauses_in_execution_order if self.in_execution_order else self.clauses
        self.get_alias_table_map(json)
        seq = ' '.join(
            part
            for clause in clauses
            for part in [getattr(self, clause)(json)]
            if part
        )
        self.pop_table_alias_stack()
        return seq

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            return 'SELECT {0}'.format(self.dispatch(json['select']))

    @debug_wrapper
    def from_(self, json):
        is_join = False
        if 'from' in json:
            from_ = json['from']

            if isinstance(from_, dict):
                return 'FROM {0}'.format(self.dispatch(from_))

            if not isinstance(from_, list):
                from_ = [from_]
            parts = []
            for token in from_:
                for join_key in join_keywords:
                    if join_key in token:
                        is_join = True
                parts.append(self.dispatch(token, is_table=isinstance(token, text)))
            joiner = ' ' if is_join else ', '
            rest = joiner.join(parts)
            return 'FROM {0}'.format(rest)

    @debug_wrapper
    def where(self, json):
        if 'where' in json:
            return 'WHERE {0}'.format(self.dispatch(json['where']))

    @debug_wrapper
    def groupby(self, json):
        if 'groupby' in json:
            return 'GROUP BY {0}'.format(self.dispatch(json['groupby']))

    @debug_wrapper
    def having(self, json):
        if 'having' in json:
            return 'HAVING {0}'.format(self.dispatch(json['having']))

    @debug_wrapper
    def orderby(self, json):
        if 'orderby' in json:
            orderby = json['orderby']
            if isinstance(orderby, dict):
                orderby = [orderby]
            return 'ORDER BY {0}'.format(
                ','.join(['{0} {1}'.format(self.dispatch(o), o.get('sort', '').upper()).strip() for o in orderby]))

    @debug_wrapper
    def limit(self, json):
        if 'limit' in json:
            return 'LIMIT {0}'.format(self.dispatch(json['limit']))

    @debug_wrapper
    def offset(self, json):
        if 'offset' in json:
            return 'OFFSET {0}'.format(self.dispatch(json['offset']))
