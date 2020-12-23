"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Tokenize a SQL query given its AST.
"""

import copy
import functools
from mo_future import string_types, text

from moz_sp.debugs import debug_wrapper
from moz_sp.keywords import join_keywords
from moz_sp.sql_normalizer import SchemaGroundedTraverser
from moz_sp.formatting import always, should_quote, not_number_date_field, never
import moz_sp.utils as utils
from src.utils.utils import is_number, strip_quotes


TABLE = 0
FIELD = 1
RESERVED_TOKEN = 2
VALUE = 3


def add_parentheses(x):
    if len(x) == 2 and isinstance(x[0], list):
        if x[0][0] == '(' and x[0][-1] == ')':
            return x
        return ['('] + x[0] + [')'], [RESERVED_TOKEN] + x[1] + [RESERVED_TOKEN]
    else:
        if x[0] == '(' and x[-1] == ')':
            return x
        return ['('] + x + [')']


def append_to_list(x, a, in_front=False):
    if len(a) == 2 and isinstance(a[0], list):
        if in_front:
            return a[0] + x[0], a[1] + x[1]
        else:
            return x[0] + a[0], x[1] + a[1]
    else:
        if in_front:
            return a + x[0], [RESERVED_TOKEN for _ in a] + x[1]
        else:
            return x[0] + a, x[1] + [RESERVED_TOKEN for _ in a]


def connect_by_keywords(x1, x2, a):
    if len(a) == 2 and isinstance(a[0], list):
        return x1[0] + a[0] + x2[0], x1[1] + a[1] + x2[1]
    else:
        return x1[0] + a + x2[0], x1[1] + [RESERVED_TOKEN for _ in a] + x2[1]


def list_join(L, x):
    """
    Joint a list of lists (L) with a specific item to form a new flat list.
    """
    if isinstance(x, string_types):
        x = (x, RESERVED_TOKEN)
    if len(L) == 0:
        return [], []
    out, out_types = copy.deepcopy(L[0][0]), copy.deepcopy(L[0][1])
    for v, t in L[1:]:
        if x:
            out += [x[0]]
            out_types += [x[1]]
        out += v
        out_types += t
    return out, out_types


def Operator(op, parentheses=False):
    def func(self, json):
        arguments = []
        for v in json:
            if isinstance(v, dict) and utils.is_subquery(v):
                arguments.append(add_parentheses(self.dispatch(v)))
            else:
                arguments.append(self.dispatch(v))
        out = list_join(arguments, op)
        if parentheses:
            out = add_parentheses(out)
        return out
    return func


def escape(identifier, value_tokenize, ansi_quotes, should_quote):
    """
    Escape identifiers.

    ANSI uses single quotes, but many databases use back quotes.
    """
    tokens = value_tokenize(strip_quotes(identifier))
    token_types = [VALUE for _ in tokens]

    if not should_quote(identifier):
        return tokens, token_types
    else:
        quote = '"' if ansi_quotes else '`'
        return [quote] + [x.replace("'", '"') for x in tokens] +  [quote], \
               [RESERVED_TOKEN] + token_types + [RESERVED_TOKEN]


class Tokenizer(SchemaGroundedTraverser):

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

    def __init__(self, value_tokenize, schema=None, keep_singleton_fields=False, caseless=True, ansi_quotes=True,
                 no_join_condition=False, omit_from_clause=False, in_execution_order=False, atomic_value=False,
                 num_token=None, str_token=None, verbose=False):
        """
        :param value_tokenize: tokenizer for constant values in the SQL query
        :param schema:
        :param keep_singleton_fields:
        :param caseless:
        :param ansi_quotes:
        :param no_join_condition:
        :param omit_from_clause:
        :param in_execution_order:
        :param atomic_value:
        :param verbose:
        """
        super().__init__(schema, verbose)
        self.value_tokenize = value_tokenize
        self.keep_singleton_fields = keep_singleton_fields
        self.caseless = caseless
        self.ansi_quotes = ansi_quotes
        self.no_join_condition = no_join_condition
        self.omit_from_clause = omit_from_clause
        self.in_execution_order = in_execution_order

        self.atomic_value = atomic_value
        self.num_token = num_token
        self.str_token = str_token
        self.constants = []

    @debug_wrapper
    def tokenize(self, json):
        if 'union' in json:
            tokens, token_types = self.union(json['union'])
        elif 'intersect' in json:
            tokens, token_types = self.intersect(json['intersect'])
        elif 'except' in json:
            tokens, token_types = self.except_(json['except'])
        else:
            tokens, token_types = self.query(json)

        if self.caseless:
            tokens_ = []
            for t, t_type in zip(tokens, token_types):
                if not (t.startswith('[') and t.endswith(']')) and t_type != VALUE:
                    tokens_.append(t.lower())
                else:
                    tokens_.append(t)
            return tokens_, token_types
        else:

            return tokens, token_types

    @debug_wrapper
    def dispatch(self, json, is_table=False, should_quote=should_quote):
        if isinstance(json, list):
            return self.delimited_list(json)
    
        if isinstance(json, dict):
            if len(json) == 0:
                return [], []
            elif 'value' in json:
                return self.value(json)
            elif 'from' in json:
                # Nested query 'from'
                return add_parentheses(self.tokenize(json))
            elif 'query' in json:
                # Nested query 'query'
                nested_query_tokens = self.tokenize(json['query'])
                if 'name' in json:
                    return connect_by_keywords(
                        add_parentheses(nested_query_tokens), self.dispatch(json['name'], is_table=True), ['AS'])
                else:
                    return add_parentheses(nested_query_tokens)
            elif 'union' in json:
                # Nested query 'union'
                return add_parentheses(self.union(json['union']))
            elif 'intersect' in json:
                return add_parentheses(self.intersect(json['intersect']))
            elif 'except' in json:
                return add_parentheses(self.except_(json['except']))
            else:
                return self.op(json)
    
        if not isinstance(json, string_types):
            json = text(json)
        if is_table and json.lower() == 't0':
            return self.value_tokenize(json), [RESERVED_TOKEN, RESERVED_TOKEN]
        if self.keep_singleton_fields and (is_table or self.is_field(json) or json == '*'):
            if is_table:
                return [json], [TABLE]
            else:
                return [json], [FIELD]
        if self.atomic_value:
            self.constants.append(escape(json, self.value_tokenize, self.ansi_quotes, never))
            if is_number(json):
                return [self.num_token], [VALUE]
            else:
                return [self.str_token], [VALUE]
        else:
            return escape(json, self.value_tokenize, self.ansi_quotes, should_quote)
        # print(json, type(json))
        # tokens = self.value_tokenize(text(json))
        # token_types = [VALUE for _ in tokens]
        # return tokens, token_types

    @debug_wrapper
    def delimited_list(self, json):
        return list_join([self.dispatch(element) for element in json], ',')

    @debug_wrapper
    def value(self, json):
        parts = self.dispatch(json['value'], is_table=('is_table' in json))
        if 'name' in json:
            parts = connect_by_keywords(parts, self.dispatch(json['name'], should_quote=never), ['AS'])
        return parts

    @debug_wrapper
    def op(self, json):
        if 'on' in json:
            return self._on(json)

        # PATCHING:
        #   properly handle cases where the SQL misses a JOIN condition
        for join_keyword in join_keywords:
            if join_keyword in json and len(json) == 1:
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
        key = key.upper()
        if isinstance(value, dict) and len(value) == 0:
            # NOT SURE IF AN EMPTY dict SHOULD BE DELT WITH HERE, OR IN self.dispatch()
            return [key] + ['(', ')'], [RESERVED_TOKEN, RESERVED_TOKEN, RESERVED_TOKEN]
        else:
            if key in ['DISTINCT', 'ALL']:
                output = self.dispatch(value)
            else:
                output = add_parentheses(self.dispatch(value))
            return [key] + output[0], [RESERVED_TOKEN] + output[1]

    @debug_wrapper
    def _exists(self, value):
        return append_to_list(self.dispatch(value), ['IS', 'NOT', 'NULL'])

    @debug_wrapper
    def _missing(self, value):
        return append_to_list(self.dispatch(value), ['IS', 'NULL'])

    @debug_wrapper
    def _like(self, pair):
        return connect_by_keywords(self.dispatch(pair[0]), self.dispatch(pair[1]), ['LIKE'])

    @debug_wrapper
    def _nlike(self, pair):
        return connect_by_keywords(self.dispatch(pair[0]), self.dispatch(pair[1]), ['NOT', 'LIKE'])

    @debug_wrapper
    def _is(self, pair):
        return connect_by_keywords(self.dispatch(pair[0]), self.dispatch(pair[1]), ['IS'])

    @debug_wrapper
    def _in(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if valid and valid[0] != '(':
            valid = add_parentheses(valid)

        return connect_by_keywords(self.dispatch(json[0]), valid, ['IN'])

    @debug_wrapper
    def _nin(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if valid and valid[0] != '(':
            valid = add_parentheses(valid)

        return connect_by_keywords(self.dispatch(json[0]), valid, ['NOT', 'IN'])

    @debug_wrapper
    def _between(self, json):
        scope = connect_by_keywords(self.dispatch(json[1]), self.dispatch(json[2]), ['AND'])
        return connect_by_keywords(self.dispatch(json[0]), scope, ['BETWEEN'])

    @debug_wrapper
    def _case(self, checks):
        parts, parts_types = ['CASE'], [RESERVED_TOKEN]
        for check in checks:
            if isinstance(check, dict):
                parts.extend(['WHEN'] + self.dispatch(check['when'])[0])
                parts.extend(['THEN'] + self.dispatch(check['then'])[0])
                parts_types.extend([RESERVED_TOKEN] + self.dispatch(check['when'])[1])
                parts_types.extend([RESERVED_TOKEN] + self.dispatch(check['then'])[1])
            else:
                parts.extend(['ELSE'] + self.dispatch(check)[0])
                parts_types.extend([RESERVED_TOKEN] + self.dispatch(check)[1])
        parts.append('END')
        parts_types.append(RESERVED_TOKEN)
        return parts, parts_types

    @debug_wrapper
    def _literal(self, json):
        if isinstance(json, list):
            return add_parentheses(
                (functools.reduce(lambda x, y: x+y, [self._literal(v)[0] for v in json]),
                 functools.reduce(lambda x, y: x+y, [self._literal(v)[1] for v in json])))
        elif isinstance(json, string_types):
            if self.atomic_value:
                self.constants.append(escape(json, self.value_tokenize, self.ansi_quotes, never))
                if is_number(json):
                    return [self.num_token], [VALUE]
                else:
                    return [self.str_token], [VALUE]
            else:
                return escape(json, self.value_tokenize, self.ansi_quotes, always)
        else:
            tokens = self.value_tokenize(text(json))
            token_types = [VALUE for _ in tokens]
            return tokens, token_types

    @debug_wrapper
    def _on(self, json):
        for join_key in join_keywords:
            if join_key in json:
                if self.no_join_condition:
                    return self.dispatch(json[join_key], is_table=isinstance(json[join_key], string_types))
                elif 'on' not in json:
                    join_content = self.dispatch(json[join_key], is_table=isinstance(json[join_key], string_types))
                    return append_to_list(join_content, [join_key.upper()], in_front=True)
                else:
                    join_content = connect_by_keywords(self.dispatch(json[join_key],
                                                                     is_table=isinstance(json[join_key], string_types)),
                                                       self.dispatch(json['on']), ['ON'])
                    return append_to_list(join_content, [join_key.upper()], in_front=True)
        raise AttributeError('Unrecognized JOIN keyword: {}'.format(json))

    @debug_wrapper
    def union(self, json):
        return list_join([self.tokenize(query) for query in json], 'UNION')

    @debug_wrapper
    def intersect(self, json):
        return list_join([self.tokenize(query) for query in json], 'INTERSECT')

    @debug_wrapper
    def except_(self, json):
        return list_join([self.tokenize(query) for query in json], 'EXCEPT')

    @debug_wrapper
    def query(self, json):
        clauses = self.clauses_in_execution_order if self.in_execution_order else self.clauses
        self.get_alias_table_map(json)
        out = [[], []]
        for clause in clauses:
            part = getattr(self, clause)(json)
            if part:
                out = append_to_list(out, part)
        if self.table_alias_stack:
            self.pop_table_alias_stack()
        return out

    @debug_wrapper
    def select(self, json):
        if 'select' in json:
            return append_to_list(self.dispatch(json['select']), ['SELECT'], in_front=True)

    @debug_wrapper
    def from_(self, json):
        if self.omit_from_clause:
            return []
        is_join = False
        if 'from' in json:
            from_ = json['from']
            if isinstance(from_, dict):
                return append_to_list(self.dispatch(from_), ['FROM'], in_front=True)
            if not isinstance(from_, list):
                from_ = [from_]
            parts = []
            for token in from_:
                for join_key in join_keywords:
                    if join_key in token:
                        is_join = True
                parts.append(self.dispatch(token, is_table=isinstance(token, string_types)))
            joiner = None if is_join else ','
            rest = list_join(parts, joiner)
            return append_to_list(rest, ['FROM'], in_front=True)

    @debug_wrapper
    def where(self, json):
        if 'where' in json:
            return append_to_list(self.dispatch(json['where']), ['WHERE'], in_front=True)

    @debug_wrapper
    def groupby(self, json):
        if 'groupby' in json:
            return append_to_list(self.dispatch(json['groupby']), ['GROUP BY'], in_front=True)

    @debug_wrapper
    def having(self, json):
        if 'having' in json:
            return append_to_list(self.dispatch(json['having']), ['HAVING'], in_front=True)

    @debug_wrapper
    def orderby(self, json):
        if 'orderby' in json:
            orderby = json['orderby']
            if isinstance(orderby, dict):
                orderby = [orderby]
            mark = []
            for o in orderby:
                order_by = self.dispatch(o)
                sort_by = o.get('sort', '')
                if sort_by:
                    order_by[0].append(sort_by)
                    order_by[1].append(RESERVED_TOKEN)
                mark.append(order_by)
            mark = list_join(mark, ',')
            return append_to_list(mark, ['ORDER BY'], in_front=True)

    @debug_wrapper
    def limit(self, json):
        if 'limit' in json:
            return append_to_list(self.dispatch(json['limit']), ['LIMIT'], in_front=True)

    @debug_wrapper
    def offset(self, json):
        if 'offset' in json:
            return append_to_list(self.dispatch(json['offset']), ['OFFSET'], in_front=True)
