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
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
###############################################

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from collections import Mapping
import json
from threading import Lock

from mo_future import text, number_types, binary_type, items, string_types
from pyparsing import ParseException, ParseResults

from moz_sp.debugs import all_exceptions
from moz_sp.extractors.foreign_key_extractor import ForeignKeyCandidateExtractor
from moz_sp.extractors.table_extractor import TableExtractor
from moz_sp.extractors.value_extractor import ValueExtractor
from moz_sp.formatting import Formatter
from moz_sp.formatting import not_number_date_field, not_number_date_field_table
from moz_sp.schema_consistency_checker import SchemaConsistencyChecker
from moz_sp.sql_normalizer import SchemaGroundedTraverser, Denormalizer, Normalizer
from moz_sp.sql_parser import SQLParser
import moz_sp.sql_execution_order_parser as eo_parser
from moz_sp.sql_tokenizer import Tokenizer


parseLocker = Lock()  # ENSURE ONLY ONE PARSING AT A TIME


def parse(sql):
    with parseLocker:
        try:
            all_exceptions.clear()
            sql = sql.rstrip().rstrip(";")
            parse_result = SQLParser.parseString(sql, parseAll=True)
            return _scrub(parse_result)
        except Exception as e:
            if isinstance(e, ParseException) and e.msg == "Expected end of text":
                problems = all_exceptions.get(e.loc, [])
                expecting = [
                    f
                    for f in (set(p.msg.lstrip("Expected").strip() for p in problems)-{"Found unwanted token"})
                    if not f.startswith("{")
                ]
                raise ParseException(sql, e.loc, "Expecting one of (" + (", ".join(expecting)) + ")")
            raise e


def eo_parse(sql):
    with parseLocker:
        try:
            all_exceptions.clear()
            sql = sql.rstrip().rstrip(";")
            parse_result = eo_parser.SQLParser.parseString(sql, parseAll=True)
            return _scrub(parse_result)
        except Exception as e:
            if isinstance(e, ParseException) and e.msg == "Expected end of text":
                problems = all_exceptions.get(e.loc, [])
                expecting = [
                    f
                    for f in (set(p.msg.lstrip("Expected").strip() for p in problems)-{"Found unwanted token"})
                    if not f.startswith("{")
                ]
                raise ParseException(sql, e.loc, "Expecting one of (" + (", ".join(expecting)) + ")")
            raise e


def format(_json, schema, **kwargs):
    return Formatter(schema, **kwargs).format(_json)


def tokenize(sql, value_tokenize, parsed=False, **kwargs):
    ast = sql if parsed else parse(sql)
    tokenizer = Tokenizer(value_tokenize, **kwargs)
    tokens, token_types = tokenizer.tokenize(ast)
    if tokenizer.atomic_value:
        return tokens, token_types, tokenizer.constants
    else:
        return tokens, token_types


def denormalize(sql, schema, return_parse_tree=False, **kwargs):
    dn = Denormalizer(schema, **kwargs)
    ast = sql if isinstance(sql, dict) else parse(sql)
    dn.denormalize(ast)
    if return_parse_tree:
        return ast, dn.contains_self_join
    else:
        dn_sql = format(ast, schema, quote_values=not_number_date_field, should_quote=not_number_date_field_table)
        return dn_sql, dn.contains_self_join


def shallow_normalize(sql, schema):
    ast = parse(sql)
    return format(ast, schema)


def check_schema_consistency(sql, schema, in_execution_order=False, verbose=True):
    if isinstance(sql, dict):
        ast = sql
    else:
        try:
            # check SQL syntax
            if in_execution_order:
                ast = eo_parse(sql)
            else:
                ast = parse(sql)
        except Exception as e:
            if verbose:
                print(str(e))
            return False
    scc = SchemaConsistencyChecker(schema)
    try:
        scc.check(ast)
        return True
    except Exception as e:
        if verbose:
            print(str(e))
        return False


def convert_to_execution_order(sql, schema):
    ast = parse(sql)
    eo_sql = format(ast, schema, in_execution_order=True)
    return eo_sql


def restore_clause_order(sql, schema, check_schema_consistency_=True, verbose=False):
    try:
        eo_ast = eo_parse(sql)
        if check_schema_consistency_:
            schema_consist = check_schema_consistency(eo_ast, schema, verbose=verbose)
            if schema_consist:
                sql = format(eo_ast, schema)
                return sql, True, True
            else:
                return None, True, False
        else:
            sql = format(eo_ast, schema)
            return sql, True, None
    except Exception as e:
        error_msg = str(e)
        if verbose:
            if error_msg.startswith('Expect'):
                print('Parsing error: {}'.format(sql))
            else:
                print(error_msg)
        return None, False, False


def add_join_condition(tokens, schema):
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        new_tokens.append(token)
        # TODO: Differentiate between a FROM keyword and a natural language "from" appeared in a constant.
        if token.lower() == 'from':
            if i + 1 >= len(tokens):
                i += 1
                continue
            # look ahead
            join_tables = []
            j = i + 1
            while j < len(tokens):
                if schema.is_table_name(tokens[j]):
                    join_tables.append(tokens[j])
                    j += 1
                else:
                    if tokens[j] == '(':
                        # heuristically detect nested query
                        stack = [tokens[j]]
                        k = j + 1
                        while k < len(tokens) and stack:
                            if tokens[k] == '(':
                                stack.append(tokens[k])
                            elif tokens[k] == ')':
                                stack.pop()
                            k += 1
                        if stack:
                            raise ValueError('Unmatched parentheses in FROM clause: {}'.format(tokens[j:]))
                        else:
                            if tokens[j+1].lower() == 'from':
                                join_tables.append(tokens[j:k])
                                j = k
                            else:
                                raise ValueError('Unrecognized parenthese content in FROM clause {}'.format(tokens[j:k]))
                    elif tokens[j].lower() in ['select', 'where', 'group by', 'order by', 'having', 'limit', 'offset']:
                        break
                    else:
                        raise ValueError('Non-table tokens found in FROM clause: {}'.format(tokens[j]))
            if not join_tables:
                raise ValueError('Empty FROM clause: {}'.format(tokens[i:]))
            i = j
            if len(join_tables) == 1:
                if isinstance(join_tables[0], string_types):
                    new_tokens += join_tables
                else:
                    new_tokens += join_tables[0]
            else:
                join_clause = [join_tables[0]]
                for k in range(1, len(join_tables)):
                    if isinstance(join_tables[k-1], string_types) and isinstance(join_tables[k], string_types):
                        table1 = join_tables[k - 1]
                        table2 = join_tables[k]
                        foreign_key_pairs = schema.get_foreign_keys_between_tables(table1, table2)
                        if foreign_key_pairs is None:
                            fn1 = schema.get_table_by_name(table1).fields[0].signature
                            fn2 = schema.get_table_by_name(table2).fields[0].signature
                        else:
                            fn1 = schema.get_field_signature(foreign_key_pairs[0][0])
                            fn2 = schema.get_field_signature(foreign_key_pairs[0][1])
                        # TODO: Predict different types of JOIN.
                        join_clause.extend(['JOIN', table2, 'ON', fn1, '=', fn2])
                    else:
                        # TODO: Synthesize JOIN condition for derived tables.
                        join_clause.extend(['JOIN'] + join_tables[k])
                new_tokens += join_clause
        else:
            i += 1
    return new_tokens


def extract_tables(sql, schema):
    ast = sql if isinstance(sql, dict) else parse(sql)
    te = TableExtractor(schema)
    te.extract(ast)
    return te.tables


def extract_foreign_keys(sql, schema, **kwargs):
    fke = ForeignKeyCandidateExtractor(schema, **kwargs)
    ast = sql if isinstance(sql, dict) else parse(sql)
    fke.extract(ast)
    return fke.foreign_keys_readable, sorted(list(fke.foreign_keys), key=lambda x:x[0])


def extract_values(sql, schema):
    ast = sql if isinstance(sql, dict) else parse(sql)
    ve = ValueExtractor(schema)
    ve.extract(ast)
    return ve.values


def _scrub(result):
    if isinstance(result, text):
        return result
    elif isinstance(result, binary_type):
        return result.decode('utf8')
    elif isinstance(result, number_types):
        return result
    elif not result:
        return {}
    elif isinstance(result, (list, ParseResults)):
        if not result:
            return None
        elif len(result) == 1:
            return _scrub(result[0])
        else:
            output = [
                rr
                for r in result
                for rr in [_scrub(r)]
                if rr != None
            ]
            # IF ALL MEMBERS OF A LIST ARE LITERALS, THEN MAKE THE LIST LITERAL
            if all(isinstance(r, number_types) for r in output):
                pass
            elif all(isinstance(r, number_types) or (isinstance(r, Mapping) and "literal" in r.keys()) for r in output):
                output = {"literal": [r['literal'] if isinstance(r, Mapping) else r for r in output]}
            return output
    elif not items(result):
        return {}
    else:
        return {
            k: vv
            for k, v in result.items()
            for vv in [_scrub(v)]
            if vv != None
        }


_ = json.dumps
