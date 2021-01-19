# encoding: utf-8

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Traverse a SQL AST and extract the set of tables in it.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from mo_future import string_types, text

from moz_sp.debugs import debug_wrapper
from moz_sp.keywords import join_keywords
import moz_sp.utils as utils
from src.utils.utils import to_indexable


def remove_redundancy(l):
    l_uni = []
    seen = set()
    for x in l:
        if not x in seen:
            l_uni.append(x)
            seen.add(x)
    return l_uni

class TableExtractor(object):

    def __init__(self, schema):
        self.schema = schema
        self.tables = []

    @debug_wrapper
    def extract(self, json):
        self.tables = []
        self.dispatch(json)
        self.tables = remove_redundancy(self.tables)

    @debug_wrapper
    def record_table(self, table_name):
        table_name = to_indexable(table_name)
        self.tables.append(table_name)

    @debug_wrapper
    def dispatch(self, json):
        if isinstance(json, list):
            for item in json:
                self.dispatch(item)
        elif isinstance(json, dict):
            for key in json:
                if key == 'from':
                    self.from_(json['from'])
                else:
                    self.dispatch(json[key])

    @debug_wrapper
    def from_(self, from_):
        if not isinstance(from_, list):
            from_ = [from_]
        for item in from_:
            if isinstance(item, string_types):
                self.record_table(item)
            elif isinstance(item, dict):
                if any([jk in item for jk in join_keywords]):
                    join_item = item['join']
                    if isinstance(join_item, string_types):
                        self.record_table(join_item)
                    else:
                        self.process_table_item(join_item)
                elif 'query' in item:
                    self.extract(item)
                elif 'name' in item or 'value' in item:
                    self.process_table_item(item)
            else:
                raise ValueError

    @debug_wrapper
    def process_table_item(self, item):
        if 'value' in item:
            value = item['value']
            if utils.is_subquery(value):
                self.extract(value)
            else:
                self.record_table(value)