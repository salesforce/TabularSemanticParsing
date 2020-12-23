"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import re


table_pattern = re.compile('[A-Za-z_]\w+|t')
alias_pattern = re.compile('([A-Z_]+alias\d)|'
                           '(T\d)|'
                           '(t\d)')
alias_id_pattern = re.compile('\d+')
alias_id_revtok_pattern = re.compile('\d+ ')
field_pattern = re.compile('([A-Z_]{1,100}alias\d+\.[A-Za-z_]\w+)|'
                           '([A-Za-z_]{1,100}\d+\.[A-Za-z_]\w+)|'
                           '([A-Za-z_]\w+\.[A-Za-z_]\w+)|'
                           '(T\d\.[A-Za-z_]\w+)|'
                           '(t\d\.[A-Za-z_]\w+)')
number_pattern = re.compile('\d+((\.\d+)|(,\d+))?')
time_pattern = re.compile('(\d{2}:\d{2}:\d{2})|(\d{2}:\d{2})')
datetime_pattern = re.compile('(\d{4})-(\d{2})-(\d{2})( (\d{2}):(\d{2}):(\d{2}))?')


DERIVED_TABLE_PREFIX = 'DERIVED_TABLE'
DERIVED_FIELD_PREFIX = 'DERIVED_FIELD'


def is_derived_table(s):
    return s.startswith(DERIVED_TABLE_PREFIX)


def is_derived_field(s):
    return s.startswith(DERIVED_FIELD_PREFIX)


def is_derived(s):
    return is_derived_table(s) or is_derived_field(s)


def is_subquery(json):
    if not isinstance(json, dict):
        return False
    return 'from' in json or \
           'query' in json or \
           'union' in json or \
           'intersect' in json or \
           'except' in json