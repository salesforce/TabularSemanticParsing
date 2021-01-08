"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Encode DB content.
"""

import difflib
from mo_future import string_types
from rapidfuzz import fuzz
import src.utils.utils as utils
from src.utils.utils import deprecated


class Match(object):
    def __init__(self, start, size):
        self.start = start
        self.size = size


def is_span_separator(c):
    return c in '\'"()`,.?! '


def split(s):
    return [c.lower() for c in s.strip()]


def prefix_match(s1, s2):
    i, j = 0, 0
    for i in range(len(s1)):
        if not is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    else:
        return False


def get_effecitve_match_source(s, start, end):
    _start = -1

    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if is_span_separator(s[i]):
            _start = i
            break

    if _start < 0:
        return None

    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if is_span_separator(s[i]):
            _end = i
            break

    if _end < 0:
        return None

    while(_start < len(s) and is_span_separator(s[_start])):
        _start += 1
    while(_end >= 0 and is_span_separator(s[_end])):
        _end -= 1

    return Match(_start, _end - _start + 1)


def get_matched_entries(s, field_values, m_theta=0.85, s_theta=0.85):
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split(s)
    else:
        n_grams = s

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, string_types):
            continue
        fv_tokens = split(field_value)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0:
            source_match = get_effecitve_match_source(n_grams, match.a, match.a + match.size)
            if source_match and source_match.size > 1:
                match_str = field_value[match.b:match.b + match.size]
                source_match_str = s[source_match.start:source_match.start+source_match.size]
                c_match_str = match_str.lower().strip()
                c_source_match_str = source_match_str.lower().strip()
                c_field_value = field_value.lower().strip()
                if c_match_str and not utils.is_number(c_match_str) and not utils.is_common_db_term(c_match_str):
                    if utils.is_stopword(c_match_str) or utils.is_stopword(c_source_match_str) or \
                            utils.is_stopword(c_field_value):
                        continue
                    if c_source_match_str.endswith(c_match_str + '\'s'):
                        match_score = 1.0
                    else:
                        if prefix_match(c_field_value, c_source_match_str):
                            match_score = fuzz.ratio(c_field_value, c_source_match_str) / 100
                        else:
                            match_score = 0
                    if (utils.is_commonword(c_match_str) or utils.is_commonword(c_source_match_str) or
                            utils.is_commonword(c_field_value)) and match_score < 1:
                        continue
                    s_match_score = match_score
                    if match_score >= m_theta and s_match_score >= s_theta:
                        if field_value.isupper() and match_score * s_match_score < 1:
                            continue
                        matched[match_str] = (field_value, source_match_str, match_score, s_match_score, match.size)

    if not matched:
        return None
    else:
        return sorted(matched.items(), key=lambda x:(1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]), reverse=True)


@deprecated
def split_old(s):
    return [' '] + [c.lower() for c in s.strip()] + [' ']


@deprecated
def source_match_score(s, start, end):
    _start = -1

    for i in range(start, start-2, -1):
        if i < 0:
            _start = i + 1
            break
        if not s[i].strip():
            _start = i
            break

    if _start < 0:
        return 0

    _end = -1
    for i in range(end-1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if not s[i].strip() or s[i] == ',':
            _end = i
            break

    if _end < 0:
        return 0

    fuzzy_match_size = _end + 1 - _start
    fuzzy_match_score = (end - start) / fuzzy_match_size
    return fuzzy_match_score


