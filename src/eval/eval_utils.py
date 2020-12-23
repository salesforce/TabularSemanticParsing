"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import difflib

from src.data_processor.tokenizers import standardise_blank_spaces


def equal_ignoring_trivial_diffs(s1, s2, verbose=False):

    s1 = standardise_blank_spaces(s1)
    s2 = standardise_blank_spaces(s2)

    # track parentheses
    p_stack = []
    for v in difflib.ndiff(s1, s2):
        if not v[0].strip():
            # character matched
            continue
        if v.strip() in ['+', '-']:
            # ignore differences in white spaces
            continue
        if v in ['+ (', '- (']:
            if len(p_stack) > 0:
                # mismatched nested parentheses are unusual
                # print_input()
                return False
            p_stack.append(v)
        elif v == '+ )':
            if len(p_stack) < 1 or p_stack[-1] != '+ (':
                # print_input()
                return False
            p_stack.pop()
        elif v == '- )':
            if len(p_stack) < 1 or p_stack[-1] != '- (':
                # print_input()
                return False
            p_stack.pop()
        else:
            # print_input()
            return False

    if len(p_stack) > 0:
        # print_input()
        return False
    else:
        return True