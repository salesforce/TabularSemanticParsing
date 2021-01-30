"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['g'
                           'loss'], tokens['after']):
        ret += g + a
    return ret.strip()
