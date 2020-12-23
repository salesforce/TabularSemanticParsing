"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
DEBUG = False

all_exceptions = {}


def record_exception(instring, loc, expr, exc):
    # if DEBUG:
    #     print ("Exception raised:" + _ustr(exc))
    es = all_exceptions.setdefault(loc, [])
    es.append(exc)


def nothing(*args):
    pass


def debug_wrapper(func):
    def func_wrapper(*args, **kwargs):
        if DEBUG:
            print(func.__name__)
        return func(*args, **kwargs)
    return func_wrapper


if DEBUG:
    debug = (None, None, None)
else:
    debug = (nothing, nothing, record_exception)