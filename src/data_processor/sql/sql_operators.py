"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
"""
SQL operator vocabularies.
"""

from src.data_processor.vocab_utils import Vocabulary


field_types = Vocabulary('field_types')
field_types.index_token('not_a_field')
field_types.index_token('text')
field_types.index_token('number')
field_types.index_token('time')
field_types.index_token('boolean')
field_types.index_token('others')


field_vocab = Vocabulary('field')
field_vocab.index_token('*')


arithmetic_ops = Vocabulary('arithmetic_ops')
arithmetic_ops.index_token('+')
arithmetic_ops.index_token('-')
arithmetic_ops.index_token('*')
arithmetic_ops.index_token('/')


aggregation_ops = Vocabulary('aggregation_ops')
aggregation_ops.index_token('')
aggregation_ops.index_token('max')
aggregation_ops.index_token('min')
aggregation_ops.index_token('count')
aggregation_ops.index_token('sum')
aggregation_ops.index_token('avg')


condition_ops = Vocabulary('condition_ops')
condition_ops.index_token('=')
condition_ops.index_token('>')
condition_ops.index_token('<')
condition_ops.index_token('>=')
condition_ops.index_token('<=')
condition_ops.index_token('BETWEEN')
condition_ops.index_token('LIKE')
condition_ops.index_token('IN')


logical_ops = Vocabulary('logical_ops')
logical_ops.index_token('AND')
logical_ops.index_token('OR')


int_vocab = Vocabulary('numerical_value')


value_vocab = Vocabulary('value')
value_vocab.index_token('t')
value_vocab.index_token('f')
value_vocab.index_token('m')
value_vocab.index_token('yes')
value_vocab.index_token('no')
value_vocab.index_token('"')
value_vocab.index_token('!')
value_vocab.index_token('%')
value_vocab.index_token('(')
value_vocab.index_token(')')
value_vocab.index_token('[')
value_vocab.index_token(']')
value_vocab.index_token('{')
value_vocab.index_token('}')
value_vocab.index_token('^')
value_vocab.index_token('$')
value_vocab.index_token('.')
value_vocab.index_token('-')
value_vocab.index_token('*')
value_vocab.index_token('+')
value_vocab.index_token('?')
value_vocab.index_token('|')
value_vocab.index_token('/')
value_vocab.index_token('\\')
value_vocab.index_token(':')


value_vocab_revtok = Vocabulary('value_revtok')
value_vocab_revtok.index_token(' T ')
value_vocab_revtok.index_token(' F ')
value_vocab_revtok.index_token(' M ')
value_vocab_revtok.index_token(' yes ')
value_vocab_revtok.index_token(' no ')
value_vocab_revtok.index_token(' "')
value_vocab_revtok.index_token(' " ')
value_vocab_revtok.index_token('" ')
value_vocab_revtok.index_token('!')
value_vocab_revtok.index_token('% ')
value_vocab_revtok.index_token('%')
value_vocab_revtok.index_token('(')
value_vocab_revtok.index_token(')')
value_vocab_revtok.index_token('[')
value_vocab_revtok.index_token(']')
value_vocab_revtok.index_token('{')
value_vocab_revtok.index_token('}')
value_vocab_revtok.index_token('^')
value_vocab_revtok.index_token('$')
value_vocab_revtok.index_token('.')
value_vocab_revtok.index_token('-')
value_vocab_revtok.index_token('*')
value_vocab_revtok.index_token('+')
value_vocab_revtok.index_token('?')
value_vocab_revtok.index_token('|')
value_vocab_revtok.index_token('/')
value_vocab_revtok.index_token('/ ')
value_vocab_revtok.index_token('\\')
value_vocab_revtok.index_token('\\ ')
value_vocab_revtok.index_token(':')