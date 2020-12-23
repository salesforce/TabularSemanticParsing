"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Utilities for managing input and output vocabularies.
"""
import collections
import numpy as np


functional_token_index = collections.OrderedDict({
    'start_token': ' <START> ',
    'unk_token': 'UNK',
    'eos_token': ' <EOS> ',
    'pad_token': ' <PAD> ',
    'unk_table_token': 'UNK_TABLE',
    'unk_field_token': 'UNK_FIELD',
    'table_token': 'TABLE',
    'field_token': 'FIELD',
    'value_token': 'VALUE',
    'num_token': 'NUM',
    'str_token': 'STRING'
})
functional_tokens = set(functional_token_index.values())


class VocabularyEntry(object):

    def __init__(self, idx, token, in_vocab, frequency=-1):
        self.idx = idx
        self.token = token
        self.in_vocab = in_vocab
        self.frequency = frequency


class Vocabulary(object):
    """
    A class for managing vocabularies and utility
    """

    def __init__(self, tag='', func_token_index=None, tu=None):
        self.tag = tag
        self.ind_ = dict()
        self.rev_ = dict()
        self.size = 0

        fti = func_token_index if func_token_index is not None else functional_token_index

        # decoder specific
        self.start_token = None if tu else fti['start_token']
        self.eos_token = None if tu else fti['eos_token']

        self.unk_token = tu.unk_token if tu else fti['unk_token']
        self.pad_token = tu.pad_token if tu else fti['pad_token']

        if not tu:
            if func_token_index is not None:
                for v in ['start_token', 'unk_token', 'eos_token', 'pad_token']:
                    self.index_token(fti[v], True, -1)

    def index_token(self, v, in_vocab=True, frequency=-1, check_for_seen_vocab=False):
        if check_for_seen_vocab:
            assert (not v in self.ind_)
        idx = len(self.ind_)
        vocab_entry = VocabularyEntry(idx, v, in_vocab, frequency)
        self.ind_[v] = vocab_entry
        self.rev_[idx] = v
        if in_vocab:
            self.size += 1

    def contains(self, v):
        return v in self.ind_

    def is_unknown(self, v):
        if v in self.ind_:
            return not self.ind_[v].in_vocab
        else:
            return True

    def to_idx(self, v):
        if not v in self.ind_:
            return self.unk_id
        v_ent = self.ind_[v]
        if v_ent.in_vocab:
            return v_ent.idx
        else:
            return self.unk_id

    def to_token(self, idx):
        return self.rev_[idx]

    def to_list(self):
        return [x for x in sorted(self.ind_.items(), key=lambda x: x[1].idx)]

    def to_dict(self):
        vocab_dict = dict()
        for v in self.ind_:
            vocab_dict[v] = self.ind_[v].idx
        return vocab_dict

    def merge_with(self, vocab):
        for v in vocab.ind_:
            if not self.contains(v):
                self.index_token(v, vocab.ind_[v].in_vocab, vocab.ind_[v].frequency)

    def save_to_disk(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as o_f:
            for (v, v_ent) in sorted(self.ind_.items(), key=lambda x: x[1].idx):
                o_f.write('{}\t{}\n'.format(v, v_ent.frequency))
        print('{} vocabulary ({}) saved to {}'.format(self.tag, self.full_size, out_path))

    def is_empty(self):
        return not self.ind_

    @property
    def full_size(self):
        return len(self.ind_)

    @property
    def start_id(self):
        if self.start_token is None:
            return None
        v_ent = self.ind_[self.start_token]
        return v_ent.idx

    @property
    def eos_id(self):
        if self.eos_token is None:
            return None
        v_ent = self.ind_[self.eos_token]
        return v_ent.idx

    @property
    def unk_id(self):
        if self.unk_token is None:
            return None
        v_ent = self.ind_[self.unk_token]
        return v_ent.idx

    @property
    def pad_id(self):
        v_ent = self.ind_[self.pad_token]
        return v_ent.idx


class SQLVocabulary(Vocabulary):

    def __init__(self, tag, func_token_index):
        super().__init__(tag, func_token_index)
        fti = func_token_index if func_token_index is not None else functional_token_index
        self.unk_table_token = fti['unk_table_token']
        self.unk_field_token = fti['unk_field_token']
        self.table_token = fti['table_token']
        self.field_token = fti['field_token']
        self.value_token = fti['value_token']
        self.num_token = fti['num_token']
        self.str_token = fti['str_token']

        for v in ['unk_table_token',
                  'unk_field_token',
                  'table_token',
                  'field_token',
                  'value_token',
                  'num_token',
                  'str_token']:
            self.index_token(fti[v], True, -1)


    @property
    def clause_mask(self):
        mask = np.zeros(self.size)
        mask[self.to_idx('select')] = 1
        mask[self.to_idx('from')] = 1
        mask[self.to_idx('where')] = 1
        mask[self.to_idx('group by')] = 1
        mask[self.to_idx('having')] = 1
        mask[self.to_idx('order by')] = 1
        mask[self.to_idx('limit')] = 1
        return mask

    @property
    def op_mask(self):
        mask = np.zeros(self.size)
        mask[self.to_idx('=')] = 1
        mask[self.to_idx('>')] = 1
        mask[self.to_idx('<')] = 1
        mask[self.to_idx('>=')] = 1
        mask[self.to_idx('<=')] = 1
        mask[self.to_idx('!=')] = 1
        mask[self.to_idx('like')] = 1
        mask[self.to_idx('in')] = 1
        mask[self.to_idx('between')] = 1
        return mask

    @property
    def join_mask(self):
        mask = np.zeros(self.size)
        mask[self.to_idx('join')] = 1
        return mask

    @property
    def unk_table_id(self):
        if self.unk_table_token is None:
            return None
        v_ent = self.ind_[self.unk_table_token]
        return v_ent.idx

    @property
    def unk_field_id(self):
        if self.unk_field_token is None:
            return None
        v_ent = self.ind_[self.unk_field_token]
        return v_ent.idx

    @property
    def value_id(self):
        if self.value_token is None:
            return None
        v_ent = self.ind_[self.value_token]
        return v_ent.idx

    @property
    def num_id(self):
        if self.num_token is None:
            return None
        v_ent = self.ind_[self.num_token]
        return v_ent.idx

    @property
    def str_id(self):
        if self.str_token is None:
            return None
        v_ent = self.ind_[self.str_token]
        return v_ent.idx

    @property
    def table_id(self):
        if self.table_token is None:
            return None
        v_ent = self.ind_[self.table_token]
        return v_ent.idx

    @property
    def field_id(self):
        if self.field_token is None:
            return None
        v_ent = self.ind_[self.field_token]
        return v_ent.idx


def is_functional_token(v):
    return v in functional_tokens


value_vocab = Vocabulary('value', functional_token_index)
value_vocab.index_token('%')
value_vocab.index_token('_')
value_vocab.index_token('|')
value_vocab.index_token('&')
value_vocab.index_token('\\')
value_vocab.index_token('^')
value_vocab.index_token('$')
value_vocab.index_token('(')
value_vocab.index_token(')')
value_vocab.index_token('[')
value_vocab.index_token(']')
value_vocab.index_token('{')
value_vocab.index_token('}')
value_vocab.index_token('*')
value_vocab.index_token('+')
value_vocab.index_token('?')
value_vocab.index_token('-')
value_vocab.index_token('.')
value_vocab.index_token('0')
value_vocab.index_token('1')
value_vocab.index_token('2')
value_vocab.index_token('3')
value_vocab.index_token('4')
value_vocab.index_token('5')
value_vocab.index_token('6')
value_vocab.index_token('7')
value_vocab.index_token('8')
value_vocab.index_token('9')
value_vocab.index_token('##0')
value_vocab.index_token('##1')
value_vocab.index_token('##2')
value_vocab.index_token('##3')
value_vocab.index_token('##4')
value_vocab.index_token('##5')
value_vocab.index_token('##6')
value_vocab.index_token('##7')
value_vocab.index_token('##8')
value_vocab.index_token('##9')
value_vocab.index_token('y')
value_vocab.index_token('n')
value_vocab.index_token('yes')
value_vocab.index_token('no')
value_vocab.index_token('true')
value_vocab.index_token('false')
