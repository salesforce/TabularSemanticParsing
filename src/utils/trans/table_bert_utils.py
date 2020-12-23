"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Huggingface pretrained BERT model utilities.
"""

import os
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'table-bert-checkpoint'))
bt = tokenizer

pad_token = tokenizer.pad_token
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
mask_token = tokenizer.mask_token
unk_token = tokenizer.unk_token
pad_id = bt.convert_tokens_to_ids(pad_token)
cls_id = bt.convert_tokens_to_ids(cls_token)
sep_id = bt.convert_tokens_to_ids(sep_token)
mask_id = bt.convert_tokens_to_ids(mask_token)
unk_id = bt.convert_tokens_to_ids(unk_token)

table_marker = '[TAB]'
field_marker = '[COL]'
value_marker = '[ROW]'
asterisk_marker = '[unused52]'
table_marker_id = bt.convert_tokens_to_ids(table_marker)
field_marker_id = bt.convert_tokens_to_ids(field_marker)
value_marker_id = bt.convert_tokens_to_ids(value_marker)
asterisk_marker_id = bt.convert_tokens_to_ids(asterisk_marker)

text_field_marker = '[unused53]'
number_field_marker = '[unused54]'
time_field_marker = '[unused55]'
boolean_field_marker = '[unused56]'
other_field_marker = '[unused57]'
text_field_marker_id = bt.convert_tokens_to_ids(text_field_marker)
number_field_marker_id = bt.convert_tokens_to_ids(number_field_marker)
time_field_marker_id = bt.convert_tokens_to_ids(time_field_marker)
boolean_field_marker_id = bt.convert_tokens_to_ids(boolean_field_marker)
other_field_marker_id = bt.convert_tokens_to_ids(other_field_marker)


typed_field_markers = [
    text_field_marker,
    number_field_marker,
    time_field_marker,
    boolean_field_marker,
    other_field_marker
]