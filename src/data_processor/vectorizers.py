"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Collection of vectorizers (token list to index list) and de_vectorizers (index_list to token list).
"""
from mo_future import string_types

from moz_sp.sql_tokenizer import TABLE, FIELD, RESERVED_TOKEN, VALUE
from src.data_processor.sql.sql_reserved_tokens import digits


# --- Vectorizers -- #

def vectorize(tokens, vocab):
    return [vocab.to_idx(x) for x in tokens]


def vectorize_singleton(tokens, token_types, vocab):
    ids = []
    for x, t in zip(tokens, token_types):
        if t == TABLE:
            ids.append(vocab.table_id)
        elif t == FIELD:
            ids.append(vocab.field_id)
        elif t == VALUE:
            if x in digits or vocab.is_unknown(x):
                ids.append(vocab.value_id)
            else:
                ids.append(vocab.to_idx(x))
        elif t == RESERVED_TOKEN:
            ids.append(vocab.to_idx(x))
        else:
            raise NotImplementedError
        assert(ids[-1] != vocab.unk_id)
    return ids


def vectorize_ptr_in(tokens, out_vocab, capture_enc_dec_overlap_lexicon=True):
    """
    Map each token in the input sequence to
        - its index in the output vocab (if in the output vocab) or
        - its copy index = out_vocab_size + its position in the input (if not in the output vocab)

    :return ptr_ids:
    :return unique_input_ids: a dictionary that maps each input token to a unique index (if not in the output vocab,
        map to out_vocab_size + the position of its first occurrence in the input)
    """
    ptr_ids = []
    unique_input_ids = {}
    for i, token in enumerate(tokens):
        if capture_enc_dec_overlap_lexicon and not out_vocab.is_unknown(token):
            # some words are in the output vocab
            ptr_ids.append(out_vocab.to_idx(token))
        else:
            # no input word is in the output vocab
            if not token in unique_input_ids:
                idx = out_vocab.size + i
                ptr_ids.append(idx)
                unique_input_ids[token] = idx
            else:
                ptr_ids.append(unique_input_ids[token])
    return ptr_ids, unique_input_ids


def vectorize_ptr_out(tokens, out_vocab, unique_input_ids, relaxed_matching=False):
    """
    Pointer-generator output.

    Map tokens in the output sequence to output vocab indices (in output vocab) or copy indices
    (not in output vocab).
    """
    ptr_ids = []
    for i, token in enumerate(tokens):
        if not out_vocab.is_unknown(token):
            ptr_ids.append(out_vocab.to_idx(token))
        elif token in unique_input_ids:
            ptr_ids.append(unique_input_ids[token])
        else:
            if relaxed_matching and token.lower() in unique_input_ids:
                ptr_ids.append(unique_input_ids[token.lower()])
            else:
                ptr_ids.append(out_vocab.unk_id)

    return ptr_ids


def vectorize_field_ptr_out(tokens, token_types, out_vocab, unique_input_ids, max_memory_size, schema=None,
                            num_included_nodes=None, relaxed_matching=False):
    """
    Pointer-generator output with field and table reference grounded to the schema.
    """
    assert(len(tokens) == len(token_types))
    ptr_ids = []
    for i, token in enumerate(tokens):
        assert(isinstance(token, string_types))
        token_type = token_types[i]
        if token_type in [TABLE, FIELD]:
            schema_pos = schema.get_schema_pos(token)
            if schema_pos < num_included_nodes:
                ptr_ids.append(out_vocab.size + max_memory_size + schema_pos)
            else:
                if token_type == TABLE:
                    ptr_ids.append(out_vocab.unk_table_id)
                else:
                    ptr_ids.append(out_vocab.unk_field_id)
        elif token_type == RESERVED_TOKEN:
            ptr_ids.append(out_vocab.to_idx(token))
        else:
            assert(token_type == VALUE)
            if not out_vocab.is_unknown(token):
                ptr_ids.append(out_vocab.to_idx(token))
            elif token in unique_input_ids:
                ptr_ids.append(unique_input_ids[token])
            else:
                if relaxed_matching and token.lower() in unique_input_ids:
                    ptr_ids.append(unique_input_ids[token.lower()])
                else:
                    ptr_ids.append(out_vocab.value_id)

    return ptr_ids


# --- De-vectorizers -- #

def de_vectorize(vec_cpu, rev_vocab, post_process, return_tokens=False):
    tokens = []
    for j in range(len(vec_cpu)):
        token_id = int(vec_cpu[j])
        if j == 0 and token_id == rev_vocab.start_id:
            continue
        if token_id == rev_vocab.eos_id or token_id == rev_vocab.pad_id:
            break
        tokens.append(rev_vocab.to_token(token_id))
    if return_tokens:
        return tokens
    s = post_process(tokens)
    return s


def de_vectorize_ptr(vec_cpu, rev_vocab, memory, post_process, return_tokens=False):
    tokens = []
    for j in range(len(vec_cpu)):
        token_id = int(vec_cpu[j])
        if j == 0 and token_id == rev_vocab.start_id:
            continue
        if token_id == rev_vocab.eos_id or token_id == rev_vocab.pad_id:
            break
        if token_id < rev_vocab.size:
            tokens.append(rev_vocab.to_token(token_id))
        else:
            memory_pos = token_id - rev_vocab.size
            tokens.append(memory[memory_pos])
    if return_tokens:
        return tokens
    s = post_process(tokens)
    return s


def de_vectorize_field_ptr(vec_cpu, rev_vocab, memory, schema, table_po=None, field_po=None, post_process=None,
                           return_tokens=False):
    tokens = []
    for j in range(len(vec_cpu)):
        token_id = int(vec_cpu[j])
        if j == 0 and token_id == rev_vocab.start_id:
            continue
        if token_id == rev_vocab.eos_id or token_id == rev_vocab.pad_id:
            break
        if token_id < rev_vocab.size:
            tokens.append(rev_vocab.to_token(token_id))
        else:
            memory_pos = token_id - rev_vocab.size
            if memory_pos < len(memory):
                tokens.append(memory[memory_pos])
            else:
                schema_pos = memory_pos - len(memory)
                tokens.append(schema.get_signature_by_schema_pos(schema_pos, table_po=table_po, field_po=field_po))
    if return_tokens:
        return tokens
    s = post_process(tokens)
    return s
