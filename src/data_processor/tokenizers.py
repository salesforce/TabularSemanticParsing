"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Collection of tokenizers (readable sequence to token list) and de_tokenizers (token list to readable sequence).
"""
import copy
import json
import re

from mo_future import string_types
import moz_sp
from moz_sp.sql_tokenizer import TABLE, FIELD, RESERVED_TOKEN, VALUE
from moz_sp.utils import alias_pattern, alias_id_revtok_pattern
import src.data_processor.revtok_tokenizer as revtok
import src.utils.utils as utils


# --- Tokenizers -- #

def get_tokenizers(args):
    if args.pretrained_transformer == 'null':
        text_tokenize = revtok_tokenize_with_functional
        program_tokenize = revtok_sql_tokenize
        post_process = revtok_de_tokenize
        transformer_utils = None
    else:
        transformer_utils = utils.get_trans_utils(args)
        text_tokenize = transformer_utils.tokenizer.tokenize
        def p_tokenize(sql, **kwargs):
            return sql_tokenize(sql, text_tokenize, **kwargs)
        program_tokenize = p_tokenize
        def p_detokenize(tokens, **kwargs):
            return trans_de_tokenize(tokens, transformer_utils, **kwargs)
        post_process = p_detokenize
    return text_tokenize, program_tokenize, post_process, transformer_utils


def sql_tokenize(sql, value_tokenize, return_token_types=False, **kwargs):
    if isinstance(sql, string_types):
        sql = standardise_blank_spaces(sql)
        try:
            ast = moz_sp.parse(sql)
        except Exception:
            return value_tokenize(sql)
    else:
        ast = sql

    output = moz_sp.tokenize(ast, value_tokenize, parsed=True, **kwargs)

    if return_token_types:
        return output
    else:
        return output[0]


wikisql_agg_ops = ['%', 'max', 'min', 'count', 'sum', 'avg']
wikisql_cond_ops = ['=', '>', '<', 'op']

def wikisql_vec_to_struct(v, vocab, memory, token_starts, token_ends, text, tu, adjust_cond_val=True):

    def get_field_id(x):
        assert(x >= (vocab.size + len(memory) + 2))
        schema_pos = x - vocab.size - len(memory)
        # * [T] [C] [C] ...
        field_id = schema_pos - 2
        return field_id

    def get_agg_op_id(x):
        assert(x < vocab.size)
        t = vocab.to_token(x)
        agg_id = list.index(wikisql_agg_ops, t)
        return agg_id

    def get_cond_op_id(x):
        assert(x < vocab.size)
        t = vocab.to_token(x)
        cond_op_id = list.index(wikisql_cond_ops, t)
        return cond_op_id

    def get_val(ts):
        s = tu.tokenizer.convert_tokens_to_string(ts)
        s = s.replace(' \' s', '\'s')
        s = s.replace(' \' t', '\'t')
        return s.strip()

    def match_ignoring_spaces(span, text):
        c_indices = [i for i, c in enumerate(text) if c != ' ']
        text_ns = text.replace(' ', '')
        span_ns = span.replace(' ', '')
        c_start = text_ns.find(span_ns)
        if c_start >= 0:
            c_end = c_start + len(span_ns)
            start = c_indices[c_start]
            if c_end < len(c_indices):
                end = c_indices[c_end]
            else:
                end = len(text)
            return text[start:end].strip()
        else:
            return None

    def adjust_unmatched_cond_val(s, cv_token_ids, text):
         # (I) If the value string is an entire span copied from the original input, crop out this span
        if len(cv_token_ids) > 0 and cv_token_ids[0] >= 0 and cv_token_ids[-1] >= 0:
            t_start = token_starts[cv_token_ids[0]]
            t_end = token_ends[cv_token_ids[-1]]
            if t_start < t_end:
                s = text[t_start:t_end]
                return s
        # (II) match s with original string ignoring spaces
        m = match_ignoring_spaces(s.strip(), text.strip())
        if m:
            return m

        # (III) Try all other corrections
        if s.startswith('" '):
            s = '"' + s[2:]
        if s.endswith(' "'):
            s = s[:-2] + '"'
        if s in text:
            return s
        s = s.replace(' ( ', ' (')
        if s in text:
            return s
        s = s.replace(' ) ', ') ')
        s = s.replace(' )"', ')"')
        s = re.sub(re.compile(r' \)$'), ')', s)
        if s in text:
            return s
        s = s.replace(' - ', '-')
        s = re.sub(re.compile(r' -$'), '-', s)
        if s in text:
            return s
        s = s.replace(' = ', '=')
        s = re.sub(re.compile(r' =$'), '=', s)
        if s in text:
            return s
        s = s.replace(' % ', '%')
        s = re.sub(re.compile(r' %$'), '%', s)
        if s in text:
            return s
        s = s.replace(' %)', '%)')
        if s in text:
            return s
        s = s.replace(' / ', '/')
        s = re.sub(re.compile(r' /$'), '/', s)
        if s in text:
            return s
        s = s.replace('  \\', ' \\')
        s = re.sub(re.compile(r' \\\\$'), '\\\\', s)
        if s in text:
            return s
        # process comma
        s = s.replace(' , ', ', ')
        s = re.sub(re.compile(r' ,$'), ',', s)
        s = s.replace(', ', ',')
        if s in text:
            return s
        # process dot
        s = s.replace(' . ', '. ')
        s = re.sub(re.compile(r' \.$'), '\.', s)
        i = 0
        t = []
        while i < len(s):
            t.append(s[i])
            if s[i] == '.':
                if i + 1 < len(s) and s[i + 1] == ' ':
                    if i + 2 < len(s) and s[i + 2].isnumeric():
                        i += 1
            i += 1
        s = ''.join(t)
        if s in text:
            return s
        s = s.replace(' .)', '.)')
        if s in text:
            return s
        s = s.encode().replace(b' \xe2\x80\x93 ', b'\xe2\x80\x93').decode()
        if s in text:
            return s

        # (IV). Encoding issue
        text = text.encode().replace(b'\xc2\xa0', b' ').decode()
        m = match_ignoring_spaces(s.strip(), text.strip())
        if m:
            return m

        return s

    def get_cond_struct(v, i):
        x = int(v[i])
        c_field_id = get_field_id(x)
        i += 1
        x = int(v[i])
        cond_op_id = get_cond_op_id(x)
        i += 1
        cond_val_tokens, cv_token_ids = [], []
        while i < len(v):
            x = int(v[i])
            if x < vocab.size:
                t = vocab.to_token(x)
                if t == 'and':
                    assert(len(cond_val_tokens) > 0)
                    cond_val = get_val(cond_val_tokens)
                    if cond_val.lower() not in text_lower and adjust_cond_val:
                        cond_val = adjust_unmatched_cond_val(cond_val, cv_token_ids, text)
                    return [c_field_id, cond_op_id, cond_val], i + 1
                else:
                    cond_val_tokens.append(t)
                cv_token_ids.append(-1)
            else:
                memory_pos = x - vocab.size
                assert(memory_pos < len(memory))
                cond_val_tokens.append(memory[memory_pos])
                cv_token_ids.append(memory_pos)
            i += 1
        assert(len(cond_val_tokens) > 0)
        cond_val = get_val(cond_val_tokens)
        if cond_val.lower() not in text_lower and adjust_cond_val:
            cond_val = adjust_unmatched_cond_val(cond_val, cv_token_ids, text)
        return [c_field_id, cond_op_id, cond_val], i

    text_lower = text.lower()
    assert(len(memory) == len(token_starts) == len(token_ends))

    for i in range(len(v)):
        if v[i] == vocab.eos_id:
            break
    v = v[:i]
    assert(len(v) >= 2)

    struct = {
        'sel': None,
        'conds': [],
        'agg': 0
    }

    # select clause
    x = int(v[0])
    struct['sel'] = get_field_id(x)
    x = int(v[1])
    struct['agg'] = get_agg_op_id(x)

    if len(v) == 2:
        return struct

    # where clause
    x = int(v[2])
    assert(x < vocab.size)
    t = vocab.to_token(x)
    assert(t == 'where')

    i = 3
    while i < len(v):
        cond, i = get_cond_struct(v, i)
        struct['conds'].append(cond)

    return struct

def wikisql_struct_to_tokens(ast, schema, tu):
    """
    :param ast:
        "sql": {
                "sel": 2,
                "conds": [
                    [
                        0,
                        0,
                        "South Australia"
                    ]
                ],
                "agg": 0
            }
    :param schema:
    """
    tokens, token_types = [], []

    select_field_id = ast['sel']
    agg_op_id = ast['agg']
    tokens.append(schema.get_field(select_field_id).signature)
    tokens.append(wikisql_agg_ops[agg_op_id])
    token_types.append(FIELD)
    token_types.append(RESERVED_TOKEN)

    conds = ast['conds']
    if len(conds) > 0:
        tokens.append('where')
        token_types.append(RESERVED_TOKEN)
        for i, (field_id, cond_op_id, val) in enumerate(conds):
            assert(cond_op_id < 3)
            cond_tokens = [schema.get_field(field_id).signature, wikisql_cond_ops[cond_op_id]]
            cond_token_types = [FIELD, RESERVED_TOKEN]
            val = str(val)
            val_tokens = tu.tokenizer.tokenize(val)
            cond_tokens.extend(val_tokens)
            cond_token_types.extend([VALUE for _ in val_tokens])

            if i > 0:
                tokens.append('and')
                token_types.append(RESERVED_TOKEN)
            tokens.extend(cond_tokens)
            token_types.extend(cond_token_types)

    return tokens, token_types


def revtok_tokenize(*args, **kwargs):
    return revtok.tokenize(*args, **kwargs)

def revtok_tokenize_with_functional(s, functional_tokens=[]):
    return tokenize_with_functional(s, functional_tokens, tokenize=revtok_tokenize)

def revtok_sql_tokenize(sql, keep_singleton_fields=False):

    def replace_alias_with_table_name(sql):
        for m in re.findall(alias_pattern, sql):
            if m[0] and table_name != 'DERIVED_TABLE':
                m = m[0]
                table_name = m.split('alias', 1)[0]
                sql = sql.replace(m.split('.', 1)[0] + '.', table_name + '.')
        return sql

    sql = standardise_blank_spaces(sql)
    if keep_singleton_fields:
        sql = replace_alias_with_table_name(sql)

    # replace alias names with a string which does not contain numbers
    # alias2nonum, nonum2alias = {}, {}
    # alias_pattern = re.compile('\w+alias\d+')
    # for m in re.findall(alias_pattern, sql):
    #     num = re.search('\d+', m).group(0)
    #     if num:
    #         nonum = m.replace(num, n2w.convert(int(num)).upper())
    #         alias2nonum[m] = nonum
    #         nonum2alias[nonum] = m
    #         sql = sql.replace(m, nonum)
    # replace variable names with a string which does not contain numbers
    # var2nonum, nonum2var = {}, {}
    # var_pattern = re.compile('[a-z]+\d')
    # for m in re.findall(var_pattern, sql):
    #     num = re.search('\d+', m).group(0)
    #     if num:
    #         nonum = m.replace(num, n2w.convert(int(num)).upper())
    #         var2nonum[m] = nonum
    #         nonum2var[nonum] = m
    #         sql = sql.replace(m, nonum)
    # replace time expressions with a string which does not contain colon
    # te2nocolon, nocolon2te = dict(), dict()
    # for m in re.findall(time_pattern, sql):
    #     te = None
    #     for te in m:
    #         if te:
    #             break
    #     assert(te is not None)
    #     nocolon = te.replace(':', COLON_PLACE_HOLDER)
    #     te2nocolon[te] = nocolon
    #     nocolon2te[nocolon] = te
    #     sql = sql.replace(te, nocolon)

    if keep_singleton_fields:
        skipped_punctuations = {'_', '.'}
    else:
        skipped_punctuations = {'_'}
    tokens = revtok.tokenize(sql, skipped_punctuations=skipped_punctuations)

    # restore time expressions
    # for i, token in enumerate(tokens):
    #     nocolon = token.strip()
    #     if nocolon in nocolon2te:
    #         te = nocolon2te[nocolon]
    #         tokens[i] = token.replace(nocolon, te)
    # restore alias names
    # if len(alias2nonum) > 0:
    #     for i, token in enumerate(tokens):
    #         nonum = token.strip()
    #         if nonum in nonum2alias:
    #             alias = nonum2alias[nonum]
    #             tokens[i] = tokens[i].replace(nonum, alias)
    # restore variables
    # if len(var2nonum) > 0:
    #     for i, token in enumerate(tokens):
    #         nonum = token.strip()
    #         if nonum in nonum2var:
    #             var = nonum2var[nonum]
    #             tokens[i] = tokens[i].replace(nonum, var)
    return process_alias_tokenization(tokens)


# --- De-tokenizers --- #

def trans_de_tokenize(tokens, tu,
                      process_dot=True,
                      process_quote=True,
                      process_paratheses=True,
                      process_underscore=True,
                      process_dash=True,
                      process_unequal_sign=True):
    if process_unequal_sign:
        for i, token in enumerate(tokens):
            if token == '<>':
                tokens[i] = '!='

    out = tu.tokenizer.convert_tokens_to_string(tokens)
    if process_quote:
        parts = out.split('"')
        new_out = ''
        for i, part in enumerate(parts):
            if i % 2 == 0:
                new_out += part
            else:
                new_out += '"{}"'.format(part.strip())
        out = new_out

        out = out.replace(" 's", "'s")

        i = 0
        new_out = ''
        num_quotes = 0
        while i < len(out):
            if out[i] != "'":
                new_out += out[i]
                i += 1
            elif out[i] == "'":
                num_quotes += 1
                if num_quotes % 2 == 1:
                    new_out += out[i]
                    j = i + 1
                    while j < len(out) and out[j] == ' ':
                        j += 1
                    i = j
                else:
                    new_out = new_out.strip() + out[i]
                    i += 1
        out = new_out

    if process_dot:
        out = out.replace(' . ', '.')
    if process_paratheses:
        out = out.replace('( ', '(').replace(' )', ')')
    if process_underscore:
        out = out.replace(' _ ', '_')
    if process_dash:
        out = out.replace(' - ', '-')

    return out


def revtok_de_tokenize(tokens):
    return revtok.detokenize(tokens)


# --- Utilities --- #

def standardise_blank_spaces(query):
    """
    split on special characters except _.:-

    Code adapted from:
    https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/canonicaliser.py
    """
    in_squote, in_dquote = False, False
    tmp_query = []
    pos = 0
    while pos < len(query):
        char = query[pos]
        pos += 1
        # Handle whether we are in quotes
        if char in ["'", '"']:
            if not (in_squote or in_dquote):
                tmp_query.append(" ")
            in_squote, in_dquote = update_quotes(char, in_squote, in_dquote)
            tmp_query.append(char)
            if not (in_squote or in_dquote):
                tmp_query.append(" ")
        elif in_squote or in_dquote:
            tmp_query.append(char)
        elif char in "!=<>,;()[]{}+*/\\#":
            tmp_query.append(" ")
            tmp_query.append(char)
            while pos < len(query) and query[pos] in "!=<>+*" and char in "!=<>+*":
                tmp_query.append(query[pos])
                pos += 1
            tmp_query.append(" ")
        else:
            tmp_query.append(char)
    new_query = ''.join(tmp_query)

    # Remove blank spaces just inside quotes:
    tmp_query = []
    in_squote, in_dquote = False, False
    prev = None
    prev2 = None
    for char in new_query:
        skip = False
        for quote, symbol in [(in_squote, "'"), (in_dquote, '"')]:
            if quote:
                if char in " \n"  and prev == symbol:
                    skip = True
                    break
                if char in " \n"  and prev == "%" and prev2 == symbol:
                    skip = True
                    break
                elif char == symbol and prev in " \n":
                    tmp_query.pop()
                elif char == symbol and prev == "%" and prev2 in " \n":
                    tmp_query.pop(len(tmp_query) - 2)
        if skip:
            continue

        in_squote, in_dquote = update_quotes(char, in_squote, in_dquote)
        tmp_query.append(char)
        prev2 = prev
        prev = char
    new_query = ''.join(tmp_query)

    # Replace single quotes with double quotes where possible
    tmp_query = []
    in_squote, in_dquote = False, False
    pos = 0
    while pos < len(new_query):
        char = new_query[pos]
        if (not in_dquote) and char == "'":
            to_add = [char]
            pos += 1
            saw_double = False
            while pos < len(new_query):
                tchar = new_query[pos]
                if tchar == '"':
                    saw_double = True
                to_add.append(tchar)
                if tchar == "'":
                    break
                pos += 1
            if not saw_double:
                to_add[0] = '"'
                to_add[-1] = '"'
            tmp_query.append(''.join(to_add))
        else:
            tmp_query.append(char)

        in_squote, in_dquote = update_quotes(char, in_squote, in_dquote)

        pos += 1
    new_query = ''.join(tmp_query)

    # remove repeated blank spaces
    new_query = ' '.join(new_query.split())

    # Remove spaces that would break SQL functions
    new_query = "COUNT(".join(new_query.split("count ("))
    new_query = "LOWER(".join(new_query.split("lower ("))
    new_query = "MAX(".join(new_query.split("max ("))
    new_query = "MIN(".join(new_query.split("min ("))
    new_query = "SUM(".join(new_query.split("sum ("))
    new_query = "AVG(".join(new_query.split("avg ("))
    new_query = "COUNT(".join(new_query.split("COUNT ("))
    new_query = "LOWER(".join(new_query.split("LOWER ("))
    new_query = "MAX(".join(new_query.split("MAX ("))
    new_query = "MIN(".join(new_query.split("MIN ("))
    new_query = "SUM(".join(new_query.split("SUM ("))
    new_query = "AVG(".join(new_query.split("AVG ("))
    new_query = "COUNT( *".join(new_query.split("COUNT(*"))
    new_query = "YEAR(CURDATE())".join(new_query.split("YEAR ( CURDATE ( ) )"))

    return new_query


def update_quotes(char, in_single, in_double):
    """
    Code adapted from:
    https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/canonicaliser.py
    """
    if char == '"' and not in_single:
        in_double = not in_double
    elif char == "'" and not in_double:
        in_single = not in_single
    return in_single, in_double


def process_alias_tokenization(tokens):
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.endswith('alias ') \
                and (i < len(tokens) - 1 and re.fullmatch(alias_id_revtok_pattern, tokens[i + 1])) \
                and (token[:-6].isupper()):
            processed_tokens.append('{} '.format(token[:-6]))
            processed_tokens.append('alias' + tokens[i + 1])
            i += 2
        else:
            processed_tokens.append(token)
            i += 1
    return processed_tokens


def tokenize_with_functional(s, functional_tokens, tokenize):
    # var_pattern = re.compile('[a-z]+\d')
    tokens = s.split()
    processed_tokens = []

    for token in tokens:
        if not token.strip():
            continue
        if token in functional_tokens:
            processed_tokens.append(token)
        else:
            # m = var_pattern.fullmatch(token)
            # if m:
            #     num = re.search('\d+', token).group(0)
            #     nonum = token.replace(num, n2w.convert(int(num)).upper())
            #     processed_token = tokenize(nonum)[0]
            #     processed_tokens.append(processed_token.replace(nonum, token))
            # else:
            tokens_ = tokenize(token)
            for i, token_ in enumerate(tokens_):
                # if is_number(token_) and not token_.endswith(' ') and i < len(tokens_) - 1:
                #         and not (len(tokens_[i+1]) == 1 and unicodedata.category(tokens_[i+1]) == 'Po'):
                #     token_ += ' '
                processed_tokens.append(token_)
    return processed_tokens


