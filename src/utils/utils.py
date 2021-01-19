"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Utility functions.
"""
import collections
import datetime
import functools
import inspect
import random
import re
import string
import warnings

# from nltk.corpus import stopwords
# try:
#     _stopwords = set(stopwords.words('english'))
# except LookupError:
#     import nltk
#     nltk.download('stopwords')
#     _stopwords = set(stopwords.words('english'))
_stopwords = {'who', 'ourselves', 'down', 'only', 'were', 'him', 'at', "weren't", 'has', 'few', "it's", 'm', 'again',
              'd', 'haven', 'been', 'other', 'we', 'an', 'own', 'doing', 'ma', 'hers', 'all', "haven't", 'in', 'but',
              "shouldn't", 'does', 'out', 'aren', 'you', "you'd", 'himself', "isn't", 'most', 'y', 'below', 'is',
              "wasn't", 'hasn', 'them', 'wouldn', 'against', 'this', 'about', 'there', 'don', "that'll", 'a', 'being',
              'with', 'your', 'theirs', 'its', 'any', 'why', 'now', 'during', 'weren', 'if', 'should', 'those', 'be',
              'they', 'o', 't', 'of', 'or', 'me', 'i', 'some', 'her', 'do', 'will', 'yours', 'for', 'mightn', 'nor',
              'needn', 'the', 'until', "couldn't", 'he', 'which', 'yourself', 'to', "needn't", "you're", 'because',
              'their', 'where', 'it', "didn't", 've', 'whom', "should've", 'can', "shan't", 'on', 'had', 'have',
              'myself', 'am', "don't", 'under', 'was', "won't", 'these', 'so', 'as', 'after', 'above', 'each', 'ours',
              'hadn', 'having', 'wasn', 's', 'doesn', "hadn't", 'than', 'by', 'that', 'both', 'herself', 'his',
              "wouldn't", 'into', "doesn't", 'before', 'my', 'won', 'more', 'are', 'through', 'same', 'how', 'what',
              'over', 'll', 'yourselves', 'up', 'mustn', "mustn't", "she's", 're', 'such', 'didn', "you'll", 'shan',
              'when', "you've", 'themselves', "mightn't", 'she', 'from', 'isn', 'ain', 'between', 'once', 'here',
              'shouldn', 'our', 'and', 'not', 'too', 'very', 'further', 'while', 'off', 'couldn', "hasn't", 'itself',
              'then', 'did', 'just', "aren't"}

_commonwords = {
    'no', 'yes', 'many'
}

string_types = (type(b''), type(u''))


SEQ2SEQ = 0
SEQ2SEQ_PG = 1
BRIDGE = 2

model_index = {
    'seq2seq': SEQ2SEQ,
    'seq2seq.pg': SEQ2SEQ_PG,
    'bridge': BRIDGE
}


# --- string utilities --- #

def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except:
        return False


def is_stopword(s):
    return s.strip() in _stopwords


def is_commonword(s):
    return s.strip() in _commonwords


def is_common_db_term(s):
    return s.strip() in ['id']


def to_string(v):
    if isinstance(v, bytes):
        try:
            s = v.decode('utf-8')
        except UnicodeDecodeError:
            s = v.decode('latin-1')
    else:
        s = str(v)
    return s


def encode_str_list(l, encoding):
    return [x.encode(encoding) for x in l]


def list_to_hist(l):
    hist = collections.defaultdict(int)
    for x in l:
        hist[x] += 1
    return hist


def remove_parentheses_str(s):
    return re.sub(r'\([^)]*\)', '', s).strip()


def strip_quotes(s):
    start = 0
    while start < len(s):
        if s[start] in ['"', '\'']:
            start += 1
        else:
            break
    end = len(s)
    while end > start:
        if s[end-1] in ['"', '\'']:
            end -= 1
        else:
            break
    if start == end:
        return ''
    else:
        return s[start:end]


def to_indexable(s, caseless=True):
    """
    Normalize table and column surface form to facilitate matching.
    """
    """replace_list = [
        ('crs', 'course'),
        ('mgr', 'manager')
    ]

    check_replace_list = {
        'stu': 'student',
        'prof': 'professor',
        'res': 'restaurant',
        'cust': 'customer',
        'ref': 'reference',
        'dept': 'department',
        'emp': 'employee'
    }

    def to_indexable_name(name):
        name = name.strip().lower()
        tokens = name.split()
        if tokens:
            tokens = functools.reduce(lambda x, y: x + y, [token.split('_') for token in tokens])
        else:
            if verbose:
                print('Warning: name is an empty string')
        for i, token in enumerate(tokens):
            if token in check_replace_list:
                tokens[i] = check_replace_list[token]

        n_name = ''.join(tokens)

        for s1, s2 in replace_list:
            n_name = n_name.replace(s1, s2)
        return n_name

    if '.' in s:
        s1, s2 = s.split('.', 1)
        return to_indexable_name(s1) + '.' + to_indexable_name(s2)
    else:
        return to_indexable_name(s)
    """
    if caseless:
        s = s.lower()
    return ''.join(s.replace('_', '').split())


def restore_feature_case(features, s, bu):
    tokens, starts, ends  = [], [], []

    i = 0
    for feat in features:
        if feat.endswith('##'):
            feat_ = feat[:-2]
        elif feat.startswith('##'):
            feat_ = feat[2:]
        elif feat == bu.unk_token:
            feat_ = '*'
            feat = '*'
        else:
            feat_ = feat

        if i >= len(s):
            print('Warning: feature and string mismatch: {}'.format(s))
            return features, [], []

        while not s[i].strip():
            i += 1
        token = s[i:i+len(feat_)]
        starts.append(i)
        ends.append(i+len(feat_))

        if feat.endswith('##'):
            token += '##'
        if feat.startswith('##'):
            token = '##' + token
        if len(token) != len(feat):
            print('Warning: feat and token mismatch: {}'.format(feat, token))
            return features, [], []

        tokens.append(token)
        i = i + len(feat_)
    return tokens, starts, ends


def get_sub_token_ids(question_tokens, span_ids, tu):
    st, ed = span_ids
    prefix_tokens = question_tokens[:st]
    prefix = tu.tokenizer.convert_tokens_to_string(prefix_tokens)
    prefix_sub_tokens = tu.tokenizer.tokenize(prefix)

    span_tokens = question_tokens[st:ed]
    span = tu.tokenizer.convert_tokens_to_string(span_tokens)
    span_sub_tokens = tu.tokenizer.tokenize(span)

    return len(prefix_sub_tokens), len(prefix_sub_tokens) + len(span_sub_tokens)


def get_trans_utils(args):
    if args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased'):
        import src.utils.trans.bert_utils as bu
        return bu
    elif args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-cased'):
        import src.utils.trans.bert_cased_utils as bcu
        return bcu
    elif args.pretrained_transformer.startswith('roberta-'):
        import src.utils.trans.roberta_utils as ru
        return ru
    elif args.pretrained_transformer == 'null':
        return None
    else:
        raise NotImplementedError


def get_random_tag(k=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


def get_time_tag():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


# --- other utilities --- #

def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


if __name__ == '__main__':
    text = 'how is LifE %-%'
    features = bu.tokenizer.tokenize(text)
    tokens, starts, ends = restore_feature_case(features, text, bu)
    print('Text: {}'.format(text))
    print('Features: {}'.format(features))
    print('Tokens: {}'.format(tokens))
    print('Starts: {}'.format(starts))
    print('Ends: {}'.format(ends))
