"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Build vocabulary. No longer actively used given pre-trained language models.
"""
import collections

from src.data_processor.data_loader import load_parsed_sqls
import src.data_processor.tokenizers as tok
from src.data_processor.path_utils import get_vocab_path
from src.data_processor.processor_utils import AugmentedText2SQLExample, get_ast
from src.data_processor.sql.sql_reserved_tokens import sql_reserved_tokens, sql_reserved_tokens_revtok
from src.data_processor.vocab_utils import functional_token_index, functional_tokens, Vocabulary
from src.utils.utils import SEQ2SEQ, SEQ2SEQ_PG, BRIDGE
import src.utils.utils as utils


def build_vocab(args, dataset, schema_graphs):
    """
    Construct vocabularies.

    This function saves to disk:
    - text vocab: consists of tokens appeared in the natural language query and schema
    - program vocab: consists of tokens appeared in the program
    - schema vocab: consists of table and field names from the schema
    - world vocab: consists of tokens in the program that does not come from any of the above category
      (which likely needed to be inferred from world knowledge)
    """
    print('Constructing vocabulary...')

    text_tokenize, program_tokenize, _, tu = tok.get_tokenizers(args)
    if args.pretrained_transformer:
        sql_reserved_vocab = sql_reserved_tokens
    else:
        sql_reserved_vocab = sql_reserved_tokens_revtok
    parsed_programs = load_parsed_sqls(args, augment_with_wikisql=args.augment_with_wikisql)

    schema_graphs.lexicalize_graphs(tokenize=text_tokenize, normalized=(args.model_id in [BRIDGE]))

    # compute text and program vocab
    text_hist, program_hist = collections.defaultdict(int), collections.defaultdict(int)
    world_vocab = Vocabulary('world')

    for split in ['train', 'dev', 'test']:
        if not split in dataset:
            continue
        data_split = dataset[split]
        for i, example in enumerate(data_split):
            if isinstance(example, AugmentedText2SQLExample):
                continue
            schema_graph = schema_graphs.get_schema(example.db_id)
            text = example.text
            if args.pretrained_transformer:
                text_tokens = text_tokenize(text)
            else:
                text_tokens = text_tokenize(text.lower(), functional_tokens)
            for word in text_tokens:
                text_hist[word] += 1
            for program in example.program_list:
                ast, _ = get_ast(program, parsed_programs, args.denormalize_sql, schema_graph)
                if ast:
                    program = ast
                program_tokens = program_tokenize(program, omit_from_clause=args.omit_from_clause,
                                                  no_join_condition=args.no_join_condition)
                for token in program_tokens:
                    program_hist[token] += 1
                    if split == 'train':
                        if not token in text_tokens and not sql_reserved_vocab.contains(token):
                            world_vocab.index_token(token, in_vocab=True)
            if i > 0 and i % 5000 == 0:
                print('{} examples processed'.format(i))

    if args.pretrained_transformer.startswith('bert') or args.pretrained_transformer == 'table-bert':
        text_hist = dict()
        for v in tu.tokenizer.vocab:
            text_hist[v] = tu.tokenizer.vocab[v]
        for v in tu.tokenizer.added_tokens_encoder:
            text_hist[v] = tu.tokenizer.convert_tokens_to_ids(v)
        schema_lexical_vocab = None
    elif args.pretrained_transformer.startswith('roberta'):
        text_hist = tu.tokenizer.encoder
        schema_lexical_vocab = None
    else:
        schema_lexical_vocab = schema_graphs.get_lexical_vocab()

    export_vocab(text_hist, program_hist, schema_lexical_vocab, world_vocab, args)


def export_vocab(text_hist, program_hist, schema_lexical_vocab, world_vocab, args):

    if schema_lexical_vocab is not None:
        # Merge the lexicon based on the natural language text and the database schema
        for v in schema_lexical_vocab:
            text_hist[v] = -1

    text_vocab = Vocabulary('text', func_token_index=functional_token_index, tu=utils.get_trans_utils(args))
    full_vocab = Vocabulary('full', func_token_index=functional_token_index, tu=utils.get_trans_utils(args))
    for v in text_hist:
        text_vocab.index_token(v, True, text_hist[v])
    text_vocab_path = get_vocab_path(args, 'nl')
    text_vocab.save_to_disk(text_vocab_path)

    program_vocab = Vocabulary('program', func_token_index=functional_token_index)
    for v in program_hist:
        program_vocab.index_token(v, True, program_hist[v])
    program_vocab_path = get_vocab_path(args, 'cm')
    program_vocab.save_to_disk(program_vocab_path)

    # Combine text and program vocabularies
    full_vocab.merge_with(text_vocab)
    full_vocab.merge_with(program_vocab)
    full_vocab_path = get_vocab_path(args, 'full')
    full_vocab.save_to_disk(full_vocab_path)

    world_vocab_path = get_vocab_path(args, 'world')
    world_vocab.save_to_disk(world_vocab_path)