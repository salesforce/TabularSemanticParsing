"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Codalab leaderboard submission.
"""
import src.data_processor.data_loader as data_loader
from src.data_processor.processors.data_processor_spider import preprocess_example
from src.data_processor.path_utils import get_checkpoint_path
import src.data_processor.schema_loader as schema_loader
from src.data_processor.sql.sql_reserved_tokens import sql_reserved_tokens, sql_reserved_tokens_revtok
import src.data_processor.tokenizers as tok
from src.data_processor.vocab_utils import functional_token_index, Vocabulary
from src.semantic_parser.learn_framework import EncoderDecoderLFramework
import src.utils.utils as utils

from src.parse_args import args

import torch
if not args.data_parallel:
    torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model ID
if args.predict_tables:
    args.model = args.model + '.pt'
args.model_id = utils.model_index[args.model]
assert(args.model_id is not None)


def inference(sp):
    text_tokenize, program_tokenize, post_process, table_utils = tok.get_tokenizers(args)
    schema_graphs = schema_loader.load_schema_graphs_spider(args.codalab_data_dir, 'spider', db_dir=args.codalab_db_dir)
    schema_graphs.lexicalize_graphs(
        tokenize=text_tokenize, normalized=(args.model_id in [utils.BRIDGE]))
    sp.schema_graphs = schema_graphs
    text_vocab = Vocabulary('text', func_token_index=functional_token_index, tu=table_utils)
    for v in table_utils.tokenizer.vocab:
        text_vocab.index_token(v, True, table_utils.tokenizer.convert_tokens_to_ids(v))
    program_vocab = sql_reserved_tokens if args.pretrained_transformer else sql_reserved_tokens_revtok
    vocabs = {
        'text': text_vocab,
        'program': program_vocab
    }
    examples = data_loader.load_data_split_spider(args.codalab_data_dir, 'dev', schema_graphs)
    print('{} {} examples loaded'.format(len(examples), 'dev'))

    for i, example in enumerate(examples):
        schema_graph = schema_graphs.get_schema(example.db_id)
        preprocess_example('dev', example, args, None, text_tokenize, program_tokenize, post_process, table_utils,
                           schema_graph, vocabs)
    print('{} {} examples processed'.format(len(examples), 'dev'))

    sp.load_checkpoint(get_checkpoint_path(args))
    sp.eval()

    out_dict = sp.inference(examples,
                            restore_clause_order=args.process_sql_in_execution_order,
                            check_schema_consistency_=True,
                            inline_eval=False,
                            verbose=False)

    assert(sp.args.prediction_path is not None)
    out_txt = sp.args.prediction_path
    with open(out_txt, 'w') as o_f:
        for pred_sql in out_dict['pred_decoded']:
            o_f.write('{}\n'.format(pred_sql[0]))
        print('Model predictions saved to {}'.format(out_txt))


def run_inference(args):
    if args.model in ['bridge',
                      'seq2seq',
                      'seq2seq.pg']:
        sp = EncoderDecoderLFramework(args)
    else:
        raise NotImplementedError

    sp.cuda()

    with torch.set_grad_enabled(False):
        inference(sp)


if __name__ == '__main__':
    run_inference(args)