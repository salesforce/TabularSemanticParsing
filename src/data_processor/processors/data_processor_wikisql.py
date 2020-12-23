"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Preprocessing WikiSQL examples released by Zhong et al. 2017.
"""
import numpy as np
import src.utils.utils as utils
import scipy.sparse as ssp

import moz_sp.sql_tokenizer as sql_tokenizer
from src.data_processor.processor_utils import get_table_aware_transformer_encoder_inputs
from src.data_processor.processor_utils import get_transformer_output_value_mask
from src.data_processor.processor_utils import Text2SQLExample
from src.data_processor.processor_utils import START_TOKEN, EOS_TOKEN, NUM_TOKEN, STR_TOKEN
from src.data_processor.vocab_utils import functional_tokens
import src.data_processor.tokenizers as tok
import src.data_processor.vectorizers as vec
from src.utils.utils import BRIDGE, is_number

RESERVED_TOKEN_TYPE = sql_tokenizer.RESERVED_TOKEN


def preprocess_example(split, example, args, parsed_programs, text_tokenize, program_tokenize, post_process,
                       table_utils, schema_graph, vocabs, verbose=False):
    tu = table_utils
    text_vocab = vocabs['text']
    program_vocab = vocabs['program']

    def get_memory_values(features, raw_text, args):
        if args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased'):
            return utils.restore_feature_case(features, raw_text, tu)
        else:
            return features

    def get_text_schema_adjacency_matrix(text_features, s_M):
        schema_size = s_M.shape[0]
        text_size = len(text_features)
        full_size = schema_size + text_size
        M = ssp.lil_matrix((full_size, full_size), dtype=np.int)
        M[-schema_size:, -schema_size:] = s_M
        return M

    # sanity check
    ############################
    query_oov = False
    denormalized = False
    schema_truncated = False
    token_restored = True
    ############################

    # Text feature extraction and set program ground truth list
    if isinstance(example, Text2SQLExample):
        if args.pretrained_transformer:
            text_features = text_tokenize(example.text)
            text_tokens, token_starts, token_ends = get_memory_values(text_features, example.text, args)
            if not token_starts:
                token_restored = False
        else:
            text_tokens = text_tokenize(example.text, functional_tokens)
            text_features = [t.lower() for t in text_tokens]
        example.text_tokens = text_features
        example.text_ptr_values = text_tokens
        example.text_token_starts = token_starts
        example.text_token_ends = token_ends
        example.text_ids = vec.vectorize(text_features, text_vocab)
        example.text_ptr_input_ids = vec.vectorize(text_features, text_vocab)
        program_list = example.program_list
        example.values = [(schema_graph.get_field(cond[0]).signature, cond[2])
                          for cond in example.program_ast_list_[0]['conds']
                          if (isinstance(cond[2], str) and not is_number(cond[2]))]
    else:
        text_tokens = example.example.text_ptr_values
        text_features = example.example.text_tokens
        program_list = example.example.program_list

    # Schema feature extraction
    if args.model_id in [BRIDGE]:
        question_encoding = example.text if args.use_picklist else None
        tables = sorted([schema_graph.get_table_id(t_name) for t_name in example.gt_table_names]) \
            if args.use_oracle_tables else None
        table_po, field_po = schema_graph.get_schema_perceived_order(tables)
        schema_features, matched_values = schema_graph.get_serialization(
            tu, flatten_features=True, table_po=table_po, field_po=field_po,
            use_typed_field_markers=args.use_typed_field_markers, use_graph_encoding=args.use_graph_encoding,
            question_encoding=question_encoding, top_k_matches=args.top_k_picklist_matches,
            num_values_per_field=args.num_values_per_field, no_anchor_text=args.no_anchor_text)
        example.matched_values = matched_values
        example.input_tokens, example.input_ptr_values, num_excluded_tables, num_excluded_fields = \
            get_table_aware_transformer_encoder_inputs(text_tokens, text_features, schema_features, table_utils)
        schema_truncated = (num_excluded_fields > 0)
        num_included_nodes = schema_graph.get_num_perceived_nodes(table_po) + 1 - num_excluded_tables - num_excluded_fields
        example.ptr_input_ids = vec.vectorize(example.input_tokens, text_vocab)
        if args.read_picklist:
            example.transformer_output_value_mask, value_features, value_tokens = \
                get_transformer_output_value_mask(example.input_tokens, matched_values, tu)
        example.primary_key_ids = schema_graph.get_primary_key_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.foreign_key_ids = schema_graph.get_foreign_key_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.field_type_ids = schema_graph.get_field_type_ids(num_included_nodes, table_po=table_po, field_po=field_po)
        example.table_masks = schema_graph.get_table_masks(num_included_nodes, table_po=table_po, field_po=field_po)
        example.field_table_pos = schema_graph.get_field_table_pos(num_included_nodes, table_po=table_po, field_po=field_po)
        example.schema_M = schema_graph.adj_matrix
        example.M = get_text_schema_adjacency_matrix(text_features, example.schema_M)
    else:
        num_included_nodes = schema_graph.num_nodes

    # Value copy feature extraction
    if args.read_picklist:
        constant_memory_features = text_features + value_features
        constant_memory = text_tokens + value_tokens
        example.text_ptr_values = constant_memory
    else:
        constant_memory_features = text_features
    constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(constant_memory_features, program_vocab)
    if isinstance(example, Text2SQLExample):
        example.text_ptr_value_ids = constant_ptr_value_ids
    example.ptr_value_ids = constant_ptr_value_ids + [program_vocab.size + len(constant_memory_features) + x
                                                      for x in range(num_included_nodes)]

    if not args.leaderboard_submission:
        for j, program in enumerate(program_list):
            if isinstance(example, Text2SQLExample):
                # Model II. Bridge output
                program_singleton_field_tokens, program_singleton_field_token_types = \
                    tok.wikisql_struct_to_tokens(example.program_ast_, schema_graph, tu)
                program_singleton_field_tokens = [START_TOKEN] + program_singleton_field_tokens + [EOS_TOKEN]
                program_singleton_field_token_types = \
                    [RESERVED_TOKEN_TYPE] + program_singleton_field_token_types + [RESERVED_TOKEN_TYPE]
                example.program_singleton_field_tokens_list.append(program_singleton_field_tokens)
                example.program_singleton_field_token_types_list.append(program_singleton_field_token_types)
                program_singleton_field_input_ids = vec.vectorize_singleton(
                    program_singleton_field_tokens, program_singleton_field_token_types, program_vocab)
                example.program_singleton_field_input_ids_list.append(program_singleton_field_input_ids)
            else:
                # Model II. Bridge output
                example.program_singleton_field_input_ids_list.append(
                    example.example.program_singleton_field_input_ids_list[j])
                program_singleton_field_tokens = example.example.program_singleton_field_tokens_list[j]
                program_singleton_field_token_types = example.example.program_singleton_field_token_types_list[j]

            program_field_ptr_value_ids = vec.vectorize_field_ptr_out(program_singleton_field_tokens,
                                                                      program_singleton_field_token_types,
                                                                      program_vocab,
                                                                      constant_unique_input_ids,
                                                                      max_memory_size=len(constant_memory_features),
                                                                      schema=schema_graph,
                                                                      num_included_nodes=num_included_nodes)
            example.program_text_and_field_ptr_value_ids_list.append(program_field_ptr_value_ids)

            table_ids = [schema_graph.get_table_id(table_name) for table_name in example.gt_table_names_list[j]]
            example.table_ids_list.append(table_ids)
            assert ([schema_graph.get_table(x).name for x in table_ids] == example.gt_table_names)

            # sanity check
            ############################
            #   NL+Schema pointer output contains tokens that does not belong to any of the following categories
            if verbose:
                if program_vocab.unk_id in program_field_ptr_value_ids:
                    unk_indices = [i for i, x in enumerate(program_field_ptr_value_ids) if x == program_vocab.unk_id]
                    print('OOV II: {}'.format(' '.join([program_singleton_field_tokens[i] for i in unk_indices])))
                    example.pretty_print(schema=schema_graph,
                                         de_vectorize_ptr=vec.de_vectorize_ptr,
                                         de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                         rev_vocab=program_vocab,
                                         post_process=post_process,
                                         use_table_aware_te=(args.model_id in [BRIDGE]))
                    query_oov = True
            if program_vocab.unk_field_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [BRIDGE]))
            if program_vocab.unk_table_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [BRIDGE]))
            ############################

            # Store the ground truth queries after preprocessing to run a relaxed evaluation or
            # to evaluate with partial queries
            if split == 'dev':
                input_tokens = text_tokens
                if args.model_id in [BRIDGE]:
                    _p = vec.de_vectorize_field_ptr(program_field_ptr_value_ids, program_vocab, input_tokens,
                                                    schema=schema_graph, post_process=post_process)
                else:
                    _p = program
                example.gt_program_list.append(_p)

            # sanity check
            ############################
            # try:
            #     assert(equal_ignoring_trivial_diffs(_p, program.lower(), verbose=True))
            # except Exception:
            #     print('_p:\t\t{}'.format(_p))
            #     print('program:\t{}'.format(program))
            #     print()
            #     import pdb
            #     pdb.set_trace()
            ############################

        example.run_unit_tests()

    return query_oov, denormalized, schema_truncated, token_restored
