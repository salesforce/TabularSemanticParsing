#-*- coding: utf-8 -*-

"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Encoder-decoder learning framework.
"""

import random
from tqdm import tqdm

import torch

import moz_sp
from src.common.learn_framework import LFramework
from src.common.nn_modules import MaskedCrossEntropyLoss
import src.common.ops as ops
import src.data_processor.data_loader as data_loader
from src.data_processor.processor_utils import get_table_aware_transformer_encoder_inputs, \
    get_transformer_output_value_mask
from src.data_processor.processor_utils import SPIDER, WIKISQL
from src.data_processor.schema_graph import DUMMY_REL
import src.data_processor.tokenizers as tok
import src.data_processor.vectorizers as vec
from src.semantic_parser.ensemble import ensemble_beam_search
from src.semantic_parser.seq2seq import Seq2Seq
from src.semantic_parser.seq2seq_ptr import PointerGenerator
from src.semantic_parser.bridge import Bridge
import src.eval.eval_tools as eval_tools
import src.eval.spider.evaluate as spider_eval_tools
from src.eval.wikisql.lib.query import Query
from src.utils.utils import SEQ2SEQ, SEQ2SEQ_PG, BRIDGE


class EncoderDecoderLFramework(LFramework):

    def __init__(self, args):
        super().__init__(args)
        vocabs = data_loader.load_vocabs(args)
        self.in_vocab = vocabs['text']
        self.out_vocab = vocabs['program']

        # Construct NN model
        if self.model_id == BRIDGE:
            self.mdl = Bridge(args, self.in_vocab, self.out_vocab)
        elif self.model_id == SEQ2SEQ_PG:
            self.mdl = PointerGenerator(args, self.in_vocab, self.out_vocab)
        elif self.model_id == SEQ2SEQ:
            self.mdl = Seq2Seq(args, self.in_vocab, self.out_vocab)
        else:
            raise NotImplementedError

        # Specify loss function
        if self.args.loss == 'cross_entropy':
            self.loss_fun = MaskedCrossEntropyLoss(self.mdl.out_vocab.pad_id)
        else:
            raise NotImplementedError

        # Optimizer
        self.define_optimizer()
        self.define_lr_scheduler()

        # Post-process
        _, _, self.output_post_process, _ = tok.get_tokenizers(args)

        print('{} module created'.format(self.model))

    def get_text_masks(self, encoder_input_ids):
        return encoder_input_ids[1]

    def get_schema_masks(self, encoder_input_ptr_ids, transformer_output_masks=None):
        if transformer_output_masks is not None:
            encoder_input_ptr_ids, _ = ops.batch_binary_lookup(
                encoder_input_ptr_ids, transformer_output_masks, pad_value=self.in_vocab.pad_id)
        if self.args.use_typed_field_markers:
            schema_masks = (encoder_input_ptr_ids == self.tu.table_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.text_field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.number_field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.time_field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.boolean_field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.other_field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.asterisk_marker_id)
        else:
            schema_masks = (encoder_input_ptr_ids == self.tu.table_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.field_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.primary_key_marker_id) | \
                           (encoder_input_ptr_ids == self.tu.asterisk_marker_id)
        return schema_masks

    def loss(self, formatted_batch):
        outputs = self.forward(formatted_batch)
        if self.model_id in [SEQ2SEQ_PG, BRIDGE]:
            decoder_ptr_value_ids, _ = formatted_batch[4]
            left_shift_targets = ops.left_shift_pad(decoder_ptr_value_ids, self.out_vocab.pad_id)
        else:
            decoder_input_ids, _ = formatted_batch[1]
            left_shift_targets = ops.left_shift_pad(decoder_input_ids, self.out_vocab.pad_id)
        loss = self.loss_fun(outputs, left_shift_targets)
        loss /= self.num_accumulation_steps
        return loss

    def forward(self, formatted_batch, model_ensemble=None):
        encoder_input_ids = formatted_batch[0]
        decoder_input_ids = formatted_batch[1][0] if self.training else None
        if self.model_id in [SEQ2SEQ_PG, BRIDGE]:
            encoder_ptr_input_ids = formatted_batch[2]
            encoder_ptr_value_ids, _ = formatted_batch[3]
            decoder_ptr_value_ids = formatted_batch[4][0] if self.training else None
            text_masks = self.get_text_masks(encoder_input_ids)
            if self.model_id in [BRIDGE]:
                transformer_output_value_masks = formatted_batch[5][0]
                schema_masks = self.get_schema_masks(encoder_ptr_input_ids[0])
                schema_memory_masks = formatted_batch[6][0]
                feature_ids = formatted_batch[8]
                if model_ensemble:
                    assert(not self.training)
                    outputs = ensemble_beam_search(model_ensemble, encoder_ptr_input_ids, encoder_ptr_value_ids,
                                                   text_masks, schema_masks, feature_ids, None,
                                                   transformer_output_value_masks, schema_memory_masks)
                else:
                    outputs = self.mdl(encoder_ptr_input_ids, encoder_ptr_value_ids,
                                       text_masks, schema_masks, feature_ids,
                                       transformer_output_value_masks=transformer_output_value_masks,
                                       schema_memory_masks=schema_memory_masks,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_ptr_value_ids=decoder_ptr_value_ids)
            else:
                outputs = self.mdl(encoder_ptr_input_ids, encoder_ptr_value_ids, text_masks,
                                   decoder_input_ids=decoder_input_ids,
                                   decoder_ptr_value_ids=decoder_ptr_value_ids)
        elif self.model_id == SEQ2SEQ:
            outputs = self.mdl(encoder_input_ids, decoder_input_ids)
        else:
            raise NotImplementedError
        return outputs

    def inference(self, examples, decode_str_output=True, restore_clause_order=False, pred_restored_cache=None,
                  check_schema_consistency_=True, engine=None, inline_eval=False, model_ensemble=None, verbose=False):
        # sanity check
        if self.args.leaderboard_submission or self.args.demo:
            assert (not verbose and not inline_eval and not self.args.use_oracle_tables)

        pred_list, pred_score_list, pred_decoded_list, pred_decoded_score_list = [], [], [], []
        if restore_clause_order:
            if pred_restored_cache is None:
                pred_restored_cache = dict()
        if self.save_vis:
            text_ptr_weights_vis, pointer_vis = [], []

        num_error_cases = 0
        for batch_start_id in tqdm(range(0, len(examples), self.dev_batch_size)):
            mini_batch = examples[batch_start_id:batch_start_id + self.dev_batch_size]
            formatted_batch = self.format_batch(mini_batch)
            outputs = self.forward(formatted_batch, model_ensemble)
            if self.model_id in [SEQ2SEQ_PG, BRIDGE]:
                preds, pred_scores, text_p_pointers, text_ptr_weights, seq_len = outputs
                text_p_pointers.unsqueeze_(2)
                p_pointers = torch.cat([1 - text_p_pointers, text_p_pointers], dim=2)
            elif self.model_id == SEQ2SEQ:
                preds, pred_scores, text_ptr_weights, seq_len = outputs
                p_pointers = None
            else:
                raise NotImplementedError

            pred_list.append(preds)
            pred_score_list.append(pred_scores)
            if decode_str_output or verbose:
                for i in range(len(mini_batch)):
                    example = mini_batch[i]
                    db_name = example.db_name
                    schema = self.schema_graphs[db_name]
                    table_po, field_po = None, None
                    if self.args.use_oracle_tables:
                        # TODO: The implementation below is incorrect.
                        if self.args.num_random_tables_added > 0:
                            table_po, field_po = formatted_batch[-1][i]

                    exp_output_strs, exp_output_scores, exp_seq_lens, exp_correct = [], [], [], []

                    if inline_eval:
                        if example.dataset_id == SPIDER:
                            gt_program_list = example.program_list
                            gt_program_ast = example.program_ast_list_[0] \
                                if example.program_ast_list_ else example.program
                            hardness = spider_eval_tools.Evaluator().eval_hardness(
                                gt_program_ast, db_dir=self.args.db_dir, db_name=example.db_name)
                        elif example.dataset_id == WIKISQL:
                            gt_program_list = example.program_ast_list_
                        else:
                            raise NotImplementedError
                        if example.dataset_id == WIKISQL:
                            hardness = 'easy'

                    if self.decoding_algorithm == 'beam-search':
                        for j in range(self.beam_size):
                            beam_id = i * self.beam_size + j
                            post_processed_output = self.post_process_nn_output(
                                beam_id, example.dataset_id, example, preds, schema, text_ptr_weights, p_pointers,
                                table_po=table_po, field_po=field_po, verbose=verbose)
                            if post_processed_output:
                                pred_sql = post_processed_output[0]
                                # print('{}\t{}'.format(pred_sql, float(pred_scores[beam_id])))
                                if restore_clause_order:
                                    if pred_restored_cache and db_name in pred_restored_cache and \
                                            pred_sql in pred_restored_cache[db_name]:
                                        restored_pred, grammatical, schema_consistent = pred_restored_cache[db_name][pred_sql]
                                    else:
                                        restored_pred, grammatical, schema_consistent = moz_sp.restore_clause_order(
                                            pred_sql, schema, check_schema_consistency_=check_schema_consistency_,
                                            verbose=verbose)
                                        if pred_restored_cache and check_schema_consistency_:
                                            # TODO: we don't cache the results when check_schema_consistency_ is off to
                                            # avoid logging false negatives
                                            if db_name not in pred_restored_cache:
                                                pred_restored_cache[db_name] = dict()
                                            pred_restored_cache[db_name][pred_sql] = restored_pred, grammatical, \
                                                                                     schema_consistent
                                    if check_schema_consistency_ and not schema_consistent:
                                        restored_pred = None
                                    pred_sql = restored_pred
                                else:
                                    if check_schema_consistency_:
                                        if not moz_sp.check_schema_consistency(
                                                pred_sql, schema, in_execution_order=self.args.process_sql_in_execution_order):
                                            pred_sql = None
                                if pred_sql and self.args.execution_guided_decoding:
                                    assert(engine is not None)
                                    try:
                                        pred_query = Query.from_dict(pred_sql, ordered=False)
                                        pred_ex = engine.execute_query(example.db_name, pred_query, lower=True)
                                        if not pred_ex:
                                            pred_sql = None
                                    except Exception:
                                        pred_sql = None
                            else:
                                pred_sql = None
                            # if not pred_sql:
                            #     pred_sql = self.get_dummy_prediction(schema)
                            if pred_sql:
                                exp_output_strs.append(pred_sql)
                                exp_output_scores.append(float(pred_scores[beam_id]))
                                exp_seq_lens.append(int(seq_len[beam_id]))
                                if self.save_vis:
                                    self.save_vis_parameters(post_processed_output, text_ptr_weights_vis, pointer_vis)
                                if inline_eval:
                                    results = eval_tools.eval_prediction(
                                        pred=pred_sql,
                                        gt_list=gt_program_list,
                                        dataset_id=example.dataset_id,
                                        db_name=example.db_name,
                                        in_execution_order=(self.args.process_sql_in_execution_order and
                                                            not restore_clause_order))
                                    correct, _, _ = results
                                    exp_correct.append(correct)
                                    correct_ = correct[1] if isinstance(correct, tuple) else correct
                                    if correct_:
                                        break
                    else:
                        raise NotImplementedError
                    num_preds = len(exp_output_strs)
                    pred_decoded_list.append(exp_output_strs)
                    pred_decoded_score_list.append(exp_output_scores[:num_preds])
                    if verbose:
                        predictions = zip(exp_output_strs, exp_output_scores, exp_seq_lens, exp_correct)
                        is_error_case = self.print_predictions(batch_start_id + i, example, hardness, predictions, schema)
                        if is_error_case:
                            num_error_cases += 1
                            print('Error Case {}'.format(num_error_cases))
                            print()
                            # if num_error_cases == 50:
                            #     import sys
                            #     sys.exit()
                    if not pred_decoded_list[-1] and not self.args.demo:
                        pred_decoded_list[-1].append(self.get_dummy_prediction(schema))
                        pred_decoded_score_list[-1].append(-ops.HUGE_INT)

        out_dict = dict()
        out_dict['preds'] = ops.pad_and_cat(pred_list, self.out_vocab.pad_id)
        out_dict['pred_scores'] = torch.cat(pred_score_list)
        if decode_str_output:
            out_dict['pred_decoded'] = pred_decoded_list
            out_dict['pred_decoded_scores'] = pred_decoded_score_list
        if restore_clause_order:
            out_dict['pred_restored_cache'] = pred_restored_cache

        if self.save_vis:
            vis_dict = dict()
            vis_dict['text_attention_vis'] = text_ptr_weights_vis
            vis_dict['text_pointer_vis'] = pointer_vis
            for key in vis_dict:
                if key.endswith('_vis'):
                    if key.endswith('_attention_vis'):
                        attn_target_label = key.split('_')[0]
                        self.vis_writer.save_cross_attention(vis_dict[key], attn_target_label)
                    if key.endswith('_pointer_vis'):
                        self.vis_writer.save_pointer(vis_dict[key], 'all')

        return out_dict

    def format_batch(self, mini_batch):

        def get_decoder_input_ids():
            if self.training:
                if self.model_id in [BRIDGE]:
                    X = [exp.program_singleton_field_input_ids for exp in mini_batch]
                else:
                    X = [exp.program_input_ids for exp in mini_batch]
                return ops.pad_batch(X, self.mdl.out_vocab.pad_id)
            else:
                return None

        def get_encoder_attn_mask(table_names, table_masks):
            schema_pos = [schema_graph.get_schema_pos(table_name) for table_name in table_names]
            encoder_attn_mask = [1 for _ in range(exp.num_text_tokens)]
            # asterisk marker
            encoder_attn_mask.append(1)
            is_selected_table = False
            for j in range(1, len(table_masks)):
                if j in schema_pos:
                    encoder_attn_mask.append(1)
                    is_selected_table = True
                elif table_masks[j] == 1:
                    # mask current table
                    encoder_attn_mask.append(0)
                    is_selected_table = False
                else:
                    if is_selected_table:
                        encoder_attn_mask.append(1)
                    else:
                        encoder_attn_mask.append(0)
            return encoder_attn_mask

        super().format_batch(mini_batch)
        encoder_input_ids = ops.pad_batch([exp.text_ids for exp in mini_batch], self.mdl.in_vocab.pad_id)
        decoder_input_ids = get_decoder_input_ids()

        table_samples = []

        if self.model_id == SEQ2SEQ:
            return encoder_input_ids, decoder_input_ids
        elif self.model_id in [BRIDGE]:
            encoder_ptr_input_ids, encoder_ptr_value_ids, decoder_ptr_value_ids = [], [], []
            primary_key_ids, foreign_key_ids, field_type_ids, table_masks, table_positions, table_field_scopes, \
                field_table_pos, transformer_output_value_masks, schema_memory_masks = [], [], [], [], [], [], [], [], []
            for exp in mini_batch:
                schema_graph = self.schema_graphs.get_schema(exp.db_id)
                # exp.pretty_print(example_id=0,
                #                  schema=schema_graph,
                #                  de_vectorize_ptr=vec.de_vectorize_ptr,
                #                  de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                #                  rev_vocab=self.out_vocab,
                #                  post_process=self.output_post_process,
                #                  use_table_aware_te=(self.model_id in [BRIDGE]))
                # import pdb
                # pdb.set_trace()
                if self.training:
                    # Compute schema layout
                    if exp.gt_table_names_list:
                        gt_tables = set([schema_graph.get_table_id(t_name) for t_name in exp.gt_table_names])
                    else:
                        gt_table_names = [token for token, t in
                                          zip(exp.program_singleton_field_tokens, exp.program_singleton_field_token_types) if t == 0]
                        gt_tables = set([schema_graph.get_table_id(t_name) for t_name in gt_table_names])
                    # [Hack] Baseball database has a complex schema which does not fit the input size of BERT. We select
                    # the ground truth tables and randomly add a few other tables for training.
                    if schema_graph.name.startswith('baseball'):
                        tables = list(gt_tables)
                        tables += random.sample([i for i in range(schema_graph.num_tables) if i not in gt_tables],
                                                 k=min(random.randint(1, 7), schema_graph.num_tables - len(gt_tables)))
                    else:
                        tables = list(range(schema_graph.num_tables))
                    if self.args.table_shuffling:
                        table_to_drop = random.choice(tables)
                        if table_to_drop not in gt_tables:
                            if random.uniform(0, 1) < 0.3:
                                tables = [x for x in tables if x != table_to_drop]
                        table_po, field_po = schema_graph.get_schema_perceived_order(
                            tables, random_table_order=True, random_field_order=self.args.random_field_order)
                    else:
                        table_po, field_po = schema_graph.get_schema_perceived_order(
                            tables, random_table_order=False, random_field_order=self.args.random_field_order)

                    # Schema feature extraction
                    question_encoding = exp.text if self.args.use_picklist else None
                    schema_features, matched_values = schema_graph.get_serialization(
                        self.tu, flatten_features=True, table_po=table_po, field_po=field_po,
                        use_typed_field_markers=self.args.use_typed_field_markers,
                        use_graph_encoding=self.args.use_graph_encoding,
                        question_encoding = question_encoding,
                        top_k_matches=self.args.top_k_picklist_matches,
                        num_values_per_field=self.args.num_values_per_field,
                        no_anchor_text=self.args.no_anchor_text,
                        verbose=False)
                    ptr_input_tokens, ptr_input_values, num_excluded_tables, num_excluded_fields = \
                        get_table_aware_transformer_encoder_inputs(
                            exp.text_ptr_values, exp.text_tokens, schema_features, self.tu)
                    assert(len(ptr_input_tokens) <= self.tu.tokenizer.max_len)
                    if num_excluded_fields > 0:
                        print('Warning: training input truncated')
                    num_included_nodes = schema_graph.get_num_perceived_nodes(tables) + 1 \
                                         - num_excluded_tables - num_excluded_fields
                    encoder_ptr_input_ids.append(self.tu.tokenizer.convert_tokens_to_ids(ptr_input_tokens))
                    if self.args.read_picklist:
                        exp.transformer_output_value_mask, value_features, value_tokens = \
                            get_transformer_output_value_mask(ptr_input_tokens, matched_values, self.tu)
                        transformer_output_value_masks.append(exp.transformer_output_value_mask)
                    primary_key_ids.append(schema_graph.get_primary_key_ids(num_included_nodes, table_po, field_po))
                    foreign_key_ids.append(schema_graph.get_foreign_key_ids(num_included_nodes, table_po, field_po))
                    field_type_ids.append(schema_graph.get_field_type_ids(num_included_nodes, table_po, field_po))
                    table_masks.append(schema_graph.get_table_masks(num_included_nodes, table_po, field_po))

                    # Value copy feature extraction
                    if self.args.read_picklist:
                        constant_memory_features = exp.text_tokens + value_features
                        constant_memory = exp.text_ptr_values + value_tokens
                        exp.text_ptr_values = constant_memory
                    else:
                        constant_memory_features = exp.text_tokens
                    constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(
                        constant_memory_features, self.out_vocab)
                    encoder_ptr_value_ids.append(
                        constant_ptr_value_ids + [self.out_vocab.size + len(constant_memory_features) + x
                                                  for x in range(num_included_nodes)])
                    program_field_ptr_value_ids = \
                        vec.vectorize_field_ptr_out(exp.program_singleton_field_tokens,
                                                    exp.program_singleton_field_token_types,
                                                    self.out_vocab, constant_unique_input_ids,
                                                    max_memory_size=len(constant_memory_features),
                                                    schema=schema_graph,
                                                    num_included_nodes=num_included_nodes)
                    decoder_ptr_value_ids.append(program_field_ptr_value_ids)
                else:
                    encoder_ptr_input_ids = [exp.ptr_input_ids for exp in mini_batch]
                    encoder_ptr_value_ids = [exp.ptr_value_ids for exp in mini_batch]
                    decoder_ptr_value_ids = [exp.program_text_and_field_ptr_value_ids for exp in mini_batch] \
                        if self.training else None
                    primary_key_ids = [exp.primary_key_ids for exp in mini_batch]
                    foreign_key_ids = [exp.foreign_key_ids for exp in mini_batch]
                    field_type_ids = [exp.field_type_ids for exp in mini_batch]
                    table_masks = [exp.table_masks for exp in mini_batch]
                    # TODO: here we assume that all nodes in the schema graph are included
                    table_pos, table_field_scope = schema_graph.get_table_scopes(schema_graph.num_nodes)
                    table_positions.append(table_pos)
                    table_field_scopes.append(table_field_scope)
                    if self.args.read_picklist:
                        transformer_output_value_masks.append(exp.transformer_output_value_mask)

            encoder_ptr_input_ids = ops.pad_batch(encoder_ptr_input_ids, self.mdl.in_vocab.pad_id)
            encoder_ptr_value_ids = ops.pad_batch(encoder_ptr_value_ids, self.mdl.in_vocab.pad_id)
            schema_memory_masks = ops.pad_batch(schema_memory_masks, pad_id=0) \
                if (self.args.use_pred_tables and not self.training) else (None, None)
            decoder_ptr_value_ids = ops.pad_batch(decoder_ptr_value_ids, self.mdl.out_vocab.pad_id) \
                if self.training else None
            primary_key_ids = ops.pad_batch(primary_key_ids, self.mdl.in_vocab.pad_id)
            foreign_key_ids = ops.pad_batch(foreign_key_ids, self.mdl.in_vocab.pad_id)
            field_type_ids = ops.pad_batch(field_type_ids, self.mdl.in_vocab.pad_id)
            table_masks = ops.pad_batch(table_masks, pad_id=0)
            transformer_output_value_masks = ops.pad_batch(transformer_output_value_masks, pad_id=0, dtype=torch.uint8) \
                if self.args.read_picklist else (None, None)
            if not self.training:
                table_positions = ops.pad_batch(table_positions, pad_id=-1) \
                    if self.args.process_sql_in_execution_order else (None, None)
                table_field_scopes = ops.pad_batch_2D(table_field_scopes, pad_id=0) \
                    if self.args.process_sql_in_execution_order else (None, None)
            graphs = None
            return encoder_input_ids, decoder_input_ids, encoder_ptr_input_ids, encoder_ptr_value_ids, \
                   decoder_ptr_value_ids, transformer_output_value_masks, schema_memory_masks, graphs, \
                   (primary_key_ids, foreign_key_ids, field_type_ids, table_masks, table_positions,
                    table_field_scopes, field_table_pos), table_samples
        elif self.model_id in [SEQ2SEQ_PG]:
            encoder_ptr_input_ids = [exp.ptr_input_ids for exp in mini_batch]
            encoder_ptr_value_ids = [exp.ptr_value_ids for exp in mini_batch]
            decoder_ptr_value_ids = [exp.program_text_ptr_value_ids for exp in mini_batch]
            encoder_ptr_input_ids = ops.pad_batch(encoder_ptr_input_ids, self.mdl.in_vocab.pad_id)
            encoder_ptr_value_ids = ops.pad_batch(encoder_ptr_value_ids, self.mdl.in_vocab.pad_id)
            decoder_ptr_value_ids = ops.pad_batch(decoder_ptr_value_ids, self.mdl.out_vocab.pad_id)
            return encoder_input_ids, decoder_input_ids, encoder_ptr_input_ids, encoder_ptr_value_ids, \
                   decoder_ptr_value_ids
        else:
            raise NotImplementedError

    def post_process_nn_output(self, idx, dataset_id, example, decoder_outputs, schema=None,
                               text_ptr_weights=None, p_pointers=None, table_po=None, field_po=None, verbose=False):
        decoder_output = ops.var_to_numpy(decoder_outputs[idx])
        if dataset_id == WIKISQL:
            try:
                output_str = tok.wikisql_vec_to_struct(decoder_output, self.out_vocab,
                                                       example.text_ptr_values,
                                                       example.text_token_starts,
                                                       example.text_token_ends,
                                                       example.text, self.tu)
            except Exception:
                output_str = None
        else:
            out_tokens = self.de_vectorize(decoder_output, self.out_vocab, example.text_ptr_values, schema,
                                           table_po=table_po, field_po=field_po, return_tokens=True)
            if self.args.no_join_condition:
                assert(schema is not None)
                try:
                    out_tokens = moz_sp.add_join_condition(out_tokens, schema)
                except ValueError as e:
                    if verbose:
                        print(str(e))
                    return None
            output_str = self.output_post_process(out_tokens)
            output_str = output_str.replace(self.out_vocab.num_token, '1').replace('<NUM>', '1')
            output_str = output_str.replace(self.out_vocab.str_token, '"string"').replace('<STRING>', "string")
        if self.save_vis:
            text_ptr_weights_vis = (example.text_ptr_values, out_tokens, ops.var_to_numpy(text_ptr_weights[idx]))
            if self.model_id in [SEQ2SEQ_PG, BRIDGE]:
                pointer_vis = (out_tokens, ops.var_to_numpy(p_pointers[idx]))
                return output_str, text_ptr_weights_vis, pointer_vis
            else:
                return output_str, text_ptr_weights_vis
        else:
            return output_str,

    def get_dummy_prediction(self, schema):
        """
        Return a dummy SQL query given a specific database.
        """
        return 'SELECT * FROM {}'.format(schema.table_rev_index[0].name)

    def de_vectorize(self, p_cpu, out_vocab, input_ptr_values, schema=None, table_po=None, field_po=None,
                     return_tokens=False):
        # convert output prediction vector into a human readable string
        if self.model_id in [SEQ2SEQ_PG]:
            return vec.de_vectorize_ptr(p_cpu, out_vocab, memory=input_ptr_values,
                                        post_process=self.output_post_process, return_tokens=return_tokens)
        elif self.model_id in [BRIDGE]:
            return vec.de_vectorize_field_ptr(p_cpu, out_vocab, memory=input_ptr_values, schema=schema,
                                              table_po=table_po, field_po=field_po,
                                              post_process=self.output_post_process, return_tokens=return_tokens)
        elif self.model_id == SEQ2SEQ:
            return vec.de_vectorize(p_cpu, out_vocab, post_process=self.output_post_process,
                                    return_tokens=return_tokens)
        else:
            raise NotImplementedError

    def print_predictions(self, example_id, example, hardness, predictions, schema):
        inspect_error_cases = False
        output_strs = []

        for i, prediction in enumerate(predictions):
            pred_sql, pred_score, sql_len, correct = prediction
            if isinstance(correct, tuple):
                correct = correct[1]
            correct_badge = '[CORRE]' if correct else '[WRONG]'
            if example.dataset_id == WIKISQL:
                pred_sql = str(pred_sql)
            output_strs.append('{} {} Pred {}:\t{} ({:.3f}) (length={})'.format(
                correct_badge, '[{}]'.format(hardness), i, pred_sql.encode('utf-8'), float(pred_score), int(sql_len)))
            if i == 0 and not correct:
                inspect_error_cases = True

        if not output_strs:
            output_strs.append('{} {} Pred {}:\t{} ({:.3f})'.format('[WRONG]', hardness, 0, 'No valid output!', 0))
            inspect_error_cases = True

        if not output_strs or output_strs[0].startswith('[WRONG]'): # and hardness == 'medium':
            example.pretty_print(example_id=example_id,
                                 schema=schema,
                                 de_vectorize_ptr=vec.de_vectorize_ptr,
                                 de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                 rev_vocab=self.out_vocab,
                                 post_process=self.output_post_process,
                                 use_table_aware_te=(self.model_id in [BRIDGE]))
            for output_str in output_strs:
                print(output_str)

        return inspect_error_cases

    def save_vis_parameters(self, outputs, text_ptr_weights_vis, pointer_vis):
        text_ptr_weights_vis.append(outputs[1])
        if self.model_id in [SEQ2SEQ_PG, BRIDGE]:
            pointer_vis.append(outputs[2])
