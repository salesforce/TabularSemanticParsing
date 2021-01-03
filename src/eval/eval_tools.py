"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Compute SQL generation evaluation metrics.
"""

from src.data_processor.processor_utils import WIKISQL
import src.data_processor.vectorizers as vec
from src.eval.eval_utils import *
import src.eval.spider.evaluate as spider_eval_tools
from src.eval.wikisql.lib.dbengine import DBEngine
import src.eval.wikisql.evaluate as wikisql_eval_tools
from src.data_processor.processor_utils import SPIDER
from src.utils.utils import encode_str_list, list_to_hist


def get_exact_set_match_metrics(examples, pred_list, verbose=False, vocabs=None, schema_graphs=None, clauses=None):
    assert(len(examples) == len(pred_list))

    esm = ExactSetMatch(vocabs)

    metrics = {
        'select': 0.0,
        'groupBy': 0.0,
        'orderBy': 0.0,
        'from': 0.0,
        'where': 0.0,
        'having': 0.0,
        'limit': 0.0
    }
    for i, (example, example_pred) in enumerate(zip(examples, pred_list)):
        schema_graph = schema_graphs.get_schema(example.db_id)
        sql_correct, select_correct, group_by_correct, order_by_correct, from_correct, where_correct, having_correct, \
        limit_correct = \
            esm.eval_example(example_pred, example, verbose=verbose, example_id=i, schema_graph=schema_graph)
        if sql_correct:
            metrics['sql'] += 1.0
        if select_correct:
            metrics['select'] += 1.0
        if group_by_correct:
            metrics['groupBy'] += 1.0
        if order_by_correct:
            metrics['orderBy'] += 1.0
        if from_correct:
            metrics['from'] += 1.0
        if where_correct:
            metrics['where'] += 1.0
        if having_correct:
            metrics['having'] += 1.0
        if limit_correct:
            metrics['limit'] += 1.0

    avg_metrics = 0
    if clauses is None:
        clauses = metrics.keys()
    for key in clauses:
        metrics[key] /= len(examples)
        avg_metrics += metrics[key]
    avg_metrics /= (len(metrics) - 1)
    metrics['average'] = avg_metrics

    return metrics


class ExactSetMatch(object):

    def __init__(self, vocabs):
        self.field_vocab = vocabs['field']
        self.aggregation_ops = vocabs['aggregation_ops']
        self.arithmetic_ops = vocabs['arithmetic_ops']
        self.condition_ops = vocabs['condition_ops']
        self.logical_ops = vocabs['logical_ops']
        self.value_vocab = vocabs['value']

    def eval_example(self, example_pred, example, verbose=False, example_id=None, schema_graph=None):
        example_pred = self.pack_sql(example_pred)
        gt_ast = example.program_ast
        s_correct, g_correct, o_correct, f_correct, w_correct, h_correct, l_correct = self.match(example_pred, gt_ast)
        sql_correct = s_correct and g_correct and o_correct and f_correct and w_correct and h_correct and l_correct

        if not sql_correct and verbose:
            assert(schema_graph is not None)

            text = example.text
            text_tokens = vec.de_vectorize(example.text_ids, return_tokens=True)
            text_ptr_values = example.text_ptr_values
            print('Example {}'.format(example_id))
            print('NL:\t{}'.format(text.encode('utf-8')))
            print('NL tokens:\t{}'.format(encode_str_list(text_tokens, 'utf-8')))
            print('NL tokens (original):\t{}'.format(encode_str_list(text_ptr_values, 'utf-8')))
            print('NL tokens (recovered): {}'.format(
                vec.de_vectorize_ptr(example.text_ptr_value_ids, self.value_vocab, text_ptr_values).encode('utf-8'),
                return_tokens=True))

            for i, program in enumerate(example.program_list):
                print('Target {}'.format(i))
                print('- string: {}'.format(program.encode('utf-8')))

            badges = [
                get_badge(sql_correct),
                get_badge(s_correct),
                get_badge(g_correct),
                get_badge(o_correct),
                get_badge(f_correct),
                get_badge(w_correct),
                get_badge(h_correct),
                get_badge(l_correct)
            ]
            print(' '.join(badges))

            serializer = SQLSerializer(schema_graph, self.field_vocab, self.aggregation_ops, self.arithmetic_ops,
                                       self.condition_ops, self.logical_ops, self.value_vocab)
            print('select clause: {}'.format(serializer.serialize_select(example_pred['select'])))
            print('group by clause: {}'.format(serializer.serialize_group_by(example_pred['groupBy'])))
            print('order by clause: {}'.format(serializer.serialize_order_by(example_pred['orderBy'])))

        return sql_correct, s_correct, g_correct, o_correct, f_correct, w_correct, h_correct, l_correct

    def match(self, pred, gt_ast):
        gt_s = gt_ast['select']
        pred_s = pred['select']
        s_correct = self.match_select(pred_s, gt_s)

        gt_g = gt_ast['groupBy']
        pred_g = pred['groupBy']
        g_correct = self.match_group_by(pred_g, gt_g)

        gt_o = gt_ast['orderBy']
        pred_o = pred['orderBy']
        o_correct = self.match_order_by(pred_o, gt_o)

        gt_f = gt_ast['from']
        f_correct = True

        gt_w = gt_ast['where']
        w_correct = True

        gt_h = gt_ast['having']
        h_correct = True

        gt_l = gt_ast['limit']
        l_correct = True

        return s_correct, g_correct, o_correct, f_correct, w_correct, h_correct, l_correct

    def match_select(self, pred, gt):
        if not gt and not pred:
            return True
        if not gt or not pred:
            return False
        return pred[0] == gt[0] and orderless_match(pred[1], gt[1])

    def match_group_by(self, pred, gt):
        return orderless_match(pred, gt)

    def match_order_by(self, pred, gt):
        if not gt and not pred:
            return True
        if not gt or not pred:
            return False
        return pred[0] == gt[0] and orderless_match(pred[1], gt[1])

    def pack_sql(self, pred):
        return {
            'select': self.pack_select_clause(pred['select']),
            'groupBy': self.pack_group_by_clause(pred['groupBy']),
            'orderBy': self.pack_order_by_clause(pred['orderBy']),
            'from': self.pack_from_clause(pred['from']),
            'where': self.pack_where_clause(pred['where']),
            'having': self.pack_having_clause(pred['having']),
            'limit': self.pack_limit_clause(pred['limit'])
        }

    def pack_select_clause(self, pred):
        fields, aggs, distincts, vu_aggs, cla_distinct = pred
        assert (len(fields) == len(vu_aggs))
        assert (len(aggs) == len(vu_aggs))
        assert (len(distincts) == len(vu_aggs))
        cla_distinct = (cla_distinct[0] == 1)
        out = (cla_distinct, self.pack_select_field_expression_list(fields, aggs, distincts, vu_aggs))
        return out

    def pack_select_field_expression_list(self, fields, aggs, distincts, vu_aggs):
        out = []
        for fields_, aggs_, distincts_, vu_agg in zip(fields, aggs, distincts, vu_aggs):
            out.append((vu_agg, self.pack_field_expression(fields_, aggs_, distincts_)))
        return out

    def pack_group_by_clause(self, pred):
        fields, aggs, distincts = pred
        assert(len(fields) == len(aggs))
        assert(len(distincts) == len(aggs))
        out = []
        for fields_, aggs_, distincts_ in zip(fields, aggs, distincts):
            out.append(self.pack_field_expression(fields_, aggs_, distincts_))
        return out

    def pack_order_by_clause(self, pred):
        fields, aggs, distincts, ascs = pred
        assert(len(fields) == len(ascs))
        assert(len(aggs) == len(ascs))
        assert(len(distincts) == len(ascs))
        asc = 'asc' if ascs[0] == 1 else 'desc'
        field_expression_list = []
        for fields_, aggs_, distincts_ in zip(fields, aggs, distincts):
            field_expression_list.append(self.pack_field_expression(fields_, aggs_, distincts_))
        return asc, field_expression_list

    def pack_from_clause(self, pred):
        pass

    def pack_where_clause(self, pred):
        pass

    def pack_having_clause(self, pred):
        pass

    def pack_limit_clause(self, pred):
        pass

    def pack_field_expression(self, fields, aggs, distincts):
        # convert the in-order serialization of a field expression tree back to the tree structure
        assert(len(fields) % 2 == 1)
        i = 0
        if fields[i] < self.arithmetic_ops.full_size:
            # current node is an op
            if len(fields) > 1:
                second_arg = self.pack_field_expression(fields[i+2:], aggs[i+2:], distincts[i+2:])
                if second_arg is None:
                    return None
                return fields[i], self.pack_field(fields[i+1], aggs[i+1], distincts[i+1]), second_arg
            else:
                return None
        else:
            # current node is a field
            return self.pack_field(fields[i] - self.arithmetic_ops.full_size, aggs[i], distincts[i])

    def pack_field(self, field, agg, distinct):
        distinct = (distinct == 1)
        return agg, field, distinct


def orderless_match(x, y):
    return list_to_hist(x) == list_to_hist(y)


def get_exact_match_metrics(examples, pred_list, in_execution_order=False, engine=None):
    assert(len(examples) == len(pred_list))

    num_top_1_c, num_top_2_c, num_top_3_c, num_top_5_c, num_top_10_c = 0, 0, 0, 0, 0
    num_ex_top_1_c, num_ex_top_2_c, num_ex_top_3_c, num_ex_top_5_c, num_ex_top_10_c = 0, 0, 0, 0, 0
    table_errs = []

    for i, example in enumerate(examples):
        top_k_preds = pred_list[i]
        em_recorded, ex_recorded = False, False
        for j, pred in enumerate(top_k_preds):
            if example.dataset_id == SPIDER:
                gt_program_list = example.program_list
            elif example.dataset_id == WIKISQL:
                gt_program_list = example.program_ast_list_
            else:
                gt_program_list = example.gt_program_list

            results = eval_prediction(pred, gt_program_list, example.dataset_id,
                                      db_name=example.db_name, in_execution_order=in_execution_order,
                                      engine=engine)
            if j == 0:
                table_errs.append(results[-1])

            if example.dataset_id == SPIDER:
                em_correct = results[0]
            elif example.dataset_id == WIKISQL:
                ex_correct = results[0][0]
                em_correct = results[0][1]
            else:
                raise NotImplementedError

            if example.dataset_id == WIKISQL:
                if j >= 10:
                    break
                if em_correct and not em_recorded:
                    if j == 0:
                        num_top_1_c += 1
                    if j < 2:
                        num_top_2_c += 1
                    if j < 3:
                        num_top_3_c += 1
                    if j < 5:
                        num_top_5_c += 1
                    if j < 10:
                        num_top_10_c += 1
                    em_recorded = True
                if ex_correct and not ex_recorded:
                    if j == 0:
                        num_ex_top_1_c += 1
                    if j < 2:
                        num_ex_top_2_c += 1
                    if j < 3:
                        num_ex_top_3_c += 1
                    if j < 5:
                        num_ex_top_5_c += 1
                    if j < 10:
                        num_ex_top_10_c += 1
                    ex_recorded = True
            else:
                if em_correct:
                    if j == 0:
                        num_top_1_c += 1
                    if j < 2:
                        num_top_2_c += 1
                    if j < 3:
                        num_top_3_c += 1
                    if j < 5:
                        num_top_5_c += 1
                    if j < 10:
                        num_top_10_c += 1
                    break

    metrics = dict()
    metrics['top_1_em'] = float(num_top_1_c) / len(examples)
    metrics['top_2_em'] = float(num_top_2_c) / len(examples)
    metrics['top_3_em'] = float(num_top_3_c) / len(examples)
    metrics['top_5_em'] = float(num_top_5_c) / len(examples)
    metrics['top_10_em'] = float(num_top_10_c) / len(examples)

    if example.dataset_id == WIKISQL:
        metrics['top_1_ex'] = float(num_ex_top_1_c) / len(examples)
        metrics['top_2_ex'] = float(num_ex_top_2_c) / len(examples)
        metrics['top_3_ex'] = float(num_ex_top_3_c) / len(examples)
        metrics['top_5_ex'] = float(num_ex_top_5_c) / len(examples)
        metrics['top_10_ex'] = float(num_ex_top_10_c) / len(examples)

    assert(len(table_errs) == len(examples))
    metrics['table_err'] = sum(table_errs) / len(examples)
    return metrics


def eval_prediction(pred, gt_list, dataset_id, db_name=None, in_execution_order=False, engine=None):
    if dataset_id == SPIDER:
        try:
            return spider_eval_tools.evaluate_single_query_with_multiple_ground_truths(
                pred, [(gt, db_name) for gt in gt_list], in_execution_order=in_execution_order)
        except Exception as e:
            print(str(e))
            return False, 'easy', 0
    elif dataset_id == WIKISQL:
        assert(len(gt_list) == 1)
        gt = {'sql': gt_list[0], 'table_id': db_name}
        pred = {'query': pred, 'table_id': db_name}
        ex_correct, em_correct = wikisql_eval_tools.eval_fun(gt, pred, engine)
        return (ex_correct, em_correct), 'easy', 0
    else:
        raise NotImplementedError


def correct_ignoring_trivial_diffs(s, ground_truths):
    for gt in ground_truths:
        if equal_ignoring_trivial_diffs(s, gt):
            return True
    return False


def get_badge(correct):
    if correct:
        return '[CORRECT]'
    else:
        return '[WRONG]'
