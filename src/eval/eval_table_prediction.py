"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np


def eval_table_f1(examples, pred_tables_list):
    metrics = dict()
    prec_micro, recall_micro, f1_micro = [], [], []
    for example, pred_tables in zip(examples, pred_tables_list):
        gt_tables = set(example.gt_table_names)
        pred_tables = pred_tables[0]
        assert(len(pred_tables) > 0)
        correct = gt_tables & pred_tables
        precision = len(correct) / len(pred_tables)
        recall = len(correct) / len(gt_tables)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if recall != 1:
            pass
            # print('GT: {}'.format(gt_sql))
            # print('GT tables: {}'.format(sorted(gt_tables)))
            # print('PR: {}'.format(pred_sql))
            # print('Pred tables: {}'.format(sorted(pred_tables)))
            # print()
            # import pdb
            # pdb.set_trace()

        prec_micro.append(precision)
        recall_micro.append(recall)
        f1_micro.append(f1)

    metrics['precision'] = np.mean(prec_micro)
    metrics['recall'] = np.mean(recall_micro)
    metrics['f1'] = np.mean(f1_micro)
    return metrics