"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Evaluate schema linking approaches.
"""
import numpy as np


class SchemaLinkingEvaluator(object):
    def __init__(self):
        self.precs, self.recalls, self.f1s = [], [], []

    def eval_const_f1(self, ground_truth_values, pred_values, eval_field=False):
        if len(ground_truth_values) == 0 and len(pred_values) == 0:
            prec, recall, f1 = 1, 1, 1
        else:
            if eval_field:
                gt_set = set([(x.lower(), y.lower()) for x, y in ground_truth_values])
                p_set = set([(x.lower(), y.lower()) for x, y in pred_values])
            else:
                gt_set = set([x.lower() for x in ground_truth_values])
                p_set = set([x.lower() for x in pred_values])

            num_gt_matched = 0
            for gt_v in gt_set:
                if gt_v in p_set:
                    num_gt_matched += 1

            num_p_matched = 0
            for p_v in p_set:
                if p_v in gt_set:
                    num_p_matched += 1

            prec = num_p_matched / len(pred_values) if len(pred_values) > 0 else 0
            recall = num_gt_matched / len(ground_truth_values) if len(ground_truth_values) > 0 else 1
            f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0

        self.precs.append(prec)
        self.recalls.append(recall)
        self.f1s.append(f1)

    def print(self, split=''):
        print('--- {} value extraction performance ---'.format(split))
        print('micro precision = {}'.format(np.mean(self.precs)))
        print('micro recall = {}'.format(np.mean(self.recalls)))
        print('micro F1 = {}'.format(np.mean(self.f1s)))