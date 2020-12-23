"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Class for recording dataset statistics.
"""
import numpy as np


class DatasetStatistics(object):
    def __init__(self):
        self.num_examples = 0
        self.num_oov = 0
        self.num_denormalization_failed = 0
        self.num_schema_truncated = 0
        self.num_token_restored = 0
        self.max_ptr_span_size = 0

        self.num_text_tokens = []
        self.num_input_tokens = []
        self.num_cm_tokens = []
        self.num_cm_whole_field_tokens = []

    def accumulate(self, ds):
        self.num_examples += ds.num_examples
        self.num_oov += ds.num_oov
        self.num_denormalization_failed += ds.num_denormalization_failed
        self.num_schema_truncated += ds.num_schema_truncated
        self.num_token_restored += ds.num_token_restored
        if ds.max_ptr_span_size > self.max_ptr_span_size:
            self.max_ptr_span_size = ds.max_ptr_span_size

        self.num_text_tokens += ds.num_text_tokens
        self.num_input_tokens += ds.num_input_tokens
        self.num_cm_tokens += ds.num_cm_tokens
        self.num_cm_whole_field_tokens += ds.num_cm_whole_field_tokens

    def print(self, split=''):
        print('********** {} Data Statistics ***********'.format(split))
        print('OOV observed in {}/{} examples'.format(self.num_oov, self.num_examples))
        print('Denormalization skipped for {}/{} examples'.format(self.num_denormalization_failed, self.num_examples))
        print('Schema truncated for {}/{} examples'.format(self.num_schema_truncated, self.num_examples))
        print('Token restored for {}/{} examples'.format(self.num_token_restored, self.num_examples))
        if len(self.num_text_tokens) > 0:
            print('+ text sizes')
            print('# text tokens (avg) = {}'.format(np.mean(self.num_text_tokens)))
            print('# text tokens (min) = {}'.format(np.min(self.num_text_tokens)))
            print('# text tokens (max) = {}'.format(np.max(self.num_text_tokens)))
            print('+ input sizes')
            print('input size (avg) = {}'.format(np.mean(self.num_input_tokens)))
            print('input size (min) = {} '.format(np.min(self.num_input_tokens)))
            print('input size (max) = {}'.format(np.max(self.num_input_tokens)))
            print('+ program sizes')
            print('# program tokens (avg) = {}\t# program whole field tokens = {} (avg)\t'.format(
                np.mean(self.num_cm_tokens), np.mean(self.num_cm_whole_field_tokens)))
            print('# program tokens (min) = {}\t# program whole field tokens = {} (min)\t'.format(
                np.min(self.num_cm_tokens), np.min(self.num_cm_whole_field_tokens)))
            print('# program tokens (max) = {}\t# program whole field tokens = {} (max)\t'.format(
                np.max(self.num_cm_tokens), np.max(self.num_cm_whole_field_tokens)))
            print('max pointer span size = {}'.format(self.max_ptr_span_size))


