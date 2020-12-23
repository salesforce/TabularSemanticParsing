"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Rule-based perturbation of natural language questions to increase language diversity.
"""
import copy
import json
import os
import random
import sys


punctuations = [
    '.',
    '..',
    '...',
    '!',
    '?',
    '!!',
    '??',
    '!!!',
    '???'
]


class NaturalLanguageAugmentor(object):
    def __init__(self, data_augmentation_factor):
        self.data_augmentation_factor = data_augmentation_factor

    def augment_spider_dataset(self, data):
        new_data = copy.deepcopy(data)
        data_hash = set()
        for example in data:
            text = example['question']
            aug_texts = {text}
            aug_texts = aug_texts.union(set(self.remove_head_verb(list(aug_texts))))
            aug_texts = aug_texts.union(set(self.enumerate_articles(list(aug_texts))))
            aug_texts = aug_texts.union(set(self.add_punctuation(list(aug_texts))))
            aug_texts = list(aug_texts)
            if aug_texts:
                random.shuffle(aug_texts)
                print('[S]:\t{}'.format(text))
                for i, aug_text in enumerate(aug_texts):
                    print('AS-{}:\t{}'.format(i, aug_text))
                    if i < self.data_augmentation_factor:
                        new_exp = {
                            'db_id': example['db_id'],
                            'query': example['query'],
                            'question': aug_text
                        }
                        new_exp_key = json.dumps(new_exp, sort_keys=True)
                        if not new_exp_key in data_hash:
                            new_data.append(new_exp)
                            data_hash.add(new_exp_key)
        return new_data

    def add_punctuation(self, sents):
        aug_sents = []
        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            punc_in_the_end = True
            while (punc_in_the_end):
                punc_in_the_end = False
                for punc in punctuations:
                    if sent.endswith(punc):
                        sent = sent[:-len(punc)]
                        punc_in_the_end = True
        aug_sents.append(sent)
        for punc in punctuations:
            aug_sents.append(sent + punc)
        return aug_sents

    def enumerate_articles(self, sents):
        aug_sents = []
        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            tokens = sent.split()
            article_indices = []
            for i, token in enumerate(tokens):
                if token == 'a' or token == 'an' or token == 'the':
                    article_indices.append(i)
            for article_idx in article_indices:
                aug_sents.append(' '.join([token for i, token in enumerate(tokens) if i != article_idx]))
            if len(article_indices) > 1:
                aug_sents.append(' '.join([token for i, token in enumerate(tokens) if i not in article_indices]))
        return aug_sents

    def remove_head_verb(self, sents):
        prefices = [
            'What is',
            'What are',
            'Return',
            'Return all',
            'Return me',
            'Return me all',
            'Find',
            'Find all',
            'Find me',
            'Find me all',
            'Give',
            'Give all',
            'Give me',
            'Give me all',
            'Show',
            'Show all',
            'Show me',
            'Show me all',
            'List',
            'List all'
        ]

        aug_sents = []
        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            if sent.endswith('.') or sent.endswith('?'):
                sent = sent[:-1]
            for prefix in prefices:
                if sent.startswith(prefix) and not sent.startswith(prefix + ' me'):
                    aug_sents.append(sent.replace(prefix, '').strip())
                if sent.startswith(prefix.lower()) and not sent.startswith(prefix.lower() + ' me'):
                    aug_sents.append(sent.replace(prefix.lower(), '').strip())
        return aug_sents


def augment_spider_data():
    data_dir = sys.argv[1]
    data_augmentation_factor = int(sys.argv[2])

    in_json = os.path.join(data_dir, 'fine-tune.json')
    out_json = os.path.join(data_dir, 'fine-tune.aug-{}.json'.format(data_augmentation_factor))
    nl_aug = NaturalLanguageAugmentor(data_augmentation_factor)
    with open(in_json) as f:
        dataset = json.load(f)
        new_dataset = nl_aug.augment_spider_dataset(dataset)
        with open(out_json, 'w') as o_f:
            json.dump(new_dataset, o_f, indent=4)
            print('{} examples saved to {}'.format(len(new_dataset), out_json))


if __name__ == '__main__':
    augment_spider_data()
