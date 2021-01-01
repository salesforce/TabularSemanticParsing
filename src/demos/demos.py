"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Demo Interfaces.
"""
import os
import sys
import time

import src.data_processor.data_loader as data_loader
import src.data_processor.processor_utils as data_utils
from src.data_processor.schema_graph import SchemaGraph, SchemaGraphs
from src.data_processor.schema_loader import load_schema_graphs
from src.data_processor.path_utils import get_model_dir, get_checkpoint_path
from src.data_processor.processor_utils import WIKISQL
from src.data_processor.processors.data_processor_spider import preprocess_example
import src.data_processor.schema_loader as schema_loader
import src.data_processor.tokenizers as tok
from src.semantic_parser.learn_framework import EncoderDecoderLFramework
from src.trans_checker.trans_checker import TranslatabilityChecker
from src.utils.utils import model_index
from src.utils.utils import SEQ2SEQ, SEQ2SEQ_PG, BRIDGE


def load_semantic_parser(args):
    if args.model in model_index:
        sp = EncoderDecoderLFramework(args)
    else:
        raise NotImplementedError
    sp.load_checkpoint(get_checkpoint_path(args))
    sp.cuda()
    sp.eval()
    return sp


def load_confusion_span_detector(args):
    tc = TranslatabilityChecker(args)
    if args.checkpoint_path is not None:
        tc.load_checkpoint(args.checkpoint_path)
    else:
        print('Warning: translatability checker checkpoint not specified')
        return None
    tc.cuda()
    tc.eval()
    return tc


def demo_preprocess(args, example, vocabs=None, schema_graph=None):
    text_tokenize, program_tokenize, post_process, tu = tok.get_tokenizers(args)
    if not schema_graph:
        schema_graphs = load_schema_graphs(args)
        schema_graph = schema_graphs.get_schema(example.db_id)
    schema_graph.lexicalize_graph(tokenize=text_tokenize, normalized=(args.model_id in [BRIDGE]))
    preprocess_example('test', example, args, {}, text_tokenize, program_tokenize, post_process, tu, schema_graph, vocabs)


class Text2SQLWrapper(object):
    def __init__(self, args, cs_args, schema):
        self.args = args
        self.text_tokenize, _, _, self.tu = tok.get_tokenizers(args)

        # Vocabulary
        self.vocabs = data_loader.load_vocabs(args)

        # Confusion span detector
        self.confusion_span_detector = load_confusion_span_detector(cs_args)

        # Text-to-SQL model
        self.semantic_parser = load_semantic_parser(args)
        self.semantic_parser.schema_graphs = SchemaGraphs()
        if schema is not None:
            self.add_schema(schema)

        # When generating SQL in execution order, cache reordered SQLs to save time
        if args.process_sql_in_execution_order:
            self.pred_restored_cache = self.semantic_parser.load_pred_restored_cache()
        else:
            self.pred_restored_cache = None

    def confusion_span_detection(self, example):
        text_tokens = example.text_tokens
        confusion_span = self.confusion_span_detector.inference([example])
        confusion_span = confusion_span[0]
        if confusion_span[0] == 0:
            return True, None, None
        else:
            if confusion_span[1] - confusion_span[0] + 1 >= 5:
                return False, None, None
            else:
                confuse_span = self.tu.tokenizer.convert_tokens_to_string(text_tokens[confusion_span[0] - 1 : confusion_span[1]])
                return False, confuse_span, None

    def translate(self, example):
        """
        :param text: natural language question
        :return: SQL query corresponding to the input question
        """
        start_time = time.time()
        output = self.semantic_parser.inference([example], restore_clause_order=self.args.process_sql_in_execution_order,
                                                pred_restored_cache=self.pred_restored_cache, verbose=False)
        if len(output['pred_decoded'][0]) > 1:
            pred_sql = output['pred_decoded'][0][0]
        else:
            pred_sql = None
        print('inference time: {:.2f}s'.format(time.time() - start_time))
        return pred_sql

    def process(self, text, schema_name, verbose=False):
        schema = self.semantic_parser.schema_graphs[schema_name]
        start_time = time.time()
        example = data_utils.Text2SQLExample(data_utils.OTHERS, schema.name,
                                             db_id=self.semantic_parser.schema_graphs.get_db_id(schema.name))
        example.text = text
        demo_preprocess(self.args, example, self.vocabs, schema)
        print('data processing time: {:.2f}s'.format(time.time() - start_time))

        if self.confusion_span_detector:
            translatable, confuse_span, replace_span = self.confusion_span_detection(example)
        else:
            translatable, confuse_span, replace_span = True, None, None

        sql_query = None
        if translatable:
            print('Translatable!')
            sql_query = self.translate(example)
            if verbose:
                print('Text: {}'.format(text))
                print('SQL: {}'.format(sql_query))
                print()
        else:
            print('Untranslatable!')

        output = dict()
        output['translatable'] = translatable
        output['sql_query'] = sql_query
        output['confuse_span'] = confuse_span
        output['replace_span'] = replace_span
        return output

    def add_schema(self, schema):
        schema.lexicalize_graph(tokenize=self.text_tokenize)
        if schema.name not in self.semantic_parser.schema_graphs.db_index:
            self.semantic_parser.schema_graphs.index_schema_graph(schema)

    def schema_exists(self, schema_name):
        return schema_name in self.semantic_parser.schema_graphs.db_index


def demo_table(args, sp):
    """
    Run the semantic parser from the standard input.
    """
    sp.load_checkpoint(get_checkpoint_path(args))
    sp.eval()

    vocabs = data_loader.load_vocabs(args)

    table_array = [
        ['name', 'age', 'gender'],
        ['John', 18, 'male'],
        ['Kate', 19, 'female']
    ]
    table_name = 'employees'
    schema_graph = schema_loader.SchemaGraph(table_name)
    schema_graph.load_data_from_2d_array(table_array)

    sys.stdout.write('Enter a natural language question: ')
    sys.stdout.write('> ')
    sys.stdout.flush()
    text = sys.stdin.readline()

    while text:
        example = data_utils.Text2SQLExample(0, table_name, 0)
        example.text = text
        demo_preprocess(args, example, vocabs, schema_graph)
        output = sp.forward([example])
        for i, sql in enumerate(output['pred_decoded'][0]):
            print('Top {}: {}'.format(i, sql))
        sys.stdout.flush()
        sys.stdout.write('\nEnter a natural language question: ')
        sys.stdout.write('> ')
        text = sys.stdin.readline()
