"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Sequence-to-sequence encoder-decoder module.
"""

import torch
import torch.nn as nn

from src.common.nn_modules import *
import src.utils.utils as utils


class EncoderDecoder(nn.Module):
    """
    Interface class.
    """
    def __init__(self, args, in_vocab=None, out_vocab=None):
        super().__init__()
        self.share_vocab = args.share_vocab
        self.in_vocab, self.out_vocab = in_vocab, out_vocab
        self.input_vocab_size = self.in_vocab.full_size
        self.output_vocab_size = self.out_vocab.size
        self.tu = utils.get_trans_utils(args)
        self.max_in_seq_len = self.tu.tokenizer.max_len if args.pretrained_transformer != 'null' else args.max_in_seq_len
        self.max_out_seq_len = args.max_out_seq_len
        self.encoder_input_dim = args.encoder_input_dim
        self.decoder_input_dim = args.decoder_input_dim
        self.emb_dropout = args.emb_dropout_rate
        self.res_dropout = (args.res_input_dropout_rate, args.res_layer_dropout_rate)
        self.cross_attn_dropout = args.cross_attn_dropout_rate
        self.cross_attn_num_heads = args.cross_attn_num_heads
        self.xavier_initialization = args.xavier_initialization

        self.pretrained_transformer = args.pretrained_transformer
        self.pretrained_lm_dropout = args.pretrained_lm_dropout_rate
        self.fix_pretrained_transformer_parameters = args.fix_pretrained_transformer_parameters
        self.data_parallel = args.data_parallel

        self.dataset_name = args.dataset_name

        self.encoder_embeddings = None
        self.decoder_embeddings = None
        self.encoder = None
        self.decoder = None

    def define_embeddings(self):
        if self.pretrained_transformer != 'null':
            print('pretrained_transformer = {}'.format(self.pretrained_transformer))
            print('fix_pretrained_transformer_parameters = {}'.format(self.fix_pretrained_transformer_parameters))
            print()
            self.encoder_embeddings = TransformerHiddens(self.pretrained_transformer, dropout=self.pretrained_lm_dropout,
                                                         requires_grad=(not self.fix_pretrained_transformer_parameters))
            if self.data_parallel:
                self.encoder_embeddings = nn.DataParallel(self.encoder_embeddings)
        else:
            self.encoder_embeddings = Embedding(
                self.input_vocab_size, self.encoder_input_dim, dropout=self.emb_dropout, requires_grad=True)

        if self.share_vocab:
            assert(not self.pretrained_transformer)
            self.decoder_embeddings = self.encoder_embeddings
        else:
            self.decoder_embeddings = Embedding(
                self.output_vocab_size, self.decoder_input_dim, dropout=self.emb_dropout, requires_grad=True)

    def get_segment_and_position_ids(self, encoder_input_ids):
        batch_size, input_size = encoder_input_ids.size()
        position_ids = ops.arange_cuda(input_size).unsqueeze(0).expand_as(encoder_input_ids)
        # [CLS] w1 w2 ... [SEP] * [T] ...
        # 0     0  0  ...  0  1 1 ...
        seg1_end_pos = torch.nonzero(encoder_input_ids == self.tu.sep_id)[:, 1].view(batch_size, 2)[:, 0]
        segment_ids = (position_ids > seg1_end_pos.unsqueeze(1)).long()
        # position_ids = position_ids * (1 - segment_ids)
        # position_ids = position_ids * (1 - segment_ids) + (seg1_end_pos + 1).unsqueeze(1) * segment_ids
        # import pdb
        # pdb.set_trace()
        position_ids = None
        return segment_ids, position_ids
