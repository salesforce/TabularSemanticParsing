"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Sequence-to-sequence with attention model.
"""

import torch
import torch.nn as nn

from src.semantic_parser.decoding_algorithms import beam_search
from src.common.nn_modules import WeightDropoutLSTM, LSTMWithPacking, \
    MultiHead, AttentionDotProduct, ConcatAndProject
import src.common.ops as ops
from src.semantic_parser.decoder import Decoder
from src.semantic_parser.encoder_decoder import EncoderDecoder
from src.utils.utils import SEQ2SEQ


class Seq2Seq(EncoderDecoder):
    def __init__(self, args, in_vocab=None, out_vocab=None):
        super().__init__(args, in_vocab, out_vocab)
        self.model_id = SEQ2SEQ
        self.encoder_hidden_dim = args.encoder_hidden_dim if args.encoder_hidden_dim > 0 else args.encoder_input_dim
        self.decoder_hidden_dim = args.decoder_hidden_dim if args.decoder_hidden_dim > 0 else args.decoder_input_dim
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_layer_dropout = args.rnn_layer_dropout_rate
        self.rnn_weight_dropout = args.rnn_weight_dropout_rate
        self.attn_dropout = args.cross_attn_dropout_rate
        self.ff_dropouts = (args.ff_input_dropout_rate, args.ff_hidden_dropout_rate, args.ff_output_dropout_rate)
        self.decoding_algorithm = args.decoding_algorithm
        self.beam_size = args.beam_size
        self.bs_alpha = args.bs_alpha

        self.define_modules()

    def forward(self, inputs, targets=None):
        # Encoder operations
        # => [batch_size, input_seq_len]
        inputs, input_masks = inputs
        inputs_embedded = self.encoder_embeddings(inputs)
        encoder_hiddens, hidden = self.encoder(inputs_embedded, input_masks)

        # Decoder operations
        # => [batch_size, target_seq_len]
        if self.training:
            # compute training objectives
            assert(targets is not None)
            targets_embedded = self.decoder_embeddings(targets)
            outputs, _, _ = self.decoder(targets_embedded, hidden, encoder_hiddens, input_masks)
            return outputs
        else:
            with torch.no_grad():
                # autoregressuve decoding
                if self.decoding_algorithm == 'beam-search':
                    return beam_search(self.bs_alpha,
                                       self.model_id,
                                       self.decoder,
                                       self.decoder_embeddings,
                                       self.max_out_seq_len,
                                       self.beam_size,
                                       hidden,
                                       encoder_hiddens,
                                       input_masks)
                else:
                    raise NotImplementedError

    def define_modules(self):
        self.define_embeddings()
        self.encoder = RNNEncoder(self.encoder_input_dim,
                                  self.encoder_hidden_dim,
                                  self.num_rnn_layers,
                                  self.rnn_layer_dropout,
                                  self.rnn_weight_dropout)
        self.decoder = RNNDecoder(self.decoder_input_dim,
                                  self.decoder_hidden_dim,
                                  self.num_rnn_layers,
                                  self.out_vocab,
                                  self.rnn_layer_dropout,
                                  self.rnn_weight_dropout,
                                  self.ff_dropouts,
                                  [(self.decoder_hidden_dim,
                                    self.encoder_hidden_dim,
                                    self.cross_attn_num_heads,
                                    self.attn_dropout)])


class RNNEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, rnn_weight_dropout):
        super().__init__()
        self.rnn = LSTMWithPacking(input_dim=input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                   bidirectional=True, dropout=dropout, rnn_weight_dropout=rnn_weight_dropout)

    def forward(self, inputs, input_masks):
        """
        :param inputs: [batch_size, seq_len, input_dim]
        :param input_masks: [batch_size, seq_len]
            binary mask where the padding entries are set to 1
            if set to None,
                use a zero mask (equivalent to no padding)
        :return outputs: [batch_size, seq_len, num_directions*hidden_dim]
        :return h: [num_layers*num_directions, batch_size, hidden_size]
        :return c: [num_layers*num_directions, batch_size, hidden_size]
        """
        if input_masks is None:
            input_masks = ops.int_zeros_var_cuda([inputs.size(0), inputs.size(1)])
        max_seq_len = input_masks.size(1)
        input_sizes = max_seq_len - torch.sum(input_masks, dim=1).long()
        outputs, (h, c) = self.rnn(inputs, input_sizes.cpu())
        num_layers = self.rnn.num_layers
        assert(len(c) == self.rnn.num_directions * num_layers)
        if self.rnn.bidirectional:
            h = ops.pack_bidirectional_lstm_state(h, num_layers)
            c = ops.pack_bidirectional_lstm_state(c, num_layers)
        return outputs, (h, c)


class RNNDecoder(Decoder):
    """
    LSTM Decoder with Attention.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, vocab, rnn_layer_dropout, rnn_weight_dropout,
                 ff_dropouts, attn_params, return_hiddens=False):
        super().__init__(input_dim, hidden_dim, vocab, return_hiddens)

        context_dim = sum([v_dim for _, v_dim, _, _ in attn_params])
        self.rnn = WeightDropoutLSTM(input_size=input_dim + context_dim,
                                     hidden_size=hidden_dim,
                                     bidirectional=False,
                                     num_layers=num_layers,
                                     dropout=rnn_layer_dropout,
                                     rnn_weight_dropout=rnn_weight_dropout)

        attn_query_dim, attn_value_dim, attn_num_heads, attn_dropout = attn_params[0]
        attn_ = AttentionDotProduct(dropout=attn_dropout, causal=False)
        self.attn = MultiHead(attn_query_dim,
                              attn_value_dim,
                              attn_value_dim,
                              attn_num_heads,
                              attn_)
        self.attn_combine = ConcatAndProject(hidden_dim + context_dim, hidden_dim, ff_dropouts[0], activation='tanh')


    def forward(self, inputs, hidden, encoder_hiddens, encoder_masks):
        """
        :param inputs: [batch_size, seq_len(=1), input_dim]
        :param hidden: (h, c)
            h - [num_layers, batch_size, hidden_dim]
            c - [num_layers, batch_size, hidden_dim]
        :param encoder_hiddens: [batch_size, input_seq_len, hidden_dim]
        :param encoder_masks: [batch_size, input_seq_len]
        :return outputs: [batch_size, seq_len(=1), output_vocab_size]
        :return hidden: updated decoder hidden state
        :return attn_weights: [batch_size, num_head, seq_len(=1), encoder_seq_len]
        """
        outputs, hiddens = [], []
        seq_attn_weights = []
        for i in range(inputs.size(1)):
            input_embedded = inputs[:, i:i + 1, :]
            # [batch_size, 1, hidden_dim]
            output, hidden = self.rnn(input_embedded, hidden)
            if self.training and self.return_hiddens:
                hiddens.append(hidden)
            # [batch_size, 1, hidden_dim], [batch_size, num_head, seq_len(=1), encoder_seq_len]
            attn_vec, attn_weights = self.attn(output, encoder_hiddens, encoder_hiddens, encoder_masks)
            # [batch_size, 1, hidden_dim]
            output = self.attn_combine(output, attn_vec)
            # [batch_size, 1, output_vocab_size]
            output = self.out(output)
            outputs.append(output)
            seq_attn_weights.append(attn_weights)
        if self.training and self.return_hiddens:
            return torch.cat(outputs, dim=1), hidden, torch.cat(seq_attn_weights, dim=2), self.cat_lstm_hiddens(hiddens)
        else:
            return torch.cat(outputs, dim=1), hidden, torch.cat(seq_attn_weights, dim=2)

    def merge_multi_head_attention(self, attn_weights, use_mean=False):
        if use_mean:
            return attn_weights.mean(dim=1)
        else:
            return attn_weights[:, -1, :, :]

    def get_input_feed(self, input):
        return input
