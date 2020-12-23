"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Pointer-Generator Network.
"""

import torch

from src.common.nn_modules import PointerSwitch, selective_read
import src.common.ops as ops
from src.semantic_parser.decoding_algorithms import beam_search
from src.semantic_parser.seq2seq import Seq2Seq, RNNEncoder, RNNDecoder
from src.utils.utils import SEQ2SEQ_PG


class PointerGenerator(Seq2Seq):
    def __init__(self, args, in_vocab=None, out_vocab=None):
        super().__init__(args, in_vocab, out_vocab)
        self.model_id = SEQ2SEQ_PG

    def forward(self, encoder_ptr_input_ids, encoder_ptr_value_ids, decoder_input_ids=None, decoder_ptr_value_ids=None):
        # Encoder operations
        # => [batch_size, input_seq_len]
        inputs, input_masks = encoder_ptr_input_ids
        if self.pretrained_transformer:
            segment_ids, position_ids = self.get_segment_and_position_ids(inputs)
            inputs_embedded, _ = self.encoder_embeddings(
                inputs, input_masks, segments=segment_ids, position_ids=position_ids)
        else:
            inputs_embedded = self.encoder_embeddings(inputs)
        encoder_base_hiddens, hidden = self.encoder(inputs_embedded, input_masks)
        encoder_hiddens = encoder_base_hiddens

        # Decoder operations
        # => [batch_size, target_seq_len]
        if self.training:
            # compute training objectives
            targets_embedded = self.decoder_embeddings(decoder_input_ids)
            outputs = self.decoder(targets_embedded,
                                   hidden,
                                   encoder_hiddens,
                                   input_masks,
                                   encoder_ptr_value_ids=encoder_ptr_value_ids,
                                   decoder_ptr_value_ids=decoder_ptr_value_ids)
            return outputs[0]
        else:
            with torch.no_grad():
                if self.decoding_algorithm == 'beam-search':
                    return beam_search(self.bs_alpha,
                                       self.model_id,
                                       self.decoder,
                                       self.decoder_embeddings,
                                       self.max_out_seq_len,
                                       self.beam_size,
                                       hidden,
                                       encoder_hiddens,
                                       input_masks,
                                       encoder_ptr_value_ids)
                else:
                    raise NotImplementedError

    def define_modules(self):
        self.define_embeddings()
        self.encoder = RNNEncoder(self.encoder_input_dim,
                                  self.encoder_hidden_dim,
                                  self.num_rnn_layers,
                                  self.rnn_layer_dropout,
                                  self.rnn_weight_dropout)
        self.decoder = PointerGeneratorDecoder(self.decoder_input_dim,
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


class PointerGeneratorDecoder(RNNDecoder):
    """
    LSTM Pointer-Generator Decoder.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, vocab, rnn_layer_dropout, rnn_weight_dropout,
                 ff_dropouts, attn_params):
        super().__init__(input_dim, hidden_dim, num_layers, vocab, rnn_layer_dropout, rnn_weight_dropout,
                         ff_dropouts, attn_params)
        context_dim = sum([v_dim for v_dim, _, _ in attn_params])
        self.pointer_switch = PointerSwitch(query_dim=hidden_dim, key_dim=context_dim, input_dropout=ff_dropouts[0])

    def forward(self, input_embedded, hidden, encoder_hiddens, encoder_hidden_masks, pointer_context=None,
                encoder_ptr_value_ids=None, decoder_ptr_value_ids=None, last_output=None):
        """
        :param input_embedded: [batch_size, seq_len(=1), input_dim]
        :param hidden: (h, c)
            h - [num_layers, batch_size, hidden_dim]
            c - [num_layers, batch_size, hidden_dim]
        :param encoder_hiddens: [batch_size, encoder_seq_len, hidden_dim]
        :param encoder_hidden_masks: [batch_size, encoder_seq_len]
        :param pointer_context
            p_pointer - [batch_size, seq_len(=1), 1]
            attn_weights - [batch_size, seq_len(=1), attn_value_dim]
        :param encoder_ptr_value_ids: [batch_size, encoder_seq_len]
            Mapping element in the memory to the pointing-generating vocabulary. If None, the pointing and generating
            vocabularies do not overlap.
        :param decoder_ptr_value_ids: [batch_size, decoder_seq_len]
            Decoder output ground truth. Used during training only.
        :param last_output: [batch_size, seq_len(=1)]
            Decoding result of previous step.
        :return outputs: [batch_size, seq_len(=1), output_vocab_size]
        :return hidden: updated decoder hidden state
        :return pointer_context
            p_pointer - [batch_size, seq_len(=1), 1]
            attn_weights - [batch_size, num_head, seq_len(=1), encoder_seq_len]
        """
        if not self.training:
            assert(decoder_ptr_value_ids is None)
        batch_size = len(input_embedded)

        # unpack input_embedded
        if pointer_context:
            p_pointer, attn_weights = pointer_context
        else:
            p_pointer = ops.zeros_var_cuda([batch_size, 1, 1])
            attn_weights = ops.zeros_var_cuda([batch_size, self.attn.num_heads, 1, encoder_hiddens.size(1)])

        outputs, hiddens = [], []
        seq_attn_weights = []
        seq_p_pointers = []

        for i in range(input_embedded.size(1)):
            input_ = input_embedded[:, i:i + 1, :]
            # compute selective read
            if self.training and decoder_ptr_value_ids is not None:
                if i == 0:
                    last_output = ops.int_fill_var_cuda([batch_size, 1], self.vocab.start_id)
                else:
                    last_output = decoder_ptr_value_ids[:, i:i+1]
            else:
                assert(last_output is not None)
            # [batch_size, encoder_seq_len]
            selective_attn = selective_read(encoder_ptr_value_ids, encoder_hiddens,
                                            self.merge_multi_head_attention(attn_weights), last_output)
            # [batch_size, 1, input_dim + self.attn_value_dim]
            input_ = torch.cat([input_, selective_attn], dim=2)

            output, hidden = self.rnn(input_, hidden)
            if self.training and self.return_hiddens:
                hiddens.append(hidden)
            # a) compute attention vector and attention weights
            # [batch_size, 1, attn_value_dim], [batch_size, num_head, 1, encoder_seq_len]
            attn_vec, attn_weights = self.attn(output, encoder_hiddens, encoder_hiddens, encoder_hidden_masks)
            # b) compute pointer-generator switch
            # [batch_size, 1, 1]
            p_pointer = self.pointer_switch(output, attn_vec)
            # c) update output state
            output = self.attn_combine(output, attn_vec)
            # d.1) compute generation prob
            # [batch_size, 1, output_vocab_size]
            gen_logit = self.out(output)
            # d.2) merge pointing and generation prob
            # [batch_size, 1, output_vocab_size + max_in_seq_len]
            if encoder_ptr_value_ids is None:
                point_gen_prob = torch.cat([(1 - p_pointer) * torch.exp(gen_logit),
                                            p_pointer * self.merge_multi_head_attention(attn_weights)], dim=2)
            else:
                gen_prob_zeros_pad = ops.zeros_var_cuda((batch_size, 1, encoder_ptr_value_ids.size(1)))
                weighted_gen_prob = torch.cat([(1 - p_pointer) * torch.exp(gen_logit), gen_prob_zeros_pad], dim=2)
                weighted_point_prob = p_pointer * self.merge_multi_head_attention(attn_weights)
                point_gen_prob = weighted_gen_prob.scatter_add_(
                    dim=2, index=encoder_ptr_value_ids.unsqueeze(1), src=weighted_point_prob)
            point_gen_logit = ops.safe_log(point_gen_prob)

            outputs.append(point_gen_logit)
            seq_attn_weights.append(attn_weights)
            seq_p_pointers.append(p_pointer)

        if self.training and self.return_hiddens:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2)), \
                   self.cat_lstm_hiddens(hiddens)
        else:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2))

    def get_input_feed(self, input):
        """
        Get the input indices accepted by the decoder during autoregressuve decoding.
        """
        # set copied tokens to UNK
        vocab_mask = (input < self.vocab_size).long()
        return vocab_mask * input + (1 - vocab_mask) * self.vocab.unk_id