"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

    BRIDGE Text-to-SQL model (Lin et. al. 2020).

    Highlights:
        1. BRIDGE is a Seeq2Seq framework with bi-directional encoder and autoregressive pointer-generator.
        2. BRIDGE uses multi-head attention between the encoder and decoder LSTM layers. The last attention head is also
            used as the pointer.
        3. BRIDGE assumes the input/output sequences are processed for code synthesis. Especially, units that need to be
            copied as a single unit (e.g. a table name, a field name, a value phrase etc.) needs to be handled so in the
            preprocessing.
        4. The BRIDGE encoder design is heavily inspired by the SQLova model (Hwang et al. 2019)
            https://github.com/naver/sqlova
"""

import torch
import torch.nn as nn

from src.semantic_parser.decoding_algorithms import beam_search
from src.common.nn_modules import Embedding, ConcatAndProject, FusionLayer, Feedforward, Linear, PointerSwitch, \
    SelfAttentionLayer, selective_read
import src.common.ops as ops
from src.data_processor.sql.sql_operators import field_types
from src.semantic_parser.seq2seq_ptr import PointerGenerator, RNNEncoder, RNNDecoder
from src.utils.utils import BRIDGE


class Bridge(PointerGenerator):
    def __init__(self, args, in_vocab=None, out_vocab=None):
        self.num_const_attn_layers = args.num_const_attn_layers
        self.use_lstm_encoder = args.use_lstm_encoder
        self.use_meta_data_encoding = args.use_meta_data_encoding
        self.use_graph_encoding = args.use_graph_encoding
        super().__init__(args, in_vocab, out_vocab)
        self.model_id = BRIDGE

    def forward(self, encoder_ptr_input_ids, encoder_ptr_value_ids, text_masks, schema_masks, feature_ids,
                transformer_output_value_masks=None, schema_memory_masks=None, decoder_input_ids=None,
                decoder_ptr_value_ids=None):
        # Encoder operations
        # => [batch_size, input_seq_len]
        inputs, input_masks = encoder_ptr_input_ids
        if self.pretrained_transformer:
            segment_ids, position_ids = self.get_segment_and_position_ids(inputs)
            inputs_embedded, _ = self.encoder_embeddings(
                inputs, input_masks, segments=segment_ids, position_ids=position_ids)
        else:
            inputs_embedded = self.encoder_embeddings(inputs)
        encoder_hiddens, encoder_hidden_masks, constant_hidden_masks, schema_hidden_masks, hidden = \
            self.encoder(inputs_embedded,
                         input_masks,
                         text_masks,
                         schema_masks,
                         feature_ids,
                         transformer_output_value_masks)
        # Decoder operations
        # => [batch_size, target_seq_len]
        if self.training:
            # compute training objectives
            targets_embedded = self.decoder_embeddings(decoder_input_ids)
            assert(schema_memory_masks is None)
            outputs = self.decoder(targets_embedded,
                                   hidden,
                                   encoder_hiddens,
                                   encoder_hidden_masks,
                                   memory_masks=schema_memory_masks,
                                   encoder_ptr_value_ids=encoder_ptr_value_ids,
                                   decoder_ptr_value_ids=decoder_ptr_value_ids)
            return outputs[0]
        else:
            with torch.no_grad():
                if self.decoding_algorithm == 'beam-search':
                    # [batch_size, schema_seq_len]
                    table_masks, _ = feature_ids[3]
                    table_pos, _ = feature_ids[4]
                    if table_pos is not None:
                        table_field_scope, _ = feature_ids[5]
                        db_scope = (table_pos, table_field_scope)
                    else:
                        db_scope = None
                    return beam_search(self.bs_alpha,
                                       self.model_id,
                                       self.decoder,
                                       self.decoder_embeddings,
                                       self.max_out_seq_len,
                                       self.beam_size,
                                       hidden,
                                       encoder_hiddens=encoder_hiddens,
                                       encoder_masks=encoder_hidden_masks,
                                       constant_hidden_masks=constant_hidden_masks,
                                       schema_hidden_masks=schema_hidden_masks,
                                       table_masks=table_masks,
                                       encoder_ptr_value_ids=encoder_ptr_value_ids,
                                       schema_memory_masks=schema_memory_masks,
                                       db_scope=db_scope,
                                       no_from=(self.dataset_name == 'wikisql'))
                else:
                    raise NotImplementedError

    def define_modules(self):
        if not self.use_lstm_encoder:
            self.encoder_hidden_dim = self.encoder_input_dim
        self.define_embeddings()
        self.encoder = SchemaAwareTransformerEncoder(self.in_vocab,
                                                     self.out_vocab,
                                                     self.encoder_input_dim,
                                                     self.encoder_hidden_dim,
                                                     self.decoder_hidden_dim,
                                                     self.num_rnn_layers,
                                                     self.num_const_attn_layers,
                                                     self.rnn_layer_dropout,
                                                     self.rnn_weight_dropout,
                                                     self.emb_dropout,
                                                     self.res_dropout,
                                                     self.ff_dropouts,
                                                     self.use_lstm_encoder,
                                                     self.use_meta_data_encoding,
                                                     self.use_graph_encoding)
        self.decoder = BridgeDecoder(self.decoder_input_dim,
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


class SchemaEncoder(nn.Module):
    """
    DB-schema Encoder.
    """
    def __init__(self, hidden_dim, feat_dim, feat_emb_dropout, res_dropout, ff_dropouts, use_graph_encoding=False):
        super().__init__()
        self.primary_key_embeddings = Embedding(2, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.foreign_key_embeddings = Embedding(2, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.field_type_embeddings = Embedding(field_types.size, feat_dim, dropout=feat_emb_dropout, requires_grad=True)
        self.use_graph_encoding = use_graph_encoding

        if use_graph_encoding:
            self.feature_fusion_layer = ConcatAndProject(
                hidden_dim + feat_dim, hidden_dim, dropout=ff_dropouts[0], activation='relu')
        else:
            self.feature_fusion_layer = Feedforward(
                hidden_dim + 3 * feat_dim, hidden_dim, hidden_dim, dropouts=ff_dropouts)

        self.field_table_fusion_layer = FusionLayer(
                hidden_dim, hidden_dim, res_dropout, ff_dropouts=ff_dropouts)

    def forward(self, input_hiddens, feature_ids):
        table_masks, _ = feature_ids[3]
        field_masks = (1 - table_masks).unsqueeze(2).float()
        field_type_embeddings = self.field_type_embeddings(feature_ids[2][0]) * field_masks
        if self.use_graph_encoding:
            schema_hiddens = self.feature_fusion_layer(input_hiddens, field_type_embeddings)
        else:
            primary_key_embeddings = self.primary_key_embeddings(feature_ids[0][0]) * field_masks
            foreign_key_embeddings = self.foreign_key_embeddings(feature_ids[1][0]) * field_masks
            schema_hiddens = self.feature_fusion_layer(torch.cat([input_hiddens,
                                                                  primary_key_embeddings,
                                                                  foreign_key_embeddings,
                                                                  field_type_embeddings], dim=2))
            # schema_hiddens = self.feature_fusion_layer(input_hiddens,
            #                                            primary_key_embeddings,
            #                                            foreign_key_embeddings,
            #                                            field_type_embeddings)
        # field_table_pos = feature_ids[4][0]
        # table_hiddens = ops.batch_lookup_3D(input_hiddens, field_table_pos) * field_masks
        # table_hiddens[:, 0, :] = 0
        # schema_hiddens = self.field_table_fusion_layer(schema_hiddens, table_hiddens)
        return schema_hiddens


class SchemaAwareTransformerEncoder(nn.Module):
    """
    DB-schema-aware Transformer Encoder.
    """
    def __init__(self, in_vocab, out_vocab, input_dim, hidden_dim, decoder_hidden_dim, num_layers,
                 num_const_attn_layers, rnn_layer_dropout, rnn_weight_dropout, feat_emb_dropout, res_dropout,
                 ff_dropouts, use_lstm_encoder=False, use_meta_data_encoding=False, use_graph_encoding=False):
        super().__init__()
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_lstm_encoder = use_lstm_encoder
        self.use_meta_data_encoding = use_meta_data_encoding
        self.use_graph_encoding = use_graph_encoding
        if self.use_lstm_encoder:
            self.bilstm_encoder = RNNEncoder(input_dim, hidden_dim, num_layers, rnn_layer_dropout, rnn_weight_dropout)
            self.text_encoder = RNNEncoder(hidden_dim, hidden_dim, num_layers, rnn_layer_dropout, rnn_weight_dropout)
        else:
            self.hidden_proj = Linear(self.hidden_dim, 2*self.decoder_hidden_dim)
        if num_const_attn_layers > 0:
            self.constant_encoder = SelfAttentionLayer(self.hidden_dim, self.hidden_dim, num_heads=8,
                                                       res_dropout=res_dropout, ff_dropouts=ff_dropouts)
        else:
            self.constant_encoder = None
        self.num_const_attn_layers = num_const_attn_layers
        if self.use_meta_data_encoding:
            self.schema_encoder = SchemaEncoder(self.hidden_dim, self.hidden_dim, feat_emb_dropout, res_dropout,
                                                ff_dropouts, use_graph_encoding=use_graph_encoding)

    def forward(self, inputs_embedded, input_masks, text_masks, schema_masks, feature_ids,
                transformer_output_value_masks=None, return_separate_hiddens=False):
        """
        :param inputs: [batch_size, seq_len]
        :param inputs_embedded: [batch_size, seq_len, input_dim]
        :param input_masks: [batch_size, seq_len]
            binary mask where the padding entries are set to 1
            if set to None,
                use an a zero mask (equivalent to no padding)
        :param text_masks: [batch_size, max_text_seq_len]
        :param schema_masks: [batch_size, seq_len]
        :param feature_ids: [batch_size, schema_seq_len]
        :param transformer_output_value_masks: [batch_size, seq_len]
        :param return_separate_hiddens: If set, return separate text and schema hiddens also.
        """
        if self.use_lstm_encoder:
            encoder_base_hiddens, _ = self.bilstm_encoder(inputs_embedded, input_masks)
        else:
            encoder_base_hiddens = inputs_embedded

        # -- Text Encoder
        # [batch_size, seq_len]
        # [CLS] + text + [SEP] + schema + [SEP]
        # 0 1 1 1 ... 0 0 0
        text_start_offset = 1
        # [batch_size, text_size, hidden_size]
        text_embedded = encoder_base_hiddens[:, text_start_offset:text_masks.size(1) + text_start_offset, :]
        if self.use_lstm_encoder:
            text_hiddens, hidden = self.text_encoder(text_embedded, text_masks)
        else:
            text_hiddens = text_embedded
            hidden = torch.split(self.hidden_proj(text_hiddens[:, -1, :]).unsqueeze(0), self.decoder_hidden_dim, dim=2)
            hidden = (hidden[0].contiguous(), hidden[1].contiguous())

        if transformer_output_value_masks is not None:
            values_embedded, values_masks = ops.batch_binary_lookup_3D(
                encoder_base_hiddens, transformer_output_value_masks, pad_value=0)
            constant_hiddens, constant_hidden_masks = ops.merge_padded_seq_3D(
                text_hiddens, text_masks, values_embedded, values_masks)
            if self.num_const_attn_layers > 0:
                for i in range(self.num_const_attn_layers):
                    constant_hiddens, _ = self.constant_encoder(constant_hiddens, constant_hidden_masks)
        else:
            constant_hiddens = text_hiddens
            constant_hidden_masks = text_masks

        # -- Schema Encoder
        # [batch_size, schema_size]
        schema_hiddens, schema_hidden_masks = ops.batch_binary_lookup_3D(
            encoder_base_hiddens, schema_masks, pad_value=0)
        if self.use_meta_data_encoding:
            schema_hiddens = self.schema_encoder(schema_hiddens, feature_ids)

        if return_separate_hiddens:
            return constant_hiddens, constant_hidden_masks, schema_hiddens, schema_hidden_masks, hidden
        else:
            # -- Merge text and schema encodings
            encoder_hiddens, encoder_hidden_masks = ops.merge_padded_seq_3D(
                constant_hiddens, constant_hidden_masks, schema_hiddens, schema_hidden_masks)
            return encoder_hiddens, encoder_hidden_masks, constant_hidden_masks, schema_hidden_masks, hidden


class BridgeDecoder(RNNDecoder):
    """
    Bridge Decoder.
    The decoder operates auto-regressively.
    At each step, it makes a three-way decision of
        (1) Outputs a SQL keyword
        (2) Copying a DB schema element (table or column)
        (3) Copy a text span
    """

    def __init__(self, input_dim, hidden_dim, num_layers, vocab, rnn_layer_dropout, rnn_weight_dropout,
                 ff_dropouts, attn_params):
        super().__init__(input_dim, hidden_dim, num_layers, vocab, rnn_layer_dropout, rnn_weight_dropout,
                         ff_dropouts, attn_params)
        query_dim = attn_params[0][0]
        context_dim = sum([v_dim for _, v_dim, _, _ in attn_params])
        self.pointer_switch = PointerSwitch(query_dim=query_dim, key_dim=context_dim, input_dropout=ff_dropouts[0])

    def forward(self, input_embedded, hidden, encoder_hiddens, encoder_hidden_masks, pointer_context=None,
                vocab_masks=None, memory_masks=None, encoder_ptr_value_ids=None, decoder_ptr_value_ids=None,
                last_output=None):
        """
        :param input_embedded: [batch_size, seq_len(=1), input_dim]
        :param hidden: (h, c)
            h - [num_layers, batch_size, hidden_dim]
            c - [num_layers, batch_size, hidden_dim]
        :param pointer_context
            p_pointer - [batch_size, seq_len(=1), 1]
            attn_weights - [batch_size, num_head, seq_len(=1), attn_value_dim]
        :param encoder_hiddens: [batch_size, encoder_seq_len, hidden_dim]
        :param encoder_hidden_masks: [batch_size, encoder_seq_len]
        :param pointer_context:
        :param vocab_masks: [batch_size, vocab_size] binary mask in which the banned vocab entries are set to 0 and the
            rest are set to 1.
        :param memory_masks: [batch_size, memory_seq_len] binary mask in which the banned memory entries are set
            to 0 and the rest are set to 1.
        :param encoder_ptr_value_ids: [batch_size, encoder_seq_len] mapping element in the memory to the
            pointing-generating vocabulary. If None, the pointing and generating libraries do not overlap.
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
        assert (encoder_hiddens.size(1) == encoder_ptr_value_ids.size(1))
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
                last_output = decoder_ptr_value_ids[:, i:i + 1]
            else:
                assert(last_output is not None)

            # [batch_size, encoder_seq_len]x
            select_attn = selective_read(encoder_ptr_value_ids, encoder_hiddens,
                                         self.merge_multi_head_attention(attn_weights), last_output)
            # [batch_size, 1, input_dim + self.attn_value_dim]
            input_sa = torch.cat([input_, p_pointer * select_attn], dim=2)
            output, hidden = self.rnn(input_sa, hidden)
            if self.training and self.return_hiddens:
                hiddens.append(hidden)
            # a) compute attention vector and attention weights
            # [batch_size, 1, attn_value_dim], [batch_size, num_head, 1, encoder_seq_len]
            attn_vec, attn_weights = self.attn(output, encoder_hiddens, encoder_hiddens, encoder_hidden_masks)
            # b) compute pointer-generator switch
            # [batch_size, 1, 3]
            p_pointer = self.pointer_switch(output, attn_vec)
            # c) update output state
            output = self.attn_combine(output, attn_vec)
            # d.1) compute generation prob
            # [batch_size, 1, output_vocab_size]
            gen_logit = self.out(output)
            gen_prob = torch.exp(gen_logit)
            # TODO: vocab_mask implementation in progress
            # assert(vocab_masks is None)
            # if vocab_masks is not None:
            #     gen_prob *= vocab_masks.float().unsqueeze(1)
            # d.2) compute schema element pointing prob

            # d.3) compute text span pointing prob
 
            # d.4) merge d.1, d.2 and d.3
            # [batch_size, 1, output_vocab_size + max_in_seq_len]
            point_prob = self.merge_multi_head_attention(attn_weights)
            if memory_masks is not None:
                point_prob *= memory_masks.float().unsqueeze(1)
            weighted_point_prob = p_pointer * point_prob
            if encoder_ptr_value_ids is None:
                point_gen_prob = torch.cat([(1 - p_pointer) * gen_prob, weighted_point_prob], dim=2)
            else:
                gen_prob_zeros_pad = ops.zeros_var_cuda((batch_size, 1, encoder_hiddens.size(1)))
                weighted_gen_prob = torch.cat([(1 - p_pointer) * gen_prob, gen_prob_zeros_pad], dim=2)
                point_gen_prob = weighted_gen_prob.scatter_add_(index=encoder_ptr_value_ids.unsqueeze(1),
                                                                src=weighted_point_prob, dim=2)
            point_gen_logit = ops.safe_log(point_gen_prob)

            outputs.append(point_gen_logit), seq_attn_weights.append(attn_weights),\
            seq_p_pointers.append(p_pointer)

        if self.training and self.return_hiddens:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2)), \
                   self.cat_lstm_hiddens(hiddens)
        else:
            return torch.cat(outputs, dim=1), hidden, \
                   (torch.cat(seq_p_pointers, dim=1), torch.cat(seq_attn_weights, dim=2))

    def get_input_feed(self, input, memory_inputs):
        """
        Get the input indices accepted by the decoder during autoregressuve decoding.
        """
        # set copied tokens to UNK
        vocab_mask = (input < self.vocab_size).long()
        point_mask = 1 - vocab_mask
        memory_pos = (input - self.vocab_size) * point_mask
        point_input = ops.batch_lookup(memory_inputs, memory_pos, vector_output=False)
        input_feed = vocab_mask * input + (1 - vocab_mask) * point_input
        return input_feed
