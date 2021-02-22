"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Customized nn modules.
 Code adapted from https://github.com/salesforce/decaNLP/blob/master/models/common.py
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from transformers import BertModel, RobertaModel
import src.common.ops as ops


class Embedding(nn.Module):

    def __init__(self, vocab_size, vocab_dim, dropout, requires_grad, pretrained_vocab_vectors=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.embeddings = nn.Embedding(vocab_size, vocab_dim)
        self.embeddings.weight.requires_grad = requires_grad
        self.dropout = nn.Dropout(dropout)
        if pretrained_vocab_vectors is not None:
            assert(vocab_size == pretrained_vocab_vectors.size(0))
            assert(vocab_dim == pretrained_vocab_vectors.size(-1))
            self.embeddings.weight.data = pretrained_vocab_vectors

    def forward(self, x):
        return self.dropout(self.embeddings(x))

    def set_embeddings(self, W):
        self.embeddings.weight.data = W


class TransformerHiddens(nn.Module):
    """
    Pre-trained BERT contextualized embeddings.
    """
    def __init__(self, model, dropout=0.0, requires_grad=False):
        super().__init__()
        if model.startswith('bert'):
            self.trans_parameters = BertModel.from_pretrained(model)
        elif model.startswith('roberta'):
            self.trans_parameters = RobertaModel.from_pretrained(model)
        elif model == 'table-bert':
            self.trans_parameters = BertModel.from_pretrained(os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'utils/trans/table-bert-checkpoint'
            ))
        else:
            return NotImplementedError
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.requires_grad = requires_grad

    def forward(self, inputs, input_masks, segments=None, position_ids=None, output_all_encoded_layers=False):
        last_hidden_states, pooler_output = (self.trans_parameters(
            inputs, token_type_ids=segments, position_ids=position_ids, attention_mask=(~input_masks)))
        return self.dropout(last_hidden_states), pooler_output


class WeightDropoutLSTM(nn.Module):
    """
    A wrapper class that implements weight dropout LSTM.
    """
    def __init__(self, input_size, hidden_size, bidirectional=False, num_layers=1, dropout=0.0,
                 rnn_weight_dropout=0.0, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirecitonal = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_weight_dropout = rnn_weight_dropout
        base_rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        if rnn_weight_dropout > 0:
            self.rnn = WeightDrop(base_rnn,
                [name for name, _ in base_rnn.named_parameters() if name.startswith('weight_hh_l')],
                dropout=rnn_weight_dropout)
        else:
            self.rnn = base_rnn

    def forward(self, inputs, hidden=None):
        return self.rnn(inputs, hidden)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            if not self.training:
                w = w.data
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class LSTMWithPacking(nn.Module):
    """
    A wrapper class that packs the input sequences and unpacks output sequences of LSTMs.
    """
    def __init__(self, input_dim, output_dim, bidirectional=False, num_layers=1, dropout=0.0, rnn_weight_dropout=0.0,
                 batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        hidden_dim = int(output_dim / self.num_directions)
        self.rnn = WeightDropoutLSTM(input_dim,
                                     hidden_dim,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     rnn_weight_dropout=rnn_weight_dropout,
                                     bidirectional=bidirectional,
                                     batch_first=batch_first)


    def forward(self, inputs, input_sizes, hidden=None):
        # Pack the input
        sorted_input_sizes, sorted_indices = torch.sort(input_sizes, descending=True)
        inputs = inputs[sorted_indices] if self.batch_first else inputs[:, sorted_indices]
        # [batch_size, seq_len, input_dim]
        packed_inputs = pack(inputs, sorted_input_sizes, batch_first=self.batch_first)

        outputs, (h, c) = self.rnn(packed_inputs, hidden)

        # Unpack
        outputs, _ = unpack(outputs, batch_first=self.batch_first)
        _, rev_indices = torch.sort(sorted_indices, descending=False)
        # [batch_size, seq_len, num_directions*hidden_dim]
        outputs = outputs[rev_indices] if self.batch_first else outputs[:, rev_indices]
        assert(outputs.size(2) == self.output_dim)
        # [num_layers*num_directions, batch_size, hidden_size]
        h = h[:, rev_indices, :]
        assert(h.size(0) == self.num_layers * self.num_directions)
        assert(h.size(2) == self.rnn.hidden_size)
        # [num_layers*num_directions, batch_size, hidden_size]
        c = c[:, rev_indices, :]
        assert (c.size(0) == self.num_layers * self.num_directions)
        assert (c.size(2) == self.rnn.hidden_size)

        return outputs, (h, c)


class LogSoftmaxOutput(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = Linear(input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x))


class LayerNorm(nn.Module):

    def __init__(self, layer_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(layer_dim))
        self.beta = nn.Parameter(torch.zeros(layer_dim))
        self.eps = ops.EPSILON

    def forward(self, x):
        x1 = None
        if type(x) is tuple:
            assert(len(x) == 2)
            x = x[0]
            x1 = x[1]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        if x1 is not None:
            return output, x1
        else:
            return output


class ResidualConnectionWrapper(nn.Module):

    def __init__(self, mdl, dropout):
        super().__init__()
        self.mdl = mdl
        res_dropout, layer_dropout = dropout
        self.res_dropout = nn.Dropout(res_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)

    def forward(self, *args, **kwargs):
        outputs = self.mdl(*args, **kwargs)
        if type(outputs) is tuple:
            assert(len(outputs) == 2)
            return self.res_dropout(args[0]) + self.layer_dropout(outputs[0]), outputs[1]
        else:
            return self.res_dropout(args[0]) + self.layer_dropout(outputs)


class ResidualBlock(nn.Module):
    """
    Residual Connection with Layernorm.
    """
    def __init__(self, mdl, output_dim, dropout, ignore_arg_indices=None):
        super().__init__()
        self.mdl_with_residual_connection = ResidualConnectionWrapper(mdl, dropout)
        self.layer_norm = LayerNorm(output_dim)
        self.ignore_arg_indices = ignore_arg_indices

    def forward(self, *args, **kwargs):
        return self.layer_norm(self.mdl_with_residual_connection(*args, **kwargs))


class FusionLayer(nn.Module):
    def __init__(self, query_dim, value_dim, dropout, ff_dropouts=(0.0, 0.0, 0.0)):
        assert(query_dim == value_dim)
        hidden_dim = query_dim
        super().__init__()
        res_dropout, layer_dropout = dropout
        self.res_dropout = nn.Dropout(res_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.layer_norm = LayerNorm(hidden_dim)
        self.res_feed_forward = ResidualBlock(
            Feedforward(query_dim, hidden_dim, value_dim, ff_dropouts), value_dim, dropout)

    def forward(self, x, feat):
        additive_fuse = self.layer_norm(self.res_dropout(x) + self.layer_dropout(feat))
        return self.res_feed_forward(additive_fuse)


class ConcatAndProject(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation=None, bias=True):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.linear1 = Linear(input_dim, output_dim, bias=bias)
        self.activation = activation

    def forward(self, *args):
        input = self.input_dropout(torch.cat(args, dim=-1))
        if self.activation is None:
            return self.linear1(input)
        else:
            return getattr(torch, self.activation)(self.linear1(input))


class Linear(nn.Linear):
    """
    Apply linear projection to the last dimention of a tensor.
    """
    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):
    """
    Apply feedforward computation to the last dimension of a tensor.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropouts, activation='relu', bias1=True, bias2=True):
        super().__init__()
        self.activation = activation 
        self.input_dropout = nn.Dropout(dropouts[0])
        self.hidden_dropout = nn.Dropout(dropouts[1])
        self.linear1 = Linear(input_dim, hidden_dim, bias=bias1)
        self.linear2 = Linear(hidden_dim, output_dim, bias=bias2)

    def forward(self, x):
        return self.linear2(
                 self.hidden_dropout(
                   getattr(torch, self.activation)(
                     self.linear1(
                        self.input_dropout(x)))))


class AttentionDotProduct(nn.Module):

    def __init__(self, dropout, causal, return_attn_vec=True, return_normalized_weights=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.return_attn_vec = return_attn_vec
        self.return_normalized_weights = return_normalized_weights

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size, query_seq_len, query_dim]
        :param key: [batch_size, key_seq_len, key_dim]
        :param value: [batch_size, key_seq_len, value_dim]
        :param mask: [batch_size, key_seq_len]
            binary mask where the padding entries are set to 1
        :return: attn_vec: [batch_size, query_seq_len, value_dim]
        """
        # [batch_size, query_seq_len, key_seq_len]
        attn_weights = ops.matmul(query, key.transpose(1, 2))
        if (query.size(1) == key.size(1)) and self.causal:
            causal_mask = ops.fill_var_cuda((query.size(1), key.size(1)), 1).triu(1)
            attn_weights -= causal_mask.unsqueeze(0) * ops.HUGE_INT
        if mask is not None:
            attn_weights.data.masked_fill_(mask.unsqueeze(1).expand_as(attn_weights), -ops.HUGE_INT)
        attn_weights /= np.sqrt(key.size(-1))
        if self.return_normalized_weights:
            attn_weights = F.softmax(attn_weights, -1)

        if self.return_attn_vec:
            assert(self.return_normalized_weights)
            # [batch_size, query_seq_len, value_dim]
            attn_vec = ops.matmul(attn_weights, self.dropout(value))
            return attn_vec, attn_weights
        else:
            return attn_weights


class Attention(nn.Module):
    """
    Attention module where the attention weights are computed w/ a feedforward network.
    """
    def __init__(self, query_dim, key_dim, ff_dropouts, dropout, causal, return_attn_vec=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.ffn = Feedforward(query_dim + key_dim, key_dim, 1, ff_dropouts, activation='tanh', bias1=True, bias2=False)
        self.return_attn_vec = return_attn_vec

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size, query_seq_len, query_dim]
        :param key: [batch_size, key_seq_len, key_dim]
        :param value: [batch_size, key_seq_len, value_dim]
        :param mask: [batch_size, key_seq_len]
            binary mask where the padding entries are set to 1
        :return: attn_vec: [batch_size, query_seq_len, value_dim]
        """
        # [batch_size, query_seq_len, key_seq_len]
        batch_size = query.size(0)
        query_seq_len = query.size(1)
        key_seq_len = key.size(1)
        tiled_seq_len = query_seq_len * key_seq_len
        tiled_query = query.unsqueeze(2).repeat(1, 1, key_seq_len, 1).view(batch_size, tiled_seq_len, -1)
        tiled_key = key.repeat(1, query_seq_len, 1)
        attn_weights = self.ffn(torch.cat([tiled_query, tiled_key], dim=2)).view(batch_size, query_seq_len, key_seq_len)

        if (query.size(1) == key.size(1)) and self.causal:
            causal_mask = ops.fill_var_cuda((query.size(1), key.size(1)), 1).triu(1)
            attn_weights -= causal_mask.unsqueeze(0) * ops.HUGE_INT
        if mask is not None:
            attn_weights.data.masked_fill_(mask.unsqueeze(1).expand_as(attn_weights), -ops.HUGE_INT)
        attn_weights /= np.sqrt(key.size(-1))
        attn_weights = F.softmax(attn_weights, -1)

        if self.return_attn_vec:
            # [batch_size, query_seq_len, value_dim]
            attn_vec = ops.matmul(attn_weights, self.dropout(value))
            return attn_vec, attn_weights
        else:
            return attn_weights


class MultiHead(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, num_heads, attention, use_wo=True, return_attn_vec=True):
        super().__init__()
        self.attention = attention
        assert(self.attention.return_attn_vec == return_attn_vec)
        self.wq = Linear(query_dim, key_dim, bias=False)
        self.wk = Linear(key_dim, key_dim, bias=False)
        self.wv = Linear(value_dim, value_dim, bias=False)
        self.num_heads = num_heads
        self.use_wo = use_wo
        self.return_attn_vec = return_attn_vec
        if use_wo:
            self.wo = Linear(value_dim, key_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size, query_seq_len, query_dim]
        :param key: [batch_size, key_seq_len, key_dim]
        :param value: [batch_size, key_seq_len, value_dim]
        :param mask: [batch_size, key_seq_len]
            binary mask where the padding entries are set to 1
        :return: multi_head_attn: [batch_size, query_seq_len, value_dim]
        :return: multi_head_attn_weights: [batch_size, query_seq_len]
        """
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.num_heads, -1) for x in (query, key, value))
        multi_head_attn_vecs, multi_head_attn_weights = [], []
        for q, k, v in zip(query, key, value):
            if self.return_attn_vec:
                head_attn_vec, head_attn_weights = self.attention(q, k, v, mask)
                multi_head_attn_vecs.append(head_attn_vec)
            else:
                head_attn_weights = self.attention(q, k, v, mask)
            multi_head_attn_weights.append(head_attn_weights.unsqueeze(1))
        multi_head_attn_weights = torch.cat(multi_head_attn_weights, dim=1)
        if self.return_attn_vec:
            multi_head_attn = torch.cat(multi_head_attn_vecs, dim=2)
            if self.use_wo:
                return self.wo(multi_head_attn), multi_head_attn_weights
            else:
                return multi_head_attn, multi_head_attn_weights
        else:
            return multi_head_attn_weights


class SelfAttentionLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, attn_dropout=0.0, res_dropout=0.0,
                 ff_dropouts=(0.0, 0.0, 0.0)):
        super().__init__()
        attn_ = AttentionDotProduct(attn_dropout, causal=False)
        self.res_self_attn = ResidualBlock(
            MultiHead(input_dim, input_dim, input_dim, num_heads, attn_), input_dim, res_dropout,
            ignore_arg_indices=[1])
        self.res_feed_forward = ResidualBlock(
            Feedforward(input_dim, hidden_dim, hidden_dim, ff_dropouts), input_dim, res_dropout)

    def forward(self, x, mask=None, input_history=None):
        """
        :param x: [batch_size, seq_len, input_size]
        :param mask: [batch_size, seq_len]
            binary mask where the padding entries are set to 1
        :param input_history: input_history: input states before the current step. (default: None)
        :return: [batch_size, seq_len, output_size]
        """
        if input_history is not None:
            x, sa_weights = self.res_self_attn(x, input_history, input_history)
        else:
            x, sa_weights = self.res_self_attn(x, x, x, mask=mask)
        return self.res_feed_forward(x), sa_weights


class RelationalAttentionSingleHead(nn.Module):

    def __init__(self, sparse, attn_dropout=0.0):
        super().__init__()
        self.attn = AttentionDotProduct(
            attn_dropout, causal=False, return_attn_vec=False, return_normalized_weights=False)
        self.sparse = sparse

    def forward(self, query, key, value, mask, M, r_k, r_v=None):
        """
        :param query: [batch_size, query_seq_len, query_dim]
        :param key: [batch_size, key_seq_len, key_dim]
        :param value: [batch_size, key_seq_len, value_dim]
        :param mask: [batch_size, key_seq_len] binary mask where the padding entries are set to 1
        :param M: [batch_size, query_seq_len, key_seq_len] adjacency matrix
            M[i, j] represents the relationship of j -> i
                M[i, j] = 0 indicates a null relationship.
        :param r_k: [num_edge_labels, key_dim]
        :param r_v: [num_edge_labels, value_dim] If set to None, r_v = r_k.
        """
        if r_v is None:
            r_v = r_k
        # [batch_size, query_seq_len, key_seq_len]
        r_mask = (M > 0).float()
        # [batch_size, query_seq_len, key_seq_len]
        base_attn_weights = self.attn(query, key, value, mask)
        # [batch_size, query_seq_len, key_seq_len, key_dim]
        r_k_M = r_k[M] * r_mask.unsqueeze(3)
        # [batch_size, query_seq_len, key_seq_len, value_dim]
        r_v_M = r_v[M] * r_mask.unsqueeze(3)
        # [batch_size, query_seq_len, key_seq_len]
        r_attn_weights = torch.matmul(query.unsqueeze(2), r_k_M.transpose(2, 3)).squeeze(2) / np.sqrt(r_k.size(1))
        attn_weights = base_attn_weights + r_attn_weights
        attn_weights = attn_weights - (1 - r_mask) * ops.HUGE_INT
        attn_weights = base_attn_weights
        attn_weights = F.softmax(attn_weights, -1)
        # [batch_size, query_seq_len, value_dim]
        base_attn_vec = torch.matmul(attn_weights, value)
        r_attn_vec = torch.matmul(attn_weights.unsqueeze(2), r_v_M).squeeze(2)
        attn_vec = base_attn_vec + r_attn_vec
        return attn_vec, attn_weights


class RelationalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, sparse, attn_dropout=0.0, use_wo=True):
        super().__init__()
        self.attn = RelationalAttentionSingleHead(sparse, attn_dropout)
        self.wq = Linear(query_dim, key_dim, bias=False)
        self.wk = Linear(key_dim, key_dim, bias=False)
        self.wv = Linear(value_dim, value_dim, bias=False)
        self.num_heads = num_heads
        self.use_wo = use_wo
        if use_wo:
            self.wo = Linear(value_dim, key_dim, bias=False)

    def forward(self, query, key, value, mask, M, r_k, r_v=None):
        """
       :param query: [batch_size, query_seq_len, query_dim]
       :param key: [batch_size, key_seq_len, key_dim]
       :param value: [batch_size, key_seq_len, value_dim]
       :param mask: [batch_size, key_seq_len] binary mask where the padding entries are set to 1
       :param M: [batch_size, seq_len, seq_len] adjacency matrix
           M[i, j] represents the relationship of j -> i
       :param r_k: [num_edge_labels, key_dim]
       :param r_v: [num_edge_labels, value_dim] If set to None, r_v = r_k.
       """
        if r_v is None:
            r_v = r_k
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.num_heads, -1) for x in (query, key, value))
        r_k, r_v = (x.chunk(self.num_heads, -1) for x in (r_k, r_v))
        multi_head_attn_vecs, multi_head_attn_weights = [], []
        for q, k, v, rk, rv in zip(query, key, value, r_k, r_v):
            h_attn_vec, h_attn_weights = self.attn(q, k, v, mask, M, rk, rv)
            multi_head_attn_vecs.append(h_attn_vec)
            multi_head_attn_weights.append(h_attn_weights.unsqueeze(1))
        # [batch_size, query_seq_len, value_dim]
        multi_head_attn_vec = torch.cat(multi_head_attn_vecs, dim=2)
        # [batch_size, num_heads, query_seq_len, key_seq_len]
        multi_head_attn_weights = torch.cat(multi_head_attn_weights, dim=1)
        return multi_head_attn_vec, multi_head_attn_weights


class RelationalSelfAttentionLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads, sparse, attn_dropout=0.0, res_dropout=0.0,
                 ff_dropouts=(0.0, 0.0, 0.0)):
        super().__init__()
        attn_ = RelationalAttention(input_dim, input_dim, input_dim, num_heads, sparse, attn_dropout)
        self.res_self_attn = ResidualBlock(attn_, input_dim, res_dropout, ignore_arg_indices=[1])
        self.res_feed_forward = ResidualBlock(
            Feedforward(input_dim, hidden_dim, input_dim, ff_dropouts), input_dim, res_dropout)

    def forward(self, x, mask, M, r_k, r_v=None):
        x, sa_weights = self.res_self_attn(x, x, x, mask, M, r_k, r_v)
        return self.res_feed_forward(x), sa_weights


class PointerSwitch(nn.Module):

    def __init__(self, query_dim, key_dim, input_dropout):
        super().__init__()
        self.project = ConcatAndProject(query_dim + key_dim, 1, input_dropout, activation=None)

    def forward(self, query, key):
        return torch.sigmoid(self.project(query, key))


class MultiTargetPointerSwitch(nn.Module):

    def __init__(self, query_dim, key_dim, num_targets, input_dropout):
        super().__init__()
        self.project = ConcatAndProject(query_dim + key_dim, num_targets, input_dropout, activation=None)

    def forward(self, *args):
        return F.softmax(self.project(*args), dim=2)


class CoattentiveLayer(nn.Module):
    """
    Coattention.
    """
    def __init__(self, p_dim, q_dim, dropout=0.0, symmetric=False):
        super().__init__()
        self.project = Linear(q_dim, p_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.symmetric = symmetric

    def forward(self, p, q, p_masks, q_masks):
        """
        :param p: [batch_size, p_seq_len, p_dim]
        :param q: [batch_size, q_seq_len, q_dim]
        :param p_masks: [batch_size, p_seq_len]
        :param q_masks: [batch_sizse, q_seq_len]
        :return: p_to_q_attn_vec: [batch_size, p_seq_len, q_dim + p_dim]
            attention vector computed given p as the key and q as the context
        :return: q_to_p_attn_vec: [batch_size, q_seq_len, p_dim]
            attention vector computed given q as the key and p as the context
        """
        _, p_seq_len, _ = p.size()
        _, q_seq_len, _ = q.size()
        # [batch_size, q_seq_len, p_dim]
        q_proj = self.dropout(self.project(q))
        # [batch_size, p_seq_len, q_seq_len]
        affinity = p.bmm(q_proj.transpose(1,2))
        # [batch_size, p_seq_len (normalized), q_seq_len]
        attn_over_p = self.normalize(affinity, p_masks)
        # [batch_size, q_seq_len (normalized), p_seq_len]
        attn_over_q = self.normalize(affinity.transpose(1,2), q_masks)
        # [batch_size, q_seq_len, p_dim]
        q_to_p_attn_vec = self.attn(attn_over_p, p)
        # [batch_size, p_seq_len, q_dim + p_dim]
        if self.symmetric:
            p_to_q_attn_vec = self.attn(attn_over_q, q)
        else:
            p_to_q_attn_vec = self.attn(attn_over_q, torch.cat([q, q_to_p_attn_vec], dim=2))
        return p_to_q_attn_vec, q_to_p_attn_vec

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -ops.HUGE_INT)
        return F.softmax(raw_scores, dim=1)


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary entropy loss with logits with certain output positions masked.
    """
    def __init__(self, pad_id=2, weight=None, size_average=True):
        super().__init__()
        self.pad_id = pad_id
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, inputs, targets):
        assert(inputs.size() == targets.size())
        target_mask = (targets != self.pad_id)
        masked_inputs = inputs[target_mask]
        masked_targets = targets[target_mask]
        loss = self.loss(masked_inputs, masked_targets)
        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with certain output positions masked.
    """
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, inputs, targets):
        """
        :param inputs: [batch_size, seq_len, vocab_size]
        :param targets: [batch_size, seq_len]
        """
        target_mask = (targets != self.pad_idx)
        target_vec_mask = target_mask.unsqueeze(-1).expand_as(inputs)
        vocab_size = inputs.size(-1)
        masked_inputs = inputs[target_vec_mask].view(-1, vocab_size)
        masked_targets = targets[target_mask]
        if masked_targets.nelement() == 0:
            return 0
        loss = F.nll_loss(masked_inputs, masked_targets)
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        if loss > 1e8:
            import pdb
            pdb.set_trace()
        return loss


def selective_read(encoder_ptr_value_ids, memory_hiddens, attn_weights, last_output):
    """
    :param encoder_ptr_value_ids:
    :param memory_hiddens: [batch_size, seq_len, hidden_dim]
    :param attn_weights: [batch_size, 1, seq_len]
    :param last_output
    :return:
    """
    point_mask = (encoder_ptr_value_ids == last_output).float()
    weights = point_mask * attn_weights.squeeze(1)
    batch_size = memory_hiddens.size(0)
    weight_normalizer = weights.sum(dim=1)
    weight_normalizer = weight_normalizer + (weight_normalizer == 0).float() * ops.EPSILON
    return (weights.unsqueeze(2) * memory_hiddens).sum(dim=1, keepdim=True) / weight_normalizer.view(batch_size, 1, 1)
