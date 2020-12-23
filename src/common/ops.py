"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Pytorch functions.
"""

import itertools
import numpy as np

import torch
import torch.nn as nn


EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31


def merge_padded_seq_3D(hiddens1, masks1, hidden2, masks2):
    batch_size = len(hiddens1)
    seq_len1 = masks1.size(1) - masks1.sum(dim=1)
    seq_len2 = masks2.size(1) - masks2.sum(dim=1)
    merged_size = seq_len1 + seq_len2
    max_merged_size = int(merged_size.max())
    res1 = max_merged_size - hiddens1.size(1)
    merged_hiddens = torch.cat([hiddens1, zeros_var_cuda([batch_size, res1, hiddens1.size(2)])], dim=1)
    scatter_index2 = seq_len1.unsqueeze(1) + batch_arange_cuda(batch_size, hidden2.size(1))
    scatter_index_masks2 = (scatter_index2 < max_merged_size)
    scatter_index2 *= scatter_index_masks2.long()
    merged_hiddens.scatter_add_(index=scatter_index2.unsqueeze(2).expand_as(hidden2),
                                src=hidden2 * scatter_index_masks2.unsqueeze(2).float(), dim=1)
    merged_hidden_masks = batch_arange_cuda(batch_size, max_merged_size) >= merged_size.unsqueeze(1)
    return merged_hiddens, merged_hidden_masks


def batch_lookup(M, idx, vector_output=True):
    """
    Perform batch lookup on matrix M using indices idx.
    :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
    :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
    :param vector_output: If set, return a 1-D vector when sample size is 1.
    :return samples: [batch_size, sample_size] where samples[i, j] = M[i, idx[i, j]]
    """
    batch_size, w = M.size()
    batch_size2, sample_size = idx.size()
    assert(batch_size == batch_size2)

    if sample_size == 1 and vector_output:
        samples = torch.gather(M, 1, idx).view(-1)
    else:
        samples = torch.gather(M, 1, idx)
    return samples


def batch_lookup_3D(M, idx):
    """
    Perform batch look up on a 3D tensor M using indices idx.
    :param M: [batch_size, seq_len, dim] Each row of M is an independent feature set.
    :param idx: [batch_size, sample_size] Each row of idx is the indices of the items in the corresponding row of M.
    :return features: [batch_size, sample_size, dim] where samples[i, j, k] = M[i, idx[i, j], k]
    """
    batch_size, seq_len, dim = M.size()
    _, sample_size = idx.size()
    M = M.view(batch_size*seq_len, dim)
    offset = long_var_cuda(torch.arange(batch_size).unsqueeze(1))
    idx = idx + offset * seq_len
    idx = idx.view(-1)
    # [batch_size*sample_size, dim]
    features = torch.index_select(M, 0, idx)
    return features.view(batch_size, sample_size, dim)


def batch_binary_lookup(M, b_idx, pad_value):
    """
    Perform batch look up on a 2D tensor M using a binary mask.
    :param M: [batch_size, seq_len]
    :param b_idx: [batch_size, seq_len]
    :return output: [batch_size, sum(b_idx == 1)]
    """
    batch_size = M.size(0)
    seq_len = b_idx.sum(1, keepdim=True)
    max_seq_len = int(seq_len.max())
    output_masks = batch_arange_cuda(batch_size, max_seq_len) >= seq_len
    pad_len = max_seq_len - seq_len
    max_pad_len = int(pad_len.max())
    M = torch.cat([M, fill_var_cuda([batch_size, max_pad_len], pad_value, dtype=M.dtype)], dim=1)
    pad_b_idx = batch_arange_cuda(batch_size, max_pad_len) < pad_len
    b_idx = torch.cat([b_idx, pad_b_idx], dim=1)
    output = M[b_idx].view(batch_size, max_seq_len)
    return output, output_masks


def batch_binary_lookup_3D(M, b_idx, pad_value):
    """
    Perform batch look up on a 3D tensor M using a binary mask.
    :param M: [batch_size, seq_len, dim] Each row of M is an independent feature set.
    :param b_idx: [batch_size, seq_len] The 1s in each row in idx indicates the features to be selected from M.
    :return output: [batch_size, sum(b_idx == 1), dim]
    """
    # Pad binary indices
    batch_size = M.size(0)
    hidden_dim = M.size(2)
    seq_len = b_idx.sum(1, keepdim=True)
    max_seq_len = int(seq_len.max())
    output_masks = batch_arange_cuda(batch_size, max_seq_len) >= seq_len
    pad_len = max_seq_len - seq_len
    max_pad_len = int(pad_len.max())
    M = torch.cat([M, fill_var_cuda([batch_size, max_pad_len, hidden_dim], pad_value, dtype=M.dtype)], dim=1)
    pad_b_idx = batch_arange_cuda(batch_size, max_pad_len) < pad_len
    b_idx = torch.cat([b_idx, pad_b_idx], dim=1)
    output = M[b_idx].view(batch_size, max_seq_len, hidden_dim)
    return output, output_masks


def soft_embedding_lookup(embeddings, prob):
    """
    Warning: only apply on embeddings of small size. Otherwise this operation will be very costly.
    """
    input_dim = prob.dim()
    if input_dim == 3:
        assert(prob.size(1) == 1)
        prob = prob.squeeze(1)
    vocab_size = prob.size(-1)
    indices = arange_cuda(vocab_size)
    embedded = embeddings(indices)
    soft_embedded = torch.matmul(prob, embedded)
    if input_dim == 3:
        return soft_embedded.unsqueeze(1)
    else:
        return soft_embedded


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    elif x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze_(-2)
    elif x.dim() - 1 == y.dim():
        return (x @ y.unsqueeze(-2)).squeeze_(-2)
    else:
        raise AttributeError('matmul: Unmatched input dimension x: {} y: {}'.format(x.size(), y.size()))


def one_hot(x, max_size):
    y = torch.zeros(list(x.size()) + [max_size])
    y.scatter(-1, x, 1)
    return y


def to_binary_array(x, size):
    x_bin = np.zeros(size)
    x_bin[x] = 1
    return x_bin


def pack_bidirectional_lstm_state(state, num_layers):
    """
    Pack the hidden state of a BiLSTM s.t. the first dimension equals to the number of layers.
    """
    assert (len(state) == 2 * num_layers)
    _, batch_size, hidden_dim = state.size()
    layers = state.view(num_layers, 2, batch_size, hidden_dim).transpose(1, 2).contiguous()
    state = layers.view(num_layers, batch_size, -1)
    return state


def unpack_bidirectional_lstm_state(state, num_directions=2):
    """
    Unpack the packed hidden state of a BiLSTM s.t. the first dimension equals to the number of layers multiplied by
    the number of directions.
    """
    batch_size = state.size(1)
    new_hidden_dim = int(state.size(2) / num_directions)
    return torch.stack(torch.split(state, new_hidden_dim, dim=2), dim=1).view(-1, batch_size, new_hidden_dim)


def pad_and_cat(a, padding_value, padding_dim=1, dtype=torch.long, fill_empty_batch=True, return_masks=False):

    def vectorize(a):
        if dtype == torch.uint8:
            a = [byte_var_cuda(x) for x in a]
        elif dtype == torch.int:
            a = [int_var_cuda(x) for x in a]
        elif dtype == torch.long:
            a = [long_var_cuda(x) for x in a]
        else:
            a = [var_cuda(x) for x in a]
        return a

    if not list(itertools.chain(*a)):
        # "a" contains only empty vectors
        if fill_empty_batch:
            a[0].append(padding_value)
        else:
            return a
    if type(a[0]) is list or type(a[0]) is np.ndarray:
        a = vectorize(a)
    if a[0].dim() == 1:
        a = [x.unsqueeze(0) for x in a]
    max_dim_size = max([x.size()[padding_dim] for x in a])

    if return_masks:
        masks = byte_ones_var_cuda([len(a), max_dim_size])
    padded_a = []
    for i, x in enumerate(a):
        if return_masks:
            masks[i][:x.size()[padding_dim]] = 0
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[padding_dim]
            if x.dim() == 2:
                padded_a.append(pad_1d_right(x, res_len, padding_value))
            elif x.dim() == 3:
                padded_a.append(torch.cat([x, fill_var_cuda([x.size(0), res_len, x.size(2)], padding_value)],
                                          dim=padding_dim))
            else:
                raise NotImplementedError
        else:
            padded_a.append(x)
    padded_a = torch.cat(padded_a, dim=0)

    if return_masks:
        return padded_a, masks
    else:
        return padded_a


def pad_and_cat_2d(a, padding_value, padding_dim=2, dtype=torch.long, list_padding_value=None, fill_empty_batch=True):
    if not list(itertools.chain(*a)):
        # "a" contains only empty vectors
        if fill_empty_batch:
            a[0].append([padding_value])
        else:
            return a
    batch_size = len(a)
    w = max([len(l) for l in a])
    padded_a = []
    if list_padding_value is None:
        list_padding_value = padding_value
    for l in a:
        if len(l) < w:
            padded_a.append(l + [[list_padding_value]] * (w - len(l)))
        else:
            padded_a.append(l)
    flat_a = [x for l in padded_a for x in l]
    padded_flat_a = pad_and_cat(flat_a, padding_value, padding_dim=(padding_dim-1), dtype=dtype)
    return padded_flat_a.view(batch_size, w, -1)


def pad_and_cat_matrices(a, padd_value, dtype=torch.long):

    def vectorize(a):
        if dtype == torch.uint8:
            a = [byte_var_cuda(x) for x in a]
        elif dtype == torch.int:
            a = [int_var_cuda(x) for x in a]
        elif dtype == torch.long:
            a = [long_var_cuda(x) for x in a]
        else:
            a = [var_cuda(x) for x in a]
        return a

    if type(a[0]) is list or type(a[0]) is np.ndarray:
        a = vectorize(a)

    max_x_size = max([x.size(0) for x in a])
    max_y_size = max([y.size(0) for y in a])

    padded_a = []
    for x in a:
        x_size, y_size = x.size()
        x_res = max_x_size - x_size
        y_res = max_y_size - y_size
        pad = nn.ConstantPad2d((0, x_res, 0, y_res), padd_value)
        padded_a.append(pad(x))

    padded_a = [x.unsqueeze(0) for x in padded_a]
    return torch.cat(padded_a, dim=0)


def pad_1d_right(x, padding_size, padding_value):
    pad = nn.ConstantPad1d((0, padding_size), padding_value)
    return pad(x)


def pad_1d_left(x, padding_size, pad_id):
    pad = nn.ConstantPad1d((padding_size, 0), pad_id)
    return pad(x)


def right_shift_pad(x, pad_id):
    if x.size(1) == 1:
        return int_fill_var_cuda(x.size(), pad_id)
    return pad_1d_left(x[:, :-1], 1, pad_id)


def left_shift_pad(x, pad_id):
    if x.size(1) == 1:
        return int_fill_var_cuda(x.size(), pad_id)
    return pad_1d_right(x[:, 1:], 1, pad_id)


def pad_batch(batch_seq_ids, pad_id, dtype=torch.long):
    padded_seq = pad_and_cat(batch_seq_ids, pad_id, dtype=dtype)
    pad_mask = (padded_seq == pad_id)
    return padded_seq, pad_mask


def pad_batch_2D(batch_2D_seq_ids, pad_id, dtype=torch.long, output_2d_tensor=False):
    padded_2D_seq = pad_and_cat_2d(batch_2D_seq_ids, pad_id, dtype=dtype)
    pad_mask = (padded_2D_seq == pad_id)
    if output_2d_tensor:
        batch_size = padded_2D_seq.size(0)
        padded_2D_seq = padded_2D_seq.view(batch_size, -1)
        pad_mask = pad_mask.view(batch_size, -1)
    return padded_2D_seq, pad_mask


def tile_along_beam(x, beam_size, dim=0):
    bs = x.size(dim)
    tile_indices = arange_cuda(bs).view(bs, 1).repeat(1, beam_size).view(bs*beam_size)
    return torch.index_select(x, dim, tile_indices)
    # batch_size = len(x)
    # full_size = batch_size * beam_size
    # if x.dim() == 1:
    #     return x.usqueeze(1).repeat(-1, beam_size).continguous().view(-1)
    # if x.dim() == 2:
    #     return x.unsqueeze(1).repeat(-1, beam_size, -1).continguous().view(full_size, -1)
    # if x.dim() == 3:
    #     dim1 = x.size(1)
    #     return x.unsqueeze(1).repeat(-1, beam_size, -1, -1).continguous().view(full_size, dim1, -1)
    # if x.dim() == 4:
    #     _, dim1, dim2, _ = x.size(1)
    #     return x.unsqueeze(1).repeat(-1, beam_size, -1, -1, -1).continguous().view(full_size, dim1, dim2, -1)
    # raise NotImplementedError


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1), dtype=torch.float)
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return encodings


def arange_cuda(x, dtype=torch.long):
    return torch.arange(x, dtype=dtype).cuda()


def batch_arange_cuda(batch_size, x, dtype=torch.long):
    return zeros_var_cuda(batch_size, dtype=dtype).unsqueeze(1) + \
           arange_cuda(x, dtype=dtype).unsqueeze(0)


def byte_ones_var_cuda(s, requires_grad=False):
    return torch.ones(s, dtype=torch.uint8, requires_grad=requires_grad).cuda()


def ones_var_cuda(s, requires_grad=False, dtype=torch.float32):
    return torch.ones(s, requires_grad=requires_grad, dtype=dtype).cuda()


def int_ones_var_cuda(s, requires_grad=False):
    return torch.ones(s, dtype=torch.long, requires_grad=requires_grad).cuda()


def zeros_like_cuda(x, requires_grad=False, dtype=torch.float32):
    return torch.zeros_like(x, requires_grad=requires_grad, dtype=dtype).cuda()


def byte_zeros_var_cuda(s, requires_grad=False):
    return torch.zeros(s, dtype=torch.uint8, requires_grad=requires_grad).cuda()


def zeros_var_cuda(s, requires_grad=False, dtype=torch.float32):
    return torch.zeros(s, requires_grad=requires_grad, dtype=dtype).cuda()


def int_zeros_var_cuda(s, requires_grad=False):
    return torch.zeros(s, dtype=torch.long, requires_grad=requires_grad).cuda()


def int_fill_var_cuda(s, value, requires_grad=False):
    return torch.zeros(s, dtype=torch.long, requires_grad=requires_grad).cuda() + value


def fill_var_cuda(s, value, dtype=None, requires_grad=False):
    return torch.zeros(s, dtype=dtype, requires_grad=requires_grad).cuda() + value


def byte_var_cuda(x, requires_grad=False):
    tx = torch.ByteTensor(x).cuda()
    if requires_grad:
        tx.requires_grad_()
    return tx


def int_var_cuda(x, requires_grad=False):
    tx = torch.IntTensor(x).cuda()
    if requires_grad:
        tx.requires_grad_()
    return tx


def long_var_cuda(x, requires_grad=False):
    tx = torch.LongTensor(x).cuda()
    if requires_grad:
        tx.requires_grad_()
    return tx


def var_cuda(x, requires_grad=False):
    tx = torch.Tensor(x).cuda()
    if requires_grad:
        tx.requires_grad_()
    return tx


def var(x, requires_grad=False):
    tx = torch.Tensor(x)
    if requires_grad:
        tx.requires_grad_()
    return tx


def var_to_numpy(x):
    if type(x) is list:
        return [xx.cpu().numpy() for xx in x]
    else:
        return x.cpu().numpy()


def safe_log(x):
    return torch.log(x + EPSILON) # - (x == 0).float() * 1024


def initialize_module(mdl, method='xavier'):
    print('Model initialization ({})'.format(method))
    print('--------------------------')
    num_display = 500
    count = 0
    if method == 'xavier':
        for name, param in mdl.named_parameters():
            if 'trans_parameters' in name:
                print('{} (skipped)'.format(name))
                continue
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                if count < num_display:
                    print(name)
            elif 'weight' in name or name.endswith('embeddings'):
                nn.init.xavier_normal_(param)
                if count < num_display:
                    print('{} done'.format(name))
            count += 1
    if count >= num_display:
        print('...')
    print('--------------------------')
