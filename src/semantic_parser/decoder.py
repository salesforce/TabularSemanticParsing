"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Sequence decoder module.
"""

import torch
import torch.nn as nn

from src.common.nn_modules import LogSoftmaxOutput


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab, return_hiddens=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.return_hiddens = return_hiddens
        self.out = LogSoftmaxOutput(hidden_dim, vocab.size)

    def cat_lstm_hiddens(self, hiddens):
        """
        :param hiddens: hidden state output of the decoder LSTM at different steps.
        :return seq_hiddens: ([num_layers*num_directions, batch_size, seq_len, hidden_dim],
                              [num_layers*num_directions, batch_size, seq_len, hidden_dim])
        """
        return (torch.cat([h[0].unsqueeze(2) for h in hiddens], dim=2),
                torch.cat([h[1].unsqueeze(2) for h in hiddens], dim=2))

    def get_input_feed(self, *args, **kwargs):
        """
        Interface.
        """
        return
