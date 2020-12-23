"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Visualization saver.
"""

import os
import pickle

from src.data_processor.path_utils import safe_mkdir_hier


class LayerVisualizationDataWriter(object):

    def __init__(self, log_dir, verbose=False):
        self.log_dir = log_dir
        self.verbose = verbose
        if not os.path.exists(self.log_dir):
            safe_mkdir_hier('.', self.log_dir)

    def save_cross_attention(self, vis_data, attn_target):
        out_pkl = os.path.join(self.log_dir, '{}_attention.pkl'.format(attn_target))
        with open(out_pkl, 'wb') as o_f:
            pickle.dump(vis_data, o_f)
        if self.verbose:
            print('* {} attention visualization saved to {}'.format(attn_target, out_pkl))

    def save_pointer(self, vis_data, pointer_target):
        out_pkl = os.path.join(self.log_dir, '{}_pointer.pkl'.format(pointer_target))
        with open(out_pkl, 'wb') as o_f:
            pickle.dump(vis_data, o_f)
        if self.verbose:
            print('* {} pointer visualization saved to {}'.format(pointer_target, out_pkl))
