"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Experiment Hyperparameters.
"""

import os
from src.parse_args import parser


code_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
args = ['--demo', '--ensemble_inference', '--gpu', '0']
with open(os.path.join(code_base_dir, 'configs', 'bridge', 'spider-bridge-bert-large.sh')) as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            name, value = line.strip().split('=')
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if value.lower() in ['true', 'false']:
                if value.lower() == 'true':
                    args.append('--{}'.format(name))
            else:
                args.extend(['--{}'.format(name), value])
args = parser.parse_args(args)