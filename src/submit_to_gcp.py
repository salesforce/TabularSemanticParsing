"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import random
import string
import subprocess
import sys
import yaml


def run():
    pod_name = sys.argv[1]
    temp_dir = 'yaml'
    yaml_file_name = sys.argv[2]
    command = sys.argv[3]

    in_temp = os.path.join(temp_dir, yaml_file_name)
    with open(in_temp) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
        # print(json.dumps(content, indent=4))

    random_tag = get_random_tag(k=12)
    pod_name_tagged = pod_name + '-{}'.format(random_tag)
    content['metadata']['name'] = pod_name_tagged
    containers = content['spec']['containers']
    for i, container in enumerate(containers):
        container['name'] = pod_name + '-{}'.format(i)
        container['command'][2] += command

    out_yaml = os.path.join(temp_dir, '{}.yaml'.format(pod_name_tagged))
    with open(out_yaml, 'w') as o_f:
        yaml.dump(content, o_f)

    print('run: {}'.format(['kubectl', 'create', '-f', out_yaml]))
    subprocess.run(['kubectl', 'create', '-f', out_yaml])


def get_random_tag(k=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


if __name__ == '__main__':
    run()