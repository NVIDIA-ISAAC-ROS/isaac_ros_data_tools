#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse as ap
import math
from pathlib import Path

import numpy as np


def compare(
    tensor1_path: Path,
    tensor2_path: Path,
    output_directory: Path,
    rel_tol: float,
    abs_tol: float
):
    tensor_list1 = np.load(tensor1_path, allow_pickle=True)
    tensor_list2 = np.load(tensor2_path, allow_pickle=True)

    assert len(tensor_list1.keys()) == len(tensor_list2.keys()), \
        'Tensor lists are not the same length'

    logfile_path = output_directory / \
        f'comparator_logfile_{tensor1_path.stem}_{tensor2_path.stem}.txt'
    print(f'Logfile path: {logfile_path}')
    with open(logfile_path, 'w+') as logfile:

        for tensor1_name, tensor2_name in zip(tensor_list1, tensor_list2):
            print(f'T1 name: \t{tensor1_name}')
            print(f'T2 name: \t{tensor2_name}')

            tensor1, tensor2 = tensor_list1[tensor1_name], tensor_list2[tensor2_name]

            print(f'T1 shape: \t{tensor1.shape}')
            print(f'T2 shape: \t{tensor2.shape}')

            count_good, count_bad = 0, 0

            logfile.write(
                f'Comparing: {tensor1_path.name} vs. {tensor2_path.name}\n')
            logfile.write(
                f'Relative Tolerance: {rel_tol} \tAbsolute Tolerance: {abs_tol}\n\n')

            for (index1, e1), (index2, e2) in zip(
                np.ndenumerate(tensor1), np.ndenumerate(tensor2)
            ):
                status_string = ''
                if math.isclose(e1, e2, rel_tol=rel_tol, abs_tol=abs_tol):
                    count_good += 1
                    status_string = 'GOOD'
                else:
                    count_bad += 1
                    status_string = 'BAD'

                logfile.writelines([
                    status_string + '\n',
                    f'T1 \t{index1}: {e1:.10f}\n',
                    f'T2 \t{index2}: {e2:.10f}\n',
                    '\n'
                ])

            logfile.write(f'Count GOOD: {count_good} vs. Count BAD: {count_bad}\n')

    print(f'Count GOOD: {count_good}')
    print(f'Count BAD: {count_bad}')


if __name__ == '__main__':
    parser = ap.ArgumentParser()

    parser.add_argument('tensor1_filename')
    parser.add_argument('tensor2_filename')
    parser.add_argument('output_directory')
    parser.add_argument('--abs-tol', '-a', default=0.0, type=float,
                        help='Absolute numerical error acceptable per element')
    parser.add_argument('--rel-tol', '-r', default=0.05, type=float,
                        help='Relative percentage error acceptable per element')

    args = parser.parse_args()

    compare(Path(args.tensor1_filename), Path(args.tensor2_filename), Path(args.output_directory),
            rel_tol=args.rel_tol, abs_tol=args.abs_tol)
