# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchao.quantization import Int4WeightOnlyConfig, quantize_

model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))

config = Int4WeightOnlyConfig(
    group_size=32,
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
)

quantize_(model, config)
