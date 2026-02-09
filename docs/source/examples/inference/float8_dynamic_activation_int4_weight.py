# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchao.quantization import Float8DynamicActivationInt4WeightConfig, quantize_

model = nn.Sequential(nn.Linear(2048, 2048, device="cuda"))
quantize_(model, Float8DynamicActivationInt4WeightConfig())
