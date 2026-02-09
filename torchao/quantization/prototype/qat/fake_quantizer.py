# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.quantization.qat.fake_quantizer import (
    IntxFakeQuantizer as FakeQuantizer,
)

__all__ = [
    "FakeQuantizer",
]
