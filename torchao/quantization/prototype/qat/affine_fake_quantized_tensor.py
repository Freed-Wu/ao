# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.quantization.qat.affine_fake_quantized_tensor import (
    _AffineFakeQuantizedTensor,
    _to_affine_fake_quantized,
)

__all__ = [
    "_AffineFakeQuantizedTensor",
    "_to_affine_fake_quantized",
]
