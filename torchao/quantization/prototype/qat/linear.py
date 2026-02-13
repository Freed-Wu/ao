# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.quantization.qat.linear import (
    FakeQuantizedLinear,
    Int4WeightOnlyQATLinear,
    Int4WeightOnlyQATQuantizer,
    Int8DynActInt4WeightQATLinear,
    Int8DynActInt4WeightQATQuantizer,
    disable_4w_fake_quant,
    disable_8da4w_fake_quant,
    enable_4w_fake_quant,
    enable_8da4w_fake_quant,
)

__all__ = [
    "disable_4w_fake_quant",
    "disable_8da4w_fake_quant",
    "enable_4w_fake_quant",
    "enable_8da4w_fake_quant",
    "FakeQuantizedLinear",
    "Int4WeightOnlyQATLinear",
    "Int4WeightOnlyQATQuantizer",
    "Int8DynActInt4WeightQATLinear",
    "Int8DynActInt4WeightQATQuantizer",
]
