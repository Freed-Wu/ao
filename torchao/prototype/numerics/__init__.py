import math
import types
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from torchao.core.config import AOBaseConfig
from torchao.quantization import Int4Tensor, Int4WeightOnlyConfig
from torchao.quantization.quant_api import _module_extra_repr
from torchao.quantization.quantize_.workflows.int4.int4_tensor import (
    int4_row_quantize_zp,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import TorchAOBaseTensor


@dataclass
class ObserverConfig(AOBaseConfig):
    step: str = "observe"


@register_quantize_module_handler(ObserverConfig)
def _observer_config_transform(
    module: torch.nn.Module, config: ObserverConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    tensor = getattr(module, parameter_name)
    new_tensor = GPTQObserverTensor.from_hp(tensor)
    setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module

class ObserverTensor(TorchAOBaseTensor):
    """
    We create ObserverTensor with two modes, OBSERVE and REPLAY.

    if in OBSERVE mode, when it comes across a mm it will add the input to saved activations, and return a meta tensor.

    if in REPLAY mode, when it comes across a meta input to mm, it will pop an input from the saved activations, and return the quantized mm output.

    Then to sequentially quantize we can do the following:

    quantize_(layer N, ObserverTensor.OBSERVE)

    for batch in calibration_dataset:
        model(batch)

    # repeat below for all layers
    layer1.calculate_qparams_gptq()
    quantize(layer N, ObserverTensor.REPLAY)
    quantize(layer N+1, ObserverTensor.OBSERVE)

    for batch in calibration_dataset:
        model(batch.to(meta))

    quantize(layer N, Int4Tensor)
    move layer N to meta
    """

    tensor_data_names = ["hp_data"]
    tensor_attribute_names = ["observed_data"]
    optional_tensor_attribute_names = []

    def __new__(cls, hp_data: torch.Tensor, observed_data: List[torch.Tensor] = []):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, observed_data: List[torch.Tensor] = []):
        super().__init__()
        self.hp_data = hp_data
        self.observed_data = observed_data

    @classmethod
    def from_hp(cls, hp_tensor):
        return ObserverTensor(hp_tensor, [])

    def update(self, input_tensor):
        self.observed_data.append(input_tensor.detach())

implements = ObserverTensor.implements
implements_torch_function = ObserverTensor.implements_torch_function
aten = torch.ops.aten


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor.update(input_tensor.detach())
    return F.linear(input_tensor, weight_tensor.hp_data, bias)


@implements(aten.bmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor.update(input_tensor.detach())
    return func(input_tensor, weight_tensor.hp_data)


@dataclass
class GPTQConfig(AOBaseConfig):
    acceleration_config = Int4WeightOnlyConfig()
    percdamp: int = 0.1
    gptq_quantize_block_size = 256


@register_quantize_module_handler(GPTQConfig)
def _gptq_config_transform(
    module: torch.nn.Module, config: GPTQConfig, *, parameter_name="weight"
) -> torch.nn.Module:
    tensor = getattr(module, parameter_name)
    assert isinstance(tensor, GPTQObserverTensor)
    new_tensor = gptq_quantize(tensor.hessian, tensor.hp_data, config)
    setattr(module, parameter_name, nn.Parameter(new_tensor, requires_grad=False))
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module

def gptq_quantize(H, W, config):
    block_size = [1, config.acceleration_config.group_size]
    gptq_quantize_block_size = config.gptq_quantize_block_size
    percdamp = config.percdamp
    group_size = config.acceleration_config.group_size

    assert W.dim() == 2
    assert group_size > 0

    W = W.view(-1, W.shape[-1]).detach()
    columns = W.shape[1]
    device = W.device

    gptq_quantize_block_size = math.ceil(gptq_quantize_block_size / group_size) * group_size

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    all_qparams = []


    for (W_quantize_block, block_start) in zip(
        torch.split(W, gptq_quantize_block_size, dim=1),
        range(0, columns, gptq_quantize_block_size),
    ):
        block_end = min(block_start + gptq_quantize_block_size, columns)

        Err1 = torch.zeros_like(W_quantize_block, dtype=H.dtype)
        Hinv_quantize_block = Hinv[block_start:block_end, block_start:block_end]


        for (W_group, group_start) in zip(
            torch.split(W_quantize_block, group_size, dim=1),
            range(block_start, block_end, group_size),
        ):
            group_end = min(group_start + group_size, columns)

            if group_start % group_size == 0:
                # calculate qparams once per group
                _, scale, zero = int4_row_quantize_zp(
                    W_group, group_size
                )
                all_qparams.append((scale, zero))
            
            # within each group
            for i in range(group_start-block_start, group_end-block_start):
                w = W_quantize_block[:, i].unsqueeze(1)

                q = Int4Tensor.int4_row_quantize_zp_precomputed_qparams(
                    w, scale, zero, group_size
                )
                dq = (
                    Int4Tensor(
                        qdata=q,
                        scale=scale,
                        zero_point=zero,
                        block_size=block_size,
                        shape=q.shape,
                    ).dequantize()
                )

                err1 = (w - dq) / Hinv_quantize_block[i, i]
                W_quantize_block[:, i:] -= (
                    err1.matmul(Hinv_quantize_block[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1.flatten()

        W[:, block_end:] -= Err1.matmul(
            Hinv[block_start:block_end, block_end:]
        )

    if "cuda" in device.type:
        torch.cuda.synchronize()

    final_qparams = [torch.cat(x, dim=0) for x in zip(*all_qparams)]
    return Int4Tensor.from_hp_scale_and_zero_point(
        W,
        block_size,
        final_qparams[0].to(W.dtype),
        final_qparams[1].to(W.dtype)
    )
    
class GPTQObserverTensor(ObserverTensor):
    tensor_data_names = ["hp_data", "hessian"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = ["total_batches"]

    def __new__(cls, hp_data: torch.Tensor, hessian: torch.Tensor, total_batches: int = 0):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, hessian: torch.Tensor, total_batches: int = 0):
        super(ObserverTensor).__init__()
        self.total_batches = total_batches
        self.hessian = hessian
        self.hp_data = hp_data

    @classmethod
    def from_hp(cls, hp_tensor):
        return cls(
            hp_tensor,
            torch.tensor([],
                         device=hp_tensor.device,
                         dtype=torch.float),
            0,
        )

    def update(self, input_tensor):
        H = 0 if len(self.hessian) == 0 else self.hessian

        x = input_tensor.float()
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= self.total_batches / (self.total_batches + n)
        self.total_batches += n

        x = ((2 / self.total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())
        self.hessian = H
