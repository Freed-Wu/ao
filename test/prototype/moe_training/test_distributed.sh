# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

NGPU=${NGPU:-4}
torchrun --nproc_per_node=$NGPU --local-ranks-filter=0 -m pytest test/prototype/moe_training/test_distributed.py -s -v
