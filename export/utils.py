"""
Export utilities for MobileCLIP and other models.

This module contains helper functions that were previously sourced from
external reference repositories, bundled here for seamless setup.
"""

import copy

import torch
from torch import nn


def reparameterize_model(model: nn.Module) -> nn.Module:
    """
    Reparameterize a model by merging training-time branches.

    MobileCLIP (and MobileOne architecture) uses multi-branch training-time
    blocks for better gradient flow during training. Before inference/export,
    these branches must be merged into single convolutions for efficiency.

    This function recursively finds all modules with a `reparameterize()` method
    and calls it to merge the branches.

    Original source: Apple's ml-mobileclip repository
    https://github.com/apple/ml-mobileclip/blob/main/mobileclip/modules/common/mobileone.py

    Args:
        model: PyTorch model with reparameterizable modules

    Returns:
        Reparameterized model (deep copy with merged branches)
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
