import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch

from cgp.nn.mace.modules.cue_utils import CuEquivarianceConfig
from e3nn import o3


def get_transfer_keys() -> List[str]:
    """Get list of keys that need to be transferred"""
    return [
        "nn.node_embedding.linear.weight",
        "nn.radial_embedding.bessel_fn.bessel_weights",
        "nn.readouts.0.linear.weight",
        *[f"nn.readouts.1.linear_{i}.weight" for i in range(1, 3)],
    ] + [
        s
        for j in range(2)
        for s in [
            f"nn.interactions.{j}.linear_up.weight",
            *[f"nn.interactions.{j}.conv_tp_weights.layer{i}.weight" for i in range(4)],
            f"nn.interactions.{j}.linear.weight",
            f"nn.interactions.{j}.skip_tp.weight",
            f"nn.products.{j}.linear.weight",
        ]
    ]


def get_kmax_pairs(max_L: int, correlation: int) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on max_L and correlation"""
    if correlation == 2:
        raise NotImplementedError("Correlation 2 not supported yet")
    if correlation == 3:
        return [[0, max_L], [1, 0]]
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(max_L, correlation)

    for i, kmax in kmax_pairs:
        wm = torch.concatenate(
            [
                source_dict[
                    f"nn.products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                ]
                for k in range(kmax + 1)
                for j in ["_max", ".0", ".1"]
            ],
            dim=1,
        )
        target_dict[f"nn.products.{i}.symmetric_contractions.weight"] = wm
    return target_dict


def convert_weights(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    model_params: Dict[str, any],
    # max_L: int,
    # correlation: int,
):
    max_L = o3.Irreps(model_params["hidden_irreps"]).lmax
    correlation = model_params["correlation"]
    
    # Transfer main weights
    transfer_keys = get_transfer_keys()
    for key in transfer_keys:
        if key in source_dict:  # Check if key exists
            target_dict[key] = source_dict[key]
        else:
            logging.warning(f"Key {key} not found in source model")

    # Transfer symmetric contractions
    target_dict = transfer_symmetric_contractions(source_dict, target_dict, max_L, correlation)

    # Unsqueeze linear and skip_tp layers
    for key in source_dict.keys():
        if any(x in key for x in ["linear", "skip_tp"]) and "weight" in key and "nn." in key:
            target_dict[key] = target_dict[key].unsqueeze(0)

    # Transfer remaining matching keys
    transferred_keys = set(transfer_keys)
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}

    if remaining_keys:
        for key in remaining_keys:
            if source_dict[key].shape == target_dict[key].shape:
                logging.debug(f"Transferring additional key: {key}")
                target_dict[key] = source_dict[key]
            else:
                logging.warning(
                    f"Shape mismatch for key {key}: "
                    f"source {source_dict[key].shape} vs target {target_dict[key].shape}"
                )
    
    return target_dict

   


