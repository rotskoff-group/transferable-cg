from typing import Any, Dict
import torch
from e3nn import o3

def extract_config_mace_model(model: torch.nn.Module) -> Dict[str, Any]:
    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    model_mlp_irreps = (
        o3.Irreps(str(model.readouts[-1].hidden_irreps))
        if model.num_interactions.item() > 1
        else 1
    )
    mlp_irreps = o3.Irreps(f"{model_mlp_irreps.count((0, 1))}x0e")
    try:
        correlation = (
            len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = model.products[0].symmetric_contractions.contraction_degree
    config = {
        "nn_cutoff": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": (mlp_irreps if model.num_interactions.item() > 1 else 1),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "correlation": correlation,
        "gate": (
            model.readouts[-1]  # pylint: disable=protected-access
            .non_linearity._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "use_cueq": model.cueq_config != None
    }
    return config