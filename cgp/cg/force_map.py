import torch

def get_local_atoms(positions, cg_indices, cutoff, pad_value=-1):
    """Find non-CG atoms within cutoff of each CG bead (by mean distance across frames).

    Parameters
    ----------
    positions  : (n_frames, n_atoms, 3)
    cg_indices : (n_beads,)
    cutoff     : float, in same units as positions

    Returns
    -------
    local_indices : (n_beads, m_max) original atom indices, padded with pad_value
    """
    n_atoms = positions.shape[1]

    cg_mask = torch.zeros(n_atoms, dtype=torch.bool, device=positions.device)
    cg_mask[cg_indices] = True
    free_indices = torch.where(~cg_mask)[0]
    free_positions = positions[:, free_indices, :]
    cg_positions = positions[:, cg_indices, :]

    # (n_beads, n_frames, n_free, 3)
    diff = (free_positions[None, :, :, :]
            - cg_positions.permute(1, 0, 2)[:, :, None, :])

    # (n_beads, n_free)
    mean_distances = torch.norm(diff, dim=-1).mean(dim=1)
    within_cutoff = mean_distances < cutoff

    counts = within_cutoff.sum(dim=1)
    m_max = int(counts.max().item())

    order = torch.argsort(~within_cutoff, dim=1, stable=True)[:, :m_max]
    local_indices_full = free_indices[order]

    valid = torch.arange(m_max, device=positions.device)[None, :] < counts[:, None]
    local_indices = torch.where(valid, local_indices_full,
                                torch.full_like(local_indices_full, pad_value))
    return local_indices


def _fit_weights_single_bead(f_rep, f_local, valid):
    """Fit optimal force weights for a single CG bead.

    Parameters
    ----------
    f_rep   : (n_frames, 3)
    f_local : (m_max, n_frames, 3)
    valid   : (m_max,) bool

    Returns
    -------
    w : (m_max,)
    """
    f_local = f_local * valid[:, None, None].float()
    f_local_flat = f_local.reshape(f_local.shape[0], -1)   # (m_max, 3*n_frames)
    f_rep_flat = f_rep.reshape(-1)                           # (3*n_frames,)
    n = f_rep_flat.shape[0]

    A = (f_local_flat @ f_local_flat.T) / n
    b = (f_local_flat @ f_rep_flat) / n
    A = A + 1e-6 * torch.eye(A.shape[0], device=A.device)

    w = torch.linalg.solve(A, -b)
    return w * valid.float()


def _apply_weights_single_bead(f_rep, f_local, valid, w):
    """Apply precomputed weights to get optimal CG force for a single bead.

    Parameters
    ----------
    f_rep   : (n_frames, 3)
    f_local : (m_max, n_frames, 3)
    valid   : (m_max,) bool
    w       : (m_max,)

    Returns
    -------
    cg_force : (n_frames, 3)
    """
    f_local = f_local * valid[:, None, None].float()
    return f_rep + torch.einsum('m,mfd->fd', w, f_local)


def get_all_optimal_cg_forces(positions, forces, cg_indices, cutoff,
                               fit_positions=None, fit_forces=None, pad_value=-1):
    """Compute optimal CG forces for all beads.

    Weights are fit on fit_positions/fit_forces (defaults to positions/forces if not
    provided) and then applied to all frames of positions/forces. This allows using a
    random subset for the least-squares solve while applying the result to all frames.

    Parameters
    ----------
    positions      : (n_frames, n_atoms, 3)
    forces         : (n_frames, n_atoms, 3)
    cg_indices     : (n_beads,)
    cutoff         : float, in same units as positions
    fit_positions  : (n_fit, n_atoms, 3) or None  — used to find local atoms and fit weights
    fit_forces     : (n_fit, n_atoms, 3) or None  — used to fit weights

    Returns
    -------
    cg_forces : (n_beads, n_frames, 3)
    """
    if fit_positions is None:
        fit_positions = positions
    if fit_forces is None:
        fit_forces = forces

    local_indices = get_local_atoms(fit_positions, cg_indices, cutoff, pad_value)
    valid = local_indices != pad_value
    safe_indices = torch.where(valid, local_indices, torch.zeros_like(local_indices))

    # Fit weights on the (possibly subsampled) frames
    f_rep_fit = fit_forces[:, cg_indices, :].permute(1, 0, 2)       # (n_beads, n_fit, 3)
    f_local_fit = fit_forces[:, safe_indices, :].permute(1, 2, 0, 3)  # (n_beads, m_max, n_fit, 3)
    weights = torch.vmap(_fit_weights_single_bead)(f_rep_fit, f_local_fit, valid)  # (n_beads, m_max)

    # Apply weights to all frames
    f_rep_all = forces[:, cg_indices, :].permute(1, 0, 2)           # (n_beads, n_frames, 3)
    f_local_all = forces[:, safe_indices, :].permute(1, 2, 0, 3)    # (n_beads, m_max, n_frames, 3)
    cg_forces = torch.vmap(_apply_weights_single_bead)(
        f_rep_all, f_local_all, valid, weights
    )  # (n_beads, n_frames, 3)

    return cg_forces
