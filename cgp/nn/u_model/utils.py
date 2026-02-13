import torch
from torch.autograd import grad


@torch.jit.unused
def compute_divergence_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    trace = torch.zeros(1).to(positions.device)
    for i, grad_elem in enumerate(forces.reshape(-1)):
        hess_row = torch.autograd.grad(
            outputs=grad_elem,
            inputs=positions,
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        trace += hess_row[i // 3, i % 3]
    return trace


@torch.jit.unused
def compute_divergence_vmap(
    score: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    try:
        score = score.reshape(-1, 1)
        mask = torch.eye(
            score.shape[0], device=score.device, dtype=score.dtype
        ).unsqueeze(-1)
        trace = torch.autograd.grad(
            outputs=score,
            inputs=x,
            grad_outputs=mask,
            create_graph=True,
            retain_graph=True,  # Retain graph for multiple samples
            is_grads_batched=True,
        )[0]
        trace.reshape(score.shape[0], -1)
        trace = trace.diagonal().sum()
    except RuntimeError as e:
        print(e)
        trace = compute_divergence_loop(score, x)
    return trace


@torch.jit.unused
def hutchinson_trace(score, x, num_samples=10, vectorize=False):
    if vectorize:
        v = (
            torch.randint(0, 2, (num_samples, *x.shape), device=x.device).float() * 2
            - 1
        )
        grads = torch.autograd.grad(
            outputs=score,
            inputs=x,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,  # Retain graph for multiple samples
            is_grads_batched=True,
        )[0]
        trace = (grads * v).reshape(num_samples, -1).sum(dim=1).mean()
        return trace
    else:
        trace = torch.zeros(1, device=x.device, dtype=x.dtype)
        for _ in range(num_samples):
            v = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1
            grads = torch.autograd.grad(
                outputs=score,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,  # Retain graph for multiple samples
            )[0]
            trace += (grads * v).sum()
        trace /= num_samples
        return trace


def stein_trace(f_x, f, x, eps=0.001, num_samples=10):
    trace = torch.zeros(1, device=x.device, dtype=x.dtype)
    for _ in range(num_samples):
        e = torch.randn_like(x, device=x.device)
        f_e = f(x + (eps * e)) - f_x
        trace += (e * f_e).sum()
    return (trace / num_samples) / eps


def get_eval_fn(self):
    if self.nn.return_forces:

        def get_force(pos, features, batch_size):
            # pos.requires_grad = True
            features["nn_features"]["batch_size"] = batch_size
            features["prior_features"]["batch_size"] = batch_size
            if (
                "return_load_balance" in features["nn_features"].keys()
                and features["nn_features"]["return_load_balance"]
            ):
                force, loss = self.nn(pos, **features)
                return force, loss
            else:
                force = self.nn(pos, **features)
            return force
    else:

        def get_force(pos, features, batch_size):
            # pos.requires_grad = True
            features["nn_features"]["batch_size"] = batch_size
            features["prior_features"]["batch_size"] = batch_size
            if (
                "return_load_balance" in features["nn_features"].keys()
                and features["nn_features"]["return_load_balance"]
            ):
                energy, loss = self.nn(pos, **features)
                force = -grad(
                    energy,
                    pos,
                    create_graph=True,
                    retain_graph=True,
                    grad_outputs=torch.ones_like(energy),
                )[0]
                return force, loss
            energy = self.nn(pos, **features)
            force = -grad(
                energy,
                pos,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones_like(energy),
            )[0]
            return force

    if self.hparams.loss_type == "force matching":
        if self.hparams.load_balance_lambda > 0:
            if self.hparams.l1_lambda > 0:

                def loss(batch, batch_idx, prefix):
                    (pos, features), target_forces, target_energies, batch_size = batch
                    pos.requires_grad = True
                    features["nn_features"]["return_load_balance"] = True
                    pred_forces, load_balance_loss = get_force(
                        pos, features, batch_size
                    )
                    reg_l1_loss = self.nn.l1_loss()
                    mse_force_loss = ((pred_forces - target_forces) ** 2).sum(-1).mean()
                    total_loss = (
                        mse_force_loss
                        + self.hparams.l1_lambda * reg_l1_loss
                        + self.hparams.load_balance_lambda * load_balance_loss
                    )
                    metrics = {
                        f"{prefix}/MSEForce": mse_force_loss,
                        f"{prefix}/RegL1": reg_l1_loss,
                        f"{prefix}/LoadBalance": load_balance_loss,
                        f"{prefix}/TotalLoss": total_loss,
                        "hp_metric": total_loss,
                    }
                    self.log_dict(
                        metrics,
                        on_epoch=True,
                        on_step=self.hparams.on_step,
                        sync_dist=self.hparams.sync_dist,
                        batch_size=batch_size,
                    )
                    return total_loss
            else:

                def loss(batch, batch_idx, prefix):
                    (pos, features), target_forces, target_energies, batch_size = batch
                    pos.requires_grad = True
                    features["nn_features"]["return_load_balance"] = True
                    pred_forces, load_balance_loss = get_force(
                        pos, features, batch_size
                    )
                    mse_force_loss = ((pred_forces - target_forces) ** 2).sum(-1).mean()
                    total_loss = (
                        mse_force_loss
                        + self.hparams.load_balance_lambda * load_balance_loss
                    )
                    metrics = {
                        f"{prefix}/MSEForce": mse_force_loss,
                        f"{prefix}/LoadBalance": load_balance_loss,
                        f"{prefix}/TotalLoss": total_loss,
                        "hp_metric": total_loss,
                    }
                    self.log_dict(
                        metrics,
                        on_epoch=True,
                        on_step=self.hparams.on_step,
                        sync_dist=self.hparams.sync_dist,
                        batch_size=batch_size,
                    )
                    return total_loss
        else:
            if self.hparams.l1_lambda > 0:

                def loss(batch, batch_idx, prefix):
                    (pos, features), target_forces, target_energies, batch_size = batch
                    pos.requires_grad = True
                    pred_forces = get_force(pos, features, batch_size)
                    reg_l1_loss = self.nn.l1_loss()
                    mse_force_loss = ((pred_forces - target_forces) ** 2).sum(-1).mean()
                    total_loss = mse_force_loss + self.hparams.l1_lambda * reg_l1_loss
                    metrics = {
                        f"{prefix}/MSEForce": mse_force_loss,
                        f"{prefix}/RegL1": reg_l1_loss,
                        f"{prefix}/TotalLoss": total_loss,
                        "hp_metric": total_loss,
                    }
                    self.log_dict(
                        metrics,
                        on_epoch=True,
                        on_step=self.hparams.on_step,
                        sync_dist=self.hparams.sync_dist,
                        batch_size=batch_size,
                    )
                    return total_loss
            else:

                def loss(batch, batch_idx, prefix):
                    (pos, features), target_forces, target_energies, batch_size = batch
                    pos.requires_grad = True
                    pred_forces = get_force(pos, features, batch_size)
                    mse_force_loss = ((pred_forces - target_forces) ** 2).sum(-1).mean()
                    total_loss = mse_force_loss
                    metrics = {
                        f"{prefix}/MSEForce": mse_force_loss,
                        f"{prefix}/TotalLoss": total_loss,
                        "hp_metric": total_loss,
                    }
                    self.log_dict(
                        metrics,
                        on_epoch=True,
                        on_step=self.hparams.on_step,
                        sync_dist=self.hparams.sync_dist,
                        batch_size=batch_size,
                    )
                    return total_loss
    elif self.hparams.loss_type == "score matching":
        # beta = 1.677398444995886 # in units of mol/kcal
        beta = 503.21953349876577 / self.model_temperature  # in units of mol/kcal
        assert self.hparams.load_balance_lambda == 0, (
            "Score matching loss does not support load balance loss"
        )

        if self.hparams.div_method == "hutchinson":

            def get_trace(score, x, score_fn=None):
                return hutchinson_trace(
                    score,
                    x,
                    self.hparams.div_samples,
                    vectorize=self.hparams.vectorize_div,
                )
        elif self.hparams.div_method == "stein":

            def get_trace(score, x, score_fn=None):
                return stein_trace(
                    score,
                    score_fn,
                    x,
                    eps=self.hparams.div_epsilon,
                    num_samples=self.hparams.div_samples,
                )
        elif self.hparams.div_method == "exact":
            if self.hparams.vectorize_div:

                def get_trace(score, x, score_fn=None):
                    return compute_divergence_vmap(score, x)
            else:

                def get_trace(score, x, score_fn=None):
                    return compute_divergence_loop(score, x)
        else:
            raise ValueError(f"Unknown divergence method: {self.hparams.div_method}")

        if self.hparams.l1_lambda > 0.0:

            def loss(batch, batch_idx, prefix):
                (pos, features), _, _, batch_size = batch
                pos.requires_grad = True
                def get_score(x):
                    return get_force(x, features, batch_size) * beta

                score = get_score(pos)
                divergence = get_trace(score, pos, get_score) / pos.shape[0]
                score_loss = (score**2).sum(-1).mean() + (2 * divergence)
                reg_l1_loss = self.nn.l1_loss()
                total_loss = score_loss + self.hparams.l1_lambda * reg_l1_loss
                metrics = {
                    f"{prefix}/ScoreLoss": score_loss,
                    f"{prefix}/RegL1": reg_l1_loss,
                    f"{prefix}/TotalLoss": total_loss,
                    "hp_metric": total_loss,
                }
                self.log_dict(
                    metrics,
                    on_epoch=True,
                    on_step=self.hparams.on_step,
                    sync_dist=self.hparams.sync_dist,
                    batch_size=batch_size,
                )
                return total_loss
        else:

            def loss(batch, batch_idx, prefix):
                (pos, features), _, _, batch_size = batch
                pos.requires_grad = True
                def get_score(x):
                    return get_force(x, features, batch_size) * beta

                score = get_score(pos)
                divergence = get_trace(score, pos, get_score) / pos.shape[0]
                score_loss = (score**2).sum(-1).mean() + (2 * divergence)
                metrics = {
                    f"{prefix}/ScoreLoss": score_loss,
                    f"{prefix}/TotalLoss": score_loss,
                    "hp_metric": score_loss,
                }
                self.log_dict(
                    metrics,
                    on_epoch=True,
                    on_step=self.hparams.on_step,
                    sync_dist=self.hparams.sync_dist,
                    batch_size=batch_size,
                )
                return score_loss
    else:
        raise ValueError(f"Unknown loss type: {self.hparams.loss_type}")
    return loss, get_force
