import lightning as L
import torch
from torch.autograd import grad
import numpy as np
from .cg_model import CGModel
from .utils import get_eval_fn
   
class UModel(L.LightningModule):
    def __init__(self, nn_model_name, nn_model_args,
                 prior_model_name, prior_model_args,
                 return_forces,
                 model_temperature,
                 loss_type,
                 div_method,
                 div_samples,
                 div_epsilon,
                 vectorize_div,
                 l1_lambda,
                 load_balance_lambda,
                 layers_to_freeze,
                 optimizer,
                 optimizer_args,
                 lr_scheduler, lr_scheduler_args,
                 monitor, sync_dist, on_step):
        super().__init__()
        assert not (nn_model_name == None and prior_model_name == None)
        self.use_nn = nn_model_name != None
        self.use_prior = prior_model_name != None
        self.nn = CGModel(nn_model_name,
                          nn_model_args,
                          prior_model_name,
                          prior_model_args,
                          return_forces)
        
        if type(layers_to_freeze) == str:
            layers_to_freeze = np.load(layers_to_freeze)
        for name, param in self.nn.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
        self.return_forces = return_forces
        self.model_temperature = model_temperature
        self.save_hyperparameters()
        self._loss, self._get_force = get_eval_fn(self)


    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)


    def _shared_eval(self, batch, batch_idx, prefix):
        return self._loss(batch, batch_idx, prefix)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        (pos, features), target_forces, target_energies, batch_size = batch
        pos.requires_grad = True
        pred_forces = self._get_force(pos, features, batch_size)
        mse_force_loss = ((pred_forces - target_forces) ** 2).sum(-1).mean()
        metrics = {f"test/MSEForce": mse_force_loss}
        self.log_dict(metrics, on_epoch=True, 
                                  on_step=self.hparams.on_step, 
                                  sync_dist=self.hparams.sync_dist,
                                  batch_size=batch_size)


    def forward(self, inputs, target):
        return self.nn(inputs, target)

    def configure_optimizers(self):

        u_optimizer = getattr(torch.optim,
                              self.hparams.optimizer)
        optimizer_args = []
        if self.use_nn:
            optimizer_args.append({'params': self.nn.nn.parameters(),
                                   'lr': self.hparams.optimizer_args.nn_lr})
        if self.use_prior:
            optimizer_args.append({'params': self.nn.prior.parameters(),
                                   'lr': self.hparams.optimizer_args.prior_lr})
        u_optimizer = u_optimizer(optimizer_args)
        u_scheduler = getattr(torch.optim.lr_scheduler,
                              self.hparams.lr_scheduler)
        u_scheduler = u_scheduler(u_optimizer,
                                  **self.hparams.lr_scheduler_args)
        to_return = {"optimizer": u_optimizer, "lr_scheduler": u_scheduler,
                     "monitor": self.hparams.monitor}
        return to_return

    def load_model_from_ckpt(self, filename):
        print(f"Loading model from {filename}")
        all_weights = torch.load(filename, weights_only=False)["state_dict"]
        model_weights = {k[3:]: v for k, v in all_weights.items()
                         if k.startswith("nn.")}
        self.nn.load(model_weights, self.hparams.nn_model_name, self.hparams.nn_model_args)



