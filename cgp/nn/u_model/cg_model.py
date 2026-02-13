import cgp.nn
import torch.nn as nn
import numpy as np

class CGModel(nn.Module):
    def __init__(self,
                 nn_model_name,
                 nn_model_args,
                 prior_model_name,
                 prior_model_args,
                 return_forces):
        super().__init__()
        # Both nn_model_name and prior_model_name cannot be None
        assert not (nn_model_name == None and prior_model_args == None)

        if nn_model_name == None: # No neural network, only prior
            def nn_model(pos, **kwargs):
                return 0.0
            self.nn = nn_model
        else:
            if "return_forces" in nn_model_args.keys():
                nn_model_args["return_forces"] = return_forces
            self.nn = getattr(cgp.nn, nn_model_name)
            self.nn = self.nn(**nn_model_args)


        if prior_model_name == None: # No prior model, only neural network
            def prior_model(pos, **kwargs):
                return 0.0
            self.prior = prior_model
        else:
            if "return_forces" in prior_model_args.keys():
                prior_model_args["return_forces"] = return_forces
            self.prior = getattr(cgp.nn, prior_model_name)
            self.prior = self.prior(**prior_model_args)
        self.return_forces = return_forces

    def forward(self, x, nn_features, prior_features):
        """
        x : (batch_size * num_nodes) x 3
        features : batch_size x num_nodes x 1
        """
        if "return_load_balance" in nn_features.keys() and nn_features["return_load_balance"]:
            u_nn, loss = self.nn(x, **nn_features)
            u_prior = self.prior(x, **prior_features)
            return u_nn + u_prior, loss
        return self.nn(x, **nn_features) + self.prior(x, **prior_features) ##TODO: Fix with load balance loss
    
    def load(self, model_weights, model_name, model_params):
        try:
            self.load_state_dict(model_weights, strict=False)
        except Exception as e:
            if model_name == "MACE":
                target_dict = self.state_dict()
                print("MACE model loading failed, trying to convert weights")
                if model_params["use_cueq"]:
                    model_weights = cgp.nn.mace.tools.convert_e3nn_cueq.convert_weights(
                        model_weights,
                        target_dict,
                        model_params
                    )
                else:
                    model_weights = cgp.nn.mace.tools.convert_cueq_e3nn.convert_weights(
                        model_weights,
                        target_dict,
                        model_params
                    )
                self.load_state_dict(model_weights, strict=True)
            else:
                raise e


    def l1_loss(self):
        l1_loss = self.prior.l1_loss()
        return l1_loss