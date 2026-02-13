import os
import hydra
from cgp.nn import create_dataset_from_path, create_lightning_model, create_dataloaders
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import warnings
import torch
import numpy as np


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="test_u")
def main(cfg):
    global_args = cfg.global_args
    accelerator = global_args["accelerator"]  # "cuda" or "cpu"
    config_path = global_args["config_path"]
    ckpt_path = global_args["ckpt_path"]
    # model_folder = global_args["model_folder"]
    model_cfg = OmegaConf.load(config_path)

    nn_config = model_cfg["nn"]
    prior_config = model_cfg["prior"]
    train_config = model_cfg["train"]
    dataset_config = model_cfg["dataset"]

    if not model_cfg["global_args"]["use_nn"]:
        nn_config = None
    if not model_cfg["global_args"]["use_prior"]:
        prior_config = None

    report_metric_per_protein_save_folder = (
        global_args.report_metric_per_protein_save_folder
    )
    if global_args["use_training_dataset"]:
        assert dataset_config["dataset_split_args"]["test"] > 0
        global_args = model_cfg["global_args"]

    else:  # Use dataset all for testing
        dataset_config["dataset_split_args"]["train"] = 0.0
        dataset_config["dataset_split_args"]["val"] = 0.0
        dataset_config["dataset_split_args"]["test"] = 1.0
        dataset_config["batch_size"] = 1

    train_config["trainer_args"][
        "strategy"
    ] = "auto"  # do not use ddp since ddp replicates samples to ensure all devices have same batch size
    if accelerator == "cuda":
        train_config["trainer_args"]["devices"] = 1 # use 1 GPU
    else:
        train_config["trainer_args"]["devices"] = "auto"

    L.seed_everything(**train_config["seed_args"])  # ensure same split as training

    u_model = create_lightning_model(
        nn_config=nn_config,
        prior_config=prior_config,
        train_config=train_config,
    )

    if report_metric_per_protein_save_folder is None:
        dataset_path = global_args.dataset_folder_name
        dataset = create_dataset_from_path(
            dataset_config["model"],
            dataset_path,
            dataset_config["data_in_memory"],
            dataset_config["edge_args"],
            individual_protein_dataset=global_args.individual_protein_datasets,
            requires_esm2_embeddings=nn_config["model"] == "Transformer"
        )
        _, _, test_dataloader = create_dataloaders(
            dataset,
            nn_config=nn_config,
            prior_config=prior_config,
            dataset_config=dataset_config,
            train_dataset_fraction=dataset_config["train_dataset_fraction"],
        )

        u_model.load_model_from_ckpt(ckpt_path)
        u_model.eval()
        version_num = config_path.split("/")[-2]

        logger = TensorBoardLogger(save_dir="./", version=version_num)
        trainer = L.Trainer(
            logger=logger, **train_config["trainer_args"], inference_mode=False
        )
        with torch.enable_grad():
            trainer.test(model=u_model, dataloaders=test_dataloader)
    else:
        all_mse = []
        assert global_args.individual_protein_datasets != None
        u_model.load_model_from_ckpt(ckpt_path)
        u_model.eval()
        version_num = config_path.split("/")[-2]
        save_dir = f"{report_metric_per_protein_save_folder}/{version_num}"
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, f"{save_dir}/config.yaml")

        protein_names = np.load(global_args.individual_protein_datasets)
        for protein in protein_names:
            dataset_path = f"{global_args.dataset_folder_name}/{protein}"
            dataset = create_dataset_from_path(
                dataset_config["model"],
                dataset_path,
                dataset_config["data_in_memory"],
                dataset_config["edge_args"],
                requires_esm2_embeddings=nn_config["model"] == "Transformer"
            )
            _, _, test_dataloader = create_dataloaders(
                dataset,
                nn_config=nn_config,
                prior_config=prior_config,
                dataset_config=dataset_config,
                train_dataset_fraction=dataset_config["train_dataset_fraction"],
            )

            trainer = L.Trainer(
                logger=False, **train_config["trainer_args"], inference_mode=False
            )
            with torch.enable_grad():
                trainer.test(model=u_model, dataloaders=test_dataloader)
            all_mse.append(trainer.callback_metrics["test/MSEForce"])
        all_mse = np.array(all_mse)
        np.save(f"{save_dir}/mse_per_protein.npy", all_mse)


if __name__ == "__main__":
    main()
