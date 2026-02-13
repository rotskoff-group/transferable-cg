import os
import hydra
from cgp.nn import create_dataset_from_path, create_lightning_model, create_dataloaders
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import warnings


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="nn_potential")
def main(cfg):
    nn_config = cfg.nn
    prior_config = cfg.prior
    train_config = cfg.train
    dataset_config = cfg.dataset

    if not cfg.global_args.use_nn:
        nn_config = None
    if not cfg.global_args.use_prior:
        prior_config = None

    L.seed_everything(**train_config["seed_args"])
    u_model = create_lightning_model(
        nn_config=nn_config, prior_config=prior_config, train_config=train_config
    )

    warnings.warn("Not checking dataset validity")
    if train_config["lightning_model_args"]["loss_type"] == "force matching":
        requires_forces = True
        requires_energies = False
    elif train_config["lightning_model_args"]["loss_type"] == "score matching":
        requires_forces = False
        requires_energies = False
    else:
        raise ValueError(
            f"Unknown loss type: {train_config['lightning_model_args']['loss_type']}"
        )
    dataset = create_dataset_from_path(
        dataset_config["model"],
        dataset_config["dataset_folder_name"],
        dataset_config["data_in_memory"],
        dataset_config["edge_args"],
        individual_protein_dataset=dataset_config["individual_protein_datasets"],
        requires_forces=requires_forces,
        requires_energies=requires_energies,
    )

    if dataset_config["adaption_dataset_folder_name"] is not None:
        adaption_dataset = create_dataset_from_path(
            dataset_config["model"],
            dataset_config["adaption_dataset_folder_name"],
            dataset_config["data_in_memory"],
            dataset_config["edge_args"],
            requires_forces=requires_forces,
            requires_energies=requires_energies,
        )
    else:
        adaption_dataset = None
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset,
        nn_config,
        prior_config,
        dataset_config,
        adaption_dataset=adaption_dataset,
        train_dataset_fraction=dataset_config["train_dataset_fraction"],
        requires_forces=requires_forces,
        requires_energies=requires_energies,
    )

    resume_training_path = train_config["resume_training_path"]
    if resume_training_path is not None:
        ckpt_path = resume_training_path
        if ckpt_path.endswith("/"):
            version_num = ckpt_path.split("/")[-4]
            model_path = "/".join(ckpt_path.split("/")[:-3])
        else:
            version_num = ckpt_path.split("/")[-3]
            model_path = "/".join(ckpt_path.split("/")[:-2])

        print(f"Restarting training from {model_path}")

        every_epoch_checkpoint_callback = ModelCheckpoint(
            filename="model_{epoch:02d}",
            every_n_train_steps=500,
            save_top_k=-1,
            dirpath=f"{model_path}/checkpoints",
        )
        best_checkpoint_callback = ModelCheckpoint(
            filename="best_model",
            monitor=train_config["lightning_model_args"]["monitor"],
            mode="min",
            save_top_k=1,
            dirpath=f"{model_path}/checkpoints",
        )
        logger = TensorBoardLogger(save_dir="./", version=version_num)

        trainer = L.Trainer(
            callbacks=[every_epoch_checkpoint_callback, best_checkpoint_callback],
            logger=logger,
            **train_config["trainer_args"],
        )

        trainer.fit(u_model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    else:
        every_epoch_checkpoint_callback = ModelCheckpoint(
            filename="model_{epoch:02d}", every_n_train_steps=500, save_top_k=-1
        )
        best_checkpoint_callback = ModelCheckpoint(
            filename="best_model",
            monitor=train_config["lightning_model_args"]["monitor"],
            mode="min",
            save_top_k=1,
        )
        logger = TensorBoardLogger(save_dir="./")
        trainer = L.Trainer(
            callbacks=[every_epoch_checkpoint_callback, best_checkpoint_callback],
            logger=logger,
            **train_config["trainer_args"],
        )

        if trainer.global_rank == 0:
            train_folder_name = trainer.logger.log_dir
            os.makedirs(train_folder_name, exist_ok=True)
            OmegaConf.save(cfg, f"{train_folder_name}/config.yaml")
        if cfg.global_args.load_model is not None:
            # Transfer learning
            load_filename = cfg.global_args.load_model
            print(f"Loading model from {load_filename}")
            u_model.load_model_from_ckpt(load_filename)

        trainer.fit(u_model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
