import os
import numpy as np
import hydra
import copy
from cgp.cg import create_cg_model, concatenate_cg_datasets
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="u_dataset")
def main(cfg):
    cg_config = cfg.cg
    multiple_protein_names = cfg.global_args.multiple_protein_names
    is_individual_dataset_made = cfg.global_args.is_individual_dataset_made
    root_save_folder_name = cfg.global_args.root_save_folder_name
    root_data_folder_name = cfg.global_args.root_data_folder_name
    concatenate_datasets = cfg.global_args.concatenate_datasets

    if multiple_protein_names is not None:
        assert cg_config["cg_model_args"]["protein_name"] is None
        multiple_protein_names = np.load(multiple_protein_names)

        all_sp_filenames = []
        for protein_name in multiple_protein_names:
            save_folder_name = f"{root_save_folder_name}/{protein_name}/"
            if is_individual_dataset_made:
                all_sp_filenames.append(
                    f"{root_data_folder_name}/{protein_name}/dataset.hdf5"
                )
                continue

            os.makedirs(save_folder_name, exist_ok=True)

            cg_config_sp = copy.deepcopy(cg_config)
            OmegaConf.update(
                cg_config_sp, "cg_model_args.protein_name", str(protein_name)
            )

            # Creating cg_model automatically creates dataset
            cg_model = create_cg_model(
                save_folder_name, root_data_folder_name, cg_config_sp
            )
            all_sp_filenames.append(cg_model.cg_dataset_filename)
            OmegaConf.save(cfg, f"{save_folder_name}/dataset_config.yaml")

        if concatenate_datasets:
            concatenate_cg_datasets(
                all_sp_filenames,
                root_save_folder_name,
                cg_config["cg_model_args"]["store_in_dataset_args"],
            )
        OmegaConf.save(cfg, f"{root_save_folder_name}/dataset_config.yaml")

    else:
        protein_name = cg_config["cg_model_args"]["protein_name"]
        save_folder_name = f"{root_save_folder_name}/{protein_name}/"
        os.makedirs(save_folder_name, exist_ok=True)
        create_cg_model(save_folder_name, root_data_folder_name, cg_config)
        OmegaConf.save(cfg, f"{save_folder_name}/dataset_config.yaml")
