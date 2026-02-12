import copy
import numpy as np
import torch
import cgp.nn
from omegaconf import OmegaConf
import h5py

    
def get_collate_fn(nn_config, prior_config,
                   requires_forces=True, requires_energies=False):
    assert not (nn_config == None and prior_config == None)
    if prior_config != None:
        prior_collate_fn= getattr(cgp.nn, prior_config["collate_fn"])
    else:
        def prior_collate_fn(**kwargs):
            return {}
    if nn_config != None:
        nn_collate_fn = getattr(cgp.nn, nn_config["collate_fn"])
    else:
        def nn_collate_fn(**kwargs):
            return {}
    
    def collate_features_fn(data):
        return {"nn_features": nn_collate_fn(**data), "prior_features": prior_collate_fn(**data)}
    
    def collate_fn(data):
        (x_positions,
        y_f,
        y_e,
        angle_indices,
        angle_features,
        dihedral_indices,
        dihedral_features,
        bond_distance_indices,
        bond_distance_features,
        nonbonded_edge_indices,
        nonbonded_edge_features,
        edge_indices, # edges
        atom_features,
        esm2_embeddings, 
        update_edge_indices) = zip(*data)

        update_edge_indices = np.any(update_edge_indices)
        features = {
            "angle_indices": angle_indices,
            "angle_features": angle_features,
            "dihedral_indices": dihedral_indices,
            "dihedral_features": dihedral_features,
            "bond_distance_indices": bond_distance_indices,
            "bond_distance_features": bond_distance_features,
            "nonbonded_edge_indices": nonbonded_edge_indices,
            "nonbonded_edge_features": nonbonded_edge_features,
            "edge_indices": edge_indices,
            "atom_features": atom_features,
            "esm2_embeddings": esm2_embeddings,
            "update_edge_indices": update_edge_indices
        }
    
        batch_size = len(x_positions)
        x_positions = torch.cat(x_positions, dim=0)
        if requires_forces:
            y_f = torch.cat(y_f, dim=0)
        if requires_energies:
            y_e = torch.cat(y_e, dim=0)
        features = collate_features_fn(features)
        features["nn_features"]["batch_size"] = batch_size
        features["prior_features"]["batch_size"] = batch_size
        return ([x_positions, features], 
                y_f, y_e, batch_size)
    return collate_fn, collate_features_fn



def get_hdf5_fn(cg_dataset_filename, individual_protein_dataset):
    """Gets the dataset from a coarse-grained model
    Args:
        cg_dataset_filename (str): The name of the file where the dataset is saved
    Returns:
        cg_hdf5: A dictionary with the dataset
    """
    if individual_protein_dataset != None:
        def get_hdf5_data():
            protein_names = np.load(individual_protein_dataset)
            return [h5py.File(f"{cg_dataset_filename}/{p}/dataset.hdf5", "r") for p in protein_names]
    else:
        def get_hdf5_data():
            return h5py.File(cg_dataset_filename, "r")
    return get_hdf5_data

def create_dataset_from_path(model, dataset_folder_name, data_in_memory,
                             edge_args, individual_protein_dataset=None,
                             requires_forces=True, requires_energies=False,
                             requires_esm2_embeddings=False,
                             random_rotate=False):
    """Creates a dataset from a path to a hdf5 file
    Args:
        cg_dataset_filename (str): The name of the file where the dataset is saved
        nn_config (dict): A dictionary with the configuration for the neural network
    Returns:
        dataset: A (subclassed) PyTorch Dataset
    """
    dataset = getattr(cgp.nn, model)
    if individual_protein_dataset == None:
        dataset_path = f"{dataset_folder_name}/dataset.hdf5"
    else:
        dataset_path = f"{dataset_folder_name}"

    get_hdf5_data = get_hdf5_fn(dataset_path, 
                                individual_protein_dataset)
    dataset = dataset(get_hdf5_data=get_hdf5_data,
                      data_in_memory=data_in_memory,
                      requires_forces=requires_forces,
                      requires_energies=requires_energies,
                      requires_esm2_embeddings=requires_esm2_embeddings,    
                      random_rotate=random_rotate,
                      **edge_args)

    return dataset

def create_dataloaders(dataset, nn_config, prior_config, 
                       dataset_config,
                       adaption_dataset=None, 
                       train_dataset_fraction=None,
                       requires_forces=True, 
                       requires_energies=False):
    """Creates dataloaders from a dataset
    Args:
        dataset: A (subclassed) PyTorch Dataset
        nn_config (dict): A dictionary with the configuration for the neural network
    Returns:
        train_loader, val_loader, test_loader: PyTorch Dataloaders
    """
    train, val, test = torch.utils.data.random_split(dataset,
                                                     list(dataset_config["dataset_split_args"].values()))
    collate_fn, _ = get_collate_fn(nn_config, prior_config,
                                   requires_forces=requires_forces,
                                   requires_energies=requires_energies)
    
    if train_dataset_fraction != 1.0:
        subsample_indices_train = np.random.choice(np.arange(len(train)), 
                                                int(len(train) * train_dataset_fraction), replace=False)
        subsample_indices_val = np.random.choice(np.arange(len(val)),
                                                int(len(val) * train_dataset_fraction), replace=False)
        train = torch.utils.data.Subset(train, subsample_indices_train)
        val = torch.utils.data.Subset(val, subsample_indices_val)

    if adaption_dataset is not None:
        adaption_train, adaption_val, adaption_test = torch.utils.data.random_split(adaption_dataset,
                                                                                     list(dataset_config["dataset_split_args"].values()))
        train = torch.utils.data.ConcatDataset([train, adaption_train])
        val = torch.utils.data.ConcatDataset([val, adaption_val])
        test = torch.utils.data.ConcatDataset([test, adaption_test])

    if len(train) > 0:
        train_loader = torch.utils.data.DataLoader(train, batch_size=dataset_config["batch_size"], shuffle=True,
                                                num_workers=dataset_config["num_workers"], collate_fn=collate_fn,
                                                pin_memory=True)
    else:
        train_loader = None

    if len(val) > 0:
        val_loader = torch.utils.data.DataLoader(val, batch_size=dataset_config["batch_size"], shuffle=False,
                                             num_workers=dataset_config["num_workers"], collate_fn=collate_fn)
    else:
        val_loader = None
    if len(test) > 0:
        test_loader = torch.utils.data.DataLoader(test, batch_size=dataset_config["batch_size"], shuffle=False,
                                                num_workers=0,
                                                collate_fn=collate_fn)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader

def create_lightning_model(nn_config, prior_config, train_config):
    """Creates a neural network model
    Args:
        nn_config (dict): A dictionary with the configuration for the neural network
        train_config (dict): A dictionary with the configuration for the training
    Returns:
        lightning_model: A LightningModule model
    """
    assert not (nn_config == None and prior_config == None)
    if nn_config != None:
        nn_model = nn_config["model"]
        nn_model_args = nn_config["model_args"]
    else:
        nn_model = None
        nn_model_args = None
    if prior_config != None:
        prior_model = prior_config["model"]
        prior_model_args = prior_config["model_args"]
    else:
        prior_model = None
        prior_model_args = None
    lightning_model = getattr(cgp.nn, train_config["lightning_model"])
    # Make the nn in the potential so Lightning is aware of hyperparameters

    lightning_model = lightning_model(nn_model_name=nn_model,
                                      nn_model_args=nn_model_args,
                                      prior_model_name=prior_model,
                                      prior_model_args=prior_model_args,
                                      **train_config["lightning_model_args"])
    return lightning_model


def check_configs_equal(config1, config2, keys_to_test):
    """Compares two configurations
    Args:
        config1 (dict): A dictionary with the configuration
        config2 (dict): A dictionary with the configuration
    Returns:
        bool: True if the configurations are the same, False otherwise
    """
    config1 = copy.deepcopy(config1)
    config2 = copy.deepcopy(config2)
    configs_equal = True
    for key in keys_to_test:
        if OmegaConf.select(config1, key) != OmegaConf.select(config2, key):
            configs_equal = False
    return configs_equal



