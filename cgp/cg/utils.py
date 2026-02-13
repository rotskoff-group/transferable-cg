import cgp.cg
import h5py
import numpy as np
import dask.array as da
from .res_info import (all_bond_types,
                         all_angle_types, all_dihedral_angle_types, all_pairwise_types,
                         amber_res_code_to_standard, cg_bead_mass)
import torch

def create_cg_model(save_folder_name, root_data_folder_name, cg_config):
    """Creates a coarse-grained model from a configuration dictionary
    Args:
        save_folder_name (str): The name of the folder where to save the information
        root_data_folder_name (str): The name of the root data folder
        cg_config (dict): A dictionary with the configuration for the coarse-grained model
    Returns:
        cg_model: A coarse-grained model
    """
    cg_model = getattr(cgp.cg, cg_config['cg_model'])
    cg_model_config = cg_config['cg_model_args']
    cg_model = cg_model(save_folder_name, 
                        root_data_folder_name=root_data_folder_name,    
                        **cg_model_config)
    return cg_model


def concatenate_cg_datasets(all_sp_filenames, root_save_folder_name, store_in_dataset_args):
    """Concatenates a list of datasets
    Args:
        all_datasets: A list of file paths to the h5py datasets
        save_folder_name (str): The name of the folder where the dataset is saved
    Returns:
        dataset: A (subclassed) PyTorch Dataset
    """
    all_h5py_files = []
    for sp_filename in all_sp_filenames:
        print(sp_filename)
        h5py_file = h5py.File(sp_filename, "r")
        all_h5py_files.append(h5py_file)

    length_units = [h5_file.attrs["length_units"] for h5_file in all_h5py_files]
    assert np.unique(length_units).shape[0] == 1
    length_units = length_units[0]

    energy_units = [h5_file.attrs["energy_units"] for h5_file in all_h5py_files]
    assert np.unique(energy_units).shape[0] == 1
    energy_units = energy_units[0]
    assert store_in_dataset_args["store_positions"] == True

    num_data_points_per_dataset = [h5_file["positions"].shape[0]
                                   for h5_file in all_h5py_files]
    num_data_points = sum(num_data_points_per_dataset)
    num_data_points_per_dataset_cum = np.append(0,
                                                np.cumsum(num_data_points_per_dataset))

    num_cg_beads_per_dataset = np.array([h5_file["positions"].shape[1]
                                for h5_file in all_h5py_files])
    max_cg_beads = max(num_cg_beads_per_dataset)



    cg_dataset_filename = f"{root_save_folder_name}/dataset.hdf5"
    file = h5py.File(cg_dataset_filename, "w")

    if store_in_dataset_args["store_positions"]:
        file.create_dataset("positions", (num_data_points, max_cg_beads, 3),
                               dtype='f4')
    if store_in_dataset_args["store_forces"]:
        file.create_dataset("forces", (num_data_points, max_cg_beads, 3),
                            dtype='f4')
    if store_in_dataset_args["store_energies"]:
        file.create_dataset("energies", (num_data_points, 1),
                            dtype='f4')

    if store_in_dataset_args["store_features"]:
        num_features = [h5_file["features"].shape[2]
                        for h5_file in all_h5py_files]
        assert np.unique(num_features).shape[0] == 1
        num_features = num_features[0]
        file.create_dataset("features", (num_data_points, max_cg_beads, num_features),
                            dtype='i4', fillvalue=-1)
        
    if store_in_dataset_args["store_atom_features"]:
        # num_atom_features = [h5_file["atom_features"].shape[-1] 
        #                             for h5_file in all_h5py_files]
        file.create_dataset("atom_features", (num_data_points, max_cg_beads),
                            dtype='i4', fillvalue=-1)
        
    if store_in_dataset_args["store_bond_distance_features"]:
        # num_bonds_per_dataset = [h5_file["bond_distance_features"].shape[-1] for h5_file in all_h5py_files]
        num_bonds_per_dataset = num_cg_beads_per_dataset - 1
        max_num_bonds = max_cg_beads - 1
        file.create_dataset("bond_distance_features", (num_data_points, max_num_bonds),
                            dtype='i4', fillvalue=-1)
        file.create_dataset("bond_distance_indices", (num_data_points, max_num_bonds, 2),
                            dtype='i4', fillvalue=-1)
    
        
    if store_in_dataset_args["store_bond_angle_features"]:
        # num_angles_per_dataset = [h5_file["angle_features"].shape[-1] for h5_file in all_h5py_files]
        num_angles_per_dataset = num_cg_beads_per_dataset - 2
        max_num_angles = max_cg_beads - 2
        file.create_dataset("angle_features", (num_data_points, max_num_angles),
                            dtype='i4', fillvalue=-1)
        file.create_dataset("angle_indices", (num_data_points, max_num_angles, 3),
                            dtype='i4', fillvalue=-1)
        
    if store_in_dataset_args["store_dihedral_angle_features"]:
        # num_dihedral_angles_per_dataset = [h5_file["dihedral_features"].shape[-1] 
        #                                    for h5_file in all_h5py_files]
        num_dihedral_angles_per_dataset = num_cg_beads_per_dataset - 3
        max_num_dihedral_angles = max_cg_beads - 3
        file.create_dataset("dihedral_features", (num_data_points, max_num_dihedral_angles),
                            dtype='i4', fillvalue=-1)
        file.create_dataset("dihedral_indices", (num_data_points, max_num_dihedral_angles, 4),
                            dtype='i4', fillvalue=-1)
        
    if store_in_dataset_args["store_nonbonded_features"]:
        num_pairwise_per_dataset = [h5_file["nonbonded_edge_features"].shape[1] 
                                    for h5_file in all_h5py_files]
        max_num_pairwise = max(num_pairwise_per_dataset)
        file.create_dataset("nonbonded_edge_features", (num_data_points, max_num_pairwise, 2),
                            dtype='i4', fillvalue=-1)
        file.create_dataset("nonbonded_edge_indices", (num_data_points, max_num_pairwise, 2),
                            dtype='i4', fillvalue=-1)
    

    file.attrs["length_units"] = length_units
    file.attrs["energy_units"] = energy_units
    
    

    for i in range(len(all_h5py_files)):
        print(i)
        h5py_file = all_h5py_files[i]

        start = num_data_points_per_dataset_cum[i]
        end = num_data_points_per_dataset_cum[i + 1]
        if store_in_dataset_args["store_positions"]:
            cg_positions = da.from_array(h5py_file["positions"], chunks="auto")
            da.store(cg_positions, 
                     file["positions"],
                     regions=(slice(start, end),
                              slice(0, int(num_cg_beads_per_dataset[i]))))
        if store_in_dataset_args["store_forces"]:
            cg_forces = da.from_array(h5py_file["forces"], chunks="auto")
            da.store(cg_forces, 
                     file["forces"],
                     regions=(slice(start, end),
                              slice(0, int(num_cg_beads_per_dataset[i]))))
        if store_in_dataset_args["store_energies"]:
            cg_energies = da.from_array(h5py_file["energies"], chunks="auto")
            da.store(cg_energies, 
                     file["energies"],
                     regions=(slice(start, end),
                              slice(None)))
        if store_in_dataset_args["store_features"]:
            cg_features = da.from_array(h5py_file["features"], chunks="auto")
            da.store(cg_features, 
                     file["features"],
                     regions=(slice(start, end),
                              slice(0, int(num_cg_beads_per_dataset[i]))))
        if store_in_dataset_args["store_atom_features"]: 
            cg_features = da.from_array(h5py_file["atom_features"], chunks="auto")
            da.store(cg_features, 
                     file["atom_features"],
                     regions=(slice(start, end),
                              slice(0, int(num_cg_beads_per_dataset[i]))))
        if store_in_dataset_args["store_bond_distance_features"]:
            cg_bond_distance_features = da.from_array(h5py_file["bond_distance_features"], chunks=(10, num_bonds_per_dataset[i]))
            cg_bond_distance_indices = da.from_array(h5py_file["bond_distance_indices"], chunks=(10, num_bonds_per_dataset[i], 2))
            da.store(cg_bond_distance_features, file["bond_distance_features"],
                     regions=(slice(start, end),
                              slice(0, int(num_bonds_per_dataset[i]))))
            da.store(cg_bond_distance_indices, file["bond_distance_indices"],
                     regions=(slice(start, end),
                              slice(0, int(num_bonds_per_dataset[i])),
                              slice(0, 2)))

        if store_in_dataset_args["store_bond_angle_features"]:
            cg_angle_features = da.from_array(h5py_file["angle_features"], chunks=(10, num_angles_per_dataset[i]))
            cg_angle_indices = da.from_array(h5py_file["angle_indices"], chunks=(10, num_angles_per_dataset[i], 3))
            da.store(cg_angle_features, file["angle_features"],
                     regions=(slice(start, end),
                              slice(0, int(num_angles_per_dataset[i]))))
            da.store(cg_angle_indices, file["angle_indices"],
                     regions=(slice(start, end),
                              slice(0, int(num_angles_per_dataset[i])),
                              slice(0, 3)))
        if store_in_dataset_args["store_dihedral_angle_features"]:
            cg_dihedral_features = da.from_array(h5py_file["dihedral_features"], chunks=(10, num_dihedral_angles_per_dataset[i]))
            cg_dihedral_indices = da.from_array(h5py_file["dihedral_indices"], chunks=(10, num_dihedral_angles_per_dataset[i], 4))
            da.store(cg_dihedral_features, file["dihedral_features"],
                     regions=(slice(start, end),
                              slice(0, int(num_dihedral_angles_per_dataset[i]))))
            da.store(cg_dihedral_indices, file["dihedral_indices"],
                     regions=(slice(start, end),
                              slice(0, int(num_dihedral_angles_per_dataset[i])),
                              slice(0, 4)))
        if store_in_dataset_args["store_nonbonded_features"]:
            cg_pairwise_features = da.from_array(h5py_file["nonbonded_edge_features"], chunks=(10, num_pairwise_per_dataset[i], 2))
            cg_pairwise_indices = da.from_array(h5py_file["nonbonded_edge_indices"], chunks=(10, num_pairwise_per_dataset[i], 2))
            da.store(cg_pairwise_features, file["nonbonded_edge_features"],
                     regions=(slice(start, end),
                              slice(0, int(num_pairwise_per_dataset[i])),
                              slice(0, 2)))
            da.store(cg_pairwise_indices, file["nonbonded_edge_indices"],
                     regions=(slice(start, end),
                              slice(0, int(num_pairwise_per_dataset[i])),
                              slice(0, 2)))



        start = num_data_points_per_dataset_cum[i]
        end = num_data_points_per_dataset_cum[i + 1]


    file.flush()
    file.close()

def get_bond_distance_features(cg_topology, batch_size, as_tensor=False, device=None):
    all_bond_indices = [[i, i + 1]
                            for i in range(cg_topology.n_atoms - 1)]
    all_bond_indices = np.array(all_bond_indices)
    all_bond_features = [all_bond_types[cg_topology.atom(atom_index_1).name,
                                        cg_topology.atom(atom_index_2).name]
                            for (atom_index_1, atom_index_2) in all_bond_indices]

    all_bond_features = np.array(all_bond_features)

    if batch_size > 0:
        all_bond_features = np.expand_dims(all_bond_features, axis=0)
        all_bond_features = np.repeat(all_bond_features, batch_size,
                                        axis=0)

        all_bond_indices = np.expand_dims(all_bond_indices, axis=0)
        all_bond_indices = np.repeat(all_bond_indices, batch_size,
                                        axis=0)
    if as_tensor:
        all_bond_indices = torch.tensor(all_bond_indices).to(device)
        all_bond_features = torch.tensor(all_bond_features).to(device)
    return all_bond_indices, all_bond_features

def get_bond_angle_features(cg_topology, batch_size, as_tensor=False, device=None):    
    all_angle_indices = [[i, i + 1, i + 2]
                             for i in range(cg_topology.n_atoms - 2)]
    all_angle_indices = np.array(all_angle_indices)

    all_angle_features = [all_angle_types[amber_res_code_to_standard[cg_topology.atom(atom_index_1).residue.name],
                                          amber_res_code_to_standard[cg_topology.atom(atom_index_2).residue.name],
                                          amber_res_code_to_standard[cg_topology.atom(atom_index_3).residue.name],
                                          cg_topology.atom(atom_index_1).name,
                                          cg_topology.atom(atom_index_2).name,
                                          cg_topology.atom(atom_index_3).name]
                            for (atom_index_1, atom_index_2, atom_index_3) in all_angle_indices]

    all_angle_features = np.array(all_angle_features)

    if batch_size > 0:
        all_angle_features = np.expand_dims(all_angle_features, axis=0)
        all_angle_features = np.repeat(all_angle_features,
                                        batch_size, axis=0)

        all_angle_indices = np.expand_dims(all_angle_indices, axis=0)
        all_angle_indices = np.repeat(all_angle_indices,
                                        batch_size, axis=0)
    if as_tensor:
        all_angle_indices = torch.tensor(all_angle_indices).to(device)
        all_angle_features = torch.tensor(all_angle_features).to(device)
    return all_angle_indices, all_angle_features

def get_dihedral_angle_features(cg_topology, batch_size, as_tensor=False, device=None):
    all_dihdral_indices = [[i, i + 1, i + 2, i + 3]
                               for i in range(cg_topology.n_atoms - 3)]
    all_dihdral_indices = np.array(all_dihdral_indices)

    all_dihedral_features = [all_dihedral_angle_types[amber_res_code_to_standard[cg_topology.atom(atom_index_1).residue.name],
                                                      amber_res_code_to_standard[cg_topology.atom(atom_index_2).residue.name],
                                                      amber_res_code_to_standard[cg_topology.atom(atom_index_3).residue.name],
                                                      amber_res_code_to_standard[cg_topology.atom(atom_index_4).residue.name],
                                                      cg_topology.atom(atom_index_1).name,
                                                      cg_topology.atom(atom_index_2).name,
                                                      cg_topology.atom(atom_index_3).name,
                                                      cg_topology.atom(atom_index_4).name]
                                for (atom_index_1, atom_index_2, atom_index_3, atom_index_4) in all_dihdral_indices]
    all_dihedral_features = np.array(all_dihedral_features)

    if batch_size > 0:
        all_dihedral_features = np.expand_dims(all_dihedral_features, axis=0)
        all_dihedral_features = np.repeat(all_dihedral_features,
                                            batch_size, axis=0)

        all_dihdral_indices = np.expand_dims(all_dihdral_indices, axis=0)
        all_dihdral_indices = np.repeat(all_dihdral_indices,
                                        batch_size, axis=0)
    if as_tensor:
        all_dihdral_indices = torch.tensor(all_dihdral_indices).to(device)
        all_dihedral_features = torch.tensor(all_dihedral_features).to(device)
    return all_dihdral_indices, all_dihedral_features

def get_atom_features(cg_topology, batch_size, as_tensor=False, device=None):
    # all_atom_features = [(res_code_to_feat_number[amber_res_code_to_standard[cg_topology.atom(i).residue.name]], 
    #                       atom_type_to_feat_number[cg_topology.atom(i).name]) for i in range(cg_topology.n_atoms)]
    all_atom_features = [all_pairwise_types[amber_res_code_to_standard[cg_topology.atom(i).residue.name],
                                                    cg_topology.atom(i).name] for i in range(cg_topology.n_atoms)]
    all_atom_features = np.array(all_atom_features)

    if batch_size > 0:
        all_atom_features = np.expand_dims(all_atom_features, axis=0)
        all_atom_features = np.repeat(all_atom_features, batch_size,
                                        axis=0)

    if as_tensor:
        all_atom_features = torch.tensor(all_atom_features).to(device)
    return all_atom_features

def get_nonbonded_features(cg_topology, batch_size, as_tensor=False, device=None):
    all_nonbonded_indices = ([[i, j]
                        for i in range(cg_topology.n_atoms)
                        for j in range(i + 4, cg_topology.n_atoms)])
    
    all_nonbonded_features = ([[all_pairwise_types[amber_res_code_to_standard[cg_topology.atom(atom_index_1).residue.name],
                                                        cg_topology.atom(atom_index_1).name],
                                                        all_pairwise_types[amber_res_code_to_standard[cg_topology.atom(atom_index_2).residue.name],
                                                        cg_topology.atom(atom_index_2).name]]                  
                                                for (atom_index_1, atom_index_2) in all_nonbonded_indices])
    if batch_size > 0:
        all_nonbonded_indices = np.array(all_nonbonded_indices)
        all_nonbonded_indices = np.expand_dims(all_nonbonded_indices, axis=0)
        all_nonbonded_indices = np.repeat(all_nonbonded_indices, batch_size,
                                                    axis=0)
        
        all_nonbonded_features = np.array(all_nonbonded_features)
        all_nonbonded_features = np.expand_dims(all_nonbonded_features, axis=0)
        all_nonbonded_features = np.repeat(all_nonbonded_features, batch_size,
                                       axis=0)
    
    if as_tensor:
        all_nonbonded_indices = torch.tensor(all_nonbonded_indices).to(device)
        all_nonbonded_features = torch.tensor(all_nonbonded_features).to(device)
    
    return all_nonbonded_indices, all_nonbonded_features

def get_esm2_embeddings(cg_topology, batch_size, as_tensor=False, device=None):
    residues = [str(residue.code) for residue in cg_topology.residues]
    sequence = "".join(residues)
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
             results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
    token_representations = results["representations"][model.num_layers]
    esm2_embeddings = token_representations[0, 1:len(sequence)+1].cpu().numpy()
    
    if batch_size > 0:
        esm2_embeddings = np.expand_dims(esm2_embeddings, axis=0)
        esm2_embeddings = np.repeat(esm2_embeddings, batch_size,
                                        axis=0)
    if as_tensor:
        esm2_embeddings = torch.tensor(esm2_embeddings).to(device)
        
    return esm2_embeddings

def get_cg_bead_masses(cg_topology):
    h_mass = 1.00784
    o_mass = 15.999
    num_atoms = cg_topology.n_atoms
    num_residues = cg_topology.n_residues
    masses = []
    for i in range(num_atoms):
        atom = cg_topology.atom(i)
        res_name = amber_res_code_to_standard[atom.residue.name]
        atom_name = atom.name
        mass = cg_bead_mass[res_name, atom_name]
        if atom.residue.index == 0 and res_name != "ACE" and atom_name == "N":
            mass += h_mass
        elif atom.residue.index == num_residues - 1 and res_name != "NME" and atom_name == "C":
            mass += o_mass
            mass += h_mass
        masses.append(mass)
    masses = np.array(masses).astype(np.float32)
    return masses
    
    
    