import torch

atom_type_to_feat_number = {"CA": 0, "C": 1, "N": 2, "CH3": 0}


res_code_to_feat_number = {"ACE": 0,
                           "NME": 1,
                           "ALA": 2,
                           "ARG": 3,
                           "ASN": 4,
                           "ASP": 5,
                           "ASH": 5, # Neutral ASP
                           "CYS": 6,
                           "GLN": 7,
                           "GLU": 8,
                           "GLH": 8, # Neutral GLU
                           "GLY": 9,
                           "HID": 10, # H on delta N
                           "HIE": 10, # H on epsilon N
                           "HIP": 10, # Protonated HIS
                           "HIS": 10,
                           "ILE": 11,
                           "LEU": 12,
                           "LYS": 13,
                           "LYN": 13, # Protonated LYS
                           "MET": 14,
                           "PHE": 15,
                           "PRO": 16,
                           "SER": 17,
                           "THR": 18,
                           "TRP": 19,
                           "TYR": 20,
                           "VAL": 21,
                           "CYX": 22} # Cysteine with disulfide bond

amber_res_code_to_standard = {"ACE": "ACE",
                           "NME": "NME",
                           "ALA": "ALA",
                           "ARG": "ARG",
                           "ASN": "ASN",
                           "ASP": "ASP",
                           "ASH": "ASP", # Neutral ASP
                           "CYS": "CYS",
                           "GLN": "GLN",
                           "GLU": "GLU",
                           "GLH": "GLU", # Neutral GLU
                           "GLY": "GLY",
                           "HID": "HIS", # H on delta N
                           "HIE": "HIS", # H on epsilon N
                           "HIP": "HIS", # Protonated HIS
                           "HIS": "HIS",
                           "ILE": "ILE",
                           "LEU": "LEU",
                           "LYS": "LYS",
                           "LYN": "LYS", # Protonated LYS
                           "MET": "MET",
                           "PHE": "PHE",
                           "PRO": "PRO",
                           "SER": "SER",
                           "THR": "THR",
                           "TRP": "TRP",
                           "TYR": "TYR",
                           "VAL": "VAL",
                           "CYX": "CYX"} # Cysteine with disulfide bond
res_code_cg = [ "ACE",
                "NME",
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
                "GLU",
                "GLY",
                "HIS",
                "ILE",
                "LEU",
                "LYS",
                "MET",
                "PHE",
                "PRO",
                "SER",
                "THR",
                "TRP",
                "TYR",
                "VAL",
                "CYX"]

all_bond_types = {("C", "N"): 0,
                  ("N", "CA"): 1,
                  ("CA", "C"): 2,
                  ("CH3", "C"): 3,
                  ("N", "C"): 4}

non_cap_residues = [res_code
                    for res_code in res_code_cg
                    if res_code not in ["ACE", "NME"]]
non_n_cap_residues = [res_code
                      for res_code in res_code_cg
                      if res_code != "ACE"]
non_c_cap_residues = [res_code
                      for res_code in res_code_cg
                      if res_code != "NME"]


num_non_cap_residues = len(non_cap_residues)
num_non_n_cap_residues = len(non_n_cap_residues)
num_non_c_cap_residues = len(non_c_cap_residues)
num_residues = len(res_code_cg)

all_angle_types = ({(res, res, res, "N", "CA", "C"): i for (i, res) in enumerate(non_cap_residues)}
                   # res_1 can't be ACE or NME # res_2 can't be ACE
                   | {(res_1, res_1, res_2, "CA", "C", "N"): (num_non_cap_residues
                                                              + i * (num_non_n_cap_residues) + j)
                      for (i, res_1) in enumerate(non_cap_residues)
                      for (j, res_2) in enumerate(non_n_cap_residues)}
                   # res_1 can't be NME # res_2 can't be ACE or NME
                   | {(res_1, res_2, res_2, "C", "N", "CA"): (num_non_cap_residues
                                                              + (num_non_n_cap_residues * num_non_cap_residues)
                                                              + i * (num_non_cap_residues) + j)
                       for (i, res_1) in enumerate(non_c_cap_residues)
                       for (j, res_2) in enumerate(non_cap_residues)}
                   | {("ACE", "ACE", res, "CH3", "C", "N"): (num_non_cap_residues
                                                             + (num_non_n_cap_residues * num_non_cap_residues)
                                                             + (num_non_c_cap_residues * num_non_cap_residues)
                                                             + i)
                       for (i, res) in enumerate(non_cap_residues)}
                   | {(res, "NME", "NME", "C", "N", "C"): (2 * num_non_cap_residues
                                                           + (num_non_n_cap_residues * num_non_cap_residues)
                                                           + (num_non_c_cap_residues * num_non_cap_residues)
                                                           + i)
                       for (i, res) in enumerate(non_cap_residues)})

# phi, psi, cap dihedrals
all_dihedral_angle_types = ({(res_1, res_2, res_2, res_2, "C", "N", "CA", "C"): i * num_non_cap_residues + j
                             for (i, res_1) in enumerate(non_c_cap_residues)
                             for (j, res_2) in enumerate(non_cap_residues)}
                            | {(res_1, res_1, res_1, res_2, "N", "CA", "C", "N"): (i * num_non_n_cap_residues + j
                                                                                   + (num_non_cap_residues * num_non_c_cap_residues))
                               for (i, res_1) in enumerate(non_cap_residues)
                               for (j, res_2) in enumerate(non_n_cap_residues)}
                            | {(res_1, res_1, res_2, res_2, "CA", "C", "N", "CA"): ((num_non_cap_residues * num_non_c_cap_residues)
                                                                                    + (num_non_cap_residues * num_non_n_cap_residues)
                                                                                    + (i * num_non_n_cap_residues + j))
                               for (i, res_1) in enumerate(non_cap_residues)
                               for (j, res_2) in enumerate(non_c_cap_residues)}
                            | {("ACE", "ACE", res, res, "CH3", "C", "N", "CA"): ((num_non_cap_residues * num_non_c_cap_residues)
                                                                                 + (num_non_cap_residues * num_non_c_cap_residues)
                                                                                 + (num_non_cap_residues * num_non_n_cap_residues)
                                                                                 + i)
                               for (i, res) in enumerate(non_cap_residues)}
                            | {(res, res, "NME", "NME", "CA", "C", "N", "C"): ((num_non_cap_residues * num_non_c_cap_residues)
                                                                               + (num_non_cap_residues * num_non_c_cap_residues)
                                                                               + (num_non_cap_residues * num_non_n_cap_residues)
                                                                               + num_non_cap_residues
                                                                               + i)
                               for (i, res) in enumerate(non_cap_residues)})

all_pairwise_types = ({(res, atom): (i * 3 + k)
                          for (i, res) in enumerate(non_cap_residues)
                          for (k, atom) in enumerate(["N", "CA", "C"])}
                         | {("ACE", atom): (num_non_cap_residues * 3 + k)
                            for (k, atom) in enumerate(["CH3", "C"])}
                         | {("NME", atom): (num_non_c_cap_residues* 3 + k - 1)
                            for (k, atom) in enumerate(["N", "C"])})

cg_bead_mass = {('ACE', 'C'): 43.044831,
                ('ALA', 'N'): 15.014667,
                ('ALA', 'C'): 28.01021,
                ('ALA', 'CA'): 28.053348000000007,
                ('ARG', 'N'): 15.014667,
                ('ARG', 'C'): 28.01021,
                ('ARG', 'CA'): 114.16942400000002,
                ('ASN', 'N'): 15.014667,
                ('ASN', 'C'): 28.01021,
                ('ASN', 'CA'): 71.07822500000002,
                ('ASP', 'N'): 15.014667,
                ('ASP', 'C'): 28.01021,
                ('ASP', 'CA'): 71.05504100000002,
                ('CYS', 'N'): 15.014667,
                ('CYS', 'C'): 28.01021,
                ('CYS', 'CA'): 60.11884800000001,
                ('GLN', 'N'): 15.014667,
                ('GLN', 'C'): 28.01021,
                ('GLN', 'CA'): 85.10489900000002,
                ('GLU', 'N'): 15.014667,
                ('GLU', 'C'): 28.01021,
                ('GLU', 'CA'): 85.08171500000002,
                ('GLY', 'N'): 15.014667,
                ('GLY', 'C'): 28.01021,
                ('GLY', 'CA'): 14.026674,
                ('HIS', 'N'): 15.014667,
                ('HIS', 'C'): 28.01021,
                ('HIS', 'CA'): 94.11502200000001,
                ('ILE', 'N'): 15.014667,
                ('ILE', 'C'): 28.01021,
                ('ILE', 'CA'): 70.13337000000001,
                ('LEU', 'N'): 15.014667,
                ('LEU', 'C'): 28.01021,
                ('LEU', 'CA'): 70.13337000000001,
                ('LYS', 'N'): 15.014667,
                ('LYS', 'C'): 28.01021,
                ('LYS', 'CA'): 86.15598400000002,
                ('MET', 'N'): 15.014667,
                ('MET', 'C'): 28.01021,
                ('MET', 'CA'): 88.17219600000001,
                ('PHE', 'N'): 15.014667,
                ('PHE', 'C'): 28.01021,
                ('PHE', 'CA'): 104.149816,
                ('PRO', 'N'): 14.00672,
                ('PRO', 'C'): 28.01021,
                ('PRO', 'CA'): 55.09874900000001,
                ('SER', 'N'): 15.014667,
                ('SER', 'C'): 28.01021,
                ('SER', 'CA'): 44.052778,
                ('THR', 'N'): 15.014667,
                ('THR', 'C'): 28.01021,
                ('THR', 'CA'): 58.07945200000002,
                ('TRP', 'N'): 15.014667,
                ('TRP', 'C'): 28.01021,
                ('TRP', 'CA'): 143.186043,
                ('TYR', 'N'): 15.014667,
                ('TYR', 'C'): 28.01021,
                ('TYR', 'CA'): 120.149246,
                ('VAL', 'N'): 15.014667,
                ('VAL', 'C'): 28.01021,
                ('VAL', 'CA'): 56.106696000000014,
                ('NME', 'C'): 15.0154,
                ('NME', 'N'): 17.03022}



