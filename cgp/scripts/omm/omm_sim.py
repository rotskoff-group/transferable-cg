import os                   
import hydra
import cgp.omm
import numpy as np
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="omm_sim")
def main(cfg):
    global_args = cfg.global_args
    
    folder_name = global_args.folder_name
    amber_filename = global_args.amber_filename
    save_filename = global_args.save_filename
    num_data_points = global_args.num_data_points
    save_freq = global_args.save_freq
    starting_positions = global_args.starting_positions

    omm_config = cfg.omm
    simulation_args = omm_config.simulation_args
    solvent_args = omm_config.solvent_args

    tw_args = cfg.tw

    if simulation_args["chk_freq"] is None:
        simulation_args["chk_freq"] = save_freq * num_data_points



    chk_files = [f for f in os.listdir(folder_name) if f.endswith(".chk")]
    chk_files = [f for f in chk_files
                 if f[0:len(amber_filename)] == amber_filename]
    if len(chk_files) == 0:
        max_chk = 0
    else:
        max_chk = max([int(f.split("_")[-1].split(".")[0]) for f in chk_files]) + 1
    

    if save_filename is not None:
        save_filename = f"{folder_name}{save_filename}"
    OmegaConf.save(cfg, f"{save_filename}_config_{max_chk}.yaml")



    amber_filename = f"{folder_name}{amber_filename}"
    p = getattr(cgp.omm, omm_config["simulation_type"])
    p = p(filename = amber_filename, chk = max_chk, 
          simulation_args=simulation_args, solvent_args=solvent_args,
          save_filename=save_filename)

    if starting_positions != None and os.path.isfile(starting_positions) and max_chk == 0:
        starting_positions = np.load(starting_positions)
        p.update_positions_and_velocities(positions=starting_positions,
                                          velocities=True,
                                          length_units="angstroms",
                                          time_units="picoseconds")

    p.generate_trajectory(num_data_points=num_data_points,
                            save_freq=save_freq,
                            tw_args=tw_args)




if __name__ == "__main__":
    main()