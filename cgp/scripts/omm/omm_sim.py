import os
import sys
import traceback
import hydra
import cgp.omm
import numpy as np
from omegaconf import OmegaConf
from cgp.omm.utils import parse_checkpoints, concatenate_trajectories


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="omm_sim")
def main(cfg):
    global_args = cfg.global_args
    error_log = global_args.error_log

    try:
        _run(cfg, global_args)
    except Exception:
        if error_log is None:
            raise
        with open(error_log, "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)


def _run(cfg, global_args):
    amber_filename = global_args.amber_filename
    save_folder = global_args.save_folder
    save_filename = global_args.save_filename
    num_data_points = global_args.num_data_points
    save_freq = global_args.save_freq
    chk_freq = global_args.chk_freq
    frames_per_file = global_args.frames_per_file
    burn_in_data_points = global_args.burn_in_data_points
    starting_positions = global_args.starting_positions

    if num_data_points is not None:
        assert num_data_points % frames_per_file == 0, (
            f"num_data_points ({num_data_points}) must be an exact multiple of "
            f"frames_per_file ({frames_per_file}). Choose values such that the "
            f"remainder is  zero."
        )

    omm_config = cfg.omm
    simulation_args = OmegaConf.to_container(omm_config.simulation_args, resolve=True)
    solvent_args = (
        OmegaConf.to_container(omm_config.solvent_args, resolve=True)
        if omm_config.solvent_args is not None
        else None
    )

    tw_args = cfg.tw

    full_save_filename = os.path.join(save_folder, save_filename)

    name = os.path.basename(full_save_filename)
    traj_dir = os.path.join(save_folder, "trajectories")
    chk_dir = os.path.join(save_folder, "checkpoints")

    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(chk_dir, exist_ok=True)

    # i_last_final: index of the last fully written trajectory file (-1 if none)
    # i_cur:        index of the trajectory file to write next (= i_last_final + 1)
    # j_cur:        index of the last in-progress checkpoint within i_cur (-1 if none)
    i_cur, j_cur, i_last_final = parse_checkpoints(chk_dir, name)

    if chk_freq is None:
        chk_freq = frames_per_file

    start_frame = (j_cur + 1) * chk_freq
    total_frames = i_cur * frames_per_file + start_frame

    if num_data_points is not None and total_frames >= num_data_points:
        print(f"Target of {num_data_points} frames reached. Exiting.")
        return

    main_fresh = i_cur == 0 and j_cur == -1

    # --- burn-in checkpoint state ---
    burn_in_complete = False
    burn_in_start_frame = 0
    bj_cur = -1
    if burn_in_data_points is not None:
        _, bj_cur, bi_last_final = parse_checkpoints(chk_dir, f"{name}_burn_in")
        burn_in_complete = bi_last_final >= 0
        if not burn_in_complete:
            burn_in_start_frame = (bj_cur + 1) * chk_freq

    if burn_in_data_points is not None:
        truly_fresh = main_fresh and not burn_in_complete and bj_cur == -1
    else:
        truly_fresh = main_fresh

    if j_cur >= 0:
        chkpt_filename = os.path.join(chk_dir, f"{name}_{i_cur}_{j_cur}.chk")
    elif i_last_final >= 0:
        chkpt_filename = os.path.join(chk_dir, f"{name}_{i_last_final}_final.chk")
    elif burn_in_complete:
        chkpt_filename = os.path.join(chk_dir, f"{name}_burn_in_0_final.chk")
    elif burn_in_data_points is not None and bj_cur >= 0:
        chkpt_filename = os.path.join(chk_dir, f"{name}_burn_in_0_{bj_cur}.chk")
    else:
        chkpt_filename = None

    if truly_fresh:
        OmegaConf.save(cfg, f"{full_save_filename}_config.yaml")

    p = getattr(cgp.omm, omm_config["simulation_type"])
    p = p(
        filename=amber_filename,
        simulation_args=simulation_args,
        solvent_args=solvent_args,
        save_filename=full_save_filename,
        chkpt_filename=chkpt_filename,
    )

    if (
        starting_positions is not None
        and os.path.isfile(starting_positions)
        and truly_fresh
    ):
        starting_positions = np.load(starting_positions)
        p.update_positions_and_velocities(
            positions=starting_positions,
            velocities=True,
            length_units="angstroms",
            time_units="picoseconds",
        )

    if burn_in_data_points is not None and not burn_in_complete:
        burn_in_traj_path = os.path.join(chk_dir, f"{name}_burn_in.hdf5")
        p.generate_trajectory(
            traj_path=burn_in_traj_path,
            chk_dir=chk_dir,
            name=f"{name}_burn_in",
            traj_idx=0,
            frames_per_file=burn_in_data_points,
            save_freq=save_freq,
            chk_freq=chk_freq,
            start_frame=burn_in_start_frame,
            end_frame=burn_in_data_points,
            tw_args=tw_args,
        )

    while num_data_points is None or total_frames < num_data_points:
        budget_remaining = (
            (num_data_points - total_frames)
            if num_data_points is not None
            else frames_per_file
        )
        end_frame = min(start_frame + budget_remaining, frames_per_file)

        traj_path = os.path.join(traj_dir, f"{name}_{i_cur}.hdf5")
        p.generate_trajectory(
            traj_path=traj_path,
            chk_dir=chk_dir,
            name=name,
            traj_idx=i_cur,
            frames_per_file=frames_per_file,
            save_freq=save_freq,
            chk_freq=chk_freq,
            start_frame=start_frame,
            end_frame=end_frame,
            tw_args=tw_args,
        )

        total_frames += end_frame - start_frame

        if end_frame >= frames_per_file:
            i_cur += 1
            start_frame = 0
        else:
            i_cur += 1
            break

    out_path = f"{full_save_filename}_traj_all.hdf5"
    concatenate_trajectories(traj_dir, name, i_cur, out_path)
    print(f"Saved concatenated trajectories to {out_path}")


if __name__ == "__main__":
    main()
