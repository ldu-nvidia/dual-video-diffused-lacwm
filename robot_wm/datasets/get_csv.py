import csv
import os

import h5py


def get_traj_ids(h5_path):
    """Return all top-level group names inside the HDF5 file."""
    trajs = []
    try:
        with h5py.File(h5_path, "r") as f:
            for key in f["data"].keys():
                trajs.append(key)
    except Exception as e:
        print(f"[ERROR] Could not read {h5_path}: {e}")
    return trajs


def find_hdf5_files(
    root_dir,
    output_csv="/svl/u/ravenh/lacwm/robot_world_models-raven-lam/robot_wm/datasets/csv_files/libero.csv",
):
    train_rows = []
    val_rows = []

    # Scan directories recursively
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".hdf5"):
                full_path = os.path.abspath(os.path.join(dirpath, filename))

                # Determine train vs val
                is_val = "unseen" in dirpath

                # Extract trajectories
                traj_ids = get_traj_ids(full_path)

                # If file has no traj, still keep one row with empty ID
                if not traj_ids:
                    row = [full_path, ""]
                    (val_rows if is_val else train_rows).append(row)
                else:
                    for tid in traj_ids:
                        row = [full_path, tid]
                        (val_rows if is_val else train_rows).append(row)

    # ---- Write train CSV ----
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "traj_id"])
        writer.writerows(train_rows)

    # ---- Write val CSV ----
    val_csv = output_csv.replace(".csv", "_val.csv")
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "traj_id"])
        writer.writerows(val_rows)

    print(f"✔ Train HDF5 files: {len(train_rows)} trajectories")
    print(f"✔ Val HDF5 files:   {len(val_rows)} trajectories")
    print(f"✔ Saved to:\n    {output_csv}\n    {val_csv}")


if __name__ == "__main__":
    root = "/viscam/projects/dexs2r/libero_data"
    find_hdf5_files(root)
