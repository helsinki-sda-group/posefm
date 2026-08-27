from posefm import PoseFM
from utils.metrics import get_metrics


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os

def singlepu(config, id):
    # Predict and measure uncertainty
    posefm = PoseFM(config)
    posefm.predict_poses()
    scale = np.array([0.13,  0.13,  0.13,  0.013,  0.013,  0.013]).reshape((1, -1))

    mean = np.concatenate([posefm.poses[i]["mean"] for i in posefm.poses], axis=0)
    PoseFM.save_trajectory(mean * scale, f"outputs/{id}/mean.txt")
    std = np.concatenate([posefm.poses[i]["std"] for i in posefm.poses], axis=0)
    PoseFM.save_trajectory(std, f"outputs/{id}/std.txt")

    trans_std = np.mean(std[:, :3], axis=1) / np.mean(mean[:, :3], axis=1)
    rot_std = np.mean(std[:, 3:] / mean[:, 3:])
    print(f"Trans mean std: {np.mean(trans_std)}, rot mean std: {rot_std}")

    traj, gt_traj = posefm.create_trajectory(scale=scale)
    pred = np.concatenate([posefm.poses[i]["pred"] for i in posefm.poses], axis=0)
    gt = np.concatenate([posefm.poses[i]["gt"] for i in posefm.poses], axis=0)

    PoseFM.save_trajectory(pred * scale, f"outputs/{id}/pred_poses.txt")
    PoseFM.save_trajectory(gt , f"outputs/{id}/gt_poses.txt")

    # Save individual trajectories
    multitrajs = None
    for i in posefm.poses:
        if i == -1:
            continue
        if multitrajs is None:
            multitrajs = posefm.poses[i]["samples"]
            print(posefm.poses[i]["samples"].shape)
        else:
            multitrajs = np.concatenate([multitrajs, posefm.poses[i]["samples"]], axis=0)
    for i in range(multitrajs.shape[1]):
        PoseFM.save_trajectory(multitrajs[:,i,:] * scale, f"outputs/{id}/pred_poses_{i}.txt")
    filename = "traj"
    pred_path = f"outputs/{id}/pred_{filename}.txt"
    gt_path = f"outputs/{id}/gt_{filename}.txt"
    PoseFM.save_trajectory(traj, pred_path)
    PoseFM.save_trajectory(gt_traj, gt_path)
    ause(pred*scale, gt, std)
    return get_metrics(pred_path, gt_path)

def multipu():
    import tomllib
    config_path = "configs/tartantest.toml"
    with open(config_path, 'rb') as file:
        config = tomllib.load(file)
    conditions = [f"ME00{i}" for i in range(0, 8)]
    conditions.extend([f"MH00{i}" for i in range(0, 8)])
    conditions = ["MH004"]
    res_dict = {}
    model = config["model"]["model_path"].lstrip("ckpts/").rstrip(".pth")
    for cond in conditions:
        config["data"]["test"]["data_path"] = f"{config["data"]["test"]["data_path"]}/{cond}"
        id = f"uncert/{cond}/{model}"
        res = singlepu(config, id)
        res_dict[cond] = res
    ates = []
    for cond in res_dict:
        print(f"Cond: {cond} - ATE:\n{res_dict[cond]['ATE(m)']}")
        ates.append(res_dict[cond]['ATE(m)'])
    print(ates)

def _ause_curve(df):
    df = df.reset_index(drop=True)

    n = len(df)

    coverage = np.arange(1, n + 1) / n

    # Uncertainty-based sparsification
    df_uncert = df.sort_values(
        "std",
        ascending=True
    ).reset_index(drop=True)

    uncertainty_risk = (
        df_uncert["err"].cumsum()
        / np.arange(1, n + 1)
    )

    
    # Oracle sparsification
    df_oracle = df.sort_values(
        "err",
        ascending=True
    ).reset_index(drop=True)

    oracle_risk = (
        df_oracle["err"].cumsum()
        / np.arange(1, n + 1)
    )

    
    # Random sparsification
    random_risk = df["err"].mean()

    
    # Sparsification error
    sparsification_error = (
        uncertainty_risk - oracle_risk
    )

    
    # Plot
    # plt.figure(figsize=(8, 6))

    # plt.plot(
    #     coverage[n_5::n_5],
    #     uncertainty_risk[n_5::n_5],
    #     label="Uncertainty-based",
    #     linewidth=2
    # )

    # plt.plot(
    #     coverage[n_5::n_5],
    #     oracle_risk[n_5::n_5],
    #     label="Oracle",
    #     linewidth=2,
    #     linestyle="--"
    # )

    # plt.axhline(
    #     random_risk,
    #     color="red",
    #     linestyle=":",
    #     label=f"Random ({random_risk:.3f})"
    # )

    # plt.xlabel("Coverage")
    # plt.ylabel("Mean error")
    # plt.title("Sparsification Curve")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    
    # AUSE
    ause = np.trapz(
        sparsification_error,
        coverage
    )

    # Normalized AUSE
    if random_risk > 0:
        ause_normalized = ause / random_risk
    else:
        ause_normalized = np.nan

    print(f"AUSE:              {ause:.6f}")
    print(f"Normalized AUSE:   {ause_normalized:.6f}")

    return ause_normalized

def ause(pred, gt, std):
    
    errort = np.abs(pred[:, :3] - gt[:, :3])
    com_errot = np.linalg.norm(errort, axis=1)
    com_stdt = np.linalg.norm(std[:, :3], axis=1)
    dft = pd.DataFrame({"err": com_errot, "std": com_stdt})
    _ause_curve(dft)

    # Rotation error
    R_pred = R.from_rotvec(pred[:, 3:])
    R_gt   = R.from_rotvec(gt[:, 3:])
    R_err = R_pred.inv() * R_gt
    # degrees
    com_error = np.rad2deg(R_err.magnitude())

    # Rotation uncertainty
    com_stdr = np.linalg.norm(std[:, 3:], axis=1)

    df = pd.DataFrame({"err": com_error, "std": com_stdr})
    _ause_curve(df)

def ause_file():
    data_dir = f"outputs/uncert/MH004/TartanVO_0303_021727_ck"
    gt_path = os.path.join(data_dir, "gt_poses.txt")
    pred_path = os.path.join(data_dir, "pred_poses.txt")
    std_path = os.path.join(data_dir, "std.txt")
    df = {}
    if not (
        os.path.exists(gt_path)
        and os.path.exists(pred_path)
        and os.path.exists(std_path)):
        raise ValueError

    gt = np.loadtxt(gt_path)
    pred = np.loadtxt(pred_path)
    std = np.loadtxt(std_path)
    ause(pred, gt, std)

if __name__ == "__main__":
    multipu()
    # ause_file()