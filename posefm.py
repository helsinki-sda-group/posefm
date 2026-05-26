# Copyright (c) 2026 Dominik Kuczkowski, SDA group, University of Helsinki
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

import torch
from tqdm import tqdm
import numpy as np
from voloader.transformation import ses2poses_quat, tartan2kitti

from utils.utils import create_data_loader, sample_model, load_state_dicts
from utils.metrics import get_metrics
from models.PoseFMNet import PoseFMNet

import tomllib
from pathlib import Path
from argparse import ArgumentParser
import os

class PoseFM:
    def __init__(self, config: str|dict):
        """PoseFM: Relative camera pose estimation for visual odometry using flow matching.
        Args:
            config (str|dict): Path to the configuration file or dict with config.
        """
        # Load config
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, 'rb') as file:
                self.config = tomllib.load(file)
        
        self.device = torch.device(self.config["posefm"]["device"])
        model_config = self.config["model"]
        self.model = PoseFMNet(**model_config["params"] if "params" in model_config else {}).to(self.device).float()
        self.model = load_state_dicts(self.model, config=model_config, device=self.device)
        self.dataloader = create_data_loader(**self.config["data"]["test"])["test"]
        self.cond = self.config["posefm"]["conditioning"]
        if self.cond not in ["flow", "img"]:
            raise ValueError(f"Unsupported conditiong type. Given: {self.cond}")
        self.numsamples = self.config["posefm"]["numsamples"] if "numsamples" in self.config["posefm"] else 1
    
    def predict_poses(self):
        """Predict camera poses using a trained model."""
        self.model.eval()
        with torch.no_grad():
            self.poses = {}
            for i, batch in enumerate(tqdm(self.dataloader, desc="Predicting poses")):  
                self.poses[i] = {}
                batch_size = batch["relpose"].shape[0]

                if self.cond == 'flow':
                    cond_batch = torch.cat([batch['flow'], batch['intrinsic']], dim=1).to(self.device, non_blocking=True)
                elif self.cond == 'img':
                    cond_batch = [batch['img1'].to(self.device, non_blocking=True), 
                                batch['img2'].to(self.device, non_blocking=True),
                                batch['intrinsic'].to(self.device, non_blocking=True)]
                
                sampled_poses = []
                for _ in range(self.numsamples):
                    sampled_poses.append(sample_model(self.model, 
                                        cond_batch, 
                                        self.device,
                                        batch_size,
                                        **self.config["posefm"]["sampling"]))


                if "relpose" in batch:
                    self.poses[i]["gt"] = batch["relpose"].numpy()
                    trans_scale = np.linalg.norm(batch["relpose"][:, :3], axis=1)
                
                samples = []
                for pose in sampled_poses:
                    pose = pose.cpu().numpy()
                    if "relpose" in batch:
                        # Perform translation scale alignment
                        scale = trans_scale / np.linalg.norm(pose[:, :3], axis=1)
                        pose[:, :3] = pose[:,:3] * scale[..., None]
                    samples.append(pose)

                self.poses[i]["pred"] = samples[0]
                self.poses[i]["samples"] = np.array(samples).transpose((1, 0, 2))

                if self.numsamples > 1:
                    self.poses[i]["mean"] = np.mean(self.poses[i]["samples"], axis=1)
                    self.poses[i]["std"] = np.std(self.poses[i]["samples"], axis=1)
    
    def create_trajectory(self, quat: bool = False, scale = None):
        """Create a 3D trajectory from the predicted relative poses.
        Args:
            quat (bool): If True, the poses are in quaternion format. If False, they are in KITTI rotation matrix format. Default: False
            scale (np.ndarray, optional): Training dataset derived std scale for predicitons. 
        Returns:
            trajectory (np.ndarray): Sequence of global poses based on the predicted relative poses.
        """
        trajectory = []
        gt_trajectory = []
        for ind, batch in self.poses.items():
            if ind < 0:
                continue

            if "mean" in batch:
                pred = batch["mean"]
            else:
                pred = batch["pred"]

            if scale is not None:
                pred = pred * scale
            
            gt = batch["gt"]
            trajectory.append(pred)
            gt_trajectory.append(gt)

        trajectory = ses2poses_quat(np.concatenate(trajectory, axis=0))
        gt_trajectory = ses2poses_quat(np.concatenate(gt_trajectory, axis=0))
        
        if quat:
            return trajectory, gt_trajectory
        
        return tartan2kitti(trajectory), tartan2kitti(gt_trajectory)

    @staticmethod 
    def save_trajectory(trajectory: np.ndarray, path: str):
        """Save the trajectory to a file.
        Args:
            trajectory (np.ndarray): Trajectory to save.
            path (str): Path to save the trajectory.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True) # Create directories if needed
        np.savetxt(path, trajectory, fmt='%.6f')

if __name__ == "__main__":
    parser = ArgumentParser(description="Run PoseFM with specified config")
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to the TOML configuration file"
    )
    parser.add_argument("--out", type=str, help="Output direcotry path")
    args = parser.parse_args()

    posefm = PoseFM(args.config)
    posefm.predict_poses() # Predict relative camera poses
    scale = np.array([0.13,  0.13,  0.13,  0.013,  0.013,  0.013]).reshape((1, -1))
    traj, gt_traj = posefm.create_trajectory(scale=scale) # Combine predicted poses into a trajectory
    os.makedirs(args.out, exist_ok=True)
    PoseFM.save_trajectory(traj, args.out + "/pred_traj.txt")

    # Runs when test data provided GT poses 
    if gt_traj is not None:
        PoseFM.save_trajectory(gt_traj, args.out + "/gt_traj.txt")
        metrics = get_metrics(args.out + "/pred_traj.txt", args.out + "/gt_traj.txt")
        print(metrics)