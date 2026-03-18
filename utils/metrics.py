from evo.core import metrics
from evo.core.units import Unit
from evo.tools import file_interface
from evo.tools import plot
import matplotlib.pyplot as plt
import numpy as np

def APE(pred_traj_path, gt_traj_path, plot_traj=False):
    """
    Calculate absolute pose error metrics. The metrics are:
    ATE (Absolute Translation (Trajectory) Error) - RMSE of translation between estimate and gt
    ARE (Absolute Rotation Error) - RMSE of rotation betwen estimate ang gt
    
    Args:
        pred_traj_path (str): Path to the predicted trajectory file.
        gt_traj_path (str): Path to the ground truth trajectory file.
        plot (bool): Flag that enables plotting the trajectory. Default: False
        
    Returns:
        dict: Metrics values accessed under their abbreviations.
    """
    # Load the predicted and ground truth trajectories
    pred_traj = file_interface.read_kitti_poses_file(pred_traj_path)
    gt_traj = file_interface.read_kitti_poses_file(gt_traj_path)
    path_length = gt_traj.path_length
    # Align points
    pred_traj.align(gt_traj, correct_scale=True)

    # Calculate translational error
    tran_ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    tran_ape_metric.process_data((gt_traj, pred_traj))

    tran_ape_stat = tran_ape_metric.get_statistic(metrics.StatisticsType.rmse)

    # Calculate rotational error
    rot_ape_metric = metrics.APE(metrics.PoseRelation.rotation_part)
    rot_ape_metric.process_data((gt_traj, pred_traj))

    rot_ape_stat = rot_ape_metric.get_statistic(metrics.StatisticsType.rmse)

    if plot_traj:
        fig = plt.figure()
        traj_by_label = {
            
            "estimate (aligned)": pred_traj,
            "reference": gt_traj
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()
    return {"GT len (m)": np.round(path_length, 2).item(), 
            "ATE(m)": np.round(tran_ape_stat, 3).item(), 
            "ATE(%)": np.round(tran_ape_stat / path_length * 100, 2).item(), 
            "ARE(temp)": np.round(rot_ape_stat, 3).item()}

def RPE(pred_traj_path, gt_traj_path):
    """
    Calculate relative pose error metrics. The metrics are:
    RTE (Relative Translation Error) - RMSE of translation drift between estimate and gt
    RRE (Relative Rotation Error) - RMSE of relative rotation drift betwen estimate ang gt
    
    Args:
        pred_traj_path (str): Path to the predicted trajectory file.
        gt_traj_path (str): Path to the ground truth trajectory file.
        
    Returns:
        dict: Metrics values accessed under their abbreviations.
    """
    # Load the predicted and ground truth trajectories
    pred_traj = file_interface.read_kitti_poses_file(pred_traj_path)
    gt_traj = file_interface.read_kitti_poses_file(gt_traj_path)
    delta = 1 # Calculate error per frame
    tran_rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta, Unit.frames, all_pairs=False)
    tran_rpe_metric.process_data((gt_traj, pred_traj))

    tran_rpe_stat = tran_rpe_metric.get_statistic(metrics.StatisticsType.rmse)

    rot_rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_part, delta, Unit.frames, all_pairs=False)
    rot_rpe_metric.process_data((gt_traj, pred_traj))
    rot_rpe_stat = rot_rpe_metric.get_statistic(metrics.StatisticsType.rmse)

    return {"RTE (m)": np.round(tran_rpe_stat, 3).item(), "RRE(temp)": np.round(rot_rpe_stat, 3).item()}

def get_metrics(pred_traj_path, gt_traj_path):
    """Provide both absolute and relative metrics in one dict"""
    apes = APE(pred_traj_path, gt_traj_path, plot_traj=False)
    rpes = RPE(pred_traj_path, gt_traj_path)

    apes.update(rpes)
    return apes
