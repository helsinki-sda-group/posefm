from pathlib import Path

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import DataLoader, random_split
from voloader.utils import AsChannelFirstTensor, Compose, CropCenter, ResizeScaleFlowTensor, RandomCropResizeTensor
from voloader.tartanair import TartanDataset
from voloader.kitti import KITTIOdometryDataset
from voloader.tum import TumDataset


# Model wrapper
class WrappedCondModel(ModelWrapper):
    def __init__(self, cond_batch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_batch = cond_batch
        if hasattr(self.model, "flow") and self.model.sample==0:
            self.model.flow = None
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if len(t.shape) < 2:
            t = t.repeat(x.shape[0], 1)
        v, _ = self.model(x, t, self.cond_batch)
        return v

def convert_vonet_checkpoint(old_checkpoint):
    if "state_dict" in old_checkpoint:
        old_state_dict = old_checkpoint["state_dict"]
    else:
        old_state_dict = old_checkpoint

    new_state_dict = {}
    counter_total = 0
    counter_update = 0
    for old_key, value in old_state_dict.items():

        if old_key.startswith("flowPoseNet."):
            new_key = "poseNet." + old_key[len("flowPoseNet."):]
            counter_update += 1
        elif old_key.startswith("flowNet."):
            new_key = old_key

        else:
            new_key = old_key

        if new_key in new_state_dict:
            raise RuntimeError(
                f"Checkpoint key collision: {new_key}"
            )

        new_state_dict[new_key] = value
        counter_total +=1
    print(f"Total keys: {counter_total}, updated keys: {counter_update}")
    return new_state_dict

def load_state_dicts(model, config, device, optim=None):
    """Load state dicts if path provided in the config"""
    # Load model state if specified in the config
    if "model_path" in config:
        model_path = config["model_path"]
        if Path(model_path).is_file():
            state_dict = torch.load(model_path, map_location=device)
            state_dict["model_state_dict"] = convert_vonet_checkpoint(state_dict["model_state_dict"])
            model.load_state_dict(state_dict["model_state_dict"])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model path {model_path} is not a file or does not exist. \
                  Initializing model without loading state.")
    elif "checkpoint_path" in config:
        # Load model and optim from checkpoint
        checkpoint_path = config["checkpoint_path"]
        if Path(checkpoint_path).is_file():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            msg = ''
            if optim is not None:
                optim.load_state_dict(checkpoint["optim_state_dict"])
                msg = "and optim "
            print(f"Model {msg}state dicts loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint path {checkpoint_path} is not a file or does not exist. \
                  Initializing model without loading state.")
    else:
        print("Didn't find model or checkpoint path in config.")
    
    if "freeze" in config and config["freeze"] == True:
        # Freeze encoder part of the pretrained model
        print("Freezing the frontend")
        for param in model.flowNet.parameters():
            param.requires_grad = False

    if optim is not None:
        return model.to(device), optim
    return model.to(device)

def create_data_loader(dataset_name: str, 
                       data_path: str, 
                       batch_size: int, 
                       val_split: float = 0.0, 
                       combined: bool = True, 
                       num_workers: int = 0,
                       modality: str = "all",
                       seed: int = None,
                       sequences: list = ["00"]):
    """
    Create data loaders for specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for the dataloader.
        val_split (float): Fraction of the dataset to be used for validation. Default is 0.0 (no validation split).
        combined (bool): Parameter of the TartanDataset, whether to combine multiple trajectories into one dataset.
        num_workers (int): Number of worker threads for data loading. Default is 0, which means data loading will be done in the main process.
        modality (str): Type of data to be loaded by the dataset. Available options: all, img, flow. Ground truth pose and intrinsic layer are always loaded.
        seed (int): Random generator seed. Default: None
        sequences (list): KITTI sequences selected for evalutation. By default the method will be run on sequence 00. 
        
    Returns:
        dataloaders (dict): Dictionary containing at least one of train, validation, and test dataloaders.
    """
    
    match dataset_name:
        case "tartantrain":
            pose_std = [ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013]
            
            transform = Compose([CropCenter((448, 640)),
                                 AsChannelFirstTensor(), 
                                 RandomCropResizeTensor(size=(448, 640), scale=(0.16, 1.0), ratio=(1.3, 1.55)),
                                 ResizeScaleFlowTensor((112, 160), 20.0), 
                                 ])
            dataset = TartanDataset(data_path, train=True, combined=combined, transform=transform, 
                                    modality=modality, std=pose_std)
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            if val_split > 0.0:
                # Split dataset into train and test based on val_split
                total_size = len(dataset)
                print(f"Total dataset size: {total_size}")
                val_size = int(total_size * val_split)
                train_size = total_size - val_size
                print(f"Train size: {train_size}, Val size: {val_size}")
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=g)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            else:
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=g)
                val_loader = None
            dataloaders = {
                "train": train_loader,
                "val": val_loader,
                "test": None
                }

        case "tartantest":
            transform = Compose([CropCenter((448, 640)), AsChannelFirstTensor(), ResizeScaleFlowTensor((112, 160), 20.0)])
            dataset = TartanDataset(data_path, train=False, combined=combined, transform=transform, modality=modality)
            dataloaders = {
                "train": None,
                "val": None,
                "test": DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            }

        case "kittitest":
            transform = Compose([CropCenter((448, 640)), AsChannelFirstTensor(), ResizeScaleFlowTensor((112, 160), 20.0)])
            
            if len(sequences) == 0 or not isinstance(sequences, list):
                sequences = ["00"]

            dataset = KITTIOdometryDataset(data_path, sequences=sequences, train=False, combined=False, transform=transform)

            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            dataloaders = {
                "train": None,
                "val": None,
                "test": test_loader
            }
        case "tumtest":
            transform = Compose([CropCenter((448, 640)), AsChannelFirstTensor(), ResizeScaleFlowTensor((112, 160), 20.0)])
            dataset = TumDataset(data_path, transform=transform)
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            dataloaders = {
                "train": None,
                "val": None,
                "test": test_loader
            }
        case _:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return dataloaders


def sample_pose_uniform(batch_size: int, device: torch.device, std: float = 1, torch_g=None, numpy_g=None):
    """Sample random poses uniformly in the space of translations and rotations.
    Args:
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to run the sampling on.
        torch_g: Torch random generator. Default: None
        numpy_g: Numpy random generator. Default: None
    Returns:
        poses (torch.Tensor): Sampled poses as a tensor of shape (batch_size, 6).
    """

    translation = std * torch.randn((batch_size, 3), generator=torch_g, dtype=torch.float32, device=device)  # Random translation with 0 mean and std
    rotation = torch.tensor(R.random(batch_size, rng=numpy_g).as_rotvec(), dtype=torch.float32, device=device)  # Random rotation
    poses = torch.cat([translation, rotation], dim=1).to(device)
    return poses


def sample_model(model, 
                 cond_batch, 
                 device, 
                 batch_size,
                 num_t=10, 
                 step_size=None,
                 torch_g=None,
                 numpy_g=None
                 ):
    """Sample from the model using the provided conditioning batch.
    Args:
        model (torch.nn.Module): The model to sample from.
        cond_batch (torch.Tensor): Conditioning batch, e.g., optical flow.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size
        num_t (int): Number of time steps to sample. Default: 10.
        step_size (float): Step size for the ODE solver. If given than num_t is ignored Default: None.
        torch_g: Torch random generator. Default: None
        numpy_g: Numpy random generator. Default: None
    Returns:
        final_poses (torch.Tensor): Sampled final poses for the input batch."""


    x_init = sample_pose_uniform(batch_size, device, std=1, torch_g=torch_g, numpy_g=numpy_g)  # sample initial poses randomly
    T = torch.linspace(0,1,num_t, dtype=torch.float32)  # sample times

    wrapped_vf = WrappedCondModel(cond_batch, model).to(device)
    solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
    T = T.to(device=device)
    final_poses = solver.sample(time_grid=T, x_init=x_init, method='midpoint', 
                                step_size=step_size, return_intermediates=False)  # sample from the model
    return final_poses


class TerminalLogger:
    def log(self, metrics, step=None):
        if step is not None:
            print(f"Step {step}: {metrics}")
        else:
            print(f"{metrics}")
