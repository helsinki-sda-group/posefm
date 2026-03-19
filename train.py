# Copyright (c) 2026 Dominik Kuczkowski, SDA group, University of Helsinki
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

from utils.utils import create_data_loader, sample_pose_uniform, load_state_dicts, TerminalLogger
from models.PoseFMNet import PoseFMNet

import time
import tomllib
import os
import argparse

import torch
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from tqdm import tqdm


class Trainer:
    def __init__(self, config_path: str):
        """Trainer class to handle PoseFM training.
        Args:
            config_path (str): Path to the configuration toml file.
        """
        
        # Load config
        self.config_path = config_path
        with open(config_path, 'rb') as file:
            self.config = tomllib.load(file)
        
        self.device = torch.device(self.config["trainer"]["device"])
        self.cond = self.config["trainer"]["conditioning"] # This should be a valid string
        if self.cond not in ["flow", "img"]:
            self.cond = False

        # Instantiate the model
        model_config = self.config["model"]
        self.model = PoseFMNet(**model_config["params"] if "params" in model_config else {}).to(self.device).float()
        # Init optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config["trainer"]["lr"]) 
        # Load state dicts if provided
        self.model, self.optim = load_state_dicts(model=self.model, optim=self.optim, config=model_config, device=self.device)
        # Instantiate an affine path object
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Create data loader, assuming the config contains section data.train
        self.dataloaders = create_data_loader(**self.config["data"]["train"]) # The config has to match the function signature
        self.train_loader = self.dataloaders["train"]
        self.val_loader = self.dataloaders["val"]
        
        # Initialize empty list for loggers
        self.loggers = []

    def register_logger(self, logger):
        self.loggers.append(logger)

    def sample_path_and_calc_loss(self, batch):
        # Fetch gt data
        x_1 = batch["relpose"].to(self.device, non_blocking=True)
        # Sample random translation and rotation
        x_0 = sample_pose_uniform(x_1.shape[0], self.device, std=1)

        if self.cond == 'flow':
            x_1_cond = torch.cat([batch['flow'], batch['intrinsic']], dim=1).to(self.device, non_blocking=True)
        elif self.cond == 'img':
            x_1_cond = [batch['img1'].to(self.device, non_blocking=True), 
                        batch['img2'].to(self.device, non_blocking=True),
                        batch['intrinsic'].to(self.device, non_blocking=True)]
        # Sample time
        t = torch.rand(x_1.shape[0], dtype=torch.float32).to(self.device) 

        # Sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        model_input = [path_sample.x_t, path_sample.t.unsqueeze(-1)]
        if self.cond:
            model_input.append(x_1_cond)
        
        pred = self.model(*model_input)

        loss = ((pred - path_sample.dx_t)**2).sum(dim=-1)
        loss = loss.mean()

        return loss

    def train(self):
        """Train the model."""
        epochs = self.config["trainer"]["epochs"]
        best_val_loss = 1e6
        for ep in range(epochs):
            self.model.train()
            losses = []
            num_batches = len(self.train_loader)
            for i, batch in enumerate(pbar := tqdm(self.train_loader, desc=f"Epoch {ep+1}/{epochs}")):
                self.optim.zero_grad() 
                loss = self.sample_path_and_calc_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optim.step()
                losses.append(loss.cpu().item())
                # Log loss
                pbar.set_postfix(ordered_dict={"current_loss": loss.item()})
                # Per batch logging
                for logger in self.loggers:
                    logger.log({"batch_train_loss": loss.item(), "batch": ep*num_batches+i})

            metrics = {}
            metrics["val_loss"] = self.validate()
            # Add training loss to metrics dict
            metrics["train_loss"] = torch.tensor(losses).mean().item()
            metrics["epoch"] = ep
            for logger in self.loggers:
                logger.log(metrics)
            # Save best model
            if metrics["val_loss"] < best_val_loss:
                self.save_model(checkpoint=True)
                best_val_loss = metrics["val_loss"]
                print(f"Found new best model. Val loss: {best_val_loss}, epoch: {ep+1}")
    
    def validate(self):
        """Validation step. 
        Returns:
            val_loss (float): Average validation loss
        """
        self.model.eval()
        losses = []
        for i, batch in enumerate(pbar := tqdm(self.val_loader, desc="Validation")):
            with torch.no_grad():
                loss = self.sample_path_and_calc_loss(batch)
                losses.append(loss.cpu().item())
                pbar.set_postfix(ordered_dict={"current_loss": loss.item()})
        return torch.tensor(losses).mean().item()
    
    def save_model(self, checkpoint=False):
        """Save model weights."""
        if "save_dir" not in self.config["model"]:
            print("Model save directory is not specified in the config. Defaulting to current dir.")
            model_dir = "."
        else:
            model_dir = self.config["model"]["save_dir"]
            os.makedirs(model_dir, exist_ok=True)

        suffix = '_ckpt' if checkpoint else ''
        model_path = f"{model_dir}/{self.config['model']['type']}_{time.strftime('%m%d_%H%M%S')}{suffix}.pth"
        to_save = {"model_state_dict": self.model.state_dict()}
        if checkpoint:
            to_save["optim_state_dict"] = self.optim.state_dict()
        torch.save(to_save, model_path)
        msg = f"Model {'and optim ' if checkpoint else ''}saved to {model_path}"
        print(msg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the trainer with specified config")
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to the TOML configuration file"
    )
    args = parser.parse_args()

    config_path = args.config
    trainer = Trainer(config_path)
    # Register loggers
    logger = TerminalLogger()
    trainer.register_logger(logger)

    # Start the training process
    trainer.train()
    trainer.save_model()