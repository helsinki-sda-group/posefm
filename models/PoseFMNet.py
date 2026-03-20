# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Modifications Copyright (c) 2026 Dominik Kuczkowski, SDA group, University of Helsinki

import torch
import torch.nn as nn

from .FMNet import FMNet
from .WAFT.waft_a1 import ViTWarpV8

import json
from pathlib import Path
import argparse


class WAFTWrapper(ViTWarpV8):
    def __init__(self, args):
        super().__init__(args)
    def forward(imgs: list):
        out = super().forward(imgs[0], imgs[1])
        return out['flow'][-1]


class PoseFMNet(nn.Module):
    def __init__(self, full=True, cache=False, frontend="PWC", *args, **kwargs):
        super(PoseFMNet, self).__init__()
        if frontend == "PWC":
            from .PWC import PWCDCNet
            self.flowNet = PWCDCNet()
        elif frontend == "WAFT":
            config_path = Path(__file__).parent / "WAFT/config/a1/tar-c-t.json"
            with open(config_path) as f:
                config = json.load(f)
                config.update(**kwargs)
                args = argparse.Namespace()
                args_dict = args.__dict__
                for key, value in config.items():
                    args_dict[key] = value

            self.flowNet = WAFTWrapper(args)
        else:
            raise ValueError(f"Unsupported frontend: {frontend}")
        self.frontend = frontend
        self.poseNet = FMNet(**kwargs)
        self.full = full # Run full pipeline or only pose prediction
        self.cache = cache # User parameter that enables caching
        self.flow = None

    def forward(self, n, t, x):
        if self.full:
            if not self.cache or self.flow is None:
                flow = self.flowNet(x[0:2])
                self.flow = flow
            flow_input = torch.cat( (self.flow, x[2]), dim=1 )  
        else:
            flow_input = x
        # Estimate pose using flow matching
        pose = self.poseNet(n, t, flow_input)
        return pose
