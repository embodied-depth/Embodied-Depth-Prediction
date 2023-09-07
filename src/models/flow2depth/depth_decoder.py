# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from src.models.layers import *


def dist_T(t):
    I = torch.eye(3).repeat(t.shape[0], t.shape[1], 1, 1).to(t.device)
    dist = torch.sqrt(
            (t[:, :, :3, 3].abs()).sum(dim=(-1))  + \
            2. / 3. * ( I - t[:,:, :3, :3])[:,:, (0,1,2), (0,1,2)].sum(dim=-1)
    )
    dist[dist.isnan()] = 0
    return dist

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, bayes=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

'''
V5: Tailor to the Multi-input (including Single) Resnet encoders, use pose_dist as decay coef for the embeddings
Receive multi-frame images (may including depth) in every layer with shape (B, T, c, h, w)
'''
class DepthDecoderv5(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, 
                    bayes=False, frame_num=1, input_pose=False, multi_pose=False, pose_dist_info=False, pose_rot=False, pose_decay=False):
        super(DepthDecoderv5, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.input_pose = input_pose
        self.multi_pose = multi_pose
        self.pose_rot = pose_rot
        self.pose_dist_info = pose_dist_info
        self.pose_decay = pose_decay
        self.pose_dim = 6 if pose_rot else 2
        if self.pose_dist_info:
            self.pose_dim += 1 

        self.decayT = 0.5

        self.num_ch_enc = num_ch_enc 
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        if self.input_pose:
            if self.multi_pose:
                for i in range(len(self.num_ch_enc)):
                    self.num_ch_enc[i] += self.pose_dim * frame_num
            else:
                self.num_ch_enc[-1] += self.pose_dim * frame_num

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, input_poses=None):
        self.outputs = {}

        B, T, _, _, _ = input_features[-1].shape
        # pose_dist_decay 
        if (input_poses is not None) and self.pose_decay:
            pose_dist = dist_T(input_poses).reshape(B, T, 1, 1, 1)
            pose_dist_decay = torch.exp(-pose_dist / self.decayT)
            for i in range(len(input_features)):
                x = input_features[i]
                B, T, C, H, W = x.shape
                input_features[i] = (x * pose_dist_decay).reshape(B, -1, H, W)
        else:
            for i in range(len(input_features)):
                x = input_features[i]
                input_features[i] = x.flatten(1, 2)

        x = input_features[-1]
        # add pose info
        if self.input_pose:
            pose_info = input_poses[:, :, (0,2), 3].reshape(B, -1, 1, 1)
            if self.pose_rot:
                roat = R.from_matrix(input_poses.flatten(0,1)[:, :3, :3].cpu().numpy())  # (B,T,4,4) -> (B,4)
                quat = roat.as_quat()
                quat = torch.from_numpy(quat).reshape(B, -1, 1, 1).to(x.device).to(x.dtype)
                pose_info = torch.cat([pose_info, quat], dim=1)

            x = torch.cat([x, pose_info.repeat((1, 1,x.shape[-2],x.shape[-1]))], dim=1)

        # decode
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # Add pose info in every layer
            if i > 0 and self.input_pose and self.multi_pose:
                x = torch.cat([x, pose_info.repeat((1, 1,x.shape[-2],x.shape[-1]))], dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


'''
T: Tailor to the Multi-input (including Single) Resnet encoders
Delete Pose

Receive multi-frame images (may including depth) in every layer with shape (B, T, c, h, w)
'''
class DepthDecoderT(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, 
                    bayes=False, frame_num=1, input_pose=False, multi_pose=False, pose_dist_info=False, pose_rot=False, pose_decay=False):
        super(DepthDecoderT, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.input_pose = input_pose
        self.multi_pose = multi_pose
        self.pose_rot = pose_rot
        self.pose_dist_info = pose_dist_info
        self.pose_decay = pose_decay
        self.pose_dim = 6 if pose_rot else 2
        if self.pose_dist_info:
            self.pose_dim += 1 

        self.decayT = 0.5

        self.num_ch_enc = num_ch_enc // frame_num
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, bayes=bayes)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, input_poses=None):
        self.outputs = {}

        B, T, _, _, _ = input_features[-1].shape
        # pose_dist_decay 
        for i in range(len(input_features)):
            x = input_features[i]
            input_features[i] = x.flatten(0, 1)

        x = input_features[-1]

        # decode
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # Add pose info in every layer
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        _, _, H, W = self.outputs[("disp", i)].shape
        self.outputs[("disp", i)] = self.outputs[("disp", i)].reshape(B, T, 1, H, W)
        return self.outputs
