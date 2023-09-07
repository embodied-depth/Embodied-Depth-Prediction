from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def depth_to_disp(_depth, min_depth, max_depth):
    mask = _depth != 0

    depth = torch.clamp(_depth, min_depth, max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp_ = (scaled_disp - min_disp) / (max_disp - min_disp) * mask
    return disp_

class MultiMonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_pose=False,
                 input_depth=False,
                 num_past_frame_input=1,
                 video_interval=1,
                 input_T_dim=None):
        super(MultiMonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.num_past_frame_input = num_past_frame_input
        self.video_interval = video_interval
        self.input_T_dim = input_T_dim

        self.read_center_id = -min([min(frame_idxs), -video_interval * num_past_frame_input]) \
                        if len(self.input_T_dim) > 1 else \
                       -min(frame_idxs) + video_interval * num_past_frame_input
        self.read_len = self.read_center_id + max(self.frame_idxs) + 1

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.input_depth = input_depth

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.load_pose  = load_pose

    def preprocess(self, _colors, _depths, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        outputs = {}
        outputs['depth_gt'] = torch.tensor(_depths[self.read_center_id]).unsqueeze(0)

        for s in range(self.num_scales):
            colors = [self.to_tensor(self.resize[s](color)) for color in _colors]
            depths = torch.tensor(np.stack(_depths)).unsqueeze(1)
            #depths = [self.to_tensor(self.resize[s](depth)) for depth in _depths]
            for idx in self.frame_idxs:
                # GT color and depth for warpping and eval
                outputs[('color', idx, s)] = colors[self.read_center_id + idx]
                outputs[('color_aug', idx, s)] = torch.stack([color_aug(colors[self.read_center_id + idx])])

            for idx in self.input_T_dim:
                # Color as input
                read_idxs = slice(self.read_center_id - self.num_past_frame_input * self.video_interval,
                                 self.read_center_id + 1,
                                  self.video_interval)
                outputs[('color_aug', idx, s)] = torch.stack([color_aug(c) for c in colors[read_idxs]])
                outputs[('color', 0)] = torch.stack([c for c in colors[read_idxs]])[:-1]

                # Depth as input
                if self.input_depth:
                    depth_input = torch.stack([d for d in depths[read_idxs]]) 
                    if self.depth_noise:
                        depth_input = depth_input + 0.05 * depth_input / 2. * torch.randn_like(depth_input)
                        depth_input[depth_input > 4.] = 0
                        outputs[('depth_input', idx, s)] = depth_to_disp(depth_input, 0.1, 40)
                    else:
                        depth_input[depth_input > 4.] = 0
                        outputs[('depth_input', idx, s)] = depth_to_disp(depth_input, 0.1, 40)

        return outputs

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images, with shape (B, 1, 3, H, W)
            ("color_aug", <frame_id>, <scale>)      for augmented colour images with shape (B, num_past_frame_input, 3, H, W),
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= 1 // (2 ** scale)
            K[1, :] *= 1 // (2 ** scale)
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # Read Color and Depth
        do_color_aug = self.is_train and random.random() > 0.5
        #do_color_aug = False
        color_aug = transforms.ColorJitter(
                        self.brightness, 
                        self.contrast, 
                        self.saturation, 
                        self.hue) if do_color_aug else (lambda x: x)
        ### do_flip = self.is_train and random.random() > 0.5
        do_flip = False

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1].split('_')[0])

        colors = self.get_color(folder, frame_index)
        depths = self.get_depth(folder, frame_index)

        inputs.update(self.preprocess(colors, depths, color_aug))

        # Read pose for warpping and input
        warp_pose, warp_pose_inv, v_pose, poses = self.get_pose(folder, frame_index)
        for i, f in enumerate(self.frame_idxs):
            inputs[("cam_T_cam", 0, f)] = warp_pose[i]
            if warp_pose_inv is not None:
                inputs[("cam_T_cam_inv", 0, f)] = warp_pose_inv[i]

        inputs['v_poses'] = torch.tensor(np.stack(v_pose))

        # For optical flow input
        inputs['flowpose'] = poses

        return inputs

    def get_color(self, folder, frame_index):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index):
        raise NotImplementedError

    def get_pose(self, folder, frame_index):
        raise NotImplementedError

