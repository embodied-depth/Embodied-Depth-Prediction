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
#from wrapper.my_utils import cam1_to_cam0

def cam1_to_cam0(T_world_camera1, T_world_camera0):
    T_camera1_world = np.linalg.pinv(T_world_camera1)
    T_camera1_camera0 = np.matmul(T_camera1_world, T_world_camera0)
    return T_camera1_camera0


def traj_loader(path, start_idx):
    with open(path, 'r') as f:
        lines = f.readlines()
        out_ = np.zeros((len(lines), 4 ,4))
        for i, l in enumerate(lines):
            num = l.split(' ')[1:]
            out_[i, :3, :] = np.array([float(n) for n in num ]).reshape(-1, 4)
            out_[i, 3, 3] = 1

    return out_

class EmRealDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        height
        width
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                data,
                frame_idxs,
                 height,
                 width,
                 num_scales,
                 is_train=True,
                 load_pose=False,
                 input_depth=False):
        super(EmRealDataset, self).__init__()

        images = data['color']  
        depths = data['depth'] 
        poses  = data['Ext'] 

        self.images = images
        self.depths = depths
        self.poses = poses
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs
        self.frame_interval = max(frame_idxs)

        self.to_tensor = transforms.ToTensor()
        self.input_depth = input_depth
        self.is_train = is_train

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

        self.load_depth = True
        self.load_pose  = load_pose

        self.K = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.inputs_memo = {}
        for scale in range(self.num_scales):
            K = self.K.copy()

            #K[0, :] *= self.width // (2 ** scale)
            #K[1, :] *= self.height // (2 ** scale)
            K[0, :] *= 1 // (2 ** scale)
            K[1, :] *= 1 // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            self.inputs_memo[("K", scale)] = torch.from_numpy(K)
            self.inputs_memo[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        self.length = None
        self.length = self.__len__()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f)).unsqueeze(0)
            elif "color_pass" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


    def __len__(self):
        if self.length == None: 
            self.length = 0
            self.len_mark = []
            for img_list, depth_list, pose_list in zip(self.images, self.depths, self.poses):
                assert img_list.shape[0] == depth_list.shape[0]
                assert img_list.shape[0] == pose_list.shape[0]
                self.length += img_list.shape[0] - 2 * self.frame_interval
                self.len_mark.append(self.length)

        return self.length

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
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
        group_index = 0
        for i in range(len(self.len_mark) - 1):
            if self.len_mark[i+1] > index:
                frame_index = index - self.len_mark[i]
            else:
                group_index += 1
                if group_index == len(self.len_mark) - 1:
                    frame_index = index - self.len_mark[-1]

        frame_index += self.frame_interval # skip index=0 and -1, cuz where no 
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        #do_flip = self.is_train and random.random() > 0.5
        do_flip = False

        frame_idxs = self.frame_idxs #[0, -self.frame_interval, self.frame_interval]
        for i in frame_idxs:
            color = Image.fromarray(self.images[group_index][frame_index + i]).convert('RGB')
            if do_flip:
                color = color.transpose(Image.FLIP_LEFT_RIGHT)
            inputs[("color", i, -1)] = color

        for scale in range(self.num_scales):
            inputs[("K", scale)] = self.inputs_memo[("K", scale)].clone()
            inputs[("inv_K", scale)] = self.inputs_memo[("inv_K", scale)].clone()


        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        depth_gt = self.depths[group_index][frame_index]
        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        self.preprocess(inputs, color_aug)
        for i in frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_pose:
            for i in frame_idxs[1:]:
                inputs[("cam_T_cam", 0, i)] = torch.from_numpy(cam1_to_cam0(self.poses[frame_index + i],
                                                        self.poses[frame_index]))

            poses = [ self.poses[frame_index + f]  for f in self.frame_idxs]
            poses_ref_in_other = np.stack([(np.matmul(np.linalg.inv(pose), poses[0])) for pose in poses[3:]])
            inputs['flowpose'] = poses_ref_in_other


        return inputs

