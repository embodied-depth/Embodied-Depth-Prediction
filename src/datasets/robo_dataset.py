#import sys
#from pathlib import Path
#base_dir = Path(__file__).absolute().parent.parent
#sys.path.append(str(base_dir))

import os
import skimage.transform
import numpy as np
import torch
import PIL.Image as pil
from scipy.signal import medfilt2d

from src.datasets.base_dataset import MultiMonoDataset

def cam1_to_cam0(_world_camera1, _world_camera0):
    T_camera1_world = np.linalg.pinv(_world_camera1)
    T_camera1_camera0 = np.matmul(T_camera1_world, _world_camera0)
    return T_camera1_camera0

def traj_loader(path, start_idx):
    with open(path, 'r') as f:
        lines = f.readlines()
        out_ = np.zeros((len(lines), 4 ,4), dtype=np.float32)
        for i, l in enumerate(lines):
            num = l.split(' ')[1:]
            out_[i, :3, :] = np.array([float(n) for n in num ]).reshape(-1, 4)
            out_[i, 3, 3] = 1

    return out_

class MRealRAWDataset(MultiMonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(MRealRAWDataset, self).__init__(*args, **kwargs)

        self.use_odom = False
        # focal length 904.62 mm
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # Takes from this issue https://github.com/facebookresearch/habitat-sim/issues/80
        self.K = np.array([[904.62 , 0,     640., 0],
                           [0,      904.62 , 360., 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 192) # (1024, 320) #(640, 192)
        self.K[0] = self.K[0]   * self.full_res_shape[0] / 1280.
        self.K[1] = self.K[1]   * self.full_res_shape[1] / 720.

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.exts = {}
        self.color_memo = {}
        self.dep_memo = {}
        self.depth_noise = False

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index):
        f_str = "color/frame{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index):
        colors = []
        for i in range(self.read_len):
            idx = frame_index + i - self.read_center_id
            color = self.loader(self.get_image_path(folder, idx))
            colors.append(color)

        return colors

    def get_depth(self, folder, frame_index):
        depths = []
        for i in range(self.read_len):
            idx = frame_index + i - self.read_center_id
            f_str = "depth/dframe{}.npy".format(frame_index + i - self.read_center_id)
            depth_path = os.path.join(
                   self.data_path, folder, f_str)
            depth = np.load(depth_path) / 1000.
            depth = skimage.transform.resize(
                    depth, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
            depths.append(depth.astype(np.float32))

        return depths

    def get_pose(self, folder, frame_index):
        if folder not in self.exts.keys():
            if self.use_odom:
                pose_dir = os.path.join(self.data_path, folder, 'odom.npy')
            else:
                pose_dir = os.path.join(self.data_path, folder, 'poses.npy')
            self.exts[folder] = np.load(pose_dir).astype(np.float32)

        warp_pose = []
        video_pose = []
        for f in self.frame_idxs:
            warp_pose.append(cam1_to_cam0(
                self.exts[folder][frame_index + f ],
                 self.exts[folder][frame_index ],
            ))

        for i in self.input_T_dim:
            buf = []
            for f in range(-self.num_past_frame_input * self.video_interval, 1, self.video_interval):
                buf.append(cam1_to_cam0(
                    self.exts[folder][frame_index + i + f ],
                    self.exts[folder][frame_index + i],
                ))
            video_pose.append(buf)

        poses = [ self.exts[folder][i + frame_index] for i in self.frame_idxs]
        if len(poses) > 3:
            poses_ref_in_other = np.stack([(np.matmul(np.linalg.inv(pose), poses[0])) for pose in poses[3:]]) # (#ofRefFrames, 4, 4)
        else:
            #print("warning")
            poses_ref_in_other = np.stack(poses)

        return warp_pose, None, video_pose, poses_ref_in_other


  
