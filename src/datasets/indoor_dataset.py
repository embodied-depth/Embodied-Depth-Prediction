from __future__ import absolute_import, division, print_function
import sys
#from pathlib import Path
#base_dir = Path(__file__).absolute().parent.parent
#sys.path.append(str(base_dir))

import os
import numpy as np

from src.datasets.base_dataset import MultiMonoDataset
from src.my_utils import correct_yz, cam1_to_cam0


class MHabitatRAWDataset(MultiMonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(MHabitatRAWDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # Takes from this issue https://github.com/facebookresearch/habitat-sim/issues/80
        self.K = np.array([[320, 0, 320, 0],
                           [0,  96, 96, 0],
                           [0,   0,  1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        #self.K[0, :] *= self.width
        #self.K[1, :] *= self.height
        self.full_res_shape = (640, 192)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.depth_noise = True
        self.exts = {}
        self.color_memo = {}
        self.dep_memo = {}

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index ):
        f_str = "color/{:08d}{}".format(frame_index, self.img_ext)
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
            f_str = "depth/{:08d}.npy".format(idx)
            depth_path = os.path.join(
                        self.data_path, folder, f_str)
            depth = np.load(depth_path)
            depths.append(depth.astype(np.float32))

        return depths

    def get_pose(self, folder, frame_index):
        '''
        return: 
        warp_pose: relative pose used in the loss computation
        video_pose: relative pose used when inputting video and its corresponding poses
        poses_ref_in_other: relative pose used for deriving depth from optical flows

        '''

        if folder not in self.exts.keys():
            pose_dir = os.path.join(self.data_path, folder, 'poses.npy')
            self.exts[folder] = correct_yz(np.load(pose_dir))

        warp_pose = []
        warp_pose_inv = []
        for f in self.frame_idxs:
            warp_pose.append(cam1_to_cam0(
                self.exts[folder][frame_index + f ],
                 self.exts[folder][frame_index ],
            ))
            warp_pose_inv.append(cam1_to_cam0(
                self.exts[folder][frame_index], 
                self.exts[folder][frame_index+f]
                ))
           


        video_pose = []
        for i in self.input_T_dim:
            buf = []
            for f in range(-self.num_past_frame_input * self.video_interval, 1, self.video_interval):
                buf.append(cam1_to_cam0(
                    self.exts[folder][frame_index + i  ] ,
                    self.exts[folder][frame_index + i + f]
                ))

            video_pose.append(buf)

        poses = [ (self.exts[folder][frame_index + f]) for f in self.frame_idxs]
        if len(poses) > 3:
            poses_ref_in_other = np.stack([(np.matmul(np.linalg.pinv(pose), poses[0])) for pose in poses[3:]]) # (#ofRefFrames, 4, 4)
        else:
            poses_ref_in_other = np.stack(poses)

        return warp_pose, warp_pose_inv, video_pose, poses_ref_in_other



if __name__ == '__main__':
    from utils import readlines
    from torch.utils.data import DataLoader

    data_path = '/Users/mac/Desktop/Habitat/interactivity_output'
    fpath = '/Users/mac/Desktop/Habitat/all_files.txt'
    train_filenames = readlines(fpath)

    train_dataset = IndoorRAWDataset(
            data_path, train_filenames, 544, 720,
            [0, -1, 1], 4, is_train=True, img_ext='.png')
    train_loader = DataLoader(
        train_dataset, 2, True,
        num_workers=0, pin_memory=True, drop_last=True)

    for i in train_loader:
        print(i.keys())
        break
