import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from imageio import get_writer
from flow2depth import generate_image_homogeneous_coordinates, rot_bearing_mul, triangulation

import matplotlib.pyplot as plt


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    if 'png' in imfile:
        img = img[:3]
    return img[None].to(DEVICE)


def viz(img, flo, pred_depth):
    img = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    #img_flo = np.concatenate([img_flo, display_d(pred_depth).flatten(0,1).permute(1,2,0).repeat(1,1,3).numpy()], axis=0)
    #img_flo = np.concatenate([img_flo, colormap(pred_depth, torch_transpose=False).flatten(0,1).permute(1,2,0)], axis=0)
    #d_ = 
    depth = (colormap(pred_depth, torch_transpose=False).reshape((192, 640, 3)) * 255 ).astype(np.uint8)
    img_flo = np.concatenate([img_flo, depth], axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return img_flo
    return img_flo[:, :, [2,1,0]]
    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    
#_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
_DEPTH_COLORMAP = plt.get_cmap('Spectral', 256)  # for plotting
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis

def display_d(depth, ax=None):
    d_max = 10.
    d_min = 0.1
    depth[depth<d_min] = d_min
    depth[depth>d_max] = d_max

    depth = depth / (d_max - d_min)
    return depth * 255.

def traj_loader(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        out_ = np.zeros((len(lines), 4 ,4), dtype=np.float32)
        for i, l in enumerate(lines):
            num = l.split(' ')[1:]
            out_[i, :3, :] = np.array([float(n) for n in num ]).reshape(-1, 4)
            out_[i, 3, 3] = 1
    return out_


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    log = get_writer('/data/scratch/qqbao/DEBUG/flow_colormap_2flow.mp4', fps=45)

    poses_all = np.load(args.path + '../poses.npy')
    y_correct = np.array([[1., 0, 0, 0],
                    [0, -1., 0, 0],
                    [0, 0 , -1., 0],
                    [0, 0 , 0, 1.]], dtype=np.float32)

    fc = np.array([320., 96.])
    cc = np.array([320., 96.])
    height =  192
    width  = 640

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for i, (imfile1, imfile2, imfile3) in enumerate(zip(images[:-6], images[3:-3], images[6:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            image3 = load_image(imfile3)

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)


            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0
            image3 = 2 * (image3 / 255.0) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()
            image3 = image3.contiguous()

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low, flow_up2 = model(image1, image3, iters=20, test_mode=True)

            flow_up = flow_up.cpu()
            flow_up2 = flow_up2.cpu()

            poses = [
                np.matmul(poses_all[i], y_correct),
                np.matmul(poses_all[i+3], y_correct),
                np.matmul(poses_all[i+6], y_correct)]

            poses_ref_in_other = [torch.from_numpy(np.matmul(np.linalg.inv(pose), poses[0])) for pose in poses[1:]]
            rots_ref_in_other = torch.from_numpy(np.stack([pose[0:3, 0:3] for pose in poses_ref_in_other])).unsqueeze(0)
            ts = torch.from_numpy(np.stack([pose[0:3, 3] for pose in poses_ref_in_other])).unsqueeze(0)
            homo =  generate_image_homogeneous_coordinates(fc, cc, width, height).permute(2, 0, 1).unsqueeze(0)
            bearings_ref_in_other = [rot_bearing_mul(rots_ref_in_other[:, k, ...], homo) for k in range(rots_ref_in_other.shape[1])]
            pred_depth, _ = triangulation(bearings_ref_in_other, ts, [flow_up, flow_up2], residual=True)


            log.append_data(viz(image1, flow_up, pred_depth))

    log.close()

def demo2(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    log = get_writer('/data/scratch/qqbao/DEBUG/REALDATA_colormap_2flow_interval16_new.mp4', fps=45)

    poses_all = traj_loader(os.path.join(args.path, 'new_trajectory.txt'))

    height = 480 #720 
    width  = 640
    fc = np.array([904.62, 904.62]).astype(float) 
    cc = np.array([640., 360.]).astype(float)
    fc[0] *= width / 1280.
    cc[0] *= width / 1280.
    fc[1] *= height / 720.
    cc[1] *= height / 720.
    print('FC, cc', self.fc, self.cc)
    #fc = np.array([904.62, 640.]) * width / 1280
    #cc = np.array([904.62, 360.]) * height / 720

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, 'color', '*.png')) + \
                 glob.glob(os.path.join(args.path, 'color', '*.jpg'))
        
        images = sorted(images)
        images = [os.path.join(args.path,'color', 'frame{}.jpg'.format(i)) for i in range(1, 1000) ]
        for i, (imfile1, imfile2, imfile3) in enumerate(zip(images[:-16], images[13:-3], images[16:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            image3 = load_image(imfile3)

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low, flow_up2 = model(image1, image3, iters=20, test_mode=True)

            flow_up = flow_up.cpu()
            flow_up2 = flow_up2.cpu()

            poses = [
                poses_all[i],
                poses_all[i+13],
                poses_all[i+16]]

            poses_ref_in_other = [torch.from_numpy(np.matmul(np.linalg.inv(pose), poses[0])) for pose in poses[1:]]
            rots_ref_in_other = torch.from_numpy(np.stack([pose[0:3, 0:3] for pose in poses_ref_in_other])).unsqueeze(0)
            ts = torch.from_numpy(np.stack([pose[0:3, 3] for pose in poses_ref_in_other])).unsqueeze(0)
            homo =  generate_image_homogeneous_coordinates(fc, cc, width, height).permute(2, 0, 1).unsqueeze(0)
            bearings_ref_in_other = [rot_bearing_mul(rots_ref_in_other[:, k, ...], homo) for k in range(rots_ref_in_other.shape[1])]
            pred_depth, _ = triangulation(bearings_ref_in_other, ts, [flow_up, flow_up2], residual=True)


            log.append_data(viz(image1, flow_up, pred_depth))

    log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
