#import sys
#sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Grayscale

from src.models.flow2depth.RAFT.core.raft import RAFT
from src.models.flow2depth.RAFT.core.utils import flow_viz
from src.models.flow2depth.RAFT.core.utils.utils import InputPadder
import multiprocessing

#from core.raft import RAFT
#from core.utils import flow_viz
#from core.utils.utils import InputPadder

from torch import nn
import torch.nn.functional as F

class FlowNet(nn.Module):
    def __init__(self, args):
        super(FlowNet, self).__init__()
        self.args = args

        raft = torch.nn.DataParallel(RAFT(args))
        raft.load_state_dict(torch.load(args.flow_model_path))
        self.flownet = raft.module
        self.flownet = self.flownet.to(args.device)
        self.flownet.eval()

    def forward(self, img0, img_refs):
        with torch.no_grad():
            flow_ups = []
            for img_ref in img_refs:
                _, flow_up = self.flownet(img0, img_ref, iters=20, test_mode=True)
                flow_ups.append(flow_up)

        return torch.stack(flow_ups, dim=1)

def get_flow(imgs):
	out_ = cv2.calcOpticalFlowFarneback(imgs[0], imgs[1], None, 0.5, 10, 80, 3, 7, 1.5, 0)
	return torch.tensor(out_).permute(2, 0, 1)


class FlowBaseline(nn.Module):
    def __init__(self, args):
        super(FlowBaseline, self).__init__()
        self.args = args
        self.grayscale = Grayscale(num_output_channels=1)

    def forward(self, _img0, _img_refs, iters=20, test_mode=True):                                                                                  
        flow_ups = []                                                                                                                               
        img0 = (self.grayscale(_img0.cpu()).squeeze().numpy() * 255).astype(np.uint8)                                                               
        img_refs = (self.grayscale(_img_refs.cpu()).squeeze().numpy() * 255).astype(np.uint8)                                                      
        if len(_img0) > 1:
            slices = [(img0[i], img_refs[i]) for i in range(len(img0))]
            # Create a multiprocessing pool
            pool = multiprocessing.Pool(processes=len(img0))
            # Apply the process_slice function to each slice in parallel
            results = pool.map(get_flow, slices)
            # Close the pool to free up resources
            pool.close()
            # Wait for all processes to finish
            pool.join()
            flow_ups = results
        else:
            flow_ups = [get_flow((img0, img_refs))]
			
        #for i0, img_ref in zip(img0, img_refs):                                                                                                     
        #    flow_up = cv2.calcOpticalFlowFarneback(i0, img_ref, None, 0.5, 10, 80, 10, 7, 1.5, 0)
        #    flow_ups.append(torch.tensor(flow_up).to(_img0.device).permute(2, 0, 1))
                                     
        return None, torch.stack(flow_ups, dim=0).to(_img0.device)


class Flow2Depth(nn.Module):
    def __init__(self, args,):
        super(Flow2Depth, self).__init__()
        self.args = args

        if args.use_opencv_flow:
            self.flownet = FlowBaseline(args)
        else:
            raft = torch.nn.DataParallel(RAFT(args))
            raft.load_state_dict(torch.load(args.flow_model_path))
            self.flownet = raft.module
            self.flownet = self.flownet.to(args.device)
            self.flownet.eval()

        #self.height = 480 #720 
        #self.width  = 640
        self.height = args.height
        self.width  = args.width
        if args.dataset == 'real':
            #self.fc = np.array([904.62, 640.]).astype(float) * self.width / 1280
            #self.cc = np.array([904.62, 360.]).astype(float)* self.height / 720
            self.fc = np.array([904.62, 904.62]).astype(float) 
            self.cc = np.array([640., 360.]).astype(float)
            self.fc[0] *= self.width / 1280.
            self.cc[0] *= self.width / 1280.
            self.fc[1] *= self.height / 720.
            self.cc[1] *= self.height / 720.
            print('FC, cc', self.fc, self.cc)

        elif args.dataset == 'habitat':
            self.fc = np.array([320., 96.])
            self.cc = np.array([320., 96.])
        else:
            raise NotImplementedError('No support for {} dataset. Need to know its Camera Intrinsics'.format(self.opt.dataset))

        self.homo = generate_image_homogeneous_coordinates(self.fc, self.cc, self.width, self.height).permute(2, 0, 1).unsqueeze(0)

    def forward(self, img0, img_refs, poses):
        with torch.no_grad():

            flow_ups = []
            for img_ref in img_refs:
                _, flow_up = self.flownet(img0, img_ref, iters=20, test_mode=True)
                flow_ups.append(flow_up)

            batch_size = img0.shape[0]
            device = img0.device

            #poses_ref_in_other = [torch.from_numpy(np.matmul(np.linalg.inv(pose), poses[0])).to(device) for pose in poses[1:]]
            #rots_ref_in_other = torch.from_numpy(np.stack([pose[0:3, 0:3] for pose in poses])).to(device)
            #ts = torch.from_numpy(np.stack([pose[0:3, 3] for pose in poses])).to(device)
            rots = poses[:,:,  :3, :3]
            ts = poses[:, :, :3, 3]
            homo = self.homo.repeat(batch_size, 1, 1, 1).to(device)
            bearings_ref_in_other = [rot_bearing_mul(rots[:, k, ...], homo) for k in range(rots.shape[1])]

            pred_depth, _ = self.triangulation(bearings_ref_in_other, ts, flow_ups, residual=True)

        return depth_to_disp(pred_depth, min_depth=self.args.min_depth, max_depth=self.args.max_depth)
        #return pred_depth

    def triangulation(self, bearings_ref_in_other, t_ref_in_other, flows, residual=False):
        rs, ss = self.pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=False)
        # get output = (z, residual, hessian)
        outputs = [ls_2view(*r_s) for r_s in zip(rs, ss)]
        # weighted sum of z with weight hessian
        
        hessian = sum([output[2] for output in outputs])
        pred_depths = sum([output[0] * output[2] for output in outputs]) / (hessian + 1e-12)

        if residual:
            # hessian*(z* - z)^2 + residual
            error = torch.sqrt(
                sum([output[2] * (pred_depths - output[0]) ** 2 + output[1] for output in outputs]).clamp_min(0))
            sqrt_hessian = torch.sqrt(hessian)
            return pred_depths, (error, sqrt_hessian)
        else:
            return pred_depths

    def flow2bearing(self, flow, fc, cc, normalize=True):
        assert len(flow.shape) == 4
        height, width = flow.shape[2:4]
        device = flow.device
        xx, yy = np.meshgrid(range(width), range(height))
        pixel = torch.zeros_like(flow)
        match = [flow[:, 0, ...] + torch.from_numpy(xx).to(device), flow[:, 1] + torch.from_numpy(yy).to(device)]
        pixel[:, 0] = (match[0] - cc[0]) / fc[0]
        pixel[:, 1] = (match[1] - cc[1]) / fc[1]
        pixel = torch.cat((pixel, torch.ones_like(pixel[:, 0:1])), dim=1)

        if normalize:
            pixel = F.normalize(pixel)
        return pixel

    def pre_triangulation(self, bearings_ref_in_other, t_ref_in_other, flows, concat=True):
        device = flows[0].device
        fc = torch.from_numpy(self.fc).to(device)
        cc = torch.from_numpy(self.cc).to(device)

        #fc = torch.from_numpy(np.array([320., 96.])).to(device) #/ resize
        #cc = torch.from_numpy(np.array([320., 96.])).to(device) #/ resize
        #fc = torch.from_numpy(np.array([904.62, 640.])).to(device) / 2.#/ resize
        #cc = torch.from_numpy(np.array([904.62, 360.])).to(device) * 480. / 720. #/ resize

        bearings_other = [self.flow2bearing(flow, fc, cc, normalize=True) for flow in flows]
        ss = [torch.cross(bearings_other[k], bearings_ref_in_other[k], dim=1) for k in
                range(len(bearings_other))]
        rs = [torch.cross(bearings_other[k], t_ref_in_other[:, k, :, None, None].expand_as(bearings_other[k]), dim=1)
                for k in range(len(bearings_other))]

        if concat:
            s = torch.cat(ss, dim=1)
            r = torch.cat(rs, dim=1)
            return r, s
        else:
            return rs, ss

def depth_to_disp(_depth, min_depth, max_depth):
    #mask1 = ( (_depth == 0).mean((-1, -2), dtype=torch.float) < 0.5).reshape(_depth.shape[0], 1, 1, 1).repeat(1, *_depth.shape[1:])
    #mask2 = _depth != 0
    #mask = mask1 * mask2
    
    mask = _depth != 0
    depth = torch.clamp(_depth, min_depth, max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp_ = (scaled_disp - min_disp) / (max_disp - min_disp)* mask
    return disp_

def generate_image_homogeneous_coordinates(fc, cc, image_width, image_height):
    homogeneous = np.zeros((image_height, image_width, 3))
    homogeneous[:, :, 2] = 1

    xx, yy = np.meshgrid([i for i in range(0, image_width)], [i for i in range(0, image_height)])
    homogeneous[:, :, 0] = (xx - cc[0]) / fc[0]
    homogeneous[:, :, 1] = (yy - cc[1]) / fc[1]

    return torch.from_numpy(homogeneous.astype(np.float32))

def rot_bearing_mul(rot, bearing):
    # rot: B x 3 x 3, bearing: B x 3 x H x W
    product = torch.bmm(rot, bearing.view(bearing.shape[0], 3, -1))
    return product.view(bearing.shape)

def ls_2view(r, s):
    hessian = (s * s).sum(dim=1, keepdims=True)
    z = -(s * r).sum(dim=1, keepdims=True) / (hessian + 1e-30)
    e = (r * r).sum(dim=1, keepdims=True) - hessian * (z ** 2)

    invalid_mask = (z <= 0.1)
    invalid_mask |= (z >= 40)
    # invalid_mask |= (e > 0.015 ** 2)
    z[invalid_mask] = 0
    e[invalid_mask] = 0
    hessian[invalid_mask] = 0
    return z, e, hessian


def triangulation(bearings_ref_in_other, t_ref_in_other, flows, residual=False):
    rs, ss = pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=False)
    # get output = (z, residual, hessian)
    outputs = [ls_2view(*r_s) for r_s in zip(rs, ss)]
    # weighted sum of z with weight hessian
    
    hessian = sum([output[2] for output in outputs])
    pred_depths = sum([output[0] * output[2] for output in outputs]) / (hessian + 1e-12)

    if residual:
        # hessian*(z* - z)^2 + residual
        error = torch.sqrt(
            sum([output[2] * (pred_depths - output[0]) ** 2 + output[1] for output in outputs]).clamp_min(0))
        sqrt_hessian = torch.sqrt(hessian)
        return pred_depths, (error, sqrt_hessian)
    else:
        return pred_depths

def flow2bearing(flow, fc, cc, normalize=True):
    assert len(flow.shape) == 4
    height, width = flow.shape[2:4]
    device = flow.device
    xx, yy = np.meshgrid(range(width), range(height))
    pixel = torch.zeros_like(flow)
    match = [flow[:, 0, ...] + torch.from_numpy(xx).to(device), flow[:, 1] + torch.from_numpy(yy).to(device)]
    pixel[:, 0] = (match[0] - cc[0]) / fc[0]
    pixel[:, 1] = (match[1] - cc[1]) / fc[1]
    pixel = torch.cat((pixel, torch.ones_like(pixel[:, 0:1])), dim=1)

    if normalize:
        pixel = F.normalize(pixel)
    return pixel

def pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=True):
    device = flows[0].device

    fc = torch.from_numpy(np.array([320., 96.])).to(device) #/ resize
    cc = torch.from_numpy(np.array([320., 96.])).to(device) #/ resize
    #fc = torch.from_numpy(np.array([904.62, 640.])).to(device) / 2.#/ resize
    #cc = torch.from_numpy(np.array([904.62, 360.])).to(device) * 480. / 720. #/ resize

    bearings_other = [flow2bearing(flow, fc, cc, normalize=True) for flow in flows]
    ss = [torch.cross(bearings_other[k], bearings_ref_in_other[k], dim=1) for k in
            range(len(bearings_other))]
    rs = [torch.cross(bearings_other[k], t_ref_in_other[:, k, :, None, None].expand_as(bearings_other[k]), dim=1)
            for k in range(len(bearings_other))]

    if concat:
        s = torch.cat(ss, dim=1)
        r = torch.cat(rs, dim=1)
        return r, s
    else:
        return rs, ss
