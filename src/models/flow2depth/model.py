import os
import json
from PIL import Image  # using pillow-simd for increased speed
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.my_utils import *
from src.models.utils import *
from src.models.layers import *

from src.models.flow2depth import RGBDEncoder, RGBDEncoderv2, DepthDecoderv5, DepthDecoderT #, UpdateLayerv2
from src.models.flow2depth.RAFT.flow2depth import Flow2Depth

def cam1_to_cam0(_world_camera1, _world_camera0):
    T_camera1_world = np.linalg.pinv(_world_camera1)
    T_camera1_camera0 = np.matmul(T_camera1_world, _world_camera0)
    return T_camera1_camera0

def depth_to_disp(_depth, min_depth, max_depth):
    mask = _depth != 0

    depth = torch.clamp(_depth, min_depth, max_depth)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp_ = (scaled_disp - min_disp) / (max_disp - min_disp) * mask
    return disp_

class DepthModel:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = self.opt.pose_model_type != 'gt'
        self.input_pose = self.opt.input_pose

        self.frames_to_load_test = [0]
        self.frames_to_load_train = self.opt.frame_ids.copy()
        for idx in range(-self.opt.num_past_frame_input * self.opt.video_interval, 1, self.opt.video_interval):
            if idx not in self.frames_to_load_test:
                self.frames_to_load_test.append(idx)
                self.frames_to_load_train.append(idx)

        print(self.frames_to_load_train)
        print(self.frames_to_load_test)
        # TODO: clarify this GUY!!
        if self.opt.rgbdcat:
            self.models["encoder"] = RGBDEncoder(
        			self.opt.num_layers,
        			self.opt.weights_init == "pretrained", 
        			num_input_images=1,  #self.opt.num_past_frame_input+1 ,
        			#num_input_images=self.opt.num_past_frame_input+1 ,
        			input_depth=self.opt.depth_encoder) 
        else:
            self.models["encoder"] = RGBDEncoderv2(
                    self.opt.num_layers,
        			self.opt.weights_init == "pretrained", 
        			num_input_images=1,  #self.opt.num_past_frame_input+1 ,
        			#num_input_images=self.opt.num_past_frame_input+1 ,
        			input_depth=self.opt.depth_encoder) 

        self.models["depth"] = DepthDecoderv5(
                self.models["encoder"].num_ch_enc, 
                self.opt.scales, 
                bayes=self.opt.bayes, 
                frame_num=self.opt.num_past_frame_input + 1 ,
                input_pose=self.opt.input_pose,
                multi_pose=self.opt.multi_pose,
                pose_dist_info=self.opt.pose_dist_info,
                pose_rot=self.opt.pose_rot)


        for k in self.models.keys():
            self.models[k].to(self.device)
            if k != 'refine':
                self.parameters_to_train += list(self.models[k].parameters())

        if self.opt.flow2depth:
            self.models["flow2depth"] = Flow2Depth(self.opt)
            self.models["flow2depth"].to(self.device)
            if self.opt.flow2depth_train:
                self.parameters_to_train += list(self.models["flow2depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                    input_depth=self.opt.input_depth)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
            else:
                raise NotImplementedError


        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)


        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w) #if self.opt.dataset != 'habitat' else HabBackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w) #if self.opt.dataset != 'habitat' else  HabProject3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

    def forward(self, inputs):
        outputs, losses = self._process_batch(inputs)
        return outputs, losses

    def _preprocess(self, obs, ):
        '''
        Input: 
            obs: Dict{'color_sensor': np.ndarray, 'depth_sensor': np.ndarray} 
        Output:
            img_: torch.Tensor (B,  3, H, W)
            depth_: numpy.ndarray  (B,  1, H, W)
        '''
        correct = torch.tensor(np.array([[1., 0, 0, 0],
                          [0, -1., 0, 0],
                          [0, 0 , -1., 0],
                          [0, 0 , 0, 1.]], dtype=np.float32))
        correct_f = lambda x: torch.matmul(x, correct)

        if type(obs) in [list, deque] :
            inputs = {}
            series = [self._preprocess(o) for o in obs]

            minus_len = -min(self.frames_to_load_test)
            data_len = len(obs) 
            for i in self.frames_to_load_test:
                inputs[('color', i, 0)] = torch.stack([obs['rgb'] for obs in series[minus_len+i: data_len+i]]).to(self.device)
                inputs[('flowpose', i)] = torch.stack(
                                                        [
                                                           cam1_to_cam0(
                                                                        correct_f(series[idx +  i]['pose']),
                                                                        correct_f(series[idx     ]['pose'])) 
                                                            for idx in range(minus_len+i, data_len+i)
                                                        ]
                                                    ).to(self.device)

            inputs['gt_depth'] = np.stack([obs['gt_depth'] for obs in series[minus_len: data_len]])
             
        else:
            to_tensor = transforms.ToTensor()
            img_ = to_tensor(Image.fromarray(obs['color_sensor']).convert('RGB')).unsqueeze(0).to(torch.float32)
            depth_ = np.expand_dims(obs['depth_sensor'], 0)
            pose_ = torch.tensor(obs['Ext'])

            inputs = {
                'rgb': img_,
                'gt_depth': depth_,
                'pose': pose_
            }

        return inputs

    def pred_depth(self, obs):
        #input_color, gt_depth = self._preprocess(obs)
        inputs = self._preprocess(obs)
        gt_depth = inputs['gt_depth']
        with torch.no_grad():

            input_rgb = inputs[('color', 0, 0)]

            if self.opt.depth_encoder:
                if self.opt.flow2depth:
                    img0 = inputs["color", 0, 0].clone().flatten(0,1)
                    img_refs = [inputs["color", i, 0].clone().flatten(0,1) for i in self.frames_to_load_test[1:]]
                    poses = torch.stack([inputs[('flowpose',i)] for i in self.frames_to_load_test[1:]], dim=1)

                    input_depth = self.models['flow2depth'](img0, img_refs, poses).unsqueeze(1)
                    inputs[('depth_input', 0, 0)]  = input_depth
                else:
                    input_depth = inputs[('depth_input', 0, 0)] 
            else:
                input_depth = None

            features = self.models["encoder"](input_rgb, input_depth) 
            output = self.models["depth"](features, inputs["v_poses"].squeeze()) if self.input_pose else self.models["depth"](features)

            pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if self.opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            # Disp -> depth
            #pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            #mask = gt_depth > 0 
            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                pred_depth *= ratio


            MIN_DEPTH = self.opt.min_depth
            MAX_DEPTH = self.opt.max_depth
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        return pred_depth.squeeze()

    def pred_pose(self, obs):
        input_color, gt_depth = self._preprocess(obs)
        with torch.no_grad():
        
            all_color_aug = torch.cat([input_color[i:i+1] for i in range(input_color.shape[0])], 1)

            features = [self.models['pose_encoder'](all_color_aug)]
            axisangle, translation = self.models['pose'](features)

            pred_poses = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()

        return pred_poses

    def _process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.geo_consist:
            input_rgb = torch.cat([inputs["color_aug", i, 0].squeeze() for i in self.opt.frame_ids]).unsqueeze(1)
            input_depth = torch.cat([inputs[('depth_input', i, 0)] for i in self.opt.frame_ids ]) if self.opt.depth_encoder else None
        else:
            input_rgb = inputs["color_aug", 0, 0]

            if self.opt.depth_encoder:
                if self.opt.flow2depth:
                    img0 = inputs["color", 0, 0].clone().squeeze()
                    img_refs = [inputs["color", i, 0].clone().squeeze() for i in self.frames_to_load_test[1:]]
                    poses = inputs['flowpose']

                    input_depth = self.models['flow2depth'](img0, img_refs, poses).unsqueeze(1)
                    inputs[('depth_input', 0, 0)]  = input_depth
                else:
                    input_depth = inputs[('depth_input', 0, 0)] 
            else:
                input_depth = None

        features = self.models["encoder"](input_rgb, input_depth) 

        outputs = self.models["depth"](features, inputs["v_poses"].squeeze()) if self.input_pose else self.models["depth"](features)

        outputs.update(self._predict_poses(inputs, features))

        self._generate_images_pred(inputs, outputs)
        losses = self._compute_losses(inputs, outputs)

        return outputs, losses

    def _predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.opt.pose_model_type == 'gt':
            for f_i in self.opt.frame_ids[1:]:
                outputs[("cam_T_cam", 0, f_i)] = inputs[("cam_T_cam", 0, f_i)].to(torch.float32)
            return outputs

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            raise NotImplementedError

        return outputs

    def _generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.geo_consist:
                outputs[("disp", scale)] = outputs[("disp", scale)].chunk(len(self.opt.frame_ids), 0)[0]
            else:
                outputs[("disp", scale)] = outputs[("disp", scale)]

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            

            if self.opt.geo_consist:
                depth = depth.chunk(len(self.opt.frame_ids), 0)
            else:
                depth = [depth]

            outputs[("depth", 0, scale)] = depth[0]

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth[0], inputs[("inv_K", source_scale)])
                pix_coords, warp_depth = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[('warp_depth', frame_id, scale)] = warp_depth.unsqueeze(1)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if self.opt.geo_consist:
                    outputs[("depth",frame_id, scale)] = F.grid_sample(
                        depth[i+1],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="zeros")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def _compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def _compute_depth_loss(self, disp, _gt_depth):
        # scale the ground truth
        assert not _gt_depth.isnan().any(), '  '.format(_gt_depth.isnan.sum())
        #mask = (F.interpolate(_gt_depth, disp.shape[-2:], mode='bilinear') > 0 ).detach()

        #gt_disp = 1 / (_gt_depth + 0.01)
        #depth = torch.clamp(F.interpolate(gt_disp, disp.shape[-2:], mode='bilinear'),
        #                    0.,
        #                    100.)
        
        _gt_depth   = _gt_depth.unsqueeze(1)
        mask = torch.logical_and(_gt_depth > 0, disp>0)
        _loss = F.l1_loss(torch.log(disp[mask]), torch.log(_gt_depth[mask]))
        return _loss

    def _compute_depth_consistency(self, outputs):
        loss_ = 0
        for i in self.opt.frame_ids[1:]:
            mask = outputs[("depth", i, 0)] > 0
            diff = (torch.log(outputs[("depth", i, 0)][mask]) - torch.log(outputs[('warp_depth', i, 0)][mask])).abs()
            loss_ += diff.mean()

        return loss_    

    def _compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self._compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self._compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            
            #if ('depth_input', 0, 0) in inputs.keys():
            #    disp_loss = self._compute_depth_loss(disp, inputs[('depth_input', 0, 0)][:, -1])
            #    loss += disp_loss * self.opt.disp_loss_weight / (2 ** scale)
            #    losses["loss/disp_loss{}".format(scale)] = disp_loss

            if scale == 0 and ('depth_input', 0, 0) in inputs.keys():
                gt_disp = depth_to_disp(inputs[('depth_gt')][:, -1], self.opt.min_depth, self.opt.max_depth)
                disp_loss = self._compute_depth_loss(disp, gt_disp)
                loss += disp_loss * self.opt.disp_loss_weight / (2 ** scale)
                losses["loss/disp_loss{}".format(scale)] = disp_loss

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            if ('disp_old', 0) in outputs.keys():
                idloss = F.mse_loss(disp , outputs[('disp_old', 0)][:, -1]) * 0.01
                loss += idloss
                losses['loss/id_loss']= idloss.detach().cpu().data

            if self.opt.geo_consist:
                geo_consist_loss = self._compute_depth_consistency(outputs)
                loss += self.opt.geo_consist * geo_consist_loss
                losses["loss/geo_loss{}".format(scale)] = geo_consist_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses['loss/smooth_loss']= smooth_loss.detach().cpu().data


        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses



    def train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def save_model(self, path, e):
        """Save model weights to disk
        """
        save_folder = os.path.join(path, "models", "weights_{}".format(e))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

    def load_model(self, path=None):
        """Load model(s) from disk
        """
        if path is None:
            self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
            folder = self.opt.load_weights_folder
        else:
            folder = path

        assert os.path.isdir(folder), \
            "Cannot find folder {}".format(folder)
        print("loading model from folder {}".format(folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
