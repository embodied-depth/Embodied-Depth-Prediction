import os
import sys
import json
from PIL import Image  # using pillow-simd for increased speed
from pathlib import Path
from collections import deque

import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
from tensorboardX import SummaryWriter


from src.models.layers import *
from src.models.utils import *
from src.datasets  import MHabitatRAWDataset, MRealRAWDataset, EmDataset 
from src.my_utils import *

from src.models.base_trainer import BaseTrainer


class MyTrainer(BaseTrainer):
    def __init__(self, opt):
        from src.models.flow2depth.model import DepthModel

        self.model = DepthModel(opt)

        super().__init__(opt)

        self.num_scales = len(self.opt.scales)

        param_to_train = self.model.parameters_to_train #if self.opt.model_type != 'ours' else \
        #[
        #        {'params': self.model.parameters_to_train},
        #        {'params': self.model.models['flow2depth'].parameters(), 'lr': 1e-7}
        #    ]
        self.model_optimizer = optim.Adam(
                                    param_to_train, 
                                    self.opt.learning_rate, 
                                    weight_decay=0,
                                    eps=self.opt.epsilon)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        self.save_opts()

    def reset_model(self):
        if self.opt.model_type == 'monodepth':
            from src.models.monodepth.model import DepthModel
        elif self.opt.model_type == 'ours':
            from src.models.flow2depth.model import DepthModel
        self.model = DepthModel(self.opt)
        self.model_optimizer = optim.Adam(self.model.parameters_to_train, self.opt.learning_rate, weight_decay=0, eps=self.opt.epsilon)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

    def train_offline(self):
        """Run the entire training pipeline
        """
        img_ext = '.png' if self.opt.png else '.jpg'
        train_filenames = readlines(self.fpath.format("train{}".format(self.opt.train_scene_id)))
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.model.frames_to_load_train, 
            num_scales=self.num_scales, is_train=True, img_ext=img_ext, 
            load_pose=self.opt.pose_model_type=='gt',
            input_depth=self.opt.depth_encoder,
            num_past_frame_input=0,
            video_interval=self.opt.video_interval,
            input_T_dim=self.opt.input_T_dim 
            )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.writers = {}
        for mode in ["train", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.epoch = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self._run_epoch()
            print("Epoch {}, training step {}, lr = {}".format(self.epoch, self.step, self.model_lr_scheduler.get_last_lr()))

    def _run_epoch(self):
        """Run a single epoch of training and validation
        """

        self.model_lr_scheduler.step()
        print("Training")
        self.model.train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.model.forward(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            if self.opt.clip > 0 :
                torch.nn.utils.clip_grad_norm_(self.model.parameters_to_train, self.opt.clip)
            self.model_optimizer.step()
            #self.model_lr_scheduler.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 3000
            late_phase = self.step % 3000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    depth_pred = outputs[("depth", 0, 0)]
                    depth_gt = inputs["depth_gt"]
                    mask = depth_gt > 0
                    depth_gt = depth_gt[mask]
                    depth_pred = depth_pred[mask]

                    depth_errors = compute_depth_errors(depth_gt, depth_pred)
                    for metric, val in depth_errors.items():
                        losses[metric] = np.array(depth_errors[metric].cpu().data)

                self.log("train", inputs, outputs, losses)

            #if self.step % 2048 == 10:
            #    self.model.eval()
            #    self.test(self.writers['test'], self.step)
            #    self.model.train()

            self.step += 1

        if self.epoch % 2 == 0:
            self.set_eval()
            self.val(self.writers['train'], self.step)
            if self.epoch % 4 == 0:
                self.save_model(self.log_path, 'e_{}'.format(self.epoch ))
            self.test(self.writers['test'], self.step)
            self.set_train()


    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def process_batch(self, inputs):
        '''
        return:
            outputs
            losses 
        '''
        return self.model.forward(inputs)


    def save_model(self, path, name):
        '''
        Save all model ckpts and optimizer in the following path
         `[tensorboard_dir]/[self.log_path]/models/weights_{epoch}_val_{valacc}/`
        '''
        self.model.save_model(path, name)
        #save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        #torch.save(self.model_optimizer.state_dict(), save_path)


    def load_model(self, path):
        self.model.load_model(path)


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if type(mode) == str:
            writer = self.writers[mode]
        else:
            writer = mode

        if type(outputs) == list:
            sigmas = self.get_uncertainty(outputs)
            outputs = outputs[0]
        else:
            sigmas = None

        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image(
                "gt_disp/{}".format(j),
                colorize(depth_to_disp(inputs["depth_gt"][j].cpu().detach(), 
                                self.opt.min_depth,
                                self.opt.max_depth)), 
                 self.step)
            if ('depth_input', 0, 0) in inputs.keys():
                writer.add_image(
                    "input_disp/{}".format(j),
                    colorize(inputs[('depth_input', 0, 0)][j, -1].cpu().detach()), 
                    self.step)

            if sigmas is not None:
                writer.add_image(
                    "depth_Nsigma/{}".format(j),
                    normalize_image(sigmas[j]), self.step)

            if not(("color", self.opt.frame_ids[1], 0) in inputs.keys()):
                break
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j,:3].cpu().data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j,:3].cpu().data, self.step)
                    if self.opt.geo_consist and frame_id!= 0:
                        writer.add_image(
                            "computed_depth_{}/{}".format(frame_id, j),
                            colorize(normalize_image(outputs[("warp_depth", frame_id, 0)][j]).cpu().detach()), self.step)
                        writer.add_image(
                            "warp_depth_{}/{}".format(frame_id, j),
                            colorize(normalize_image(outputs[("depth", frame_id, 0)][j].cpu().detach())), self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    colorize(outputs[("disp", s)][j].cpu().detach()), self.step)

                if ("disp_old", s) in outputs.keys():
                    writer.add_image(
                    "disp_old_{}/{}".format(s, j),
                    colorize(outputs[("disp_old", s)][j, -1].cpu().detach()), self.step)
                

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                #elif not self.opt.disable_automasking:
                #    writer.add_image(
                #        "automask_{}/{}".format(s, j),
                #        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def special_train_offline(self):
        """Run the entire training pipeline
        """
        img_ext = '.png' if self.opt.png else '.jpg'
        train_filenames = readlines(self.fpath.format("train{}".format(self.opt.train_scene_id)))
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.model.frames_to_load_train, 
            num_scales=1, is_train=True, img_ext=img_ext, 
            load_pose=self.opt.pose_model_type=='gt',
            input_depth=self.opt.depth_encoder,
            num_past_frame_input=0,
            video_interval=self.opt.video_interval,
            input_T_dim=self.opt.input_T_dim 
            )
        
        whole_length = len(train_dataset)

        self.writers = {}
        for mode in ["train", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.epoch = 0
        self.start_time = time.time()

        start_len = 6000
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch % 1 == 0:
                current_train_set = Subset(train_dataset, [i for i in range(start_len)])
                self.writers['train'].add_scalar('Explor/data_num', start_len, self.step)
                self.train_loader = DataLoader(
                    current_train_set , self.opt.batch_size, True,
                    num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
                if self.epoch >= 0:
                    start_len += 3000
                    start_len = min(start_len, whole_length)

            self._run_epoch()
            print("Epoch {}, training step {}, len_data={}, lr = {}".format(self.epoch, self.step, len(current_train_set), self.model_lr_scheduler.get_last_lr()))



if __name__ == "__main__":
    from monodepth2.options import MonodepthOptions

    options = MonodepthOptions()
    opts = options.parse()

    trainer = ModelTrainer(opts)
    trainer.train_offline()
    #trainer.test()
