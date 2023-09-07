import os
import sys
import json
import glob
from PIL import Image  # using pillow-simd for increased speed
from pathlib import Path
from collections import deque

import time
from tqdm import tqdm
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


class BaseTrainer:
    def __init__(self, options):
        '''
        Compulsory-specified attribute:
            self.model
                must have two attributes 
                `frames_to_load_train` decide the number of frames from Dataset input 
                `frames_to_load_test` decide the number of frames when the model online preprocesses the data from environment
        Attention: Model should be defined before super().__init__()
        '''

        self.opt = options
        log_path = os.path.join(self.opt.log_dir, self.opt.model_type + '_' + self.opt.agent_type +  '_' + self.opt.model_name )
        if not self.opt.test:
            times = len(glob.glob( os.path.join(log_path, 'log*'), recursive=False))
            self.log_path = os.path.join(log_path,  'log' + str(times))
        else:
            self.log_path = log_path
            

        ## Val and test dataset
        self.fpath = os.path.join(str(Path(__file__).absolute().parent.parent.parent), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(self.fpath.format("val{}".format(self.opt.test_scene_id)))
        test_filenames = readlines(self.fpath.format("test{}".format(self.opt.test_scene_id)))

        datasets_dict = {
                         "habitat": MHabitatRAWDataset,
                         'real': MRealRAWDataset
                         }
        self.dataset = datasets_dict[self.opt.dataset]

        num_scales = 4 #if self.opt.model_type == 'manydepth' else 1

        img_ext = '.png' if self.opt.png else '.jpg'
        val_dataset = self.dataset(
            data_path=self.opt.data_path,
            filenames=val_filenames,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.model.frames_to_load_train,
            num_scales=num_scales,
            is_train=False,
            img_ext=img_ext,
            input_depth=False,
            num_past_frame_input=0,
            video_interval=self.opt.video_interval,
            input_T_dim=self.opt.input_T_dim )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=True)

        test_dataset = self.dataset(
            data_path=self.opt.data_path,
            filenames=test_filenames,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.model.frames_to_load_train,
            num_scales=num_scales,
            is_train=False,
            img_ext=img_ext,
            input_depth=False,
            num_past_frame_input=0,
            video_interval=self.opt.video_interval,
            input_T_dim=self.opt.input_T_dim )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=12, pin_memory=True, drop_last=True)

        ## Log variable
        self.step = 0
        self.epoch = 0
        self.val_history_best = 0

    def train_offline(self):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def process_batch(self, inputs):
        '''
        return:
            outputs
            losses 
        '''
        pass

    def log(self, logger, inputs, outputs, loss):
        pass

    def save_model(self, path, name):
        '''
        Save all model ckpts and optimizer in the following path
         `[tensorboard_dir]/[self.log_path]/models/weights_{epoch}_val_{valacc}/`
        '''
        #save_folder = os.path.join(self.log_path, "models", "weights_{}".format(cur_step))
        #save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        #torch.save(self.model_optimizer.state_dict(), save_path)
        pass

    def load_model(self, path):
        pass



    # Above need to specify; below has been written

    def train_one_step(self, data, logger=None, cur_step=0, dataweight=None):
        self.set_train()

        dataset = EmDataset(
                            data, 
                            self.model.frames_to_load_train,
                            self.opt.height,
                            self.opt.width,
                            self.num_scales,
                            is_train=True,
                            load_pose=(self.opt.pose_model_type == 'gt')
                            )

        dataloader = DataLoader(
            dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        loss_buf = []
        for i in range(self.opt.update_times):
            for inputs in dataloader:
                outputs, losses = self.process_batch(inputs)

                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()
                loss_buf.append(losses["loss"].cpu().data)

                if self.step % 1024 == 0:
                    logger.add_scalar('loss/loss', losses["loss"].cpu().data, self.step)
                    logger.add_scalar('lr_rate', self.model_lr_scheduler.get_last_lr()[0], self.step)

                early_phase = self.step % 400 == 0 and self.step < 5000
                late_phase = self.step % 4000 == 0

                if (early_phase or late_phase) and (len(outputs.keys()) > 2):
                    self.log(logger, inputs, outputs, losses)

                self.step += 1
                if self.step % 2048 == 0:
                    self.set_eval()
                    self.val(logger, self.step)
                    #self.test(logger, self.step)
                    self.set_train()

                if (self.step ) % 8000 == 0:
                    self.save_model(self.log_path, 'latest')

        self.model_lr_scheduler.step()
        self.set_eval()
        metrics = self.test(logger, self.step)
        

        acc_ = metrics['test/a1']()
        self.set_eval()
        print("Action Step {}, training step {}, lr = {}".format(cur_step, self.step, self.model_lr_scheduler.get_last_lr()))


        del dataset
        del dataloader

        # For manydepth 
        if hasattr(self.opt, "freeze_teacher_epoch" ):
            self.epoch += 1
            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()


        return acc_

    def cal_acc(self, b):
        with torch.no_grad():
            outputs, losses = self.process_batch(b)

            depth_pred  = outputs[('depth', 0, 0)].cpu()
            depth_gt = b['depth_gt'].cpu()

            mask = depth_gt > 0
            #scale_coef  = 1 if self.opt.dataset == 'real' else torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
            scale_coef  =  torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])

            depth_pred *= scale_coef  #torch.median(depth_gt) / torch.median(depth_pred)
            #depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
            depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)
            depth_pred_ = depth_pred.clone()

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

            depth_errors_ = compute_depth_errors(depth_gt, depth_pred)
        return depth_pred_, depth_errors_, losses

    def test(self, logger, cur_step, final=False, load_best=True):

        d_videos = []
        loss_buf = AvgMeter()
        depth_metric_names = [
            "test/abs_rel", "test/sq_rel", "test/rms", "test/log_rms", "test/a1", "test/a2", "test/a3"] + \
            ["test/>5m_abs_rel", "test/>5m_sq_rel", "test/>5m_rms", "test/>5m_log_rms", "test/>5m_a1", "test/>5m_a2", "test/>5m_a3"]
        metrics = {k: AvgMeter() for k in depth_metric_names}

        if final:
            if load_best:
                self.load_best_model()

        with torch.no_grad():
            for b in tqdm(self.test_loader):
                #videos.append(b[('color', 0, 0)].cpu())
                depth_pred, depth_errors, losses = self.cal_acc(b)
                loss_buf += losses["loss"].cpu().data
                d_videos.append(depth_pred)
                for k in depth_errors.keys():
                    metric_name = k.replace('de/', 'test/').replace('da/', 'test/')
                    metrics[metric_name] += depth_errors[k].cpu().data


        d_videos = torch.cat(d_videos).unsqueeze(0)

        if final:
            if self.opt.save_pred_npy:
                np.save(os.path.join(self.log_path, 
                                     self.opt.model_name.split('/')[0] + '_'+self.opt.model_type + '_' + self.opt.agent_type + '_scene{}.npy'.format(self.opt.test_scene_id)), 
                        d_videos.numpy() )
            print('Final Metircs')
            main_table_path = os.path.join(Path(self.log_path).parent, 'table.csv')
            write_metrics = (not os.path.exists(main_table_path))
            with open(main_table_path, 'a') as f:
                if write_metrics:
                    for k in metrics.keys():
                        f.write(k + ',')
                    f.write('\n')

                for k, v in metrics.items():
                    print("{}: {} \t".format(k, v()))
                    f.write( '{},'.format(v()) )
                f.write('\n')

            # calculate mean and std so far
            summary_path = os.path.join(Path(self.log_path).parent, 'summary.csv')
            tb = np.loadtxt(main_table_path, delimiter=",", dtype=str)
            data = tb[1:, :-1].astype(float)
            name = tb[0]
            mean = data.mean(0)
            std = data.std(0)
            with open(summary_path, 'w') as f:
                for k in name:
                    f.write(k + ',')
                f.write('\n')
                for k in mean:
                    f.write('{:.03f},'.format(k))
                f.write('\n')
                for k in std:
                    f.write('{:.03f},'.format(k))
                f.write('\n')

        else:
            logger.add_scalar("Eval_loss", loss_buf(), cur_step)
            if cur_step % 7 == 0:
                logger.add_video('Eval/depth', depth_to_rgb(d_videos.numpy(), self.opt.max_depth), fps=60, global_step=cur_step)
            if self.opt.save_pred_npy:
                np.save(os.path.join(self.log_path, 
                                     '/our_pred_depth_scene{}_e{}.npy'.format(self.opt.test_scene_id, self.epoch)), 
                        d_videos.numpy() )
            for k, v in metrics.items():
                logger.add_scalar("{}".format(k), 
                                    v(),
                                    cur_step
                                    )

            return metrics

    def val(self, logger, cur_step):
        loss_buf = AvgMeter()
        depth_metric_names = [
            "val/abs_rel", "val/sq_rel", "val/rms", "val/log_rms", "val/a1", "val/a2", "val/a3"] + \
            ["val/>5m_abs_rel", "val/>5m_sq_rel", "val/>5m_rms", "val/>5m_log_rms", "val/>5m_a1", "val/>5m_a2", "val/>5m_a3"]
        metrics = {k: AvgMeter() for k in depth_metric_names}

        with torch.no_grad():
            for b in self.val_loader:
                depth_pred, depth_errors, losses = self.cal_acc(b)
                for k in depth_errors.keys():
                    metric_name = k.replace('de/', 'val/').replace('da/', 'val/')
                    metrics[metric_name] += depth_errors[k].cpu().data
                    loss_buf += losses["loss"].cpu().data

            logger.add_scalar("val_loss", loss_buf(), cur_step)
            for k, v in metrics.items():
                logger.add_scalar("{}".format(k), 
                                    v(),
                                    cur_step
                                    )

            if (self.val_history_best < metrics['val/a1']()):
                self.val_history_best = metrics['val/a1']()
                self.save_model(self.log_path, 'e_{}_val_{}'.format(cur_step, self.val_history_best ))
                self.test(logger, cur_step, final=False)
    
    def load_best_model(self):
        weights_names = glob.glob(os.path.join(self.log_path, 'models', 'weights*val*'), recursive=False)
        print('Choose from {} ...'.format(weights_names))
        best = 0
        best_name = None
        for weight_name in weights_names:
            acc = float(weight_name.split('val_')[-1])
            if best < acc:
                best = acc
                best_name = weight_name

        print('Load {} ...'.format(best_name))
        self.load_model(best_name)

        for n in weights_names:
            if n != best_name:
                for root, dirs, files in os.walk(n, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
        
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def log_trajectory(self, env, data, logger, cur_step):
        
        gt_poses = [  d['pos'] for d in data]

        INTERVAL = self.opt.frame_ids[-1]
        vis_gt_poses = gt_poses[0 ::INTERVAL]

        img = vis_map(env, vis_gt_poses, None)

        logger.add_image('Train/Trajectory', img, cur_step)


