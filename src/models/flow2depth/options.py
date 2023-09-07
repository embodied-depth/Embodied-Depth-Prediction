# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


def flow2depthOptions(parser):
     # PATHS
     parser.add_argument("--data_path",
                              type=str,
                              help="path to the training data",
                              default=os.path.join(file_dir, "kitti_data"))
     parser.add_argument("--log_dir",
                              type=str,
                              help="log directory",
                              default=os.path.join(os.path.expanduser("~"), "tmp"))

     # TRAINING options
     parser.add_argument("--model_name",
                              type=str,
                              help="the name of the folder to save the model in",
                              default="mdp")
     parser.add_argument("--split",
                              type=str,
                              help="which training split to use",
                              #choices=["eigen_zhou", "eigen_full", "odom", "benchmark", 'habitat','habitat2', 'habitat_pose', 'minikitti', 'real'],
                              default="eigen_zhou")
     parser.add_argument("--num_layers",
                              type=int,
                              help="number of resnet layers",
                              default=18,
                              choices=[18, 34, 50, 101, 152])
     parser.add_argument("--dataset",
                              type=str,
                              help="dataset to train on",
                              default="kitti",
                              choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", 'habitat', 'minikitti', 'real'])
     parser.add_argument("--height",
                              type=int,
                              help="input image height",
                              default=192)
     parser.add_argument("--width",
                              type=int,
                              help="input image width",
                              default=640)
     parser.add_argument("--disparity_smoothness",
                              type=float,
                              help="disparity smoothness weight",
                              default=1e-3)

     parser.add_argument("--disp_loss_weight",
                         help="disparity loss with gt depth",
                              type=float,
                              default=0.)
     parser.add_argument("--geo_consist",
                              help="geometry consistency loss weight",
                              type=float,
                              default=0.)

     parser.add_argument("--scales",
                              nargs="+",
                              type=int,
                              help="scales used in the loss",
                              default=[0, 1, 2, 3])
     parser.add_argument("--min_depth",
                              type=float,
                              help="minimum depth",
                              default=0.1)
     parser.add_argument("--max_depth",
                              type=float,
                              help="maximum depth",
                              default=100.0)
     parser.add_argument("--use_stereo",
                              help="if set, uses stereo pair for training",
                              action="store_true")
     parser.add_argument("--frame_ids",
                              nargs="+",
                              type=int,
                              help="frames to load",
                              default=[0, -1, 1])
     parser.add_argument("--warp_ids",
                              nargs="+",
                              type=int,
                              help="frames to load",
                              default=[0, -1, 1])

     # OPTIMIZATION options
     parser.add_argument("--batch_size",
                              type=int,
                              help="batch size",
                              default=12)
     parser.add_argument("--learning_rate",
                              type=float,
                              help="learning rate",
                              default=1e-4)

     parser.add_argument('--wdecay', type=float, default=0.0001)
     parser.add_argument('--epsilon', type=float, default=1e-8)
     parser.add_argument('--clip', type=float, default=-1.0)
     

     parser.add_argument('--num_steps', type=int, default=10000)
     parser.add_argument("--num_epochs",
                              type=int,
                              help="number of epochs",
                              default=20)
     parser.add_argument("--scheduler_step_size",
                              type=int,
                              help="step size of the scheduler",
                              default=15)

     # ABLATION options
     parser.add_argument("--warmup_orinet",
                         action="store_true",
                         default=False)

     parser.add_argument("--transform_in_refine",
                         action="store_true",
                         default=False)

     parser.add_argument("--flow2depth",
                         action="store_true",
                         default=False)

     parser.add_argument("--rgbdcat",
                         action="store_true",
                         default=False)

     parser.add_argument("--use_opencv_flow",
                              help="if set, and flow2depth is set meanwhile, use opencv func to generate optical flows",
                              action="store_true")

     ## RAFT options
     parser.add_argument("--flow_model_path",
                              default='')

     parser.add_argument("--small",
                         action="store_true",
                              default=False)
                              
     parser.add_argument("--device",
                              default='cuda')

     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
     parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')


     ### Depth decoder ablation
     parser.add_argument("--stack_video",
                         action="store_true",
                         default=False)

     parser.add_argument("--multi_pose",
                         action="store_true",
                         default=False)

     parser.add_argument("--pose_dist_info",
                         action="store_true",
                         default=False)

     parser.add_argument("--pose_rot",
                         action="store_true",
                         default=False)

     parser.add_argument("--pose_decay",
                         action="store_true",
                         default=False)

     parser.add_argument("--depth_encoder",
                         action="store_true",
                         default=False)


     parser.add_argument("--bayes",
                              action="store_true")
     parser.add_argument("--input_depth",
                              help="if set, input noisy depth map",
                              action="store_true")
     parser.add_argument("--input_pose",
                              help="if set, input noisy depth map",
                              action="store_true")



     parser.add_argument("--v1_multiscale",
                              help="if set, uses monodepth v1 multiscale",
                              action="store_true")
     parser.add_argument("--avg_reprojection",
                              help="if set, uses average reprojection loss",
                              action="store_true")
     parser.add_argument("--disable_automasking",
                              help="if set, doesn't do auto-masking",
                              action="store_true")
     parser.add_argument("--predictive_mask",
                              help="if set, uses a predictive masking scheme as in Zhou et al",
                              action="store_true")
     parser.add_argument("--no_ssim",
                              help="if set, disables ssim in the loss",
                              action="store_true")
     parser.add_argument("--weights_init",
                              type=str,
                              help="pretrained or scratch",
                              default="pretrained",
                              choices=["pretrained", "scratch"])
     parser.add_argument("--pose_model_input",
                              type=str,
                              help="how many images the pose network gets",
                              default="pairs",
                              choices=["pairs", "all"])
     parser.add_argument("--pose_model_type",
                              type=str,
                              help="normal or shared",
                              default="separate_resnet",
                              choices=["posecnn", "separate_resnet", "shared", 'gt'])

     # SYSTEM options
     parser.add_argument("--no_cuda",
                              help="if set disables CUDA",
                              action="store_true")

     # LOADING options
     parser.add_argument("--load_weights_folder",
                              type=str,
                              help="name of model to load")
     parser.add_argument("--models_to_load",
                              nargs="+",
                              type=str,
                              help="models to load",
                              default=["encoder", "depth", "pose_encoder", "pose"])

     # LOGGING options
     parser.add_argument("--log_frequency",
                              type=int,
                              help="number of batches between each tensorboard log",
                              default=250)
     parser.add_argument("--save_frequency",
                              type=int,
                              help="number of epochs between each save",
                              default=12)

     # EVALUATION options
     parser.add_argument("--eval_stereo",
                              help="if set evaluates in stereo mode",
                              action="store_true")
     parser.add_argument("--eval_mono",
                              help="if set evaluates in mono mode",
                              action="store_true")
     parser.add_argument("--disable_median_scaling",
                              help="if set disables median scaling in evaluation",
                              action="store_true")
     parser.add_argument("--pred_depth_scale_factor",
                              help="if set multiplies predictions by this number",
                              type=float,
                              default=1)
     parser.add_argument("--ext_disp_to_eval",
                              type=str,
                              help="optional path to a .npy disparities file to evaluate")
     parser.add_argument("--eval_split",
                              type=str,
                              default="eigen",
                              choices=[
                                   "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "habitat", 'habitat_pose'],
                              help="which split to run eval on")
     parser.add_argument("--save_pred_disps",
                              help="if set saves predicted disparities",
                              action="store_true")
     parser.add_argument("--no_eval",
                              help="if set disables evaluation",
                              action="store_true")
     parser.add_argument("--eval_eigen_to_benchmark",
                              help="if set assume we are loading eigen results from npy but "
                                   "we want to evaluate using the new benchmark.",
                              action="store_true")
     parser.add_argument("--eval_out_dir",
                              help="if set will output the disparities to this folder",
                              type=str)
     parser.add_argument("--post_process",
                              help="if set will perform the flipping post processing "
                                   "from the original monodepth paper",
                              action="store_true")

     return parser 

