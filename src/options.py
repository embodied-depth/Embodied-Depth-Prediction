from __future__ import absolute_import, division, print_function

import os
import argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepth options")

        # Embodied env setting
        self.parser.add_argument("--scene_id",
                            type=int,
                            default=4)
        self.parser.add_argument("--train_scene_id",
                            type=int,
                            help='id for offline data training',
                            default=4)
        self.parser.add_argument("--test_scene_id",
                            type=int,
                            default=4)
        self.parser.add_argument("--cfg_path",
                        type=str,
                        default='configs/default_cfg.yaml')

        # Visalization setting
        self.parser.add_argument("--video_prefix",
                                        type=str,
                                        default='')
        self.parser.add_argument("--clipmax",
                                type=int,
                                default=80 )
        
        self.parser.add_argument("--max_frame_interval",
                                   type=int,
                                   default=6)
        self.parser.add_argument("--min_frame_interval",
                                   type=int,
                                   default=1)

        self.parser.add_argument("--update_times",
                                   type=int,
                                   default=1,
                                   help='number of Data Training for one iteration data collection')

       # Dataset setting 
        self.parser.add_argument("--png",
                              help="if set, trains from raw KITTI png files (instead of jpgs)",
                              action="store_true")
        self.parser.add_argument("--video_interval",
                            type=int,
                            default=1)

        self.parser.add_argument("--input_T_dim",
                            nargs="+",
                            type=int,
                            help="time index used in consistency loss",
                            default=[0])

        
        self.parser.add_argument("--num_past_frame_input",
                                type=int,
                                help="frames to load",
                                default=0)
        
        self.parser.add_argument("--num_workers",
                              type=int,
                              help="number of dataloader workers",
                              default=12)

        # Ablation
        self.parser.add_argument("--model_type",
                            choices=['monodepth', 'scdepth', 'manydepth', 'flow2depth', 'leres'])
        self.parser.add_argument("--agent_type",
                            choices=['random', 'chamfer', 'mae', 'frontier', 'mix', 'edge', 'forward'],
                            default='random')
        self.parser.add_argument("--reset_halfway",
                                help="Whether only test in the offline mode",
                                default=False,
                                action="store_true")

        self.parser.add_argument("--full_trainset",
                                help="Whether use full trainset directly in the offline mode",
                                default=False,
                                action="store_true")

        self.parser.add_argument("--test",
                                help="Whether only test in the offline mode",
                                default=False,
                                action="store_true")

        self.parser.add_argument("--save_pred_npy",
                                help="Whether only test in the offline mode",
                                default=False,
                                action="store_true")

    def parse(self):
        subparser = self.parser.add_subparsers()

        from src.models import flow2depthOptions 
        #from src.models import monodepthOptions 
        #from src.models import manydepthOptions 
        #from src.models import scdepthOptions 
        #from src.models import leresOptions 

        flow2depthOptions(subparser.add_parser('flow2depth'))
        #monodepthOptions(subparser.add_parser('monodepth'))
        #manydepthOptions(subparser.add_parser('manydepth'))
        #scdepthOptions(subparser.add_parser('scdepth'))
        #leresOptions(subparser.add_parser('leres'))

        self.options = self.parser.parse_args()
        return self.options
