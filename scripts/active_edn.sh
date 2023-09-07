#!/bin/bash
LOGDIR=/path/to/logdir
DATADIR=/path/to/data #offline data path
RAFT_MODEL_PATH=/path/to/RAFT #you may need to download the weights from offical RAFT repo

DEVICE=0
for SCENE in 0 
do
CUDA_VISIBLE_DEVICES=$DEVICE python  embodied_depth.py  \
                    --cfg_path configs/template.yaml \
                    --scene_id $SCENE \
                    --test_scene_id $SCENE \
                    --update_times 1 \
                    --agent_type mix \
                    --num_workers  12 \
                    --num_past_frame_input 1 \
                    --input_T_dim 0  \
                    --video_interval 5 \
                    --png \
                    \
                    \
                    --model_type flow2depth \
                    flow2depth \
                    \
                    \
                    --log_dir $LOGDIR/cat_repeat_base/our \
					--rgbdcat \
                    --model_name _Scene$SCENE \
                    --models_to_load encoder depth \
                    --height 192 \
                    --width 640 \
                    --data_path $DATADIR\
                    --dataset habitat \
                    --split random \
                    --max_depth 40 \
                    --min_depth 0.1 \
                    --batch_size 12 \
                    --frame_ids 0  -4  4 \
                    --scales 0 \
                    --pose_model_type gt \
                    --log_frequency 20 \
                    --learning_rate 1e-4 \
                    --scheduler_step_size 20 \
                    --num_steps 100000 \
                    --depth_encoder \
                     --flow2depth \
                     --flow_model_path $RAFT_MODEL_PATH    \
                     --wdecay 0   &
done
