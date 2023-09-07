## Embodied Depth Prediction
Website: https://embodied-depth.github.io

## Env Setup
### Overview
1. Install [Habitat Simulation](https://github.com/facebookresearch/habitat-sim/tree/v0.2.2)
2. Install dependencies, `pip install -r requirements.txt`
3. [Optional] If you want to use ROS, refer to official [Rospy](http://wiki.ros.org/rospy) document or install it with [conda](https://anaconda.org/conda-forge/ros-rospy). Currently, we have not integrated ROS into conda environment. Potentially, you may try [RoboStack](https://robostack.github.io/index.html).

Here we give a recommended setup in conda environment

### [Recommended] Conda Env Setup
```
conda create -n habitat python=3.8 cmake=3.14.0
conda activate habitat
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
pip install -r requirements.txt
```

## Dataset Setup
### Simulation
Download 3D scenes of [Matterport3D](https://niessner.github.io/Matterport/), [Gibbson](http://gibsonenv.stanford.edu/database/), and name their directories `mp3d` and `gibson` respectively. Potentially, you can download the test scenes in Habitat-sim for a try:
```python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/```

### Real
Download at [Google Drive](https://drive.google.com/file/d/1iGNgROAAMpKzwmJaU7lzsRQmchgVdMzz/view?usp=sharing). For each data folder, there contains one or more trajectories recorded, e.g., `traj0`, `traj1`, ...,`trajN`, in which `color/` and `depth` include RGB (`.jpg`) and Lidar-generated Depth maps (`.npy`).

## Sim Exp
1. Enter the `ENV.SCENE_PATH` with the scene directory in `configs/template.yaml`
2. Enter the `LOGDIR`, `DATADIR`, and `RAFT_MODEL_DIR` in `scripts/active_edn.sh`
3. run `bash scriput/active_edn.sh`


## Reference
We thanks the following works as our works' foundation:
- [Monocular depth estimation from a single image](https://github.com/nianticlabs/monodepth2)
- [A flexible, high-performance 3D simulator for Embodied AI research](https://github.com/facebookresearch/habitat-sim)
- [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT)