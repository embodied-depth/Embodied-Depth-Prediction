import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

#import imageio.v2 as imageio
try:
    import habitat_sim
    import magnum as mn
    from habitat.utils.visualizations import maps
except ImportError:
    pass

try:
    from src.ros import DataListener, Controller
except ImportError:
    print('no rospy in lib')

class CollisionError(Exception):
    def __init__(self, error_info=None):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class SimEnv:
    def __init__(self, config, scene_id):
        self.config = config
        base_dir = self.config.SCENE_PATH
        self.SCENE_MAP = \
            {
                0: base_dir + 'mp3d/JeFG25nYj2p/JeFG25nYj2p.glb',
                1: base_dir + 'gibson/Herricks.glb', 
                2: base_dir + 'mp3d/17DRP5sb8fy/17DRP5sb8fy.glb',
                3: base_dir + 'gibson/Eastville.glb',
                4: base_dir + 'mp3d/2t7WUuJeko7/2t7WUuJeko7.glb',
                5: base_dir + 'gibson/Corder.glb',
            }

        self.type = 'sim'
        # Create the simulator
        self.SCENE_PATH = self.SCENE_MAP[scene_id]
        cfg = self._get_env_cfg()
        self._sim = habitat_sim.Simulator(cfg)
        self._sim.seed(self.config.SEED)

        self.physical_bounds = self._sim.pathfinder.get_bounds()

        # Create the agent to the sim
        self.agent = self._sim.initialize_agent(self.config.ENV.DEFAULT_AGENT)

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

        self._hist_turn = 'turn_right'

    def _check_collision(self, _depth):
        H, W = _depth.shape
        return (_depth[H // 2 - 10 : H // 2 + 10, W//2 - 10 : W // 2 + 10] < 0.1).all()
        
    def _step(self, action):
        if action in ['turn_left', 'turn_right']:
            observations = self._sim.step(action)
            #observations = self._sim.step('move_forward')
        else:
            observations = self._sim.step(action)
        state = self.agent.get_state()
        observations['Ext'] = np.array(self.agent.scene_node.transformation_matrix())
        observations['pos'] = state.position
        observations['angle'] = state.rotation
        return [observations]

    def step(self, action):
        '''
        Input: lists of actions 
        Return: 
            obs_: lists of observations
            collided_flag: whether collision occurs
        '''
        obs_ = []
        if type(action) == list:
            for i, a in enumerate(action):
                obs_ += self.step(a)
                if self._check_collision(obs_[-1]['depth_sensor']) or obs_[-1]['collided']:
                    return obs_, True
            return obs_, False
        else:
            obs_ += self._step(action)
            if action in ['turn_left', 'turn_right']:
                self._hist_turn = action
            return obs_

    def reset(self):
        pass
    
    def _get_env_cfg(self):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.SCENE_PATH
        sim_cfg.gpu_device_id = self.config.ENV.CUDA_ID

        # a RGB visual sensor 
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [self.config.SENSOR.HEIGHT, self.config.SENSOR.WIDTH]  
        rgb_sensor_spec.position = [0.0, self.config.SENSOR.POS_HEIGHT, 0.0]
        rgb_sensor_spec.orientation = [self.config.SENSOR.PITCH, 0.0, 0.0,]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # a Depth sensor
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [self.config.SENSOR.HEIGHT, self.config.SENSOR.WIDTH]  
        depth_sensor_1st_person_spec.position = [0.0, self.config.SENSOR.POS_HEIGHT, 0.0]
        depth_sensor_1st_person_spec.orientation = [self.config.SENSOR.PITCH, 0.0, 0.0,]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_1st_person_spec]
        agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=self.config.AGENT.FORWARD_MOUNT)
        ),
        "move_backward":habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=-self.config.AGENT.FORWARD_MOUNT)
        ),

        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=self.config.AGENT.LR_ANGLE)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=self.config.AGENT.LR_ANGLE)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=self.config.AGENT.UD_ANGLE)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=self.config.AGENT.UD_ANGLE)
        ),
        }

        '''
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ] 
        '''
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class RealEnv:
    def __init__(self, no_return=False):
        #self.config = config

        self.no_return = no_return
        self.type = 'real'
        # Create the simulator

        self.control = Controller()
        self.datalistener = DataListener()

        self.physical_bounds = [(-15, -15, -15), (15, 15, 15) ]
        #self.physical_bounds = self._sim.pathfinder.get_bounds()


    def _check_collision(self, _depth):
        H, W = _depth.shape
        return (_depth[H // 2 - 10 : H // 2 + 10, W//2 - 10 : W // 2 + 10] < 0.1).all()
        
    def step(self, actions):
        '''
        Input: lists of actions 
        Return: 
            obs_: lists of observations
            collided_flag: whether collision occurs
        '''
        self.control.act(actions)
        if not self.no_return:
            obs_ = self.datalistener.output_data()

            return obs_
        else:
            return [None]

def test():
    cfg = get_config('configs/default_cfg.yaml')
    env = SimEnv(cfg)
    actions = ['turn_right', 'turn_right','move_forward', 'move_forward', 'move_forward', 'move_forward', 'move_forward',  'turn_right',] * 50

    obss = []
    trace = []
    for a in actions:
        obs = env.step(a)[0]
        obss.append(obs['color_sensor'])
        trace.append( obs['pos'] )

    print('*' * 20)
    print(obss[0].shape)

    m = MapStat(env)
    m.step(trace=trace)
    m.summary()

if __name__ == '__main__':
    from config import get_config
    from tensorboardX import SummaryWriter
    from torchvision.transforms import ToTensor
    from PIL import Image
    test()