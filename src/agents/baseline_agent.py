import random
import numpy as np
import cv2

class RandomAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.steps = cfg.FRAMENUM_PER_STEP
        
        self.action_space = {
            0: 'turn_left', 
            1: 'move_forward', 
            2: 'turn_right'
        }

    def act(self, model, observations):
        actions = self._random_sample(num=self.steps)
        return self._collect_data(actions)

    def _random_sample(self, num=1):
        out_ = [self.action_space[np.random.randint(0,3)] for _ in range(num)]
        return out_

    def _collect_data(self, policy, get_flag=False):
        if get_flag:
            return self.env.step(policy)
        else:
            return self.env.step(policy)[0]

class HeuristicRandomAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.steps = cfg.FRAMENUM_PER_STEP
        
        self.policy = {
            0: ['turn_left'] * 8 + ['move_forward'] * (self.steps - 8 ),
            1: ['turn_right'] * 8 + ['move_forward'] * (self.steps - 8 ),
            2: ['move_forward'] * self.steps
        }


    def act(self, model, observations):
        actions = self._random_sample(num=self.steps)
        return self._collect_data(actions)

    def _random_sample(self, num=1):
        out_ = self.policy[np.random.randint(0,3)] 
        return out_

    def _collect_data(self, policy, get_flag=False):
        if get_flag:
            return self.env.step(policy)
        else:
            return self.env.step(policy)[0]

class ForwardAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.steps = cfg.FRAMENUM_PER_STEP
        self.policy = {
            0: ['turn_left'] * 8 + ['move_forward'] * (self.steps - 8 ),
            1: ['turn_right'] * 8 + ['move_forward'] * (self.steps - 8 ),
            2: ['move_forward'] * self.steps
        }

        self.random_obs_count = 0
        self.explore_prob = cfg.EXPLORE_PROB

    def act(self, model, observations):

        if np.random.rand() < self.explore_prob:
            obs_ = self._random_walk()
        else:
            obs_ = []
            policy = ['move_forward'] * self.steps

            obs, collision_flag = self._collect_data(policy)
            obs_ += obs

            while collision_flag:
                policy = ['turn_right'] * 5
                obs, collision_flag = self._collect_data(policy)
                obs_ += obs

        return obs_

    def _random_walk(self):
        observations = []
        for _ in range(5):
            policy = self.policy[np.random.randint(0,3)]
            obs = self._collect_data(policy=policy)
            observations += obs

        self.random_obs_count += len(observations)
        return observations 

    def _collect_data(self, policy):
        return self.env.step(policy)