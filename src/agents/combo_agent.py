import numpy as np
from skimage.measure import label

from src.agents.frontagent import FrontAgent, Map


class MyAgent(FrontAgent):
    def __init__(self, env, cfg, model_name='monodepth', interval=1):
        super(MyAgent, self).__init__(env, cfg, model_name, interval)

        self.steps = cfg.FRAMENUM_PER_STEP
        self.ref_frame_index = cfg.REF_FRAME_INDEX

        self.exploit_counter = 9000
        self.explore_counter = 0

        self.last_policy = None

    def act(self, model, history_observations):
        if self.exploit_counter > 400:
            print('Front Explore')
            history = history_observations[-400:]
            self._update_states(model=model, history_observations=history)
            obs = self._explore_act(model, history)

            self.explore_counter += len(obs)
            if self.explore_counter > 800:
                self.exploit_counter = 0
                self.explore_counter = 0
            self.last_policy = 'frontier'
        else:
            obs = self._inconsist_act(model, history_observations)
            self.last_policy = 'inconsist'

            self.exploit_counter += len(obs)

        return obs

    def _inconsist_act(self, model, observations):
        
        if self.model_name == 'monodepth':
            self.cur_costmap = self.create_singlemap(model, observations[-1:])
            self.past_costmap = self.create_singlemap(model, observations[self.ref_frame_index:self.ref_frame_index+1])
        elif self.model_name == 'manydepth':
            self.cur_costmap = self.create_singlemap(model, observations[-self.interval-1:])
            self.past_costmap = self.create_singlemap(model, observations[self.ref_frame_index-self.interval:self.ref_frame_index+1])


        if (self.cur_costmap is not None) and (self.past_costmap is not None):
            inconsist_center, flag = self.find_inconsistent_region(self.cur_costmap, self.past_costmap)
            if flag:
                goal_idx = self.find_navigable_point(inconsist_center)
                goal = self.from_grid(goal_idx[0], goal_idx[1], self.map.shape, self.env.physical_bounds)

                self.map.visualize(goal=goal)

                return self.navigate_get_obs(model=model, current_pose=self.history_obs[-1]['Ext'], goal=goal)

        return self._random_walk()

    def find_inconsistent_region(self, map1, map2):
        occ1 = map1.get_map() > 0
        occ2 = map2.get_map() > 0
        mask1 = map1.get_map() != 0
        mask2 = map2.get_map() != 0
        inconsist = np.logical_and(mask1, mask2) * (occ1 != occ2)

        groups = label(inconsist)
        group_num = groups.max()
        max_group = []

        for i in range(1, group_num+1):
            cur_group = np.argwhere(groups == i)
            if len(cur_group) > len(max_group):
                max_group = cur_group

        if max_group != []:
            c =  np.array(max_group).mean(axis=0)
            return c.astype(int), True 
        else:
            return None, False

    def find_navigable_point(self, goal):
        p_ = np.array(goal).copy()
        while not self.map.is_navigable(p_):
            noise = 4 * np.random.normal(size=(2))
            if self.map.is_legal( np.floor(p_ + noise)):
                p_ = (p_ + noise).astype(int)
        return p_
        
    def create_singlemap(self, model, observation):
        pose = observation[-1]['Ext'] 
        if model is None:
            depth = observation[-1]['depth_sensor'] 
        else:
            depth = model.pred_depth(observation)
        points = self.backproj(depth).squeeze()

        # filter some points in terms of height
        points = points[:, points[1,:] > self.cliplow]
        points = points[:, points[1,:] < self.cliphigh]

        points = points[:, (points[2] != 0)]
        # We do not trust the farther distance estimation
        ## here z's direction is backward
        occ_points = points[:, (points[2] > self.mark_range)]

        empty_points = points.clone()
        empty_points[:3, (points[2] < self.empty_range)] *= self.empty_range /  empty_points[2, (points[2] < self.empty_range)]

        occ_proj_points = (pose[:3, :] @ occ_points.numpy()).T
        empty_proj_points = (pose[:3, :] @ empty_points.numpy()).T

        if occ_proj_points.shape[1] == 0 or occ_proj_points.shape[0] ==0:
            return

        costmap = Map(self.env, self.cfg)
        costmap.update(occ_pts=occ_proj_points, empty_pts=empty_proj_points, current_pose=pose)
        return costmap
