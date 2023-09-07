import io
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.my_utils import *
try:
    from habitat.utils.visualizations import maps
except ImportError:
    print('No habitat in the environment')

from matplotlib import pyplot as plt
from multiprocessing import Pool

from src.agents.planners import Planner
from skimage.measure import label

def euclidean(a, b):
    return np.sqrt( ((a-b)**2).mean() )

def dest_point_from_ang_dist(current_pos, angle, distance):
    z, x = current_pos
    delta_x = distance * np.cos(angle)
    delta_z = distance * np.sin(angle)
    return z+delta_z, x+delta_x

class FrontAgent:
    def __init__(self, env, cfg=None, model_name='monodepth', interval=1):
        self.cfg = cfg
        self.map = Map(env, cfg)
        self.env = env
        self.model_name = model_name

        self.interval = interval
        ## 
        self.planner = Planner(env.physical_bounds, cfg.FRONTIER.PLANNER.DILATION_SIZE, env.type)
        self.history_obs = deque(maxlen=200)
        self.reliable_traj_dist = cfg.FRONTIER.RELIABLE_TRAJ_DIST

        self.hist_frontier = deque(maxlen=3)
        self.steps = cfg.FRAMENUM_PER_STEP
        self.policy = {
            0: ['turn_left'] * 8 + ['move_forward'] * (self.steps - 8 ),
            1: ['move_forward'] * self.steps,
            2: ['turn_right'] * 8 + ['move_forward'] * (self.steps - 8 ),
        }

        if env.type == 'sim':
            #self.K = np.array([[320, 0, 320, 0],
            #                [0,  96, 96, 0],
            #                [0,   0,  1, 0],
            #                [0, 0, 0, 1]], dtype=np.float32)
            self.K = np.array([[1, 0, 0, 0],
                            [0,  1, 0, 0],
                            [0,   0,  1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
            self.backproj = self.sim_backproj
        else:
            self.K = np.array([[904.62 , 0,     640., 0],
                            [0,      904.62 , 360., 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
            self.full_res_shape = (640, 192) # (1024, 320) #(640, 192)
            self.K[0] = self.K[0]   * self.full_res_shape[0] / 1280.
            self.K[1] = self.K[1]   * self.full_res_shape[1] / 720.
            self.backproj = self.real_backproj

        self.HEIGHT = 192
        self.WIDTH = 640

        self.explore_prob = cfg.EXPLORE_PROB
        self.mark_range = self.cfg.FRONTIER.MAP_MARK_RANGE
        self.empty_range = self.cfg.FRONTIER.MAP_CLEAR_RANGE
        self.cliplow  = self.cfg.FRONTIER.PCL_HEIGHT_CLIP_LOW
        self.cliphigh = self.cfg.FRONTIER.PCL_HEIGHT_CLIP_HIGH

        # TODO:
        self.counter = 0

    def act(self, model, history_observations):
        #history = history_observations[-2000::15] + [history_observations[-1]]
        # We only assume the recent 30 frames is convincing, catering to the real world egomotion cases
        if np.random.rand() < self.explore_prob:
            obs_ = self._random_walk()
        else:
            history = history_observations[-400:]
            self._update_states(model=model, history_observations=history)
            obs_ = self._explore_act(model, history)
            #obs += self._round_look()
        return obs_


    def _update_states(self, model, history_observations):
        # We only assume the recent 30 frames is convincing, catering to the real world egomotion cases
        self.update_historyobs(history_observations)

        # Remember the place the robot went as empty [deprecated]
        #self.map._update_empty_map([obs['Ext'] for obs in history_observations])

        self.map.reset_costmap()
        sample_interval  = max(1, len(self.history_obs) // 60)
        self.update_map(model, list(self.history_obs)[::sample_interval])

        # before planning to the new area, update nearby local map
        #nearby_obs = self._round_look()
        #self.update_map(model, nearby_obs)

    def _explore_act(self, model, history_observations):

        current_pos = self.history_obs[-1]['Ext'][(2, 0), 3]

        # Find next goal
        costmap = self.map.get_map()
        frontiers = self._compute_frontiers(costmap)
        grid_idx, goal = self._pick_goal_from_frontiers(frontiers, current_pos)

        if grid_idx is not None:
            self.hist_frontier.append(grid_idx.astype(np.int))
            if len(self.hist_frontier) >2 and (self.hist_frontier[-1] == self.hist_frontier[-2]).any() and (self.hist_frontier[-2] == self.hist_frontier[-3]).any():
                print("RESET COSTMAP cuz the center does not change{}".format(self.hist_frontier))
                self.map.reset_costmap()
                return self._random_walk()


            self.map.visualize(grid_idx ) 
            return self.navigate_get_obs(model=model, current_pose=self.history_obs[-1]['Ext'], goal=goal)
        else:
            return self._random_walk()

    def navigate_get_obs(self, model, current_pose, goal):
        observations_  = []
        collide_flag = True
        counter = 0

        while collide_flag and (counter <= 5):
            if counter > 0:
                # use random to walk out of the corner
                while collide_flag:
                    #print('Collide, turn right and seek to be out')
                    policy = ['turn_right'] * 10
                    obs, collide_flag = self._collect_data(policy, get_flag=True)
                    observations_ += obs
                    policy = ['move_forward'] * 10
                    obs, collide_flag = self._collect_data(policy, get_flag=True)
                    observations_ += obs
                current_pose = observations_[-1]['Ext']

            actions, grid_path = self.planner.plan(self.map.get_map() , current_pose, goal) 
            if observations_ != []:
                self.update_map(model, observations_[-50::5])
            self.map._vis_plan(grid_path)
            obs, collide_flag = self._collect_data(actions, get_flag=True)
            observations_ += obs
            current_pose = observations_[-1]['Ext']
            counter += 1

        return observations_

    def _label_collision(self, cur_ext_matrix):
        cur_pos = [cur_ext_matrix[2, 3], cur_ext_matrix[0, 3]]
        cur_ang = np.arctan2(-cur_ext_matrix[0, 0], -cur_ext_matrix[0, 2]) 

        half_fov = np.pi / 4
        distance = 0.1 * np.sqrt(2)

        point1 = dest_point_from_ang_dist(current_pos=cur_pos, angle=cur_ang-half_fov, distance=distance)
        point2 = dest_point_from_ang_dist(current_pos=cur_pos, angle=cur_ang+half_fov, distance=distance)

        points = np.linspace(point1, point2, num=50)
        self.map._label_collision_occ(points)


    def update_historyobs(self, new_obs):
        if euclidean(new_obs[0]['Ext'][(0, 2), 3] , new_obs[-1]['Ext'][(0, 2), 3]) > self.reliable_traj_dist:
            self.history_obs = deque(new_obs)
        else:
            self.history_obs += deque(new_obs)

        while True:
            if euclidean(self.history_obs[0]['Ext'][(0, 2), 3] , self.history_obs[-1]['Ext'][(0, 2), 3]) > self.reliable_traj_dist:
                self.history_obs.popleft()
            else:
                break

        # Some statistics
        traj = np.array([obs['Ext'][(0, 2), 3] for obs in self.history_obs ])
        print('Step={}, the mean = {}, \t var = {} '.format(self.counter, traj.mean(axis=0), traj.var(axis=0)))
        self.counter += 1


    def update_map(self, model, history_observations):
        if self.model_name == 'monodepth':
            for i, obs in enumerate(tqdm(history_observations)):
                self._update_map(model, [obs])
        elif self.model_name == 'manydepth':
            for i in tqdm(range(self.interval + 1, len(history_observations))):
                self._update_map(model, history_observations[i-self.interval-1:i] )

    def _update_map(self, model, observation):
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
        self.map.update(occ_pts=occ_proj_points, empty_pts=empty_proj_points, current_pose=pose)


    def real_backproj(self, depth, batch_size=1):
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)

        inv_K = torch.from_numpy(np.linalg.pinv(self.K))
        
        meshgrid = np.meshgrid(range(self.WIDTH), range(self.HEIGHT), indexing='xy')
        
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords)
                                    
        ones = torch.ones(batch_size, 1, self.HEIGHT * self.WIDTH)

        pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, ones], 1)

        # forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:3, :3], pix_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        return cam_points


    def sim_backproj(self, depth, batch_size=1):
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)

        inv_K = torch.from_numpy(np.linalg.pinv(self.K))
        
        #meshgrid = np.meshgrid(range(self.WIDTH), range(self.HEIGHT), indexing='xy')
        meshgrid = np.meshgrid(np.linspace(-1,1,self.WIDTH), np.linspace(1,-1,self.HEIGHT), indexing='xy')
        
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords)
                                    
        ones = torch.ones(batch_size, 1, self.HEIGHT * self.WIDTH)

        pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, -ones], 1)

        # forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:3, :3], pix_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        return cam_points

    def _compute_frontiers(self, map):
        frontiers = np.zeros_like(map)
        for i in np.argwhere(map == 0):
            if i[0] == 0 or i[1] == 0 or i[0] == map.shape[0]-1 or i[1] == map.shape[1]-1:
                continue
            if (map[i[0]-1:i[0]+2, i[1]-1:i[1]+2] < 0).any():
                frontiers[i[0], i[1]] = 1

        fronts_group = label(frontiers)
        group_num = fronts_group.max()
        groups = []
        max_group = []

        out_ = []
        for i in range(1, group_num+1):
            cur_group = np.argwhere(fronts_group == i)
            if len(cur_group) > 20:
                groups += [f for f in cur_group]
                out_.append(cur_group)
            if len(cur_group) > len(max_group):
                max_group = cur_group

        if groups != []:
            self.map.visualize(None, groups, None)
        return out_


    def _pick_goal_from_frontiers(self, frontiers, current_pos):
        ## Pick the frontiers w/ max Euclidean distance 
        if frontiers != []:
            dists = []
            groups = []
            for _, group in enumerate(frontiers):
                grid_idx, goal = self._get_frontier_center(group)
                dist =  euclidean(current_pos, goal)
                if dist >= 1.:
                    dists.append(dist)
                    groups.append(group)

            if dists != []:
                dists = np.array(dists)
                dists /= dists.sum()
                grid_idx, goal = self._get_frontier_center(
                                        groups[
                                            np.random.choice(len(dists), p=dists)
                                            ]
                                        )
                return grid_idx, goal

        return None, None

    def from_grid(self, grid_x, grid_y, grid_resolution, bounds):
        lower_bound, upper_bound = bounds

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        realworld_x = lower_bound[2] + grid_x * grid_size[0]
        realworld_y = lower_bound[0] + grid_y * grid_size[1]
        return realworld_x, realworld_y

    def _get_frontier_center(self, frontiers):
        c =  np.array(frontiers).mean(axis=0)
        goal = self.from_grid(int(c[0]), int(c[1]), self.map.shape, self.env.physical_bounds)
        #print('Frontier Goal is {} (grid), {}'.format(c, goal))
        return c, goal

    def _collect_data(self, policy, get_flag=False):
        if get_flag:
            obs, collide_flag = self.env.step(policy)
            if collide_flag:
                self._label_collision(obs[-1]['Ext'])
            return obs, collide_flag
        else:
            return self.env.step(policy)[0]

    def _random_walk(self):
        observations = []
        for _ in range(5):
            policy = self.policy[np.random.randint(0,3)]
            obs = self._collect_data(policy=policy)
            observations += obs
        return observations

    def _round_look(self):
        observations = []
        policy = ['turn_left']  * 180
        obs = self._collect_data(policy=policy)
        observations += obs[1::4]
        return observations


# CostMap utils
def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.
    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])
    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])
    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 
    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1)) - 1
        max_iter_each = np.amax(np.abs(end - start), axis=1) - 1

    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    bline = np.concatenate([bline[i, range(max_iter_each[i])] for i in range(npts)])

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed
    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.
    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])

class Map():
    def __init__(self, env, cfg):
        self.env = env

        self.log_occupied = cfg.FRONTIER.MAP_LOG_OCCUPIED
        self.log_free = cfg.FRONTIER.MAP_LOG_FREE
        self.decay_factor = cfg.FRONTIER.MAP_DECAY

        if self.env.type == 'sim':
            top_down_map = maps.get_topdown_map_from_sim(
                env._sim, map_resolution=512
                ) 
            self.total = (top_down_map == 1).sum()
            self.top_down_map = maps.colorize_topdown_map(top_down_map)
        else:
            # Resolution is 0.03 meter per pixel
            self.top_down_map = np.zeros((1000, 1000, 3))

        self.shape = self.top_down_map.shape[:2]
        self.log_odds_prob = np.zeros(self.top_down_map.shape[:2])
        self.confident_empty_map = np.zeros(self.top_down_map.shape[:2])

        self.real_bounds = self.env.physical_bounds

    def is_legal(self, point):
        return (point >= 0).all() and (point < self.shape).all()

    def is_navigable(self, point):
        return self.log_odds_prob[point[0], point[1]] <= 0

    def reset_viz(self):
        if self.env.type == 'sim':
            top_down_map = maps.get_topdown_map_from_sim(
                self.env._sim, map_resolution=512
                ) 
            self.top_down_map = maps.colorize_topdown_map(top_down_map)
        else:
            self.top_down_map = np.zeros((1000, 1000, 3))

    def to_3dgrid(self, points, grid_resolution):
        '''
        points with shape (N, 2),  
        '''
        lower_bound, upper_bound = self.real_bounds

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[1],
        )
        grid_z = ((points[:, 2] - lower_bound[2]) / grid_size[0]).astype(np.int)
        grid_x = ((points[:, 0] - lower_bound[0]) / grid_size[1]).astype(np.int)
        grid_y = ((points[:, 1] - lower_bound[1]) / grid_size[1]).astype(np.int)

        return np.stack( [grid_z, grid_x, grid_y], axis=1)

    def to_grid(self, points, grid_resolution):
        '''
        points with shape (N, 2),  
        '''
        lower_bound, upper_bound = self.real_bounds

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        grid_x = ((points[:, 0] - lower_bound[2]) / grid_size[0]).astype(np.int)
        grid_y = ((points[:, 1] - lower_bound[0]) / grid_size[1]).astype(np.int)
        return np.stack( [grid_x, grid_y], axis=1)

    def _vis_costmap(self):
        for p in np.array((self.log_odds_prob>0).nonzero()).T:
            overlay = self.top_down_map.copy()
            cv2.circle(overlay, (p[1], p[0]), radius=1, color=(255, 0, 0), thickness=1)
            alpha = 0.5
            self.top_down_map = cv2.addWeighted(overlay, alpha, self.top_down_map, 1 - alpha, 0)

        for p in np.array((self.log_odds_prob<0).nonzero()).T:
            overlay = self.top_down_map.copy()
            cv2.circle(overlay, (p[1], p[0]), radius=1, color=(0, 255, 0), thickness=1)
            alpha = 0.1
            self.top_down_map = cv2.addWeighted(overlay, alpha, self.top_down_map, 1 - alpha, 0)

    def _vis_plan(self, traj):
        p = traj[0]
        cv2.circle(self.top_down_map, (p[1], p[0]), radius=7, color=(100, 250, 0), thickness=4)
        for p in traj[1:]:
            cv2.circle(self.top_down_map, (p[1], p[0]), radius=1, color=(200, 200, 0), thickness=1)
        p = traj[-1]
        cv2.circle(self.top_down_map, (p[1], p[0]), radius=7, color=(0, 0 ,250), thickness=4)

    def visualize(self, goal=None, fronts=None, name=None):
        if goal is not None:
            goal = (int(goal[0]), int(goal[1]))
            cv2.circle(self.top_down_map, (goal[1], goal[0]), radius=7, color=(0, 255, 0), thickness=3)

        if fronts is not None:
            f = np.array(fronts)
            for p in f:
                cv2.circle(self.top_down_map, (p[1], p[0]), radius=1, color=(0, 0, 255), thickness=1)

        if name is not None:
            self._vis_costmap()
            plt.axis('off')
            plt.imshow(self.top_down_map)
            if name =='buf':
                buf = io.BytesIO() 
                plt.savefig(buf, format='jpeg')
                buf.seek(0)
                image = Image.open(buf)
                image = ToTensor()(image)

                self.reset_viz()
                return image
            else:
                self.reset_viz()
                plt.savefig(name)

    def reset_costmap(self):
        self.log_odds_prob = np.zeros_like(self.log_odds_prob)
        self.reset_viz()
    
    def _update_empty_map(self, _traj):
        traj = np.stack(_traj)
        points = np.stack([traj[:, 2, 3], traj[:, 0, 3]], axis=1)
        trajs = self.to_grid(points, self.log_odds_prob.shape)
        self.confident_empty_map[trajs[:, 0], trajs[:, 1]] = -1

    def _update_3dlog_odds(self, pcls, current_pose):
        # pcls (Num, 3)


        grid_idxs = self.to_3dgrid(pcls, self.log_odds_prob.shape)

        boundz, boundx = self.log_odds_prob.shape 
        filter_grid_idxs = grid_idxs[grid_idxs[:, 0]<boundz] 
        filter_grid_idxs = filter_grid_idxs[0<= filter_grid_idxs[:, 0]] 
        filter_grid_idxs = filter_grid_idxs[filter_grid_idxs[:, 1]<boundx] 
        filter_grid_idxs = filter_grid_idxs[0<= filter_grid_idxs[:, 1]] 

        current_pose = np.expand_dims(current_pose, 0)
        start_points = self.to_3dgrid(current_pose, self.log_odds_prob.shape) #- np.array([mins]) 

        mins, maxs = [], []
        for i in range(3):
            mins.append(min(filter_grid_idxs[:, i].min(), int(start_points[:, i])))
            maxs.append(max(filter_grid_idxs[:, i].max(), int(start_points[:, i])))
        
        #occ3d = np.zeros((boundz, boundx, boundx))
        occ3d = np.zeros([maxs[i] - mins[i] + 1 for i in range(3)])

        filter_grid_idxs -= np.array([mins])
        start_points -= np.array([mins]) 
        starts = start_points.repeat(filter_grid_idxs.shape[0], axis=0)
        ## Occupied points
        occ3d[filter_grid_idxs[:, 0], filter_grid_idxs[:, 1], filter_grid_idxs[:, 2]] = 1

        ## Empty points
        bresenham_path = bresenhamline( starts,
                                        filter_grid_idxs,
                                        max_iter=-1)
        bresenham_path = bresenham_path[bresenham_path[:, 0]<boundz] 
        bresenham_path = bresenham_path[0<=bresenham_path[:, 0]] 
        bresenham_path = bresenham_path[bresenham_path[:, 1]<boundx] 
        bresenham_path = bresenham_path[0<=bresenham_path[:, 1]] 
        bresenham_path = bresenham_path[bresenham_path[:, 2]<boundx] 
        bresenham_path = bresenham_path[0<=bresenham_path[:, 2]] 
        occ3d[bresenham_path[:, 0], bresenham_path[:, 1], bresenham_path[:, 2]] = 0.5

        ## Project 3d to 2d
        occ2d = occ3d.max(axis=2)
        occupied_idx = np.argwhere(occ2d == 1) + np.array([mins[:2]]) #        
        empty_idx = np.argwhere(occ2d == 0.5) + np.array([mins[:2]])#        

        ## Update log odds 
        self.log_odds_prob *= self.decay_factor
        self.log_odds_prob[occupied_idx[:, 0], occupied_idx[:, 1]] += self.log_occupied
        self.log_odds_prob[empty_idx[:, 0], empty_idx[:, 1]] -= self.log_free

        self.log_odds_prob = np.clip(self.log_odds_prob, -1000, 1000)
        #self.log_odds_prob[self.confident_empty_map < 0] = -1000

        for p in np.array((self.log_odds_prob>0).nonzero()).T:
            cv2.circle(self.top_down_map, (p[1], p[0]), radius=1, color=(255, 0, 0), thickness=1)

    def _clip_grid(self, idxs, boundx, boundy):
        filter_grid_idxs = idxs[idxs[:, 1]<boundy] 
        filter_grid_idxs = filter_grid_idxs[0<= filter_grid_idxs[:, 1]] 
        filter_grid_idxs = filter_grid_idxs[filter_grid_idxs[:, 0]<boundx] 
        filter_grid_idxs = filter_grid_idxs[0<= filter_grid_idxs[:, 0]] 

        return filter_grid_idxs

    def _update_log_odds(self, occ_pcls, empty_pcls, current_pose):
        # pcls (Num, 3)
        self.log_odds_prob *= self.decay_factor

        boundx, boundy = self.log_odds_prob.shape 
        # Occupied
        occ_points = np.stack([occ_pcls[:, 2], occ_pcls[:, 0]], axis=1)
        occ_grid_idxs = np.unique(self.to_grid(occ_points, self.log_odds_prob.shape), axis=0)
        filter_occ_grids = self._clip_grid(occ_grid_idxs, boundx, boundy)
        self.log_odds_prob[filter_occ_grids[:, 0], filter_occ_grids[:, 1]] += self.log_occupied

        # Empty
        empty_points = np.stack([empty_pcls[:, 2], empty_pcls[:, 0]], axis=1)
        empty_grid_idxs = np.unique(self.to_grid(empty_points, self.log_odds_prob.shape), axis=0)
        filter_empty_grids = self._clip_grid(empty_grid_idxs, boundx, boundy)
        if filter_empty_grids.shape[0] > 0:
            start_points = self.to_grid(
                            np.array([current_pose[[2,0]]]), 
                            self.log_odds_prob.shape,
                            )
            starts = start_points.repeat(filter_empty_grids.shape[0], axis=0)
            bresenham_path = bresenhamline( starts,
                                            filter_empty_grids,
                                            max_iter=-1)
            bresenham_path = self._clip_grid(bresenham_path, boundx, boundy)
            self.log_odds_prob[bresenham_path[:, 0], bresenham_path[:, 1]] -= self.log_free

        # Saturate the log odds
        self.log_odds_prob = np.clip(self.log_odds_prob, -1000, 1000)

        for p in np.array((self.log_odds_prob>0).nonzero()).T:
            cv2.circle(self.top_down_map, (p[1], p[0]), radius=1, color=(255, 0, 0), thickness=1)

    def _update_log_oddsv2(self, occ_pcls, empty_pcls, current_pose):
        # pcls (Num, 3)
        self.log_odds_prob *= self.decay_factor

        occupied_map = np.zeros_like(self.log_odds_prob)
        empty_map = np.zeros_like(self.log_odds_prob)

        boundx, boundy = self.log_odds_prob.shape 
        # Occupied
        occ_points = np.stack([occ_pcls[:, 2], occ_pcls[:, 0]], axis=1)
        occ_grid_idxs = np.unique(self.to_grid(occ_points, self.log_odds_prob.shape), axis=0)
        filter_occ_grids = self._clip_grid(occ_grid_idxs, boundx, boundy)


        occupied_map[filter_occ_grids[:, 0], filter_occ_grids[:, 1]] = 1

        # Empty
        empty_points = np.stack([empty_pcls[:, 2], empty_pcls[:, 0]], axis=1)
        empty_grid_idxs = np.unique(self.to_grid(empty_points, self.log_odds_prob.shape), axis=0)
        filter_empty_grids = self._clip_grid(empty_grid_idxs, boundx, boundy)
        if filter_empty_grids.shape[0] > 0:
            start_points = self.to_grid(
                            np.array([current_pose[[2,0]]]), 
                            self.log_odds_prob.shape,
                            )
            starts = start_points.repeat(filter_empty_grids.shape[0], axis=0)
            bresenham_path = bresenhamline( starts,
                                            filter_empty_grids,
                                            max_iter=-1)
            bresenham_path = self._clip_grid(bresenham_path, boundx, boundy)

            empty_map[bresenham_path[:, 0], bresenham_path[:, 1]] = 1
  
        # Merge Occupied map and empty map
        # Keep occupied points if it is also labeled as empty
        overlap = np.logical_and(occupied_map, empty_map)
        empty_map[overlap] = 0
        self.log_odds_prob += self.log_occupied * occupied_map
        self.log_odds_prob -= self.log_free * empty_map

        # Saturate the log odds
        self.log_odds_prob = np.clip(self.log_odds_prob, -1000, 1000)

    def _label_collision_occ(self, points):
        '''
        Input:
            points: [N, 2]  in continuous space
        '''
        boundx, boundy = self.log_odds_prob.shape 
        occ_grid_idxs = np.unique(self.to_grid(points, self.log_odds_prob.shape), axis=0)
        filter_occ_grids = self._clip_grid(occ_grid_idxs, boundx, boundy)

        self.log_odds_prob[filter_occ_grids[:, 0], filter_occ_grids[:, 1]] = 1000
        for p in zip(filter_occ_grids[:, 0],filter_occ_grids[:, 1]) :
            overlay = self.top_down_map.copy()
            cv2.circle(overlay, (p[1], p[0]), radius=2, color=(10, 10, 10), thickness=1)
            alpha = 0.3
            self.top_down_map = cv2.addWeighted(overlay, alpha, self.top_down_map, 1 - alpha, 0)

    def update(self, occ_pts, empty_pts, current_pose):
        self._update_log_oddsv2(occ_pts, empty_pts, current_pose[:3, 3])
            
    def get_map(self):
        return self.log_odds_prob

    def summary(self):
        #map = (self.map / self.map.max() * 255).astype(np.uint8)
        #vis = np.stack([map, np.zeros_like(map), np.zeros_like(map)], dim=-1)
        ##vis = cv2.cvtColor(map ,cv2.COLOR_GRAY2RGB)
        #vis_ = cv2.addWeighted(top_down_map, 0.5, vis, 0.5, 0)
        plt.imshow(self.top_down_map)
        print(len(self.map.nonzero()[0]) / self.total)


