import collections
import timeit
import io
import cv2
import torch
import numpy as np
#import quaternion
try:
    from habitat.utils.visualizations import maps
except:
    pass
from torchvision.transforms import ToTensor
from PIL import Image  
import matplotlib
import matplotlib.cm
from matplotlib import pyplot as plt

PI = 3.1415926
D2PI = lambda x: x /  180 * PI
K = np.array([[1 , 0, 0., 0],
                           [0, 1, 0., 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis



def colorize(value, vmin=None, vmax=None, cmap='Spectral'):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    #else:
    #    # Avoid 0-division
    #    value = value*0.
    # squeeze last dim if it exists

    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True).transpose((2, 0, 1))[:3] # (nxmx4)
    return value

    #rgb_d_im = np.uint8(value * 255)
    #dep_ = cv2.applyColorMap(rgb_d_im, cv2.COLORMAP_MAGMA)
    #return dep_


def cam1_to_cam0(T_world_camera1, T_world_camera0):
    T_camera1_world = np.linalg.pinv(T_world_camera1)
    T_camera1_camera0 = np.matmul(T_camera1_world, T_world_camera0)
    return T_camera1_camera0


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse

class MapStat:
    def __init__(self, env):
        self.env = env
        top_down_map = maps.get_topdown_map_from_sim(
            env._sim, map_resolution=512
            ) 
        self.total = (top_down_map == 1).sum()
        self.top_down_map = maps.colorize_topdown_map(top_down_map)
        self.map = np.zeros(self.top_down_map.shape[:2])
    
    def step(self, trace):
        po = [
            maps.to_grid(
                    p[2], p[0],
                    self.map.shape,
                    sim=self.env._sim,
                )
                for p in trace
            ]
        maps.draw_path(self.top_down_map, po,  maps.MAP_SHORTEST_PATH_COLOR, thickness=2)
        for p in po:
            self.map[p] += 1

    def summary(self, path=None):
        plt.imshow(self.top_down_map)
        if path == None:
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image)
            percent = (len(self.map.nonzero()[0]) / self.total)
            return image, percent
        else:
            plt.savefig(path, format='jpeg')

def vis_map(env, gt_trace, pred_trace):
    top_down_map = maps.get_topdown_map_from_sim(
            env._sim, map_resolution=1024
        )

    gt_po = [
            maps.to_grid(
                    p[2],
                    p[0],
                    (top_down_map.shape[0], top_down_map.shape[1]),
                    sim=env._sim,
                )
                for p in gt_trace
    ]

    top_down_map = maps.colorize_topdown_map(top_down_map)
    maps.draw_path(top_down_map, gt_po, (255, 0, 0) , thickness=4)

    if pred_trace is not None:
        pred_po = [
                maps.to_grid(
                        p[2],
                        p[0],
                        (top_down_map.shape[0], top_down_map.shape[1]),
                        sim=env._sim,
                    )
                    for p in pred_trace
        ]
        maps.draw_path(top_down_map, pred_po,  (0, 255,0), thickness=4)
        for p in pred_po:
            cv2.circle(top_down_map, (p[1], p[0]), radius=5, color=(0, 200, 0), thickness=3)

    for i, p in enumerate(gt_po):
        if i == 0:
            cv2.circle(top_down_map, (p[1], p[0]), radius=8, color=(200, 0, 0), thickness=5)
        else:
            cv2.circle(top_down_map, (p[1], p[0]), radius=5, color=(200, 0, 0), thickness=3)

    plt.imshow(top_down_map)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = Image.open(buf)
    image = ToTensor()(image)
    return image


def dist(p1, p2):
    '''
    Considering the different point numbers in different areas,
    we normalize the distance according to the point numbers
    p1 : (1, 3, N_1, ) 
    p2 : (1, 3, N_2, ) 
    '''
    n_1 = p1.shape[2]
    n_2 = p2.shape[2]
    if n_1 == 0 or n_2 ==0:
        return 1000
    error = (
                (p1.unsqueeze(3).repeat(1, 1, 1, n_2) - \
                p2.unsqueeze(2).repeat(1, 1, n_1, 1, )) ** 2
            ).sum(dim=1)
    
    dist1 = error.min(dim=-1).values.mean(-1)  
    dist2 = error.min(dim=-2).values.mean(-1)
    return dist1 + dist2

def backproj(depth, K, batch_size=1):
    '''
    Input:
        depth: (*, W, H) 
        K: (4, 4)
    
    '''
    #TODO: check the 3d points' scale
    height = 192
    width = 640
    
    inv_K = torch.from_numpy(np.linalg.pinv(K))
    
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
                                   
    ones = torch.ones(batch_size, 1, height * width)

    pix_coords = torch.unsqueeze(torch.stack(
            [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    pix_coords = pix_coords.repeat(batch_size, 1, 1)
    pix_coords = torch.cat([pix_coords, ones], 1)

    cam_points = torch.matmul(inv_K[:3, :3], pix_coords)
    cam_points = depth.view(batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)

    return cam_points

def correct_yz(pose):
    '''correct pose in Habitat to the standard xyz 
    pose with shape (*, 4, 4)
    '''
    correct = np.array([[1., 0, 0, 0],
                        [0, -1., 0, 0],
                        [0, 0 , -1., 0],
                        [0, 0 , 0, 1.]], dtype=np.float32)
    return  np.matmul(pose, correct)

def split(_points):
    '''
    points: (1, 3, N) 
    '''
    points = _points[:, :, _points[0, 1, :] > 0]
    angles = torch.atan2(-points[..., 2, :], points[...,0, :]).squeeze()
    l_ = points[:,:, D2PI(75) > angles].clone()
    r_ = points[:,:, D2PI(105) < angles].clone()
    m = points[:,:, D2PI(75) < angles].clone()
    m_angles = torch.atan2(-m[..., 2, :], m[...,0, :]).squeeze()
    m_ = m[:,:, D2PI(105) > m_angles ].clone()

    return l_, m_, r_

class AvgMeter(object):
    def __init__(self):
        self.value = 0.
        self.count = 0
    def __add__(self, other:float):
        self.value = (self.value * self.count + other) / (self.count + 1)
        self.count += 1
        return self
    def __call__(self):
        return self.value 

    
class Timings:
    """Not thread-safe."""

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        """Save an update for event `name`.
        Nerd alarm: We could just store a
            collections.defaultdict(list)
        and compute means and standard deviations at the end. But thanks to the
        clever math in Sutton-Barto
        (http://www.incompleteideas.net/book/first/ebook/node19.html) and
        https://math.stackexchange.com/a/103025/5051 we can update both the
        means and the stds online. O(1) FTW!
        """
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (
            n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2
        ) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n    %s: %.6fs +- %.6fs (%.2f%%) " % (
                k,
                means[k],
                 stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.6fs" % ( total)
        return result
