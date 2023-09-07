# Modified from the work https://github.com/richardos/occupancy-grid-a-star
import math
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

from scipy import ndimage


def dist2d(point1, point2):
    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)

def _get_movements_4n():
    """
    Get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]


def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


def a_star(start_m, goal_m, gmap, movement='8N', occupancy_cost_factor=100):
    """
    A* for 2D occupancy grid.
    :param start_m: start node (x, y) in meters
    :param goal_m: goal node (x, y) in meters
    :param gmap: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :param occupancy_cost_factor: a number the will be multiplied by the occupancy probability
        of a grid map cell to give the additional movement cost to this cell (default: 3).
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """

    # get array indices of start and goal
    start = gmap.get_index_from_coordinates(start_m[0], start_m[1])
    goal = gmap.get_index_from_coordinates(goal_m[0], goal_m[1])

    # check if start and goal nodes correspond to free spaces
    #if gmap.is_occupied_idx(start):
    #    raise Exception('Start node is not traversable')

    #if gmap.is_occupied_idx(goal):
    #    raise Exception('Goal node is not traversable')

    # add start node to front
    # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
    start_node_cost = 0
    start_node_estimated_cost_to_goal = dist2d(start, goal) + start_node_cost
    front = [(start_node_estimated_cost_to_goal, start_node_cost, start, None)]

    # use a dictionary to remember where we came from in order to reconstruct the path later on
    came_from = {}

    # get possible movements
    if movement == '4N':
        movements = _get_movements_4n()
    elif movement == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # while there are elements to investigate in our front.
    while front:
        # get smallest item and remove from front.
        element = heappop(front)

        # if this has been visited already, skip it
        total_cost, cost, pos, previous = element
        if gmap.is_visited_idx(pos):
            continue

        # now it has been visited, mark with cost
        gmap.mark_visited_idx(pos)

        # set its previous node
        came_from[pos] = previous

        # if the goal has been reached, we are done!
        if pos == goal:
            break

        # check all neighbors
        for dx, dy, deltacost in movements:
            # determine new position
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_pos = (new_x, new_y)

            # check whether new position is inside the map
            # if not, skip node
            if not gmap.is_inside_idx(new_pos):
                continue

            # add node to front if it was not visited before and is not an obstacle
            if (not gmap.is_visited_idx(new_pos)): #and (not gmap.is_occupied_idx(new_pos)):
                potential_function_cost = gmap.get_data_idx(new_pos)*occupancy_cost_factor
                new_cost = cost + deltacost + potential_function_cost
                new_total_cost_to_goal = new_cost + dist2d(new_pos, goal) + potential_function_cost

                heappush(front, (new_total_cost_to_goal, new_cost, new_pos, pos))

    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    if pos == goal:
        while pos:
            path_idx.append(pos)
            # transform array indices to meters
            pos_m_x, pos_m_y = gmap.get_coordinates_from_index(pos[0], pos[1])
            path.append((pos_m_x, pos_m_y))
            pos = came_from[pos]

        # reverse so that path is from start to goal.
        path.reverse()
        path_idx.reverse()

    return path, path_idx


class OccupancyGridMap:
    def __init__(self, data_array, bounds, occupancy_threshold=0.8):
        """
        Creates a grid map
        :param data_array: a 2D array with a value of occupancy per cell (values from 0 - 1)
        :param cell_size: cell size in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
        A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """

        self.data = data_array
        self.dim_cells = data_array.shape

        self.lower_bound, self.upper_bound = bounds

        cell_size = min(abs(self.upper_bound[coord] - self.lower_bound[coord]) / 512 for coord in [0, 2])
        self.dim_meters = (self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        # 2D array to mark visited nodes (in the beginning, no node has been visited)
        self.visited = np.zeros(self.dim_cells, dtype=np.float32)

    def mark_visited_idx(self, point_idx):
        """
        Mark a point as visited.
        :param point_idx: a point (x, y) in data array
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        self.visited[x_index][y_index] = 1.0

    def mark_visited(self, point):
        """
        Mark a point as visited.
        :param point: a 2D point (x, y) in meters
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.mark_visited_idx((x_index, y_index))

    def is_visited_idx(self, point_idx):
        """
        Check whether the given point is visited.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is visited, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        if self.visited[x_index][y_index] == 1.0:
            return True
        else:
            return False

    def is_visited(self, point):
        """
        Check whether the given point is visited.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is visited, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_visited_idx((x_index, y_index))

    def get_data_idx(self, point_idx):
        """
        Get the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :return: the occupancy value of the given point
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        return self.data[x_index][y_index]

    def get_data(self, point):
        """
        Get the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :return: the occupancy value of the given point
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.get_data_idx((x_index, y_index))

    def set_data_idx(self, point_idx, new_value):
        """
        Set the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :param new_value: the new occupancy values
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        self.data[x_index][y_index] = new_value

    def set_data(self, point, new_value):
        """
        Set the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :param new_value: the new occupancy value
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        self.set_data_idx((x_index, y_index), new_value)

    def is_inside_idx(self, point_idx):
        """
        Check whether the given point is inside the map.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is inside the map, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            return False
        else:
            return True

    def is_inside(self, point):
        """
        Check whether the given point is inside the map.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is inside the map, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_inside_idx((x_index, y_index))

    def is_occupied_idx(self, point_idx):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is occupied, false otherwise
        """
        x_index, y_index = point_idx
        if self.get_data_idx((x_index, y_index)) >= self.occupancy_threshold:
            return True
        else:
            return False

    def is_occupied(self, point):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is occupied, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_occupied_idx((x_index, y_index))

    def get_index_from_coordinates(self, x, y):
        """
        Get the array indices of the given point.
        :param x: the point's x-coordinate in meters
        :param y: the point's y-coordinate in meters
        :return: the corresponding array indices as a (x, y) tuple
        """
        grid_size = (
            abs(self.upper_bound[2] - self.lower_bound[2]) / self.dim_cells[0],
            abs(self.upper_bound[0] - self.lower_bound[0]) / self.dim_cells[1],
            )
        grid_x = int((x - self.lower_bound[2]) / grid_size[0])
        grid_y = int((y - self.lower_bound[0]) / grid_size[1])
        return grid_x, grid_y

    def get_coordinates_from_index(self, x_index, y_index):
        """
        Get the coordinates of the given array point in meters.
        :param x_index: the point's x index
        :param y_index: the point's y index
        :return: the corresponding point in meters as a (x, y) tuple
        """
        grid_size = (
            abs(self.upper_bound[2] - self.lower_bound[2]) / self.dim_cells[0],
            abs(self.upper_bound[0] - self.lower_bound[0]) / self.dim_cells[1],
        )
        realworld_x = self.lower_bound[2] + x_index * grid_size[0]
        realworld_y = self.lower_bound[0] + y_index * grid_size[1]
        return realworld_x, realworld_y

    def plot(self, alpha=1, min_val=0, origin='lower'):
        """
        plot the grid map
        """
        plt.imshow(self.data, vmin=min_val, vmax=1, origin=origin, interpolation='none', alpha=alpha)
        plt.draw()



class Planner:
    def __init__(self, bounds, dilation_size, mode='sim'):
        self.bounds = bounds
        self.dilation_size = dilation_size
        if mode=='sim':
            self.turn_angle = 2.
            self.move_meter = 0.03
        elif mode=='real':
            self.turn_angle = 4.
            self.move_meter = 0.057


    def plan(self, costmap, start, goal):
        phy_path, grid_path = self.get_waypoints(costmap, start, goal)
        actions = self.path2action(phy_path, start)
        return actions, grid_path

    def get_waypoints(self, _costmap, start, goal,):
        '''
        costmap
        start: Extrinsic matrix with shape (4, 4)
        goal: [z, x]   
        '''
        start_pos = [start[2, 3], start[0, 3]]

        costmap = (_costmap > 0).astype(np.float32)
        costmap[costmap > 0] = 1000  # occupied area
        costmap[_costmap == 0] = 100 # unknown area

        # Dilate the occupied points, in case of collision
        costmap = ndimage.grey_dilation(costmap,
                                        size=self.dilation_size)

        gmap = OccupancyGridMap(
                    data_array=costmap,
                    bounds=self.bounds,
        )

        phy_path, grid_path = \
                a_star( start_pos ,
                goal_m=goal,
                gmap=gmap,
                movement='8N', 
                occupancy_cost_factor=100  )

        return phy_path, grid_path

    def path2action(self, waypoints, start):
        start_pos = [start[2, 3], start[0, 3]]
        start_ang = np.arctan2(-start[0, 0], -start[0, 2]) * 180 / np.pi

        poses = np.concatenate([np.array([start_pos]), np.array(waypoints)], axis=0)
        ang = np.arctan2(poses[1:, 0] - poses[:-1, 0],
                         poses[1:, 1] - poses[:-1, 1]) * 180 / np.pi

        angs = np.concatenate([np.expand_dims(start_ang, 0), ang], axis=0)
        rots = angs[1:] - angs[:-1]
        rots[np.abs(rots) > 180] -= np.sign(rots[np.abs(rots) > 180]) * 360
        rots = np.around(rots, decimals=1)

        dists = np.sqrt(((poses[1:, ] - poses[:-1]) ** 2).sum(axis=1))

        actions = []
        forward_counter = 0
        for r, dist in zip(rots, dists):
            if r == 0:
                forward_counter += dist
            else:
                actions +=  ['move_forward'] * int(forward_counter / self.move_meter)  
                if r < 0:
                    actions += ['turn_left'] * int( np.abs(r) // self.turn_angle)  #+ ['move_forward'] * int(np.ceil(dist))
                elif r > 0:
                    actions += ['turn_right'] * int( np.abs(r) // self.turn_angle) #+ ['move_forward'] * int(np.ceil(dist))
                forward_counter = dist

        if forward_counter != 0:
            actions +=  ['move_forward'] * int(forward_counter / self.move_meter)  

        return actions







