import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
import random
import pandas

class CarlaDataset(Dataset):
    def __init__(self, traj_path=None, map_path=None):
        # Use provided paths or default to None
        self.traj_path = traj_path or './trajectories_trafficLight_yaw.pkl'
        self.waypoints_path = map_path or './waypoints_xy.pkl'
        self.map_path = './map_dict.pkl'
        self.data = pickle.load(open(self.traj_path, 'rb'))
        self.waypoints_xy = np.array(pickle.load(open(self.waypoints_path, 'rb')))
        self.tree = cKDTree(self.waypoints_xy)
        self.radius = 30.0
        self.map_dict = pickle.load(open(self.map_path, 'rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        traj, label = self.data[index]
        traj = torch.tensor(traj.copy())
        label = torch.tensor(label.copy())

        target_coordinateinate = traj[0][-1][:2].numpy().tolist()
        indices = self.tree.query_ball_point(target_coordinateinate, self.radius)
        # 获取符合条件的坐标点
        filtered_coords = self.waypoints_xy[indices]
        nbr_waypoints = torch.tensor(filtered_coords)
        
        # 获取最近的地图dict
        distances = {key: np.linalg.norm(np.array(target_coordinateinate) - np.array(key)) for key in self.map_dict.keys()}
        closest_coordinate = min(distances, key=distances.get)
        closest_value = self.map_dict[closest_coordinate]

        lane_list = [torch.tensor(df[['x', 'y']].values, dtype=torch.float32) for df in closest_value]
        lane_tensor = torch.cat(lane_list, dim=0)
        
        
        n = len(traj)
        # 初始化一个大小为（n，2）的tensor
        ctrs = torch.zeros((n, 2))
        # 遍历前n个轨迹
        for i in range(n):
            last_row = traj[i][-1][:2].clone().detach().requires_grad_(True)
            ctrs[i] = last_row

        data = {'feat': traj, 'ctrs': ctrs, 'nbr_waypoints': nbr_waypoints, 'lane_list': lane_tensor, 'label': label}
        return data
    
    
def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)