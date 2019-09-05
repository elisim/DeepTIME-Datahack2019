from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
import os
import numpy as np
from data import *

DATA_DIR = 'data/'
TEST_DIR = 'test_data/'
REAL_DIR = 'real_data/'
IMAGES_FILE = 'images.tar'
POSE_FILE = 'pose.pkl'
SIGNATURES_FILE = 'signatures.pkl'


class OrcamDataset(Dataset):
    def __init__(self, data_dir, train=True, filter_signatures=True):
        self.data_dir = data_dir
        im_file = os.path.join(data_dir, IMAGES_FILE)
        pose_file = os.path.join(data_dir, POSE_FILE)
        sig_file = os.path.join(data_dir, SIGNATURES_FILE)
        self.pose_paths, self.keypoints, self.scores = read_pose(pose_file)
        self.signatures_paths, self.signatures = read_signatures(sig_file)
        self.signatures = torch.Tensor(self.signatures)
        self.images = Images(im_file)
        if filter_signatures:
            self.actual_paths = list(set(self.images.paths) & set(self.signatures_paths))
        else:
            raise NotImplementedErrorׂ('Should add pretrained net instead of sig')
            self.actual_paths = self.images.paths

    def __len__(self):
        return len(self.actual_paths)

    @staticmethod
    def get_label(path):
        slash = path.find('/')
        underscore = path.find('_')
        num_str = path[underscore + 1:slash]
        return int(num_str)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.actual_paths[idx]
        sign_idx = self.signatures_paths.index(path)
        signature = self.signatures[sign_idx]
        pose_idx = self.pose_paths.index(path)
        pose = self.keypoints[pose_idx]
        image = self.images[path]
        image = torch.Tensor(image.copy())
        label = self.get_label(path)
        pose_dist = get_pose_dists(pose)
        sample = {'image': image, "signature": signature, "pose": pose_dist, "label": torch.Tensor([label])}
        return sample


def get_pose_dists(pose):
    return torch.Tensor(
        [x for x in np.array([np.sqrt(sum((kp_1 - kp_2) ** 2, 0)) for kp_1 in pose for kp_2 in pose]) if x != 0])


def get_data_loaders(batch_size=16, shuffle=True, num_workers=1):
    train_dataset = OrcamDataset(REAL_DIR, train=True)
    test_dataset = OrcamDataset(TEST_DIR, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader
