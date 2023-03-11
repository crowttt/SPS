# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# operation
from . import tools

# db
from db.session import session
from db.model import SelectPair, Kinetics

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 exp_name,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.exp_name = exp_name

        self.load_data(mmap)
        self.history()

    def load_data(self, mmap):
        # data: N C V T M
        # N 代表视频的数量，通常一个 batch 有 256 个视频（其实随便设置，最好是 2 的指数）；
        # C 代表关节的特征，通常一个关节包含x,y,acc 等 3 个特征（如果是三维骨骼就是 4 个），x,y为节点关节的位置坐标，acc为置信度。
        # T 代表关键帧的数量，一般一个视频有 150 帧。
        # V 代表关节的数量，通常一个人标注 18 个关节。
        # M 代表一帧中的人数，一般选择平均置信度最高的 2 个人。

        self.label =  label

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        pair = self.pairs[index]

        data_pair_first = np.array(self.data[pair[0]])
        data_pair_sec = np.array(self.data[pair[1]])
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_pair_first = tools.random_choose(data_pair_first, self.window_size)
            data_pair_sec = tools.random_choose(data_pair_sec, self.window_size)
        elif self.window_size > 0:
            data_pair_first = tools.auto_pading(data_pair_first, self.window_size)
            data_pair_sec = tools.auto_pading(data_pair_sec, self.window_size)
        if self.random_move:
            data_pair_first = tools.random_move(data_pair_first)
            data_pair_sec = tools.random_move(data_pair_sec)

        return data_pair_first, data_pair_sec, label

    def history(self):
        selected_pairs = session.query(SelectPair).filter(
            SelectPair.exp_name == exp_name,
            SelectPair.done == True
        ).all()
        self.label = [0 if pair.first_selection > pair.sec_selection else 1 for pair in selected_pairs]
        pairs = [(pair.first, pair.second) for pair in selected_pairs]
        self.pairs = [
            (
                session.query(Kinetics.index).filter(Kinetics.name == pair[0]).first(),
                session.query(Kinetics.index).filter(Kinetics.name == pair[1]).first(),
            ) for pair in pairs
        ]
 