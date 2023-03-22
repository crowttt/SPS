# sys
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

# log
from loguru import logger


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
                 exp_name,
                 data_path,
                 label_path,
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

        logger.info('Loading all dataset')
        self.load_data(mmap)
        logger.info('Done')
        logger.info('Load label history')
        self.history()
        logger.info('Done')

    def load_data(self, mmap):
        # data: N C V T M
        # N 代表视频的数量，通常一个 batch 有 256 个视频（其实随便设置，最好是 2 的指数）；
        # C 代表关节的特征，通常一个关节包含x,y,acc 等 3 个特征（如果是三维骨骼就是 4 个），x,y为节点关节的位置坐标，acc为置信度。
        # T 代表关键帧的数量，一般一个视频有 150 帧。
        # V 代表关节的数量，通常一个人标注 18 个关节。
        # M 代表一帧中的人数，一般选择平均置信度最高的 2 个人。

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.data = self.data[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        # get data
        data_pair_first = self.data[index][0]
        data_pair_sec = self.data[index][1]
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
        data = torch.stack((torch.tensor(data_pair_first), torch.tensor(data_pair_sec)), dim=0)
        return data, label

    def history(self):
        selected_pairs = session.query(SelectPair).filter(
            SelectPair.exp_name == self.exp_name,
            SelectPair.done == True
        ).all()
        self.label = [1 if pair.first_selection > pair.sec_selection else 0 for pair in selected_pairs]
        pairs = [(pair.first, pair.second) for pair in selected_pairs]

        ######################################
        # adfsfa = [
        #     (
        #         session.query(Kinetics).get(pair[0]).index,
        #         session.query(Kinetics).get(pair[1]).index
        #     ) for pair in pairs
        # ]
        # adfsfa = list(filter(lambda x : x[0] and x[1] , adfsfa))
        ######################################
        pairs_first = [pair[0] for pair in pairs]
        pairs_second = [pair[1] for pair in pairs]
        first_kinetics = session.query(Kinetics).filter(Kinetics.name.in_(pairs_first)).all()
        second_kinetics = session.query(Kinetics).filter(Kinetics.name.in_(pairs_second)).all()
        first_kinetics = [next(q for q in first_kinetics if q.name == name) for name in pairs_first]
        second_kinetics = [next(q for q in second_kinetics if q.name == name) for name in pairs_second]

        if [kinetics.name for kinetics in first_kinetics] == pairs_first:
            print('Ok')
        if [kinetics.name for kinetics in second_kinetics] == pairs_second:
            print('Ok')

        first_kinetics = [kinetics.index for kinetics in first_kinetics]
        second_kinetics = [kinetics.index for kinetics in second_kinetics]

        # 權宜之計
        # first_kinetics_tmp = []
        # second_kinetics_tmp = []
        # for i in range(len(first_kinetics)):
        #     if first_kinetics[i].has_skeleton and second_kinetics[i].has_skeleton:
        #         first_kinetics_tmp.append(first_kinetics[i].index)
        #         second_kinetics_tmp.append(second_kinetics[i].index)
        # first_kinetics = first_kinetics_tmp
        # second_kinetics = second_kinetics_tmp
        # self.label = self.label[: len(second_kinetics)]

        # extract labeled data from total dataset
        self.data_pairs_first = self.data[first_kinetics]
        self.data_pairs_sec = self.data[second_kinetics]
        self.data = [(data_pair_first, data_pair_sec) for data_pair_first, data_pair_sec in zip(self.data_pairs_first, self.data_pairs_sec)]


class DatasetFeeder(Feeder):
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)
        self.load_label()

    def load_label(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        name = self.sample_name[index]
        
        # processing
        if self.random_choose:
            data = tools.random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = tools.auto_pading(data, self.window_size)
        if self.random_move:
            data = tools.random_move(data)

        return data, name
