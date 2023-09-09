# sys
import sys
import numpy as np
import random
import pickle

import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sqlalchemy import func, or_

# operation
from . import tools

# db
from db.session import session
from db.model import Kinetics

from config import high_risk_class, low_risk_class


class Feeder(torch.utils.data.Dataset):
    def load_data(self):
        score = pd.read_csv(self.score_path, sep='\t')
        score = score[(score['Risk Level'] == 'High') | (score['Risk Level'] == 'Low')]
        score = score.sample(frac=1, random_state=42)
        action = list(score['Action'])
        self.level = list(score['Risk Level'])
        self.label = list(score['Risk score'])

        kinetics = session.query(Kinetics).filter(Kinetics.name.in_(action)).all()
        kinetics_idx = [next(q.index for q in kinetics if q.name == name) for name in action]

        self.data = np.load(self.data_path, mmap_mode='r')
        self.data = self.data[kinetics_idx]


    def __init__(self, data_path, score_path):
        self.data_path = data_path
        self.score_path = score_path
        self.load_data()


    def __getitem__(self, index):
        label = self.label[index]

        level = self.level[index]
        if level == 'High':
            label = 1
        else:
            label = 0

        return torch.tensor(self.data[index]), torch.tensor(label)


    def __len__(self):
        return len(self.label)


class TestFeeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data()

    def load_data(self):
        high_risk_kinetic = session.query(Kinetics.index, Kinetics.label_idx, Kinetics.label).filter(
            Kinetics.label.in_(high_risk_class),
            Kinetics.has_skeleton == True
        ).all()

        low_risk_kinetic = session.query(Kinetics.index, Kinetics.label_idx, Kinetics.label).filter(
            Kinetics.label.in_(low_risk_class),
            Kinetics.has_skeleton == True
        ).all()

        data = np.load('./data/Kinetics/train_data.npy', mmap_mode='r')
        kinetic_index = [k.index for k in high_risk_kinetic + low_risk_kinetic]

        self.classes = [k.label for k in high_risk_kinetic + low_risk_kinetic]
        self.label = [1 for i in range(len(high_risk_kinetic))] + [0 for i in range(len(low_risk_kinetic))]
        self.data = data[kinetic_index]


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = self.data[index]
        label = self.label[index]
        classes = self.classes[index]
        
        # processing
        if self.random_choose:
            data = tools.random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = tools.auto_pading(data, self.window_size)
        if self.random_move:
            data = tools.random_move(data)

        return torch.tensor(data), label, classes
