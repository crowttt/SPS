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
from torchvision import datasets, transforms
from sqlalchemy import func, or_

# operation
from . import tools

# db
from db.session import session
from db.model import SelectPair, Kinetics

# log
from loguru import logger


class Feeder(torch.utils.data.Dataset):
    def load_data(self):
        # f'static/{self.exp_name}_scores.csv', sep='\t'
        score = pd.read_csv(self.score_path, sep='\t')
        score = score.sample(frac=1, random_state=42)
        action = list(score['Action'])
        level = list(score['Risk Level'])
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
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])


    def __len__(self):
        return len(self.label)
