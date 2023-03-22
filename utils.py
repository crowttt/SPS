import os
import time
from abc import ABC, abstractmethod
import yaml
import uuid
import argparse
import random
import json
import requests
import torch
from sqlalchemy import func

from db.model import SelectPair, questionnaire, Users, Kinetics
from db.session import session


class Base(ABC):
    def __init__(self, arg):
        self.arg = arg
        # self.load_arg(argv.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngpus = ngpu(self.arg.device)
        self.gpus = visible_gpu(self.arg.device)
        self.load_model()
        # self.load_data()


    @abstractmethod
    def load_model(self):
        pass


    @abstractmethod
    def load_data(self):
        pass


    @abstractmethod
    def train(self):
        pass



def load_arg(config):
    parser = argparse.ArgumentParser()
    if config is not None:
        # load config file
        with open(config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)

        parser.set_defaults(**default_arg)
        arg = parser.parse_args()
        return arg


def init_sample(config):

    exist_sample = session.query(SelectPair).filter(
         SelectPair.exp_name == config.process['exp_name']
    ).count()

    if exist_sample:
        return

    sample_name = session.query(Kinetics.name).filter(
        Kinetics.has_skeleton == True,
        Kinetics.index != None
    ).all()
    size = len(sample_name)
    sample = [random.randint(0,size) for _ in range(500)]
    sample = [sample_name[i] for i in sample]
    sample_pair = [(i, j) for i, in sample for j, in sample]
    sample_pair = list(set([tuple(sorted(pair)) for pair in sample_pair]))
    sample_pair = list(filter(lambda x: x[0] is not x[1], sample_pair))
    random.shuffle(sample_pair)
    sample_pair = sample_pair[:int(config.process['pair_batch_size'])]

    store_sample(config, sample_pair)

    dispatch(config)


def ngpu(gpus):
    """
        count how many gpus used.
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


def dispatch(config, round_num=0):
    payload = json.dumps({
        "exp_name": config.process['exp_name'],
        "round_num": round_num,
        "person": config.process['person_per_batch'],
        "size": config.process['pair_batch_size']
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request(
        "POST",
        os.environ.get("DISPATCH_URL", "http://168.138.47.102:5001/dispatch"),
        headers=headers,
        data=payload
    )


def next_round(config):
    label_in_curr_round = 0
    current_round = session.query(func.max(SelectPair.round_num)).scalar()
    while label_in_curr_round is not config.process['pair_batch_size']:
        label_in_curr_round = session.query(SelectPair).filter(
            SelectPair.exp_name == config.process['exp_name'],
            SelectPair.round_num == current_round,
            SelectPair.done == True
        ).count()
        time.sleep(5)
    return current_round + 1


def store_sample(config, sample_pair, curr_round=0):
    for pair in sample_pair:
        select_pair = SelectPair(
            pair_id=str(uuid.uuid4()),
            exp_name=config.process['exp_name'],
            first=pair[0],
            second=pair[1],
            round_num=curr_round
        )
        session.add(select_pair)
    session.commit()


def visible_gpu(gpus):
    """
        set visible gpu.

        can be a single id, or a list

        return a list of new gpus ids
    """
    # print(gpus)
    # print('--------')
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    # gpus = [0,1,2,3]
    # gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, gpus)))
    return list(range(len(gpus)))
