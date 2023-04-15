import random
from collections import namedtuple

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sqlalchemy import func, or_

from db.session import session
from db.model import SelectPair, Kinetics
from net.qnet import QNet as Net
from net.stgcn_embed import Model
from utils import visible_gpu, ngpu
from config import REDIS_ENGINE

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_candi'))

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition: tuple):
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*transition))
        else:
            self.memory = self.memory[1:]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def all(self):
        return self.memory

    def load(self, data):
        self.memory = data

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, config):
        self.config = config
        self.eval_net = Net(candi_num=config.process['candi_num'], emb_size=config.embed_args['out_channels']*2)
        self.target_net = Net(candi_num=config.process['candi_num'], emb_size=config.embed_args['out_channels']*2)
        self.experience = set()
        self.memory = Memory(config.dqn_args['memory_size'])
        self.state = []
        self.last_transition = None
        self.double_q = config.dqn_args['double_q']
        self.gamma = config.dqn_args['gamma']
        self.tau = config.dqn_args['tau']
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=config.dqn_args['learning_rate'], weight_decay = config.dqn_args['l2_norm'])
        self.loss_func = nn.MSELoss()
        self.data_path = config.embed_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpus = visible_gpu(config.device)
        self.ngpus = ngpu(config.device)
        if torch.cuda.is_available():
            self.eval_net.to(self.device)
            self.target_net.to(self.device)
        if self.ngpus > 1:
            self.eval_net = nn.DataParallel(self.eval_net, device_ids=self.gpus)
            self.target_net = nn.DataParallel(self.target_net, device_ids=self.gpus)
        self.load_data()


    def initialize(self):
        current_round = session.query(func.max(SelectPair.round_num)).scalar()
        select_pair = session.query(SelectPair).filter(
            SelectPair.exp_name == self.config.process['exp_name'],
            SelectPair.round_num == current_round,
            or_(SelectPair.first_selection > SelectPair.none_selection,
            SelectPair.sec_selection > SelectPair.none_selection)
        ).all()

        self.curr_state = [(pair.first, pair.second) for pair in select_pair]


    def embeded(self, data):
        first = [s[0] for s in data]
        second = [s[1] for s in data]
        first_idx = session.query(Kinetics.index, Kinetics.name).filter(Kinetics.name.in_(first)).all()
        second_idx = session.query(Kinetics.index, Kinetics.name).filter(Kinetics.name.in_(second)).all()
        first_idx = [next(q[0] for q in first_idx if q.name == name) for name in first]
        second_idx = [next(q[0] for q in second_idx if q.name == name) for name in second]
        emb = torch.cat([self.data[first_idx,:], self.data[second_idx,:]], 1)

        return emb


    def choose_action(self, candi_pool):
        self.initialize()
        actions = []
        self.eval_net.eval()
        for candi in candi_pool:
            candi_emb = self.embeded(candi)
            candi_emb = torch.unsqueeze(candi_emb, 0)

            state_emb = self.embeded(self.curr_state)
            state_emb = torch.unsqueeze(state_emb, 0)

            with torch.no_grad():
                actions_value = self.eval_net(state_emb.to(self.device), candi_emb.to(self.device))
            action = candi[actions_value.argmax().item()]
            actions.append(action)
        return actions


    def update_memory(self, actions, candi_pool, round_num):
        self.initialize()
        results = session.query(SelectPair).filter(
            SelectPair.exp_name == self.config.process['exp_name'],
            SelectPair.round_num == round_num
        ).all()
        results = [next(r for r in results if r.first == action[0] and r.second == action[1]) for action in actions]

        for i in range(len(results)):
            if self.last_transition and i == 0:
                candi = candi_pool[i]
                first_transition = self.last_transition + (candi,)
                self.memory.push(first_transition)

            action = actions[i]
            label_list = [results[i].first_selection, results[i].sec_selection, results[i].none_selection]
            reward = [1,0.5,-1][label_list.index(max(label_list))]

            if i is len(results) - 1:
                self.transition = (self.curr_state, action, reward, self.curr_state + [action])
            else:
                candi = candi_pool[i+1]
                self.memory.push((self.curr_state, action, reward, self.curr_state + [action], candi))
            self.curr_state = self.curr_state + [action]


    def mem_full(self):
        return self.memory.__len__() > self.config.memory_size


    def load_data(self, mmap=True):
        if mmap:
            data = np.load(self.data_path, mmap_mode='r')
        else:
            data = np.load(self.data_path)
        self.data = torch.tensor(data)


    def learn(self, iter):
        self.eval_net.train()
        if iter == 0:
            for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        batch_size = self.config.dqn_args['batch_size']

        transitions = self.memory.sample(self.config.dqn_args['batch_size'])
        batch = Transition(*zip(*transitions))


        lengths = [len(state) for state in list(batch.state)]
        lengths_b_s = torch.tensor(lengths, dtype=torch.int64)
        b_s = self.embeded([x for state in batch.state for x in state])
        b_s = torch.split(b_s, lengths)
        b_s = pad_sequence(b_s, batch_first=True).to(self.device)


        lengths = [len(state) for state in list(batch.next_state)]
        lengths_b_s_ = torch.tensor(lengths, dtype=torch.int64)
        b_s_ = self.embeded([x for next_state in batch.next_state for x in next_state])
        b_s_ = torch.split(b_s_, lengths)
        b_s_ = pad_sequence(b_s_, batch_first=True).to(self.device)


        b_a_emb = torch.unsqueeze(self.embeded(list(batch.action)), 1)


        # reward
        b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1)).to(self.device)


        next_candi_emb = self.embeded([x for candi in list(batch.next_candi) for x in candi])
        next_candi_emb = torch.split(next_candi_emb, self.config.process['candi_num'])
        next_candi_emb = torch.stack(next_candi_emb).to(self.device)


        q_eval = self.eval_net(b_s, b_a_emb, lengths_b_s , choose_action=False)


        if self.double_q:
            index = self.eval_net(b_s_,next_candi_emb, lengths_b_s_).argmax(dim=1)
            best_actions_emb = next_candi_emb[torch.arange(next_candi_emb.size(0)), index, :]
            best_actions_emb = torch.unsqueeze(best_actions_emb, 1)
            q_target = b_r + self.gamma *( self.target_net(b_s_,best_actions_emb, lengths_b_s_,choose_action=False).detach())
        else:
            q_target = b_r + self.gamma*((self.target_net(b_s_,next_candi_emb,lengths_b_s_).detach()).max(dim=1).view(self.batch_size,1))
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"iter: {iter}")
        print(f'loss: {loss.item()}')
