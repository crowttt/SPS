import random
from collections import namedtuple

import torch
import torch.nn as nn

from db.session import session
from db.model import SelectPair

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_candi'))

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition: tuple):
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*transition))
            return True
        else:
            return False
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, config):
        self.eval_net, self.target_net = Net(config), Net(config)
        self.experience = set()
        self.memory = Memory(config.memory_size)
        self.curr_state = []
        self.config = config
        self.last_transition = None


    def initialize(self):
        select_pair = session.query(SelectPair).filter(
            SelectPair.exp_name == config.process['exp_name']
        ).all()

        for pair in select_pair:
            label_list = [pair.first_selection, pair.sec_selection, pair.none_selection]
            self.cur_state.append((pair.first, pair.second, label_list.index(max(label_list)))


    def choose_action(self, candi_pool):
        # self.action = ?
        pass


    def update_memory(self, actions, candi_pool, round_num):
        candi_pool = {action: candi for action, candi in zip(actions, candi_pool)}
        results = session.query(SelectPair).filter(
            SelectPair.exp_name == self.config.process['exp_name'],
            SelectPair.round_num == round_num
        ).all()
        results = [next(r for r in results if r.first == action[0] and r.second == action[1]) for action in actions]
        
        for i in len(results):
            if self.last_transition and i is 0:
                candi = candi_pool[i]
                first_transition = self.last_transition + (candi,)
                self.memory.push(first_transition)

            action = actions[i]
            label_list = [results[i].first_selection, results[i].sec_selection, results[i].none_selection]
            reward = label_list.index(max(label_list))
            new_stat = self.curr_state + [(action[0], action[1], reward)]

            if i is len(results) - 1:
                self.transition = (self.curr_state, action, reward, new_stat)
            else:
                candi = candi_pool[i+1]
                self.memory.push((self.curr_state, action, reward, new_stat, candi))
            self.curr_state = new_stat


    def learn(self):
    	pass


    def mem_full(self):
        return self.memory.__len__() > config.memory_size