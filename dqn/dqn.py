import random
from collections import namedtuple
import itertools

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from db.session import session
from db.model import SelectPair, Kinetics
from net.qnet import QNet as Net
from net.stgcn_embed import Model

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
        self.config = config
        self.eval_net = Net(candi_num=config.process['candi_num'], emb_size=config.process['out_channels']*2)
        self.target_net = Net(candi_num=config.process['candi_num'], emb_size=config.process['out_channels']*2)
        self.experience = set()
        self.memory = Memory(config.memory_size)
        self.state = []
        self.last_transition = None
        self.double_q = config.dqn_args['double_q']
        self.gamma = config.dqn_args['gamma']
        self.tau = config.dqn_args['tau']
        self.embeddor = Model(**(config.embed_args))
        self.optimizer = optim.Adam(itertools.chain(self.eval_net.parameters(),self.gcn_net.parameters()), lr=config.dqn_args['learning_rate'], weight_decay = config.dqn_args['l2_norm'])
        self.loss_func = nn.MSELoss()
        if torch.cuda.is_available():
            self.embeddor.to(torch.device('cuda'))
            self.embeddor.load_state_dict(torch.load(config.pretrained['embed']))

        self.load_data()


    def initialize(self):
        select_pair = session.query(SelectPair).filter(
            SelectPair.exp_name == self.config.process['exp_name'],
            SelectPair.round_num == 0
        ).all()

        self.cur_state = [(pair.first, pair.second) for pair in select_pair]
        # for pair in select_pair:
            # label_list = [pair.first_selection, pair.sec_selection, pair.none_selection]
            # self.cur_state.append((pair.first, pair.second, label_list.index(max(label_list))))


    def choose_action(self, candi_pool):
        self.state = [self.curr_state]
        actions = []
        for candi in candi_pool:
            first = [c[0] for c in candi]
            second = [c[1] for c in candi]
            first_idx = session.query(Kinetics.index).filter(Kinetics.name.in_(first)).all()
            second_idx = session.query(Kinetics.index).filter(Kinetics.name.in_(second)).all()
            first_idx = [next(q[0] for q in first_idx if q.name == name) for name in first]
            second_idx = [next(q[0] for q in second_idx if q.name == name) for name in second]
            candi_data = [np.array((self.data[f], self.data[s])) for f, s in zip(first_idx, second_idx)]
            candi_emb = [self.embeddor(torch.tensor(d)) for d in candi_data]
            candi_emb = [d.view(-1) for d in candi_emb]
            candi_emb = torch.stack(candi_emb)
            candi_emb = torch.unsqueeze(candi_emb, 0)

            curr_state = self.state[-1]
            first = [s[0] for s in curr_state]
            second = [s[1] for s in curr_state]
            first_idx = session.query(Kinetics.index).filter(Kinetics.name.in_(first)).all()
            second_idx = session.query(Kinetics.index).filter(Kinetics.name.in_(second)).all()
            first_idx = [next(q[0] for q in first_idx if q.name == name) for name in first]
            second_idx = [next(q[0] for q in second_idx if q.name == name) for name in second]
            state_data = [np.array((self.data[f], self.data[s])) for f, s in zip(first_idx, second_idx)]
            state_emb = [self.embeddor(torch.tensor(d)) for d in state_data]
            state_emb = [d.view(-1) for d in state_emb]
            state_emb = torch.stack(state_emb)
            state_emb = torch.unsqueeze(state_emb, 0)

            actions_value = self.eval_net(state_emb, candi_emb)
            action = candi[actions_value.argmax().item()]
            self.state.append(curr_state + [action])
            actions.append(action)
        self.curr_state = self.state[-1]
        return actions


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
            # new_stat = self.curr_state + [action]

            if i is len(results) - 1:
                # self.transition = (self.curr_state, action, reward, new_stat)
                self.transition = (self.state[i], action, reward, self.state[i+1])
            else:
                candi = candi_pool[i+1]
                # self.memory.push((self.curr_state, action, reward, new_stat, candi))
                self.memory.push((self.state[i], action, reward, self.state[i+1], candi))
            # self.curr_state = new_stat


    def mem_full(self):
        return self.memory.__len__() > self.config.memory_size


    def load_data(self, mmap=True):
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)


    def learn(self):
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        self.config.dqn_args['batch_size']

        transitions = self.memory.sample(self.config.dqn_args['batch_size'])
        batch = Transition(*zip(*transitions))

        # state 和 next_state 都是所選物件的 sequence 
        # self.gcn_net 算的是一個 seqnence 的 embedding vector
        b_s = self.gcn_net(list(batch.state))       #[N*max_node*emb_dim]
        b_s_ = self.gcn_net(list(batch.next_state))

        # self.gcn_net.embedding 是一個 nn.Embedding
        b_a = torch.LongTensor(np.array(batch.action).reshape(-1, 1))  #[N*1]
        b_a_emb =self.gcn_net.embedding(b_a)       #[N*1*emb_dim]

        # reward
        b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1))

        # 產生 candi_emb，算 target net 的時候要用
        next_candi = torch.LongTensor(list(batch.next_candi))
        next_candi_emb = self.gcn_net.embedding(next_candi)    #[N*k*emb_dim]

        q_eval = self.eval_net(b_s, b_a_emb,choose_action=False)


        if self.double_q:
            best_actions = torch.gather(input=next_candi, dim=1, index=self.eval_net(b_s_,next_candi_emb).argmax(dim=1).view(self.batch_size,1))
            best_actions_emb = self.gcn_net.embedding(best_actions)
            q_target = b_r + self.gamma *( self.target_net(b_s_,best_actions_emb,choose_action=False).detach())
        else:
            q_target = b_r + self.gamma*((self.target_net(b_s_,next_candi_emb).detach()).max(dim=1).view(self.batch_size,1))
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()