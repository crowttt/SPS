import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class QNet(nn.Module):
    def __init__(self, candi_num, emb_size, x_dim=800, state_dim=800, hidden_dim=800, layer_num=1):
        super(QNet, self).__init__()
        #self.duling = duling
        self.candi_num = candi_num
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        #V(s)
        self.fc2_value = nn.Linear(hidden_dim, hidden_dim)
        self.out_value = nn.Linear(hidden_dim, 1)
        #Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_dim+emb_size, hidden_dim)   #hidden_dim + emb_size
        self.out_advantage = nn.Linear(hidden_dim,1)


    def forward(self, x,y,lengths_x=None, choose_action = True):
        """
        L:32
        D:50
        K:200
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        if lengths_x is not None:
            x = pack_padded_sequence(x, lengths_x.to('cpu'), batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        x = F.relu(self.fc1(h))
        #v(s)
        value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2) #[N*1*1]
        #Q(s,a)
        if choose_action:
            x = x.repeat(1,self.candi_num,1)
        state_cat_action = torch.cat((x,y),dim=2)
        advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2) #[N*K]

        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
        else:
            qsa = advantage + value

        return qsa