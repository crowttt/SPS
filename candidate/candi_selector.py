import sys
import random
import pickle
import torch
import torch.nn as nn 
from tqdm import tqdm
from net.st_gcn import Model
from utils import Base, ngpu
from feeder.ranknet import Feeder, DatasetFeeder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Candidate(Base):

    def load_data(self):
        if self.arg.ranknet_feeder_args:
            self.data_loader = torch.utils.data.DataLoader(
                dataset=Feeder(self.arg.process['exp_name'], **self.arg.ranknet_feeder_args),
                batch_size=self.arg.ranknet_train_arg['batch_size'],
                shuffle=True,
                num_workers=self.arg.num_worker * self.ngpus,
                drop_last=True
            )

            self.unlabel_data_loader = torch.utils.data.DataLoader(
                # dataset=DatasetFeeder(**self.arg.ranknet_feeder_args),
                dataset=DatasetFeeder(
                    data_path=self.arg.ranknet_feeder_args['data_path'],
                    label_path=self.arg.ranknet_feeder_args['label_path']
                ),
                batch_size=self.arg.ranknet_train_arg['batch_size'] * 32,
                shuffle=True,
                num_workers=self.arg.num_worker * self.ngpus,
                drop_last=True
            )
    
    def load_model(self):
        self.model = Model(**(self.arg.ranknet_args))
        self.model.apply(weights_init)
        self.model = self.model.to(self.device)
        if self.ngpus > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.arg.ranknet_train_arg['base_lr'])

    def train(self):
        min_loss = sys.float_info.max
        for i in range(self.arg.ranknet_train_arg['num_epoch']):
            pbar = tqdm(self.data_loader)
            total_loss = 0.0
            self.model.train()
            for _, (data, y) in enumerate(pbar):
                data = torch.split(data, 1, dim=1)
                if torch.cuda.is_available():
                    input1 = torch.squeeze(data[0], dim=1).float().to(self.device)
                    input2 = torch.squeeze(data[1], dim=1).float().to(self.device)
                    y = y.float().to(self.device)

                score1 = self.model(input1)
                score2 = self.model(input2)
                y_pred = torch.squeeze(torch.sigmoid(score1 - score2), dim=1)
                self.optimizer.zero_grad()
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print("Training: ", i, " traing loss: ", total_loss / len(self.data_loader))
            if total_loss / len(self.data_loader) <= min_loss:
                min_loss = total_loss / len(self.data_loader)
                torch.save({ 
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, 'pretrain/ranknet/ranknet.pt')

    def ranker(self):
        self.rank = []
        pbar = tqdm(self.unlabel_data_loader)
        self.model = Model(**(self.arg.ranknet_args))
        pre_model = torch.load(self.arg.pretrained['ranknet'])
        self.model.load_state_dict(pre_model['model_state_dict'], False)
        self.model = self.model.to(self.device)
        if self.ngpus > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.eval()
        for _, (data, name) in enumerate(pbar):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = data.float().to(self.device)
                score = self.model(data)
            self.rank = self.rank + [(n,float(s)) for n,s in zip(name,score)]
        self.rank = sorted(self.rank, key=lambda x: x[1])
        self.rank = [x[0] for x in self.rank]
        # with open('./data/Kinetics/train_label.pkl', 'rb') as f:
        #     self.rank, _ = pickle.load(f)
        # self.rank = [x[:-5] for x in self.rank]


    def candidate(self, k):
        top_k = self.rank[:k]
        last_k = self.rank[-k:]
        permutation = [(i, j) for i in top_k for j in last_k]

        res = []
        for i in range(k):
            candi = random.sample(permutation, self.arg.process['candi_num'])
            permutation = list(set(permutation) - set(candi))
            res.append(candi)

        return res
