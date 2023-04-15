import sys
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm

from feeder.eval import Feeder
from feeder.ranknet import DatasetFeeder
from net.stgcn_embed import Model
from utils import Base, ngpu


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


class evaluation(Base):
    
    # def __init__(self):
    #     super.__init__()

    def load_data(self):
        self.exp_name = self.arg.process['exp_name']
        self.data_path = self.arg.ranknet_feeder_args['data_path']
        dataset = Feeder(self.data_path, f'static/{self.exp_name}_scores.csv')
        test_dataset = DatasetFeeder(**self.arg.ranknet_feeder_args)
        train_size = int(0.8 * len(dataset))  # 80% for training

        # set the size of the test set
        test_size = len(dataset) - train_size

        # split the dataset into training and test sets using SubsetRandomSampler
        train_sampler = SubsetRandomSampler(range(train_size))
        eval_sampler = SubsetRandomSampler(range(train_size, len(dataset)))
        test_sampler = SubsetRandomSampler(np.random.choice(range(len(test_dataset)), int(0.2 * len(test_dataset))))

        self.train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=self.arg.num_worker * self.ngpus,
            sampler=train_sampler
        )
        self.eval_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=self.arg.num_worker * self.ngpus,
            sampler=eval_sampler
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=32,
            num_workers=self.arg.num_worker * self.ngpus,
            sampler=test_sampler
        )


    def load_model(self):
        self.model = Model(**(self.arg.ranknet_args))
        self.model.apply(weights_init)
        self.model = self.model.to(self.device)
        if self.ngpus > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.loss_func = nn.MSELoss(reduce=True, size_average=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)



    def train(self):
        self.load_data()
        min_loss = sys.float_info.max
        train_loss = []
        test_loss = []
        for i in range(100):
            pbar = tqdm(self.train_loader)
            total_loss = 0.0
            self.model.train()
            for _, (data, label) in enumerate(pbar):
                if torch.cuda.is_available():
                    data = data.float().to(self.device)
                    label = label.float().to(self.device)

                score = self.model(data)
                self.optimizer.zero_grad()
                # print(score.view(-1).size())
                # print(label.size())
                loss = self.loss_func(score.view(-1), label)
                # print(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_loss.append(total_loss / len(self.train_loader))
            print("Training: ", i, " traing loss: ", total_loss / len(self.train_loader))


            total_loss = 0.0
            self.model.eval()
            pbar = tqdm(self.eval_loader)
            for _, (data, label) in enumerate(pbar):
                if torch.cuda.is_available():
                    data = data.float().to(self.device)
                    label = label.float().to(self.device)
                score = self.model(data)

                loss = self.loss_func(score, label)

                total_loss += loss.item()
            test_loss.append(total_loss / len(self.eval_loader))
            print("Testing: ", i, " testing loss: ", total_loss / len(self.eval_loader))

            if total_loss / len(self.eval_loader) <= min_loss:
                min_loss = total_loss / len(self.eval_loader)
            
                torch.save({ 
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict()}, f"static/{self.arg.process['exp_name']}.pt")

        print("Min loss: ",min_loss)
        np.savetxt(f"static/train_{self.exp_name}.csv", np.asarray( train_loss ), delimiter=",")
        np.savetxt(f"static/test_{self.exp_name}.csv", np.asarray( test_loss ), delimiter=",")


    def test(self):
        self.load_data()
        pbar = tqdm(self.test_loader)
        pre_model = torch.load(f"static/{self.arg.process['exp_name']}.pt")
        self.model.load_state_dict(pre_model['model_state_dict'], False)
        self.model = self.model.to(self.device)
        # if self.ngpus > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.eval()
        pbar = tqdm(self.test_loader)

        result = []
        for _, (data, name) in enumerate(pbar):
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = data.float().to(self.device)
                score = self.model(data)
            result = result + [(n,float(s)) for n,s in zip(name, score)]
            # result.append((name, score, label))
        result = pd.DataFrame(result, columns=['name', 'score'])
        result.to_csv(f"static/{self.arg.process['exp_name']}_result.csv", sep='\t')
