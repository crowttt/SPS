from net.st_gcn import Model
from utils import Base
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
                num_workers=self.arg.num_worker * ngpu(
                    self.arg.device),
                drop_last=True
            )

            self.unlabel_data_loader = torch.utils.data.DataLoader(
                dataset=DatasetFeeder(**self.arg.ranknet_feeder_args),
                batch_size=self.arg.ranknet_train_arg['batch_size'],
                shuffle=False,
                num_workers=self.arg.num_worker * ngpu(
                    self.arg.device),
                drop_last=True
            )
    
    def load_model(self):
        self.model = Model(**(self.arg.ranknet_args))
        self.model.apply(weights_init)
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.arg.ranknet_train_arg['base_lr'])

    def train(self):
        for _ in range(self.arg.ranknet_train_arg['num_epoch']):
            for input1, input2, y in self.data_loader:
                score1 = self.model(input1)
                score2 = self.model(input2)
                y_pred = torch.sigmoid(score1 - score2)

                self.optimizer.zero_grad()
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()

    def ranker(self):
        self.rank = []
        for data, name in self.unlabel_data_loader:
            score = self.model(data)
            self.rank = self.rank + [(n,float(s)) for n,s in zip(name,score)]
        self.rank = sorted(self.rank, key=lambda x: x[1])


    def candidate(self, k):
        top_k = self.rank[:k]
        last_k = self.rank[-k:]
        res = [[i[0], j[0]] for i in top_k for j in last_k]
        return res
