from net.st_gcn import Model
from utils.base import Base
from feeder.ranknet import Feeder


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
                dataset=Feeder(**self.arg.ranknet_feeder_args),
                batch_size=self.arg.ranknet_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * ngpu(
                    self.arg.device),
                drop_last=True
            )
    
    def load_model(self):
        self.model = Model(**(self.arg.ranknet_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        pass

