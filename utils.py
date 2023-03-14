from abc import ABC, abstractmethod
import yaml


class Base(ABC):
    def __init__(self, arg):
        self.arg = arg
        # self.load_arg(argv.config)
        self.load_model()
        self.load_data()


    # def load_arg(self, config):
    #     parser = argparse.ArgumentParser()
    #     if config is not None:
    #         # load config file
    #         with open(config, 'r') as f:
    #             default_arg = yaml.load(f, Loader=yaml.FullLoader)

    #         parser.set_defaults(**default_arg)
    #         arg = parser.parse_args()
    #         self.arg = arg


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


def init_sample():
    pass
