from abc import ABC, abstractmethod
import yaml


class Base(ABC):
    def __init__(self, argv):
        self.load_arg(argv.config)
        self.load_model()
        self.load_data()


    def load_arg(self, config):

        if config is not None:
            # load config file
            with open(config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)


    @abstractmethod
    def load_model(self):
        pass


    @abstractmethod
    def load_data(self):
        pass


    @abstractmethod
    def train(self):
        pass
