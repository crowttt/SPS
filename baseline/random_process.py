from utils import load_arg, init_sample, get_next_round

from db.session import session
from db.model import SelectPair


class RandomProcess:
    def __init__(self, argv=None):
        self.config = load_arg(argv.config)

    def start(self):
        init_sample(self.config)
        # get_next_round(self.config)