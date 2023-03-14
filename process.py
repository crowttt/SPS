from candidate.candi_selector import Candidate
from utils import load_arg


class Process:
    def __init__(self, argv=None):
        self.config = load_arg(argv.config)
        self.candidate = Candidate(self.config)

    def start(self):
        for _ in range(self.config.process['round']):
            self.candidate.train()
            self.candidate.ranker()
            candi_pool = self.candidate.candidate()
