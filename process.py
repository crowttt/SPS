import time
import random
from candidate.candi_selector import Candidate
from utils import load_arg, init_sample, next_round, store_sample
from loguru import logger

from db.session import session
from db.model import SelectPair



class Process:
    def __init__(self, argv=None):
        self.config = load_arg(argv.config)
        self.current_round = 0
        init_sample(self.config)
        logger.info('[INIT SAMPLE] Generate intialized sample')
        self.candidate = Candidate(self.config)
        logger.info('[CANDIDATE] Initialize candidate generate')

    def start(self):
        while self.current_round < int(self.config.process['round']):
            logger.info(f'[ROUND] {self.current_round}th round')
            self.current_round = next_round(self.config)
            candidate_pool = self.get_candidate()
            logger.info(f'[CANDIDATE POOL] Generate candidate pool')
            self.picker(candidate_pool)
                
    def get_candidate(self):
        self.candidate.load_data()
        logger.info(f'Start to train ranker')
        self.candidate.train()
        logger.info(f'Ranking all kinetics data')
        self.candidate.ranker()
        return self.candidate.candidate(self.config.process['pair_batch_size'])


    def picker(self, candidate_pool):
        num_to_select = self.config.process['pair_batch_size']
        sample_to_push = random.sample(candidate_pool, num_to_select)

        store_sample(self.config, sample_to_push, self.current_round)
        dispatch(self.config, self.current_round)
