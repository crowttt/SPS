import time
import random
from candidate.candi_selector import Candidate
from dqn.dqn import DQN
from utils import load_arg, init_sample, next_round, store_action
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
        self.dqn = DQN(self.config)
        logger.info('[DQN] Initialize deep Q-learning model')

        self.current_round = next_round(self.config)
        logger.info(f'[ROUND] {self.current_round}th round')


    def start(self):
        while self.current_round < int(self.config.process['round']):
            if self.current_round == 1:
                self.dqn.initialize()
            candidate_pool = self.get_candidate()
            logger.info(f'[CANDIDATE POOL] Generate candidate pool')

            actions = self.choose_action(candidate_pool)
            logger.info(f'[ACTION] Choose actions from candidate pool')

            next_round = next_round(self.config)
            logger.info(f'[NEXT ROUND] Next round is {self.current_round}th round')

            self.dqn.update_memory(actions, candidate_pool, self.current_round)
            self.dqn.learn()
            self.current_round = next_round


    def get_candidate(self):
        self.candidate.load_data()
        logger.info(f'Start to train ranker')
        self.candidate.train()
        logger.info(f'Ranking all kinetics data')
        self.candidate.ranker()
        return self.candidate.candidate(self.config.process['pair_batch_size'])


    def choose_action(self, candidate_pool, random_mode=False):
        num_to_select = self.config.process['pair_batch_size']

        if random_mode:
            action = [random.choice(candi) for candi in candidate_pool]
        else:
            actions = dqn.choose_action()

        store_action(self.config, action, self.current_round)
        dispatch(self.config, self.current_round)

        return action
