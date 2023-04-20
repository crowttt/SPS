import time
import random
from candidate.candi_selector import Candidate
from dqn.dqn import DQN
from utils import load_arg, init_sample, get_next_round, store_action, dispatch
from loguru import logger

import torch
from db.session import session
from db.model import SelectPair
from config import REDIS_ENGINE
import pickle



class Process:
    def __init__(self, argv=None):
        self.config = load_arg(argv.config)
        init_sample(self.config)
        logger.info('[INIT SAMPLE] Generate intialized sample')
        self.candidate = Candidate(self.config)
        logger.info('[CANDIDATE] Initialize candidate generate')
        self.dqn = DQN(self.config)
        logger.info('[DQN] Initialize deep Q-learning model')

        self.next_round = get_next_round(self.config)
        self.current_round = self.next_round - 1

        if REDIS_ENGINE.get('next') == None:
            REDIS_ENGINE.set('next', 't')
        self.reload()


    def start(self):
        while self.current_round <= int(self.config.process['round']):
            if REDIS_ENGINE.get('next') == b't':
                REDIS_ENGINE.set('next', 'f')
                logger.info(f'[START] Start {self.next_round}th round')
                logger.info(f'[CANDIDATE POOL] Generating candidate pool')
                candidate_pool = self.get_candidate()
                logger.info(f'[CANDIDATE POOL] Done')

                logger.info(f'[ACTION] Choosing actions from candidate pool')
                actions = self.choose_action(candidate_pool)
                logger.info(f'[ACTION] Done')

                logger.info(f'[USER] Waiting for user response ...')
                self.next_round = get_next_round(self.config)
                self.current_round = self.next_round - 1
                logger.info(f'[USER] Done')

                logger.info(f'[MEMORY] Updating memory')
                self.update_memory(actions, candidate_pool)
                logger.info(f'[MEMORY] Done')

            logger.info(f'[DQN] Start to train DQN {self.next_round}th round')
            for itr in range(self.config.process['update_times']):
                self.dqn.learn(itr)
            REDIS_ENGINE.set('next', 't')
            self.init_status()
            
            torch.save({
                'model_state_dict': self.dqn.eval_net.state_dict(),
                'optimizer_state_dict': self.dqn.optimizer.state_dict()}, f'pretrain/dqn/{self.config.process["exp_name"]}.pt')

            logger.info(f'[DQN] Done')


    def get_candidate(self):
        self.candidate.load_data()
        logger.info(f'Start to train ranker')
        self.candidate.train()
        logger.info(f'Ranking all kinetics data')
        self.candidate.ranker()
        candidate_pool = self.candidate.candidate(self.config.process['pair_batch_size'])
        REDIS_ENGINE.set('candi', pickle.dumps(candidate_pool))
        REDIS_ENGINE.set('candi_flag', 't')
        return candidate_pool


    def choose_action(self, candidate_pool, random_mode=False):
        if random_mode:
            actions = [random.choice(candi) for candi in candidate_pool]
        else:
            actions = self.dqn.choose_action(candidate_pool)

        store_action(self.config, actions, self.next_round)
        status = dispatch(self.config, self.next_round)

        while(status != 200):
            status = dispatch(self.config, self.next_round)

        REDIS_ENGINE.set('actions', pickle.dumps(actions))
        REDIS_ENGINE.set('action_flag', 't')
        return actions


    def init_status(self):
        REDIS_ENGINE.set('candi_flag', 'f')
        REDIS_ENGINE.set('action_flag', 'f')
        REDIS_ENGINE.set('memory_flag', 'f')


    def reload(self):
        if REDIS_ENGINE.get('next') == b't':
            return
        logger.info(f'[RELOAD] Go to {self.current_round}th reload process')
        if REDIS_ENGINE.get('candi_flag') == b't':
            logger.info(f'[RELOAD] Reload candiidate')
            candidate_pool = pickle.loads(REDIS_ENGINE.get('candi'))
        else:
            logger.info(f'[CANDIDATE POOL] Generating candidate pool')
            candidate_pool = self.get_candidate()
            logger.info(f'[CANDIDATE POOL] Done')

        if REDIS_ENGINE.get('action_flag') == b't':
            logger.info(f'[RELOAD] Reload action')
            actions = pickle.loads(REDIS_ENGINE.get('actions'))
        else:
            logger.info(f'[ACTION] Choosing actions from candidate pool')
            actions = self.choose_action(candidate_pool)
            logger.info(f'[ACTION] Done')

        logger.info(f'[USER] Waiting for user response ...')
        self.next_round = get_next_round(self.config)
        logger.info(f'[USER] Done')

        if REDIS_ENGINE.get('memory_flag') == b't':
            logger.info(f'[RELOAD] Reload memory')
            self.dqn.memory.load(pickle.loads(REDIS_ENGINE.get('memory')))
        else:
            logger.info(f'[MEMORY] Updating memory')
            self.update_memory(actions, candidate_pool)
            logger.info(f'[MEMORY] Done')


    def update_memory(self, actions, candidate_pool):
        self.dqn.update_memory(actions, candidate_pool, self.current_round)
        REDIS_ENGINE.set('memory', pickle.dumps(self.dqn.memory.all()))
        REDIS_ENGINE.set('memory_flag', 't')     
