from collections import deque
from typing import *
import numpy as np 
import random
from models.MultiTimeFrame_10_20_60 import Model
import torch
from torch import nn


class Agent(object):
    def __init__(self, 
                 num_feats: int, 
                 num_actions: int, 
                 max_replay_mem: int, 
                 target_update_thresh: int, 
                 replay_batch_size: int, 
                 timeframes: List = [10,20,60],
                 gamma: float = 0.5):
        self.main_net = Model(num_features=num_feats, num_actions=num_actions, timeframes=timeframes)
        self.target_net = Model(num_features=num_feats, num_actions=num_actions, timeframes=timeframes)

        self.target_update_ctr = 0
        self.target_update_thresh = target_update_thresh

        self.replay_memory = deque(maxlen=max_replay_mem) #make quite large since we're mixing up the num of stocks 4K? 5k?
        # self.begin_replay_at = begin_replay_at #min mem size to start replaying
        #     #above is important because we're going to sample way too much from beginning
        self.replay_batch_size = replay_batch_size

        self.num_actions = num_actions
        

        self.eps = 1.0
        self.eps_decay = 0.995
        self.min_eps = 0.01
        self.gamma = gamma
        self.inventory = []
        
    def transfer_weights(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def action(self, state: List[np.ndarray]):
        if random.random() <= self.eps:
            return random.randrange(self.num_actions)
        return self.main_net.predict(state, return_logits=False) #sample from a probability density instead???

    def replay(self):
        if len(self.replay_memory) < self.replay_batch_size: #skip if not enough samples
            return

        memory_samples = random.sample(self.replay_memory, self.replay_batch_size) #randomly sample (n = batch_size) elements from memory

        s = np.vstack([x[0] for x in memory_samples]) #(batch, context_pd, feats)
        a = np.array([x[1] for x in memory_samples])
        r = np.array([x[2] for x in memory_samples])
        s_prime = np.vstack([x[3] for x in memory_samples]) #(batch, context_pd, feats)
        done_mask = np.array([x[4] for x in memory_samples])



        if self.num_actions == 2:
            a[a > 0] = 1

        target_rewards = torch.Tensor(r) + self.gamma * torch.max(self.target_net.predict(s_prime,
                                                                               return_logits=True), dim=1).values


        target_rewards[done_mask] = torch.Tensor(r[done_mask])  

        curr_pred = self.main_net.predict(s, return_logits=True).numpy()
        curr_pred[np.arange(len(a)), a] = target_rewards
        self.main_net.train(1, s, curr_pred)


        self.target_update_ctr += 1
        if self.target_update_ctr >= self.target_update_thresh:
            self.transfer_weights()
            self.target_update_ctr = 0
    
    def decay_epsilon(self): #end of every episode
        if self.eps > self.min_eps:
            self.eps *= self.eps_decay

    def save_model(self, path: str):
        if path.find(".pt") == -1:
            path = path + ".pt"
        torch.save(self.main_net.state_dict(), path)
        print(f"Saved main model at {path}")

    def load_model(self, path: str):
        if path.find(".pt") == -1:
            path = path + ".pt"
        self.main_net.load_state_dict(torch.load(path))
    