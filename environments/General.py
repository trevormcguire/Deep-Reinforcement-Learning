import numpy as np 
from typing import *
import utils.functional as F 

class Environment(object):
    """
    ------
    PARAMS
    ------
        1. buy_on -> must be open or close
                > Will be open or close T + 1 since this algorithm will be run overnight 
        2. sell_on -> must be open or close
                > Will be open or close T + 1 since this algorithm will be run overnight 
        3. data -> dictionary of ticker: prices
                > 'prices' must be numpy array
        4. rand_start -> if True (default), selects a random starting idx
        5. period -> amount of trading days for agent to consider
        6. reward_function -> Initialized Class from utils.rewards.py to 
                Ex: reward_function=WeightedMean()
    ------
        Note: Ensure OHLC is the first 4 values of last dimension in array passed
    """
    def __init__(self, 
                 data: dict, 
                 buy_on: str, 
                 sell_on: str,
                 period: int,
                 reward_function: Callable,
                 rand_start: bool = True):
        assert type(data) is dict, "data must be a dictionary of ticker: prices where prices is numpy array"
        assert buy_on.lower() in ["open", "close"], f"buy on must be eiter open or close. You passed {buy_on}"
        assert sell_on.lower() in ["open", "close"], f"buy on must be eiter open or close. You passed {sell_on}"
        self.rand_start = rand_start
        self.period = period
        self.data = data
        self.holding_position = False
        self.buy_on = buy_on.lower()
        self.sell_on = sell_on.lower()
        self.action_map = {
            0: "hold", 
            1: "enter", 
            2: "exit"
            }
        self.reward_handler = reward_function
        self.ppt = []
        self.reset()
        self.episode_profits = []

    def reset(self):
        """
        Switch ticker we are working with randomly and reset environment 
        """
        self.ticker = np.random.choice(list(self.data.keys()))
        self.prices = self.data[self.ticker]
        self.len_prices = len(self.prices)
        self.holding_position = False
        self.entry_price, self.exit_price = None, None
        self.curr_idx = np.random.choice(np.arange(self.len_prices - (self.period * 2))) if self.rand_start else 0 

    def get_state(self):
        state = np.array([self.prices[self.curr_idx: self.curr_idx+self.period]]) #shape=(1, self.period, self.num_feats)
        self.curr_ohlc = state[-1][:4] #last row
        self.next_ohlc = self.prices[self.curr_idx+1: self.curr_idx + 1 + self.period][0][:4] #first row
        return state

    def get_next_state(self):
        return np.array([self.prices[self.curr_idx + 1: self.curr_idx + 1 + self.period]]) #shape=(1, self.period, self.num_feats)

    def __get_curr_price(self):
        if self.buy_on == "close":
            return self.next_ohlc[-1]
        return self.next_ohlc[0]


    def step(self, action: int) -> Tuple:
        """
        Terminating conditions:
            1. Exit Position
            2. Reach end of time series
        """
        reward = 0.0
        terminate = False
        action = self.action_map[action]

        if self.holding_position:
            curr_price = self.__get_curr_price()
            curr_profit = (curr_price - self.entry_price) / self.entry_price
            self.ppt.append(curr_profit * 100)
            reward = self.reward_handler.give_reward(self.ppt) #iwm for holds too

        if action == "enter" and not self.holding_position:
            self.entry_price = self.__get_curr_price()
            self.holding_position = True
            print(f"Bought {self.ticker} @ {self.entry_price} ||")

        elif action == "exit" and self.holding_position:
            self.exit_price = curr_price
            self.episode_profits.append(curr_profit)
            terminate = True
            self.holding_position = False
            self.ppt = [] #reset
            print(f"Sold {self.ticker} @ {self.exit_price} || Profit: {curr_profit * 100:.2f}")

        if self.curr_idx + 1 > self.len_prices - self.period: #test if next state will out run the length of our data
            terminate = True
            next_state = None #reached end of timeseries, next is None
        else:
            next_state = self.get_next_state()

        if not terminate:
            self.curr_idx += 1
        
        return reward, next_state, terminate
        