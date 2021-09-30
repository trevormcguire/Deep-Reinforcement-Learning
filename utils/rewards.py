import numpy as np
from typing import *
import utils.functional as F 

class Reward(object):
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None, tanh: bool = True) -> float:
        if profit == None:
            profit = np.sum(ppt)
        if tanh:
            return np.tanh(profit)
        return profit
    
class WeightedMean(Reward):
    """
    -------
    Takes a array of profits per tick and applies a weighted or (inverse weighted) 
        transformation to get the reward.
    -------
        1. weighted mean -> Tail values will be weighed more
        2. inverse weighted mean -> Head values will be weighed more
    -------
    """
    def __init__(self, pos="inverse weighted", neg="weighted"):
        self.func_map = {
            "inverse weighted":F.inv_weighted_mean, 
            "weighted": F.weighted_mean,
            }
        allowable_params = list(self.func_map.keys())
        assert pos in allowable_params and neg in allowable_params,\
        f"params 'pos' and 'neg' must be one of the following {allowable_params}"
        self.pos = pos
        self.neg = neg
    
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        if profit < 0:
            return np.tanh(self.func_map[self.neg](ppt))
        elif profit > 0:
            return np.tanh(self.func_map[self.pos](ppt))
        return 0

class AdjustedWeightedMean(Reward):
    """
    -------
    Takes a array of profits per tick and applies a weighted or (inverse weighted) 
        transformation to get the reward.
    -------
        1. weighted mean -> Tail values will be weighed more
        2. inverse weighted mean -> Head values will be weighed more
    -------
    """
    def __init__(self, pos="inverse weighted", neg="weighted"):
        self.func_map = {
            "inverse weighted":F.inv_weighted_mean, 
            "weighted": lambda profit: np.sum(profit),
            }
        allowable_params = list(self.func_map.keys())
        assert pos in allowable_params and neg in allowable_params,\
        f"params 'pos' and 'neg' must be one of the following {allowable_params}"
        self.pos = pos
        self.neg = neg
    
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        if profit < 0:
            return np.tanh(self.func_map[self.neg](ppt))
        elif profit > 0:
            return np.tanh(self.func_map[self.pos](ppt))
        return 0


class MDD(Reward):
    """
    Averages profit with Maximum Drawdown
    """
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        return np.tanh((profit + np.min(ppt)) / 2)

class MeanCumPPT(Reward):
    """
    Mean Cumulative Sum of PPT
    """
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        return np.tanh(np.mean(np.cumsum(ppt)))

class Sharpe(Reward):
    """
    Sharpe Ratio
    """
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        return np.tanh(np.sum(ppt) / (np.std(ppt) * 2))


class AdjSharpe(Reward):
    """
    Adjusted Sharpe Ratio
        -> Only considers drawdowns for std calculation 
    """
    def give_reward(self, ppt: np.ndarray, profit: Union[float, int] = None) -> float:
        profit = super().give_reward(ppt, profit, tanh=False)
        ppt = np.array(ppt)
        neg_dd = ppt[ppt < 0]
        if len(neg_dd) > 0:
            return np.sum(ppt) / (np.std(neg_dd) * 2)
        return np.tanh(np.sum(ppt))


