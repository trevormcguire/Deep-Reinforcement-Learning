from typing import *
import numpy as np
import torch
from torch import nn
from utils import functional as F 

class Model(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 num_actions: int, 
                 timeframes: List[int] = [10,20,60], 
                 detect_important_levels: bool = True):
        super(Model, self).__init__()
        self.num_gru_layers = 2
        self.detect_important_levels = detect_important_levels
        self.num_features = num_features
        if detect_important_levels:
            self.num_features += 4

        self.timeframes = timeframes

        self.gru_small = nn.GRU(self.num_features, 32, self.num_gru_layers, batch_first=True)
        self.gru_med = nn.GRU(self.num_features, 64, self.num_gru_layers, batch_first=True)
        self.gru_large = nn.GRU(self.num_features, 128, self.num_gru_layers, batch_first=True)

        self.dense_small = nn.Linear(32, 16)
        self.dense_med = nn.Linear(64, 16)
        self.dense_large = nn.Linear(128, 16)

        self.dense_out = nn.Linear(16*3, num_actions)
        self.relu = nn.ReLU()

        #get rid of the below--we have to calculate loss manually
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def __ensure_is_tensor(self, 
                           matrix: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return torch.Tensor(matrix) if type(matrix) is np.ndarray else matrix

    def __ensure_is_tensor_list(self, 
                                feat_list: List[Union[np.ndarray, torch.Tensor]]) -> List[torch.Tensor]:
        return [self.__ensure_is_tensor(x) for x in feat_list]
    
    def __split_timeframes(self, X: np.ndarray) -> List[np.ndarray]:
        states = []
        for t in self.timeframes:
            states.append(F.normalize3D(X[:,-t:])) #take most (t) most recent idxs for each timeframe
        return states

    def forward(self, X: Tuple[Union[np.ndarray, torch.Tensor]]):
        X = self.__split_timeframes(X)
        if self.detect_important_levels:
            X = F.stack_regression(X)
        
        smallX, medX, largeX = self.__ensure_is_tensor_list(X)

        out_small, _ = self.gru_small(smallX, torch.zeros(self.num_gru_layers, smallX.size(0), 32))
        
        out_med, _ = self.gru_med(medX, torch.zeros(self.num_gru_layers, medX.size(0), 64))
        
        out_large, _ = self.gru_large(largeX, torch.zeros(self.num_gru_layers, largeX.size(0), 128))

        out_small = out_small[:, -1, :]
        out_med = out_med[:, -1, :]
        out_large = out_large[:, -1, :]

        out_small = self.relu(self.dense_small(out_small))
        out_med = self.relu(self.dense_med(out_med))
        out_large = self.relu(self.dense_large(out_large))

        out = torch.cat([out_small, out_med, out_large], dim=1)
        out = self.dense_out(out)
        return out

    def predict(self, 
                X: Tuple[Union[np.ndarray, torch.Tensor]],
                return_logits: bool = False) -> torch.Tensor:
        with torch.no_grad():
            logits = self(X)
            if return_logits:
                return logits
            return torch.argmax(logits).item() 

    def train(self, 
              num_epochs: int, 
              X: Tuple[Union[np.ndarray, torch.Tensor]],
              y: Union[np.ndarray, torch.Tensor]):
        
        y = self.__ensure_is_tensor(y)

        for e in range(num_epochs):
            self.optimizer.zero_grad()
            yhat = self(X) #(batch, num_actions)
            loss = self.criterion(yhat, y)
            loss.backward()
            self.optimizer.step()