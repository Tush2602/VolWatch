import copy 
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn

from models.base import Forecaster

class _LSTMNet(nn.Module):
    def __init__(self, n_features : int, hidden:int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)    # 4 input features per day , 32 hidden units in LSTM
        self.head = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class LSTMForecaster(Forecaster):
    """Read the past window trading days in order and predict Next day volatility"""
    name = "LSTM"
    INPUTS = ["vol", "abs_return", "vix", "vix_change"]
    LOG_COLS = ["vol", "abs_return", "vix"]

    def __init__(self, window:int= 14, hidden : int = 32, lr:float = 1e-3, epochs: int= 30,
                batch_size: int= 64, patience : int= 8, holdout : float= 0.15, seed : int =42):
                self.window =window
                self.hidden =hidden
                self.lr= lr
                self.epochs = epochs
                self.batch_size =batch_size
                self.patience = patience
                self.holdout =holdout
                self.seed =seed
                self.net= None

    #data preparation
    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.INPUTS].copy()
        for col in self.LOG_COLS:
            x[col] = np.log1p(x[col])
        return x.to_numpy(dtype=np.float32)

    def _scale(self, x: np.ndarray)-> np.array:
        return (x - self.x_mean)/self.x_std

    def _windows(self, x:np.ndarray) -> np.ndarray:
        """Row i of the result holds days i-window+1 ... i, so it never looks forward."""
        w = self.window
        return np.stack([x[i-w+1 : i+1] for i in range(w-1, len(x))])
    
    def fit(self, train: pd.DataFrame)-> "LSTMForecaster":
        x_raw = self._prepare(train)
        self.x_mean = x_raw.mean(axis=0)
        self.x_std = x_raw.std(axis=0) + 1e-8

        y = train['target'].to_numpy(dtype=np.float32)
        self.y_mean, self.y_std = float(y.mean()), float(y.std())
        self.min_forecast = float(max(y.min(), 0.1))

        xw = self._windows(self._scale(x_raw))   #shape (87, 14, 4) 87 window size (total_Days - 14+1)
        yw= (y[self.window - 1:] - self.y_mean) / self.y_std   #answer of above created windows   #(87,)
        split  =int(len(xw) * (1 -self.holdout))
        if split < 10 or len(xw) - split < 5:
            raise ValueError("not enough training rows for this window size")
        
        x_tr, y_tr = torch.from_numpy(xw[:split]), torch.from_numpy(yw[:split])
        x_va ,y_va = torch.from_numpy(xw[split:]), torch.from_numpy(yw[split:])

        torch.manual_seed(self.seed)
        net = _LSTMNet(len(self.INPUTS), self.hidden)
        optimizer = torch.optim.Adam(net.parameters(), lr= self.lr)
        loss_fn = nn.MSELoss()

        best_loss , best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(self.epochs):
            net.train()
            order = torch.randperm(len(x_tr))
            for i in range(0, len(order), self.batch_size):
                idx = order[i:1+self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(net(x_tr[idx]), y_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss= loss_fn(net(x_va),y_va).item()
            if val_loss < best_loss -1e-5:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(net.state_dict()), 0
            
            else: 
                bad_epochs+=1
                if bad_epochs >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.epochs_run = epoch+1
        return self

    def predict(self, features:pd.DataFrame) -> pd.Series:
        if self.net is None:
            raise RuntimeError("call fit() before predict()")
        
        out = pd.Series(np.nan, index=features.index, name="forecast")
        if len(features) < self.window:
            return out
        xw = torch.from_numpy(np.ascontiguousarray(self._windows(self._scale(self._prepare(features)))))
        with torch.no_grad():
            pred = self.net(xw).numpy() * self.y_std + self.y_mean
        out.iloc[self.window - 1:] = np.maximum(pred, self.min_forecast)
        return out

