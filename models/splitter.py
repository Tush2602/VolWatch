from dataclasses import dataclass
import pandas as pd 

@dataclass
class Splits:
    train : pd.DataFrame
    test : pd.DataFrame
    valid : pd.DataFrame

class Timesplitter:
    def __init__(self, train_end : str = '2017-12-31', valid_end : str='2019-12-31'):
        self.train_end = train_end
        self.valid_end = valid_end

    def split(self, table : pd.DataFrame)-> Splits:
        labeled = table.dropna(subset=['target'])
        idx = labeled.index

        train = labeled[idx <= self.train_end]
        valid = labeled[(idx > self.train_end) & (idx <= self.valid_end)]
        test = labeled[(idx > self.valid_end)]

        return Splits(train=train.iloc[:-1], test=test, valid = valid.iloc[:-1])
