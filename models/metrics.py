import numpy as np 

class Metrics:
    FLOOR = 1e-4   # to avoid divide by zero error 
    @staticmethod
    def rmse(actual, forecast) -> float:
        a,f = np.asarray(actual, float), np.asarray(forecast, float)
        return float(np.sqrt(np.mean((a-f)**2)))

    @staticmethod
    def mae(actual, forecast) -> float:
        a, f = np.asarray(actual, float), np.asarray(forecast, float)
        return float(np.mean(np.abs(a-f)))

    @classmethod
    def qlike(cls, actual, forecast)-> float:
        a = np.maximum(np.asarray(actual, float), cls.FLOOR)
        f = np.maximum(np.asarray(forecast, float), cls.FLOOR)
        ratio = (a/f)**2
        return float(np.mean(ratio - np.log(ratio) - 1))
    
    @classmethod
    def summarize(cls, actual, forecast)-> float:
        return {
            "rmse" : cls.rmse(actual, forecast),
            'mae': cls.mae(actual, forecast),
            'qlike' : cls.qlike(actual, forecast)
        }
    