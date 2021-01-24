from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np

class Ensemble:
    def __init__(self,modelpreds,ytest):
        self.ytest = ytest
        self.modelpreds = modelpreds
        self.preds = np.array(self._get_predictions())
        self.acc = accuracy_score(self.ytest,self.preds)
        pass
    
    def _get_predictions(self):
        preds = [np.argmax(np.bincount(np.array(tup))) for tup in zip(*self.modelpreds)]
        return preds

class EnsembleRegression:
    def __init__(self,modelpreds,ytest):
        self.ytest = ytest
        self.modelpreds = modelpreds
        self.preds = np.array(self._get_predictions())
        self.mse = mean_squared_error(self.ytest,self.preds)
        pass
    
    def _get_predictions(self):
        preds = [np.mean(tup) for tup in zip(*self.modelpreds)]
        return preds
 