from sklearn.metrics import accuracy_score
import numpy as np

class Ensemble:
    def __init__(self,models,Xtests,ytest):
        self.all_models = models
        self.Xtests = Xtests
        self.ytest = ytest
        
        self.preds = np.array(self._get_predictions())
        self.acc = accuracy_score(self.ytest,self.preds)
        pass
    
    def _get_predictions(self):
        predictions = [m.predict(Xtest) for m,Xtest in zip(self.all_models,self.Xtests)]
        preds = [np.argmax(np.bincount(np.array(tup))) for tup in zip(*predictions)]
        return preds
        