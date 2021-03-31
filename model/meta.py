from sklearn.metrics import accuracy_score
import numpy as np


class Ensemble:
    def __init__(self, model_preds, ytest=None):
        self.ytest = ytest
        self.model_preds = model_preds
        self.preds = np.array(self._get_predictions())
        if self.ytest is not None:
            self.acc = accuracy_score(self.ytest, self.preds)
        else:
            self.acc = None
        pass

    def _get_predictions(self):
        preds = [np.argmax(np.bincount(np.array(tup))) for tup in zip(*self.model_preds)]
        return preds
