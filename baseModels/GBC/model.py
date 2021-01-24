#import modules

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm


class GBC:
    
    def __init__(self,Xtrain,Xtest,ytrain,ytest,random_seed=None,pca_comp=20,nest=15,lrate=0.1,mdepth=3,ssample=1,optimize=False,verbose=True):
        np.random.seed(random_seed)
        
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
        pipeline = self._make_pipeline(pca_comp,nest,lrate,mdepth,ssample)
        
        self.model = pipeline.fit(self.Xtrain,self.ytrain)
        self.ypredtrain = self.model.predict(self.Xtrain)
        self.ypredtest = self.model.predict(self.Xtest)
        self.acc_train = accuracy_score(self.ytrain,self.ypredtrain)
        self.acc_test = accuracy_score(self.ytest,self.ypredtest)
        
        
        if verbose:
            print('-'*5+'Initial Model Evaluation'+'-'*5)
            print('-'*5+'Training Accuracy:'+str(self.acc_train)+'-'*5)
            print('-'*5+'Testing Accuracy:'+str(self.acc_test)+'-'*5)
        
        # Hyperparameter Optimization
        
        if optimize:
            print('-'*5+'Hyperparameter Optimization'+'-'*5)

            parameters = {'pca__n_components':[1,5,20,40],
                         'GBC__n_estimators':[15,25,100],
                         'GBC__learning_rate':[0.1,0.5,1],
                         'GBC__max_depth':[1,3,5]}

            self.grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,scoring='accuracy',verbose=10)
            self.grid.fit(self.Xtrain, self.ytrain)
            
            # print evaluation results

            print("score = %3.2f" %(self.grid.score(self.Xtest,self.ytest)))

            print(self.grid.best_params_)
            
            best_pipeline = self.grid.best_estimator_
            
            self.model = best_pipeline
        
                
        
    def _make_pipeline(self,n_comp,n,lr,md,ss):
        steps = [('pca',PCA(n_components=n_comp)),('GBC',GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=md,subsample=ss))]
        pipe = Pipeline(steps)
        return pipe
