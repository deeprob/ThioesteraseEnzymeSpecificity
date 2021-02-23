#import modules

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm


class GBC:
    
    def __init__(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,random_seed=None,pca_comp=20,nest=15,lrate=0.1,mdepth=3,ssample=1,optimize=False,verbose=True):
        np.random.seed(random_seed)
        
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.ytrain = ytrain
        self.yvalid = yvalid
        
        pipeline = self._make_pipeline(pca_comp,nest,lrate,mdepth,ssample)
        
        self.model = pipeline        
        self.model.fit(self.Xtrain,self.ytrain) 
        
        self.ypredtrain = self.model.predict(self.Xtrain)
        self.ypredvalid = self.model.predict(self.Xvalid)
        self.acc_train = accuracy_score(self.ytrain,self.ypredtrain)
        self.acc_valid = accuracy_score(self.yvalid,self.ypredvalid)
        
        
        if verbose:
            print('-'*5+'Initial Model Evaluation'+'-'*5)
            print('-'*5+'Training Accuracy:'+str(self.acc_train)+'-'*5)
            print('-'*5+'Testing Accuracy:'+str(self.acc_valid)+'-'*5)
        
        # Hyperparameter Optimization
        
        if optimize:
            if verbose:
                print('-'*5+'Hyperparameter Optimization'+'-'*5)

            if self.Xtrain.shape[1]<75:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5*shape),int(0.6*shape),int(0.75*shape)]
            else:
                try_pca= [40,55,75]


            parameters = {'pca__n_components':try_pca,
                         'GBC__n_estimators':[15,25,100],
                         'GBC__learning_rate':[0.1,0.5,1],
                         'GBC__max_depth':[1,3,5]}

            self.grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,scoring='accuracy',verbose=10)
            self.grid.fit(self.Xtrain, self.ytrain)
            
            # print evaluation results

            if verbose:

                print("score = %3.2f" %(self.grid.score(self.Xvalid,self.yvalid)))

                print(self.grid.best_params_)
            
            best_pipeline = self.grid.best_estimator_
            
            self.model = best_pipeline
        
            self.ypredtrain = self.model.predict(self.Xtrain)
            self.ypredvalid = self.model.predict(self.Xvalid)
            self.acc_train = accuracy_score(self.ytrain,self.ypredtrain)
            self.acc_valid = accuracy_score(self.yvalid,self.ypredvalid)

        if Xtest:
            self.X=np.concatenate((self.Xtrain,self.Xvalid),axis=0)
            self.y=np.concatenate((self.ytrain,self.yvalid),axis=0)
            self.model.fit(self.X,self.y)
            self.yhattrain = self.model.predict(self.X)
            self.yhattest = self.model.predict(self.Xtest)
            self.acc_tr = accuracy_score(self.y,self.yhattrain)

        
    def _make_pipeline(self,n_comp,n,lr,md,ss):
        steps = [('normalize',Normalizer()),('pca',PCA(n_components=n_comp)),('GBC',GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=md,subsample=ss))]
        pipe = Pipeline(steps)
        return pipe
