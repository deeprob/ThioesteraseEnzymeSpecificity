#import modules

import pandas as pd 
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline


class SVM:
    
    def __init__(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,random_seed=None,pca_comp=40,regC=1,kern='rbf',probability=False,optimize=False,verbose=True,classweight=None,multi_jobs=True):
        np.random.seed(random_seed)
        
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.ytrain = ytrain
        self.yvalid = yvalid
        
        pipeline = self._make_pipeline(pca_comp,regC,kern,random_seed,probability,classweight)
        
        self.model = pipeline
        
        self.model.fit(self.Xtrain,self.ytrain) 
        
        self.ypredtrain = self.model.predict(self.Xtrain)
        self.ypredvalid = self.model.predict(self.Xvalid)
        self.acc_train = accuracy_score(self.ytrain,self.ypredtrain)
        self.acc_valid = accuracy_score(self.yvalid,self.ypredvalid)
        
        if verbose:
            print('-'*5+'Initial Model Evaluation'+'-'*5)
            print('-'*5+'Training Accuracy:'+str(self.acc_train)+'-'*5)
            print('-'*5+'Validation Accuracy:'+str(self.acc_valid)+'-'*5)
        
        # Hyperparameter Optimization
        
        if optimize:
            if verbose:
                print('-'*5+'Hyperparameter Optimization'+'-'*5)
            
            if self.Xtrain.shape[1]<55:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5*shape),int(0.6*shape)]
            else:
                try_pca= [40,55]

            parameters = {'pca__n_components':try_pca,
                         'SVM__C':[0.1,1,20,30],
                         'SVM__kernel':['linear','rbf']}
            
            if multi_jobs:
                self.n_jobs=-1
            else:
                self.n_jobs=1

            self.grid = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=self.n_jobs,scoring='accuracy',verbose=0)
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
            
        if Xtest is not None:
            self.Xtest=Xtest
            self.X=np.concatenate((self.Xtrain,self.Xvalid),axis=0)
            self.y=np.concatenate((self.ytrain,self.yvalid),axis=0)
            self.model.fit(self.X,self.y)
            self.yhattrain = self.model.predict(self.X)
            self.yhattest = self.model.predict(self.Xtest)
            self.acc_tr = accuracy_score(self.y,self.yhattrain)
            
            pass
            

                
        
    def _make_pipeline(self,n_comp,c,k,rs,prob,cw):
        steps = [('normalize',Normalizer()),('pca',PCA(n_components=n_comp,random_state=rs)),('SVM',SVC(C=c,gamma='scale',kernel=k,random_state=rs,max_iter=-1,probability=prob,class_weight=cw))]
        pipe = Pipeline(steps)
        return pipe

class SVMRegressor:
    
    def __init__(self,Xtrain,Xvalid,ytrain,yvalid,Xtest,random_seed=None,pca_comp=20,regC=1,kern='rbf',optimize=False,verbose=True):
        np.random.seed(random_seed)
        
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.ytrain = ytrain
        self.yvalid = yvalid
        
        pipeline = self._make_pipeline(pca_comp,regC,kern,random_seed)
        
        self.model = pipeline.fit(self.Xtrain,self.ytrain)
        
        self.ypredtrain = self.model.predict(self.Xtrain)
        self.ypredvalid = self.model.predict(self.Xvalid)
        self.error_train = mean_squared_error(self.ytrain,self.ypredtrain)
        self.error_valid = mean_squared_error(self.yvalid,self.ypredvalid)
        
        
        if verbose:
            print('-'*5+'Initial Model Evaluation'+'-'*5)
            print('-'*5+'Training Accuracy:'+str(self.error_train)+'-'*5)
            print('-'*5+'Testing Accuracy:'+str(self.error_valid)+'-'*5)
        
        # Hyperparameter Optimization
        
        if optimize:
            print('-'*5+'Hyperparameter Optimization'+'-'*5)

            if self.Xtrain.shape[1]<75:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5*shape),int(0.6*shape),int(0.75*shape)]
            else:
                try_pca= [40,55,75]

            parameters = {'pca__n_components':try_pca,
                         'SVM__C':[0.1,1,20,30],
                         'SVM__kernel':['linear','rbf','sigmoid']}

            self.grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,scoring='neg_mean_squared_error',verbose=10)
            self.grid.fit(self.Xtrain, self.ytrain)
            
            # print evaluation results

            print("score = %3.2f" %(self.grid.score(self.Xvalid,self.yvalid)))

            print(self.grid.best_params_)
            
            best_pipeline = self.grid.best_estimator_
            
            self.model = best_pipeline
        
            self.ypredtrain = self.model.predict(self.Xtrain)
            self.ypredvalid = self.model.predict(self.Xvalid)
            self.error_train = mean_squared_error(self.ytrain,self.ypredtrain)
            self.error_valid = mean_squared_error(self.yvalid,self.ypredvalid)
                
        
    def _make_pipeline(self,n_comp,c,k,rs):
        steps = [('pca',PCA(n_components=n_comp,random_state=rs)),('SVM',SVR(C=c,gamma='scale',kernel=k,max_iter=-1))]
        pipe = Pipeline(steps)
        return pipe
