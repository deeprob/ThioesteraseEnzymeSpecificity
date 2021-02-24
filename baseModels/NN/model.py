#import modules

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm


class NN:
    
    def __init__(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,random_seed=None,pca_comp=20,hlayers=(10,),lrateinit=0.1,regparam=0.01,optimize=False,verbose=True):
        np.random.seed(random_seed)
        
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.ytrain = ytrain
        self.yvalid = yvalid
        
        pipeline = self._make_pipeline(pca_comp,hlayers,lrateinit,regparam)
        
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
                         'NN__hidden_layer_sizes':[(10,5,),(5,),(10,)],
                         'NN__learning_rate_init':[0.1,0.01,0.001],
                         'NN__alpha':[0.01,0.001,0.0001]}

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
                
        
    def _make_pipeline(self,n_comp,h,lr,reg):
        steps = [('scaler',StandardScaler()),('pca',PCA(n_components=n_comp)),('NN',MLPClassifier(hidden_layer_sizes=h,activation='logistic',solver='adam',learning_rate='adaptive',learning_rate_init=lr,alpha=reg,max_iter=200))]
        pipe = Pipeline(steps)
        return pipe
