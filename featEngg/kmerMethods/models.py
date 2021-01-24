#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np

import ngramEnc
import gaacEnc

import sys
sys.path.append('../../')
from baseModels.SVM.model import SVM
from baseModels.GBC.model import GBC
from baseModels.NN.model import NN

class Model:
    def __init__(self,Xtrain,Xtest,ytrain,ytest,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=1,
                 kernparam='linear',nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(5,),
                 lrateinit=0.1,regparam=0.01,random_seed=None,inc_count=False,optimizeQ=False,verboseQ=False):
        
        self.ytrain,self.ytest = ytrain,ytest
        self.pca_components=pca_components
        self.optimizeQ=optimizeQ
        self.verboseQ=verboseQ
        self.rs=random_seed
        
        if SVM:
            self.regCparam=regCparam
            self.kernparam=kernparam
            
        elif GBC:
            self.nestparam=nestparam
            self.lrateparam=lrateparam
            self.mdepthparam=mdepthparam


        elif NN:
            self.hlayer=hlayer
            self.lrateparam=lrateinit
            self.reg=regparam

            
        else:
            raise ValueError('No model initiated')
            
    def get_SVM(self):
        return SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,regC=self.regCparam,kern=self.kernparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
    
    def get_GBC(self):
        return GBC(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,nest=self.nestparam,lrate=self.lrateparam,mdepth=self.mdepthparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

    def get_NN(self):
        return NN(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,hlayers=self.hlayer,lrateinit=self.lrateparam,regparam=self.reg,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)


    
class NGModel(Model):
    # NGram model
    def __init__(self,Xtrain,Xtest,ytrain,ytest,k=7,s=1,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=1,kernparam='linear',nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(5,),lrateinit=0.1,regparam=0.01,random_seed=None,inc_count=False,optimizeQ=False,verboseQ=False):
        
        super().__init__(Xtrain,Xtest,ytrain,ytest)

        self.Xtrain_raw,self.Xtest_raw = Xtrain,Xtest
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s,inc_count)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        self.pca_components=pca_components
        self.optimizeQ=optimizeQ
        self.verboseQ=verboseQ
        self.rs=random_seed

        
        if SVM:
            self.SVMobject = self.get_SVM()
            self.model = self.SVMobject.model
            
        elif GBC:
            self.GBCobject=self.get_GBC()
            self.model=self.GBCobject.model

        elif NN:
            self.NNobject=self.get_NN()
            self.model=self.NNobject.model
            
        else:
            raise ValueError('No model initiated')
        

class GAACModel(Model):
    # GAAC-NGram model
    def __init__(self,Xtrain,Xtest,ytrain,ytest,k=7,s=1,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=20,kernparam='rbf',nestparam=15,lrateparam=0.5,mdepthparam=3,ssampleparam=1,hlayer=(5,),lrateinit=0.1,regparam=0.01,inc_count=True,optimizeQ=False,verboseQ=False,random_seed=None):

        self.gc = gaacEnc.GAAC()
        X_gaac_train = self.gc.transform(Xtrain)
        X_gaac_test = self.gc.transform(Xtest)
        self.Xtrain_raw,self.ytrain,self.Xtest_raw,self.ytest = X_gaac_train,ytrain,X_gaac_test,ytest
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s,inc_count)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)

        super().__init__(Xtrain,Xtest,ytrain,ytest,SVM,GBC,NN,pca_components,regCparam,
                 kernparam,nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(5,),
                 lrateinit=0.1,regparam=0.01,random_seed=None,inc_count=False,optimizeQ=False,verboseQ=False)
        
        if SVM:
            self.SVMobject = self.get_SVM()
            self.model = self.SVMobject.model
            
        elif GBC:
            self.GBCobject=self.get_GBC()
            self.model=self.GBCobject.model
            
        elif NN:
            self.NNobject=self.get_NN()
            self.model=self.NNobject.model

            
        else:
            raise ValueError('No model initiated')


# In[11]:


if __name__=='__main__':
    import helper
    import pandas as pd
    from sklearn.model_selection import train_test_split
    enz_datafile = '../../data/SeqFile/EnzymeSequence.csv'
    label_file = '../../data/LabelFiles/EnzymeLabelsMultiClass.csv'
    df1 = pd.read_csv(enz_datafile,header=None)
    df2 = pd.read_csv(label_file,header=None)
    df = df1.merge(df2,on=0)
    enz_names = df[0].values
    X = df.iloc[:,1].values
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test,enz_train,enz_test = train_test_split(X, y,enz_names, test_size=0.25, random_state=7)
    ngmodel = NGModel(X_train,X_test,y_train,y_test)
    gaamodel = GAACModel(X_train,X_test,y_train,y_test)
    print(ngmodel.SVMobject.acc_test,gaamodel.SVMobject.acc_test)

