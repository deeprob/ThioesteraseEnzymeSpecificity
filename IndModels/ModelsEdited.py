#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pickle
import numpy as np

sys.path.append('../')
from FeatEngg import ngramEnc,ctdEnc,gaacEnc
from mySVM.model import SVM
from GBClassifier.model import GBC
from NNClassifier.model import NN

class Model:
    def __init__(self,Xtrain,Xtest,ytrain,ytest,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=1,kernparam='linear',nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(5,),lrateinit=0.1,regparam=0.01,random_seed=None,inc_count=False,optimizeQ=False,verboseQ=False):
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
        super().__init__(Xtrain,Xtest,ytrain,ytest,SVM,GBC,NN,pca_components,regCparam,kernparam,nestparam,lrateparam,mdepthparam,ssampleparam,hlayer,lrateinit,regparam,random_seed,inc_count,optimizeQ,verboseQ)
       
        self.Xtrain_raw,self.Xtest_raw,self.ytrain,self.ytest = Xtrain,Xtest,ytrain,ytest
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s,inc_count)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        
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

        super().__init__(Xtrain,Xtest,ytrain,ytest,SVM,GBC,NN,pca_components,regCparam,kernparam,nestparam,lrateparam,mdepthparam,ssampleparam,hlayer,lrateinit,regparam,random_seed,inc_count,optimizeQ,verboseQ)
        
        self.gc = gaacEnc.GAAC()
        X_gaac_train = self.gc.transform(Xtrain)
        X_gaac_test = self.gc.transform(Xtest)
        self.Xtrain_raw,self.Xtest_raw = X_gaac_train,X_gaac_test
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s,inc_count)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)

        
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

