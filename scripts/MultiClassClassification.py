#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modules
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

sys.path.append('../')
from baseModels.SVM.model import SVM
from ensemble.model import Ensemble


# In[2]:
class Base:
    def __init__(self,SVM=True,GBC=False,NN=False,pca_components=55,regCparam=30,
        kernparam='rbf',nestparam=250,lrateparam=0.01,mdepthparam=5,ssampleparam=1,hlayer=(50,10),
        lrateinit=0.01,regparam=0.1,random_seed=None,optimizeQ=False,verboseQ=False):
        
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
            
    def get_SVM(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None):
        return SVM(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=self.pca_components,regC=self.regCparam,kern=self.kernparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
    
    def get_GBC(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None):
        return GBC(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=self.pca_components,nest=self.nestparam,lrate=self.lrateparam,mdepth=self.mdepthparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

    def get_NN(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None):
        return NN(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=self.pca_components,hlayers=self.hlayer,lrateinit=self.lrateparam,regparam=self.reg,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)


class TEClassification(Base):
    
    def __init__(self,enzseqdata,testenzseqdata,labelfile,trainfeaturefiledirs,testfeaturefiledirs,model='SVM',random_seed=None,pca_components=55,n_models=17,validation_fraction=0.25):
        
        self.random_seed = random_seed
        self.model=model
        self.default_pca_components = pca_components
        self.n_models = n_models
        self.validation_fraction = validation_fraction
        self.test = True if testfeaturefiledirs else False
        
        
        #initialize super class
        if self.model=='SVM':
            super().__init__(optimizeQ=False)
        else:
            if self.model=='GBC':
                super().__init__(SVM=False,GBC=True)
            elif self.model=='NN':
                super().__init__(SVM=False,NN=True)
            else:
                raise ValueError('Wrong Model Assigned')
        
        self.object_map = {'SVM':self.get_SVM,'NN':self.get_NN,'GBC':self.get_GBC}
        
        # original data based on which everything is obtained
        df1 = pd.read_csv(enzseqdata,header=None)
        df2 = pd.read_csv(labelfile,header=None)
        self.train_df = df1.merge(df2,on=0)
        
        self.enz_names = self.train_df[0].values
        self.X = self.train_df.iloc[:,1].values
        self.y = self.train_df.iloc[:,-1].values
        
        # training and validation data for general use
        self.X_train, self.X_valid, self.y_train, self.y_valid,self.enz_train,self.enz_valid = train_test_split(self.X, self.y,self.enz_names, test_size=self.validation_fraction, random_state=self.random_seed)
        
        self.label_file = labelfile
        
        # test data
        if self.test:
            self.test_df = pd.read_csv(testenzseqdata,header=None)
            self.testenz_names = self.test_df[0].values
            self.X_test = self.test_df.iloc[:,1].values
        else:
            self.X_test=None
            
            
        # kmer and gaakmer
        ng = ngModel(self.X_train,self.X_valid,self.X_test)
        gaang = gaangModel(self.X_train,self.X_valid,self.X_test)
        kmernames = ['kmer','gaakmer']
        kmerObjs = [self.get_model_online(ng.Xtrain,ng.Xvalid,self.y_train,self.y_valid,ng.Xtest),self.get_model_online(gaang.Xtrain,gaang.Xvalid,self.y_train,self.y_valid,gaang.Xtest)]

        
        #generate a list of names from the directories
        trainfeatfiles = [d+f.name for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]            
        self.featnames = [f.name.replace('.csv.gz','') for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
        
        if self.test:
            testfeatfiles = [d+f.name for d in testfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
            func_iter = list(zip(trainfeatfiles,testfeatfiles))
            assert [f.name for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]==[f.name for d in testfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
            self.objects=list(itertools.starmap(self.get_model_feat,func_iter))

        else:
            # getting all SVM objects together
            self.objects = list(map(self.get_model_feat,trainfeatfiles))
            
        self.featnames.extend(kmernames)
        self.objects.extend(kmerObjs)
            
        
        # select only the best models based on training or validation
        self.best_idx,self.best_models = self.select_top_models(self.objects)
        self.best_model_names = np.array(self.featnames)[self.best_idx]
        
        # getting all model predictions together for ensemble
        if not self.test:
            self.all_model_preds = [o.ypredvalid for o in self.best_models]
            self.en = Ensemble(self.all_model_preds,self.y_valid)
            self.precision = precision_score(self.y_valid,self.en.preds,labels=[3],average='micro')
            
        else:
            self.all_model_preds = [o.yhattest for o in self.best_models]
            self.en = Ensemble(self.all_model_preds)
        
        pass
    
    def get_model_online(self,X_train,X_valid,y_train,y_valid,X_test=None):

        if X_train.shape[1]<self.default_pca_components:
            self.pca_components = int(0.75*X_train.shape[1])
        else:
            self.pca_components=self.default_pca_components
            
        if self.test:
            obj = self.object_map[self.model](X_train,X_valid,y_train,y_valid,X_test)
        else:
            obj = self.object_map[self.model](X_train,X_valid,y_train,y_valid)
        return obj
    
    
    def get_model_feat(self,featfilename,testfeatfilename=None):
        
        df1 = pd.read_csv(featfilename,header=None)
        df2 = pd.read_csv(self.label_file,header=None)
        df_feat = df1.merge(df2,on=0).set_index(0)
        df_feat_train = df_feat.loc[self.enz_train]
        df_feat_valid = df_feat.loc[self.enz_valid]
        X_train_feat,y_train_feat = df_feat_train.iloc[:,0:-1].values,df_feat_train.iloc[:,-1].values
        X_valid_feat,y_valid_feat = df_feat_valid.iloc[:,0:-1].values,df_feat_valid.iloc[:,-1].values

        if X_train_feat.shape[1]<self.default_pca_components:
            self.pca_components = int(0.75*X_train_feat.shape[1])
        else:
            self.pca_components=self.default_pca_components
            
        if self.test:
            df_feat_test = pd.read_csv(testfeatfilename,header=None).set_index(0)
            X_test_feat = df_feat_test.loc[self.testenz_names].values
            if X_train_feat.shape[1] != X_test_feat.shape[1]:
                print(featfilename)
            obj = self.object_map[self.model](X_train_feat,X_valid_feat,y_train_feat,y_valid_feat,X_test_feat)
        else:
            obj = self.object_map[self.model](X_train_feat,X_valid_feat,y_train_feat,y_valid_feat)
        return obj
        
    def select_top_models(self,Os):
        o_valid_accs = [o.acc_valid for o in Os] if self.test else [o.acc_train for o in Os] 
        sorted_idx = np.argsort(o_valid_accs)[::-1]
        best_idx = sorted_idx[:self.n_models]
        return best_idx,np.array(Os)[best_idx]
        