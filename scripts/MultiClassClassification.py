#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modules
import os
import sys
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split

sys.path.append('../')
from ensemble.model import Ensemble
from baseModels.SVM.model import SVM
from baseModels.GBC.model import GBC
from baseModels.NN.model import NN
from featEngg.online.kmerMethods.models import ngModel,gaangModel


# In[2]:
class Base:
    def __init__(self,SVM=True,GBC=False,NN=False,pca_components=55,regCparam=20,
        kernparam='rbf',nestparam=100,lrateparam=0.1,mdepthparam=3,ssampleparam=1,hlayerparam=(10,),
        lrateinitparam=0.001,regparam=0.001,random_seed=None,optimizeQ=False,verboseQ=False,multi_jobsQ=False):
        
        self.default_pca_components=pca_components
        self.default_optimizeQ=optimizeQ
        self.default_verboseQ=verboseQ
        self.default_multi_jobsQ=multi_jobsQ
        self.default_rs=random_seed
        
        if SVM:
            self.default_regCparam=regCparam
            self.default_kernparam=kernparam
            
        elif GBC:
            self.default_nestparam=nestparam
            self.default_lrateparam=lrateparam
            self.default_mdepthparam=mdepthparam


        elif NN:
            self.default_hlayersparam=hlayerparam
            self.default_lrateinitparam=lrateinitparam
            self.default_regparam=regparam

            
        else:
            raise ValueError('No model initiated')
            
    def get_SVM(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,param_dict=dict()):
        
        if 'pca_comp' not in param_dict:
            param_dict['pca_comp']=self.default_pca_components
        if 'regC' not in param_dict:
            param_dict['regC']=self.default_regCparam
        if 'kern' not in param_dict:
            param_dict['kern']=self.default_kernparam
            
        
        
        return SVM(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=param_dict['pca_comp'],regC=param_dict['regC'],kern=param_dict['kern'],optimize=self.default_optimizeQ,verbose=self.default_verboseQ,random_seed=self.default_rs,multi_jobs=self.default_multi_jobsQ)
    
    def get_GBC(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,param_dict=dict()):
        
        if 'pca_comp' not in param_dict:
            param_dict['pca_comp']=self.default_pca_components
        if 'nest' not in param_dict:
            param_dict['nest']=self.default_nestparam
        if 'lrate' not in param_dict:
            param_dict['lrate']=self.default_lrateparam
        if 'mdepth' not in param_dict:
            param_dict['mdepth']=self.default_mdepthparam
            
            
        return GBC(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=param_dict['pca_comp'],nest=param_dict['nest'],lrate=param_dict['lrate'],mdepth=param_dict['mdepth'],optimize=self.default_optimizeQ,verbose=self.default_verboseQ,random_seed=self.default_rs,multi_jobs=self.default_multi_jobsQ)

    def get_NN(self,Xtrain,Xvalid,ytrain,yvalid,Xtest=None,param_dict=dict()):
        
        if 'pca_comp' not in param_dict:
            param_dict['pca_comp']=self.default_pca_components
        if 'hlayers' not in param_dict:
            param_dict['hlayers']=self.default_hlayersparam
        if 'lrate' not in param_dict:
            param_dict['lrateinit']=self.default_lrateinitparam
        if 'regparam' not in param_dict:
            param_dict['regparam']=self.default_regparam

        return NN(Xtrain,Xvalid,ytrain,yvalid,Xtest,pca_comp=param_dict['pca_comp'],hlayers=param_dict['hlayers'],lrateinit=param_dict['lrateinit'],regparam=param_dict['regparam'],optimize=self.default_optimizeQ,verbose=self.default_verboseQ,random_seed=self.default_rs,multi_jobs=self.default_multi_jobsQ)


class TEClassification(Base):
    
    def __init__(self,enzseqdata,testenzseqdata,labelfile,trainfeaturefiledirs,testfeaturefiledirs,use_feat=None,hyperparamfile=None,model='SVM',random_seed=None,pca_components=55,n_models=5,validation_fraction=0.25,optimize=False):
        
        self.random_seed = random_seed
        self.model=model
        self._pca_components = pca_components
        self.n_models = n_models
        self.validation_fraction = validation_fraction
        self.optimize=optimize
        self.test = True if testfeaturefiledirs else False
        
        
        #initialize super class
        if self.model=='SVM':
            super().__init__(pca_components=self._pca_components,random_seed=self.random_seed,optimizeQ=self.optimize)
        else:
            if self.model=='GBC':
                super().__init__(pca_components=self._pca_components,random_seed=self.random_seed,SVM=False,GBC=True,optimizeQ=self.optimize)
            elif self.model=='NN':
                super().__init__(pca_components=self._pca_components,random_seed=self.random_seed,SVM=False,NN=True,optimizeQ=self.optimize)
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
        
        self.df_hyperparam = pd.read_csv(hyperparamfile).set_index('feat_name') if hyperparamfile else None
        
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
        kmerObjs = [self.get_model_online('kmer',ng.Xtrain,ng.Xvalid,self.y_train,self.y_valid,ng.Xtest),self.get_model_online('gaakmer',gaang.Xtrain,gaang.Xvalid,self.y_train,self.y_valid,gaang.Xtest)]

        
        #generate a list of names from the directories
        self.trainfeatfiles = [d+f.name for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]            
        self.featnames = [f.name.replace('.csv.gz','') for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
        
        if self.test:
            self.testfeatfiles = [d+f.name for d in testfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
            func_iter = list(zip(self.featnames,self.trainfeatfiles,self.testfeatfiles))
            assert [f.name for d in trainfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]==[f.name for d in testfeaturefiledirs for f in os.scandir(d) if f.name.endswith('.csv.gz')]
            self.objects=list(itertools.starmap(self.get_model_offline,func_iter))

        else:
            # getting all objects together
            func_iter = list(zip(self.featnames,self.trainfeatfiles))
            self.objects = list(itertools.starmap(self.get_model_offline,func_iter))
            
        self.featnames.extend(kmernames)
        self.objects.extend(kmerObjs)
            
        if use_feat is not None:
            assert self.n_models == len(use_feat)
            self.best_model_names, self.best_models = self.select_predef_models(use_feat,self.objects)
        
        else:
            # select only the best models based on training or validation
            self.best_model_names, self.best_models = self.select_top_models(self.objects)

        
        # getting all model predictions together for ensemble
        if not self.test:
            self.all_model_preds = [o.ypredvalid for o in self.objects]
            self.best_model_preds = [o.ypredvalid for o in self.best_models]
            self.en = Ensemble(self.best_model_preds,self.y_valid)
            self.precision = precision_score(self.y_valid,self.en.preds,labels=[3],average='micro')
            self.recall = recall_score(self.y_valid,self.en.preds,labels=[3],average='micro')
            
        else:
            self.best_model_valid_preds = [o.ypredvalid for o in self.best_models]
            self.en_valid = Ensemble(self.best_model_valid_preds,self.y_valid)
            self.precision = precision_score(self.y_valid,self.en_valid.preds,labels=[3],average='micro')
            self.recall = recall_score(self.y_valid,self.en_valid.preds,labels=[3],average='micro')
            self.best_model_preds = [o.yhattest for o in self.best_models]
            self.en = Ensemble(self.best_model_preds)
        
        pass
    
    def get_model_online(self,model_name,X_train,X_valid,y_train,y_valid,X_test=None):

        if X_train.shape[1]<self._pca_components:
            self.default_pca_components = int(0.75*X_train.shape[1])
        else:
            self.default_pca_components=self._pca_components
            
        if self.df_hyperparam is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyperparam.loc[model_name,'pca_comp'] 
                param_dict_['regC'] = self.df_hyperparam.loc[model_name,'regC']
                param_dict_['kernel'] = self.df_hyperparam.loc[model_name,'kernel']
        
        else:
            param_dict_ = dict()
            
        if self.test:
            obj = self.object_map[self.model](X_train,X_valid,y_train,y_valid,X_test,param_dict=param_dict_)
        else:
            obj = self.object_map[self.model](X_train,X_valid,y_train,y_valid,param_dict=param_dict_)
        return obj
    
    
    def get_model_offline(self,featname,featfilename,testfeatfilename=None):

        df1 = pd.read_csv(featfilename,header=None)
        df2 = pd.read_csv(self.label_file,header=None)
        df_feat = df1.merge(df2,on=0).set_index(0)
        df_feat_train = df_feat.loc[self.enz_train]
        df_feat_valid = df_feat.loc[self.enz_valid]
        X_train_feat,y_train_feat = df_feat_train.iloc[:,0:-1].values,df_feat_train.iloc[:,-1].values
        X_valid_feat,y_valid_feat = df_feat_valid.iloc[:,0:-1].values,df_feat_valid.iloc[:,-1].values

        if X_train_feat.shape[1]<self._pca_components:
            self.default_pca_components = int(0.75*X_train_feat.shape[1])
        else:
            self.default_pca_components=self._pca_components
        
            
        if self.df_hyperparam is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyperparam.loc[featname,'pca_comp'] 
                param_dict_['regC'] = self.df_hyperparam.loc[featname,'regC']
                param_dict_['kernel'] = self.df_hyperparam.loc[featname,'kernel']
        
        else:
            param_dict_ = dict()
            
        if self.test:

            df_feat_test = pd.read_csv(testfeatfilename,header=None).set_index(0)
            X_test_feat = df_feat_test.loc[self.testenz_names].values
            if X_train_feat.shape[1] != X_test_feat.shape[1]:
                print(featfilename)
                
                
            obj = self.object_map[self.model](X_train_feat,X_valid_feat,y_train_feat,y_valid_feat,X_test_feat,param_dict=param_dict_)
            


        else:
            obj = self.object_map[self.model](X_train_feat,X_valid_feat,y_train_feat,y_valid_feat,param_dict=param_dict_)
        return obj

        
    def select_top_models(self,Os):
        o_valid_accs = [o.acc_valid for o in Os] #if self.test else [o.acc_train for o in Os] 
        sorted_idx = np.argsort(o_valid_accs)[::-1]
        best_idx = sorted_idx[:self.n_models]
        
        return np.array(self.featnames)[best_idx],np.array(Os)[best_idx]
    
    def select_predef_models(self,pre_def_models,Os):
        model_objs = []
        
        for fname, fobj in zip(self.featnames, self.objects):
            if fname in pre_def_models:
                model_objs.append(fobj)
                
        return pre_def_models,model_objs
    

    def get_best_hp(self,model_obj):
        return tuple(model_obj.grid.best_params_.values()) if self.optimize else None
    
    
    def get_best_hps(self):
        hps = list(map(self.get_best_hp, self.objects))
        return list(zip(self.featnames, hps))
