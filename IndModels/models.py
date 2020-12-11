import sys
import pickle
import numpy as np

sys.path.append('../')
from FeatEngg import ngramEnc,ctdEnc,gaacEnc
from mySVM.model import SVM
from GBClassifier.model import GBC
from NNClassifier.model import NN

class PosModel:
    # Position-Model By Mike
    def __init__(self,pickle_file_path,enzs,X,y,tr_idx,te_idx,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=1,kernparam='linear',nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(10,),lrateinit=0.1,regparam=0.01,random_seed=None,optimizeQ=False,verboseQ=False):
        self.file = pickle.load(open(pickle_file_path,'rb'))
        self.Pos_model_dict = {name:feat for name,feat in zip(self.file[0],self.file[1])}
        self.Xtrain,self.Xtest = self.get_pos_model_feat(self.Pos_model_dict,enzs,tr_idx,te_idx)
        self.ytrain,self.ytest = y[tr_idx],y[te_idx]
        self.pca_components=pca_components
        self.optimizeQ=optimizeQ
        self.verboseQ=verboseQ
        self.rs=random_seed
        
        if SVM:
            self.regCparam=regCparam
            self.kernparam=kernparam
            self.SVMobject = self.get_SVM()
            self.model = self.SVMobject.model
            
        elif GBC:
            self.nestparam=nestparam
            self.lrateparam=lrateparam
            self.mdepthparam=mdepthparam
            self.GBCobject=self.get_GBC()
            self.model=self.GBCobject.model

        elif NN:
            self.hlayer=hlayer
            self.lrateparam=lrateinit
            self.reg=regparam
            self.NNobject=self.get_NN()
            self.model=self.NNobject.model
            
        else:
            raise ValueError('No model initiated')
    
    def get_pos_model_feat(self,pos_model_dict,enz_names,train_idx,test_idx):
        X_train = []
        X_test = []
        for enz_tr in enz_names[train_idx]:
            X_train.append(pos_model_dict[enz_tr])
        for enz_te in enz_names[test_idx]:
            X_test.append(pos_model_dict[enz_te])
        return np.array(X_train),np.array(X_test)
    
    def get_SVM(self):
        return SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,regC=self.regCparam,kern=self.kernparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
    
    def get_GBC(self):
        return GBC(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,nest=self.nestparam,lrate=self.lrateparam,mdepth=self.mdepthparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

    def get_NN(self):
        return NN(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,hlayers=self.hlayer,lrateinit=self.lrateparam,regparam=self.reg,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

class NGModel:
    # NGram model
    def __init__(self,enzs,X,y,tr_idx,te_idx,k=7,s=1,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=1,kernparam='linear',nestparam=100,lrateparam=0.1,mdepthparam=1,ssampleparam=1,hlayer=(5,),lrateinit=0.1,regparam=0.01,random_seed=None,optimizeQ=False,verboseQ=False):
        self.Xtrain_raw,self.ytrain,self.Xtest_raw,self.ytest = X[tr_idx],y[tr_idx],X[te_idx],y[te_idx]
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        self.pca_components=pca_components
        self.optimizeQ=optimizeQ
        self.verboseQ=verboseQ
        self.rs=random_seed
        
        if SVM:
            self.regCparam=regCparam
            self.kernparam=kernparam
            self.SVMobject = self.get_SVM()
            self.model = self.SVMobject.model
            
        elif GBC:
            self.nestparam=nestparam
            self.lrateparam=lrateparam
            self.mdepthparam=mdepthparam
            self.GBCobject=self.get_GBC()
            self.model=self.GBCobject.model

        elif NN:
            self.hlayer=hlayer
            self.lrateparam=lrateinit
            self.reg=regparam
            self.NNobject=self.get_NN()
            self.model=self.NNobject.model
            
        else:
            raise ValueError('No model initiated')
        
    def get_SVM(self):
        return SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,regC=self.regCparam,kern=self.kernparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
    
    def get_GBC(self):
        return GBC(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,nest=self.nestparam,lrate=self.lrateparam,mdepth=self.mdepthparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

    def get_NN(self):
        return NN(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,hlayers=self.hlayer,lrateinit=self.lrateparam,regparam=self.reg,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

class GAACModel:
    # GAAC-NGram model
    def __init__(self,enzs,X,y,tr_idx,te_idx,k=7,s=1,SVM=True,GBC=False,NN=False,pca_components=40,regCparam=20,kernparam='rbf',nestparam=15,lrateparam=0.5,mdepthparam=3,ssampleparam=1,hlayer=(5,),lrateinit=0.1,regparam=0.01,optimizeQ=False,verboseQ=False,random_seed=None):
        self.gc = gaacEnc.GAAC()
        X_gaac = self.gc.transform(X)
        
        self.Xtrain_raw,self.ytrain,self.Xtest_raw,self.ytest = X_gaac[tr_idx],y[tr_idx],X_gaac[te_idx],y[te_idx]
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        self.pca_components=pca_components
        self.optimizeQ=optimizeQ
        self.verboseQ=verboseQ
        self.rs=random_seed
        
        if SVM:
            self.regCparam=regCparam
            self.kernparam=kernparam
            self.SVMobject = self.get_SVM()
            self.model = self.SVMobject.model
            
        elif GBC:
            self.nestparam=nestparam
            self.lrateparam=lrateparam
            self.mdepthparam=mdepthparam
            self.GBCobject=self.get_GBC()
            self.model=self.GBCobject.model
            
        elif NN:
            self.hlayer=hlayer
            self.lrateparam=lrateinit
            self.reg=regparam
            self.NNobject=self.get_NN()
            self.model=self.NNobject.model

            
        else:
            raise ValueError('No model initiated')
        

    def get_SVM(self):
        return SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,regC=self.regCparam,kern=self.kernparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
    
    def get_GBC(self):
        return GBC(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,nest=self.nestparam,lrate=self.lrateparam,mdepth=self.mdepthparam,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)

    def get_NN(self):
        return NN(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=self.pca_components,hlayers=self.hlayer,lrateinit=self.lrateparam,regparam=self.reg,optimize=self.optimizeQ,verbose=self.verboseQ,random_seed=self.rs)
