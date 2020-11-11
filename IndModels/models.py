import sys
import pickle
import numpy as np

sys.path.append('../')
from FeatEngg import ngramEnc,ctdEnc,gaacEnc
from mySVM.model import SVM

class PosModel:
    # Position-Model By Mike
    def __init__(self,pickle_file_path,enzs,X,y,tr_idx,te_idx,pca_components=40,regCparam=1,kernparam='linear',optimizeQ=False,verboseQ=False,randomseed=None):
        self.file = pickle.load(open(pickle_file_path,'rb'))
        self.Pos_model_dict = {name:feat for name,feat in zip(self.file[0],self.file[1])}
        self.Xtrain,self.Xtest = self.get_pos_model_feat(self.Pos_model_dict,enzs,tr_idx,te_idx)
        self.ytrain,self.ytest = y[tr_idx],y[te_idx]
        self.SVMobject = SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=pca_components,regC=regCparam,kern=kernparam,optimize=optimizeQ,verbose=verboseQ,random_seed=randomseed)
        self.model = self.SVMobject.model
    
    def get_pos_model_feat(self,pos_model_dict,enz_names,train_idx,test_idx):
        X_train = []
        X_test = []
        for enz_tr in enz_names[train_idx]:
            X_train.append(pos_model_dict[enz_tr])
        for enz_te in enz_names[test_idx]:
            X_test.append(pos_model_dict[enz_te])
        return np.array(X_train),np.array(X_test)
    
class NGModel:
    # NGram model
    def __init__(self,enzs,X,y,tr_idx,te_idx,k=7,s=1,pca_components=40,regCparam=1,kernparam='linear',optimizeQ=False,verboseQ=False,randomseed=None):
        self.Xtrain_raw,self.ytrain,self.Xtest_raw,self.ytest = X[tr_idx],y[tr_idx],X[te_idx],y[te_idx]
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        self.SVMobject = SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=pca_components,regC=regCparam,kern=kernparam,optimize=optimizeQ,verbose=verboseQ,random_seed=randomseed)
        self.model = self.SVMobject.model
        
class GAACModel:
    # GAAC-NGram model
    def __init__(self,enzs,X,y,tr_idx,te_idx,k=7,s=1,pca_components=40,regCparam=20,kernparam='rbf',optimizeQ=False,verboseQ=False,randomseed=None):
        self.gc = gaacEnc.GAAC()
        X_gaac = self.gc.transform(X)
        
        self.Xtrain_raw,self.ytrain,self.Xtest_raw,self.ytest = X_gaac[tr_idx],y[tr_idx],X_gaac[te_idx],y[te_idx]
        self.ng = ngramEnc.Ngram(self.Xtrain_raw,k,s)
        self.ng.fit()
        self.Xtrain,self.Xtest = self.ng.transform(self.Xtrain_raw),self.ng.transform(self.Xtest_raw)
        self.SVMobject = SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,pca_comp=pca_components,regC=regCparam,kern=kernparam,optimize=optimizeQ,verbose=verboseQ,random_seed=randomseed)
        self.model = self.SVMobject.model
