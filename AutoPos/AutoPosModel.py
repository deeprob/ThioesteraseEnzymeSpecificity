#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,chi2
from sklearn.model_selection import train_test_split
import sys


# In[15]:


sys.path.append('../')

import helper
from mySVM.model import SVM


# In[19]:


class AutoPosModel:
    def __init__(self,Xtrain,Xtest,ytrain,ytest,scoring_func,savedfilename,n_pos):
        self.Xtrain,self.Xtest,self.ytrain,self.ytest = Xtrain,Xtest,ytrain,ytest
        if self.Xtrain.ndim==1:
            self.Xtrain = np.array(list(map(list,self.Xtrain)))
            self.Xtest = np.array(list(map(list,self.Xtest)))
        self.scfunc = scoring_func
        self.filename =  savedfilename
        self.n_pos = n_pos
        pass
    
    def createEncodedArray(self,X,bestpos):
        X = np.array(list(map(list,X)))
        AAs = ['-','A','C','D','E','F','G','H','I','K','L', #11
                'M','N','P','Q','R','S','T','V','W','X','Y'] #22
        OHE_dict = dict(zip(AAs,range(len(AAs))))
        X_enc = np.zeros((X.shape[0],len(bestpos)*len(AAs)))


        for pos_i,pos in enumerate(bestpos):
            for x_i,x in enumerate(X):

                #get the value of the AA
                AAval = x[pos]
                #get the column index that will be 1
                y_i = pos_i*len(AAs) + OHE_dict[AAval]
                y_i_gaa = pos_i*len(AAs)
                X_enc[x_i,y_i] = 1
        return X_enc
    
    def createEncodedArraywithGAA(self,X,bestpos):
        X = np.array(list(map(list,X)))
        AAs = ['-','A','C','D','E','F','G','H','I','K','L', #11
                'M','N','P','Q','R','S','T','V','W','X','Y'] #22
        OHE_dict = dict(zip(AAs,range(len(AAs))))
        GAAmap = {'G':'g1','A':'g1','V':'g1','L':'g1','M':'g1','I':'g1','F':'g2','Y':'g2','W':'g2',
                           'K':'g3','R':'g3','H':'g3', 'D':'g4', 'E':'g4','S':'g5','T':'g5','C':'g5','P':'g5',
                           'N':'g5','Q':'g5','X':'g6','-':'g6'}
        GAAs = sorted(set(GAAmap.values()))
        for i,gaaval in enumerate(GAAs):
            OHE_dict[gaaval] = i+len(AAs)
        X_enc = np.zeros((X.shape[0],len(bestpos)*(len(AAs)+len(GAAs))))


        for pos_i,pos in enumerate(bestpos):
            for x_i,x in enumerate(X):

                #get the value of the AA
                AAval = x[pos]
                GAAval = GAAmap[AAval]
                #get the column index that will be 1
                y_i = pos_i*(len(AAs)+len(GAAs)) + OHE_dict[AAval]
                y_i_gaa = pos_i*(len(AAs)+len(GAAs)) + OHE_dict[GAAval]
                X_enc[x_i,y_i] = 1
                X_enc[x_i,y_i_gaa] = 1
        return X_enc
    
class OhePosModel(AutoPosModel):
    
    def __init__(self,Xtrain,Xtest,ytrain,ytest,scoring_func,savedfilename,n_pos,imp=True):
        super().__init__(Xtrain,Xtest,ytrain,ytest,scoring_func,savedfilename,n_pos)
        
        self.ohe_enc = self._getOHEncoder(self.Xtrain)
        self.Xtrain_enc = self.ohe_enc.transform(self.Xtrain).toarray()
        self.skbest = self._getBestMapperObj(self.Xtrain_enc,self.ytrain,self.scfunc)
        self.lencat = self._getSeqPos2nAAmap(self.ohe_enc)
        self.numcatmap = self._getOHEpos2seqpos(self.lencat)
        self.BestPositions = self._getBestPositions(self.skbest,self.numcatmap,self.n_pos)
        if imp:
            self.BestPositionsImp = self._getBestPositionswithImportance(self.skbest,self.numcatmap,self.n_pos)
        if savedfilename:
            self._savebestpos(self.BestPositions,self.filename)
        
    
    def _getOHEncoder(self,Xarr):
        '''A one hot encoded representation of the aligned enzyme sequences'''
        ohe = OneHotEncoder()
        ohe.fit(Xarr)
        return ohe
    
    def _getSeqPos2nAAmap(self,ohe):
        '''Number of AAs identified per postion; Functions to be used when mapping back the OHE encoded position 
        to the actual position in the aligned sequence of the enzyme'''
        lencat = [(ci,lc) for ci,lc in zip(range(len(ohe.categories_)),list(map(len,ohe.categories_)))]
        return lencat
    
    def _getOHEpos2seqpos(self,lencat):
        '''mapping from the OHE encoded sequence to the aligned sequence position'''
        
        catnummap = {}
        currval = 0
        nextval = 0

        for i,j in lencat:
            nextval += j
            catnummap[i] = list(range(currval,nextval))
            currval=nextval
            
        numcatmap = {num:cat for cat,nums in catnummap.items() for num in nums}
        return numcatmap
    
    def _getBestMapperObj(self,X_ohe_enc,y,scoring_func):
        skbest = SelectKBest(score_func=scoring_func)
        skbest.fit(X_ohe_enc,y)
        return skbest
    
    def _getBestPositions(self,skbest,numcatmap,n_pos=50):
        
        bestpos = set()
        i=0
        lenbestpos = 0
        sorted_scores_index = np.argsort(skbest.scores_)[::-1]
        
        while lenbestpos<n_pos:
            if not np.isnan(skbest.scores_[sorted_scores_index[i]]):
                bestpos.add(numcatmap[sorted_scores_index[i]])
            i+=1
            lenbestpos = len(bestpos)
            
        return sorted(bestpos)
    
    def _getBestPositionswithImportance(self,skbest,numcatmap,n_pos=50):
        
        bestpos = dict()
        i=0
        lenbestpos = 0
        sorted_scores_index = np.argsort(skbest.scores_)[::-1]
        
        while lenbestpos<n_pos:
            if not np.isnan(skbest.scores_[sorted_scores_index[i]]):
                if numcatmap[sorted_scores_index[i]] not in bestpos:
                    bestpos[numcatmap[sorted_scores_index[i]]] = skbest.scores_[sorted_scores_index[i]]
            i+=1
            lenbestpos = len(bestpos)
            
        return bestpos
    
    def _savebestpos(self,bestpos,filename):
        with open(filename,'w') as f:
            for pos in bestpos:
                f.write(str(pos))
                f.write('\n')
            
        return  
    
    
    

    
class AutoPosClass(OhePosModel):
    
    def __init__(self,Xtrain,Xtest,ytrain,ytest,scoring_func,savedfilename,n_pos,imp=True,random_seed=None,pca_comp=40,regC=1,kern='rbf',probability=False,optimize=False,verbose=True,classweight=None):
        
        super().__init__(Xtrain,Xtest,ytrain,ytest,scoring_func,savedfilename,n_pos,imp)
        
        self.Xtrain,self.Xtest = self.createEncodedArray(Xtrain,self.BestPositions),self.createEncodedArray(Xtest,self.BestPositions)
        
        self.SVMobject = SVM(self.Xtrain,self.Xtest,self.ytrain,self.ytest,random_seed,pca_comp,regC,kern,probability,optimize,verbose,classweight)
        
        pass
        


# In[20]:


if __name__=='__main__':
    np.random.seed(77)
    Aligned_Data_File = '../Data/Enzyme_aligned.txt'
    df = pd.read_csv(Aligned_Data_File,header=None)
    enz_names = df[0].values
    X = df.iloc[:,1].values
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test,enz_train,enz_test = train_test_split(X, y,enz_names, test_size=0.25, random_state=7)
    sc_func = chi2
    n_positions=50
    apmodel = AutoPosModel(X_train,X_test,y_train,y_test,sc_func,None,n_positions)
    ohepos = OhePosModel(X_train,X_test,y_train,y_test,sc_func,None,n_positions,imp=True)
    apclass = AutoPosClass(X_train,X_test,y_train,y_test,sc_func,None,n_positions)
#     print(ohepos.createEncodedArray(X_train,ohepos.BestPositions).shape)


# In[ ]:




