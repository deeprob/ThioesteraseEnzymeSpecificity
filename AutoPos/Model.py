import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,chi2
import sys
sys.path.append('../')

import helper

class AutoPosModel:
    def __init__(self,Aligned_Enzyme_Datafile,scoring_func,savedfilename,n_pos,tr_idx=None):
        self.X,self.y,self.enz_names = helper.parseEnzymeFile(Aligned_Enzyme_Datafile)
        if self.X.ndim==1:
            self.X = np.array(list(map(list,self.X)))
        if tr_idx is not None:
            self.X = self.X[tr_idx,:]
            self.y = self.y[tr_idx]
        self.scfunc = scoring_func
        self.filename =  savedfilename
        self.n_pos = n_pos
        pass
    
class OhePosModel(AutoPosModel):
    
    def __init__(self,Aligned_Enzyme_Datafile,scoring_func,savedfilename,n_pos,imp=True,training=None):
        super().__init__(Aligned_Enzyme_Datafile,scoring_func,savedfilename,n_pos,training)
        
        self.ohe_enc = self._getOHEncoder(self.X)
        self.X_enc = self.ohe_enc.transform(self.X).toarray()
        self.skbest = self._getBestMapperObj(self.X_enc,self.y,self.scfunc)
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
    
    
    

    
class AEPosModel(AutoPosModel):
    def __init__(self,Aligned_Enzyme_Datafile,scoring_func,savedfilename,n_pos,runAE=False):
        super().__init__(Aligned_Enzyme_Datafile,scoring_func,savedfilename,n_pos)
            
        self.AAs = ['-','A','C','D','E','F','G','H','I','K','L',
                    'M','N','P','Q','R','S','T','V','W','X','Y']
        self.GAAmap = {'G':'g1','A':'g1','V':'g1','L':'g1','M':'g1','I':'g1','F':'g2','Y':'g2','W':'g2',
                       'K':'g3','R':'g3','H':'g3', 'D':'g4', 'E':'g4','S':'g5','T':'g5','C':'g5','P':'g5',
                       'N':'g5','Q':'g5','X':'g6','-':'g6'}
        self.ohedict = self._get_ohe_map()
        
        if runAE:
            '''run NN based AE'''
            self.Xtrain = self._create_train_set()
            self._getAE(self.Xtrain)
        
        else:
            '''Import saved model'''
            savedmodel = 'AEModel.sav'
            self.reg = pickle.load(open(savedmodel,'rb'))

        self.Xtest = self._create_test_set()
        self.encode_dict = self._getEncodedMap(self.Xtest)
        self.X_enc = np.array(list(map(self.autoencode,self.X)))
        self.skbest = self._getBestMapperObj(self.X_enc,self.y,self.scfunc)
        self.BestPositions = self._getBestPositions(self.skbest,self.n_pos)
        self._savebestpos(self.BestPositions,self.filename)
        
        
    def _get_ohe_map(self):
        OHEdict = {aa:num for aa, num in zip(self.AAs,range(len(self.AAs)))}
        GAAlist = sorted(set(self.GAAmap.values()))
        GAAnum = range(len(OHEdict),len(OHEdict)+len(GAAlist))
        for gaa,gnum in zip(GAAlist,GAAnum):
            OHEdict[gaa] = gnum
        return OHEdict
    
    def _get_OHE_val(self,aa):
        ohe = [0 for i in range(len(self.ohedict))]
        ohe[self.ohedict[aa]] = 1
        ohe[self.ohedict[self.GAAmap[aa]]] = 1
        return ohe
    
    def _create_train_set(self):
        AAs = [aa for i in range(2000) for aa in self.AAs]
        Xtrain = list(map(self._get_OHE_val,AAs))
        return np.array(Xtrain)

    def _create_test_set(self):
        Xtest = list(map(self._get_OHE_val,self.AAs))
        return np.array(Xtest)
    
    def _predict_AAs(self,predArr):
        AApredictionidx = np.argmax(predArr[:22])
        return self.AAs[AApredictionidx]
    
    def _getAE(self,Xtrain):
        hls = (25,21,15,11,7,3,1,3,7,11,15,23,26)
        self.reg = MLPRegressor(hidden_layer_sizes=hls
                   ,batch_size=10000,
                   activation = 'tanh',solver='adam',learning_rate='adaptive',
                   learning_rate_init=0.001,max_iter=5000,tol=1e-7,verbose=True,
                  n_iter_no_change=500,alpha=0.01)
        self.reg.fit(Xtrain,Xtrain)
        return 

    def _getEncodedMap(self,Xtest):
        AApred = self.reg.predict(Xtest)
        self.predictions = list(map(self._predict_AAs,AApred))
        AA2Contdict = dict(zip(self.AAs,np.ravel(self.encoder(Xtest,6))))
        return AA2Contdict

    def tanh(self,x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def encode(self,x,layer):
        linear_layer = x*self.reg.coefs_[layer] + self.reg.intercepts_[layer]
        activation = self.tanh(linear_layer)
        return activation

    def encoder(self,data,nlayers):
        data = np.asmatrix(data)
        encoder = data
        for i in range(nlayers):
            encoder = self.encode(encoder,i)
        latent = self.encode(encoder,nlayers)
        return np.asarray(latent)
    
    def autoencode(self,seq):
        x_autoencoded = []
        for aa in seq:
            x_autoencoded.append(self.encode_dict[aa])
        return np.array(x_autoencoded)
    
    
    def _getBestMapperObj(self,X_enc,y,scoring_func):
        skbest = SelectKBest(score_func=scoring_func)
        skbest.fit(X_enc,y)
        return skbest 
        
    def _getBestPositions(self,skbest,n_pos=50):
        bestpos = set()
        i=0
        lenbestpos = 0
        sorted_scores_index = np.argsort(skbest.scores_)[::-1]
        while lenbestpos<n_pos:
            bestpos.add(sorted_scores_index[i])
            i+=1
            lenbestpos = len(bestpos)
        return sorted(bestpos)
    
    
    def _savebestpos(self,bestpos,filename):
        with open(filename,'w') as f:
            for pos in bestpos:
                f.write(str(pos))
                f.write('\n')
            
        return  
      