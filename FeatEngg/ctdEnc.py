import numpy as np
import itertools

class CTD:
    def __init__(self,F='CTD'):
        '''F can be C,T,D,CT,CD,DT or CTD'''
        
        self.F = F
        # Attributes
        self.hydrophobicity_pram = {'RKEDQN':0,'GASTPHY':1,'CLVIMFWX':2}
        self.vdwvol = {'GASCTPD':0,'NVEQIL':1,'MHKFRYWX':2}
        self.polarity = {'LIFWCMVYX':0,'PATGS':1,'HQRKNED':2}  # https://onlinelibrary.wiley.com/doi/full/10.1002/%28SICI%291097-0134%2819990601%2935%3A4%3C401%3A%3AAID-PROT3%3E3.0.CO%3B2-K
        self.polarizability = {'GASDT':0,'CPNVEQILX':1,'KMHFRYW':2} # https://onlinelibrary.wiley.com/doi/full/10.1002/%28SICI%291097-0134%2819990601%2935%3A4%3C401%3A%3AAID-PROT3%3E3.0.CO%3B2-K
        self.charge = {'KR':0,'ANCQGHILMFPSTWYVX':1,'DE':2}
        self.ss = {'EALMQKRHX':0,'VIYCWFT':1,'GNPSD':2}
        self.solvacc = {'ALFCGIVWX':0,'RKQEND':1,'MPSTHY':2} #https://bioinformatics.stackexchange.com/questions/4744/which-is-the-amino-acids-classification-when-analyzing-the-solvent-accessibility
        
        self.attrdicts = list(map(self._expand_dict,[self.hydrophobicity_pram,self.vdwvol,self.polarity,self.polarizability,
                                                    self.charge,self.ss,self.solvacc]))
        
    def _expand_dict(self,mydict):
        return {k:v for keys,v in mydict.items() for k in keys}
    
    def get_ind_attr_seq(self,seq,attr_dict):
        
        attr_comp_dict = {0:0,1:0,2:0}
        attr_transition_dict = {'01':0,'12':0,'02':0}
        attr_distribution_dict = {'01':0,'025':0,'050':0,'075':0,'0100':0,
                                 '11':0,'125':0,'150':0,'175':0,'1100':0,
                                 '21':0,'225':0,'250':0,'275':0,'2100':0}
        
        
        prev = None
        
        for s in seq:
            
            if s=='-':
                continue
            
            # transition
            curr = attr_dict[s]
            if prev is not None:
                if curr!=prev:
                    if curr>prev:
                        tran_val = str(prev) + str(curr)
                    elif curr<prev:
                        tran_val = str(curr) + str(prev)
                    attr_transition_dict[tran_val] += 1
                
            prev=curr
            
            # composition
            attr_comp_dict[attr_dict[s]]+=1
            

        attr_curr_dist = {0:0,1:0,2:0}
        for sidx,s in enumerate(seq):
            if s=='-':
                continue
            curr_cat = attr_dict[s]
            total_val = attr_comp_dict[attr_dict[s]]
            attr_curr_dist[curr_cat]+=1
            
            curr_val = (attr_curr_dist[curr_cat]/total_val)*100
            
            if curr_val==100.0:
                attr_distribution_dict[str(curr_cat)+'100']=sidx+1 
            
            if curr_val>=75 and attr_distribution_dict[str(curr_cat)+'75']==0:
                attr_distribution_dict[str(curr_cat)+'75'] = sidx+1
                
            if curr_val>=50 and attr_distribution_dict[str(curr_cat)+'50']==0:
                attr_distribution_dict[str(curr_cat)+'50'] = sidx+1

            if curr_val>=25 and attr_distribution_dict[str(curr_cat)+'25']==0:
                attr_distribution_dict[str(curr_cat)+'25'] = sidx+1
          
            if attr_curr_dist[curr_cat]==1:
                attr_distribution_dict[str(curr_cat)+'1'] = sidx+1
        
        comp = [v/len(seq) for v in attr_comp_dict.values()]
        tran = [v/(len(seq)-1) for v in attr_transition_dict.values()]
        dist = [v/(len(seq)) for v in attr_distribution_dict.values()]
        
        ifdict = {'C':comp,'T':tran,'D':dist,'CT':comp+tran,'CD':comp+dist,'DT':tran+dist,'CTD':comp+tran+dist}
        
        return ifdict[self.F]
    
    
    def get_ind_attr(self,seq):
        return sum(list(itertools.starmap(self.get_ind_attr_seq,itertools.product([seq],self.attrdicts))),[])
        

    
    def get_ctd(self,sequences):
        
        return np.array(list(map(self.get_ind_attr,sequences)))
       
            
            
            
        