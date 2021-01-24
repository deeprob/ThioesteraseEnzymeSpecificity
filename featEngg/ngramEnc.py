import numpy as np

class Ngram:
    def __init__(self,sequences,n,step,inc_count=False):
        self.sequences = sequences
        self.n = n
        self.step = step
        self.inc_count=inc_count
        self.encoder_dict={}
    
    def fit(self):
        ngramdict = dict()
        for sequence in self.sequences:
            i=0
            while i+self.n<=len(sequence):
                if type(sequence)==str:
                    seq = sequence[i:i+self.n] 
                else:
                    seq = ''.join(sequence[i:i+self.n])

                if seq in ngramdict:
                    ngramdict[seq] += 1
                else:
                    ngramdict[seq] = 1
                i+=self.step

        feature_list = sorted([k for k,v in ngramdict.items() if v>1])
        self.encoder_dict = dict(zip(feature_list,list(range(len(feature_list)))))
        return 

    def transform(self,sequences):
        
        if not self.encoder_dict:
            raise ValueError('Need to fit first')

        X_motif = []

        for sequence in sequences:
            ind_vector = np.zeros(len(self.encoder_dict))

            i=0
            while i+self.n<=len(sequence):
                if type(sequence)==str:
                    seq = sequence[i:i+self.n] 
                else:
                    seq = ''.join(sequence[i:i+self.n])
                    
                if seq in self.encoder_dict:
                    if self.inc_count:
                        ind_vector[self.encoder_dict[seq]] += 1
                    else:
                        ind_vector[self.encoder_dict[seq]] = 1

                i+=self.step

            X_motif.append(ind_vector)

        return np.array(X_motif)
