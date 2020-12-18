import numpy as np

class GAAC:
    def __init__(self):
        self.groups = {'G':'a','A':'a','V':'a','L':'a','M':'a','I':'a','F':'b','Y':'b','W':'b',
                       'K':'c','R':'c','H':'c', 'D':'d', 'E':'d','S':'e','T':'e','C':'e','P':'e',
                       'N':'e','Q':'e','X':'f','-':'f'}
        
    def transform(self,sequences):
        
        X_gaac = []
        for seq in sequences:
            curr_seq = ''
            for i in seq:
                curr_seq+=self.groups[i]
                
            X_gaac.append(curr_seq)
        
        return np.array(X_gaac) 