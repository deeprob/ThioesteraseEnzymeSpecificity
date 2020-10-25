import numpy as np

class GAAC:
    def __init__(self):
        self.groups = {'G':'g1','A':'g1','V':'g1','L':'g1','M':'g1','I':'g1','F':'g2','Y':'g2','W':'g2',
                       'K':'g3','R':'g3','H':'g3', 'D':'g4', 'E':'g4','S':'g5','T':'g5','C':'g5','P':'g5',
                       'N':'g5','Q':'g5','X':'g5'}
        
    def transform(self,sequences):
        
        X_gaac = []
        for seq in sequences:
            curr_seq = ''
            for i in seq:
                curr_seq+=self.groups[i]
                
            X_gaac.append(curr_seq)
        
        return np.array(X_gaac) 