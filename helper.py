import numpy as np

# import enzyme data

def parseEnzymeFile(enzyme_datafile):
    with open(enzyme_datafile,'r') as f:
        enzyme_names = []
        X = []
        y = []
        for lines in f:
            line  = lines.strip().split(',')
            enzyme_names.append(line[0])
            X.append(line[1])
            y.append(int(line[2]))

    return np.array(X),np.array(y),np.array(enzyme_names)

# split to train-test properly

def modified_split(enzyme_names,split=0.25):
    r = range(len(enzyme_names))
    cuphea_idx = [idx for idx in r if enzyme_names[idx].startswith('Cuphea_viscosisssima')]
    rTE_idx = [idx for idx in r if enzyme_names[idx].startswith('rTE')]
    sp_idx = cuphea_idx+rTE_idx
    other_idx = [idx for idx in r if idx not in sp_idx]
    np.random.shuffle(other_idx)
    np.random.shuffle(cuphea_idx)
    np.random.shuffle(rTE_idx)
    lr_other = int(split*len(other_idx))
    lr_cuphea = int(split*len(cuphea_idx))
    lr_rTE = int(split*len(rTE_idx))
    
    test_split_idx = np.concatenate((other_idx[:lr_other],cuphea_idx[:lr_cuphea],rTE_idx[:lr_rTE]),axis=None)
    train_split_idx = np.concatenate((other_idx[lr_other:],cuphea_idx[lr_cuphea:],rTE_idx[lr_rTE:]),axis=None)

    np.random.shuffle(test_split_idx)
    np.random.shuffle(train_split_idx)
    

    return test_split_idx,train_split_idx 

def balancing_function(X_tr,y_tr):
    values,counts = np.unique(y_tr,return_counts=True)
    label_count_dict = {l:c for l,c in zip(values,counts)}
    
    # get the max count
    max_count = max(label_count_dict.values())
    
    # for the labels with less max count, randomly select their index and append that index to the array
    
    for k,v in label_count_dict.items():
        if v<max_count:
            #get the indices with those values
            indices = np.where(y_tr==k)[0]
            
            # get how training set must be duplicated
            n_duplicate = max_count-v
            
            if n_duplicate>v:
                
                times_to_add = n_duplicate//v 
                    
                X_tr_k = X_tr[indices,:]
                
                for i in range(times_to_add):
                    X_tr = np.concatenate((X_tr,X_tr_k),axis=0)
                    y_tr = np.append(y_tr,np.array([k for i in range(v)]))
                
                remaining_times = n_duplicate-times_to_add*v
                
                random_indices = np.random.choice(indices,size=remaining_times)
                
                X_tr_extra = X_tr[random_indices,:]
                y_tr_extra = np.array([k for i in range(remaining_times)])
                
                X_tr = np.concatenate((X_tr,X_tr_extra),axis=0)
                y_tr = np.append(y_tr,y_tr_extra)
                
            else:
                
                remaining_times = n_duplicate
                
                random_indices = np.random.choice(indices,size=remaining_times)
                
                X_tr_extra = X_tr[random_indices,:]
                y_tr_extra = np.array([k for i in range(remaining_times)])
                
                X_tr = np.concatenate((X_tr,X_tr_extra),axis=0)
                y_tr = np.append(y_tr,y_tr_extra)
                
                
    return X_tr,y_tr