import numpy as np
import sys

# import enzyme data

def parseEnzymeFile(enzyme_datafile):
    with open(enzyme_datafile,'r') as f:
        enzyme_names = []
        X = []
        y = []
        for lines in f:
            line  = lines.strip().split(',')
            enzyme_names.append(line[0])
            X.append(line[1].upper())
            y.append(int(line[2]))

    return np.array(X,dtype='str'),np.array(y,dtype=int),np.array(enzyme_names,dtype='str')

def parseEnzymeFileRegression(enzyme_datafile):
    with open(enzyme_datafile,'r') as f:
        enzyme_names = []
        X = []
        y = []
        for lines in f:
            line  = lines.strip().split(',')
            enzyme_names.append(line[0])
            X.append(line[1].upper())
            y.append(float(line[2]))

    return np.array(X,dtype='str'),np.array(y,dtype=float),np.array(enzyme_names,dtype='str')

# split to train-test properly

def modified_split(enzyme_names,enz_with_acc_file=None,split=0.25):
#     original_enzymes = set()
#     with open(enz_with_acc_file,'r') as f:
#         for lines in f:
#             original_enzymes.add(lines.strip())
    r = range(len(enzyme_names))
#    original_idx = [idx for idx in r if enzyme_names[idx] in original_enzymes]
    no_label_1 = np.where(enzyme_names=='Cuphea_hookeriana(short)_(ChFatB2)')[0][0]
    no_label_2 = np.where(enzyme_names=='Cuphea_viscosisssima_(CvB2MT41')[0][0]
    cuphea_idx = [idx for idx in r if enzyme_names[idx].startswith('Cuphea_viscosisssima')]
    rTE_idx = [idx for idx in r if enzyme_names[idx].startswith('rTE')]
    sp_idx = cuphea_idx+rTE_idx
    other_idx = [idx for idx in r if idx not in sp_idx] 
    cuphea_idx.remove(no_label_2)
    other_idx.remove(no_label_1)
#    np.random.shuffle(original_idx)
    np.random.shuffle(other_idx)
    np.random.shuffle(cuphea_idx)
    np.random.shuffle(rTE_idx)
    
#     total_test_len = split*len(enzyme_names)
#     utilizable_test_data = len(enzyme_names) - len(original_idx)
#     msplit = total_test_len/utilizable_test_data
    
    lr_other = int(split*len(other_idx))
    lr_cuphea = int(split*len(cuphea_idx))
    lr_rTE = int(split*len(rTE_idx))
    
    must_test_label = np.array([no_label_1,no_label_2])
    
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