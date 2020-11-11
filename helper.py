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