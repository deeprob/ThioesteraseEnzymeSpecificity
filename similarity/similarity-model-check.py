def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn


import time
import numpy as np
import multiprocessing as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


similarity_matrix = '../similarity/similarity_matrix.txt'
label_file = '../data/label/EnzymeLabelsMultiClass.csv'


def get_matrix():
    mat = np.loadtxt(similarity_matrix, delimiter=',')
    return mat


def get_labels():
    y = []
    with open(label_file, 'rt') as f:
        for lines in f:
            vals = lines.strip().split(',')
            y.append(int(vals[1]))
    y = np.array(y)
    return y


def get_train_valid_split(mat, y):
    all_indices = [i for i in range(115)]

    train_idx = np.random.choice(all_indices, size=80, replace=False)
    valid_idx = [i for i in all_indices if i not in train_idx]

    X_train = mat[train_idx, :][:, train_idx]
    X_valid = mat[valid_idx, :][:, train_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]
    return X_train, X_valid, y_train, y_valid


def get_predictions(X_train, X_valid, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='precomputed')
    neigh.fit(X_train, y_train)

    y_hat_valid = neigh.predict(X_valid)
    return y_hat_valid


def get_model_metrics(y_valid, y_hat_valid):
    acc = accuracy_score(y_valid, y_hat_valid)
    rec = recall_score(y_valid, y_hat_valid, labels=[3], average='micro')
    prec = precision_score(y_valid, y_hat_valid, labels=[3], average='micro')
    return prec, rec, acc


def save_metrics(model_metrics):
    with open('../similarity/results/model_sims.csv', 'w') as f:
        for i, j, k in model_metrics:
            f.write(f'{i},{j},{k}')
            f.write('\n')
    return


def main(random_seed):
    np.random.seed(random_seed)

    # get the matrix
    sim_mat = get_matrix()

    # get the labels
    y = get_labels()

    # get train valid data
    Xtrain, Xvalid, ytrain, yvalid = get_train_valid_split(sim_mat, y)

    # get model predictions
    y_hat = get_predictions(Xtrain, Xvalid, ytrain)

    # get model performance
    prec, rec, acc = get_model_metrics(yvalid, y_hat)

    return prec, rec, acc


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    N = 10000
    iterable = [i for i in range(N)]
    start_time = time.time()
    metrics = pool.map(main, iterable)
    end_time = time.time()
    print(end_time - start_time)
    save_metrics(metrics)
