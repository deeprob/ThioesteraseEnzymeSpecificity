import numpy as np


class GAA:
    def __init__(self):
        self.groups = {'G': 'a', 'A': 'a', 'V': 'a', 'L': 'a', 'M': 'a', 'I': 'a', 'F': 'b', 'Y': 'b', 'W': 'b',
                       'K': 'c', 'R': 'c', 'H': 'c', 'D': 'd', 'E': 'd', 'S': 'e', 'T': 'e', 'C': 'e', 'P': 'e',
                       'N': 'e', 'Q': 'e', 'X': 'f', '-': 'f'}

    def transform(self, sequences):

        X_gaa = []
        for seq in sequences:
            curr_seq = ''
            for i in seq:
                curr_seq += self.groups[i]

            X_gaa.append(curr_seq)

        return np.array(X_gaa)


class Ngram:
    def __init__(self, sequences, n, step, inc_count=False):
        self.sequences = sequences
        self.n = n
        self.step = step
        self.inc_count = inc_count
        self.encoder_dict = {}

    def fit(self):
        ngramdict = dict()
        for sequence in self.sequences:
            i = 0
            while i + self.n <= len(sequence):
                if type(sequence) == str:
                    seq = sequence[i:i + self.n]
                else:
                    seq = ''.join(sequence[i:i + self.n])

                if seq in ngramdict:
                    ngramdict[seq] += 1
                else:
                    ngramdict[seq] = 1
                i += self.step

        feature_list = sorted([k for k, v in ngramdict.items() if v > 1])
        self.encoder_dict = dict(zip(feature_list, list(range(len(feature_list)))))
        return

    def transform(self, sequences):

        if not self.encoder_dict:
            raise ValueError('Need to fit first')

        X_motif = []

        for sequence in sequences:
            ind_vector = np.zeros(len(self.encoder_dict))

            i = 0
            while i + self.n <= len(sequence):
                if type(sequence) == str:
                    seq = sequence[i:i + self.n]
                else:
                    seq = ''.join(sequence[i:i + self.n])

                if seq in self.encoder_dict:
                    if self.inc_count:
                        ind_vector[self.encoder_dict[seq]] += 1
                    else:
                        ind_vector[self.encoder_dict[seq]] = 1

                i += self.step

            X_motif.append(ind_vector)

        return np.array(X_motif)


class ngModel:
    # NGram model
    def __init__(self, Xtrain, Xvalid, Xtest=None, k=7, s=1, inc_count=False):

        self.Xtrain_raw, self.Xvalid_raw = Xtrain, Xvalid
        self.ng = Ngram(self.Xtrain_raw, k, s, inc_count)
        self.ng.fit()
        self.Xtrain, self.Xvalid = self.ng.transform(self.Xtrain_raw), self.ng.transform(self.Xvalid_raw)

        if Xtest is not None:
            self.Xtest = self.ng.transform(Xtest)
        else:
            self.Xtest = None


class gaangModel:
    # GAA-NGram model
    def __init__(self, Xtrain, Xvalid, Xtest=None, k=7, s=1, inc_count=False):

        self.gc = GAA()
        X_gaac_train = self.gc.transform(Xtrain)
        X_gaac_valid = self.gc.transform(Xvalid)
        self.Xtrain_raw, self.Xvalid_raw = X_gaac_train, X_gaac_valid
        self.ng = Ngram(self.Xtrain_raw, k, s, inc_count)
        self.ng.fit()
        self.Xtrain, self.Xvalid = self.ng.transform(self.Xtrain_raw), self.ng.transform(self.Xvalid_raw)

        if Xtest is not None:
            self.Xtest = self.ng.transform(Xtest)
        else:
            self.Xtest = None
