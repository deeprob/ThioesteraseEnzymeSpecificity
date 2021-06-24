import os
import itertools
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from .meta import Ensemble
from .base import SVM, GBC, NN
from ngrampro import NGModel, GAANGModel


class Base:
    def __init__(self, SVM=True, GBC=False, NN=False, pca_components=55, regCparam=20,
                 kernparam='rbf', nestparam=100, lrateparam=0.1, mdepthparam=3, hlayerparam=(10,),
                 lrateinitparam=0.001, regparam=0.001, random_seed=None, optimizeQ=False, verboseQ=False,
                 multi_jobsQ=False):

        self.default_pca_components = pca_components
        self.default_optimizeQ = optimizeQ
        self.default_verboseQ = verboseQ
        self.default_multi_jobsQ = multi_jobsQ
        self.default_rs = random_seed

        if SVM:
            self.default_regCparam = regCparam
            self.default_kernparam = kernparam

        elif GBC:
            self.default_nestparam = nestparam
            self.default_lrateparam = lrateparam
            self.default_mdepthparam = mdepthparam

        elif NN:
            self.default_hlayersparam = hlayerparam
            self.default_lrateinitparam = lrateinitparam
            self.default_regparam = regparam

        else:
            raise ValueError('No model initiated')

    def get_SVM(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, param_dict=dict()):

        if 'pca_comp' not in param_dict:
            param_dict['pca_comp'] = self.default_pca_components
        if 'regC' not in param_dict:
            param_dict['regC'] = self.default_regCparam
        if 'kern' not in param_dict:
            param_dict['kern'] = self.default_kernparam

        return SVM(Xtrain, Xvalid, ytrain, yvalid, Xtest, pca_comp=param_dict['pca_comp'], regC=param_dict['regC'],
                   kern=param_dict['kern'], optimize=self.default_optimizeQ, verbose=self.default_verboseQ,
                   random_seed=self.default_rs, multi_jobs=self.default_multi_jobsQ)

    def get_GBC(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, param_dict=dict()):

        if 'pca_comp' not in param_dict:
            param_dict['pca_comp'] = self.default_pca_components
        if 'nest' not in param_dict:
            param_dict['nest'] = self.default_nestparam
        if 'lrate' not in param_dict:
            param_dict['lrate'] = self.default_lrateparam
        if 'mdepth' not in param_dict:
            param_dict['mdepth'] = self.default_mdepthparam

        return GBC(Xtrain, Xvalid, ytrain, yvalid, Xtest, pca_comp=param_dict['pca_comp'], nest=param_dict['nest'],
                   lrate=param_dict['lrate'], mdepth=param_dict['mdepth'], optimize=self.default_optimizeQ,
                   verbose=self.default_verboseQ, random_seed=self.default_rs, multi_jobs=self.default_multi_jobsQ)

    def get_NN(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, param_dict=dict()):

        if 'pca_comp' not in param_dict:
            param_dict['pca_comp'] = self.default_pca_components
        if 'hlayers' not in param_dict:
            param_dict['hlayers'] = self.default_hlayersparam
        if 'lrate' not in param_dict:
            param_dict['lrateinit'] = self.default_lrateinitparam
        if 'regparam' not in param_dict:
            param_dict['regparam'] = self.default_regparam

        return NN(Xtrain, Xvalid, ytrain, yvalid, Xtest, pca_comp=param_dict['pca_comp'], hlayers=param_dict['hlayers'],
                  lrateinit=param_dict['lrateinit'], regparam=param_dict['regparam'], optimize=self.default_optimizeQ,
                  verbose=self.default_verboseQ, random_seed=self.default_rs, multi_jobs=self.default_multi_jobsQ)


class TEClassification(Base):

    def __init__(self, train_enz_seq, test_enz_seq, label_file, train_feature_dirs, test_feature_dirs, use_feat=None,
                 hyper_param_file=None, model='SVM', random_seed=None, pca_components=55, n_models=5,
                 validation_fraction=0.25, optimize=False):

        self.random_seed = random_seed
        self.model = model
        self._pca_components = pca_components
        self.n_models = n_models
        self.validation_fraction = validation_fraction
        self.optimize = optimize
        self.test = True if test_feature_dirs is not None else False

        # initialize super class
        if self.model == 'SVM':
            super().__init__(pca_components=self._pca_components, random_seed=self.random_seed, optimizeQ=self.optimize)
        else:
            if self.model == 'GBC':
                super().__init__(pca_components=self._pca_components, random_seed=self.random_seed, SVM=False, GBC=True,
                                 optimizeQ=self.optimize)
            elif self.model == 'NN':
                super().__init__(pca_components=self._pca_components, random_seed=self.random_seed, SVM=False, NN=True,
                                 optimizeQ=self.optimize)
            else:
                raise ValueError('Wrong Model Assigned')

        self.object_map = {'SVM': self.get_SVM, 'NN': self.get_NN, 'GBC': self.get_GBC}

        # original data based on which everything is obtained
        df1 = pd.read_csv(train_enz_seq, header=None)
        df2 = pd.read_csv(label_file, header=None)
        self.train_df = df1.merge(df2, on=0)

        self.enz_names = self.train_df[0].values
        self.enz_idx = np.arange(len(self.enz_names))
        self.X = self.train_df.iloc[:, 1].values
        self.y = self.train_df.iloc[:, -1].values

        self.df_hyper_param = pd.read_csv(hyper_param_file).set_index('feat_name') if hyper_param_file is not None else None

        # training and validation data for general use
        self.X_train, self.X_valid, self.y_train, self.y_valid, self.enz_train, self.enz_valid, self.enz_train_idx, self.enz_valid_idx = train_test_split(
            self.X, self.y, self.enz_names, self.enz_idx, test_size=self.validation_fraction,
            random_state=self.random_seed)

        self.label_file = label_file

        # test data
        if self.test:
            self.test_df = pd.read_csv(test_enz_seq, header=None)
            self.test_enz_names = self.test_df[0].values
            self.X_test = self.test_df.iloc[:, 1].values
        else:
            self.X_test = None

        # kmer and gaakmer
        ng = NGModel(self.X_train, self.X_valid, self.X_test)
        gaang = GAANGModel(self.X_train, self.X_valid, self.X_test)
        self.feat_names = ['kmer', 'gaakmer']
        self.objects = [self.get_model_online('kmer', ng.x_train, ng.x_valid, self.y_train, self.y_valid, ng.x_test),
                        self.get_model_online('gaakmer', gaang.x_train, gaang.x_valid, self.y_train, self.y_valid,
                                              gaang.x_test)]

        # kernels
        kernel_names = ['spectrumKernel', 'mismatchKernel', 'gappyKernel']
        self.kernel_trainfeatdir = self.get_kernel_trainfeatdirs(train_feature_dirs)

        if self.test:
            self.kernel_testfeatdir = self.get_kernel_testfeatdirs(test_feature_dirs)
            kernel_objects = [self.get_model_kernel(kn, self.kernel_trainfeatdir, self.kernel_testfeatdir) for kn in
                              kernel_names]
        else:
            kernel_objects = [self.get_model_kernel(kn, self.kernel_trainfeatdir) for kn in kernel_names]

        self.feat_names.extend(kernel_names)
        self.objects.extend(kernel_objects)

        # ifeat
        if self.test:
            self.ifeat_traindirs, self.ifeat_testdirs = self.get_ifeat_trainfeatdirs(train_feature_dirs), \
                                                        self.get_ifeat_testfeatdirs(test_feature_dirs)
            self.ifeat_names, self.ifeat_trainfeatfiles = self.get_ifeat_trainfeatfiles(self.ifeat_traindirs)
            self.ifeat_names_test, self.ifeat_testfeatfiles = self.get_ifeat_testfeatfiles(self.ifeat_testdirs)
            assert self.ifeat_names == self.ifeat_names_test
            func_iter = list(
                zip(self.ifeat_names, self.ifeat_trainfeatfiles, self.ifeat_testfeatfiles))
            ifeat_objects = list(itertools.starmap(self.get_model_ifeat, func_iter))

        else:
            self.ifeat_traindirs = self.get_ifeat_trainfeatdirs(train_feature_dirs)
            self.ifeat_names, self.ifeat_trainfeatfiles = self.get_ifeat_trainfeatfiles(self.ifeat_traindirs)
            func_iter = list(zip(self.ifeat_names, self.ifeat_trainfeatfiles))
            ifeat_objects = list(itertools.starmap(self.get_model_ifeat, func_iter))

        self.feat_names.extend(self.ifeat_names)
        self.objects.extend(ifeat_objects)

        # pssm
        if self.test:
            self.pssm_traindirs, self.pssm_testdirs = self.get_pssm_trainfeatdirs(train_feature_dirs), \
                                                        self.get_pssm_testfeatdirs(test_feature_dirs)
            self.pssm_names, self.pssm_trainfeatfiles = self.get_pssm_trainfeatfiles(self.pssm_traindirs)
            self.pssm_names_test, self.pssm_testfeatfiles = self.get_pssm_testfeatfiles(self.pssm_testdirs)
            assert self.pssm_names == self.pssm_names_test
            
            func_iter = list(
                zip(self.pssm_names, self.pssm_trainfeatfiles, self.pssm_testfeatfiles))
            pssm_objects = list(itertools.starmap(self.get_model_pssm, func_iter))

        else:
            self.pssm_traindirs = self.get_pssm_trainfeatdirs(train_feature_dirs)
            self.pssm_names, self.pssm_trainfeatfiles = self.get_pssm_trainfeatfiles(self.pssm_traindirs)
            func_iter = list(zip(self.pssm_names, self.pssm_trainfeatfiles))
            pssm_objects = list(itertools.starmap(self.get_model_pssm, func_iter))

        self.feat_names.extend(self.pssm_names)
        self.objects.extend(pssm_objects)

        if use_feat is not None:
            assert self.n_models == len(use_feat)
            self.best_model_names, self.best_models = self.select_predef_models(use_feat)

        else:
            # select only the best models based on training or validation
            self.best_model_names, self.best_models = self.select_top_models(self.objects)

        # getting all model predictions together for ensemble
        if not self.test:
            self.all_model_preds = [o.ypredvalid for o in self.objects]
            self.best_model_preds = [o.ypredvalid for o in self.best_models]
            self.en = Ensemble(self.best_model_preds, self.y_valid)
            self.precision = precision_score(self.y_valid, self.en.preds, labels=[3], average='micro')
            self.recall = recall_score(self.y_valid, self.en.preds, labels=[3], average='micro')

        else:
            self.best_model_valid_preds = [o.ypredvalid for o in self.best_models]
            self.en_valid = Ensemble(self.best_model_valid_preds, self.y_valid)
            self.precision = precision_score(self.y_valid, self.en_valid.preds, labels=[3], average='micro')
            self.recall = recall_score(self.y_valid, self.en_valid.preds, labels=[3], average='micro')
            self.best_model_preds = [o.yhattest for o in self.best_models]
            self.en = Ensemble(self.best_model_preds)

        pass

    def get_kernel_trainfeatdirs(self, trainfeatdirs):
        kernel_trainfeaturefiledirs = [d for d in trainfeatdirs if 'kernel' in d]
        assert len(kernel_trainfeaturefiledirs) == 1
        return kernel_trainfeaturefiledirs[0]

    def get_kernel_testfeatdirs(self, testfeatdirs):
        kernel_testfeaturefiledirs = [d for d in testfeatdirs if 'kernel' in d]
        assert len(kernel_testfeaturefiledirs) == 1
        return kernel_testfeaturefiledirs[0]

    def get_ifeat_trainfeatdirs(self, trainfeatdirs):
        ifeat_traindirs = [d for d in trainfeatdirs if 'ifeatpro' in d]
        assert len(ifeat_traindirs) == 1
        return ifeat_traindirs[0]

    def get_ifeat_testfeatdirs(self, testfeatdirs):
        ifeat_testdirs = [d for d in testfeatdirs if 'ifeatpro' in d]
        assert len(ifeat_testdirs) == 1
        return ifeat_testdirs[0]

    def get_pssm_trainfeatdirs(self, trainfeatdirs):
        pssm_traindirs = [d for d in trainfeatdirs if 'pssmpro' in d]
        assert len(pssm_traindirs) == 1
        return pssm_traindirs[0]

    def get_pssm_testfeatdirs(self, testfeatdirs):
        pssm_testdirs = [d for d in testfeatdirs if 'pssmpro' in d]
        assert len(pssm_testdirs) == 1
        return pssm_testdirs[0]

    def get_ifeat_trainfeatfiles(self, ifeat_traindirs):

        ifeat_trainfiles = [ifeat_traindirs + f.name for f in os.scandir(ifeat_traindirs) if
                                       f.name.endswith('.csv')]

        featnames = [f.name.replace('.csv', '') for f in os.scandir(ifeat_traindirs) if
                                       f.name.endswith('.csv')]

        return featnames, ifeat_trainfiles

    def get_ifeat_testfeatfiles(self, ifeat_testdirs):

        ifeat_testfiles = [ifeat_testdirs + f.name for f in os.scandir(ifeat_testdirs) if
                                       f.name.endswith('.csv')]

        featnames = [f.name.replace('.csv', '') for f in os.scandir(ifeat_testdirs) if
                                       f.name.endswith('.csv')]

        return featnames, ifeat_testfiles

    def get_pssm_trainfeatfiles(self, pssm_traindirs):

        pssm_trainfiles = [pssm_traindirs + f.name for f in os.scandir(pssm_traindirs) if
                                       f.name.endswith('.csv')]

        featnames = [f.name.replace('.csv', '') for f in os.scandir(pssm_traindirs) if
                                       f.name.endswith('.csv')]

        return featnames, pssm_trainfiles

    def get_pssm_testfeatfiles(self, pssm_testdirs):

        pssm_testfiles = [pssm_testdirs + f.name for f in os.scandir(pssm_testdirs) if
                                       f.name.endswith('.csv')]

        featnames = [f.name.replace('.csv', '') for f in os.scandir(pssm_testdirs) if
                                       f.name.endswith('.csv')]

        return featnames, pssm_testfiles

    def get_model_online(self, model_name, X_train, X_valid, y_train, y_valid, X_test=None):

        if X_train.shape[1] < self._pca_components:
            self.default_pca_components = int(0.75 * X_train.shape[1])
        else:
            self.default_pca_components = self._pca_components

        if self.df_hyper_param is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyper_param.loc[model_name, 'pca_comp']
                param_dict_['regC'] = self.df_hyper_param.loc[model_name, 'regC']
                param_dict_['kernel'] = self.df_hyper_param.loc[model_name, 'kernel']

        else:
            param_dict_ = dict()

        if self.test:
            obj = self.object_map[self.model](X_train, X_valid, y_train, y_valid, X_test, param_dict=param_dict_)
        else:
            obj = self.object_map[self.model](X_train, X_valid, y_train, y_valid, param_dict=param_dict_)
        return obj

    def get_model_kernel(self, featname, train_file_prefix, test_file_prefix=None):

        featnamealias_dict = {'spectrumKernel': 'spec',
                              'gappyKernel': 'gap',
                              'mismatchKernel': 'mism'}

        alias = featnamealias_dict[featname]

        train_mat_file = train_file_prefix + alias + 'mat.npz'
        train_enz_name_file = train_file_prefix + alias + 'enz_names.txt'

        X = sparse.load_npz(train_mat_file)
        train_enz_names = np.genfromtxt(train_enz_name_file, dtype=str)

        X_train_feat, X_valid_feat = X[self.enz_train_idx, :], X[self.enz_valid_idx, :]

        if self.df_hyper_param is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyper_param.loc[featname, 'pca_comp']
                param_dict_['regC'] = self.df_hyper_param.loc[featname, 'regC']
                param_dict_['kernel'] = self.df_hyper_param.loc[featname, 'kernel']

        else:
            param_dict_ = dict()

        if self.test:
            test_mat_file = test_file_prefix + alias + 'mat.npz'
            test_enz_name_file = test_file_prefix + alias + 'enz_names.txt'
            X_test_feat = sparse.load_npz(test_mat_file)
            test_enz_names = np.genfromtxt(test_enz_name_file, dtype=str)

            obj = self.object_map[self.model](X_train_feat, X_valid_feat, self.y_train, self.y_valid, X_test_feat,
                                              param_dict=param_dict_)

        else:
            obj = self.object_map[self.model](X_train_feat, X_valid_feat, self.y_train, self.y_valid,
                                              param_dict=param_dict_)

        return obj

    def get_model_ifeat(self, featname, featfilename, testfeatfilename=None):

        df1 = pd.read_csv(featfilename, header=None)
        df2 = pd.read_csv(self.label_file, header=None)
        df_feat = df1.merge(df2, on=0).set_index(0) 
        df_feat_train = df_feat.loc[self.enz_train]
        df_feat_valid = df_feat.loc[self.enz_valid]
        X_train_feat, y_train_feat = df_feat_train.iloc[:, 0:-1].values, df_feat_train.iloc[:, -1].values
        X_valid_feat, y_valid_feat = df_feat_valid.iloc[:, 0:-1].values, df_feat_valid.iloc[:, -1].values

        if X_train_feat.shape[1] < self._pca_components:
            self.default_pca_components = int(0.75 * X_train_feat.shape[1])
        else:
            self.default_pca_components = self._pca_components

        if self.df_hyper_param is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyper_param.loc[featname, 'pca_comp']
                param_dict_['regC'] = self.df_hyper_param.loc[featname, 'regC']
                param_dict_['kernel'] = self.df_hyper_param.loc[featname, 'kernel']

        else:
            param_dict_ = dict()

        if self.test:

            df_feat_test = pd.read_csv(testfeatfilename, header=None).set_index(0)
            X_test_feat = df_feat_test.loc[self.test_enz_names].values
            if X_train_feat.shape[1] != X_test_feat.shape[1]:
                print(featfilename)

            obj = self.object_map[self.model](X_train_feat, X_valid_feat, y_train_feat, y_valid_feat, X_test_feat,
                                              param_dict=param_dict_)



        else:
            obj = self.object_map[self.model](X_train_feat, X_valid_feat, y_train_feat, y_valid_feat,
                                              param_dict=param_dict_)
        return obj

    def get_model_pssm(self, featname, featfilename, testfeatfilename=None):

        df1 = pd.read_csv(featfilename, header=None)
        df2 = pd.read_csv(self.label_file, header=None)
        df_feat = df1.merge(df2, on=0).set_index(0)
        df_feat_train = df_feat.loc[self.enz_train]
        df_feat_valid = df_feat.loc[self.enz_valid]
        X_train_feat, y_train_feat = df_feat_train.iloc[:, 0:-1].values, df_feat_train.iloc[:, -1].values
        X_valid_feat, y_valid_feat = df_feat_valid.iloc[:, 0:-1].values, df_feat_valid.iloc[:, -1].values

        if X_train_feat.shape[1] < self._pca_components:
            self.default_pca_components = int(0.75 * X_train_feat.shape[1])
        else:
            self.default_pca_components = self._pca_components

        if self.df_hyper_param is not None:
            param_dict_ = dict()
            if self.model == 'SVM':
                param_dict_['pca_comp'] = self.df_hyper_param.loc[featname, 'pca_comp']
                param_dict_['regC'] = self.df_hyper_param.loc[featname, 'regC']
                param_dict_['kernel'] = self.df_hyper_param.loc[featname, 'kernel']

        else:
            param_dict_ = dict()

        if self.test:

            df_feat_test = pd.read_csv(testfeatfilename, header=None).set_index(0)
            X_test_feat = df_feat_test.loc[self.test_enz_names].values
            if X_train_feat.shape[1] != X_test_feat.shape[1]:
                print(featfilename)

            obj = self.object_map[self.model](X_train_feat, X_valid_feat, y_train_feat, y_valid_feat, X_test_feat,
                                              param_dict=param_dict_)



        else:
            obj = self.object_map[self.model](X_train_feat, X_valid_feat, y_train_feat, y_valid_feat,
                                              param_dict=param_dict_)
        return obj

    def select_top_models(self, Os):
        o_valid_accs = [o.acc_valid for o in Os]  # if self.test else [o.acc_train for o in Os]
        sorted_idx = np.argsort(o_valid_accs)[::-1]
        best_idx = sorted_idx[:self.n_models]

        return np.array(self.feat_names)[best_idx], np.array(Os)[best_idx]

    def select_predef_models(self, pre_def_models):
        model_objs = []

        for fname, fobj in zip(self.feat_names, self.objects):
            if fname in pre_def_models:
                model_objs.append(fobj)

        return pre_def_models, model_objs

    def get_best_hp(self, model_obj):
        return tuple(model_obj.grid.best_params_.values()) if self.optimize else None

    def get_best_hps(self):
        hps = list(map(self.get_best_hp, self.objects))
        return list(zip(self.feat_names, hps))
