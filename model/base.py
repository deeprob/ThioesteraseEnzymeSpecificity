import numpy as np
from scipy import sparse
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class Model:

    def __init__(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, random_seed=None,
                 optimize=False, verbose=True, multi_jobs=True):

        # setting the random seed
        np.random.seed(random_seed)

        # setting some attributes
        self.sparse = sparse.issparse(Xtrain)  # for kernel methods which scales horribly with data
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.ytrain = ytrain
        self.yvalid = yvalid
        self.Xtest = Xtest
        self.verbose = verbose
        self.optimize = optimize
        self.parallel = multi_jobs

        # setting some more attributes
        if self.Xtest is not None:
            if self.sparse:
                # concatenate does not work with sparse matrices,  scipy vstack is required here
                self.X = sparse.vstack((self.Xtrain, self.Xvalid))
            else:
                self.X = np.concatenate((self.Xtrain, self.Xvalid), axis=0)
            self.y = np.concatenate((self.ytrain, self.yvalid), axis=0)

        pass

    def _run_model(self, model_obj):
        model_obj.fit(self.Xtrain, self.ytrain)
        ypredtrain = model_obj.predict(self.Xtrain)
        ypredvalid = model_obj.predict(self.Xvalid)
        acc_train = accuracy_score(self.ytrain, ypredtrain)
        acc_valid = accuracy_score(self.yvalid, ypredvalid)
        return model_obj, ypredtrain, ypredvalid, acc_train, acc_valid

    def _run_hpopt(self, model_obj, params):
        if self.parallel:
            n_jobs = -1
        else:
            n_jobs = 1

        grid = GridSearchCV(model_obj, param_grid=params, cv=5, n_jobs=n_jobs, scoring='accuracy', verbose=0)
        grid.fit(self.Xtrain, self.ytrain)
        return grid

    pass


class SVM(Model):

    def __init__(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, random_seed=None, pca_comp=40, regC=1, kern='rbf',
                 probability=False, optimize=False, verbose=True, classweight=None, multi_jobs=True):

        super(SVM, self).__init__(Xtrain, Xvalid, ytrain, yvalid, Xtest, random_seed, optimize, verbose, multi_jobs)

        pipeline = self._make_pipeline(pca_comp, regC, kern, random_seed, probability, classweight)
        self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(pipeline)

        if self.verbose:
            print('-' * 5 + 'Initial Model Evaluation' + '-' * 5)
            print('-' * 5 + 'Training Accuracy:' + str(self.acc_train) + '-' * 5)
            print('-' * 5 + 'Validation Accuracy:' + str(self.acc_valid) + '-' * 5)

        # Hyperparameter Optimization
        if self.optimize:
            if self.verbose:
                print('-' * 5 + 'Hyperparameter Optimization' + '-' * 5)

            if self.Xtrain.shape[1] < 55:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5 * shape), int(0.6 * shape)]
            else:
                try_pca = [40, 55]

            parameters = {'pca__n_components': try_pca,
                          'SVM__C': [0.1, 1, 20, 30],
                          'SVM__kernel': ['linear', 'rbf']}

            self.grid = self._run_hpopt(self.model, parameters)

            # print evaluation results
            if self.verbose:
                print("score = %3.2f" % (self.grid.score(self.Xvalid, self.yvalid)))
                print(self.grid.best_params_)

            best_pipeline = self.grid.best_estimator_
            self.model = best_pipeline
            self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(self.model)

        if self.Xtest is not None:
            self.model.fit(self.X, self.y)
            self.yhattrain = self.model.predict(self.X)
            self.yhattest = self.model.predict(self.Xtest)
            self.acc_tr = accuracy_score(self.y, self.yhattrain)

            pass

    def _make_pipeline(self, n_comp, c, k, rs, prob, cw):
        if not self.sparse:
            steps = [('normalize', Normalizer()), ('pca', PCA(n_components=n_comp, random_state=rs)),
                     ('SVM', SVC(C=c, gamma='scale', kernel=k, random_state=rs, max_iter=-1, probability=prob,
                                 class_weight=cw))]
        else:
            steps = [('normalize', Normalizer()), ('pca', TruncatedSVD(n_components=n_comp, random_state=rs)),
                     ('SVM', SVC(C=c, gamma='scale', kernel=k, random_state=rs, max_iter=-1, probability=prob,
                                 class_weight=cw))]

        pipe = Pipeline(steps)
        return pipe


class GBC(Model):
    def __init__(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, random_seed=None, pca_comp=20, nest=15, lrate=0.1,
                 mdepth=3, ssample=1, optimize=False, verbose=True, multi_jobs=True):

        super(GBC, self).__init__(Xtrain, Xvalid, ytrain, yvalid, Xtest, random_seed, optimize, verbose, multi_jobs)

        pipeline = self._make_pipeline(pca_comp, nest, lrate, mdepth, ssample, random_seed)
        self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(pipeline)

        if verbose:
            print('-' * 5 + 'Initial Model Evaluation' + '-' * 5)
            print('-' * 5 + 'Training Accuracy:' + str(self.acc_train) + '-' * 5)
            print('-' * 5 + 'Validation Accuracy:' + str(self.acc_valid) + '-' * 5)

        # Hyperparameter Optimization
        if self.optimize:
            if self.verbose:
                print('-' * 5 + 'Hyperparameter Optimization' + '-' * 5)

            if self.Xtrain.shape[1] < 55:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5 * shape), int(0.6 * shape)]
            else:
                try_pca = [40, 55]

            parameters = {'pca__n_components': try_pca,
                          'GBC__n_estimators': [100, 250],
                          'GBC__learning_rate': [0.1, 1],
                          'GBC__max_depth': [3, 5]}

            self.grid = self._run_hpopt(self.model, parameters)

            # print evaluation results

            if self.verbose:
                print("score = %3.2f" % (self.grid.score(self.Xvalid, self.yvalid)))
                print(self.grid.best_params_)

            best_pipeline = self.grid.best_estimator_
            self.model = best_pipeline
            self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(self.model)

        if self.Xtest is not None:
            self.model.fit(self.X, self.y)
            self.yhattrain = self.model.predict(self.X)
            self.yhattest = self.model.predict(self.Xtest)
            self.acc_tr = accuracy_score(self.y, self.yhattrain)

    def _make_pipeline(self, n_comp, n, lr, md, ss, rs):
        if not self.sparse:
            steps = [('normalize', Normalizer()), ('pca', PCA(n_components=n_comp, random_state=rs)),
                     ('GBC', GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=md, subsample=ss,
                                                        n_iter_no_change=5, tol=1e-4, random_state=rs))]
        else:
            steps = [('normalize', Normalizer()), ('pca', TruncatedSVD(n_components=n_comp, random_state=rs)),
                     ('GBC', GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=md, subsample=ss,
                                                        n_iter_no_change=5, tol=1e-4, random_state=rs))]
        pipe = Pipeline(steps)
        return pipe


class NN(Model):

    def __init__(self, Xtrain, Xvalid, ytrain, yvalid, Xtest=None, random_seed=None, pca_comp=20, hlayers=(10,),
                 lrateinit=0.1, regparam=0.01, optimize=False, verbose=True, multi_jobs=True):

        super(NN, self).__init__(Xtrain, Xvalid, ytrain, yvalid, Xtest, random_seed, optimize, verbose, multi_jobs)

        pipeline = self._make_pipeline(pca_comp, hlayers, lrateinit, regparam, random_seed)
        self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(pipeline)

        if verbose:
            print('-' * 5 + 'Initial Model Evaluation' + '-' * 5)
            print('-' * 5 + 'Training Accuracy:' + str(self.acc_train) + '-' * 5)
            print('-' * 5 + 'Validation Accuracy:' + str(self.acc_valid) + '-' * 5)

        # Hyperparameter Optimization
        if self.optimize:
            if self.verbose:
                print('-' * 5 + 'Hyperparameter Optimization' + '-' * 5)

            if self.Xtrain.shape[1] < 55:
                shape = self.Xtrain.shape[1]
                try_pca = [int(0.5 * shape), int(0.6 * shape)]
            else:
                try_pca = [40, 55]

            parameters = {'pca__n_components': try_pca,
                          'NN__hidden_layer_sizes': [(10, 5,), (10,)],
                          'NN__learning_rate_init': [0.01, 0.001],
                          'NN__alpha': [0.01, 0.001]}

            self.grid = self._run_hpopt(self.model, parameters)

            # print evaluation results
            if self.verbose:
                print("score = %3.2f" % (self.grid.score(self.Xvalid, self.yvalid)))
                print(self.grid.best_params_)

            best_pipeline = self.grid.best_estimator_
            self.model = best_pipeline
            self.model, self.ypredtrain, self.ypredvalid, self.acc_train, self.acc_valid = self._run_model(self.model)

        if self.Xtest is not None:
            self.model.fit(self.X, self.y)
            self.yhattrain = self.model.predict(self.X)
            self.yhattest = self.model.predict(self.Xtest)
            self.acc_tr = accuracy_score(self.y, self.yhattrain)

    def _make_pipeline(self, n_comp, h, lr, reg, rs):

        if not self.sparse:
            steps = [('scaler', StandardScaler()), ('pca', PCA(n_components=n_comp, random_state=rs)),
                     ('NN', MLPClassifier(hidden_layer_sizes=h, activation='logistic', solver='adam',
                                          learning_rate='adaptive', learning_rate_init=lr, alpha=reg,
                                          max_iter=200, random_state=rs))]
        else:
            steps = [('scaler', StandardScaler(with_mean=False)), ('pca', TruncatedSVD(n_components=n_comp,
                                                                                       random_state=rs)),
                     ('NN', MLPClassifier(hidden_layer_sizes=h, activation='logistic', solver='adam',
                                          learning_rate='adaptive', learning_rate_init=lr, alpha=reg, max_iter=200,
                                          random_state=rs))]
        pipe = Pipeline(steps)
        return pipe
