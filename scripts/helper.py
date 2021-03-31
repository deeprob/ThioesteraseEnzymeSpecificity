from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys
sys.path.append("../")

from model.classifier import TEClassification




def get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = TEClassification(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, 
                          hyperparamfile=hyper_param_file, random_seed=rs, n_models=k, model=base_algo, optimize=opt)
    return te


def check_base_performance(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    return te.precision, te.recall, te.en.acc


# # Individual Hyperparameter Optimization



def best_hps(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    return te.get_best_hps()


# # Individual Feature Performance Measure



def get_precision(y,yhat,label=3):
    return round(precision_score(y,yhat,labels=[label],average='micro'),2)


def get_recall(y,yhat,label=3):
    return round(recall_score(y,yhat,labels=[label],average='micro'),2)


def get_accuracy(y,yhat):
    return round(accuracy_score(y,yhat),2)


def get_metrics(val_iter):
    return get_precision(*val_iter), get_recall(*val_iter), get_accuracy(*val_iter)


def get_validation_iter(obj):
    return obj.yvalid,obj.ypredvalid


def indfeat_performance(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    val_iters = list(map(get_validation_iter, te.objects))
    mets = list(map(get_metrics, val_iters))
    return list(zip(te.featnames, mets))


# # Parametric sweep of ensemble model hyperparameter k



def ensemble_pred(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    return te.y_valid,te.en.preds

def ensemble_metrics(val_iter):
    return get_metrics(val_iter)

