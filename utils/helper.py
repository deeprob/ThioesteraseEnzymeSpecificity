from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
import sys
sys.path.append("../")

from model.classifier import TEClassification




def get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = TEClassification(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, 
                          hyper_param_file=hyper_param_file, random_seed=rs, n_models=k, model=base_algo, optimize=opt)
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


def get_mcc(y,yhat):
    return round(matthews_corrcoef(y,yhat),2)


def get_metrics(val_iter):
    return get_precision(*val_iter), get_recall(*val_iter), get_accuracy(*val_iter), get_mcc(*val_iter)


def get_validation_iter(obj):
    return obj.yvalid,obj.ypredvalid


def indfeat_performance(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    val_iters = list(map(get_validation_iter, te.objects))
    mets = list(map(get_metrics, val_iters))
    return list(zip(te.feat_names, mets))


# # Parametric sweep of ensemble model hyperparameter k



def ensemble_pred(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs):
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    return te.y_valid,te.en.preds

def ensemble_metrics(val_iter):
    return get_metrics(val_iter)


# # Model evaluation all three categories

def get_metrics_lab2(val_iter):
    return get_precision(*val_iter,label=2), get_recall(*val_iter,label=2), get_accuracy(*val_iter)

def get_metrics_lab1(val_iter):
    return get_precision(*val_iter,label=1), get_recall(*val_iter,label=1), get_accuracy(*val_iter)

# # Test Predictions

def get_test_map_dict(filename):
    map_dict = dict()
    with open(filename,'r') as f:
        for lines in f:
            data = lines.strip().split(",")
            map_dict[data[0]] = data[1]
    return map_dict


def function_predict_test(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs, test_enz_map_file):
    enz_map_dict = get_test_map_dict(test_enz_map_file)
    names = []
    te = get_te(enz_file, test_enz_file, label_file, train_feat_dirs, test_feat_dirs, hyper_param_file, base_algo, k, opt, rs)
    for name,pred in zip(te.testenz_names, te.en.preds):
        if pred == 3:
            names.append(enz_map_dict[name])
    return names

