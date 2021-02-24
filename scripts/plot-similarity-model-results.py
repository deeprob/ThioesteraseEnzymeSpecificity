import matplotlib.pyplot as plt
import numpy as np


def get_model_results(filename):

    precs = []
    recs = []
    accs = []

    with open(filename, 'rt') as f:
        for lines in f:
            vals = lines.strip().split(',')
            precs.append(float(vals[0]))
            recs.append(float(vals[1]))
            accs.append(float(vals[2]))

    return precs, recs, accs


def save_fig(metric, metric_name):
    filename = f'../similarity/results/{metric_name}.png'
    plt.hist(metric, bins=5, color='b')
    plt.grid()
    plt.xlim(min(metric), max(metric) + 0.1)
    plt.xlabel(f'{metric_name} scores')
    plt.ylabel('Counts')
    plt.title(f'{metric_name} Distribution across 10,000 simulations')
    plt.savefig(filename)
    return


def save_model_report(precs, recs, accs):
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_acc = np.mean(accs)
    min_prec = min(precs)
    min_rec = min(recs)
    min_acc = min(accs)
    max_prec = max(precs)
    max_rec = max(recs)
    max_acc = max(accs)

    with open('../similarity/results/model_report.csv', 'w') as f:
        f.write(',min_precision,min_recall,min_accuracy,max_precision,max_recall,max_accuracy,mean_precision,mean_recall,mean_accuracy')
        f.write('\n')
        f.write(f'Similarity_Model,{min_prec},{min_rec},{min_acc},{max_prec},{max_rec},{max_acc},{mean_prec},{mean_rec},{mean_acc}')
        f.write('\n')
    return


if __name__ == '__main__':
    sim_model_results_file = '../similarity/results/model_sims.csv'

    precisions, recalls, accuracies = get_model_results(sim_model_results_file)

    save_fig(precisions, 'Precision')
    save_fig(recalls, 'Recall')
    save_fig(accuracies, 'Accuracy')

    save_model_report(precisions, recalls, accuracies)
