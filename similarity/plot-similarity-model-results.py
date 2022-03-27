import matplotlib.pyplot as plt
import numpy as np


def get_model_results(filename):

    precs = []
    recs = []
    accs = []
    mccs = []

    with open(filename, 'rt') as f:
        for lines in f:
            vals = lines.strip().split(',')
            precs.append(float(vals[0]))
            recs.append(float(vals[1]))
            accs.append(float(vals[2]))
            mccs.append(float(vals[3]))

    return precs, recs, accs, mccs


def save_fig(metric, metric_name):
    filename = f"../similarity/results/{metric_name}.png"
    plt.hist(metric, bins=5, color='b')
    plt.grid()
    plt.xlim(min(metric), max(metric) + 0.1)
    plt.xlabel(f'{metric_name} scores')
    plt.ylabel('Counts')
    plt.title(f'{metric_name} Distribution across 10,000 simulations')
    plt.savefig(filename)
    return


def save_prec_acc(prec, accs):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].hist(prec, color='sandybrown')
    axes[0].set_xlabel('Precision Scores')
    axes[0].set_title('Precision Score Distribution of Similarity Model')

    axes[1].hist(accs, color='darkslateblue')
    axes[1].set_xlabel('Accuracy Scores')
    axes[1].set_title('Accuracy Score Distribution of Similarity Model')

    figure.text(0, 0.5, 'Counts', va='center', rotation='vertical')
    figure.tight_layout()
    plt.savefig('../similarity/results/score_dist.png')
    return


def save_model_report(precs, recs, accs, mccs):
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_acc = np.mean(accs)
    mean_mcc = np.mean(mccs)
    min_prec = min(precs)
    min_rec = min(recs)
    min_acc = min(accs)
    min_mcc = min(mccs)
    max_prec = max(precs)
    max_rec = max(recs)
    max_acc = max(accs)
    max_mcc = max(mccs)

    with open('../similarity/results/model_report.csv', 'w') as f:
        f.write(',min_precision,min_recall,min_accuracy,min_mcc,max_precision,max_recall,max_accuracy,max_mcc,mean_precision,mean_recall,mean_accuracy,mean_mcc')
        f.write('\n')
        f.write(f'Similarity_Model,{min_prec},{min_rec},{min_acc},{min_mcc},{max_prec},{max_rec},{max_acc},{max_mcc},{mean_prec},{mean_rec},{mean_acc},{mean_mcc}')
        f.write('\n')
    return


if __name__ == '__main__':
    sim_model_results_file = '../similarity/results/model_sims.csv'

    precisions, recalls, accuracies, mccs = get_model_results(sim_model_results_file)

    save_fig(precisions, 'Precision')
    save_fig(recalls, 'Recall')
    save_fig(accuracies, 'Accuracy')
    save_fig(mccs, 'MCC')
    save_prec_acc(precisions, accuracies)
    
    save_model_report(precisions, recalls, accuracies, mccs)
