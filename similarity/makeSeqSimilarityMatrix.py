import os
import subprocess
import numpy as np
import re


files_dir = './fastafiles/'
filenames = [files_dir + f.name for f in os.scandir(files_dir) if f.name.endswith('.fa')]
similarity_matrix = np.zeros((len(filenames), len(filenames)))


def get_seq_similarity(file1, file2):
    output = subprocess.getoutput(f'bash seq_similarity.sh {file1} {file2}')
    value = output.strip().split()

    return float(value[0])


def get_num(file):
    pattern = re.compile("^.+/+enz_(\d+).fa$")
    m = re.match(pattern, file)
    return m.group(1)


def get_mat_idx(file1, file2):
    file1_idx = get_num(file1)
    file2_idx = get_num(file2)
    return int(file1_idx), int(file2_idx)


def save_similarity_matrix():
    np.savetxt('./similarity_matrix.txt', similarity_matrix, delimiter=',')
    return


if __name__ == '__main__':

    for i, file in enumerate(filenames):
        print(i)
        other_files = filenames[i+1:]
        file1 = file
        for file2 in other_files:
            sim_val = get_seq_similarity(file1, file2)
            row_idx, col_idx = get_mat_idx(file1, file2)
            similarity_matrix[row_idx, col_idx] = sim_val
            similarity_matrix[col_idx, row_idx] = sim_val

    save_similarity_matrix()






