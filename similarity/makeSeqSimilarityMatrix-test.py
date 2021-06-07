import os
import subprocess
import numpy as np
import re


files_dir = './fastafiles/'
test_files_dir = "./testfastafiles/"
filenames = [files_dir + f.name for f in os.scandir(files_dir) if f.name.endswith('.fa')]
test_filenames = [test_files_dir + f.name for f in os.scandir(test_files_dir) if f.name.endswith('.fa')]

similarity_matrix = np.zeros((len(test_filenames), len(filenames)))


def get_seq_similarity(file1, file2):
    output = subprocess.getoutput(f'bash seq_similarity.sh {file1} {file2}')
    value = output.strip().split()

    return float(value[0])


def get_num(file):
    pattern = re.compile("^.+/+enz_(\d+).fa$")
    m = re.match(pattern, file)
    return m.group(1)


def get_mat_idx(file1, file2):
    file1 = os.path.basename(file1)
    test_file_dict = {"test_enz_161.fa":0, "test_enz_176.fa":1, "test_enz_177.fa":2, "test_enz_188.fa":3}
    file1_idx = test_file_dict[file1]
    file2_idx = get_num(file2)
    return int(file1_idx), int(file2_idx)


def save_similarity_matrix():
    np.savetxt('./test_similarity_matrix.txt', similarity_matrix, delimiter=',')
    return


if __name__ == '__main__':

    for i, file in enumerate(test_filenames):
        print("*", end="")
        other_files = filenames
        file1 = file
        for file2 in other_files:
            sim_val = get_seq_similarity(file1, file2)
            row_idx, col_idx = get_mat_idx(file1, file2)
            similarity_matrix[row_idx, col_idx] = sim_val

    save_similarity_matrix()






