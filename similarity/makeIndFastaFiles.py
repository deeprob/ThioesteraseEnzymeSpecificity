fasta_file = '../data/seq/EnzymeFasta.fa'


def get_enzymes():
    enz_names = []
    sequences = []
    with open(fasta_file, 'rt') as file:
        for lines in file:
            if lines.startswith('>'):
                val = lines.strip().replace('>', '')
                enz_names.append(val)
            else:
                val = lines.strip()
                if val:
                    sequences.append(val)
    return enz_names, sequences


def make_individual_fasta_files(enz_names, sequences):

    for enz_name, enz_sequence in zip(enz_names, sequences):
        with open('fastafiles/' + enz_name + '.fa', 'w') as f:
            f.write('>'+enz_name+'\n')
            f.write(enz_sequence)
            f.write('\n')

    return


if __name__ == '__main__':
    en_names, seqs = get_enzymes()
    make_individual_fasta_files(en_names, seqs)
