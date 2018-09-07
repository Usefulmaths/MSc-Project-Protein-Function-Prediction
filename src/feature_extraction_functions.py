import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder


amino_acid_dict = {'M': 0, 'Q': 1, 'F': 2, 'Y': 3, 'C': 4, 'R': 5, 'V': 6, 'H': 7, 'Z': 8, 'E': 9, 'S': 10, 'T': 11,
                   'D': 12, 'N': 13, 'U': 14, 'I': 15, 'B': 16, 'G': 17, 'A': 18, 'K': 19, 'W': 20, 'P': 21, 'L': 22}

enc = OneHotEncoder()
enc.fit(np.array(list(amino_acid_dict.values())).reshape(-1, 1))


def o_glyc_extraction(data_directory, protein_name, key, feature_dictionary):
    '''
    Extracts the O-glycosylation feature of a protein.

    Arguments:
        data_directory: the directory in which the protein information can be
                        found.
        protein_name: the protein name under consideration.
        key: the protein key under consideration.
        feature_dictionary: the dictionary specific to the protein
                            where features are stored.
    '''
    directory = data_directory + protein_name + '/' + key

    O_glyc_dict = {'.': [1, 0, 0], 'T': [0, 1, 0], 'S': [0, 0, 1]}

    with open(directory + '.netOglyc') as f:
        lines = f.readlines()

        length = int(lines[0].split(' ')[-1][:-1])
        start_number = 1
        end_number = np.ceil(length / 80)

        start_index = int(start_number + end_number)
        end_index = int(start_index + end_number)

        sequence = []
        for line in lines[start_index:end_index]:
            sequence.append(re.sub(r'\n', '', line))

        sequence = ''.join(sequence)

        for index, amino in enumerate(sequence):
            feature = O_glyc_dict[amino]

            feature_dictionary[index].append(feature)


def n_glyc_extraction(data_directory, protein_name, key, feature_dictionary):
    '''
    Extracts the N-glycosylation feature of a protein.

    Arguments:
        data_directory: the directory in which the protein information can be
                        found.
        protein_name: the protein name under consideration.
        key: the protein key under consideration.
        feature_dictionary: the dictionary specific to the protein
                            where features are stored.
    '''
    directory = data_directory + protein_name + '/' + key

    N_glyc_dict = {'.': [1, 0], 'N': [0, 1]}

    with open(directory + '.netNglyc') as f:
        lines = f.readlines()

        length = int(lines[4].split('\t')[1].split('  ')[1][:-1])

        start_number = 5
        end_number = np.ceil(length / 80)

        start_index = int(start_number + end_number)
        end_index = int(start_index + end_number)

        sequence = []
        for line in lines[start_index:end_index]:
            sequence.append(re.sub(r'\s+\d+\n', '', line))

        sequence = ''.join(sequence)

        for index, amino in enumerate(sequence):
            feature = N_glyc_dict[amino]

            feature_dictionary[index].append(feature)


def amino_acid_extraction(data_directory, protein_name, key, feature_dictionary):
    '''
    Extracts the amino-acid sequence of a protein.

    Arguments:
        data_directory: the directory in which the protein information can be
                        found.
        protein_name: the protein name under consideration.
        key: the protein key under consideration.
        feature_dictionary: the dictionary specific to the protein
                            where features are stored.
    '''
    directory = data_directory + protein_name + '/' + key

    with open(directory + '.unmasked3.diso', 'r') as f:
        amino_acid_array = f.readlines()[5:]

        for index, amino_acid in enumerate(amino_acid_array):
            symbol = amino_acid.split()[1]
            encoded_sequence = enc.transform(
                [[amino_acid_dict[symbol]]]).toarray()
            feature_dictionary[index].append(encoded_sequence.reshape(-1, 1))


def disopred_extraction(data_directory, protein_name, key, feature_dictionary):
    '''
    Extracts the disorder predictions of a protein sequence.

    Arguments:
        data_directory: the directory in which the protein information can be
                        found.
        protein_name: the protein name under consideration.
        key: the protein key under consideration.
        feature_dictionary: the dictionary specific to the protein
                            where features are stored.
    '''
    directory = data_directory + protein_name + '/' + key

    with open(directory + '.unmasked3.diso', 'r') as f:
        amino_acid_array = f.readlines()[5:]

        for index, amino_acid in enumerate(amino_acid_array):
            info_array = amino_acid.split()
            feature = [info_array[3]]
            feature_dictionary[index].append(feature)


def secondary_structure_extraction(data_directory, protein_name, key, feature_dictionary):
    '''
    Extracts the secondary structure predictions of a protein sequence.

    Arguments:
        data_directory: the directory in which the protein information can be
                        found.
        protein_name: the protein name under consideration.
        key: the protein key under consideration.
        feature_dictionary: the dictionary specific to the protein
                            where features are stored.
    '''
    directory = data_directory + protein_name + '/' + key

    with open(directory + '.masked3.ss2', 'r') as f:
        amino_acid_array = f.readlines()[2:]

        for index, amino_acid in enumerate(amino_acid_array):
            symbol = amino_acid.split()[3:]
            feature = np.array(symbol)

            feature_dictionary[index].append(feature)
