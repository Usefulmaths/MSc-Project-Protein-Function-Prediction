import dill as pickle
import hickle
import numpy as np
import os
import re
from collections import defaultdict
from fill_proteins import Protein


class Protein(object):

    def __init__(self, name, key, go_terms):
        self.name = name
        self.key = key
        self.go_terms = go_terms
        self.features = collections.defaultdict(list)

        self.go_encodings = dict()

    def update_features(self, feature_extraction_function, data_directory):
        feature_extraction_function(
            data_directory, self.name, self.key, self.features)

    def go_term_encoding(self, type, unique_go_terms):
        self.go_encodings[type] = go_encoding(
            self.go_terms[type], unique_go_terms)

if __name__ == '__main__':

    # Load in the dictionary containing the reduced protein set.
    with open('reduced_proteins.pkl', 'rb') as f:
        reduced_proteins = pickle.load(f)

    # Directory where protein train-test splits can be found.
    data_directory = './data/full_terms_test/'

    # List the files in the directory
    go_term_files = os.listdir(data_directory)

    # Instantiate the defaultdict that will store the protein train-test splits
    go_term_datasets = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    # For each GO term file
    for file in go_term_files[1:]:
        go_term = file[:-5]

        # Extract the protein train-test for positive and negative proteins.
        with open(data_directory + file, 'r') as f:
            negative = True
            train = False

            lines = f.readlines()

            for line in lines[6:]:
                if line != '# test:\n' and line != '\n' and line != '# train:\n' and line != '# positive set :\n' and line != '# negative set :\n':
                    protein = re.sub(r'\n', '', line)

                    if negative:
                        if train:
                            go_term_datasets[go_term]["train"][
                                "negative"].append(protein)
                        else:
                            go_term_datasets[go_term]["test"][
                                "negative"].append(protein)
                    else:
                        if train:
                            go_term_datasets[go_term]["train"][
                                "positive"].append(protein)
                        else:
                            go_term_datasets[go_term]["test"][
                                "positive"].append(protein)

                if line == '# positive set :\n':
                    negative = False
                    train = False

                if line == '# train:\n':
                    train = True

    # Save the protein train-test split default dictionary
    with open('./cached_material/protein_train_test_splits.pkl', 'wb') as f:
        pickle.dump(go_term_datasets, f)
