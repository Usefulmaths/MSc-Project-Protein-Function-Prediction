from feature_extraction_functions import *
import pickle
import collections


class Protein(object):
    '''
    A class that represents a protein object,
    used to store properties of a protein such as
    its name, collection of features, and assigned
    GO terms.
    '''

    def __init__(self, name, key, go_terms):
        self.name = name
        self.key = key
        self.go_terms = go_terms
        self.features = collections.defaultdict(list)

        self.go_encodings = dict()

    def update_features(self, feature_extraction_function, data_directory):
        '''
        Extracts and adds a particular feature to the protein
        feature dictionary.

        Arguments:
            feature_extraction_function: a function that is used to extract
                                         a particularly type of feature.
            data_directory: the directory where protein features can be found
        '''
        feature_extraction_function(
            data_directory, self.name, self.key, self.features)

    def amino_acid_count(self):
        '''
        Calculate the amino acid count of the protein.

        Returns:
            amino_acid_count: number of amino acids in the protein.
        '''

        amino_acid_count = len(self.features.keys())

        return amino_acid_count

    def num_features(self):
        '''
        Calculates the number of features stored in this protein.

        Returns:
            number_of_features: the number of features stored
                                within this protein
        '''

        number_of_features = len(list(self.features.values())[0])

        return number_of_features

if __name__ == '__main__':

    # Load in the reduced protein objects
    with open('cached_material/reduced_proteins.pkl', 'rb') as f:
        proteins = pickle.load(f)

    # Directory where raw feature files can be found
    data_directory = './data/RawFeaturesFFPredRNN/'

    # Create a list of unique protein objects
    unique_proteins = set()
    for protein_set in proteins.values():
        for protein in protein_set:
            unique_proteins.add(protein)

    # Fill all the protein objects with features.
    for index, protein in enumerate(list(unique_proteins)):
        if index % 100 == 0:
            print("Proteins Created: %d" % (index), end='\r')

        protein.update_features(secondary_structure_extraction, data_directory)

    # Save the filled proteins
    with open('./cached_material/filled_reduced_proteins.pkl', 'wb') as f:
        pickle.dump(proteins, f)
