import numpy as np
import pickle
import collections
import hickle
import os
from fill_proteins import Protein


def extract_matrices(proteins, go_domain, name):
    '''
    Extracts and saves a list of feature matrices derived
    from the features stored in the protein objects as well
    as their corresponding labels.

    Arguments:
        proteins: an array of Protein objects
        go_domain: the GO sub-ontology to consider
        name: naming convention for the features extracted
    '''
    proteins = proteins[go_domain]
    proteins_flattened_features = []
    proteins_label = []
    protein_names = []

    # For each protein
    for index, protein in enumerate(proteins):
        protein_names.append(protein.name)

        if(index % 100 == 0):
            print("Converted Proteins: %d" % index, end="\r")

        # Create the features matrices
        unflattened_features = np.array(
            [features for features in protein.features.values()])

        flattened_features = []
        for features in unflattened_features:
            flattened_features_single = []
            for feature in features:
                for value in feature:
                    flattened_features_single.append(value)

            flattened_features.append(flattened_features_single)

        proteins_flattened_features.append(np.array(flattened_features))

        proteins_label.append(protein.go_encodings[go_domain])

    print("\n")
    number_of_features = proteins_flattened_features[0].shape[1]
    proteins_flattened_features = padding(
        np.array(proteins_flattened_features), number_of_features)

    proteins_label = np.array(
        proteins_label, dtype=np.int32).reshape(len(proteins), -1)

    # Save the feature matrices and labels
    for index, features in enumerate(proteins_flattened_features):
        directory = './data/' + go_domain + '/tensors/' + \
            name + '/' + protein_names[index]
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + '/features.pkl', 'wb') as f:
            pickle.dump(features, f)
        with open(directory + '/labels.pkl', 'wb') as f:
            pickle.dump(proteins_label[index], f)

        if(index % 100 == 0):
            print('Matrices saved: %d' % index, end="\r")
    print("\n")


def padding(protein_features, number_of_features):
    '''
    Arguments:
        protein_features: the features of a protein
        number_of_features: number of features under consideration

    Returns:
        padded_sequences: the padded features of a protein
    '''
    max_length = max(map(lambda x: len(x), protein_features))

    padded_sequences = []

    for sequence in protein_features:
        padded_features = []
        for index, features in enumerate(sequence):
            padded_features.append(
                np.array([float(feature) for feature in features]))

        # Pads with the value -111.0
        for i in range(max_length - len(padded_features)):
            padded_features.append(
                np.array([-111.0 for i in range(number_of_features)]))

        padded_sequences.append(padded_features)

    padded_sequences = np.array(padded_sequences)

    return padded_sequences

if __name__ == '__main__':
    # Load in the reduced proteins.
    with open('cached_material/filled_reduced_proteins.pkl', 'rb') as f:
        proteins = pickle.load(f)

    # Extract feature matrices for proteins in all three sub-ontologies
    extract_matrices(proteins, 'biological_process', name='psipred')
    extract_matrices(proteins, 'molecular_function', name='psipred')
    extract_matrices(proteins, 'cellular_component', name='psipred')
