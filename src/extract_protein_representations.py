import pickle
import numpy as np
import torch
from torch.autograd import Variable
from FFPred_model import LSTM, mask_padding

# Load in dictionary of the specific protein train-test splits.
with open('./cached_material/protein_train_test_splits.pkl', 'rb') as f:
    protein_train_test_splits = pickle.load(f)

# Load in training and test proteins for each GO term
with open('./cached_material/go_term_protein_training.pkl', 'rb') as f:
    go_term_protein_training = pickle.load(f)

# Load in GO term samples (top, middle, bottom).
with open('./cached_material/GO_term_samples.pkl', 'rb') as f:
    go_term_samples = pickle.load(f)

with open('./cached_material/protein_name_key_map.pkl', 'rb') as f:
    protein_name_key_map = pickle.load(f)

with open('/home/leloie/Codes/cached_material/best_hyper_dict.pkl', 'rb') as f:
    best_hyper_dict = pickle.load(f)

if __name__ == '__main__':

    # Directory where psipred tensors are stored
    data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'

    # GO terms to convert protein raw features into LSTM representations
    go_terms = ['GO0051047', 'GO0060828', 'GO0034765', 'GO0045893', 'GO0048839', 'GO0051048', 'GO0035725', 'GO0046777', 'GO0006163', 'GO0070925', 'GO0071417', 'GO0006310', 'GO0007067', 'GO0016485', 'GO0030335', 'GO0000209', 'GO0016568', 'GO0009101', 'GO0030258', 'GO0010628', 'GO0046474', 'GO0006898',
                'GO0072657', 'GO0032956', 'GO0006402', 'GO0006643', 'GO0044257', 'GO0030522', 'GO0006935', 'GO0046488', 'GO0006457', 'GO0016570', 'GO0045862', 'GO0002431', 'GO0045861', 'GO0007015', 'GO0071593', 'GO0050907', 'GO0016477', 'GO0007265', 'GO0009116', 'GO0051146', 'GO0030036', 'GO0007018', 'GO0048872']

    # For each GO term
    for go_term in go_terms:
        # Retrieve training and set splits
        proteins = protein_train_test_splits[go_term]

        proteins_train = proteins['train']
        proteins_test = proteins['test']

        # Concatenate the positive and negative protein data
        proteins_train = np.concatenate(
            (proteins_train['positive'], proteins_train['negative']), axis=0)
        proteins_test = np.concatenate(
            (proteins_test['positive'], proteins_test['negative']), axis=0)

        protein_features_train = []
        protein_labels_train = []

        protein_features_test = []
        protein_labels_test = []

        protein_svm_train = []
        protein_svm_test = []

        # For each protein in the training set
        for index, protein in enumerate(proteins_train):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index)

            try:
                # Load in features of that protein
                with open(data_directory + protein + '/features.pkl', 'rb') as f:
                    features = pickle.load(f)
                    protein_features_train.append(features)

            except(FileNotFoundError):
                proteins_train = list(
                    filter(lambda x: x != protein, proteins_train))
                print("%s not found." % protein)

        # For each protein in the testing set
        for index, protein in enumerate(proteins_test):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index)

            try:
                # Load in features of that protein
                with open(data_directory + protein + '/features.pkl', 'rb') as f:
                    features = pickle.load(f)
                    protein_features_test.append(features)

            except(FileNotFoundError):
                proteins_test = list(
                    filter(lambda x: x != protein, proteins_test))
                print("%s not found." % protein)

        protein_features_train = np.array(protein_features_train)
        protein_features_test = np.array(protein_features_test)

        # Convert into PyTorch Variables
        protein_features_train = Variable(
            torch.Tensor(protein_features_train)).cuda()
        protein_features_test = Variable(
            torch.Tensor(protein_features_test)).cuda()

        # Extract the best hyperparameters found for that GO term
        neurons, time_step = best_hyper_dict[go_term]

        # Set up hyperparameters of the network
        feature_size = 3
        input_size = time_step * feature_size
        num_layers = 3
        go_output_size = 381
        max_length = 2000

        # Reshape and pad for the inputing into the LSTM
        protein_features_train = protein_features_train.view(
            (-1, int(max_length / time_step), input_size))
        train_lengths, train_indices = mask_padding(
            protein_features_train, max_length)
        protein_features_train = protein_features_train[train_indices]

        # Reshape and pad for the inputing into the LSTM
        protein_features_test = protein_features_test.view(
            (-1, int(max_length / time_step), input_size))
        test_lengths, test_indices = mask_padding(
            protein_features_test, max_length)
        protein_features_test = protein_features_test[test_indices]

        # Create model
        net = LSTM(input_size, [neurons], go_output_size)
        net = net.cuda()

        print('Loading in network...')

        # File to load the optimised LSTM networks from.
        file = '/cluster/project1/FFPredRNN/data/biological_process/psipred/go_terms/' + \
            go_term + '/model_optimised.pt'

        if os.path.exists(file):
            net.load_state_dict(torch.load(file))
        else:
            print("This file does not exist: %s" % (go_term))
            continue

        # Perform a forward pass on the training proteins.
        print('Performing forward pass on training...')

        train_representations = []
        for i in range(5):
            batch_size = int(np.ceil(len(protein_features_train) / 5))

            net(protein_features_train[i * batch_size: (i + 1) * batch_size],
                train_lengths[i * batch_size: (i + 1) * batch_size])

            train_representation = net.store_hidden

            # Store the LSTM representations for the training proteins.
            train_representations.append(net.store_hidden.data.cpu().numpy())

        # Perform a forward pass on the testing.
        print('Performing forward pass on testing...')

        test_representations = []
        for i in range(5):
            batch_size = int(np.ceil(len(protein_features_test) / 5))

            net(protein_features_test[i * batch_size: (i + 1) * batch_size],
                test_lengths[i * batch_size: (i + 1) * batch_size])

            test_representation = net.store_hidden

            # Store the LSTM representations for the testing proteins.
            test_representations.append(net.store_hidden.data.cpu().numpy())

        train_representation = np.concatenate(
            tuple(train_representations), axis=0)
        test_representation = np.concatenate(
            tuple(test_representations), axis=0)

        train_copy = train_representation.copy()
        for i in range(len(train_representation)):
            train_representation[train_indices[i]] = train_copy[i]

        test_copy = test_representation.copy()
        for i in range(len(test_representation)):
            test_representation[test_indices[i]] = test_copy[i]


        # Save the training protein representations
        for index, protein in enumerate(proteins_train):
            if index % 200 == 0:
                print('Saving training protein: %d' % index)
            directory = '/cluster/project1/FFPredRNN/data/biological_process/psipred/optimised_go_terms/' + \
                go_term + '/train/' + protein

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory + '/representation.pkl', 'wb') as f:
                pickle.dump(train_representation[index], f)

        # Save the testing protein representations
        for index, protein in enumerate(proteins_test):
            if index % 200 == 0:
                print('Saving testing protein: %d' % index)

            directory = '/cluster/project1/FFPredRNN/data/biological_process/psipred/optimised_go_terms/' + \
                go_term + '/test/' + protein

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory + '/representation.pkl', 'wb') as f:
                pickle.dump(test_representation[index], f)
