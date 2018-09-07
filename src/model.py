import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
from time import time


class LSTM(nn.Module):
    '''
    A class representing a LSTM model which contains a forward method
    in order to perform a forward pass of the architecture.
    '''

    def __init__(self, input_size, number_of_neurons, go_output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.go_output_size = go_output_size

        self.lstm_layers = []

        self.lstm = nn.LSTM(input_size, number_of_neurons[
                            0], 1, batch_first=True, bidirectional=True).type(torch.FloatTensor).cuda()

        self.lstm_layers.append(self.lstm)

        for i in range(len(number_of_neurons) - 1):
            self.lstm = nn.LSTM(2 * number_of_neurons[i], number_of_neurons[
                                i + 1], 1, batch_first=True, bidirectional=True).type(torch.FloatTensor).cuda()

            self.lstm_layers.append(self.lstm)

        self.bn = nn.BatchNorm1d(number_of_neurons[-1])
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(number_of_neurons[-1], self.go_output_size)

    def forward(self, x, lengths):
        '''
        Arguments:
            x: a batch of features
            lengths: the corresponding protein lengths

        Returns: the LSTM output
        '''

        # Pad the protein sequences
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Instantiate the hidden and current state of the LSTM
        h0 = Variable(torch.zeros(
            2, x.size(0), self.number_of_neurons[0])).cuda()
        c0 = Variable(torch.zeros(
            2, x.size(0), self.number_of_neurons[0])).cuda()

        ht = h0
        ct = c0

        x = packed

        # Propagate through the LSTM layers.
        for layer in self.lstm_layers:
            lstm = layer
            x, (ht, ct) = lstm(x)

        # Unpack the sequences
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Store the last hidden-time step
        self.store_hidden = ht[-1]

        # Batch normalisation
        x = self.bn(ht[-1])

        # Tanh of the LSTM output
        x = F.tanh(x)

        # Dropout
        x = self.dropout(x)

        # Fully-connected Layer
        x = self.fc(x)

        return x


def split_sets(protein_features, train_proportion):
    '''
    Arguments:
        protein_features: a set protein sequence features.
        training_proportion: what proportion of data to go
                             into the training set.

    Returns:
        train_indices: the protein indices corresponding to the training set.
        test_indices: the protein indices corresponding to the testing set.
    '''

    # The number of proteins in the dataset
    number_of_proteins = protein_features.shape[0]

    # Shuffle the dataset
    indices = np.array(range(number_of_proteins))
    np.random.shuffle(indices)

    # Split into a training and test set
    train_indices = indices[:int(train_proportion * number_of_proteins)]
    test_indices = indices[int(train_proportion * number_of_proteins):]

    return train_indices, test_indices


def mask_padding(features, max_length):
    '''
    Arguments:
        features: a set of protein sequence features.
        max_length: the maximum length of protein sequences in dataset.

    Returns:
        lengths: the lengths of each protein in the set.
        indices: the index of where the padding starts.
    '''
    # features are batch examples

    lengths = []

    # For each protein in the dataset
    for example in features:
        # Find the indices where the padding starts
        padding = (example == -111).nonzero()

        # If there is no padding, set to the maximum.
        if len(padding) <= 0:
            size = torch.tensor(max_length / time_step)

        # Else, find length or set to 1
        else:
            if padding[0, 0] != 0:
                size = padding[0, 0]
            else:
                size = 1

        lengths.append(size)

    # Convert to PyTorch tensors.
    lengths, indices = torch.tensor(
        lengths, dtype=torch.int32).sort(descending=True)

    return lengths, indices


def f1_score_go(outputs, labels):
    '''
    Arguments:
        outputs: the outputs of the LSTM.
        labels: the corresponding labels of the data.

    Returns:
        f1_scores: an array containing F1 scores for each GO term.
        mean_f1_score: the mean of the F1 scores.
    '''

    rows, cols = outputs.shape
    f1_scores = []

    # For each GO term
    for c in range(cols):
        output = outputs[:, c]
        label = labels[:, c]

        f1 = f1_score(label, output)
        f1_scores.append(f1)

    mean_f1_score = np.mean(f1_scores)
    return f1_scores, mean_f1_score

if __name__ == '__main__':
    # Load in dictionary of the specific protein train-test splits.
    with open('/home/leloie/Codes/cached_material/protein_train_test_splits.pkl', 'rb') as f:
        protein_train_test_splits = pickle.load(f)

    # Load in training and test proteins for each GO term
    with open('/home/leloie/Codes/cached_material/go_term_protein_training.pkl', 'rb') as f:
        go_term_protein_training = pickle.load(f)

    # Load in GO term samples (top, middle, bottom).
    with open('/home/leloie/Codes/cached_material/GO_term_samples.pkl', 'rb') as f:
        go_term_samples = pickle.load(f)

    # Path to the psipred feature tensors.
    data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'

    # Set up the varying hyperparameters of the network
    layers = [[512], [256]]
    steps = [10, 20, 50]

    # GO term placeholder to be replaced when generating scripts.
    go_terms = ["go_term_placeholder"]

    # For each layer hyperparameter
    for layer in layers:

        # For each time-step parameter
        for step in steps:

            # For the GO term under consideration.
            for go_term in go_terms:

                # Retrieve the training and testing proteins the GO term.
                proteins = go_term_protein_training[go_term]

                protein_features = []
                protein_labels = []

                # Load the features and labels for the proteins.
                for index, protein in enumerate(proteins):
                    if(index % 100 == 0):
                        print("Loaded Proteins: %d" % index)

                    with open(data_directory + protein + '/features.pkl', 'rb') as f:
                        features = pickle.load(f)
                        protein_features.append(features)

                    with open(data_directory + protein + '/labels.pkl', 'rb') as f:
                        labels = pickle.load(f)
                        protein_labels.append(labels)

                protein_features = np.array(protein_features)
                protein_labels = np.array(protein_labels)

                # Split the protein set into a training and testing set.
                train_indices, test_indices = split_sets(protein_features, 0.7)

                X_train = protein_features[train_indices]
                X_test = protein_features[test_indices]

                y_train = protein_labels[train_indices]
                y_test = protein_labels[test_indices]

                # Set up the hyperparameters of the network.
                batch_size = 256
                epochs = 200
                time_step = step
                feature_size = 3
                input_size = time_step * feature_size
                num_layers = 3

                # Calculate max length of a protein for apdding.
                max_length = max(map(lambda x: len(x), protein_features))

                # Convert into PyTorch tensors.
                X_train_tensor = torch.Tensor(X_train)
                X_test_tensor = torch.Tensor(X_test)
                y_train_tensor = torch.Tensor(y_train)
                y_test_tensor = torch.Tensor(y_test)

                protein_features_tensor = torch.Tensor(protein_features)
                protein_labels_tensor = torch.Tensor(protein_labels)

                # Create a training and testing DataLoader
                train_data = TensorDataset(X_train_tensor, y_train_tensor)
                train_data_loader = DataLoader(train_data, batch_size)

                test_data = TensorDataset(X_test_tensor, y_test_tensor)
                test_data_loader = DataLoader(test_data, batch_size)

                # GO output size
                go_output_size = len(protein_labels[0])

                # Create model
                net = LSTM(input_size, layer, go_output_size)
                net = net.cuda()

                # Define the optimiser over the network parameters.
                optimizer = torch.optim.Adam(net.parameters())

                # Define the loss function.
                criterion = nn.MultiLabelSoftMarginLoss()

                # For each Epoch
                for epoch in range(epochs):
                    test_losses = []
                    start = time()

                    # For each batch in the training DataSet
                    for batch_idx, (data, target) in enumerate(train_data_loader):
                        # Make the features a PyTorch Variable
                        features = Variable(data).cuda()

                        # Reshape in order to feed into a LSTM
                        features = features.view(
                            (data.shape[0], int(max_length / time_step), input_size))

                        # Make the labels a PyTorch Variable
                        labels = Variable(target).cuda()

                        # Find the lengths and indices of each protein for
                        # padding.
                        lengths, indices = mask_padding(features, max_length)

                        features = features[indices]
                        labels = labels[indices]

                        # Reset the gradients of the optimiser to zero.
                        optimizer.zero_grad()

                        # Forward pass
                        outputs = net(features, lengths).cuda()

                        # Calculate training loss
                        train_loss = criterion(outputs, labels).cuda()

                        # Calculate gradients of the network
                        train_loss.backward()

                        # Take a gradient step update.
                        optimizer.step()

                    test_predictions = []
                    test_labels = []
                    test_outputs = []

                    # For each batch in the testing DataSet
                    for batch_idx, (data, target) in enumerate(test_data_loader):
                        # Make the features a PyTorch Variable
                        features = Variable(data).cuda()

                        # Reshape in order to feed into a LSTM
                        features = features.view(
                            (data.shape[0], int(max_length / time_step), input_size))

                        # Make the labels a PyTorch Variable
                        labels = Variable(target).cuda()

                        # Find the lengths and indices of each protein for
                        # padding.
                        lengths, indices = mask_padding(features, max_length)

                        features = features[indices]
                        labels = labels[indices]

                        # Forward pass
                        outputs = net(features, lengths).cuda()

                        # Convert output to probabilities
                        sigmoid_output = F.sigmoid(outputs)

                        # Calculate the test loss
                        test_loss = criterion(outputs, labels).cuda()

                        labels = labels.data.cpu().numpy()

                        # Make prediction for each protein function
                        predictions = sigmoid_output > 0.5

                        test_losses.append(float(test_loss))
                        test_predictions.append(predictions)
                        test_labels.append(labels)
                        test_outputs.append(
                            net.store_hidden.data.cpu().numpy())

                    test_predictions = np.concatenate(
                        tuple(test_predictions), axis=0)
                    test_labels = np.concatenate(tuple(test_labels), axis=0)

                    # Calculate F1 scores
                    test_f1_array, test_f1 = f1_score_go(
                        test_labels, test_predictions)
                    end = time()

                    # Write a log file during training
                    with open('/cluster/project1/FFPredRNN/data/logs/logs_' + str(go_term) + '.txt', 'a') as f:
                        f.write("(%s, %d, %d) -> Epoch: %d \t Test loss: %f \t Test F1 score: %f \t Time: %f \t Time Left: %f \n" % (
                            go_term, layer[0], step, epoch, np.mean(test_losses), test_f1, end - start, (epochs - epoch) * (end - start)))

                # Save the performance of the networks for each hyperparameter
                with open('/cluster/project1/FFPredRNN/data/hyperparameter_results.txt', 'a') as f:
                    f.write('%s, %d, %d, %f\n' %
                            (go_term, layer[0], step, test_f1))
