import pickle
import numpy as np
import torch
import dill as pickle
import hickle
import os
import subprocess

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from time import time
from model import LSTM, mask_padding, f1_score

if __name__ == '__main__':
    # Load in the protein train test splits.
    with open('/home/leloie/Codes/cached_material/protein_train_test_splits.pkl', 'rb') as f:
        protein_train_test_splits = pickle.load(f)

    # Load in training and test proteins for each GO term
    with open('/home/leloie/Codes/cached_material/go_term_protein_training.pkl', 'rb') as f:
        go_term_protein_training = pickle.load(f)

    # Load in GO term samples (top, middle, bottom).
    with open('/home/leloie/Codes/cached_material/GO_term_samples.pkl', 'rb') as f:
        go_term_samples = pickle.load(f)

    data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'

    # GO term placeholder to be replaced when generating script.
    go_terms = ["go_term_placeholder"]

    for go_term in go_terms:
        proteins = go_term_protein_training[go_term]

    protein_features = []
    protein_labels = []

    # Read in the protein features and labels
    for index, protein in enumerate(proteins):
        if(index % 100 == 0):
            print("Loaded Proteins: %d" % index, end="\r")

        with open(data_directory + protein + '/features.pkl', 'rb') as f:
            features = pickle.load(f)
            protein_features.append(features)

        with open(data_directory + protein + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            protein_labels.append(labels)

    # Load in the best hyperparameter dictionary
    with open('/home/leloie/Codes/cached_material/best_hyper_dict.pkl', 'rb') as f:
        best_hyper_dict = pickle.load(f)

    protein_features = np.array(protein_features)
    protein_labels = np.array(protein_labels)

    X_train = protein_features
    y_train = protein_labels

    # Set up the hyperparameters
    batch_size = 256
    epochs = 200
    neurons, time_step = best_hyper_dict[go_terms[0]]
    feature_size = 3
    input_size = time_step * feature_size
    num_layers = 3

    max_length = max(map(lambda x: len(x), protein_features))

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)

    protein_features_tensor = torch.Tensor(protein_features)
    protein_labels_tensor = torch.Tensor(protein_labels)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_data_loader = DataLoader(train_data, batch_size)

    # GO output size
    go_output_size = len(protein_labels[0])

    # Create model
    net = LSTM(input_size, [neurons], go_output_size)
    net = net

    # Define the optimiser over the network parameters.
    optimizer = torch.optim.Adam(net.parameters())

    # Define the loss function.
    criterion = nn.MultiLabelSoftMarginLoss()

    # For each epoch
    for epoch in range(epochs):
        test_losses = []
        start = time()

        # For each epoch in the training data
        for batch_idx, (data, target) in enumerate(train_data_loader):
            features = Variable(data)
            features = features.view(
                (data.shape[0], int(max_length / time_step), input_size))

            labels = Variable(target)

            # Padding
            lengths, indices = mask_padding(features, max_length)

            features = features[indices]
            labels = labels[indices]

            # Reset optimiser gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(features, lengths)

            # Training loss
            train_loss = criterion(outputs, labels)

            # Retrieve gradients
            train_loss.backward()

            # Update parameters
            optimizer.step()

        end = time()

        # Save log to keep progress of trained LSTMs.
        with open('/cluster/project1/FFPredRNN/data/optimised_logs/logs_' + str(go_term) + '.txt', 'a') as f:
            f.write("(%s, %d, %d) -> Epoch: %d \t Time: %f \t Time Left: %f \n" %
                    (go_term, neurons, time_step, epoch, end - start, (epochs - epoch) * (end - start)))

    directory = '/cluster/project1/FFPredRNN/data/biological_process/psipred/go_terms/' + go_term
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the optimised models for each GO term
    torch.save(net.state_dict(), directory + '/model_optimised.pt')
