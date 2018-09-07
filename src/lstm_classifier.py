import sys
import numpy as np
import hickle
import dill as pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
from model import LSTM, mask_padding, f1_score_go

# Set seed
np.random.seed(0)

# Load in protein train/test splits
with open('/home/leloie/Codes/cached_material/protein_train_test_splits.pkl', 'rb') as f:
    protein_train_test_splits = pickle.load(f)

# Load in GO term samples (top, middle, bottom).
with open('/home/leloie/Codes/cached_material/GO_term_samples.pkl', 'rb') as f:
    go_term_samples = pickle.load(f)

data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'
cluster_data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'

with open('/home/leloie/Codes/data/biological_process/GO_term_index_mapping.pkl', 'rb') as f:
    GO_term_index_mapping = pickle.load(f)

# GO term placeholder for script generation
go_term = 'qqq'
go_index = GO_term_index_mapping[go_term]

# Only run network on GO terms that finished training in previous experiment.
finished_terms = ['GO0006414', 'GO0016197', 'GO0043623', 'GO0007601', 'GO0048666', 'GO0006412', 'GO0006338', 'GO0042254', 'GO0048699', 'GO0030258', 'GO0035725', 'GO0016311', 'GO0046942', 'GO0051186', 'GO0032388', 'GO0006066', 'GO0010466', 'GO0006281', 'GO0009116', 'GO0010955', 'GO0006470', 'GO0050953', 'GO0043603', 'GO0098609', 'GO0060627', 'GO0030522', 'GO0034976', 'GO0051346', 'GO0044262', 'GO0006366', 'GO0035023', 'GO0006260', 'GO0006631', 'GO0045861', 'GO0018108', 'GO0032870', 'GO0006163', 'GO0044257', 'GO0048002', 'GO0050907', 'GO0009165', 'GO0009101', 'GO0009615', 'GO0007411', 'GO1901605', 'GO0022604', 'GO0009611', 'GO0051056', 'GO0070613', 'GO0006869', 'GO0090150', 'GO0045333', 'GO0045893', 'GO0051321', 'GO0006457', 'GO0046488', 'GO0006935', 'GO0009266', 'GO0051276', 'GO0050804', 'GO0016570', 'GO0007015', 'GO0050911', 'GO0015698', 'GO0072657', 'GO0016477', 'GO0046395', 'GO0031589', 'GO0032535', 'GO0060429', 'GO0015711', 'GO0051146', 'GO0016568', 'GO0019216', 'GO0007059', 'GO0006367', 'GO0044089', 'GO0051495', 'GO0006913', 'GO0071345', 'GO0008285', 'GO0060271', 'GO0007596',
                  'GO0006605', 'GO0008284', 'GO0043123', 'GO0009582', 'GO0071103', 'GO0050678', 'GO0006898', 'GO0007417', 'GO0006417', 'GO0006643', 'GO0022409', 'GO0045165', 'GO0007265', 'GO0046777', 'GO0000226', 'GO0007067', 'GO0016050', 'GO0038094', 'GO0033365', 'GO0045137', 'GO0009416', 'GO0070925', 'GO0042063', 'GO0040029', 'GO0048839', 'GO0001666', 'GO0007565', 'GO0000165', 'GO0007005', 'GO0032496', 'GO0043270', 'GO0006521', 'GO0006954', 'GO0032868', 'GO0031669', 'GO0008543', 'GO0071774', 'GO0042493', 'GO0044344', 'GO0006887', 'GO0051224', 'GO0051048', 'GO0030335', 'GO0060828', 'GO0033157', 'GO0071560', 'GO0030099', 'GO0045637', 'GO0072358', 'GO0030336', 'GO0002064', 'GO0050796', 'GO0043583', 'GO0071375', 'GO0042110', 'GO0051480', 'GO0007399', 'GO0034765', 'GO0043010', 'GO0051047', 'GO0071593', 'GO1901653', 'GO0007611', 'GO0030100', 'GO2001235', 'GO0045862', 'GO0009405', 'GO0051098', 'GO0007162', 'GO0043254', 'GO0010256', 'GO0010565', 'GO0010038', 'GO0000956', 'GO0031328', 'GO0048511', 'GO0007186', 'GO0046474', 'GO0008015', 'GO0050714', 'GO0030036', 'GO0010467', 'GO0000278', 'GO0010557', 'GO0006915']

if go_term not in finished_terms:
    sys.exit("GO term not in finished sequence.")

# Define up varying hyperparameters
layers = [[512], [256]]
steps = [50, 20, 10]

# Create protein train test splits
proteins_train = protein_train_test_splits[go_term]['train']
proteins_test = protein_train_test_splits[go_term]['test']

proteins_train = np.concatenate(
    (proteins_train['positive'], proteins_train['negative']), axis=0)
proteins_test = np.concatenate(
    (proteins_test['positive'], proteins_test['negative']), axis=0)

# For each layer of the hyperparameters
for layer in layers:

    # For each time-step in the hyperparameters
    for step in steps:
        protein_features_train = []
        protein_labels_train = []

        protein_features_test = []
        protein_labels_test = []

        # For each protein in the training set
        for index, protein in enumerate(proteins_train):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index, end="\r")

            try:
                # Load in the protein psipred sequential features
                with open(data_directory + protein + '/features.pkl', 'rb') as f:
                    features = pickle.load(f)
                    protein_features_train.append(features)

                # Load in the protein labels
                with open(data_directory + protein + '/labels.pkl', 'rb') as f:
                    labels = pickle.load(f)
                    protein_labels_train.append(labels)

            except(FileNotFoundError):
                print('%s not found.' % (protein))

        # For each protein in the testing set
        for index, protein in enumerate(proteins_test):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index, end="\r")

            try:
                # Load in the protein psipred sequential features
                with open(data_directory + protein + '/features.pkl', 'rb') as f:
                    features = pickle.load(f)
                    protein_features_test.append(features)

                # Load in the protein labels
                with open(data_directory + protein + '/labels.pkl', 'rb') as f:
                    labels = pickle.load(f)
                    protein_labels_test.append(labels)

            except(FileNotFoundError):
                print('%s not found.' % (protein))

        # Set up the training/ testing features and labels
        X_train = np.array(protein_features_train)
        y_train = np.array(protein_labels_train)[:, go_index].reshape(-1, 1)

        X_test = np.array(protein_features_test)
        y_test = np.array(protein_labels_test)[:, go_index].reshape(-1, 1)

        # Set up the hyperparameters of the LSTM
        batch_size = 256
        epochs = 200
        time_step = step
        feature_size = 3
        input_size = time_step * feature_size
        num_layers = 3

        max_length = max(map(lambda x: len(x), protein_features_train))

        X_train_tensor = torch.Tensor(X_train)
        X_test_tensor = torch.Tensor(X_test)
        y_train_tensor = torch.Tensor(y_train)
        y_test_tensor = torch.Tensor(y_test)

        # Create DataLoaders for training and testing sets.
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_data_loader = DataLoader(train_data, batch_size)

        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_data_loader = DataLoader(test_data, batch_size)

        # Create a LSTM with a single output neuron
        net = LSTM(input_size, layer, 1)
        net = net

        # Define the optimiser over the network parameters.
        optimizer = torch.optim.Adam(net.parameters())

        # Define the loss function.
        criterion = nn.MultiLabelSoftMarginLoss()

        # For each epoch
        for epoch in range(epochs):
            with open('/cluster/project1/FFPredRNN/Outputs/log_%s' % go_term, 'w') as g:
                g.write("%s: Epoch %d\n" % (go_term, epoch))

            test_losses = []

            # For each batch in the training data
            for batch_idx, (data, target) in enumerate(train_data_loader):
                    # Convert features to a PyTorch variable
                features = Variable(data)

                # Reshape to use as input to LSTM
                features = features.view(
                    (data.shape[0], max_length / time_step, input_size))

                # Convert labels to a PyTorch variable
                labels = Variable(target)

                # Mask the padding
                lengths, indices = mask_padding(features, max_length)

                features = features[indices]
                labels = labels[indices]

                # Reset gradients of optimiser to zsero
                optimizer.zero_grad()

                # Forward pass
                outputs = net(features, lengths)

                # Calculate training loss
                train_loss = criterion(outputs, labels)

                # Calculate gradients of parameters of network
                train_loss.backward()

                # Update network parameters
                optimizer.step()

            print("Epoch complete: %d" % epoch)

            test_predictions = []
            test_labels = []
            test_outputs = []

            # For each batch in the test data
            for batch_idx, (data, target) in enumerate(test_data_loader):
                # Convert features to a PyTorch variable
                features = Variable(data)

                # Reshape to use as input to LSTM
                features = features.view(
                    (data.shape[0], max_length / time_step, input_size))

                # Convert labels to a PyTorch variable
                labels = Variable(target)

                # Mask the padding
                lengths, indices = mask_padding(features, max_length)

                features = features[indices]
                labels = labels[indices]

                # Forward pass
                outputs = net(features, lengths)

                # Convert outputs to probabilities
                sigmoid_output = F.sigmoid(outputs)

                # Calculate testing loss
                test_loss = criterion(outputs, labels)

                labels = labels.data.cpu().numpy()

                # Make predictions
                predictions = sigmoid_output > 0.5

                test_losses.append(float(test_loss))
                test_predictions.append(predictions)
                test_labels.append(labels)
                test_outputs.append(net.store_hidden.data.cpu().numpy())

            test_predictions = np.concatenate(tuple(test_predictions), axis=0)
            test_labels = np.concatenate(tuple(test_labels), axis=0)

            # Calculate F1 score
            test_f1_array, test_f1 = f1_score_go(test_labels, test_predictions)

            print("Epoch: %d \t Test loss: %f \t Test F1 score: %f \t Test Precision: %f \t Test Recall: %f" % (
                epoch, np.mean(test_losses), test_f1, test_precision, test_recall))

        # Save the performance of the network with the hyper-parameters
        with open('/cluster/project1/FFPredRNN/data/lstm_standard_hyperparameter_results_2.txt', 'a') as f:
            f.write('%s, %d, %d, %f, %f, %f\n' % (go_term, layer[
                    0], step, test_f1, test_precision, test_recall))

        #directory = '/cluster/project1/FFPredRNN/data/biological_process/psipred/raw_lstm/go_terms/' + go_term
