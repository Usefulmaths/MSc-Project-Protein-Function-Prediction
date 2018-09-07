import numpy as np
import dill as pickle
import shutil
import tarfile
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Set seed
np.random.seed(0)

# Load in the protein train-test splits
with open('/home/leloie/Codes/cached_material/protein_train_test_splits.pkl', 'rb') as f:
    protein_train_test_splits = pickle.load(f)

# Load in GO term samples (top, middle, bottom).
with open('/home/leloie/Codes/cached_material/GO_term_samples.pkl', 'rb') as f:
    go_term_samples = pickle.load(f)

data_directory = '/cluster/project1/FFPredRNN/data/biological_process/tensors_psipred/'
cluster_data_directory = '/cluster/project1/FFPredRNN/data/biological_process/psipred/go_term_features/'

with open('/home/leloie/Codes/data/biological_process/GO_term_index_mapping.pkl', 'rb') as f:
    GO_term_index_mapping = pickle.load(f)
    GO_term_index_mapping = {v: k for k, v in GO_term_index_mapping.items()}

# GO index placeholder for script generation
go_index = qqq
go_term = GO_term_index_mapping[go_index]

# Extract the protein features from tar files.
tar = tarfile.open(cluster_data_directory + "%s.tar.gz" % (go_term))
tar.extractall(cluster_data_directory)
tar.close()

proteins_train = protein_train_test_splits[go_term]['train']
proteins_test = protein_train_test_splits[go_term]['test']

proteins_train = np.concatenate(
    (proteins_train['positive'], proteins_train['negative']), axis=0)
proteins_test = np.concatenate(
    (proteins_test['positive'], proteins_test['negative']), axis=0)

proteins_repr_train = []
proteins_repr_test = []
proteins_labels_train = []
proteins_labels_test = []

# For each protein in the training set
for index, protein in enumerate(proteins_train):
    if(index % 100 == 0):
        print("Loaded Proteins: %d" % index)

    try:
        # Load in the protein labels
        with open(data_directory + protein + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            proteins_labels_train.append(labels)

        # Load in the protein raw psipred features
        with open(cluster_data_directory + go_term + '/train/' + protein + '/svm_features.pkl', 'rb') as f:
            svm_features = pickle.load(f)

        features = svm_features
        proteins_repr_train.append(features)

    except(FileNotFoundError):
        proteins_train = list(filter(lambda x: x != protein, proteins_train))
        print("%s not found." % protein)

# For each protein in the testing set
for index, protein in enumerate(proteins_test):
    if(index % 100 == 0):
        print("Loaded Proteins: %d" % index)

    try:
        # Load in the protein labels
        with open(data_directory + protein + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            proteins_labels_test.append(labels)
        # Load in the protein raw psipred features
        with open(cluster_data_directory + go_term + '/test/' + protein + '/svm_features.pkl', 'rb') as f:
            svm_features = pickle.load(f)

        features = svm_features
        proteins_repr_test.append(features)

    except(FileNotFoundError):
        proteins_test = list(filter(lambda x: x != protein, proteins_test))
        print("%s not found." % protein)

# Remove directory of loaded features to save memory
shutil.rmtree(
    '/cluster/project1/FFPredRNN/data/biological_process/psipred/go_term_features/' + go_term)

# Set up the training and testing features and labels
X_train = np.array(proteins_repr_train)
X_test = np.array(proteins_repr_test)

y_train = np.array(proteins_labels_train)
y_test = np.array(proteins_labels_test)

# Convert to a single binary-output problem
y_train = y_train[:, go_index]
y_test = y_test[:, go_index]

# Hyperparameter search grid.
param_grid = [
    {'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4],
        'kernel': ['linear'], 'class_weight': [None, 'balanced']},
    {'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4], 'gamma': [1, 0.5, 3, 0.2, 10,
                                                                     0.1, 0.03, 0.01, 0.001, 1e-4], 'kernel': ['rbf'], 'class_weight': [None, 'balanced']}
]

# Instantiate SVC with RBF kernel.
svc = SVC(kernel='rbf')

# Instantiate cross validation object.
clf = GridSearchCV(estimator=svc, param_grid=param_grid,
                   scoring=make_scorer(f1_score))

# Train with training proteins
clf.fit(X_train, y_train)

# Choose the best hyperparameters
para = clf.best_params_
c_value = float(para['C'])
kernel_value = para['kernel']
cw = para['class_weight']

# Train optimised SVCs with the best hyperparameters
if kernel_value == 'rbf':
    gamma_value = float(para['gamma'])
    svc_trained = SVC(kernel=kernel_value, C=c_value,
                      gamma=gamma_value, class_weight=cw, probability=True)

else:
    svc_trained = SVC(kernel=kernel_value, C=c_value,
                      class_weight=cw, probability=True)

svc_trained.fit(X_train, y_train)

# Make predictions and calculate F1 score.
predictions = svc_trained.predict(X_test)
f1 = f1_score(predictions, y_test)

# Save the probabilities
with open('/cluster/project1/FFPredRNN/data/svm_feature_mixed_proba_3.txt', 'a') as f:
    probs = svc_trained.predict_proba(X_test)

    for i, prob in enumerate(probs):
        protein_name = proteins_test[i]
        probability_true = prob[0]
        f.write(str(go_term) + '\t' + protein_name +
                '\t' + str(probability_true) + '\n')

    f.close()

# Save the performance and hyperparameters for each GO term.
with open('/cluster/project1/FFPredRNN/data/svm_feature_mixed_hyperparameter_results_3.txt', 'a') as f:
    f.write(str(go_term) + ', ' + str(para) + ', ' + str(f1) + '\n')
    f.close()
