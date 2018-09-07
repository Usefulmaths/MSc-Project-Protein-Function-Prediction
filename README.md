# MSc Machine Learning Project: Multi-Task LSTM Representation Extraction for Protein Function Prediction
This repository stores all the necessary code to carry out protein function prediction by extracting functional feature representations from a LSTM and using them to train single binary-output SVMs for each function considered.

#### Pipeline

1. First, the GO sub-ontologies are reduced, removing degenerate GO terms from the DAG, using `create_reduced_go_set.py`

2. Next, protein objects corresponding to the proteins found within the reduced GO sub-ontologies are created using `created_proteins.py`

3. The script `fill_proteins.py` is then used to fill these protein objects with raw sequential features, utilising feature extraction methods found in `feature_extraction_functions.py`

4. These protein objects are then converted into feature matrices and label matrices using `create_protein_matrices.py`

5. Special training and testing data splits are then set up using `create_train_test_sets.py` 

6. Hyper-parameter tuning for the LSTMs to learn the functional representations is then carried out using `model.py`

7. Using the best-hyperparameters found for the LSTMs for each GO term, optimised LSTMs are then trained using `optimised_model.py`

8. These optimised models are then used to transform the protein feature matrices into functional feature representations by passing through the LSTM models, using the `extract_protein_representations.py`.

9. SVMs can then be trained using the raw features (extract raw global features using `extract_protein_psipred`) of the proteins and the LSTM representations of the proteins using `SVM_features.py` and `SVM_representations.py` respectively.

10. LSTM classifiers for the raw sequential features can be trained using the `lstm_classifier.py` script.
