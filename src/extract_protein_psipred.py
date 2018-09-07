import dill as pickle
import os
import numpy as np
import tarfile
import shutil


def make_tarfile(output_filename, source_dir):
    '''
    Creates a tar file of a directory.

    Arguments:
        output_filename: name of the tar file.
        source_dir: the directory to be converted.
    '''
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == '__main__':

    # Load in a dictionary of the protein train and test splits.
    with open('./cached_material/protein_train_test_splits.pkl', 'rb') as f:
        protein_train_test_splits = pickle.load(f)

    # Load in training and test proteins for each GO term
    with open('./cached_material/go_term_protein_training.pkl', 'rb') as f:
        go_term_protein_training = pickle.load(f)

    # Load in GO term samples (top, middle, bottom).
    with open('./cached_material/GO_term_samples.pkl', 'rb') as f:
        go_term_samples = pickle.load(f)

    # Load in a dictionary of the protein name to key map.
    with open('./cached_material/protein_name_key_map.pkl', 'rb') as f:
        protein_name_key_map = pickle.load(f)

    # Directory where psipred tensors are located.
    data_directory = './data/biological_process/tensors_psipred/'

    # GO terms to be considered.
    go_terms = ['GO0051047', 'GO0060828', 'GO0034765', 'GO0045893', 'GO0048839', 'GO0051048', 'GO0035725', 'GO0046777', 'GO0006163', 'GO0070925', 'GO0071417', 'GO0006310', 'GO0007067', 'GO0016485', 'GO0030335', 'GO0000209', 'GO0016568', 'GO0009101', 'GO0030258', 'GO0010628', 'GO0046474', 'GO0006898', 'GO0072657', 'GO0032956', 'GO0006402', 'GO0006643', 'GO0044257', 'GO0030522', 'GO0006935', 'GO0046488', 'GO0006457', 'GO0016570', 'GO0045862', 'GO0002431', 'GO0045861', 'GO0007015', 'GO0071593', 'GO0050907', 'GO0016477', 'GO0007265', 'GO0009116', 'GO0051146', 'GO0030036', 'GO0007018', 'GO0048872', 'GO0048871', 'GO0050730', 'GO0006412', 'GO0006413', 'GO1901605', 'GO0006414', 'GO0048667', 'GO0048666', 'GO0031328', 'GO0051321', 'GO0000165', 'GO0006338', 'GO0051260', 'GO0071103', 'GO0044262', 'GO0007411', 'GO0045089', 'GO0042110', 'GO0046395', 'GO0050911', 'GO0007417', 'GO0071774', 'GO0032535', 'GO0002683', 'GO0009416', 'GO0031396', 'GO0016064', 'GO0071345', 'GO0071560', 'GO0010565', 'GO0045165', 'GO0007399', 'GO0010876', 'GO0043623', 'GO0007093', 'GO0006281', 'GO0046578', 'GO0048193', 'GO0051276', 'GO0033365', 'GO0048699', 'GO0016197', 'GO0042254', 'GO0009405', 'GO0016050', 'GO0043010', 'GO0030099', 'GO0045137', 'GO0042063', 'GO0022409', 'GO0006631', 'GO0051346', 'GO0051186', 'GO0090305', 'GO0045732', 'GO0048511', 'GO0098609', 'GO0015711', 'GO0007186', 'GO0010466', 'GO0010467', 'GO0032388', 'GO0006869', 'GO0031589', 'GO0090150', 'GO0007601', 'GO0008217', 'GO0045637', 'GO0006605', 'GO1901653', 'GO1901652', 'GO0007409', 'GO0072594', 'GO0010951', 'GO0038094', 'GO0033157', 'GO0006470', 'GO0006521', 'GO0043281', 'GO0050804', 'GO0030168', 'GO2001235', 'GO2001234', 'GO0043603', 'GO0008015', 'GO0006913', 'GO0006915', 'GO0051495', 'GO0051480', 'GO0002064', 'GO0007420', 'GO0045944', 'GO0000226', 'GO0007059', 'GO0010256', 'GO0060429', 'GO0006260', 'GO0060627', 'GO0032496', 'GO0009266', 'GO0060173', 'GO0043583', 'GO0060271', 'GO0044344', 'GO0044089', 'GO0032259', 'GO0010557', 'GO0051056', 'GO0043123', 'GO0042060', 'GO0006417', 'GO0007005', 'GO0006909', 'GO0006366', 'GO0050678', 'GO0034976', 'GO0050714', 'GO0050796', 'GO0051224', 'GO0007596', 'GO0010038', 'GO0040029', 'GO0001666', 'GO0009615', 'GO0007565', 'GO0007162', 'GO0009611', 'GO0019216', 'GO0009612', 'GO0006066', 'GO0031669', 'GO0009165', 'GO0022604', 'GO0015698', 'GO0050953', 'GO0030100', 'GO0043270', 'GO0042493', 'GO0030336', 'GO0008543', 'GO0007611', 'GO0000956', 'GO0035023', 'GO0018108', 'GO0032870', 'GO0008285', 'GO0008284', 'GO0006954', 'GO0071407', 'GO0072358', 'GO0006367', 'GO0000278', 'GO0009582', 'GO0000375', 'GO0016311', 'GO0015992', 'GO0030178', 'GO0060322', 'GO0043254', 'GO0070489', 'GO0070613', 'GO0006887', 'GO0071375', 'GO0042278', 'GO0045333', 'GO0048646', 'GO0032868', 'GO0048002', 'GO0071805', 'GO0010955', 'GO0046942', 'GO0051098']

    # For each GO term
    for go_term in go_terms:

        # Retrieve proteins in the training and testing splits.
        proteins = protein_train_test_splits[go_term]

        proteins_train = proteins['train']
        proteins_test = proteins['test']

        # Concatenate the positive and negative proteins.
        proteins_train = np.concatenate((proteins_train['positive'], proteins_train['negative']), axis=0)
        proteins_test =  np.concatenate((proteins_test['positive'], proteins_test['negative']), axis=0)

        protein_svm_train = []
        protein_svm_test = []

        # For each protein in the training set
        for index, protein in enumerate(proteins_train):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index)

            # Load in the features of that protein.
            try:
                # Read in the psipred features from the FFPred results file.
                with open('./data/RawFeaturesFFPredRNN/' + protein + "/" + protein_name_key_map[protein] + '.results') as f:
                    lines = f.readlines()

                    features = [line.split('\t') for line in lines]

                    # Psipred features
                    del features[2][160:210]

                    idd = features[2][0]

                    svm_features = np.array([element for element in features[2] if element != idd][:-1], dtype=np.float32)
                    directory = './data/biological_process/psipred/go_terms/' + go_term + '/train/' + protein

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Save the SVM features for this protein
                    with open(directory + '/svm_features.pkl', 'wb') as f:
                        pickle.dump(svm_features, f)


            except(FileNotFoundError):
                    proteins_train = list(filter(lambda x: x != protein, proteins_train))
                    print("%s not found." % protein)

        # For each protein in the testing set.
        for index, protein in enumerate(proteins_test):
            if(index % 100 == 0):
                print("Loaded Proteins: %d" % index)

            # Load in the psipred features of that protein
            try:
                # Read in the psipred features from the FFPred results file.
                with open('./data/RawFeaturesFFPredRNN/' + protein + "/" + protein_name_key_map[protein] + '.results') as f:
                    lines = f.readlines()

                    features = [line.split('\t') for line in lines]

                    # Psipred features
                    del features[2][160:210]

                    idd = features[2][0]
                    svm_features = np.array([element for element in features[2] if element != idd][:-1], dtype=np.float32)

                    directory = './data/biological_process/psipred/go_terms/' + go_term + '/test/' + protein

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    with open(directory + '/svm_features.pkl', 'wb') as f:
                        pickle.dump(svm_features, f)

            except(FileNotFoundError):
                    proteins_train = list(filter(lambda x: x != protein, proteins_train))
                    print("%s not found." % protein)


        # Make a tar file of the SVM features.
        make_tarfile('./data/biological_process/psipred/go_terms/' + go_term + ".tar.gz", './data/biological_process/psipred/go_terms/' + go_term)

        # Remove the directory containing the SVM features to save memory.
        shutil.rmtree('./data/biological_process/psipred/go_terms/' + go_term)
