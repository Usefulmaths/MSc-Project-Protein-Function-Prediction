import numpy as np
import os
import re
import collections
import pickle


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


def create_GO_dictionary(file_name):
    '''
    Create a dictionary of all GO terms from full_long_summary, where
    the individual keys represent the GO terms and the values represent
    which domain the GO terms belong to.
    '''
    GO_dictionary = dict()
    with open(file_name) as f:
        data_array = f.readlines()[1:]
        for line in data_array:
            line_array = line.split('\t')

            go_term = line_array[0]
            go_domain = line_array[13]

            GO_dictionary[go_term] = go_domain

    return GO_dictionary


def assign_protein_go_terms(GO_terms_list, go_domain_directory):
    '''
    Returns a dictionary containing proteins and their assigned GO terms.
    The keys represent the protein IDs, whilst the values represent the GO
    terms that are assigned the each protein.
    '''
    protein_term_dictionary = collections.defaultdict(list)
    for GO_term in GO_terms_list:
        if GO_term == '.DS_Store':
            continue
        else:
            GO_term_name = GO_term[:-5]

            with open(go_domain_directory + GO_term, 'r') as f:
                lines = f.readlines()

                negative = True
                for line in lines:
                    if line != '# test:\n' or line != '\n' or line != '# train:\n':
                        if not negative:
                            protein = line[:-1]
                            protein_term_dictionary[
                                protein].append(GO_term_name)

                        if line == '# positive set :\n':
                            negative = False

    return protein_term_dictionary


def one_hot_encoding(numbers, max_number):
    '''
    Returns a one-hot encoded vector.
    '''
    encoded = np.zeros((max_number, 1))
    if len(numbers) > 0:
        encoded[numbers, 0] = 1

    return encoded


def go_encoding(go_terms, unique_go_terms):
    '''
    Creates a one-hot encoding for a set of GO terms
    from a protein by using the number of unique
    go terms in a specific domain.
    '''
    unique_go_dictionary = dict()
    for index, term in enumerate(unique_go_terms):
        unique_go_dictionary[term] = index

    number_terms = len(unique_go_terms)

    filtered_go_terms = filter(
        lambda go_term: go_term in unique_go_dictionary.keys(), go_terms)
    categorical = np.array(
        list(map(lambda x: unique_go_dictionary[x], filtered_go_terms)))

    encoding = one_hot_encoding(categorical, number_terms)

    return encoding


def get_unique_go_domain_terms(protein_term_dictionary, GO_dictionary):
    '''
    Returns the unique GO terms assigned to each GO domain from the protein set.
    '''
    unique_go_terms = np.unique(
        [item for sublist in protein_term_dictionary.values() for item in sublist])

    unique_mf_go_terms = [term for term in unique_go_terms if GO_dictionary[
        term] == 'molecular_function']
    unique_bp_go_terms = [term for term in unique_go_terms if GO_dictionary[
        term] == 'biological_process']
    unique_cc_go_terms = [term for term in unique_go_terms if GO_dictionary[
        term] == 'cellular_component']

    return unique_mf_go_terms, unique_bp_go_terms, unique_cc_go_terms


if __name__ == '__main__':
    reduced = False

    feature_data_directory = './data/RawFeaturesFFPredRNN/'
    protein_names = os.listdir(feature_data_directory)

    go_term_file_name = './data/full_long_summary'
    GO_dictionary = create_GO_dictionary(go_term_file_name)

    go_domain_directory = './data/full_terms_test/'
    GO_terms_list = os.listdir(go_domain_directory)

    protein_term_dictionary = assign_protein_go_terms(
        GO_terms_list, go_domain_directory)

    # Retrieves a dictionary containing the reduced
    # GO term set for each domain.
    with open('reduced_go_terms.pkl', 'rb') as f:
        reduced_go_terms = pickle.load(f)

    if reduced == False:
        unique_mf_go_terms, unique_bp_go_terms, unique_cc_go_terms = get_unique_go_domain_terms(
            protein_term_dictionary, GO_dictionary)

    else:
        unique_mf_go_terms = reduced_go_terms['molecular_function']
        unique_bp_go_terms = reduced_go_terms['biological_process']
        unique_cc_go_terms = reduced_go_terms['cellular_component']

    unique_go_terms = {'molecular_function': unique_mf_go_terms,
                       'biological_process': unique_bp_go_terms,
                       'cellular_component': unique_cc_go_terms}

    proteins = {'molecular_function': [],
                'biological_process': [],
                'cellular_component': []}

    protein_name_key_map = dict()

    # For all proteins, create a protein object.
    for protein_name in protein_names:
        if protein_name != 'README.txt' and protein_name != '.DS_Store' and protein_name not in ['A0A087WT19', 'A0A087WUN5', 'A0A087WUV6', 'A0A087WV85', 'A0A087WVP8', 'A0A087WVR2', 'A0A087WVW0', 'A0A087WWQ3', 'A0A087WWX5', 'A0A087WZ37', 'A0A087WZD6', 'A0A087X1C6', 'A0A096LNI8', 'A0A096LNY1', 'A0A096LP23', 'A0A096LPE0', 'A0A0A0MQV9', 'A0A0A0MR55', 'A0A0A0MT49', 'A0A0A0MTE0', 'A0A0A6YYL5', 'A1L4H1', 'A2A3L9', 'A2IDB1', 'A2RU14', 'A2RUT3', 'A2VDJ0', 'A5LHX3', 'A6NGB7', 'A6NIV2', 'A6NJU9', 'A6NKF7', 'A8MT69', 'A8MTW9', 'A8MUN3', 'A8MXF1', 'A9Z1X7', 'B0QYP3', 'B2RUZ4', 'B4DEB1', 'B7WNN4', 'B7Z8B3', 'B8ZZ07', 'B8ZZE7', 'B9A032', 'C9J029', 'C9J2R4', 'C9J2Y1', 'C9J6Q0', 'C9JDV1', 'C9JF34', 'C9JFV4', 'C9JG80', 'C9JIH8', 'C9JPS1', 'C9JQJ2', 'C9JQX2', 'C9JU68', 'C9JUS6', 'C9JVW0', 'C9JXX5', 'C9K0J5', 'D6R9D3', 'D6RAA4', 'D6RAW1', 'D6RC06', 'D6RD60', 'D6RE89', 'D6REX3', 'D6RHY7', 'E2RYF6', 'E5RFF7', 'E5RFI2', 'E5RGJ1', 'E5RGM4', 'E5RHM3', 'E5RHQ5', 'E5RHY0', 'E5RIP5', 'E5RIT9', 'E5RJ25', 'E5RJ92', 'E5RK20', 'E7EMY7', 'E7EN65', 'E7ENW7', 'E7EP82', 'E7EQI0', 'E7ERS3', 'E7ESD3', 'E7ETZ7', 'E7EVG6', 'E7EWS8', 'E9PF17', 'E9PHH0', 'E9PK90', 'E9PKX9', 'E9PLV7', 'E9PM49', 'E9PM76', 'E9PMH7', 'E9PNK5', 'E9PQ11', 'E9PQU0', 'E9PR12', 'E9PRD7', 'E9PSE2', 'F2Z2E0', 'F2Z2I9', 'F2Z2M2', 'F2Z307', 'F5GY34', 'F5H124', 'F5H2U0', 'F5H479', 'F5H4P1', 'F5H6S0', 'F5H846', 'F5H855', 'F6VUY7', 'F7VJQ1', 'F8VNW0', 'F8VQT7', 'F8VRU3', 'F8VSI3', 'F8VU51', 'F8VV11', 'F8VVX9', 'F8VW93', 'F8VWA9', 'F8VXV4', 'F8W883', 'F8WAX1', 'F8WB74', 'F8WB78', 'F8WB98', 'F8WBD6', 'F8WBI7', 'F8WC34', 'F8WC71', 'F8WCF0', 'F8WCF1', 'F8WCH6', 'F8WCN7', 'F8WDB3', 'F8WDE4', 'F8WDE9', 'F8WDP9', 'F8WE30', 'F8WEB6', 'F8WEI3', 'F8WEP1', 'F8WEU6', 'F8WFC7', 'G3V3L5', 'G3V4U6', 'G5E950', 'G5E966', 'G5E9R3', 'H0YAJ1', 'H0YB40', 'H0YBZ5', 'H0YCL9', 'H0YDQ6', 'H0YDT5', 'H0YG40', 'H0YH96', 'H0YIC2', 'H0YJV6', 'H0YL42', 'H0YN25', 'H3BMA4', 'H3BMG6', 'H3BMI8', 'H3BMP9', 'H3BMR7', 'H3BN16', 'H3BN66', 'H3BQ09', 'H3BQG7', 'H3BRP8', 'H3BUF8', 'H3BVI5', 'H7BXJ5', 'H7BZC8', 'H7C031', 'H7C1I2', 'H7C317', 'H7C416', 'H7C4Y2', 'H7C5J4', 'H7C5Z0', 'I1E4Y6', 'I3L0L4', 'I3L119', 'I3L1N9', 'I3L1S0', 'I3L1Z4', 'I3L266', 'I3L2L9', 'I3L393', 'I3L3P0', 'I3L424', 'I3L4E0', 'I3NI43', 'J3KQV8', 'J3KS72', 'J3KSV0', 'J3KTH8', 'J3QRN4', 'J3QS12', 'K7EIN3', 'K7EJL2', 'K7EMB9', 'K7EN30', 'K7ENT3', 'K7EP01', 'K7EP04', 'K7EPJ2', 'K7EQ67', 'K7EQ87', 'K7EQK1', 'K7EQN2', 'M0QX08', 'M0QXB8', 'M0QXC6', 'M0QXQ7', 'M0QXT6', 'M0QXY2', 'M0QY80', 'M0QYC6', 'M0QYK8', 'M0QYX1', 'M0QZM5', 'M0QZM7', 'M0QZU1', 'M0QZY0', 'M0R0D3', 'M0R0Y0', 'M0R0Z0', 'M0R1P1', 'M0R1R6', 'M0R1R8', 'M0R1S3', 'M0R2F0', 'M0R2K0', 'M0R365', 'M0R399', 'O00512', 'O00585', 'O00590', 'O14558', 'O15047', 'O15054', 'O15069', 'O15172', 'O15263', 'O15304', 'O43432', 'O43516', 'O60307', 'O60542', 'O60885', 'O75690', 'O94777', 'O95104', 'O95178', 'O95487', 'O97980', 'P01606', 'P01615', 'P01767', 'P02458', 'P02808', 'P02814', 'P04553', 'P04554', 'P07197', 'P09919', 'P0C5Y4', 'P0C7Q2', 'P0CJ69', 'P0CJ70', 'P0CJ71', 'P0CJ72', 'P0CJ73', 'P0CJ74', 'P0CJ76', 'P0CJ77', 'P0DH78', 'P0DI83', 'P0DJ93', 'P0DL12', 'P0DMP1', 'P10176', 'P12036', 'P12109', 'P13945', 'P14867', 'P15502', 'P15941', 'P17600', 'P18135', 'P23246', 'P23490', 'P26371', 'P33552', 'P35228', 'P35326', 'P35579', 'P35749', 'P49750', 'P49765', 'P51397', 'P54259', 'P56279', 'P59990', 'P59991', 'P60328', 'P60329', 'P60331', 'P60369', 'P60371', 'P60372', 'P60409', 'P60411', 'P60412', 'P60413', 'P60602', 'P60606', 'P61604', 'P62945', 'P81172', 'P84996', 'Q02880', 'Q07283', 'Q07627', 'Q08999', 'Q08AP5', 'Q12816', 'Q12988', 'Q13072', 'Q13191', 'Q13233', 'Q14004', 'Q15643', 'Q16157', 'Q16446', 'Q16513', 'Q16617', 'Q16630', 'Q16674', 'Q17R89', 'Q19AV6', 'Q27J81', 'Q30KQ5', 'Q3LHN2', 'Q3LI54', 'Q3LI62', 'Q3LI70', 'Q3SYF9', 'Q3Y452', 'Q4VCS5', 'Q4ZHG4', 'Q587I9', 'Q5BLP8', 'Q5H9F3', 'Q5JU85', 'Q5JX65', 'Q5JZ02', 'Q5ST79', 'Q5TCM9', 'Q5VTY9', 'Q5VV67', 'Q5VX84', 'Q5VX85', 'Q629K1', 'Q68CP9', 'Q69YU5', 'Q6L8H2', 'Q6L8H4', 'Q6NT89', 'Q6NVH7', 'Q6NZ36', 'Q6P0A1', 'Q6P1K1', 'Q6PUV4', 'Q6QNY0', 'Q6UXQ8', 'Q6ZMY3', 'Q6ZN18', 'Q6ZVL6', 'Q701N2', 'Q70E73', 'Q71RC9', 'Q7Z4W3', 'Q7Z7F7', 'Q86SJ6', 'Q86UU0', 'Q86UU5', 'Q86UU9', 'Q86X51', 'Q8IWJ2', 'Q8IXM6', 'Q8IZD2', 'Q8N0U2', 'Q8N0Y2', 'Q8N661', 'Q8N6C5', 'Q8N726', 'Q8N8P7', 'Q8NES8', 'Q8NHG7', 'Q8TAE6', 'Q8TAQ2', 'Q8TES7', 'Q8WVI0', 'Q8WXE0', 'Q8WXF3', 'Q8WXG6', 'Q8WYQ3', 'Q8WZ04', 'Q92617', 'Q92805', 'Q92954', 'Q969E1', 'Q96CA5', 'Q96DE5', 'Q96EV2', 'Q96HJ9', 'Q96I36', 'Q96I85', 'Q96IU2', 'Q96JP2', 'Q96KR6', 'Q96N68', 'Q96PG8', 'Q96PI1', 'Q96RJ3', 'Q96RK0', 'Q99700', 'Q99748', 'Q99758', 'Q9BRJ9', 'Q9BT30', 'Q9BTK6', 'Q9BVW6', 'Q9BXK1', 'Q9BYB0', 'Q9BYE4', 'Q9BYR0', 'Q9BYR4', 'Q9BYS1', 'Q9BZL3', 'Q9C0J8', 'Q9GZP8', 'Q9H0B3', 'Q9H195', 'Q9H321', 'Q9H3S7', 'Q9HB42', 'Q9HCU8', 'Q9NP73', 'Q9NP84', 'Q9NQC3', 'Q9NRI6', 'Q9NRJ1', 'Q9NRQ5', 'Q9NTK1', 'Q9NUB4', 'Q9NYJ1', 'Q9NZ56', 'Q9P0N5', 'Q9P1C3', 'Q9P2R6', 'Q9P2X0', 'Q9UHE5', 'Q9UHL7', 'Q9UJH8', 'Q9UL51', 'Q9ULL8', 'Q9UP65', 'Q9UPS6', 'Q9UPV0', 'Q9UPX0', 'Q9Y2A0', 'Q9Y2K3', 'Q9Y3P4', 'Q9Y5L5', 'Q9Y5M6', 'Q9Y664', 'Q9Y693', 'Q9Y6Q9', 'R4GMY8', 'S4R344', 'S4R3P1', 'S4R3Y5', 'U3KQA3', 'U3KQP2', 'V9GYI1', 'V9GZ69', 'X6R2L4']:

            protein_files = os.listdir(feature_data_directory + protein_name)
            key = re.sub(r'\..*', '', protein_files[0])

            # Retrieve GO terms assigned to this protein
            protein_go_terms = protein_term_dictionary[protein_name]

            # Split the GO terms in terms of domain.
            mf_go_terms = [term for term in protein_go_terms if GO_dictionary[
                term] == 'molecular_function']
            bp_go_terms = [term for term in protein_go_terms if GO_dictionary[
                term] == 'biological_process']
            cc_go_terms = [term for term in protein_go_terms if GO_dictionary[
                term] == 'cellular_component']

            # Remove GO terms from protein that are not in the reduced set.
            if reduced == True:
                mf_go_terms = set(unique_mf_go_terms).intersection(mf_go_terms)
                bp_go_terms = set(unique_bp_go_terms).intersection(bp_go_terms)
                cc_go_terms = set(unique_cc_go_terms).intersection(cc_go_terms)

            # Create a dictionary for a protein that maps domain to GO terms.
            protein_go_term_dict = {'molecular_function': mf_go_terms,
                                    'biological_process': bp_go_terms,
                                    'cellular_component': cc_go_terms}

            # Create a protein object with name, key, and protein_go_term
            # dictionary.
            protein = Protein(protein_name, key, protein_go_term_dict)

            protein_name_key_map[protein_name] = key

            intersections = []
            for domain in ['molecular_function', 'biological_process', 'cellular_component']:
                intersection = set(protein.go_terms[domain]).intersection(
                    unique_go_terms[domain])

                unique_terms = unique_go_terms[domain]

                unique_go_dictionary = dict()
                for index, term in enumerate(unique_terms):
                    unique_go_dictionary[term] = index

                directory = './data/' + domain
                if not os.path.exists(directory):
                    os.makedirs(directory)

                with open(directory + '/GO_term_index_mapping.pkl', 'wb') as f:
                    pickle.dump(unique_go_dictionary, f)

                # If the protein has go terms in a specified domain that are in the
                # reduced unique go term set, then compute the encoding, store it
                # in the protein object and append to the set of proteins.
                if(len(intersection) > 0):
                    protein.go_term_encoding(domain, unique_terms)
                    proteins[domain].append(protein)

    # Save the proteins.
    with open('cached_material/all_proteins.pkl', 'wb') as f:
        pickle.dump(proteins, f)

    with open('cached_material/protein_name_key_map.pkl', 'wb') as f:
        pickle.dump(protein_name_key_map, f)
