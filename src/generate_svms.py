import re
import dill as pickle

'''
The script used to generate an array of SVM models for the
biological processes domain.
'''

go_terms = ['GO0050911', 'GO0007608', 'GO0050907', 'GO0007186', 'GO0034976', 'GO0008544', 'GO0032970', 'GO0051090', 'GO0050714', 'GO0050796', 'GO0043523', 'GO0010721']

with open('/home/leloie/Codes/data/biological_process/GO_term_index_mapping.pkl', 'rb') as f:
    GO_term_index_mapping = pickle.load(f)

go_indices = [GO_term_index_mapping[go_term] for go_term in go_terms]

for i in go_indices:
	with open('svm_representation_classifier.py', 'r') as f:
		lines = f.readlines()

	with open('/cluster/project1/FFPredRNN/ArrayJobs/' + str(i) + '.py', 'w') as f:
		for line in lines:
			line = re.sub(r'qqq', str(i), line)
			f.write(line)