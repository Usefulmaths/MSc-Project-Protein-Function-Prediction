import re
import dill as pickle

'''
The script used to generate an array of LSTM models for the
biological processes domain.
'''

with open('/home/leloie/Codes/cached_material/reduced_go_terms.pkl', 'rb') as f:
    reduced_go_terms = pickle.load(f)

go_terms = reduced_go_terms['biological_process']   

with open('/home/leloie/Codes/data/biological_process/GO_term_index_mapping.pkl', 'rb') as f:
    GO_term_index_mapping = pickle.load(f)

for go_term in go_terms:
	with open('FFPred_model.py', 'r') as f:
		lines = f.readlines()

	with open('/cluster/project1/FFPredRNN/ArrayJobs/' + str(GO_term_index_mapping[go_term]) + '.py', 'w') as f:
		for line in lines:
			line = re.sub(r'go_term_placeholder', str(go_term), line)
			f.write(line)