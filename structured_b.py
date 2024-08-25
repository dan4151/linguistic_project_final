import numpy as np
import pickle
from learn_decoder import *
import pandas as pd
from utils import rank_based_accuracy, distance_of_sentence

# Loading the data from all the experiments
data_exp1 = ((pd.read_csv("neuralData_for_EXP1.csv")).drop(columns=['Unnamed: 0'])).values.tolist()
vectors_exp1 = read_matrix("vectors_180concepts.GV42B300.txt", sep=" ")
concepts = np.genfromtxt('stimuli_180concepts.txt', dtype=np.dtype('U'))
with open('EXP2.pkl', 'rb') as file:
    data = pickle.load(file)
vectors_exp2 = read_matrix('vectors_384sentences.GV42B300.average.txt', sep=" ")
passages_exp2 = [item[0][0] for item in data['keyPassages']]
topic_exp2 = [item[0] for sublist in data['keyPassageCategory'] for item in sublist.flatten()]
imaging_data_exp2 = data['Fmridata']



#Trainning on exp1 data and calculating averange_rank_accracy for exp2
accuracy_per_concept_list = []
accuracy_list = []
decoder = learn_decoder(data_exp1, vectors_exp1)


decoded_test_vectors = [np.dot(stimuli, decoder) for stimuli in imaging_data_exp2]
average_rank_accuracy = rank_based_accuracy(decoded_test_vectors, vectors_exp2)
accuracy_list.append(average_rank_accuracy)
print(average_rank_accuracy)
accuracy_dict = {}
#Accuracy analysis
for i, decoded_vec in enumerate(decoded_test_vectors):
    passage_id = data['labelsPassageForEachSentence'][i][0]
    topic_id = data['labelsPassageCategory'][passage_id - 1][0]
    topic = topic_exp2[topic_id - 1]
    if topic not in accuracy_dict.keys():
        accuracy_dict[topic] = {}
        accuracy_dict[topic]['distance_sum'] = 0
        accuracy_dict[topic]['sentence_count'] = 0
    accuracy_dict[topic]['sentence_count'] += 1
    accuracy_dict[topic]['distance_sum'] += distance_of_sentence(decoded_vec, vectors_exp2, i)
accuracy_list = []
for topic in topic_exp2:
    accuracy_list.append((topic, accuracy_dict[topic]['distance_sum']/accuracy_dict[topic]['sentence_count']))

print(sorted(accuracy_list, key=lambda y: y[1], reverse=False))

# normalization of rank accuracy, as in original study
normalized_accuracy = [(x[0], 1 - (x[1] - 1) / (len(decoded_test_vectors) - 1)) for x in accuracy_list]
print(sorted(normalized_accuracy, key=lambda y: y[1]))





