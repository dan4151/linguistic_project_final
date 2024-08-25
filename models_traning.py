import numpy as np
import pickle
from learn_decoder import *
import pandas as pd
from utils import rank_based_accuracy, distance_of_sentence, exp1_data_loader
from decoders import learn_decoders

data_exp1, vectors_exp1, concepts_exp1 = (exp1_data_loader
("neuralData_for_EXP1.csv", "vectors_180concepts.GV42B300.txt", 'stimuli_180concepts.txt'))
with open('EXP2.pkl', 'rb') as file:
    data = pickle.load(file)
vectors_exp2 = read_matrix('vectors_384sentences.GV42B300.average.txt', sep=" ")
passages_exp2 = [item[0][0] for item in data['keyPassages']]
topic_exp2 = [item[0] for sublist in data['keyPassageCategory'] for item in sublist.flatten()]
imaging_data_exp2 = data['Fmridata']
print(len(imaging_data_exp2), len(imaging_data_exp2[0]))

accuracy_per_concept_list = []
accuracy_list = []
regression_types = ["ridge", "svr", "pls", "pcr"]

decoder = {}
for reg_type in regression_types:

    decoded_test_vectors[reg_type] = [np.dot(stimuli, decoder[reg_type]) for stimuli in imaging_data_exp2]
    average_rank_accuracy[reg_type] = rank_based_accuracy(decoded_test_vectors[reg_type], vectors_exp2)

    accuracy_dict = {reg_type: {}}
    # Accuracy analysis
    for i, decoded_vec in enumerate(decoded_test_vectors[reg_type]):
        passage_id = data['labelsPassageForEachSentence'][i][0]
        topic_id = data['labelsPassageCategory'][passage_id - 1][0]
        topic = topic_exp2[topic_id - 1]
        if topic not in accuracy_dict.keys():
            accuracy_dict[reg_type][topic] = {}
            accuracy_dict[reg_type][topic]['distance_sum'] = 0
            accuracy_dict[reg_type][topic]['sentence_count'] = 0
        accuracy_dict[reg_type][topic]['sentence_count'] += 1
        accuracy_dict[reg_type][topic]['distance_sum'] += distance_of_sentence(decoded_vec, vectors_exp2, i)
    accuracy_list = []
    for topic in topic_exp2:
        accuracy_list.append(
            (topic, accuracy_dict[reg_type][topic]['distance_sum'] / accuracy_dict[reg_type][topic]['sentence_count']))

    print(accuracy_list)

    # normalization of rank accuracy, as in original study
    normalized_accuracy = [(x[0], 1 - (x[1] - 1) / (len(decoded_test_vectors[reg_type]) - 1)) for x in accuracy_list]
    print("Accuracy per concept for regression type: " + reg_type)
    print(sorted(normalized_accuracy, key=lambda y: y[1]))
    normalized_accuracy_dict[reg_type] = {}
    normalized_accuracy_dict[reg_type] = sorted(normalized_accuracy, key=lambda y: y[1])

with open('normalized_accuracy_per_reg_type.pkl', 'wb') as f:
    pickle.dump(normalized_accuracy_dict, f)


