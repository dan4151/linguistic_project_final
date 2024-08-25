import pickle
from decoders import learn_decoders
from learn_decoder import *
from utils import exp1_data_loader, distance_of_sentence, rank_based_accuracy
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table


def train_open_task_exp2():
    data_exp1, vectors_exp1, concepts_exp1 = (exp1_data_loader
                                              ("neuralData_for_EXP1.csv", "vectors_180concepts.GV42B300.txt",
                                               'stimuli_180concepts.txt'))
    with open('EXP2.pkl', 'rb') as file:
        data = pickle.load(file)
    vectors_exp2 = read_matrix('vectors_384sentences.GV42B300.average.txt', sep=" ")
    passages_exp2 = [item[0][0] for item in data['keyPassages']]
    topic_exp2 = [item[0] for sublist in data['keyPassageCategory'] for item in sublist.flatten()]
    imaging_data_exp2 = data['Fmridata']

    # Trainning on exp1 data and calculating averange_rank_accracy for exp2
    accuracy_per_concept_list = []
    regression_types = ["ridge", "svr", "pls", "pcr"]
    results_dict = {}
    for reg_type in regression_types:
        decoder = learn_decoders(data_exp1, vectors_exp1, reg_type)
        decoded_test_vectors = [np.dot(stimuli, decoder) for stimuli in imaging_data_exp2]
        accuracy_dict = {}
        # Accuracy analysis
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
            accuracy_list.append((topic, accuracy_dict[topic]['distance_sum'] / accuracy_dict[topic]['sentence_count']))

        print(sorted(accuracy_list, key=lambda y: y[1], reverse=False))

        # normalization of rank accuracy, as in original study
        normalized_accuracy = [(x[0], 1 - (x[1] - 1) / (len(decoded_test_vectors) - 1)) for x in accuracy_list]
        print(sorted(normalized_accuracy, key=lambda y: y[1]))
        results_dict[reg_type] = normalized_accuracy
    print(results_dict)
    with open('results_open_task_2.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

def plot_open_task_exp2_a():
    with open('results_open_task_2.pkl', 'rb') as file:
        data = pickle.load(file)
    topics = [item[0] for item in data['ridge']]
    accuracies_ridge = [item[1] for item in data['ridge']]
    accuracies_svr = [item[1] for item in data['svr']]
    accuracies_pls = [item[1] for item in data['pls']]
    accuracies_pcr = [item[1] for item in data['pcr']]
    x = np.arange(len(topics))
    width = 0.2  # Width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Bar positions for each method
    bar_width = width
    offset = -1.5 * bar_width
    for i, (label, accuracies) in enumerate(
            zip(['Ridge', 'SVR', 'PLS', 'PCR'], [accuracies_ridge, accuracies_svr, accuracies_pls, accuracies_pcr])):
        ax.bar(x + i * bar_width, accuracies, bar_width, label=label)

    # Customize plot
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(topics, rotation=90)
    ax.set_xlabel('Topics')
    ax.set_ylabel('Accuracy')
    ax.set_title('Normalized Rank Accuracy by Topic and Regression Method (Experiment 2 data)')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_open_task_exp2_b():
    with open('results_open_task_2.pkl', 'rb') as file:
        data = pickle.load(file)

    topics = [item[0] for item in data['ridge']]
    accuracies_ridge = [item[1] for item in data['ridge']]
    accuracies_svr = [item[1] for item in data['svr']]
    accuracies_pls = [item[1] for item in data['pls']]
    accuracies_pcr = [item[1] for item in data['pcr']]
    print("ridge: " + str(sum(accuracies_ridge) / len(accuracies_ridge)))
    print("svr: " + str(sum(accuracies_svr) / len(accuracies_svr)))
    print("pls: " + str(sum(accuracies_pls) / len(accuracies_pls)))
    print("pcr: " + str(sum(accuracies_pcr) / len(accuracies_pcr)))




if __name__ == "__main__":
    #train_open_task_exp2()
    #plot_open_task_exp2_a()
    plot_open_task_exp2_b()

