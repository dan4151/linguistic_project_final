from learn_decoder import *
import pandas as pd
from decoders import learn_decoders


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    dot_product = np.dot(x, y)
    norm_vec1 = np.linalg.norm(x)
    norm_vec2 = np.linalg.norm(y)
    return dot_product / (norm_vec1 * norm_vec2)

def rank_based_accuracy(decoded_vectors, semantic_vectors):
    rank_sums = 0
    for i, decoded_vec in enumerate(decoded_vectors):
        cosaines = [(semantic_vec, cosine_similarity(decoded_vec, semantic_vec)) for semantic_vec in semantic_vectors]
        sorted_cosaines = sorted(cosaines, key=lambda y: y[1], reverse=True)
        for j, x in enumerate(sorted_cosaines):
            if np.array_equal(x[0], semantic_vectors[i]):
                index = j + 1
                break
        rank_sums += index
    return rank_sums / len(decoded_vectors)


def rank_based_accuracy_with_concepts(decoded_test_vectors, semantic_vectors, test_concepts):
    """
    :parameter:
    decoded_test_vectors: decoded 10 imaging data of test fold
    semantic_vectors: all 180 semantic vectors
    test_concepts: 10 concepts of the semantic vectors
    :return:
    rank_sums: average rank of test vectors
    distance_per_concept: list of order of closeness of each decoded test vector to semantic vectors
    :param decoded_test_vectors:
    :param semantic_vectors:
    :param test_concepts:
    :return:
    """

    distance_per_concept = []
    rank_sum = 0
    for i, decoded_vec in enumerate(decoded_test_vectors):
        cosaines = [(semantic_vec, cosine_similarity(decoded_vec, semantic_vec)) for semantic_vec in semantic_vectors]
        sorted_cosaines = sorted(cosaines, key=lambda y: y[1], reverse=True)
        for j, x in enumerate(sorted_cosaines):
            if np.array_equal(x[0], semantic_vectors[i]):
                index = j + 1
        distance_per_concept.append((test_concepts[i], index))
        rank_sum += index
    return rank_sum / len(decoded_test_vectors), distance_per_concept

def distance_of_sentence(decoded_vec, semantic_vectors, sematic_vector_index):
    cosaines = [(semantic_vec, cosine_similarity(decoded_vec, semantic_vec)) for semantic_vec in semantic_vectors]
    sorted_cosaines = sorted(cosaines, key=lambda y: y[1], reverse=True)
    for j, x in enumerate(sorted_cosaines):
        if np.array_equal(x[0], semantic_vectors[sematic_vector_index]):
            index = j + 1
            break
    return index

def exp1_data_loader(neural_data, vectors, concepts):
    """
    :param neural_data, vectors, concepts: files of data from experiment 1
    :return: loaded data as lists
    """
    data = ((pd.read_csv(neural_data)).drop(columns=['Unnamed: 0'])).values.tolist()
    vectors = read_matrix(vectors, sep=" ")
    concepts = np.genfromtxt(concepts, dtype=np.dtype('U'))
    return data, vectors, concepts

def k_fold(fmri_data_groups, vectors, vector_grups, topic_grups, k_folds):
    accuracy_list = []
    accuracy_per_concept_list = []
    for i in range(k_folds):
        train_data = []
        train_vectors = []
        for j in range(k_folds):
            if i != j:
                train_data.extend(fmri_data_groups[j])
                train_vectors.extend(vector_grups[j])
        test_data = fmri_data_groups[i]
        test_topics = topic_grups[i]
        decoder = learn_decoder(train_data, train_vectors)
        decoded_test_vectors = [np.dot(test_vec, decoder) for test_vec in test_data]
        average_rank_accuracy, accuracy_per_concept = rank_based_accuracy_with_concepts(decoded_test_vectors, vectors,
                                                                          test_topics)
        accuracy_list.append(average_rank_accuracy)
        accuracy_per_concept_list += accuracy_per_concept
    return accuracy_list, accuracy_per_concept_list


def k_fold_open_task(fmri_data_groups, vectors, vector_grups, topic_grups, k_folds, reg_type):
    accuracy_list = []
    accuracy_per_concept_list = []
    for i in range(k_folds):
        train_data = []
        train_vectors = []
        for j in range(k_folds):
            if i != j:
                train_data.extend(fmri_data_groups[j])
                train_vectors.extend(vector_grups[j])
        test_data = fmri_data_groups[i]
        test_topics = topic_grups[i]
        decoder = learn_decoders(train_data, train_vectors, reg_type)
        decoded_test_vectors = [np.dot(test_vec, decoder) for test_vec in test_data]
        average_rank_accuracy, accuracy_per_concept = rank_based_accuracy_with_concepts(decoded_test_vectors, vectors,
                                                                          test_topics)
        accuracy_list.append(average_rank_accuracy)
        accuracy_per_concept_list += accuracy_per_concept
    return accuracy_list, accuracy_per_concept_list

