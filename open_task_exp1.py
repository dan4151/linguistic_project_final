import pickle
from learn_decoder import *
from utils import exp1_data_loader, k_fold_open_task
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd


def train_open_task_exp1():
    data_exp1, vectors_exp1, concepts_exp1 = (exp1_data_loader
    ("neuralData_for_EXP1.csv", "vectors_180concepts.GV42B300.txt", 'stimuli_180concepts.txt'))
    word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec_vectors = []
    for concept in concepts_exp1:
        if concept in word2vec_model.key_to_index:
            word2vec_vectors.append(word2vec_model[concept])
        if concept == "argumentatively":
            word2vec_vectors.append(word2vec_model['argumentative'])
    concepts_groups = np.array_split(concepts_exp1, 18)
    data_groups = np.array_split(data_exp1, 18)
    vectors_groups_Word2Vec = np.array_split(word2vec_vectors, 18)
    vectors_groups_Glove = np.array_split(vectors_exp1, 18)
    k = 18
    regression_types = ["ridge", "svr", "pls", "pcr"]
    results = {}
    for reg_type in regression_types:
        results[reg_type] = {}
        accuracy_list_Word2Vec, accuracy_per_concept_list_Word2Vec = k_fold_open_task(data_groups, word2vec_vectors,
                                                                      vectors_groups_Word2Vec, concepts_groups, k, reg_type)
        accuracy_list_Glove, accuracy_per_concept_list_Glove = k_fold_open_task(data_groups, vectors_exp1,
                                                                      vectors_groups_Glove, concepts_groups, k, reg_type)
        results[reg_type]['glove'] = accuracy_list_Glove
        results[reg_type]['Word2Vec'] = accuracy_list_Word2Vec
    with open('results_open_task_1.pkl', 'wb') as file:
        pickle.dump(results, file)


def plot_open_task_exp1_a():
    with open('results_open_task_1.pkl', 'rb') as file:
        data = pickle.load(file)
    ridge_glove = data['ridge']['glove']
    svr_glove = data['svr']['glove']
    pls_glove = data['pls']['glove']
    pcr_glove = data['pcr']['glove']
    ridge_word2vec = data['ridge']['Word2Vec']
    svr_word2vec = data['svr']['Word2Vec']
    pls_word2vec = data['pls']['Word2Vec']
    pcr_word2vec = data['pcr']['Word2Vec']
    x = range(1, 19)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(x, ridge_glove, label='ridge')
    ax1.plot(x, svr_glove, label='svr')
    ax1.plot(x, pls_glove, label='pls')
    ax1.plot(x, pcr_glove, label='pcr')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.set_title('Average Rank Accuracy per k-fold (GloVe)')
    ax1.set_xlabel('k-fold')
    ax1.set_ylabel('Average Rank Accuracy')
    ax2.plot(x, ridge_word2vec, label='ridge')
    ax2.plot(x, svr_word2vec, label='svr')
    ax2.plot(x, pls_word2vec, label='pls')
    ax2.plot(x, pcr_word2vec, label='pcr')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.set_title('Average Rank Accuracy per k-fold (Word2Vec)')
    ax2.set_xlabel('k-fold')
    ax2.set_ylabel('Average Rank Accuracy')

    # Adding a legend to distinguish the lines
    plt.legend()

    # Display the plot
    plt.show()
def plot_open_task_exp1_b():
    with open('results_open_task_1.pkl', 'rb') as file:
        data = pickle.load(file)
    ridge_glove = data['ridge']['glove']
    svr_glove = data['svr']['glove']
    pls_glove = data['pls']['glove']
    pcr_glove = data['pcr']['glove']
    ridge_word2vec = data['ridge']['Word2Vec']
    svr_word2vec = data['svr']['Word2Vec']
    pls_word2vec = data['pls']['Word2Vec']
    pcr_word2vec = data['pcr']['Word2Vec']
    best_keys_glove = []
    best_keys_word2vec = []
    for i in range(18):
        glove_values = {
            'ridge': ridge_glove[i],
            'svr': svr_glove[i],
            'pls': pls_glove[i],
            'pcr': pcr_glove[i]
        }
        best_key_glove = min(glove_values, key=glove_values.get)
        best_keys_glove.append(best_key_glove)

        # Find the maximum value and corresponding key for Word2Vec
        word2vec_values = {
            'ridge': ridge_word2vec[i],
            'svr': svr_word2vec[i],
            'pls': pls_word2vec[i],
            'pcr': pcr_word2vec[i]
        }
        best_key_word2vec = min(word2vec_values, key=word2vec_values.get)
        best_keys_word2vec.append(best_key_word2vec)
    print(best_keys_word2vec)
    print(best_keys_glove)
    df = pd.DataFrame({
        'Fold': range(1, 19),
        'Best Word2Vec Regression': best_keys_word2vec,
        'Best GloVe Regression': best_keys_glove
    })
    color_map = {
        'pcr': '#d42e33',  # Red
        'pls': '#44a754',  # Green
        'ridge': '#5998c6',  # Blue
        'svr': '#fea04f'  # orange
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    for i, key_col in enumerate(['Best Word2Vec Regression', 'Best GloVe Regression']):
        for j in range(len(df)):
            key = df[key_col].iloc[j]
            color = color_map.get(key, '#FFFFFF')  # Default to white if key not found
            table[(j + 1, i + 1)].set_facecolor(color)
            # Save the figure
    plt.show()
if __name__ == "__main__":
    #train_open_task_exp1()
    plot_open_task_exp1_b()