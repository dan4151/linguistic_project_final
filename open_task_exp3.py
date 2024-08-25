import numpy as np
import pickle
from learn_decoder import *
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import torch
import matplotlib.pyplot as plt
from utils import  k_fold_open_task
import pandas as pd
def train_open_task_3():
    with open('EXP2.pkl', 'rb') as file:
        data = pickle.load(file)
    vectors_exp2 = read_matrix('vectors_384sentences.GV42B300.average.txt', sep=" ")
    topic_exp2 = [item[0] for sublist in data['keyPassageCategory'] for item in sublist.flatten()]
    imaging_data_exp2 = data['Fmridata']
    topics_per_sentence = []
    for i in range(len(imaging_data_exp2)):
        passage_id = data['labelsPassageForEachSentence'][i][0]
        topic_id = data['labelsPassageCategory'][passage_id - 1][0]
        topics_per_sentence.append(topic_exp2[topic_id - 1])
    print(len(topics_per_sentence))
    # We will use k-fold to train and test
    k = 12
    vectors_groups_Glove = np.array_split(vectors_exp2, k)
    topics_groups = np.array_split(topics_per_sentence, k)
    imaging_data_groups = np.array_split(imaging_data_exp2, k)

    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    with open('stimuli_384sentences.txt', 'r') as file:
        sentences = file.readlines()
    sentences = [sentence.strip() for sentence in sentences]
    bert_embeddings = []
    for sentence in sentences:
        inputs = tokenizer_bert(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        sentence_embedding_cls = outputs.last_hidden_state[:, 0, :].squeeze()
        bert_embeddings.append(sentence_embedding_cls)

    vectors_groups_bert = np.array_split(bert_embeddings, k)


    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    model_gpt2 = GPT2Model.from_pretrained('gpt2')
    gpt2_embedding = []
    for sentence in sentences:
        inputs = tokenizer_gpt2(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model_gpt2(**inputs)
        sentence_embedding_avg = outputs.last_hidden_state.mean(dim=1).squeeze()
        gpt2_embedding.append(sentence_embedding_avg)

    regression_types = ["ridge", "svr", "pls", "pcr"]
    results = {}
    vectors_groups_gpt2 = np.array_split(gpt2_embedding, k)
    for reg_type in regression_types:
        results[reg_type] = {}

        accuracy_list_bert, accuracy_per_concept_list_bert = k_fold_open_task(imaging_data_groups, bert_embeddings,
                                                                            vectors_groups_bert, topics_groups, k, reg_type)
        accuracy_list_glove, accuracy_per_concept_list_glove = k_fold_open_task(imaging_data_groups, vectors_exp2,
                                                                            vectors_groups_Glove, topics_groups, k, reg_type)
        results[reg_type]['glove'] = accuracy_list_glove
        results[reg_type]['bert'] = accuracy_list_bert

    with open('results_open_task_3.pkl', 'wb') as file:
        pickle.dump(results, file)

def plot_open_task_exp3():
    with open('results_open_task_3.pkl', 'rb') as file:
        data = pickle.load(file)
    ridge_glove = data['ridge']['glove']
    ridge_bert = data['ridge']['bert']
    svr_glove = data['svr']['glove']
    svr_bert = data['svr']['bert']
    pls_glove = data['pls']['glove']
    pls_bert = data['pls']['bert']
    pcr_glove = data['pcr']['glove']
    pcr_bert = data['pcr']['bert']
    x = range(1, 13)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for GloVe
    ax1.plot(x, ridge_glove, label='Ridge', marker='o', linestyle='-', color='blue')
    ax1.plot(x, svr_glove, label='SVR', marker='o', linestyle='-', color='orange')
    ax1.plot(x, pls_glove, label='PLS', marker='o', linestyle='-', color='green')
    ax1.plot(x, pcr_glove, label='PCR', marker='o', linestyle='-', color='red')

    ax1.set_title('Average Rank Accuracy per k-fold (GloVe)')
    ax1.set_xlabel('k-fold')
    ax1.set_ylabel('Average Rank Accuracy')
    ax1.set_xticks(x)
    ax1.legend()

    # Plot for BERT
    ax2.plot(x, ridge_bert, label='Ridge', marker='o', linestyle='-', color='blue')
    ax2.plot(x, svr_bert, label='SVR', marker='o', linestyle='-', color='orange')
    ax2.plot(x, pls_bert, label='PLS', marker='o', linestyle='-', color='green')
    ax2.plot(x, pcr_bert, label='PCR', marker='o', linestyle='-', color='red')

    ax2.set_title('Average Rank Accuracy per k-fold (BERT)')
    ax2.set_xlabel('k-fold')
    ax2.set_ylabel('Average Rank Accuracy')
    ax2.set_xticks(x)
    ax2.legend()

    y_min = min(min(ridge_glove), min(svr_glove), min(pls_glove), min(pcr_glove),
                min(ridge_bert), min(svr_bert), min(pls_bert), min(pcr_bert))
    y_max = max(max(ridge_glove), max(svr_glove), max(pls_glove), max(pcr_glove),
                max(ridge_bert), max(svr_bert), max(pls_bert), max(pcr_bert))

    ax1.set_ylim(y_min - 10, y_max + 10)
    ax2.set_ylim(y_min - 10, y_max + 10)
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_open_task_exp3_b():
    with open('results_open_task_3.pkl', 'rb') as file:
        data = pickle.load(file)
    ridge_glove = data['ridge']['glove']
    ridge_bert = data['ridge']['bert']
    svr_glove = data['svr']['glove']
    svr_bert = data['svr']['bert']
    pls_glove = data['pls']['glove']
    pls_bert = data['pls']['bert']
    pcr_glove = data['pcr']['glove']
    pcr_bert = data['pcr']['bert']
    best_keys_glove = []
    best_keys_word2vec = []
    for i in range(12):
        glove_values = {
            'ridge': ridge_glove[i],
            'svr': svr_glove[i],
            'pls': pls_glove[i],
            'pcr': pcr_glove[i]
        }
        best_key_glove = min(glove_values, key=glove_values.get)
        best_keys_glove.append(best_key_glove)

        # Find the maximum value and corresponding key for Word2Vec
        bert_values = {
            'ridge': ridge_bert[i],
            'svr': svr_bert[i],
            'pls': pls_bert[i],
            'pcr': pcr_bert[i]
        }
        best_key_word2vec = min(bert_values, key=bert_values.get)
        best_keys_word2vec.append(best_key_word2vec)
    print(best_keys_word2vec)
    print(best_keys_glove)
    df = pd.DataFrame({
        'Fold': range(1, 13),
        'Best BERT Regression': best_keys_word2vec,
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
    for i, key_col in enumerate(['Best BERT Regression', 'Best GloVe Regression']):
        for j in range(len(df)):
            key = df[key_col].iloc[j]
            color = color_map.get(key, '#FFFFFF')  # Default to white if key not found
            table[(j + 1, i + 1)].set_facecolor(color)
            # Save the figure
    plt.show()
if __name__ == "__main__":
    #train_open_task_3()
    #plot_open_task_exp3()
    plot_open_task_exp3_b()

