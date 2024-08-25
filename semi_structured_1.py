import numpy as np
import pickle
from learn_decoder import *
import pandas as pd
from utils import distance_of_sentence, exp1_data_loader, k_fold
from decoders import learn_decoders
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import torch
import matplotlib.pyplot as plt


with open('EXP2.pkl', 'rb') as file:
    data = pickle.load(file)
vectors_exp2 = read_matrix('vectors_384sentences.GV42B300.average.txt', sep=" ")
passages_exp2 = [item[0][0] for item in data['keyPassages']]
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


accuracy_list_glove, accuracy_per_concept_list_glove = k_fold(imaging_data_groups, vectors_exp2,
                                                                    vectors_groups_Glove, topics_groups, k)

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
accuracy_list_bert, accuracy_per_concept_list_bert = k_fold(imaging_data_groups, bert_embeddings,
                                                                    vectors_groups_bert, topics_groups, k)

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2Model.from_pretrained('gpt2')
gpt2_embedding = []
for sentence in sentences:
    inputs = tokenizer_gpt2(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model_gpt2(**inputs)
    sentence_embedding_avg = outputs.last_hidden_state.mean(dim=1).squeeze()
    gpt2_embedding.append(sentence_embedding_avg)

vectors_groups_gpt2 = np.array_split(gpt2_embedding, k)
accuracy_list_gpt2, accuracy_per_concept_list_gpt2 = k_fold(imaging_data_groups, gpt2_embedding,
                                                                    vectors_groups_gpt2, topics_groups, k)
print(accuracy_list_gpt2)
x = range(1, 13)
plt.figure(figsize=(10, 5))

# glove
plt.plot(x, accuracy_list_glove, label='Glove', color='blue', linestyle='-', marker='o')

# bert
plt.plot(x, accuracy_list_bert, label='BERT', color='red', linestyle='-', marker='o')
plt.xticks(ticks=range(1, 13))

# GPT2
plt.plot(x, accuracy_list_gpt2, label='GPT2', color='green', linestyle='-', marker='o')
plt.xticks(ticks=range(1, 13))

# Add title and labels
plt.title('Average Rank Accuracy per k-fold')
plt.xlabel('k-fold')
plt.ylabel('Average Rank Accuracy')

# Add a legend
plt.legend()
# Display the plot
plt.show()

