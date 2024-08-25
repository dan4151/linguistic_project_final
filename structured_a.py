import numpy as np
from learn_decoder import *
import pickle
from gensim.models import KeyedVectors
from utils import k_fold


data = read_matrix("imaging_data.csv", sep=",")
glove_vectors = read_matrix("vectors_180concepts.GV42B300.txt", sep=" ")
concepts = np.genfromtxt('stimuli_180concepts.txt', dtype=np.dtype('U')) #The names of the 180 concepts

word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_vectors = []
for concept in concepts:
    if concept in word2vec_model.key_to_index:
        word2vec_vectors.append(word2vec_model[concept])
    if concept == "argumentatively":
        word2vec_vectors.append(word2vec_model['argumentative'])


# 18-k fold split
concepts_groups = np.array_split(concepts, 18)
data_groups = np.array_split(data, 18)
vectors_groups_Word2Vec = np.array_split(word2vec_vectors, 18)
vectors_groups_Glove = np.array_split(glove_vectors, 18)


k = 18
# Word2Vec Train and analysis
accuracy_list_Word2Vec, accuracy_per_concept_list_Word2Vec = k_fold(data_groups, word2vec_vectors,
                                                                    vectors_groups_Word2Vec, concepts_groups, k)

accuracy_list_Glove, accuracy_per_concept_list_Glove = k_fold(data_groups, glove_vectors,
                                                                    vectors_groups_Glove, concepts_groups, k)



print(sorted(accuracy_per_concept_list_Word2Vec, key=lambda x: x[1], reverse=False))
print(sorted(accuracy_per_concept_list_Glove, key=lambda x: x[1], reverse=False))

with open('accuracy_per_concept_list_Word2Vec.pkl', 'wb') as file:
    pickle.dump(accuracy_per_concept_list_Word2Vec, file)
with open('accuracy_per_concept_list_Glove,pkl', 'wb') as file:
    pickle.dump(accuracy_per_concept_list_Glove, file)
with open('accuracy_list_Glove.pkl', 'wb') as file:
    pickle.dump(accuracy_list_Glove, file)
with open('accuracy_list_Word2Vec.pkl', 'wb') as file:
    pickle.dump(accuracy_list_Word2Vec, file)