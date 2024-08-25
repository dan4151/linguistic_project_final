import pickle
import matplotlib.pyplot as plt
import random
import numpy as np


with open('pickled/accuracy_list_Glove.pkl', 'rb') as file:
    accuracy_list_Glove = pickle.load(file)
with open('pickled/accuracy_list_Word2Vec.pkl', 'rb') as file:
    accuracy_list_Word2Vec = pickle.load(file)
with open('pickled/accuracy_per_concept_list_Glove.pkl', 'rb') as file:
    accuracy_per_concept_list_Glove = pickle.load(file)
with open('pickled/accuracy_per_concept_list_Word2Vec.pkl', 'rb') as file:
    accuracy_per_concept_list_Word2Vec = pickle.load(file)

print(accuracy_list_Glove)
print(accuracy_list_Word2Vec)
print(accuracy_per_concept_list_Glove)
print(accuracy_per_concept_list_Word2Vec)
x = range(1, 19)
plt.figure(figsize=(10, 5))

# Plot the first list
plt.plot(x, accuracy_list_Glove, label='Glove', color='blue', linestyle='-', marker='o')

# Plot the second list
plt.plot(x, accuracy_list_Word2Vec, label='Word2Vec', color='red', linestyle='-', marker='o')
plt.xticks(ticks=range(1, 19))

# Add title and labels
plt.title('Average Rank Accuracy per k-fold')
plt.xlabel('k-fold')
plt.ylabel('Average Rank Accuracy')

# Add a legend
plt.legend()
# Display the plot
plt.show()

concepts1, values1 = zip(*accuracy_per_concept_list_Glove)
concepts2, values2 = zip(*accuracy_per_concept_list_Word2Vec)

# Plotting the data
plt.figure(figsize=(15, 10))
plt.scatter(concepts1, values1, color='blue', label='List 1')
plt.scatter(concepts2, values2, color='red', label='List 2')
plt.xticks(rotation=90)
plt.xlabel('Concept')
plt.ylabel('Number')
plt.title('Concept vs Number')
plt.legend()
plt.grid(True)
plt.show()