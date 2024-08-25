import matplotlib.pyplot as plt
import pickle

with open('normalized_accuracy_exp3.pkl', 'rb') as file:
    data = pickle.load(file)

data_sorted = sorted(data, key=lambda x: x[1])
topics = [item[0] for item in data_sorted]
scores = [item[1] for item in data_sorted]
cmap = plt.get_cmap('tab20')
colors = [cmap(i / len(topics)) for i in range(len(topics))]
plt.figure(figsize=(12, 8))
for i, (topic, score) in enumerate(data_sorted):
    plt.scatter(i, score, color=colors[i], label=topic)
plt.xticks(range(len(topics)), topics, rotation=90)
plt.ylabel('Rank Accuracy Score')
plt.title('Normalized Rank Accuracy Score per Topic EXP3 Data')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Topics')
plt.tight_layout()
plt.savefig("exp3_plot.png")
plt.show()
