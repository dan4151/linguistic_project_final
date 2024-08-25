import pickle
import numpy as np
import os
import urllib.request
import zipfile
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import f
import seaborn as sns
models = [
    Ridge(),
    Lasso(),
    ElasticNet(),
    LinearRegression(),
    SVR(),
    PLSRegression()
]
model_names = ["Ridge", "Lasso", "ElasticNet", "Linear", "SVR", "PLS"]

# Combine model names with their corresponding models (optional)
models_with_names = list(zip(model_names, models))



# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download GloVe embeddings
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = "glove.6B.zip"
glove_extract_path = "."

if not os.path.exists(glove_zip_path):
    print("Downloading GloVe embeddings...")
    urllib.request.urlretrieve(glove_url, glove_zip_path)
    print("Download complete.")

if not os.path.exists(os.path.join(glove_extract_path, "glove.6B.300d.txt")):
    print("Extracting GloVe embeddings...")
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_extract_path)
    print("Extraction complete.")


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


glove_file_path = os.path.join(glove_extract_path, "glove.6B.300d.txt")
glove_embeddings = load_glove_embeddings(glove_file_path)


def get_glove_embedding(word, embeddings_index, embedding_dim=300):
    return embeddings_index.get(word, np.zeros(embedding_dim))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    dot_product = np.dot(x, y)
    norm_vec1 = np.linalg.norm(x)
    norm_vec2 = np.linalg.norm(y)
    return dot_product / (norm_vec1 * norm_vec2)


def rank_based_accuracy(decoded_test_vectors, semantic_vectors, test_concepts):
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


def test_fmri_decoder(concepts, data, vectors, k=32):
    concepts_groups = np.array_split(concepts, k)
    data_groups = np.array_split(data, k)
    vectors_groups = np.array_split(vectors, k)
    accuracy_per_concept_list = []
    accuracy_list = []
    for i in range(18):
        train_data = []
        train_vectors = []
        for j in range(18):
            if i != j:
                train_data.extend(data_groups[j])
                train_vectors.extend(vectors_groups[j])
        test_data = data_groups[i]
        test_vectors = vectors_groups[i]
        test_concepts = concepts_groups[i]
        decoder = learn_decoder(train_data, train_vectors)
        decoded_test_vectors = [np.dot(test_vec, decoder) for test_vec in test_data]
        average_rank_accuracy, accuracy_per_concept = rank_based_accuracy(decoded_test_vectors, vectors, concepts,
                                                                          test_concepts)
        accuracy_list.append(average_rank_accuracy)
        accuracy_per_concept_list += accuracy_per_concept
    return accuracy_per_concept_list, accuracy_list


def quad_list(lst):
    quad = []
    for i in lst:
        quad.append(i)
        quad.append(i)
        quad.append(i)
        quad.append(i)
    return quad


def average_scores_of_passages(accuracy_per_concept_list):
    score_dict = {}
    for pair in accuracy_per_concept_list:
        try:
            score_dict[pair[0]] += pair[1]
        except KeyError:
            score_dict[pair[0]] = pair[1]
    for key in score_dict.keys():
        score_dict[key] /= 4
    return list(score_dict.items())


def print_by_second_item(lst):
    print(sorted(lst, key=lambda x: x[1]))


def rank_accuracies(data, tokenizer=None, model=None, BERT_flag=False):
    passages = [item[0][0] for item in data['keyPassages']]
    quadded_passages = quad_list(passages)

    # Create BERT embedding for each passage
    if BERT_flag:
        encoded_passages = [tokenizer(passage, return_tensors='pt').to(device) for passage in passages]
        outputs = [model(**enc) for enc in encoded_passages]
        hidden_states = [output.last_hidden_state for output in outputs]
        cls_embeddings = [hidden_state[:, 0, :].cpu().detach().numpy() for hidden_state in hidden_states]
        quadded_embeddings = quad_list(cls_embeddings)
        quadded_embeddings = np.array(quadded_embeddings).reshape(384, 768)
    else:
        encoded_passages = [get_glove_embedding(passage.lower(), glove_embeddings, 300) for passage in passages]
        quadded_embeddings = quad_list(encoded_passages)
        quadded_embeddings = np.array(quadded_embeddings)

    # get FMRI data
    fmri_data = data['Fmridata']
    decoder = learn_decoder(fmri_data, quadded_embeddings)

    # 32-k fold split
    k = 32

    accuracy_per_concept_list, accuracy_list = test_fmri_decoder(quadded_passages, fmri_data, quadded_embeddings)
    print(accuracy_list)
    print_by_second_item(accuracy_per_concept_list)
    print_by_second_item(average_scores_of_passages(accuracy_per_concept_list))


def brain_encoding(embedded_vectors, total_voxel_data, filename, model):
    r2_scores = []

    # Wrapping the loop with tqdm for a progress bar
    for idx, voxel_readings in tqdm(enumerate(np.transpose(total_voxel_data)), total=total_voxel_data.shape[1]):
        model.fit(embedded_vectors, voxel_readings)
        voxel_pred = model.predict(embedded_vectors)
        r2_scores.append((idx, r2_score(voxel_readings, voxel_pred)))

    r2_scores = sorted(r2_scores, reverse=True, key=lambda x: x[1])

    with open(filename, 'w') as f:
        for score in r2_scores:
            f.write(f'idx: {score[0]}, score: {score[1]}\n')

    return r2_scores


def build_array(filepath):
    index_score_dict = {}

    # Reading the CSV file
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line by ', ' to separate idx and score
            parts = line.strip().split(', ')
            # Extract the index and score
            idx = int(parts[0].split(': ')[1])
            score = float(parts[1].split(': ')[1])
            # Store in dictionary
            index_score_dict[idx] = score
    # Determine the size of the array (based on the max index)
    max_idx = max(index_score_dict.keys())

    scores_array = np.zeros(max_idx + 1)
    # Fill the numpy array with scores at the appropriate indices
    for idx, score in index_score_dict.items():
        scores_array[idx] = score
    return scores_array


def build_plot(filepath, title):
    scores_array = build_array(filepath)
    reshaped_array = scores_array.reshape(1, -1)  # This creates a 1xN array (1 row, N columns)
    plt.figure(figsize=(12, 4))  # You can adjust the figsize for better visualization
    plt.scatter(np.arange(len(scores_array)), scores_array, alpha=0.5, color='blue', s=2)
    plt.ylim(0,1)
    plt.xlabel('Voxel index')
    plt.ylabel('R2 Score')
    plt.title(title)
    plt.show()


def statistical_analysis(filepath):
    scores_array = build_array(filepath)
    df1 = 300
    df2 = 83
    F_scores = (scores_array / df1) / ((1 - scores_array) / df2)

    # Calculate the p-values for each F-statistic
    p_values = 1 - f.cdf(F_scores, df1, df2)

    # Determine significance based on the chosen alpha level (e.g., 0.05)
    alpha = 0.05
    significant_voxels = p_values < alpha

    # Calculate the proportion of significant voxels
    proportion_significant = np.mean(significant_voxels)

    print(f"Proportion of voxels with a significant R^2 score: {proportion_significant:.4f}")

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased").to(device)

    # read from files
    with open('EXP2.pkl', 'rb') as f:
        data2 = pickle.load(f)
    with open('EXP3.pkl', 'rb') as f:
        data3 = pickle.load(f)
    with open('vectors_384sentences.GV42B300.average.txt', 'r') as f:
        txt = f.read()

    # Parse the glove embedding for each sentence and the passage topics
    glv_embedding_384 = txt.split('\n')[:-1]
    glv_embedding_384 = np.array([np.array(item.split(' ')[:-1]).astype(float) for item in glv_embedding_384])
    sentences = [item[0][0] for item in data2['keySentences']]
    encoded_sentences = [tokenizer(sentence, return_tensors='pt').to(device) for sentence in sentences]
    outputs = [model(**enc) for enc in encoded_sentences]
    hidden_states = [output.last_hidden_state for output in outputs]
    bert_embeddings = [np.array(hidden_state[:, 0, :].cpu().detach().numpy()).reshape(768) for hidden_state in
                       hidden_states]

    # rank_accuracies(data2, tokenizer, model, True)
    # rank_accuracies(data2)
    brain_encoding(glv_embedding_384, data2['Fmridata'], 'bert_results_linear.txt',LinearRegression())
    #brain_encoding(bert_embeddings, data2['Fmridata'], 'bert_results_elastic.txt', ElasticNet())
    #brain_encoding(bert_embeddings, data2['Fmridata'], 'svr_results.txt', SVR())
    print("Done")


if __name__ == '__main__':
    build_plot('bert_results_elastic.txt', 'Default elastic model with bert embeddings')
    build_plot('bert_results.txt', 'Linear regression model with bert embeddings')
    build_plot('log_r2_scores.txt', 'Linear regression  model with glove embeddings')
    build_plot('svr_results.txt', 'SVR model with bert embeddings')
    #statistical_analysis('log_r2_scores.txt')
    #main()
