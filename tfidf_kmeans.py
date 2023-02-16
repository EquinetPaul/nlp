# Import necessary libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")

# Get the top keywords for each cluster
def get_top_keywords(data, clusters, labels, n_terms):
    res_clusters = {}
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    for i,r in df.iterrows():
        res_clusters[i] = [labels[t] for t in np.argsort(r)[-n_terms:]]

    return res_clusters

# Apply TF-IDF on the corpus
def apply_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectors, vectorizer

# Apply KMeans on vectors
def apply_kmeans(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(vectors)
    return kmeans

# Find the optimal number of clusters (k) using the elbow method
def find_optimal_k(vectors, k_max):
    inertias = []
    iter_max = range(2,k_max+1)
    batch_perc = 0.5
    batch_size = int(vectors.shape[0]*batch_perc)

    # Compute KMeans and get its inertia for different values of k
    for k in tqdm(iter_max):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(vectors)
        inertias.append(kmeans.inertia_)

    # Plot the inertia values vs. the number of clusters
    fig, ax = plt.subplots()
    ax.plot(range(2, len(inertias) + 2), inertias)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')
    ax.grid()

    # Plot a line connecting the first and last inertia values
    x1, y1 = 2, inertias[0]
    x2, y2 = len(inertias) + 1, inertias[-1]
    ax.plot([x1, x2], [y1, y2], 'r--')

# Calculate the maximum distance between the inertia points and the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    distances = [abs(y - (m * x + b)) / np.sqrt(1 + m**2) for x, y in enumerate(inertias[1:-1])]

    max_distance = max(distances)
    max_index = distances.index(max_distance)
    x_max, y_max = max_index + 3, inertias[max_index+1]
    best_k = max_index + 3

    # Plot a red point at the maximum distance
    ax.scatter(x_max, y_max, color='y', s=100)

    # Plot green dotted lines connecting each inertia point to the line
    for x, y in enumerate(inertias[1:-1]):
        xi = x + 3
        yi = y
        m_perp = -1/m
        b_perp = yi - m_perp * xi
        xi_intersect = (b_perp - b) / (m - m_perp)
        yi_intersect = m * xi_intersect + b
        ax.plot([xi, xi_intersect], [yi, yi_intersect], 'g:')


    return best_k, fig

def plot_tsne_pca(data, labels):
    # Get the maximum label in the dataset
    max_label = max(labels)
    # Randomly select 3000 data points
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    # Compute PCA and TSNE on the randomly selected 3000 data points
    pca = PCA(n_components=2).fit_transform(np.array(data[max_items,:].todense()))
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(np.array(data[max_items,:].todense())))

    # Randomly select 300 data points for visualization
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    # Get the labels of the selected data points
    label_subset = labels[max_items]

    # Create a custom color table for the clusters
    n_clusters = len(set(label_subset))
    cmap = get_cmap('jet', n_clusters)
    colors = cmap(np.linspace(0, 1, n_clusters))
    cmap = ListedColormap(colors)

    # Assign colors to the selected data points based on their labels
    label_subset = [cmap(i) for i in label_subset[idx]]

    # Create a scatter plot of the selected data points using the TSNE coordinates
    f, ax = plt.subplots()
    ax.scatter(tsne[idx, 0], tsne[idx, 1], color=label_subset)
    ax.set_title('TSNE Cluster Plot')

    # Add a legend to the plot
    handles = []
    for i in range(max_label + 1):
        handle = ax.scatter([], [], c=cmap(i), label='Cluster {}'.format(i))
        handles.append(handle)
    ax.legend(handles=handles)

    return f

# Main function that performs topic modeling
def compute(data, k_max, n_top_words):
    # Apply TF-IDF vectorization to the input data
    vectors, vectorizer = apply_tfidf(data)
    # Find the optimal number of clusters
    best_k, plot_inertia = find_optimal_k(vectors, k_max)
    # Apply KMeans clustering with the optimal number of clusters
    kmeans = apply_kmeans(vectors, best_k)
    # Get the top keywords for each cluster
    clusters = get_top_keywords(vectors, kmeans, vectorizer.get_feature_names_out(), n_top_words)
    # Plot the clusters using PCA and t-SNE dimension reduction techniques
    plot_clusters = plot_tsne_pca(vectors, kmeans)

    # Print the top keywords for each cluster
    for key, value in clusters.items():
        print(key, value)

    # Show the plot of inertia vs number of clusters
    plot_inertia.show()
    # Show the plot of clusters
    plot_clusters.show()
    # Wait for user input
    input()
