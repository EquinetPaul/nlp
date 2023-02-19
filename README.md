# NLP
NLP lib

# Table of content
1. [tfidf_kmeans](#tfidf_kmeans)
2. [cosine_metrics](#cosine_metrics)

# tfidf_kmeans <a id="tfidf_kmeans"></a>
This library performs TF-IDF vectorization on a corpus of documents and determines the optimal number of clusters of KMeans. Once the optimal number of clusters has been identified, the library creates a topic representation for each cluster by selecting the top n terms that are most representative of that cluster. 

## Method & Results
To determine the optimal number of clusters of KMeans we use the "Elbow method" with the inertia of each cluster.

![plot_tfidf_kmeans](https://github.com/EquinetPaul/EquinetPaul/blob/main/plot_tfidf_kmeans.PNG?raw=true)

The results of the computation is:
- Topics with representative words
- Plot used to determine the optimal number of clusters
- Plot of the clusters (reduced with tsne & pca)

## Usage:
```python
import pandas as pd
import tfidf_kmeans

# Load data
data = pd.read_csv("data.csv", encoding="utf-8")["column_to_compute"].to_list()
data = [d.lower() for d in data if type(d)==str]

# Apply Topic Modeling
tfidf_kmeans.compute(
    data, 
    k_max = 10, # Used to determine the best number of clusters in range(2, k_max)
    n_top_words = 5 # Used to show the n_top_words of each clusters for topic representation
    )
```

```bash
0 ['windows', 'drive', 'dos', 'file', 'os']
1 ['space','launch','orbit','lunar','planet']
2 ['key', 'encryption', 'keys', 'encrypted', 'security']
...
8 ['christmas', 'santa', 'gifts', 'yule', 'reindeer']
```

#### To do
- Use a metric to measure the relevance of the clusters found
- Pass custom TF-IDF and KMeans models as parameters
- Add parameters: display plots or not
- Add parameters: display logs or not

# cosine_metrics <a id="cosine_metrics"></a>
Library that computes cosine similarity on documents using different implemented methods of document representation or directly on embedding.

## Document representation methods:
1. TF-IDF 
2. Bag-of-words (CountVectorizer) 
    
## Usage:
```python
import cosine_metrics
# Load your data (docs)
```

- docs: array of string (the document to measure, your data)
- methods: "tfidf", "bow"

```python
# return the mean of the cosine similarity matrix computed on each methods
cosine_metrics.evaluate_methods(docs)
```

```python
# return the mean of the cosine similarity matrix computed on the embedding
cosine_metrics.evaluate_embeddings(embeddings)
```

```python
# return the cosine sililarity matrix computed on the specified method of document representation
cosine_metrics.compute_cosine_similarity_method(docs, method)
```


# References
- https://www.kaggle.com/code/jbencina/clustering-documents-with-tfidf-and-kmeans
- Commented with GPT
