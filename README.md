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
- Plot used to determine the optimal number of clusters (based on the evolution of the inertia)
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
    n_top_words = 5, # Used to show the n_top_words of each clusters for topic representation
    display_plots = True
    )
```

```bash
0 ['windows', 'drive', 'dos', 'file', 'os']
1 ['space','launch','orbit','lunar','planet']
2 ['key', 'encryption', 'keys', 'encrypted', 'security']
...
8 ['christmas', 'santa', 'gifts', 'yule', 'reindeer']
```

### Custom TF-IDF vectorizer
```python
import tfidf_kmeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
data = ...
stop_words = ...

# Define TF-IDF
tfidf_parameters = {
    'ngram_range' : (1, 2),
    'stop_words' : stop_words,
    'lowercase': True,
    'min_df': 1,
    'max_df': 0.8,
}
vectorizer = TfidfVectorizer(**tfidf_parameters)

# Apply Topic Modeling
tfidf_kmeans.compute(
    data,
    tfidf_vectorizer = vectorizer
    )
```

### Custom KMeans parameters
```python
import tfidf_kmeans
import pandas as pd

# Load data
data = ...

# Define K-Means parameters
kmeans_parameters = {
    'init': 'k-means++', 
    'n_init': 100, 
    'max_iter': 10, 
    'tol': 0.001
 }

# Apply Topic Modeling
tfidf_kmeans.compute(
    data,
    kmeans_parameters = kmeans_parameters
    )
```

#### To do
- Use a metric to measure the relevance of the clusters found
- Pass custom TF-IDF and KMeans models as parameters (✅)
- Add parameter: display plots or not (✅)
- Add parameter: display logs or not
- Upgrade Clusters Plot (using plotly?)
- Change the legend position for TSNE Cluster Plot
- Add parameter: apply a simple clean to data

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
