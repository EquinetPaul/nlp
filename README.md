# NLP
NLP lib 

# TFIDF - KMeans
This library performs TF-IDF vectorization on a corpus of documents and uses determine the optimal number of clusters of KMeans. Once the optimal number of clusters has been identified, the library creates a topic representation for each cluster by selecting the top n terms that are most representative of that cluster.

# cosine_metrics
Library that computes cosine similarity on documents using different implemented methods of document representation or directly on embedding.

Methods:
1. TF-IDF 
2. Bag-of-words (CountVectorizer) 
    
Usage:
- docs: array of string (the document to measure)
- methods: "tfidf", "bow"
```python
# return the mean of the cosine similarity matrix computed on each methods
evaluate_methods(docs)
```

```python
# return the mean of the cosine similarity matrix computed on the embedding
evaluate_embeddings(embeddings)
```

```python
# return the cosine sililarity matrix computed on the specified method of document representation
compute_cosine_similarity_method(docs, method)
```
