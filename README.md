# NLP
NLP lib 


### cosine_metrics
Library that computes cosine similarity on documents using different implemented methods of document representation or directly on embedding.

Methods:
1. TF-IDF 
2. Bag-of-words (CountVectorizer) 
    
Usage:
- docs: array of string (the document to measure)
- methods: "tfidf", "bow"
```python
evaluate_methods(docs)
# return the mean of the cosine similarity matrix computed on each methods
```

```python
evaluate_embeddings(embeddings)
# return the mean of the cosine similarity matrix computed on the embedding
```

```python
compute_cosine_similarity_method(docs, method)
# return the cosine sililarity matrix computed on the specified method of document representation
```
