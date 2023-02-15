# NLP
NLP lib 


### cosine_metrics
Library that allows computation of cosine similarity on documents using different implemented methods of document representation or directly on embedding.

Methods:
1. TF-IDF 
2.Bag-of-words (CountVectorizer) 
    
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
