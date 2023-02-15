# Compute cosine similarity using different document representation
## Available document representation: TF-IDF, Bag-of-words (CountVectorizer)
## To implement? Word2Vec, GloVe

#!pip install scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(docs, method="tfidf"):
  
    if method=="bow":
        vectorizer = CountVectorizer()
    if method=="tfidf":
        vectorizer = TfidfVectorizer()

    doc_representation = vectorizer.fit_transform(docs)
    
    # Calculer la similarité cosinus entre chaque paire de documents en utilisant la methode spécifiée
    similarity_matrix = cosine_similarity(doc_representation)

    return similarity_matrix
