# Compute cosine similarity using different document representation
## Available document representation: TF-IDF, Bag-of-words (CountVectorizer)
## To implement? Word2Vec, GloVe

#!pip install scikit-learn

methods = ["bow", "tfidf"]

# Compute cosine similarity
def compute_cosine_similarity(doc_representation):
    similarity_matrix = cosine_similarity(doc_representation)
    return similarity_matrix

def compute_cosine_similarity_method(docs, method="tfidf"):
  
    if method=="bow":
        vectorizer = CountVectorizer()
    if method=="tfidf":
        vectorizer = TfidfVectorizer()

    doc_representation = vectorizer.fit_transform(docs)
    
    # Calculer la similarité cosinus entre chaque paire de documents en utilisant la methode spécifiée
    similarity_matrix = compute_cosine_similarity(doc_representation)

    return similarity_matrix

def evaluate_methods(docs):
    
    results = {}
    
    for method in methods:
        results[method] = compute_cosine_similarity_method(docs, method).mean()
        
    return results

def evaluate_embeddings(embeddings):
  
    similarity_matrix = compute_cosine_similarity(embeddings)
 
    return results.mean()
