from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import sys

# Charger le modèle d'embedding
def load_embedding_model(model_name='multi-qa-MiniLM-L6-cos-v1'):
    return SentenceTransformer(model_name)

# Charger et lire tous les fichiers .txt dans un répertoire donné
def load_text_files(directory_path):
    use_cases = []
    use_case_files = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                use_cases.append(content)
                use_case_files.append(filename)
    
    return use_cases, use_case_files

# Créer l'index FAISS avec les embeddings générés
def create_faiss_index(embedding_model, texts):
    # Générer un embedding d'exemple pour définir la dimension
    sample_embedding = embedding_model.encode(texts[0])
    dimension = sample_embedding.shape[0]
    print("Dimension des embeddings :", dimension)
    
    # Créer l'index FAISS
    index = faiss.IndexFlatL2(dimension)
    
    # Générer les embeddings pour tous les textes et les ajouter à l'index
    embeddings = np.array([embedding_model.encode(text) for text in texts]).astype("float32")
    index.add(embeddings)
    
    return index

# Exécuter toutes les étapes
def setup_rag_pipeline(directory_path="Multimedia/Use_Cases", model_name='multi-qa-MiniLM-L6-cos-v1'):
    # Charger le modèle d'embedding
    embedding_model = load_embedding_model(model_name)
    
    # Charger les textes des fichiers
    use_cases, use_case_files = load_text_files(directory_path)
    
    # Créer l'index FAISS avec les embeddings
    index = create_faiss_index(embedding_model, use_cases)
    
    return embedding_model, use_cases, use_case_files, index

# Fonction de recherche dans l'index
def search_in_index(prompt, index, texts, embeddings):
    prompt_embedding = embedding_model.encode(prompt).astype("float32")
    _, indices = index.search(prompt_embedding.reshape(1, -1), k=5)
    return [texts[i] for i in indices[0]]

# Exécution principale
if __name__ == "__main__":
    
    embedding_model, use_cases, use_case_files, index = setup_rag_pipeline()
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Récupère la question depuis les arguments
 
        question_embedding = embedding_model.encode(prompt).astype("float32")

        # Recherche dans l'index FAISS
        _, indices = index.search(question_embedding.reshape(1, -1), k=3)
        relevant_texts = [use_cases[i] for i in indices[0]]
        
        # Préparer le contexte pour la réponse GPT
        context = " ".join(relevant_texts)

        print("Réponse :", context)
