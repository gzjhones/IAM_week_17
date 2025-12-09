# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import glob

# ==========================================
# 0. Configuration & Setup
# ==========================================
# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global Knowledge Base
KNOWLEDGE_BASE = []  # Formerly BASE_CONOCIMIENTO
SOURCE_FILES = []    # To track where the text came from

# NLP Tools
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

# ==========================================
# 1. Document Loading Logic
# ==========================================
def load_documents():
    """Loads all .txt files from the ./docs directory into memory."""
    global KNOWLEDGE_BASE, SOURCE_FILES
    
    # Reset lists to avoid duplicates on reload
    KNOWLEDGE_BASE = []
    SOURCE_FILES = []
    
    # Path compliant with Windows/Linux
    path = os.path.join('.', 'docs', '*.txt')
    files = glob.glob(path)
    
    print(f"--- SYSTEM: Found {len(files)} files in {path} ---")

    for filename in files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newline to create chunks
                chunks = content.split('\n\n')
                file_chunk_count = 0
                for chunk in chunks:
                    if len(chunk.strip()) > 20: # Ignore tiny chunks
                        KNOWLEDGE_BASE.append(chunk.strip())
                        SOURCE_FILES.append(os.path.basename(filename))
                        file_chunk_count += 1
                print(f" > Loaded: {os.path.basename(filename)} ({file_chunk_count} chunks)")
        except Exception as e:
            print(f" ! Error loading {filename}: {e}")

    print(f"--- SYSTEM: Total Knowledge Base Size: {len(KNOWLEDGE_BASE)} chunks ---")

# Load on startup
load_documents()

# ==========================================
# 2. NLP Engine (No LLM)
# ==========================================
def preprocess_text(text):
    """Clean and stem text for better matching."""
    tokens = nltk.word_tokenize(text.lower(), language='spanish')
    # Filter non-alphanumeric and stopwords
    clean_tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(clean_tokens)

def expand_query_wordnet(query):
    """Basic synonym expansion using WordNet."""
    words = query.split()
    expansion = set(words)
    for word in words:
        synonyms = wordnet.synsets(word, lang='spa')
        for syn in synonyms:
            for lemma in syn.lemmas('spa'):
                expansion.add(lemma.name().replace('_', ' '))
    return " ".join(list(expansion))

def analyze_sentiment(text):
    """Heuristic sentiment analysis."""
    pos_words = ["gracias", "excelente", "bueno", "ayuda", "genial", "perfecto"]
    neg_words = ["malo", "lento", "error", "queja", "horrible", "no funciona", "problema"]
    
    score = 0
    for p in text.lower().split():
        if p in pos_words: score += 1
        if p in neg_words: score -= 1
        
    if score > 0: return "positivo"
    if score < 0: return "negativo"
    return "neutral"

# ==========================================
# 3. API Routes
# ==========================================

# Endpoint to verify loaded docs (Debug)
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "documents_loaded": len(KNOWLEDGE_BASE),
        "sources": list(set(SOURCE_FILES))
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    current_context = data.get('context', {})
    
    print(f"\n[User Query]: {user_message}")
    
    # 1. Sentiment Analysis
    sentiment = analyze_sentiment(user_message)
    
    # 2. Intent Detection (Hardcoded Logic for Flows)
    # Note: To make this purely generic, you would remove this block or 
    # load these rules from a json file.
    response_text = ""
    new_context = current_context
    
    # Example Flow: "Purchase/Comprar"
    if "comprar" in user_message.lower() or current_context.get('state') == 'buying':
        if current_context.get('state') != 'buying':
            response_text = "Entendido, te guiaré en la compra. ¿Qué producto buscas?"
            new_context = {'state': 'buying', 'step': 1}
        else:
            response_text = "Perfecto, he anotado tu pedido. ¿Necesitas algo más?"
            new_context = {} 
            
        return jsonify({
            "response": response_text, 
            "context": new_context,
            "sentiment": sentiment
        })

    # 3. RAG Retrieval (TF-IDF)
    # Expand query for synonyms
    expanded_query = expand_query_wordnet(user_message)
    
    # Vectorize content + query
    # We add the query at the end of the corpus to compare it against the rest
    corpus = KNOWLEDGE_BASE + [expanded_query]
    
    try:
        vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate Cosine Similarity between Query (last item) and all Docs
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # Find best match
        best_match_idx = np.argmax(cosine_sim)
        similarity_score = cosine_sim[0][best_match_idx]
        
        print(f"[Match Info] Best Match Index: {best_match_idx} | Score: {similarity_score:.4f}")
        
        # Threshold (Adjust based on testing)
        if similarity_score > 0.1: 
            best_doc = KNOWLEDGE_BASE[best_match_idx]
            
            # Dynamic Intro based on Sentiment
            intro = ""
            if sentiment == "negativo":
                intro = "Lamento que tengas inconvenientes. Encontré esto que podría ser útil: "
            elif sentiment == "positivo":
                intro = "¡Genial! Aquí tienes la información relacionada: "
                
            response_text = f"{intro}\n\n{best_doc}"
        else:
            print("[Match Info] Score too low, returning fallback.")
            response_text = "Lo siento, no encontré información específica en mis documentos sobre eso. ¿Podrías reformular?"
            
    except Exception as e:
        print(f"[Error] TF-IDF processing failed: {e}")
        response_text = "Ocurrió un error procesando tu solicitud."

    return jsonify({
        "response": response_text, 
        "context": new_context,
        "sentiment": sentiment
    })

if __name__ == '__main__':
    print("--- STARTING LOCAL RAG AGENT ---")
    app.run(port=5000, debug=True)