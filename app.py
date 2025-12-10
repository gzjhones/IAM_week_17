# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import glob
import unicodedata
from rapidfuzz import process, fuzz

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# NLTK Configuration
nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

# ==========================================
# 1. LEXICON AND MODEL CONFIGURATION
# ==========================================
KEYWORD_MAP = {
    "donde": "direccion",
    "ubicacion": "direccion",
    "lugar": "direccion",
    "calle": "direccion",
    "queda": "direccion",
    "costo": "precio",
    "valor": "precio",
    "pagar": "precio",
    "tienes": "menu",
    "platos": "recetas",
    "comida": "recetas",
    "carta": "recetas",
}

print("--- LOADING BRAIN ---")
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

# ==========================================
# 2. DOCUMENT MANAGEMENT
# ==========================================
KNOWLEDGE_BASE = []
RECIPE_LIST = []

def clean_text(text):
    """Normalize text: lowercase and remove accents"""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def load_documents():
    """Load and index all documents from docs/ directory"""
    global KNOWLEDGE_BASE, RECIPE_LIST
    KNOWLEDGE_BASE = []
    RECIPE_LIST = []
    
    files = glob.glob(os.path.join('docs', '*.txt'))
    print("\n--- INDEXING DOCUMENTS ---")
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split('\n\n')
            for chunk in chunks:
                if len(chunk.strip()) > 10:
                    KNOWLEDGE_BASE.append(chunk.strip())
                    lines = chunk.split('\n')
                    for line in lines:
                        if "TITULO:" in line.upper():
                            title = line.split(":")[1].strip()
                            RECIPE_LIST.append(title)
                            
    print(f"--- SYSTEM READY: {len(KNOWLEDGE_BASE)} chunks | {len(RECIPE_LIST)} recipes ---")

load_documents()

# ==========================================
# 3. CORRECTION LOGIC (RAPIDFUZZ)
# ==========================================
def inject_corrections(query):
    """
    Uses Fuzzy Logic to fix typos before search.
    Example: 'paeya' -> detects 'Paella Valenciana' -> injects 'Paella'
    """
    clean_query = clean_text(query)
    words = clean_query.split()
    expanded_words = words.copy()
    
    # A. Correct Intentions (Keyword Map)
    # If user writes 'direksion', search in map keys (donde, ubicacion, etc)
    # and if not found, search in values (direccion, precio).
    vocabulary = list(KEYWORD_MAP.keys()) + list(set(KEYWORD_MAP.values()))
    
    for word in words:
        # Check if word resembles something from our key vocabulary
        match = process.extractOne(word, vocabulary, scorer=fuzz.ratio, score_cutoff=85)
        if match:
            corrected_word = match[0]
            # If correction is a key (e.g., "ubicacion"), get its canonical value ("direccion")
            final_term = KEYWORD_MAP.get(corrected_word, corrected_word)
            if final_term not in expanded_words:
                expanded_words.append(final_term)
                print(f"[RapidFuzz] Typo detected: '{word}' -> Corrected to: '{final_term}'")

    # B. Correct Recipe Names (Titles)
    # Use partial_token_sort_ratio so "receta paeya" finds "PAELLA VALENCIANA"
    if RECIPE_LIST:
        title_match = process.extractOne(
            clean_query, 
            [clean_text(t) for t in RECIPE_LIST], 
            scorer=fuzz.partial_token_sort_ratio, 
            score_cutoff=80
        )
        
        if title_match:
            # title_match[0] is the cleaned title text that matched
            # Add it to search to ensure TF-IDF finds it
            print(f"[RapidFuzz] Possible recipe detected: '{title_match[0]}' ({title_match[1]:.1f}%)")
            expanded_words.append(title_match[0])

    return " ".join(expanded_words)

# ==========================================
# 4. SEARCH ENGINE
# ==========================================
def preprocess_search(text):
    """Preprocess text for TF-IDF: clean, tokenize, stem, remove stopwords"""
    # Step 1: Basic cleaning
    clean = clean_text(text)
    # Step 2: Tokenization
    tokens = nltk.word_tokenize(clean, language='spanish')
    stem_tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(stem_tokens)

def search_context(query):
    """
    Search for relevant context in knowledge base using TF-IDF + RapidFuzz correction
    """
    # 1. APPLY RAPIDFUZZ HERE
    # This transforms "como se ase la paeya" into "como se ase la paeya paella valenciana"
    optimized_query = inject_corrections(query)
    
    clean_corpus = [clean_text(doc) for doc in KNOWLEDGE_BASE]
    # Use optimized query for vectorization
    corpus_to_vectorize = clean_corpus + [optimized_query]
    
    vectorizer = TfidfVectorizer(preprocessor=preprocess_search)
    try:
        tfidf = vectorizer.fit_transform(corpus_to_vectorize)
        cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = np.argmax(cosine)
        score = cosine[0][idx]
        
        print(f"[TF-IDF] Original Query: '{query}' | Optimized: '{optimized_query}'")
        print(f"[TF-IDF] Match Index: {idx} | Score: {score:.4f}")
        
        if score > 0.1:  # Slightly higher threshold since we're helping the engine
            return KNOWLEDGE_BASE[idx]
    except Exception as e:
        print(f"Error in vectorized search: {e}")
        
    return None

# ==========================================
# 5. CHAT ENDPOINT
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    context = search_context(user_msg)
    
    # MODIFICACIÓN AQUÍ: Instrucciones estrictas de formato Markdown
    if context:
        sys_msg = f"""Eres un Chef experto y asistente culinario amable.
        
        INFORMACIÓN RECUPERADA: 
        {context}
        
        INSTRUCCIONES DE FORMATO (MANDATORIO):
        1. Usa '### ' para los títulos de recetas.
        2. Usa '**' para negritas (ej: **Ingredientes:**).
        3. Para los ingredientes, usa SIEMPRE una lista con guiones '- '.
        4. Para la preparación, usa SIEMPRE una lista numerada '1. '.
        5. No uses bloques de código, solo texto plano con formato Markdown.
        
        Ejemplo de salida esperada:
        ### Nombre de la Receta
        **Ingredientes:**
        - 100g de harina
        - 2 huevos
        
        **Preparación:**
        1. Batir los huevos.
        2. Mezclar con harina.
        """
    else:
        sys_msg = """No encontraste información exacta. Indica cortésmente que solo tienes información sobre: Paella, Mojito, Tarta de Manzana y Guacamole."""

    prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=300, stop=["<|im_end|>"], echo=False, temperature=0.3)
        response = output['choices'][0]['text'].strip()
    except Exception as e:
        response = "Error técnico en el cerebro local."
        print(f"LLM Error: {e}")

    return jsonify({"response": response})

if __name__ == '__main__':
    print("--- LLM Server Started ---")
    app.run(port=5000, debug=True)