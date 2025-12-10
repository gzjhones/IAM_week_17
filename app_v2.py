import re
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

app = Flask(__name__)
# Allow CORS for all domains
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# NLP CONFIGURATION
# ==========================================
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

print("--- LOADING BRAIN (LLM) ---")
# Adjust model path as necessary
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048, 
    n_threads=4, 
    verbose=False
)

# Global Knowledge Base and Structure Dictionary
KNOWLEDGE_BASE = []
DOC_STRUCTURE = {}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    """Normalizes text: lowercase and removes accents for search matching."""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def parse_and_index():
    """
    Parses legal documents.
    1. Splits text by Sections and Articles.
    2. Stores raw text for specific Article queries.
    3. Generates 'Rich Synthetic Summaries' for Sections to help the LLM understand context.
    """
    global KNOWLEDGE_BASE, DOC_STRUCTURE
    KNOWLEDGE_BASE = []
    DOC_STRUCTURE = {}
    
    files = glob.glob(os.path.join('docs', '*.txt'))
    print("\n--- INDEXING WITH CONTENT PREVIEWS ---")
    
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            # Basic cleanup: remove problematic backslashes and excessive newlines
            content = content.replace('\\', ' ').replace('\n', ' ')
            
            # --- PARSING LOGIC ---
            # We split the text using a Regex that looks for "Sección X" or "Artículo Y".
            # The capture group () keeps the delimiter in the result list.
            # We use a slightly looser regex for splitting, then validate inside the loop.
            split_pattern = r'(Sección\s+\d+|Artículo\s+(?:transitorio\s+)?\d+\s*[°\.])'
            tokens = re.split(split_pattern, content, flags=re.IGNORECASE)
            tokens = [t.strip() for t in tokens if t.strip()]
            
            # Default section container
            current_sec_key = "Inicio"
            current_sec_title = "Preámbulo"
            
            if current_sec_key not in DOC_STRUCTURE:
                DOC_STRUCTURE[current_sec_key] = {"title": "", "content": []}

            i = 0
            while i < len(tokens) - 1:
                header = tokens[i]    # e.g., "Artículo 14°"
                body = tokens[i+1] if (i+1) < len(tokens) else ""
                
                # Validation: Ensure the header is actually a Section or Article 
                # and not just random text.
                if not re.match(r'(Sección|Artículo)', header, re.IGNORECASE):
                    i += 1
                    continue

                full_text = f"{header}: {body}"
                
                # --- CASE A: SECTION DETECTED ---
                if "sección" in header.lower():
                    # Attempt to extract the section title from the body
                    # (Usually the first sentence after "Sección X")
                    posible_title = body[:100].split('.')[0]
                    current_sec_title = posible_title
                    
                    current_sec_key = header.lower() # Key format: "sección 12"
                    
                    if current_sec_key not in DOC_STRUCTURE:
                        DOC_STRUCTURE[current_sec_key] = {"title": current_sec_title, "content": []}
                    
                    # Store structure info in KB
                    KNOWLEDGE_BASE.append(f"ESTRUCTURA: {header} titulada '{current_sec_title}'")
                    i += 1 # Consume the body token
                
                # --- CASE B: ARTICLE DETECTED ---
                elif "artículo" in header.lower():
                    # Anti-False-Positive:
                    # If the body is too short (e.g. "ver artículo 5"), it's likely a reference, 
                    # not a definition. We skip it.
                    if len(body) < 5: 
                        i += 1
                        continue

                    # 1. Add full text to Knowledge Base (for direct retrieval)
                    KNOWLEDGE_BASE.append(full_text)
                    
                    # 2. Add a preview of this article to the current Section's structure
                    # (This allows the Section Summary to know what its articles are about)
                    if current_sec_key in DOC_STRUCTURE:
                        summary_text = f"{header} ({body[:60]}...)" 
                        DOC_STRUCTURE[current_sec_key]["content"].append(summary_text)
                    i += 1
                else:
                    i += 1

    # --- SYNTHETIC SUMMARY GENERATION ---
    # Create "Rich Summaries" so the LLM knows what each section contains without reading every article.
    for sec_key, data in DOC_STRUCTURE.items():
        articles = data["content"]
        count = len(articles)
        
        # Create a string joining the first 60 chars of all articles in this section
        context_preview = " | ".join(articles)
        
        # NOTE: This string is in SPANISH because the LLM reads it to answer users.
        synthetic = (
            f"RESUMEN DE {sec_key.upper()}: "
            f"Trata sobre '{data['title']}'. "
            f"Contiene {count} artículos. "
            f"Temas clave: {context_preview}."
        )
        KNOWLEDGE_BASE.append(synthetic)
        # Debug print
        # print(f"Indexado: {synthetic[:100]}...")

    print(f"--- SYSTEM READY: {len(KNOWLEDGE_BASE)} chunks indexed ---")

# Run indexing on startup
parse_and_index()

# ==========================================
# SEARCH LOGIC & MEMORY
# ==========================================

# Simple in-memory variable to store the last discussed topic.
# In production, use a dict keyed by session_id.
LAST_CONTEXT_TOPIC = ""

def preprocess_search(text):
    """Tokenizes and stems query for TF-IDF."""
    clean = clean_text(text)
    tokens = nltk.word_tokenize(clean, language='spanish')
    stem_tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(stem_tokens)

def search_context(query):
    global LAST_CONTEXT_TOPIC
    query_lower = query.lower()
    
    # 1. CONTEXT MEMORY CHECK
    # If user asks "what is *this* section about?", inject the previous topic.
    if "esta seccion" in query_lower or "de que trata" in query_lower:
        if LAST_CONTEXT_TOPIC and "sección" not in query_lower:
            print(f"[MEMORY] Injecting previous context: {LAST_CONTEXT_TOPIC}")
            query_lower += f" {LAST_CONTEXT_TOPIC}"

    # 2. ROUTER: SECTION LOOKUP
    # Detects "Sección X" -> Returns the Rich Synthetic Summary
    sec_match = re.search(r'secci[oó]n\s+(\d+)', query_lower)
    if sec_match:
        target = f"sección {sec_match.group(1)}"
        LAST_CONTEXT_TOPIC = target # Update memory
        print(f"[ROUTER] Fetching summary for: {target}")
        
        for doc in KNOWLEDGE_BASE:
            if f"RESUMEN DE {target.upper()}" in doc:
                return doc

    # 3. ROUTER: ARTICLE LOOKUP
    # Detects "Artículo X" -> Returns the specific text
    art_match = re.search(r'art[ií]culo\s+(\d+)', query_lower)
    if art_match:
        target_num = art_match.group(1)
        # Strict Regex to match "Artículo X" at the beginning of the chunk
        regex = re.compile(r'artículo\s+(?:transitorio\s+)?' + re.escape(target_num) + r'[°\.]', re.IGNORECASE)
        for doc in KNOWLEDGE_BASE:
            # Ignore summaries, we want the full law text
            if "RESUMEN" not in doc and regex.match(doc): 
                return doc

    # 4. FALLBACK: TF-IDF VECTOR SEARCH
    # For conceptual questions like "requisitos presidente"
    vectorizer = TfidfVectorizer(preprocessor=preprocess_search)
    try:
        corpus = KNOWLEDGE_BASE + [query_lower]
        tfidf = vectorizer.fit_transform(corpus)
        cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = np.argmax(cosine)
        if cosine[0][idx] > 0.05:
            return KNOWLEDGE_BASE[idx]
    except Exception as e:
        print(f"Vector search error: {e}")
        
    return None

# ==========================================
# API ENDPOINT
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    context = search_context(user_msg)
    
    # Construct System Prompt (Keep in Spanish for consistency)
    if context:
        sys_msg = f"""Eres un experto en la Constitución de 1843.
        
        CONTEXTO RECUPERADO: 
        {context}
        
        INSTRUCCIONES:
        1. Si el contexto es un RESUMEN, usa los 'Temas clave' para explicar de qué trata la sección.
        2. Si te preguntan por un número de artículos, responde con el número exacto del resumen.
        3. Cita textualmente si se requiere.
        """
    else:
        sys_msg = "No encontraste información relevante en la base de datos. Pide al usuario especificar Sección o Artículo."

    # Prompt Template (ChatML format)
    prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=400, stop=["<|im_end|>"], echo=False, temperature=0.1)
        response = output['choices'][0]['text'].strip()
    except Exception as e:
        response = "Error interno del servidor."
        print(e)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)