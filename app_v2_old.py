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
from rapidfuzz import process, fuzz

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
# Asegúrate de descomentar esto la primera vez si no tienes los datos
# nltk.download('stopwords'); nltk.download('punkt')

stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

# Mapa de términos legales para mejorar la búsqueda
KEYWORD_MAP = {
    "norma": "ley",
    "regla": "ley",
    "carta magna": "constitucion",
    "derechos": "garantias",
    "obligaciones": "deberes",
    "presidente": "ejecutivo",
    "congreso": "legislativo",
    "asamblea": "legislativo",
    "senado": "legislativo",
    "diputados": "legislativo",
    "jueces": "judicial",
    "cortes": "judicial",
    "religion": "culto",
    "pais": "nacion",
    "territorio": "departamentos",
    "votacion": "elecciones"
}

print("--- CARGANDO CEREBRO (LLM) ---")
# Ajusta la ruta a tu modelo local si es diferente
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,  # Contexto aumentado para documentos legales
    n_threads=4,
    verbose=False
)

# ==========================================
# 1. GESTIÓN DE DOCUMENTOS (PARSER LEGAL)
# ==========================================
KNOWLEDGE_BASE = []
ARTICLE_LIST = [] 

def clean_text(text):
    """Normaliza texto: minúsculas y quita acentos para búsqueda interna"""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def load_documents():
    """Carga, limpia y segmenta la Constitución"""
    global KNOWLEDGE_BASE, ARTICLE_LIST
    KNOWLEDGE_BASE = []
    ARTICLE_LIST = []
    
    files = glob.glob(os.path.join('docs', '*.txt'))
    print("\n--- INDEXANDO CONSTITUCIÓN 1843 ---")
    
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # [CORRECCIÓN 1] Limpieza de metadatos del archivo específico
            # Elimina , , etc.
            # content = re.sub(r'<[A-Za-z_]+>', '', content)
            content = re.sub('\\', '', content)
            
            # Normalizar saltos de línea excesivos
            content = re.sub(r'\n+', ' ', content)
            
            # [CORRECCIÓN 2] Segmentación por Artículo
            # Usamos Regex Lookahead para dividir justo antes de "Artículo X"
            # Esto mantiene el título unido a su contenido.
            chunks = re.split(r'(?=Artículo \d+°)', content)
            
            for chunk in chunks:
                clean_chunk = chunk.strip()
                # Filtramos fragmentos vacíos o muy cortos (índices sueltos)
                if len(clean_chunk) > 20:
                    KNOWLEDGE_BASE.append(clean_chunk)
                    
                    # Extraer el título exacto para el índice de corrección
                    match = re.search(r'(Artículo \d+°)', clean_chunk)
                    if match:
                        ARTICLE_LIST.append(match.group(1))
                            
    print(f"--- SISTEMA LISTO: {len(KNOWLEDGE_BASE)} fragmentos | {len(ARTICLE_LIST)} artículos detectados ---")

load_documents()

# ==========================================
# 2. LÓGICA DE CORRECCIÓN (RAPIDFUZZ)
# ==========================================
def inject_corrections(query):
    """
    Corrige typos usando el índice de artículos y palabras clave.
    Ej: 'artikulo sinco' -> 'Artículo 5°'
    """
    clean_query = clean_text(query)
    words = clean_query.split()
    expanded_words = words.copy()
    
    # A. Corregir Intenciones (Sinónimos)
    vocabulary = list(KEYWORD_MAP.keys()) + list(set(KEYWORD_MAP.values()))
    for word in words:
        match = process.extractOne(word, vocabulary, scorer=fuzz.ratio, score_cutoff=85)
        if match:
            corrected_word = match[0]
            final_term = KEYWORD_MAP.get(corrected_word, corrected_word)
            if final_term not in expanded_words:
                expanded_words.append(final_term)

    # B. Corregir Referencias a Artículos
    if ARTICLE_LIST:
        title_match = process.extractOne(
            clean_query, 
            [clean_text(t) for t in ARTICLE_LIST], 
            scorer=fuzz.partial_token_sort_ratio, 
            score_cutoff=85
        )
        if title_match:
            # Inyectamos el título limpio encontrado para ayudar al TF-IDF
            expanded_words.append(title_match[0])

    return " ".join(expanded_words)

# ==========================================
# 3. MOTOR DE BÚSQUEDA HÍBRIDO
# ==========================================
def preprocess_search(text):
    clean = clean_text(text)
    tokens = nltk.word_tokenize(clean, language='spanish')
    stem_tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(stem_tokens)

def search_context(query):
    if not KNOWLEDGE_BASE:
        return None

    # [CORRECCIÓN 3] Búsqueda Determinista (Prioridad Alta)
    # Si el usuario pide un número específico, lo buscamos literalmente.
    number_match = re.search(r'art[ií]culo\s+(\d+)', query.lower())
    if number_match:
        target_num = number_match.group(1)
        target_str = f"Artículo {target_num}°" # Formato exacto del doc (con °)
        
        for doc in KNOWLEDGE_BASE:
            # Buscamos que el documento EMPIECE o contenga el título exacto
            if target_str in doc:
                print(f"[MATCH DIRECTO] Encontrado: {target_str}")
                return doc

    # Búsqueda Vectorial (TF-IDF) para preguntas conceptuales
    optimized_query = inject_corrections(query)
    
    clean_corpus = [clean_text(doc) for doc in KNOWLEDGE_BASE]
    corpus_to_vectorize = clean_corpus + [optimized_query]
    
    vectorizer = TfidfVectorizer(preprocessor=preprocess_search)
    try:
        tfidf = vectorizer.fit_transform(corpus_to_vectorize)
        cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = np.argmax(cosine)
        score = cosine[0][idx]
        
        print(f"[TF-IDF] Query: '{optimized_query}' | Score: {score:.4f}")
        
        if score > 0.05: # Umbral bajo para permitir coincidencias de contexto
            return KNOWLEDGE_BASE[idx]
            
    except Exception as e:
        print(f"Error búsqueda vectorial: {e}")
        
    return None

# ==========================================
# 4. CHAT ENDPOINT
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    context = search_context(user_msg)
    
    if context:
        system_instruction = f"""Eres un experto en Historia y Derecho Constitucional de Bolivia (1843).
        Responde a la pregunta del usuario basándote EXCLUSIVAMENTE en el siguiente fragmento de la Constitución.
        
        CONTEXTO OFICIAL:
        {context}
        
        INSTRUCCIONES:
        1. Cita el número del artículo si está disponible (ej: **Artículo 5°**).
        2. Si citas texto literal, usa bloque de cita (>).
        3. Si la respuesta no está en el contexto, di "No aparece en este artículo".
        """
    else:
        # Sugerencias dinámicas si no encuentra nada
        preview = ", ".join(ARTICLE_LIST[:5])
        system_instruction = f"""Eres un asistente histórico.
        No encontraste información exacta en la base de datos para la consulta.
        Dile al usuario que intente preguntar por un número de artículo específico.
        Ejemplos disponibles: {preview}...
        """

    prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=500, stop=["<|im_end|>"], echo=False, temperature=0.2)
        response = output['choices'][0]['text'].strip()
    except Exception as e:
        response = "Error interno en el modelo."
        print(f"LLM Error: {e}")

    return jsonify({"response": response})

if __name__ == '__main__':
    print("--- SERVIDOR DE HISTORIA LEGAL INICIADO ---")
    app.run(port=5000, debug=True)