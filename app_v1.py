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
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# CONFIGURACIÓN NLP
# ==========================================
# Asegurarse de tener las dependencias de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

print("--- CARGANDO CEREBRO CULINARIO (LLM) ---")
# Ajusta la ruta a tu modelo local
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048, 
    n_threads=4, 
    verbose=False
)

# Base de Conocimiento Global
KNOWLEDGE_BASE = []
RECIPE_TITLES = [] # Para búsqueda rápida por nombre exacto

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def clean_text(text):
    """Normaliza texto: minúsculas y sin acentos para búsqueda."""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

def parse_and_index():
    """
    Parsea el archivo de recetas.
    Separa bloques por 'TITULO:', 'POLITICA...' o 'DIRECCIÓN:'.
    """
    global KNOWLEDGE_BASE, RECIPE_TITLES
    KNOWLEDGE_BASE = []
    RECIPE_TITLES = []
    
    files = glob.glob(os.path.join('docs', '*.txt'))
    print("\n--- INDEXANDO RECETARIO ---")
    
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Usamos Regex para dividir el texto basándonos en las cabeceras clave
            # Capturamos el delimitador para saber qué tipo de bloque es
            split_pattern = r'(TITULO:|POLITICA DE ENVIOS:|DIRECCIÓN:)'
            tokens = re.split(split_pattern, content)
            
            # La lista tokens tendrá: ["texto irrelevante", "TITULO:", "PAELLA...", "TITULO:", "MOJITO..."]
            
            i = 1 # Empezamos en 1 porque el 0 suele ser vacío o preámbulo
            while i < len(tokens) - 1:
                header = tokens[i].strip() # Ej: TITULO:
                body = tokens[i+1].strip() # El contenido del bloque
                
                full_chunk = f"{header} {body}"
                
                # Caso A: Es una Receta
                if "TITULO" in header:
                    # Extraer el nombre de la receta para búsquedas directas
                    # El nombre suele ser la primera línea del body
                    lines = body.split('\n')
                    recipe_name = lines[0].strip()
                    
                    KNOWLEDGE_BASE.append(full_chunk)
                    RECIPE_TITLES.append(recipe_name)
                    print(f"-> Receta Indexada: {recipe_name}")
                
                # Caso B: Información General (Envíos o Dirección)
                else:
                    KNOWLEDGE_BASE.append(full_chunk)
                    print(f"-> Info Indexada: {header}")
                
                i += 2 # Avanzamos al siguiente par Header-Body

    print(f"--- SYSTEM READY: {len(KNOWLEDGE_BASE)} chunks indexed ---")

# Ejecutar indexación al inicio
parse_and_index()

# ==========================================
# LÓGICA DE BÚSQUEDA & MEMORIA
# ==========================================

# Variable simple para recordar la última receta de la que se habló
LAST_RECIPE_CONTEXT = ""

def preprocess_search(text):
    """Tokeniza y extrae raíces para TF-IDF."""
    clean = clean_text(text)
    tokens = nltk.word_tokenize(clean, language='spanish')
    stem_tokens = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(stem_tokens)

def search_context(query):
    global LAST_RECIPE_CONTEXT
    query_lower = clean_text(query)
    
    # 1. CHEQUEO DE MEMORIA CONTEXTUAL
    # Si el usuario pregunta "cómo se prepara?" o "qué ingredientes lleva?",
    # asumimos que habla de la última receta.
    if ("prepara" in query_lower or "ingredientes" in query_lower or "tiempo" in query_lower) and LAST_RECIPE_CONTEXT:
        # Si no menciona explícitamente otra receta, inyectamos la anterior
        is_new_recipe = any(clean_text(t) in query_lower for t in RECIPE_TITLES)
        if not is_new_recipe:
            print(f"[MEMORY] Inyectando contexto previo: {LAST_RECIPE_CONTEXT}")
            query_lower += f" {clean_text(LAST_RECIPE_CONTEXT)}"

    # 2. ROUTER: BÚSQUEDA DIRECTA POR NOMBRE DE RECETA
    # Si la query contiene "mojito", buscamos el chunk del Mojito directamente.
    for title in RECIPE_TITLES:
        if clean_text(title) in query_lower:
            # Buscar el chunk que contiene este título
            for doc in KNOWLEDGE_BASE:
                if f"TITULO: {title}" in doc:
                    LAST_RECIPE_CONTEXT = title # Actualizar memoria
                    return doc

    # 3. ROUTER: POLÍTICAS Y DIRECCIÓN
    if "envio" in query_lower or "costo" in query_lower or "domicilio" in query_lower:
        for doc in KNOWLEDGE_BASE:
            if "POLITICA DE ENVIOS" in doc:
                return doc
    
    if "donde" in query_lower or "ubicacion" in query_lower or "calle" in query_lower or "direccion" in query_lower:
        for doc in KNOWLEDGE_BASE:
            if "DIRECCIÓN" in doc:
                return doc

    # 4. FALLBACK: BÚSQUEDA SEMÁNTICA (TF-IDF)
    # Útil para: "tienes algo con pollo?", "quiero un postre", "receta vegana"
    vectorizer = TfidfVectorizer(preprocessor=preprocess_search)
    try:
        corpus = KNOWLEDGE_BASE + [query_lower]
        tfidf = vectorizer.fit_transform(corpus)
        # Calcular similitud del coseno entre la query (último item) y los documentos
        cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = np.argmax(cosine)
        
        # Umbral de confianza
        if cosine[0][idx] > 0.05:
            found_doc = KNOWLEDGE_BASE[idx]
            # Si encontramos una receta por semántica, actualizamos la memoria
            if "TITULO:" in found_doc:
                # Extraer título sucio para memoria
                lines = found_doc.split('\n')
                # lines[0] sería "TITULO: PAELLA..."
                LAST_RECIPE_CONTEXT = lines[0].replace("TITULO:", "").strip()
            
            return found_doc
    except Exception as e:
        print(f"Error en búsqueda vectorial: {e}")
        
    return None

# ==========================================
# ENDPOINT API
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    context = search_context(user_msg)
    
    # Prompt del Sistema Adaptado al Rol de Chef
    if context:
        sys_msg = f"""Eres un Chef experto y asistente culinario amable.
        
        INFORMACIÓN RECUPERADA DEL RECETARIO: 
        {context}
        
        INSTRUCCIONES:
        1. Responde basándote EXCLUSIVAMENTE en la información recuperada.
        2. Si das una receta, usa formato claro (Lista de Ingredientes con viñetas, Pasos numerados).
        3. Si te preguntan por envíos, menciona los costos y tiempos de la política.
        4. Sé conciso y apetitoso en tu lenguaje.
        """
    else:
        sys_msg = """No encontraste información exacta en el recetario. 
        Indica cortésmente que solo tienes información sobre: Paella, Mojito, Tarta de Manzana y Guacamole, o políticas de envío.
        """

    # Formato ChatML
    prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        # Generación
        output = LLM(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False, temperature=0.2)
        response = output['choices'][0]['text'].strip()
    except Exception as e:
        response = "Lo siento, tuve un problema en la cocina (Error del servidor)."
        print(e)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)