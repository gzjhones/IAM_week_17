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

# Configuración NLTK
nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

# ==========================================
# 1. CONFIGURACIÓN DEL LEXICÓN Y MODELO
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
    "carta": "recetas"
}

print("--- CARGANDO CEREBRO ---")
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

# ==========================================
# 2. GESTIÓN DE DOCUMENTOS
# ==========================================
BASE_CONOCIMIENTO = []
LISTA_TITULOS = []

def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto

def cargar_documentos():
    global BASE_CONOCIMIENTO, LISTA_TITULOS
    BASE_CONOCIMIENTO = []
    LISTA_TITULOS = []
    
    files = glob.glob(os.path.join('docs', '*.txt'))
    print("\n--- INDEXANDO DOCUMENTOS ---")
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split('\n\n')
            for chunk in chunks:
                if len(chunk.strip()) > 10:
                    BASE_CONOCIMIENTO.append(chunk.strip())
                    lines = chunk.split('\n')
                    for line in lines:
                        if "TITULO:" in line.upper():
                            titulo = line.split(":")[1].strip()
                            LISTA_TITULOS.append(titulo)
                            
    print(f"--- SISTEMA LISTO: {len(BASE_CONOCIMIENTO)} bloques | {len(LISTA_TITULOS)} recetas ---")

cargar_documentos()

# ==========================================
# 3. LÓGICA DE CORRECCIÓN (RAPIDFUZZ)
# ==========================================
def inyectar_correcciones(consulta):
    """
    Usa Lógica Difusa para arreglar typos antes de la búsqueda.
    Ejemplo: 'paeya' -> detecta 'Paella Valenciana' -> inyecta 'Paella'
    """
    consulta_limpia = limpiar_texto(consulta)
    palabras = consulta_limpia.split()
    palabras_expandidas = words = palabras.copy()
    
    # A. Corregir Intenciones (Keyword Map)
    # Si escribe 'direksion', busca en las claves del mapa (donde, ubicacion, etc)
    # y si no encuentra, busca en los valores (direccion, precio).
    vocabularios = list(KEYWORD_MAP.keys()) + list(set(KEYWORD_MAP.values()))
    
    for palabra in palabras:
        # Busca si la palabra se parece a algo de nuestro vocabulario clave
        match = process.extractOne(palabra, vocabularios, scorer=fuzz.ratio, score_cutoff=85)
        if match:
            palabra_corregida = match[0]
            # Si la corrección es una clave (ej: "ubicacion"), obtenemos su valor canónico ("direccion")
            termino_final = KEYWORD_MAP.get(palabra_corregida, palabra_corregida)
            if termino_final not in palabras_expandidas:
                palabras_expandidas.append(termino_final)
                print(f"[RapidFuzz] Typo detectado: '{palabra}' -> Corregido a: '{termino_final}'")

    # B. Corregir Nombres de Recetas (Títulos)
    # Usamos partial_token_sort_ratio para que "receta paeya" encuentre "PAELLA VALENCIANA"
    if LISTA_TITULOS:
        match_titulo = process.extractOne(
            consulta_limpia, 
            [limpiar_texto(t) for t in LISTA_TITULOS], 
            scorer=fuzz.partial_token_sort_ratio, 
            score_cutoff=80
        )
        
        if match_titulo:
            # match_titulo[0] es el texto del título limpio que hizo match
            # Lo agregamos a la búsqueda para asegurar que TF-IDF lo encuentre
            print(f"[RapidFuzz] Posible receta detectada: '{match_titulo[0]}' ({match_titulo[1]:.1f}%)")
            palabras_expandidas.append(match_titulo[0])

    return " ".join(palabras_expandidas)

# ==========================================
# 4. MOTOR DE BÚSQUEDA
# ==========================================
def preprocesar_busqueda(texto):
    # Paso 1: Limpieza básica
    texto_limpio = limpiar_texto(texto)
    # Paso 2: Tokenización
    tokens = nltk.word_tokenize(texto_limpio, language='spanish')
    tokens_stem = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(tokens_stem)

def buscar_contexto(consulta):
    # 1. APLICAMOS RAPIDFUZZ AQUÍ
    # Esto transforma "como se ase la paeya" en "como se ase la paeya paella valenciana"
    consulta_optimizada = inyectar_correcciones(consulta)
    
    corpus_limpio = [limpiar_texto(doc) for doc in BASE_CONOCIMIENTO]
    # Usamos la consulta optimizada para la vectorización
    corpus_para_vectorizar = corpus_limpio + [consulta_optimizada]
    
    vectorizer = TfidfVectorizer(preprocessor=preprocesar_busqueda)
    try:
        tfidf = vectorizer.fit_transform(corpus_para_vectorizar)
        cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = np.argmax(cosine)
        score = cosine[0][idx]
        
        print(f"[TF-IDF] Consulta Original: '{consulta}' | Optimizada: '{consulta_optimizada}'")
        print(f"[TF-IDF] Match Index: {idx} | Score: {score:.4f}")
        
        if score > 0.1: # Subimos un poco el umbral ya que ahora "ayudamos" al motor
            return BASE_CONOCIMIENTO[idx]
    except Exception as e:
        print(f"Error en búsqueda vectorizada: {e}")
        
    return None

# ==========================================
# 5. ENDPOINT CHAT
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    contexto = buscar_contexto(user_msg)
    
    if contexto:
        system_instruction = f"""Eres un asistente experto. Usa el siguiente CONTEXTO para responder.
        
        REGLAS VISUALES:
        1. Títulos: **NEGRITA**.
        2. Ingredientes: Lista con guiones (-).
        3. Direcciones: Usa bloque de cita (>).
        
        CONTEXTO:
        {contexto}
        """
    else:
        system_instruction = f"""Eres un asistente amable del restaurante.
        No encontraste información exacta.
        
        TEMAS DISPONIBLES:
        - Recetas: {", ".join(LISTA_TITULOS)}
        - Direcciones.
        
        Di que no entendiste bien y sugiere ver las recetas."""

    prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=300, stop=["<|im_end|>"], echo=False, temperature=0.3)
        respuesta = output['choices'][0]['text'].strip()
    except Exception as e:
        respuesta = "Error técnico en el cerebro local."
        print(f"Error LLM: {e}")

    return jsonify({"response": respuesta})

if __name__ == '__main__':
    print("--- Servidor LLM Iniciado ---")
    app.run(port=5000, debug=True)