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
import unicodedata # Para manejar tildes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración NLTK
nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')
stemmer = SnowballStemmer('spanish')
stop_words = stopwords.words('spanish')

# ==========================================
# 1. CONFIGURACIÓN DEL LEXICÓN (SINÓNIMOS)
# ==========================================
# Esto ayuda a que "donde queda" encuentre "DIRECCIÓN"
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
    "comida": "recetas"
}

# ==========================================
# 2. CARGA DEL MODELO
# ==========================================
print("--- CARGANDO CEREBRO ---")
LLM = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

# ==========================================
# 3. GESTIÓN DE DOCUMENTOS
# ==========================================
BASE_CONOCIMIENTO = []
LISTA_TITULOS = []

# Nueva funcion de limpieza profunda
def limpiar_texto(texto):
    # 1. Convertir a minusculas
    texto = texto.lower()
    # 2. Eliminar acentos (normalize)
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
                    
                    # Extraer titulos para el contexto global
                    lines = chunk.split('\n')
                    for line in lines:
                        if "TITULO:" in line.upper():
                            titulo = line.split(":")[1].strip()
                            LISTA_TITULOS.append(titulo)
                            
    print(f"--- SISTEMA LISTO: {len(BASE_CONOCIMIENTO)} bloques cargados ---")

cargar_documentos()

# ==========================================
# 4. MOTOR DE BÚSQUEDA MEJORADO
# ==========================================
def preprocesar_busqueda(texto):
    # 1. Limpieza base (tildes, minusculas)
    texto_limpio = limpiar_texto(texto)
    
    # 2. Inyeccion de sinonimos (Lexicon)
    palabras = texto_limpio.split()
    palabras_enriquecidas = palabras.copy()
    
    for palabra in palabras:
        if palabra in KEYWORD_MAP:
            # Si el usuario dice "donde", agregamos "direccion" a la busqueda oculta
            palabras_enriquecidas.append(KEYWORD_MAP[palabra])
            
    texto_expandido = " ".join(palabras_enriquecidas)
    
    # 3. Stemming estandar
    tokens = nltk.word_tokenize(texto_expandido, language='spanish')
    tokens_stem = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(tokens_stem)

def buscar_contexto(consulta):
    # Preprocesamos documentos al vuelo (idealmente esto se cachea)
    corpus_limpio = [limpiar_texto(doc) for doc in BASE_CONOCIMIENTO]
    
    # Agregamos la consulta procesada al final
    corpus_para_vectorizar = corpus_limpio + [limpiar_texto(consulta)]
    
    # Vectorizacion
    vectorizer = TfidfVectorizer(preprocessor=preprocesar_busqueda)
    tfidf = vectorizer.fit_transform(corpus_para_vectorizar)
    
    # Similitud
    cosine = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = np.argmax(cosine)
    score = cosine[0][idx]
    
    print(f"[DEBUG] Consulta: '{consulta}' | Match Index: {idx} | Score: {score:.4f}")
    
    # Umbral bajo porque ahora confiamos mas en la coincidencia semantica
    if score > 0.05:
        return BASE_CONOCIMIENTO[idx]
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
        
        REGLAS DE FORMATO VISUAL (OBLIGATORIO):
        1. Títulos de recetas: Ponlos en **Negrita** y mayúsculas.
        2. Ingredientes: Usa SIEMPRE una lista con guiones (-). Ejemplo:
           - Arroz
           - Pollo
        3. Pasos: Usa lista numerada (1. 2.).
        4. Direcciones o Ubicaciones: Si das una dirección, ponla dentro de un bloque de cita usando el signo mayor que (>). Ejemplo:
           > Calle Principal 123, Ciudad.
        
        CONTEXTO:
        {contexto}
        """
    else:
        system_instruction = f"""Eres un asistente amable.
        No encontraste información específica.
        
        TUS TEMAS DISPONIBLES:
        - Recetas: {", ".join(LISTA_TITULOS)}
        - Direcciones.
        
        REGLAS: Responde corto y sugiere ver las recetas disponibles."""

    # Prompt Template
    prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        output = LLM(prompt, max_tokens=250, stop=["<|im_end|>"], echo=False, temperature=0.3) # Temperature baja = menos creatividad/alucinacion
        respuesta = output['choices'][0]['text'].strip()
    except Exception as e:
        respuesta = "Lo siento, estoy teniendo problemas técnicos."
        print(f"Error LLM: {e}")

    return jsonify({"response": respuesta})

if __name__ == '__main__':
    app.run(port=5000, debug=True)