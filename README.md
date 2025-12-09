# 🤖 Agente Conversacional RAG Local (CPU-Only)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask)
![LLM](https://img.shields.io/badge/Model-Qwen2.5--1.5B-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Este proyecto implementa un agente conversacional autónomo basado en arquitectura **RAG (Retrieval-Augmented Generation)** que se ejecuta enteramente en local. Combina modelos de lenguaje cuantizados (GGUF) con búsqueda vectorial TF-IDF y lógica difusa para responder preguntas basadas en documentos `.txt` propios.

---

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#-requisitos-previos)
2. [Instalación del Entorno](#-instalación-del-entorno)
3. [Configuración del Modelo](#-configuración-del-modelo)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Ejecución](#-ejecución)

---

## 🛠 Requisitos Previos

* **Anaconda** o **Miniconda** instalado.
* Sistema Operativo: Windows, Linux o macOS.
* Memoria RAM: Mínimo 4GB (Recomendado 8GB+).
* No requiere GPU dedicada.

---

## 🚀 Instalación del Entorno

Sigue estos pasos para configurar el entorno virtual y las dependencias necesarias.

### 1. Crear y activar el entorno
Utilizamos Python 3.9 para asegurar compatibilidad con las librerías de `llama-cpp`.

```bash
conda create -n chatbot_llm python=3.9
conda activate chatbot_llm
conda install jupyter  
pip install flask flask-cors nltk scikit-learn numpy
pip install rapidfuzz