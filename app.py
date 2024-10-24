import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

# Inyectar CSS para cambiar las tipografías
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400&family=Lexend:wght@600&display=swap');

    h1, h2, h3 {
        font-family: 'Lexend', sans-serif;
    }

    p, div, label, span, input, textarea {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Título y subtítulo de la aplicación
st.title('Generación Aumentada por Recuperación (RAG) 💬')

# Mostrar imagen en la app
image = Image.open('Chat_pdf.png')

# Mostrar la versión de Python
st.write(f"<span>Versión de Python: {platform.python_version()}</span>", unsafe_allow_html=True)
st.image(image, width=350)

# Barra lateral con información
with st.sidebar:
   st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Entrada para la clave de API de OpenAI
ke = st.text_input('Ingresa tu Clave de API de OpenAI', type='password')
os.environ['OPENAI_API_KEY'] = ke

# Cargar archivo PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Extraer y procesar el texto del PDF
if pdf is not None:
    # Crear un lector de PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Dividir el texto en chunks (fragmentos)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)

    # Crear embeddings a partir de los fragmentos del texto
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Mostrar el campo de entrada para las preguntas
    st.subheader("Escribe lo que quieres saber sobre el documento")
    user_question = st.text_area(" ")

    if user_question:
        # Buscar en la base de conocimientos la pregunta del usuario
        docs = knowledge_base.similarity_search(user_question)

        # Cargar el modelo de lenguaje y realizar la cadena de preguntas y respuestas
        llm = OpenAI(model_name="gpt-4")
        chain = load_qa_chain(llm, chain_type="stuff")

        # Mostrar la respuesta
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
