import streamlit as st
import tempfile
import os
from PyPDF2 import PdfReader
from transformers import pipeline
from typing import ClassVar
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 🚀 LLM local basé sur Flan-T5 (petit modèle pour CPU)
class CustomLLM(LLM):
    pipeline: ClassVar = pipeline("text2text-generation", model="google/flan-t5-small")

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        output = self.pipeline(prompt, max_new_tokens=200)
        return output[0]["generated_text"]

# 🧠 Embeddings open source
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🎯 Streamlit UI
st.set_page_config(page_title="📘 Q&A - Open Source", layout="centered")
st.title("📘 Résumé & Q&A - 100% Open Source")
st.subheader("📤 Uploade ton PDF de cours")

uploaded_file = st.file_uploader("Glisse ton fichier PDF ici", type="pdf")

if uploaded_file is not None:
    st.success("✅ Fichier chargé avec succès.")

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 📖 Lecture du contenu PDF
    reader = PdfReader(tmp_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # ✂️ Découpage du texte
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # 🔎 Embeddings + index
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    # 🤖 Chargement du LLM local
    llm = CustomLLM()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # 💬 Question utilisateur
    st.subheader("💬 Pose ta question sur le document")
    question = st.text_input("Ex: Quel est le sujet principal du document ?")

    if st.button("📥 Envoyer") and question:
        try:
            with st.spinner("🔍 Génération de la réponse..."):
                answer = qa_chain.run(question)
                st.success("✅ Réponse :")
                st.write(answer)
        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
