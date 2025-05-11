import streamlit as st
import os
from pages.backend import rag_functions

st.title("GigaChat RAG Chatbot")

with st.expander("Настройки модели"):
    with st.form("setting"):
        row_1 = st.columns(3)
        with row_1[0]:
            token = st.text_input("GigaChat Token", type="password")

        with row_1[1]:
            llm_model = st.text_input("LLM model", value="GigaChat:latest")

        with row_1[2]:
            instruct_embeddings = st.text_input("Embeddings", value="sentence-transformers/all-MiniLM-L6-v2")

        row_2 = st.columns(2)
        with row_2[0]:
            vector_store_path = st.text_input("Vector Store Path", value="vector_store/tech_big")

        with row_2[1]:
            max_length = st.number_input("Максимальное число токенов", value=2000, step=1)

        create_chatbot = st.form_submit_button("Создать ассистента")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if token and create_chatbot:
    st.session_state.conversation = rag_functions.prepare_rag_llm(
        token, llm_model, instruct_embeddings, vector_store_path, temperature=1.0, max_length=max_length
    )

if "history" not in st.session_state:
    st.session_state.history = []

# Отображение истории
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод пользователя
if question := st.chat_input("Задайте вопрос"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.conversation:
        answer, _ = rag_functions.generate_answer(question, token)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})
if st.button("Создать индекс"):
    rag_functions.create_and_save_faiss_index("vector_store/tech_big/tech_bigs.txt", "vector_store/tech_big", instruct_embeddings)
    st.success("Индекс успешно создан!")
