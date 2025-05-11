import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models.gigachat import GigaChat
from langchain.agents import create_gigachat_functions_agent, AgentExecutor
import streamlit as st

chat_history_m = []


def create_and_save_faiss_index(
        text_file_path: str,
        vector_store_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """Загружает текст, разбивает на чанки, создает FAISS-индекс и сохраняет его."""
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Файл не найден: {text_file_path}")

    loader = TextLoader(text_file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vector_store_path)

    return db


def prepare_rag_llm(token, model, embeddings_name, vector_store_path, temperature, max_length):
    os.environ["SB_AUTH_DATA"] = token

    llm = GigaChat(
        model=model,
        credentials=token,
        verify_ssl_certs=False,
        profanity_check=False,
        streaming=False,
        max_tokens=max_length,
        timeout=60,
        temperature=temperature
    )

    prompt = PromptTemplate.from_template(
        '''Ты — ассистент в образовании. Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста.
Если в контексте нет информации для ответа, сообщи об этом пользователю.
Контекст: {context}
Вопрос: {input}
Ответ:'''
    )

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    retriever_tool = create_retriever_tool(
        retriever,
        "search_web",
        "Searches and returns data from documents"
    )

    @tool
    def memory_clearing() -> None:
        '''Очищает историю диалога.'''
        global chat_history_m
        chat_history_m = []

    tools = [retriever_tool, memory_clearing]
    agent = create_gigachat_functions_agent(llm, tools)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    return agent_executor


def generate_answer(question, token):
    global chat_history_m
    answer = st.session_state.conversation.invoke({"input": question})["output"]
    return answer, None
