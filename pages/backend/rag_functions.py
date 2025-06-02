import os
import streamlit as st
import pandas as pd
from typing import List, Dict
import json
from langchain_gigachat.chat_models import GigaChat
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, LLMChain
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableSequence
from langgraph.prebuilt import create_react_agent


chat_history_m = []

# ======== CQL генерация ========
def load_csv_as_context(csv_path: str, max_rows: int = 5) -> str:
    df = pd.read_csv(csv_path)
    context = df.head(max_rows).to_string(index=False)
    columns_info = ', '.join(df.columns)
    return f"Таблица с колонками: {columns_info}\nПример данных:\n{context}"

def generate_cql_query(llm, instruction: str, context: str) -> str:
    prompt = PromptTemplate.from_template("""
Ты — помощник для генерации CQL-запросов к Cassandra. Используй структуру таблицы и описание операции ниже.

Контекст:
{context}

Задание:
{instruction}

Ответ:
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, instruction=instruction)

def generate_sql_query(llm, instruction: str, context: str) -> str:
    prompt = PromptTemplate.from_template("""
Ты — помощник для генерации SQL-запросов к реляционной базе данных. Используй структуру таблицы и описание операции ниже.

Контекст:
{context}

Задание:
{instruction}

Ответ:
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, instruction=instruction)


# ======== Индексация и загрузка ========
def create_and_save_faiss_index(text_file_path: str, vector_store_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
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

# ======== Подготовка RAG ========
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
Ответ: '''
    )
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    retriever_tool = create_retriever_tool(retriever, "search_web", "Searches and returns data from documents")

    @tool
    def memory_clearing() -> None:
        '''Очищает историю диалога'''
        global chat_history_m
        chat_history_m = []

    tools = [retriever_tool, memory_clearing]
    retriever = db.as_retriever(search_kwargs={"k": 5})
    #chain: Runnable = prompt | llm | StrOutputParser()
    chain: RunnableSequence = prompt | llm | StrOutputParser()
    #agent = create_react_agent(llm, tools, prompt=AGENT_PROMPT)
    #agent = create_gigachat_functions_agent(llm, tools)
    #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent = create_react_agent(llm, tools)
    agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)
    
    return agent_executor, llm, retriever

# ======== Ответ от агента ========
def generate_answer(question):
    result = st.session_state.conversation.invoke({"input": question})
    answer = result.get("output", "")
    #answer = st.session_state.conversation.invoke({"input": question})["output"]
    # Добавляем в историю
    st.session_state.chat_history.append({"role": "user", "message": question})
    st.session_state.chat_history.append({"role": "assistant", "message": answer})
    return answer, None


def generate_quiz_from_retriever(llm, retriever, query="Создай тест по теме больших данных"):
    """
    Генерирует 6 тестовых вопросов по теме запроса на основе контекста из retriever и модели LLM.

    :param llm: Модель GigaChat или совместимая LLM.
    :param retriever: Объект retriever (например, FAISS.as_retriever()).
    :param query: Тема или вопрос, по которому нужно сгенерировать тест.
    :return: Строка с форматированными вопросами и ответами.
    """
    # 1. Получаем релевантный контекст
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Формируем prompt
    test_prompt = PromptTemplate.from_template("""
Ты — преподаватель по теме "Большие данные". Используй только предоставленный контекст, чтобы создать 6 тестовых вопросов с 4 вариантами ответа (один из них правильный).

Контекст:
{context}

Формат ответа:
1. Вопрос?
А) вариант
Б) вариант
В) вариант
Г) правильный вариант
Ответ: Г

Начни генерировать:
""")

    # Генерируем результат
    chain = LLMChain(llm=llm, prompt=test_prompt)
    result = chain.run(context=context)

    return result


def check_quiz_answers(llm, questions: List[Dict[str, str]], user_answers: List[str]) -> List[Dict[str, str]]:
    """Проверка ответов с пояснением."""
    results = []
    for q, user_ans in zip(questions, user_answers):
        correct = user_ans == q["answer"]
        explanation = ""
        if not correct:
            prompt = PromptTemplate.from_template('''
Ты — объясняющий ассистент. Объясни, почему верный ответ "{correct}", а не "{wrong}" на вопрос: {question}.
Варианты ответов: {options}
''')
            chain = LLMChain(llm=llm, prompt=prompt)
            explanation = chain.run(question=q["question"], correct=q["answer"], wrong=user_ans, options=", ".join(q["options"]))
        results.append({
            "question": q["question"],
            "your_answer": user_ans,
            "correct": q["answer"],
            "result": "✅ Правильно" if correct else "❌ Неправильно",
            "explanation": explanation if not correct else ""
        })
    return results
