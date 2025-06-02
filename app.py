import streamlit as st
import datetime
import json
from pages.backend import rag_functions
#from pages.backend.rag_functions import prepare_rag_llm, load_csv_as_context, generate_cql_query, generate_answer, \
    #generate_sql_query, generate_quiz_from_retriever
from pages/backend.rag_functions import prepare_rag_llm, load_csv_as_context, generate_cql_query, generate_answer, \
    generate_sql_query, generate_quiz_from_retriever

st.title("🎓 AI Ассистент + Генератор запросов")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.sidebar.subheader("⚙️ Настройки модели")
temperature = st.sidebar.slider("Температура генерации (креативность)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

if "conversation" not in st.session_state:
    token = st.text_input("🔑 Введите ключ GigaChat:", type="password")
    if token:
        st.session_state.conversation, st.session_state.llm,st.session_state.retriever = prepare_rag_llm(
            token=token,
            model="GigaChat:latest",
            embeddings_name="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path="vector_store/tech_big",
            temperature=temperature,
            max_length=2000,
        )
st.subheader("💬 Задай вопрос по базе знаний")
user_input = st.text_input("Вопрос")
if st.button("Ответить") and user_input:
    with st.spinner("Генерация ответа..."):
        answer, _ = generate_answer(user_input)
        st.write(answer)
        st.markdown("---")
        st.subheader("🕘 История диалога")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Вы:** {msg['message']}")
            else:
                st.markdown(f"**Ассистент:** {msg['message']}")
if st.button("🧹 Очистить историю"):
    st.session_state.chat_history = []

st.subheader("💾 Сохранение истории")

# Выбор формата сохранения
save_format = st.selectbox("Выберите формат для сохранения истории", ["JSON", "TXT"])

if st.button("💾 Сохранить историю"):
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.warning("История чата пуста.")
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if save_format == "JSON":
            filename = f"chat_history_{now}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
        else:  # TXT
            filename = f"chat_history_{now}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for msg in st.session_state.chat_history:
                    role = "Вы" if msg["role"] == "user" else "Ассистент"
                    f.write(f"{role}: {msg['message']}\n\n")

        st.success(f"История сохранена как `{filename}`")


# ======== Генерация CQL и SQL ========
st.subheader("🧠 Генерация запросов по обучающим датасетам")

col1, col2 = st.columns([1, 1])

csv_path = st.text_input("Путь к CSV-файлу", "vector_store/tech_big/russian_demography.csv")
instruction = st.text_area("Введите задание (например: 'Добавить строку о населении Москвы за 2020 год')")

with col1:
    if st.button("Сгенерировать CQL"):
        try:
            context = load_csv_as_context(csv_path)
            llm = st.session_state.llm
            cql_query = generate_cql_query(llm, instruction, context)
            st.code(cql_query, language='sql')
        except Exception as e:
            st.error(f"Ошибка при генерации CQL: {str(e)}")

with col2:
    if st.button("Сгенерировать SQL"):
        try:
            context = load_csv_as_context(csv_path)
            llm = st.session_state.llm
            sql_query = generate_sql_query(llm, instruction, context)
            st.code(sql_query, language='sql')
        except Exception as e:
            st.error(f"Ошибка при генерации SQL: {str(e)}")


# ======== Генерация теста ========
st.subheader("📚 Генерация теста по базе знаний")

# Кнопка для генерации
if st.button("📚 Сгенерировать тест"):
    with st.spinner("Генерация вопросов..."):
        try:
            quiz_text = generate_quiz_from_retriever(st.session_state.llm, st.session_state.retriever)
            if quiz_text and isinstance(quiz_text, str) and quiz_text.strip():
                # Удаляем ответы
                cleaned_quiz = "\n".join(
                    line for line in quiz_text.splitlines()
                    if not line.strip().lower().startswith("ответ")
                )
                st.session_state.generated_quiz_full = quiz_text  # Полный текст с ответами
                st.session_state.generated_quiz = cleaned_quiz     # Без ответов
                st.session_state.show_answers = False
                st.success("Тест успешно сгенерирован!")
            else:
                st.warning("Модель не вернула результат.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Отображение теста без ответов
if "generated_quiz" in st.session_state and st.session_state.generated_quiz:
    st.text_area("📄 Тест (без ответов)", value=st.session_state.generated_quiz, height=400)

    # Кнопка показать ответы
    #if st.button("✅ Проверить ответы"):
      #  st.session_state.show_answers = True

    # Показать ответы после нажатия
 #   if st.session_state.get("show_answers", False):
    #    st.text_area("🟢 Тест с ответами", value=st.session_state.generated_quiz_full, height=400)

    # Кнопка сохранить
    st.download_button(
        label="💾 Скачать тест с ответами",
        data=st.session_state.generated_quiz_full,
        file_name="quiz_with_answers.txt",
        mime="text/plain"
    )
