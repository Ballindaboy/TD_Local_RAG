import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Загрузка переменных окружения
load_dotenv()

def load_vector_db():
    """Загрузка существующей векторной базы данных"""
    if not os.path.exists("faiss_index"):
        print("Ошибка: Векторная база данных не найдена. Сначала запустите rag.py для создания базы.")
        return None
    
    # Используем FakeEmbeddings, так как база уже создана и нам не нужно вычислять новые эмбеддинги
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=1536)
    
    try:
        # Загрузка векторной БД с разрешением на десериализацию 
        # (безопасно, поскольку файл создали мы сами)
        vector_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_db
    except Exception as e:
        print(f"Ошибка при загрузке векторной базы данных: {e}")
        print("Пожалуйста, сначала запустите rag.py для создания базы.")
        return None

def create_qa_chain(vector_db):
    """Создание цепочки вопрос-ответ с локальной моделью"""
    print("Подключение к локальной модели Llama...")
    
    # Шаблон промпта для модели - улучшенная версия
    prompt_template = """<|begin_of_text|><|system|>
Ты полезный ассистент, который отвечает на вопросы пользователя на основе предоставленного контекста.
Контекст содержит личные заметки пользователя о встречах, бизнесе, и различных событиях.

Следуй этим правилам:
1. Используй ТОЛЬКО информацию из контекста для ответа
2. Давай подробные и исчерпывающие ответы, перечисляя всю релевантную информацию
3. Упоминай конкретные детали из контекста, чтобы показать, что ты используешь предоставленную информацию
4. Если в контексте есть противоречия, укажи на них
5. Если точная информация отсутствует, но есть релевантная частичная информация, предоставь то, что есть
6. Если нет релевантной информации, честно скажи "В ваших заметках нет информации по этому вопросу"

Контекст:
{context}
<|user|>
{question}
<|assistant|>
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Подключение к локальному серверу LLM
    llm = ChatOpenAI(
        model="llama-3.2-3b-instruct",  # Имя модели на локальном сервере
        temperature=0.1,
        base_url="http://127.0.0.1:1234/v1",  # URL локального сервера
    )
    
    # Настраиваем расширенный поиск с большим количеством документов
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}  # Увеличиваем количество возвращаемых документов до 8
    )
    
    # Создаем RAG цепочку с указанной моделью
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": False
        }
    )
    
    return qa_chain

def main():
    # Получаем вопрос из аргументов командной строки
    if len(sys.argv) < 2:
        print("Использование: python ask.py 'Ваш вопрос здесь'")
        return
    
    question = ' '.join(sys.argv[1:])
    
    print("Загрузка векторной базы данных...")
    vector_db = load_vector_db()
    
    if not vector_db:
        return
    
    print("Создание RAG цепочки...")
    qa_chain = create_qa_chain(vector_db)
    
    print(f"\nВопрос: {question}")
    print("Ищу ответ...")
    
    try:
        response = qa_chain.invoke({"query": question})
        print(f"\nОтвет: {response['result']}")
    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")

if __name__ == "__main__":
    main()