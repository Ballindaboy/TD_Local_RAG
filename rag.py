import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
# Для локального сервера
from langchain_openai import ChatOpenAI

# Загрузка переменных окружения из .env файла
load_dotenv()

# 1. Загрузка документов
def load_documents(directory):
    from langchain_community.document_loaders import TextLoader
    import glob
    
    documents = []
    
    # Загрузка TXT файлов
    for file_path in glob.glob(f"{directory}/**/*.txt", recursive=True):
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Загружен TXT файл: {file_path}")
        except Exception as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
    
    # Загрузка MD файлов как обычного текста
    for file_path in glob.glob(f"{directory}/**/*.md", recursive=True):
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Загружен MD файл: {file_path}")
        except Exception as e:
            print(f"Ошибка при загрузке MD файла {file_path}: {e}")
    
    return documents

# 2. Разделение документов на чанки
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# 3. Создание векторной базы данных
def create_vector_db(chunks):
    print("Используем тестовые эмбеддинги (только для демонстрации)")
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=1536)  # размерность как у OpenAI
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

# 4. Создание RAG цепочки
def create_rag_chain(vector_db, use_local_llm=True):
    if use_local_llm:
        # Использование локальной модели через API сервер
        print("Используем локальную LLM модель (Llama 3.2-3b-instruct)")
        
        # Настраиваем промпт для локальной модели
        prompt_template = """<|begin_of_text|><|system|>
Ты полезный ассистент, который отвечает на вопросы, используя только информацию из контекста. 
Если информации в контексте недостаточно, скажи, что не знаешь ответа.

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
    else:
        # Использование OpenAI API
        # Если ключ OpenAI не найден, выводим ошибку
        if not os.getenv("OPENAI_API_KEY"):
            print("Ошибка: API ключ OpenAI не найден в .env файле")
            print("Пожалуйста, добавьте OPENAI_API_KEY=ваш-ключ в .env файл")
            return None
        
        print("Используем OpenAI API")
        llm = OpenAI(temperature=0)
        PROMPT = None
    
    # Создаем RAG цепочку с указанной моделью
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT} if PROMPT else {}
    )
    return qa_chain

# 5. Ответ на вопросы
def answer_question(qa_chain, question):
    response = qa_chain.invoke({"query": question})
    return response["result"]

# Пример использования
if __name__ == "__main__":
    # Проверяем наличие файлов в папке data
    if not os.listdir("data"):
        print("Папка data пуста. Создаем пример текстового файла...")
        with open("data/sample.txt", "w") as f:
            f.write("""
            Искусственный интеллект (ИИ) — это область компьютерных наук, 
            направленная на создание систем, которые могут выполнять задачи, 
            обычно требующие человеческого интеллекта.
            
            RAG (Retrieval-Augmented Generation) — это метод, который сочетает 
            поиск информации с генеративными моделями для создания ответов, 
            основанных на конкретных данных.
            """)
    
    # Запускаем RAG
    print("Загрузка документов...")
    documents = load_documents("data")
    print(f"Загружено {len(documents)} документов")
    
    print("Разделение документов на чанки...")
    chunks = split_documents(documents)
    print(f"Создано {len(chunks)} чанков")
    
    print("Создание векторной БД...")
    vector_db = create_vector_db(chunks)
    print("Векторная БД создана и сохранена в папке faiss_index")
    
    # Используем локальную модель по умолчанию
    print("Создание RAG цепочки с локальной моделью...")
    qa_chain = create_rag_chain(vector_db, use_local_llm=True)
    
    if qa_chain:
        # Тестовые запросы
        test_questions = [
            "Что такое RAG?",
            "Какие преимущества у RAG систем?",
            "Из каких этапов состоит работа RAG системы?"
        ]
        
        print("\n=== Тестовые запросы к RAG системе ===")
        
        for question in test_questions:
            print(f"\nВопрос: {question}")
            print("Ищу ответ...")
            answer = answer_question(qa_chain, question)
            print(f"Ответ: {answer}")
        
        print("\nRAG система успешно создана и протестирована!")
        print("Теперь вы можете добавлять свои текстовые (.txt) и markdown (.md) файлы в папку data/")
        print("и задавать вопросы по этим документам.")