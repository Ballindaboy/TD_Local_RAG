import os
import sys
import threading
from dotenv import load_dotenv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QFileDialog, 
                             QHBoxLayout, QTabWidget, QLineEdit, QMessageBox,
                             QProgressBar, QSplitter, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

# Импорт функций из наших модулей
from rag import load_documents, split_documents, create_vector_db
from ask import load_vector_db, create_qa_chain

# Загрузка переменных окружения
load_dotenv()


class IndexingWorker(QThread):
    """Поток для индексации документов"""
    update_status = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, directory_path):
        super().__init__()
        self.directory_path = directory_path

    def run(self):
        try:
            # Загрузка документов
            self.update_status.emit("Загрузка документов...")
            self.update_progress.emit(10)
            documents = load_documents(self.directory_path)
            
            if not documents:
                self.finished.emit(False, "Не найдено документов в указанной папке.")
                return
                
            self.update_status.emit(f"Загружено {len(documents)} документов")
            self.update_progress.emit(30)
            
            # Разделение на чанки
            self.update_status.emit("Разделение документов на чанки...")
            chunks = split_documents(documents)
            self.update_status.emit(f"Создано {len(chunks)} чанков")
            self.update_progress.emit(60)
            
            # Создание векторной БД
            self.update_status.emit("Создание векторной БД...")
            vector_db = create_vector_db(chunks)
            self.update_progress.emit(90)
            
            self.update_status.emit("Индексация завершена успешно")
            self.update_progress.emit(100)
            self.finished.emit(True, "Индексация успешно завершена!")
        except Exception as e:
            self.update_status.emit(f"Ошибка: {str(e)}")
            self.finished.emit(False, f"Ошибка при индексации: {str(e)}")


class QAWorker(QThread):
    """Поток для обработки вопроса и получения ответа"""
    answer_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, question, qa_chain):
        super().__init__()
        self.question = question
        self.qa_chain = qa_chain

    def run(self):
        try:
            response = self.qa_chain.invoke({"query": self.question})
            self.answer_ready.emit(response["result"])
        except Exception as e:
            self.error_occurred.emit(f"Ошибка при обработке вопроса: {str(e)}")


class RAGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Инициализация переменных
        self.vector_db = None
        self.qa_chain = None
        self.data_directory = os.path.join(os.getcwd(), "data")
        
        # Настройка окна
        self.setWindowTitle("TD Local RAG")
        self.setGeometry(100, 100, 900, 700)
        
        # Создаем виджет с вкладками
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Создаем вкладки
        self.setup_indexing_tab()
        self.setup_qa_tab()
        
        # Проверяем наличие индекса при запуске
        self.check_index_exists()

    def setup_indexing_tab(self):
        """Настройка вкладки для индексации документов"""
        indexing_tab = QWidget()
        layout = QVBoxLayout()
        
        # Выбор директории
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Директория с документами:")
        self.dir_path = QLineEdit(self.data_directory)
        self.dir_path.setReadOnly(True)
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_path, 1)
        dir_layout.addWidget(self.browse_btn)
        
        # Кнопка индексации
        self.index_btn = QPushButton("Начать индексацию")
        self.index_btn.clicked.connect(self.start_indexing)
        self.index_btn.setStyleSheet("font-weight: bold; height: 30px;")
        
        # Статус и прогресс
        self.index_status = QLabel("Статус: Ожидание индексации")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # Лог индексации
        log_label = QLabel("Лог индексации:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Добавляем виджеты на вкладку
        layout.addLayout(dir_layout)
        layout.addWidget(self.index_btn)
        layout.addWidget(self.index_status)
        layout.addWidget(self.progress_bar)
        layout.addWidget(log_label)
        layout.addWidget(self.log_text)
        
        indexing_tab.setLayout(layout)
        self.tabs.addTab(indexing_tab, "Индексация документов")

    def setup_qa_tab(self):
        """Настройка вкладки для вопросов и ответов"""
        qa_tab = QWidget()
        layout = QVBoxLayout()
        
        # Верхняя панель с управлением
        controls_layout = QHBoxLayout()
        
        # Выбор модели
        model_label = QLabel("Модель:")
        self.model_selector = QComboBox()
        self.model_selector.addItem("llama-3.2-3b-instruct (локальная)")
        self.model_selector.addItem("OpenAI API (требуется ключ)")
        
        # Кнопка загрузки модели
        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model)
        
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_selector)
        controls_layout.addWidget(self.load_model_btn)
        controls_layout.addStretch(1)
        
        # Разделитель для истории разговора и ввода
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # История разговора
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_label = QLabel("История разговора:")
        self.conversation_history = QTextEdit()
        self.conversation_history.setReadOnly(True)
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.conversation_history)
        history_widget.setLayout(history_layout)
        
        # Ввод вопроса
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_label = QLabel("Ваш вопрос:")
        self.question_input = QTextEdit()
        self.question_input.setMaximumHeight(100)
        self.send_btn = QPushButton("Отправить")
        self.send_btn.clicked.connect(self.ask_question)
        self.send_btn.setEnabled(False)  # Отключено до загрузки модели
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.question_input)
        input_layout.addWidget(self.send_btn)
        input_widget.setLayout(input_layout)
        
        # Добавляем виджеты в разделитель
        splitter.addWidget(history_widget)
        splitter.addWidget(input_widget)
        splitter.setSizes([600, 200])
        
        # Добавляем все на вкладку
        layout.addLayout(controls_layout)
        layout.addWidget(splitter)
        
        qa_tab.setLayout(layout)
        self.tabs.addTab(qa_tab, "Вопросы и ответы")

    def browse_directory(self):
        """Открывает диалог выбора директории"""
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите директорию с документами")
        if dir_path:
            self.data_directory = dir_path
            self.dir_path.setText(dir_path)

    def start_indexing(self):
        """Запускает процесс индексации в отдельном потоке"""
        # Отключаем кнопки
        self.index_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        
        # Очищаем лог
        self.log_text.clear()
        
        # Запускаем индексацию в отдельном потоке
        self.worker = IndexingWorker(self.data_directory)
        self.worker.update_status.connect(self.update_indexing_status)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.indexing_finished)
        self.worker.start()

    def update_indexing_status(self, status):
        """Обновляет статус индексации"""
        self.index_status.setText(f"Статус: {status}")
        self.log_text.append(status)

    def update_progress(self, value):
        """Обновляет прогресс-бар"""
        self.progress_bar.setValue(value)

    def indexing_finished(self, success, message):
        """Обрабатывает завершение индексации"""
        # Включаем кнопки
        self.index_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Индексация завершена", message)
            # Проверяем наличие индекса
            self.check_index_exists()
        else:
            QMessageBox.warning(self, "Ошибка индексации", message)

    def check_index_exists(self):
        """Проверяет наличие индекса"""
        if os.path.exists("faiss_index"):
            self.index_status.setText("Статус: Индекс найден")
            self.tabs.setTabEnabled(1, True)  # Включаем вкладку с вопросами
        else:
            self.index_status.setText("Статус: Индекс не найден")
            self.tabs.setTabEnabled(1, False)  # Отключаем вкладку с вопросами

    def load_model(self):
        """Загружает модель и векторную БД"""
        try:
            self.load_model_btn.setText("Загрузка...")
            self.load_model_btn.setEnabled(False)
            
            # Загружаем векторную БД
            self.vector_db = load_vector_db()
            if not self.vector_db:
                QMessageBox.warning(self, "Ошибка", "Векторная база данных не найдена. Сначала запустите индексацию.")
                self.load_model_btn.setText("Загрузить модель")
                self.load_model_btn.setEnabled(True)
                return
            
            # Создаем цепочку вопрос-ответ
            use_local = self.model_selector.currentIndex() == 0
            self.qa_chain = create_qa_chain(self.vector_db, use_local_llm=use_local)
            
            # Активируем кнопку отправки вопроса
            self.send_btn.setEnabled(True)
            self.load_model_btn.setText("Модель загружена")
            
            # Добавляем сообщение в историю
            model_name = "локальная LLM" if use_local else "OpenAI API"
            self.conversation_history.append(f"<b>Система:</b> Модель {model_name} успешно загружена. Можете задавать вопросы.")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки модели", f"Не удалось загрузить модель: {str(e)}")
            self.load_model_btn.setText("Загрузить модель")
            self.load_model_btn.setEnabled(True)

    def ask_question(self):
        """Отправляет вопрос и получает ответ"""
        question = self.question_input.toPlainText().strip()
        if not question:
            return
        
        if not self.qa_chain:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель.")
            return
        
        # Добавляем вопрос в историю
        self.conversation_history.append(f"<b>Вы:</b> {question}")
        self.conversation_history.append("<b>Система:</b> Генерирую ответ...")
        self.send_btn.setEnabled(False)
        self.question_input.setEnabled(False)
        
        # Запускаем поток обработки вопроса
        self.qa_worker = QAWorker(question, self.qa_chain)
        self.qa_worker.answer_ready.connect(self.display_answer)
        self.qa_worker.error_occurred.connect(self.handle_qa_error)
        self.qa_worker.start()
        
        # Очищаем поле ввода
        self.question_input.clear()

    def display_answer(self, answer):
        """Отображает полученный ответ"""
        # Удаляем сообщение "Генерирую ответ..."
        cursor = self.conversation_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()  # Удаляем перевод строки
        
        # Добавляем ответ
        self.conversation_history.append(f"<b>Система:</b> {answer}")
        self.conversation_history.append("")  # Пустая строка для разделения
        
        # Прокручиваем историю вниз
        self.conversation_history.verticalScrollBar().setValue(
            self.conversation_history.verticalScrollBar().maximum()
        )
        
        # Включаем элементы управления
        self.send_btn.setEnabled(True)
        self.question_input.setEnabled(True)
        self.question_input.setFocus()

    def handle_qa_error(self, error_message):
        """Обрабатывает ошибки при получении ответа"""
        # Удаляем сообщение "Генерирую ответ..."
        cursor = self.conversation_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()  # Удаляем перевод строки
        
        # Добавляем сообщение об ошибке
        self.conversation_history.append(f"<b>Система:</b> <span style='color:red;'>{error_message}</span>")
        self.conversation_history.append("")  # Пустая строка для разделения
        
        # Включаем элементы управления
        self.send_btn.setEnabled(True)
        self.question_input.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGApp()
    window.show()
    sys.exit(app.exec())