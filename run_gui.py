#!/usr/bin/env python3
"""
Скрипт для запуска графического интерфейса TD Local RAG
"""

import sys
from rag_gui import QApplication, RAGApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGApp()
    window.show()
    sys.exit(app.exec())