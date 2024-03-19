from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton


class LLMInsights(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
      layout = QVBoxLayout()
      boton = QPushButton("Importar Datos")
      layout.addWidget(boton)
      