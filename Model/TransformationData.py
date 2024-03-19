from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton

class TransformationData(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        boton = QPushButton("Transformar Datos")
        layout.addWidget(boton)
        