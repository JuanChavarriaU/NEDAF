from PyQt6.QtWidgets import QDialog, QPushButton, QFileDialog, QVBoxLayout, QWidget

class ImportData(QWidget): 

   def __init__(self):
        super().__init__()
        self.initUI()

   def initUI(self):
      layout = QVBoxLayout()
      boton = QPushButton("Importar Datos")
      layout.addWidget(boton)
      



    