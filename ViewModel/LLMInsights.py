from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit
# from ViewModel import LLMBackend

class LLMInsights(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        layout = QVBoxLayout()

        #text area for conversation history
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)

        self.input_field = QLineEdit()

        self.send_button = QPushButton("Send ▶️")
        self.send_button.clicked.connect(self.on_send_clicked)


        layout.addWidget(self.text_area)
        layout.addWidget(self.input_field)
        layout.addWidget(self.send_button)
        self.setLayout(layout)

    def on_send_clicked(self):
        query = self.input_field.text()
        self.process_query(query)
        self.input_field.clear()
        

    def process_query(self, query):
        # Implement your LLM logic here

        # response = LLMBackend.answer(query)

        self.text_area.append(f"You: {query}")    
        
        # self.text_area.append(f"LLM: {response}")    