import sys
from PyQt6.QtWidgets import (
    QTabWidget,
    QMainWindow,
    QApplication
)
from Model.ImportData import ImportData
from Model.TransformationData import TransformationData
from Model.ExportData import ExportData 
from ViewModel.ExploreData import ExploreData
from ViewModel.LLMInsights import LLMInsights
from ViewModel.NetworkVisualization import NetworkVisualization 
from ViewModel.StatisticalAnalysis import StatisticalAnalysis

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("NEDAF: Network Data Analysis Framework")
        self.setGeometry(100,100,800,600)

        #crear el QTabWidget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.tabs.setMovable(True)
        self.setCentralWidget(self.tabs)
        
        #agregar pestañas
        self.importar_tab = ImportData()
        self.transformar_tab  = TransformationData()
        self.explorar_tab = ExploreData()
        self.visualizacion_tab = NetworkVisualization()
        self.analisis_tab = StatisticalAnalysis()
        self.LLM_tab = LLMInsights()
        self.exportar_tab = ExportData()

        self.tabs.addTab(self.importar_tab, "Importar Datos") 
        self.tabs.addTab(self.transformar_tab, "Transformación de Datos")
        self.tabs.addTab(self.explorar_tab, "Exploración de Datos")
        self.tabs.addTab(self.visualizacion_tab, "Visualización de Datos")
        self.tabs.addTab(self.analisis_tab, "Analisis de Datos")
        self.tabs.addTab(self.LLM_tab, "LLM Insight")
        self.tabs.addTab(self.explorar_tab, "Exportar Datos")


        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
