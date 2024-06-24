import sys
from PyQt6.QtWidgets import (
    QTabWidget,
    QMainWindow,
    QApplication, 
    QVBoxLayout, 
    QMessageBox

)
from PyQt6.QtGui import QAction
from Model.ImportData import ImportData
from Model.TransformationData import TransformationDataWindow
from Model.ExportData import ExportData 
from View.clusterLogin import ClusterLogin
from ViewModel.ExploreData import ExploreData
from ViewModel.LLMInsights import LLMInsights
from ViewModel.NetworkVisualizationMod import NetworkVisualizationMod 
from View.DataManager import DataManager


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.initUI()

        
    def initUI(self):
        self.setWindowTitle("NEDAF: Network Data Analysis Framework")
        self.setGeometry(100,100,800,600)
        self.data_manager = DataManager()
        self.tabs_init()
        self.ClusterOptionAction()
        
        
    def tabs_init(self):
        #main layout
        
        #crear el QTabWidget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South) 
        #self.tabs.setMovable(True) # puedes mover las pestañas
        self.setCentralWidget(self.tabs)
        
        #agregar pestañas
        self.importar_tab = ImportData(self.data_manager)
        self.transformar_tab  = TransformationDataWindow(self.data_manager)
        self.explorar_tab = ExploreData(self.data_manager)
        self.visualizacion_tab = NetworkVisualizationMod(self.data_manager)

        self.LLM_tab = LLMInsights()
        self.exportar_tab = ExportData()

        self.tabs.addTab(self.importar_tab, "Importar Datos") 
        self.tabs.addTab(self.transformar_tab, "Transformación de Datos")
        self.tabs.addTab(self.explorar_tab, "Exploración de Datos")
        self.tabs.addTab(self.visualizacion_tab, "Visualización de Datos")
        self.tabs.addTab(self.LLM_tab, "LLM Insight")
        self.tabs.addTab(self.exportar_tab, "Exportar Datos")
  

    def ClusterOptionAction(self):
         #creacion de QAction en el menuBar   
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Options")
        cluster_login_action = QAction("Conexión a Cluster", self)
        cluster_login_action.triggered.connect(self.showClusterLogin)
        file_menu.addAction(cluster_login_action)

    def showClusterLogin(self):
        try:
            self.clusterLogin = ClusterLogin()
            if self.clusterLogin.exec() == ClusterLogin.accepted:
                self.clusterLogin.connect_cluster()
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")  
            
                
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

