from PyQt6.QtWidgets import (QMessageBox, QPushButton, 
                             QFileDialog, QVBoxLayout, QWidget, 
                            QLabel, QTableWidget, QTableWidgetItem, 
                            QTabWidget)
from PyQt6.QtCore import QStandardPaths
from View.FileExplorer import FileExplorerWidget, SSHFileSystemModel
import dask.dataframe as dd
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
from View.DataManager import DataManager
import numpy as np

class ImportData(QWidget):

   def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        self.initUI()

   def initUI(self):
     #layout principal
      Import_layout = QVBoxLayout()
      
      self.tabExplorer = QTabWidget()
      #layout para el explorador de archivos
      self.FileExplorer()
      #Titulo
      title = QLabel("Importar Datos")
      title.setStyleSheet("font-weight: bold; font-size: 16px;")
      Import_layout.addWidget(title)

      #boton de importar archivo
      buttonLoadData = QPushButton("Importar Archivo", self)
      buttonLoadData.clicked.connect(self.LoadData)
      Import_layout.addWidget(buttonLoadData)
      
      buttonReloadData = QPushButton("Reload Data", self)
      buttonReloadData.clicked.connect(self.reloadData)
      Import_layout.addWidget(buttonReloadData)

      #Area para mostrar info del archivo cargado
      self.info_file = QLabel("Ningún archivo cargado")
      Import_layout.addWidget(self.info_file)

      #tabla para previsualizar datos
      self.data_table = QTableWidget()
      self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # read only
      
      Import_layout.addWidget(self.data_table)
      Import_layout.addWidget(self.tabExplorer)
      self.setLayout(Import_layout)
  
   def FileExplorer(self, host = None, user = None, password = None): 
     
      if host is None and user is None and password is None:
        self.fileExplorer =  FileExplorerWidget()
        self.tabExplorer.addTab(self.fileExplorer, "File Explorer")
        
      else:
        self.tabExplorer.removeTab(0)
        rFileExplorer = SSHFileSystemModel()
        self.fileExplorer =  rFileExplorer.RemoteFileExplorer(host, user, password)
        self.tabExplorer.addTab(self.fileExplorer, "Remote File Explorer")
        
      self.tabExplorer.setTabPosition(QTabWidget.TabPosition.West)
                
   def reloadData(self) -> None:
       df = self.data_manager.get_data()
       self.fill_data_table(df)


   def LoadData(self):
      #definir opciones
      
      options = (QFileDialog.Option.DontUseNativeDialog)
      default_dir = QStandardPaths.writableLocation(
          QStandardPaths.StandardLocation.DesktopLocation
      )
      file_type = "CSV Files (*.csv);; Parquet Files (*.parquet);; Excel Files (*.xlsx);; Edges Files (*.edges);; mtx Files (*.mtx)"
      self.file, _ = QFileDialog.getOpenFileName(self, "Open File", default_dir, file_type, options=options)
      try:
            if self.file.endswith('.csv') :
              #here we load csv files 
               df = pd.read_csv(self.file)
               self.info_file.setText(f"Archivo cargado: {self.file}")
            elif self.file.endswith('.parquet'):
               df = dd.read_parquet(self.file)
               self.info_file.setText(f"Archivo cargado: {self.file}")
            elif self.file.endswith('.xlsx'):
               df = pd.read_excel(self.file)
               self.info_file.setText(f"Archivo cargado: {self.file}")
            elif self.file.endswith('.edges'):
               df = pd.read_csv(self.file, sep=' ', header=None, names=["source", "destination", "weight"] or ["source", "destination"])
            elif self.file.endswith('.mtx'):
               matrixData = mmread(self.file)
               rows, cols = matrixData.nonzero()
               if isinstance(matrixData, csr_matrix):
                   weights = matrixData.data
                   df = pd.DataFrame({'source': rows, 'destination': cols, 'weight': weights})
               else:
                   df = pd.DataFrame({'source': rows, 'destination': cols})    

               


            
            self.data_manager.set_data(df)
            
            self.fill_data_table(df)
            
            return df
      except Exception as e:
            QMessageBox.critical(self, "Error", f"Ha ocurrido un error: {str(e)}")
   
   def fill_data_table(self, df: dd.DataFrame|pd.DataFrame):
       #vista previa del df
      preview_data = df.head(100).values
      #config # of rows and columns
      #self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
      self.data_table.setRowCount(len(preview_data))
      self.data_table.setColumnCount(len(preview_data[0]))
      #fill table
      for row in range(len(preview_data)):
          for column in range (len(preview_data[0])):
              item = QTableWidgetItem(str(preview_data[row][column]))
              self.data_table.setItem(row, column, item)

   
