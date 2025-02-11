import pandas as pd
import dask.dataframe as dd
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

class DataManager(QObject):
    """Clase para gestionar los datos en la aplicación."""
    
    data_loaded = pyqtSignal()
    
    def __init__(self) -> None:
        super().__init__()
        
        self.data = None
    
    def set_data(self, data: pd.DataFrame | dd.DataFrame):
        """Establece el DataFrame en el gestor."""
        try:
            self.data = data
            self.data_loaded.emit()
            #QMessageBox.StandardButton.Ok(self, "Exito", f"Datos cargados correctamente") no funciona
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")   
    
    def get_data(self):
        """Devuelve el DataFrame almacenado."""
        return self.data



