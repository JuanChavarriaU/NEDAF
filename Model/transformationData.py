import pandas as pd
import dask.dataframe as dd


class TransformationData():

    def __init__(self, dataframe: pd.DataFrame | dd.DataFrame):
           super().__init__()
           """
            Inicializa la clase con un DataFrame que contiene los datos de la red.
            :param dataframe: DataFrame de pandas con las columnas 'source', 'destination' y 'weight'.
            """
           self.dataframe = dataframe
     
           #self.graph = nx.from_pandas_edgelist(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], edge_attr=self.dataframe.columns[2])
    

    def cut_missing_values(self):
        """
        Elimina las aristas que contienen valores faltantes en alguna de sus columnas.
        """
        self.dataframe = self.dataframe.dropna()        

    def normalize_data(self):
        """
        Normaliza los valores de la columna 'weight' para que estén en el rango [0, 1].
        """
        min_val = self.dataframe[self.dataframe.columns[2]].min()
        max_val = self.dataframe[self.dataframe.columns[2]].max()
        self.dataframe[self.dataframe.columns[2]] = (self.dataframe[self.dataframe.columns[2]] - min_val) / (max_val - min_val)
       
        
        

    def get_data(self) -> pd.DataFrame | dd.DataFrame:
         """
         Devuelve el dataframe con los cambios realizados.
            """ 
         return self.dataframe