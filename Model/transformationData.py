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
    
    def delete_duplicates(self):
         """
        Elimina aristas duplicadas en el grafo. En un grafo, una arista duplicada sería aquella
        que conecta los mismos nodos con el mismo peso, por lo que este método podría no ser necesario
        si NetworkX ya maneja los duplicados de manera adecuada al crear el grafo.
        """
         self.dataframe = self.dataframe.drop_duplicates(subset=[self.dataframe.columns[0],self.dataframe.columns[1],self.dataframe.columns[2]])

    def cut_missing_values(self):
        """
        Elimina las aristas que contienen valores faltantes en alguna de sus columnas.
        """
        self.dataframe = self.dataframe.dropna(subset=[self.dataframe.columns[0],self.dataframe.columns[1],self.dataframe.columns[2]])        

    def normalize_data(self):
        """
        Normaliza los valores de la columna 'weight' para que estén en el rango [0, 1].
        """
        self.dataframe['weight'] = (self.dataframe[self.dataframe.columns[2]] - self.dataframe[self.dataframe.columns[2]].min()) / (self.dataframe[self.dataframe.columns[2]].max() - self.dataframe[self.dataframe.columns[2]].min())

