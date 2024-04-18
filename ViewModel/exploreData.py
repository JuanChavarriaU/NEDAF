import pandas as pd
import dask.dataframe as dd


class exploreData():

    def __init__(self, data: pd.DataFrame | dd.DataFrame):
        """Inicializa la clase con un DataFrame de pandas."""
        try:
            self.data = data
        except Exception as e:
            pass

    def get_summary_statistics(self):
        """Devuelve un resumen de estadísticas descriptivas para el DataFrame."""
        return self.data.describe(include='all')

    def calculate_mean(self):
        """Calcula y devuelve la media de cada columna numérica en el DataFrame."""
        return self.data.mean()

    def calculate_median(self):
        """Calcula y devuelve la mediana de cada columna numérica en el DataFrame."""
        return self.data.median()

    def calculate_variance(self):
        """Calcula y devuelve la varianza de cada columna numérica en el DataFrame."""
        return self.data.var()

    def calculate_covariance(self):
        """Calcula y devuelve la matriz de covarianza del DataFrame."""
        return self.data.cov()

    def calculate_correlation(self):
        """Calcula y devuelve la matriz de correlación del DataFrame."""
        return self.data.corr()

    def calculate_distribution(self):
        """Calcula y devuelve la distribución de frecuencias para cada columna categórica."""
        distribution = {}
        for column in self.data.select_dtypes(include=['object', 'category']):
            distribution[column] = self.data[column].value_counts()
        return distribution

    def get_unique_values(self):
        """Devuelve el número de valores únicos por columna."""
        return self.data.nunique()

    def get_missing_values(self):
        """Devuelve el número de valores faltantes por columna."""
        return self.data.isnull().sum()

    def calculate_standard_deviation(self):
        """Calcula y devuelve la desviación estándar de cada columna numérica en el DataFrame."""
        return self.data.std()

    def calculate_min_max(self):
        """Calcula y devuelve el valor mínimo y máximo de cada columna numérica en el DataFrame."""
        return {
            'min': self.data.min(),
            'max': self.data.max()
        }
    
    def isNumeric(self):
        """Determina si una columna es numérica o no"""
        return self.data.select_dtypes(include=['number']).columns