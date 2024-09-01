import pandas as pd
import dask.dataframe as dd


class exploreData():

    def __init__(self):
        super().__init__()
       
    def set_data(self, data: pd.DataFrame | dd.DataFrame):
        """Establece el DataFrame de pandas."""
        try:
            self.data = data
        except Exception as e:
            print(f"Error al establecer el DataFrame: {e}") 

    def get_summary_statistics(self, selected_column: str) -> pd.Series:
        """Devuelve un resumen de estadísticas descriptivas para la columna seleccionada del DataFrame."""
        return self.data[selected_column].describe()

    def calculate_mean(self, selected_column: str) -> float:
        """Calcula y devuelve la media de cada columna numérica en el DataFrame."""
        return self.data[selected_column].mean()

    def calculate_median(self, selected_column: str) -> float:
        """Calcula y devuelve la mediana de cada columna numérica en el DataFrame."""
       
        return self.data[selected_column].median()

    def calculate_variance(self, selected_column: str) -> float:
        """Calcula y devuelve la varianza de cada columna numérica en el DataFrame."""
        
        return self.data[selected_column].var()

    def calculate_covariance(self, selected_column: str) -> pd.DataFrame:
        """Calcula y devuelve la matriz de covarianza del DataFrame."""
        
        return self.data.cov()

    def calculate_correlation(self, selected_column: str) -> pd.DataFrame: #añadir por cual metodo realizar la correlacion!!
        """Calcula y devuelve la matriz de correlación del DataFrame."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return self.data[selected_column].corr(self.data.drop(selected_column, axis=1))

    def calculate_distribution(self, selected_column: str) -> pd.Series:
        """Calcula y devuelve la distribución de frecuencias para cada columna categórica."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return self.data[selected_column].value_counts(sort=False)

    def get_unique_values(self, selected_column: str) -> int:
        """Devuelve el número de valores únicos por columna."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return self.data[selected_column].nunique()

    def get_missing_values(self, selected_column: str) -> int:
        """Devuelve el número de valores faltantes por columna."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")	
        return self.data[selected_column].isnull().sum()

    def calculate_standard_deviation(self, selected_column: str) -> float:
        """Calcula y devuelve la desviación estándar de cada columna numérica en el DataFrame."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return self.data[selected_column].std()

    def calculate_min_max(self, selected_column: str) -> pd.DataFrame:
        """Calcula y devuelve el valor mínimo y máximo de cada columna numérica en el DataFrame."""
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return pd.DataFrame.from_dict({
            'min': self.data[selected_column].min(),
            'max': self.data[selected_column].max()
        }, orient='index', columns=["Valor"])
    
    def isNumeric(self, selected_column: str) -> bool:
        """Determina si una columna es numérica o no""" 
        if selected_column not in self.data.columns:
            raise ValueError(f"Columna {selected_column} no existe en el dataframe")
        return self.data[selected_column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']