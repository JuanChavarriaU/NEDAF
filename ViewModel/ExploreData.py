from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QComboBox, 
                            QLabel, QPushButton, QStackedWidget)
from View.DataManager import DataManager    
from ViewModel.exploreData import exploreData

class ExploreData(QWidget):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        self.data_manager.data_loaded.connect(self.on_data_loaded) # no se habilitan las tablas una vez que los datos estan cargados 
        self.initUI()

    def initUI(self):
      """ Inicializa la interfaz de usuario."""
      self.column_dropdown = QComboBox()
      
      self.column_dropdown.currentIndexChanged.connect(self.on_column_changed)
      self.stacked_widget = QStackedWidget()

      self.operation_dropdown = QComboBox()
      self.operation_dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
      self.layout = QVBoxLayout()
      self.layout.addWidget(self.column_dropdown)
      self.layout.addWidget(self.operation_dropdown)
      self.layout.addWidget(self.stacked_widget)

      self.create_summary_tab()
      self.create_distribution_tab()
      self.create_correlation_tab()

      self.setLayout(self.layout)

      self.disable_enable_tables(False)  

    def create_summary_tab(self):
        """Crea la pestaña de resumen."""
        self.summary_tab = QWidget()
        self.summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_layout.addWidget(self.summary_table)
        self.summary_tab.setLayout(self.summary_layout)
        self.operation_dropdown.addItem("Resumen estadístico", self.summary_tab)
        self.stacked_widget.addWidget(self.summary_tab) 

    def create_distribution_tab(self):
        """Crea la pestaña de distribución."""
        self.distribution_tab = QWidget()
        self.distribution_layout = QVBoxLayout()
        self.distribution_table = QTableWidget()
        self.distribution_layout.addWidget(self.distribution_table)
        self.distribution_tab.setLayout(self.distribution_layout)
        self.operation_dropdown.addItem("Distribución", self.distribution_tab)
        self.stacked_widget.addWidget(self.distribution_tab)

    def create_correlation_tab(self):
        """Crea la pestaña de correlación."""
        self.correlation_tab = QWidget()
        self.correlation_layout = QVBoxLayout()
        self.correlation_table = QTableWidget()
        self.correlation_layout.addWidget(self.correlation_table)
        self.correlation_tab.setLayout(self.correlation_layout)
        self.operation_dropdown.addItem("Correlación", self.correlation_tab)
        self.stacked_widget.addWidget(self.correlation_tab)

    def on_dropdown_changed(self, index: int):
        selected_widget = self.operation_dropdown.currentData()
        self.stacked_widget.setCurrentWidget(selected_widget)
    
    def get_numeric_columns(self):
        """Devuelve las columnas numéricas del DataFrame."""
        return self.data_manager.get_data().select_dtypes(include=['number']).columns
        
    def on_column_changed(self, index: int):
        selected_column = self.column_dropdown.currentText()
        if selected_column in self.get_numeric_columns():
            self.operation_dropdown.clear()
            self.operation_dropdown.addItems(["Resumen estadístico", "Promedio", "Mediana", "Varianza", "Covarianza", "Correlación", "Distribución", "Desviación estándar", "Min y Max", "Valores únicos", "Número de valores faltantes", "Valores faltantes"])
        else:
            self.operation_dropdown.clear()
            self.operation_dropdown.addItems(["Valores únicos", "Número de valores faltantes", "Valores faltantes"])
    
    def disable_enable_tables(self, option: bool):
        """Deshabilita o habilita todas las tablas al inicio."""
        self.summary_table.setEnabled(option)
        self.distribution_table.setEnabled(option)
        self.correlation_table.setEnabled(option)


    def on_data_loaded(self):
        """Manejador para la señal de datos cargados."""
        # Habilitar tablas
        self.disable_enable_tables(True)
        self.column_dropdown.addItems(self.data_manager.get_data().columns)
        # Llenar las tablas con datos
        #self.update_tables_with_data()


    def update_tables_with_data(self):
        """Actualiza las tablas con datos cargados."""
        data = self.data_manager.get_data()

        # Si no hay datos, salir
        if data is None:
            return
        
        # Crear una instancia de exploreData con los datos
        explore_data = exploreData(data)

        # Llenar la tabla de resumen
        self.fill_table_with_summary(self.summary_table, explore_data)

        # Llenar la tabla de distribución
        self.fill_table_with_distribution(self.distribution_table, explore_data)

        # Llenar la tabla de correlación
        self.fill_table_with_correlation(self.correlation_table, explore_data)

    def fill_table_with_summary(self, table, explore_data: exploreData):
        """Llena la tabla de resumen con los datos de explore_data."""
        summary_stats = explore_data.get_summary_statistics()
        
        # Configurar la tabla de resumen con los datos
        # Establecer número de filas y columnas
        table.setColumnCount(len(summary_stats.columns))
        table.setRowCount(len(summary_stats.index))
        
        # Establecer etiquetas de columna y fila
        table.setHorizontalHeaderLabels(summary_stats.columns.astype(str))
        table.setVerticalHeaderLabels(summary_stats.index.astype(str))

    def fill_table_with_distribution(self, table, explore_data: exploreData):
        """Llena la tabla de distribución con los datos de explore_data."""
        distribution = explore_data.calculate_distribution()
        
        # Configurar la tabla de distribución con los datos
        # Establecer número de filas
        row_count = sum(len(value) for value in distribution.values())
        table.setRowCount(row_count)
        
        # Llenar la tabla con datos de distribution
        row = 0
        for column, freq_dist in distribution.items():
            for category, count in freq_dist.items():
                item_text = f"{column}: {category} = {count}"
                table.setItem(row, 0, QTableWidgetItem(item_text))
                row += 1
    
    def fill_table_with_correlation(self, table, explore_data: exploreData):
        """Llena la tabla de correlación con los datos de explore_data."""
        correlation = explore_data.calculate_correlation()
        
        # Configurar la tabla de correlación con los datos
        # Establecer número de filas y columnas
        table.setColumnCount(len(correlation.columns))
        table.setRowCount(len(correlation.index))
        
        # Establecer etiquetas de columna y fila
        table.setHorizontalHeaderLabels(correlation.columns.astype(str))
        table.setVerticalHeaderLabels(correlation.index.astype(str))
        
        # Llenar la tabla con datos de correlation
        for i, index in enumerate(correlation.index):
            for j, column in enumerate(correlation.columns):
                item = QTableWidgetItem(str(correlation.loc[index, column]))
                table.setItem(i, j, item)