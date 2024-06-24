from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox, QProgressBar
from PyQt6.QtCore import (
  QObject,
  QRunnable,
  QThreadPool,
  QTimer,
  pyqtSignal,
  pyqtSlot,
)
from PyQt6.QtCore import QPointF
import networkx as nx
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
import numpy as np
import time
from View.DataManager import DataManager
from ViewModel import NetworkAnalysis as na
import multiprocessing
import concurrent.futures
import graph_tool.all as gt
from fa2_modified import ForceAtlas2

class NetworkVisualizationMod(QWidget):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        self.NetworkAnalysis = na.NetworkAnalysis(self.data_manager)
        self.operation_to_function_map = []  # Populate with actual functions
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()  # canvas 

        # layout para la grafica
        self.figure = pg.GraphicsLayoutWidget()
        layout.addWidget(self.figure)

        # layout para el dropdown
        self.GraphBasicMetricsDropDown = QComboBox()
        layout.addWidget(self.GraphBasicMetricsDropDown)

        # layout para el label
        self.stats_label = QLabel()
        self.stats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setWindowTitle('Network Stats')
        layout.addWidget(self.stats_label)

        # layout para el boton
        button = QPushButton("Visualizar Red")
        layout.addWidget(button)

        self.progressBar = QProgressBar()
      
        button.clicked.connect(self.visualize_networks)
      
        layout.addWidget(self.progressBar)
      
        self.setLayout(layout)

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def execute(self):
        worker = Worker()
        worker.signals.progress.connect(self.update_progress)


        self.threadpool.start(worker)

    def update_progress(self, progress):
        self.progressBar.setValue(progress)


    def on_data_loaded(self) -> None:
        """Manejador para la señal de datos cargados."""
        self.GraphBasicMetricsDropDown.addItems(["Degree", "Betweenness Centrality"])

    def on_operation_changed(self, index: int, graph):
        """Manejador para el cambio de operación seleccionado en el `operation_dropdown`."""
        try:
            selected_operation = self.GraphBasicMetricsDropDown.currentText()
            if selected_operation in self.operation_to_function_map:
                self.operation_to_function_map[selected_operation](graph)
            else:
                QMessageBox.critical(self, "Error", f"Operación no soportada")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cambiando la operación {str(e)}")

    def compute_layout(self, G: nx.Graph):
        # toma 2 minutos y 11s en ejecutar. 
        forceAtlas2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,

            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.0,
            multiThreaded=False,

            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,

            verbose=True
        )

        positions = forceAtlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100) 
        return positions

    def visualize_networks(self):
        #self.execute()
        
        # Crear el grafo
        start_time = time.perf_counter()
        self.Graph = self.NetworkAnalysis.create_network_graph(self.data_manager.get_data())
        finished_time = time.perf_counter()
        print(f"Creacion de grafo nx tomó: {finished_time - start_time}s")
        # Calcular el layout
        start_time = time.perf_counter()
        self.Graph2 = self.NetworkAnalysis.create_network_graph_graph_tool(self.data_manager.get_data())
        finished_time = time.perf_counter()
        print(f"Creacion de grafo gt tomó: {finished_time - start_time}s")
        #with concurrent.futures.ProcessPoolExecutor() as exec:
        start_time = time.perf_counter()
        positions = self.compute_layout(self.Graph)
        finished_time = time.perf_counter()
        print(f"Calculo de compute_layout tomó: {finished_time - start_time}s")

        # Visualizar el grafo
        start_time_vn = time.perf_counter()
        self.plot_graph(self.Graph, positions)
        finished_time = time.perf_counter()
        print(f"Visualizar el grafo tomó: {finished_time - start_time_vn}s")
        

    def plot_graph(self, G: nx.Graph, positions):
      
        graph = pg.GraphItem()
        self.figure.clear()
        view = self.figure.addViewBox()
        view.setAspectLocked()
        view.addItem(graph)
        
        
        nodes = list(G.nodes())
        edges = list(G.edges())
        weights = nx.get_edge_attributes(G, 'w')

        node_positions = np.array([positions[node] for node in nodes])
        adj = np.zeros((len(nodes), len(nodes)), dtype=int)
       

       
        for source, target in edges:
            adj[nodes.index(source), nodes.index(target)] = 1
       
        
        graph.setData(pos=node_positions, adj=adj, size=5, symbol='o' ,pxMode=True)
        

     
        for (source, target), weight in weights.items():
            i, j = nodes.index(source), nodes.index(target)
            edge_center = (node_positions[i] + node_positions[j]) / 2
            text = pg.TextItem(text=str(weight), anchor=(0.5, 0.5))
            text.setPos(edge_center)
            view.addItem(text)
        
class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    progress
    int progress complete,from 0-100
    """
    progress = pyqtSignal(int)

class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handle worker thread execution and reusability
    """
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs
        """
        total_n = 1000
        for n in range(total_n):
            progress_pc = int(100 * n / total_n)
            self.signals.progress.emit(progress_pc)
            time.sleep(0.001)

    
    
