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
        self.NetworkCommunities = na.NetworkCommunities()
        self.operation_to_function_map = {
            "Number of Nodes":na.networkStatistics.numberofNodes,
            "Number of Edges": na.networkStatistics.numberofEdges,
            "Maximum Degree": na.networkStatistics.maximumDegree,
            "Minimum Degree": na.networkStatistics.minumumDegree,
            "Average Degree": na.networkStatistics.averageDegree,
            "Assortativity": na.networkStatistics.assortativity,
            "Number of triangles": na.networkStatistics.numberOfTriangles,
            "Network Degree" : na.networkStatistics.networkDegree,
            "Network Density": na.networkStatistics.networkDensity,
            "Network Diameter": na.networkStatistics.networkDiameter, 
            "Network Radius": na.networkStatistics.networkRadius,
            "Network Average Clustering": na.networkStatistics.networkAverageClustering,
            "Network Average Degree Conectivity": na.networkStatistics.networkAverageDegreeConectivity,
            "Network Average Path Length": na.networkStatistics.networkAveragePathLength,
            "Network Degree Distribution": na.networkStatistics.networkDegreeDistribution,
            "Network Clustering Coefficient": na.networkStatistics.networkClusteringCoefficient,
            "Network Communities": na.NetworkCommunities.networkCommunities,
            "Network Modularity": na.NetworkCommunities.networkModularity,
            "Number of Communities": na.NetworkCommunities.NoOfCommunities,
            "Network Community Size": na.NetworkCommunities.networkCommunitySize,
            "Network Key Nodes": na.NetworkCommunities.networkKeyNodes,
            "Network Isolates": na.NetworkCommunities.networkIsolates, 
            "Network Degree Centrality": na.NetworkCommunities.networkDegreeCentrality,
            "Network Betweenness Centrality": na.NetworkCommunities.networkBetweennessCentrality,
            "Network Closeness Centrality": na.NetworkCommunities.networkClosenessCentrality,
            "Network Eigenvector Centrality": na.NetworkCommunities.networkEigenvectorCentrality,
            "Network PageRank": na.NetworkCommunities.networkPageRank
            
          }  # Populate with actual functions
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()  # canvas 

        # layout para la grafica
        self.figure = pg.PlotWidget()
        self.figure.setBackground('w')
        layout.addWidget(self.figure)

        # layout para el dropdown
        self.GraphBasicMetricsDropDown = QComboBox()
        self.GraphBasicMetricsDropDown.activated.connect(self.on_operation_changed)

        layout.addWidget(self.GraphBasicMetricsDropDown)

        # layout para el label
        self.stats_label = QLabel()
        self.stats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setWindowTitle('Network Stats')
        layout.addWidget(self.stats_label)

        # layout para el boton
        button = QPushButton("Visualizar Red")
        layout.addWidget(button)

        #self.progressBar = QProgressBar()
      
        button.clicked.connect(self.visualize_networks)
      
        #layout.addWidget(self.progressBar)
      
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
        self.GraphBasicMetricsDropDown.clear()  # Limpiar el dropdown
        self.GraphBasicMetricsDropDown.addItems([
                                                 "Number of Nodes",
                                                 "Number of Edges",
                                                 "Maximum Degree",
                                                 "Minimum Degree",
                                                 "Average Degree",
                                                 "Assortativity",
                                                 "Number of triangles",
                                                 "Network Degree",
                                                 "Network Density",
                                                 "Network Diameter", 
                                                 "Network Radius",
                                                 "Network Average Clustering",
                                                 "Network Average Degree Conectivity",
                                                 "Network Average Path Length",
                                                 "Network Degree Distribution",
                                                 "Network Clustering Coefficient",
                                                 "Network Communities",
                                                 "Network Modularity",
                                                 "Number of Communities",
                                                 "Network Community Size",
                                                 "Network Key Nodes",
                                                 "Network Isolates", 
                                                 "Network Degree Centrality",
                                                 "Network Betweenness Centrality",
                                                 "Network Closeness Centrality",
                                                 "Network Eigenvector Centrality",
                                                 "Network PageRank"])

    def on_operation_changed(self, index: int):
        """Manejador para el cambio de operación seleccionado en el `graphbasicmetrics_dropdown`."""
        try:
            selected_operation = self.GraphBasicMetricsDropDown.currentText()
            print(selected_operation)
            if selected_operation in self.operation_to_function_map:

                functionSelected = self.operation_to_function_map[selected_operation]
                
                result = functionSelected(self.Graph)
                print(f"{selected_operation}: {result}")
                self.stats_label.setText(str(result))

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
        
        
        #self.Graph2 = self.NetworkAnalysis.create_network_graph_graph_tool(self.data_manager.get_data())
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
      
        
        self.figure.clear()
        #view = self.figure.addViewBox()
        #view.setAspectLocked()

        

        graph = pg.GraphItem()
        self.figure.addItem(graph)
        
        
        nodes = list(G.nodes())
        edges = list(G.edges())
        weights = nx.get_edge_attributes(G, 'w')

        node_positions = np.array([positions[node] for node in nodes])

        adj = np.zeros((len(nodes), len(nodes)), dtype=int)
      
        for source, target in edges:
            adj[nodes.index(source), nodes.index(target)] = 1

        np.savetxt("adj_matrix.txt", adj, fmt='%d')    
        
        graph.setData(pos=node_positions, adj=adj, size=15, symbol='o' ,pxMode=True, pen=pg.mkPen(color='black', width=1))
     
        for (source, target), weight in weights.items():
            i, j = nodes.index(source), nodes.index(target)
            edge_center = (node_positions[i] + node_positions[j]) / 2
            text = pg.TextItem(text=str(weight), anchor=(0.5, 0.5))
            text.setPos(edge_center)
            self.figure.addItem(text)
        
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

    
    
