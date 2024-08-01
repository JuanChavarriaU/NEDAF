from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox, QProgressBar
from PyQt6.QtCore import (
  QObject,
  QRunnable,
  QThreadPool,
  QTimer,
  pyqtSignal,
  pyqtSlot,
)
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPen
import networkx as nx
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore
import numpy as np
import time
from View.DataManager import DataManager
from ViewModel import NetworkAnalysis as na 
from fa2_modified import ForceAtlas2
pg.setConfigOptions(antialias=True)
class NetworkVisualizationMod(QWidget):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        self.data_manager.data_loaded.connect(self.on_data_loaded)
        
        self.NetworkAnalysis = na.NetworkAnalysis()
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

      
        button.clicked.connect(self.visualize_networks)
              
      
        self.setLayout(layout)

    def visualize_networks(self):
        start_time = time.perf_counter()
        
        self.Graph = self.NetworkAnalysis.create_network_graph(self.data_manager.get_data())
        finished_time = time.perf_counter()
        print(f"Creacion de grafo nx tomó: {finished_time - start_time}s")
        # Calcular el layout
        start_time = time.perf_counter()
        
        finished_time = time.perf_counter()
        print(f"Creacion de grafo gt tomó: {finished_time - start_time}s")
        
        start_time = time.perf_counter()
        num_nodes = self.Graph.number_of_nodes()
        if num_nodes > 5000:
            positions = self.compute_large_layout(self.Graph)
        elif num_nodes >= 100:
            positions = self.compute_medium_layout(self.Graph)
        elif num_nodes < 100:
            positions = self.compute_small_layout(self.Graph)
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
            


            nodes = list(G.nodes())
            edges = list(G.edges())
            weights = nx.get_edge_attributes(G, 'w')
            if not weights:
                weights = {(u,v): 1 for u,v in edges}

            node_positions = np.array([positions[node] for node in nodes])

            adj = np.array([[nodes.index(source), nodes.index(target)] for source, target in edges])

             

            #print("Nodes:", nodes)
            #print("Edges:", edges)
            #print("Adjacency matrix:\n", adj)
            
        
            
            graph.setData(
                pos=node_positions, 
                adj=adj, 
                size=15, 
                symbol='o',
                pxMode=True,
                brush=pg.mkBrush('Purple'), 
                hoverable=True,
                useCache=False,
                pen=pg.mkPen(color='black', width=1)
            )
            self.figure.addItem(graph)


          

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


    def compute_large_layout(self, G: nx.Graph):
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

        return forceAtlas2.forceatlas2_networkx_layout(G, pos=None, iterations=100)


    def compute_medium_layout(self, G: nx.Graph):
        return nx.spring_layout(G, seed=42)  # Usar el algoritmo de Fruchterman-Reingold 
    
    def compute_small_layout(self, G: nx.Graph):
        return nx.kamada_kawai_layout(G)  # Usar KK Algo



