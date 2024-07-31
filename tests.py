import pandas as pd
import unittest
#from Model.transformationData import TransformationData
#from ViewModel.exploreData import exploreData
#from ViewModel.NetworkVisualizationMod import NetworkVisualizationMod
#from ViewModel.NetworkAnalysis import NetworkAnalysis
import timeit
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
import statistics

"""def test_normalize_data():
    # Create a sample DataFrame
    data = {
        'source': [1, 2, 3, 4],
        'destination': [2, 3, 4, 1],
        'weight': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)

    # Create an instance of TransformationData
    transformation_data = TransformationData(df)

    # Call the normalize_data method
    transformation_data.normalize_data()

    # Check if the normalized values are within the expected range
    normalized_weights = transformation_data.dataframe['normalized_weight']
    assert normalized_weights.min() >= 0 and normalized_weights.max() <= 1, "Normalized values are not in the expected range"
    print("Test passed")


class TestExploreData(unittest.TestCase):
    
    
    def setUp(self):
        
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        # Create an instance of ExploreData
        self.explore_data = exploreData()
        self.explore_data.set_data(self.data)   
    
    def test_get_unique_values(self):
        # Create a sample DataFrame
       
        # Test the get_unique_values method
        unique_values = self.explore_data.get_unique_values('A')
        self.assertEqual(unique_values, [1, 2, 3, 4, 5])
        print("Test passed")

    def test_get_summary_statistics(self):
        result = self.explore_data.get_summary_statistics('A')
        expected = self.data['A'].describe(include='all')
        pd.testing.assert_frame_equal(result, expected)

    def test_calculate_mean(self):
        result = self.explore_data.calculate_mean('A')
        expected = self.data['A'].mean()
        self.assertEqual(result, expected)

    def test_calculate_median(self):
        result = self.explore_data.calculate_median('C')
        expected = self.data['C'].median()
        self.assertEqual(result, expected)

    def test_calculate_variance(self):
        result = self.explore_data.calculate_variance('C')
        expected = self.data['C'].var()
        self.assertEqual(result, expected)    

    def test_calculate_covariance(self):
        result = self.explore_data.calculate_covariance('C')
        expected = self.data['C'].cov()
        self.assertEqual(result, expected)   

    def test_calculate_correlation(self):
        result = self.explore_data.calculate_correlation('C')
        expected = self.data['C'].corr(self.data.drop('C', axis=1))
        pd.testing.assert_frame_equal(result, expected)
    
    def test_calculate_distribution(self):
        result = self.explore_data.calculate_distribution()
        expected = {col: self.data[col].value_counts() for col in self.data.select_dtypes(include=['object', 'category'])}
        self.assertEqual(result, expected)

    def test_get_unique_values(self):
        result = self.explore_data.get_unique_values('B')
        expected = self.data['B'].nunique()
        self.assertEqual(result, expected)    

    def test_get_missing_values(self):
        result = self.explore_data.get_missing_values('B')
        expected = self.data['B'].isnull().sum()
        self.assertEqual(result, expected)    

    def test_calculate_standard_deviation(self):
        result = self.explore_data.calculate_standard_deviation('C')
        expected = self.data['C'].std()
        self.assertEqual(result, expected)    

    def test_calculate_min_max(self):
        result = self.explore_data.calculate_min_max('C')
        expected = {'min': self.data['C'].min(), 'max': self.data['C'].max()}
        self.assertEqual(result, expected)    

"""
"""
class TestPerformance():

    def __init__(self) -> None:
        super.__init__()


    def load_data(format):
        
        if format == 'mtx':
            matrixData = mmread('/home/vscode/soc-karate.mtx')
            rows, cols = matrixData.nonzero()
            if isinstance(matrixData, csr_matrix):
                weights = matrixData.data
                df = pd.DataFrame({'source': rows, 'destination': cols, 'weight': weights})
            else:
                df = pd.DataFrame({'source': rows, 'destination': cols}) 
        elif format == 'edges':
            df = pd.read_csv('/home/vscode/bio-CE-LC.edges', sep=' ', header=None, names=["source", "destination", "weight"] or ["source", "destination"])
        else:
            df = pd.read_csv('/home/vscode/od_cvegeo_09_01_2020_12_24.csv')
        
        return df
    
    def test_create_network_graph(df): 
        times = [{}]
        for i in range(1, 51, 1): #1500 iteraciones algoritmo KK 34 nodos
          times[0][i] = (timeit.timeit(lambda: NetworkAnalysis.create_network_graph(df), number=1))

        
        return times
    
    def test_compute_small_layout(G): 
        times = [{}]
        for i in range(1, 51, 1): 
          times[0][i] = (timeit.timeit(lambda: NetworkVisualizationMod.compute_small_layout(G), number=1))

        
        return times

    def test_compute_medium_layout(G): 
        times = [{}]
        for i in range(1, 51, 1): 
          times[0][i] = (timeit.timeit(lambda: NetworkVisualizationMod.compute_medium_layout(G), number=1))

        
        return times   

    def test_compute_large_layout(G): 
        times = [{}]
        for i in range(1, 50, 1): 
            times[0][i] = (timeit.timeit(lambda: NetworkVisualizationMod.compute_large_layout(G), number=1))
        
        
        return times   
      

    def test_load_data(load_data, format:str):
        times = [{}]
        for i in range(1, 50, 1): 
          times[0][i] = (timeit.timeit(lambda: load_data(format), number=1))

        return times    

    
    def plot_times(times: dict, legend, title, upperLimit):
        fig, ax = plt.subplots()
        ax.plot(times[0].keys(), times[0].values(), label=legend)  
        
        plt.grid()
        plt.title(title)
        plt.legend([legend])
        plt.xlabel('Iterations')
        plt.ylabel('Time (s)')
        #plt.ylim(0, upperLimit)
        plt.show()  
        
         
            
    #Small networks
    df = load_data("mtx")
    times = test_create_network_graph(df)
    media= statistics.mean(times[0].values())
    times2 = test_compute_small_layout(NetworkAnalysis.create_network_graph(df))
    media2 = statistics.mean(times2[0].values())
    times3 = test_load_data(load_data, 'mtx')
    media3 = statistics.mean(times3[0].values())
    plot_times(times, f'Tiempo promedio de creacion de grafo: {media:.4f}s', "Rendimiento de la creacion de una red pequeña (34 nodos)", 1) 
    plot_times(times2, f'Tiempo promedio de computacion de layout: {media2:.4f}s', "Rendimiento de la computación de layout de una red pequeña (34 nodos)", 2)
    plot_times(times3, f'Tiempo promedio de carga de datos: {media3:.4f}s', "Rendimiento de la carga de una red pequeña (34 nodos)", 1)
   
    #Medium networks 
    df2 = load_data("edges")
    G = NetworkAnalysis.create_network_graph(df2)
    times = test_create_network_graph(df2)
    media= statistics.mean(times[0].values())
    times3 = test_compute_large_layout(G)
    times2 = test_compute_medium_layout(G)
    media2 = statistics.mean(times2[0].values())
    media3 = statistics.mean(times3[0].values())
    times3 = test_load_data(load_data, 'edges')
    media3 = statistics.mean(times3[0].values())
    plot_times(times, f'Tiempo promedio de creacion de grafo: {media:.4f}s', "Rendimiento de la creacion de una red mediana (1,400 nodos)",1) 
    plot_times(times2, f'Tiempo promedio de computacion de layout: {media2:.4f}s', "Rendimiento de la computación de layout de una red mediana (1,400 nodos)", 3)
    plot_times(times3, f'Tiempo promedio de carga de datos: {media3:.4f}s', "Rendimiento de la carga de datos de una red mediana (1,400 nodos)", 3)
    
    """

    #df3 = load_data("mtx")
    #G = NetworkAnalysis.create_network_graph(df3)
    #print(G)
    #times = test_create_network_graph(df3)
    #media= statistics.mean(times[0].values())
    
    #times2 = test_compute_small_layout(G)
    #media2 = statistics.mean(times2[0].values())
    #print(media2)
    #times3 = test_load_data(load_data, 'csv')
    #media3 = statistics.mean(times3[0].values())
    #plot_times(times, f'Tiempo promedio de creacion de grafo: {media:.4f}s', "Rendimiento de la creacion de una red grande (5,749 nodos)",1) 
    #plot_times(times2, f'Tiempo promedio de computacion de layout: {media2:.4f}s', "Rendimiento de la computación de layout de una red grande (5,749 nodos)", 3)
    #plot_times(times3, f'Tiempo promedio de carga de datos: {media3:.4f}s', "Rendimiento de la carga de datos de una red grande (5,749 nodos)", 3)
    # -*- coding: utf-8 -*-
"""
Simple example of subclassing GraphItem.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from netgraph import Graph, InteractiveGraph, EditableGraph


# 4) networkx Graph and DiGraph objects (MultiGraph objects are not supported, yet)
import networkx
graph_data = networkx.karate_club_graph()


# Create a non-interactive plot:
Graph(graph_data)
plt.show()

# Create an interactive plot, in which the nodes can be re-positioned with the mouse.
# NOTE: you must retain a reference to the plot instance!
# Otherwise, the plot instance will be garbage collected after the initial draw
# and you won't be able to move the plot elements around.
# For related reasons, if you are using PyCharm, you have to execute the code in
# a console (Alt+Shift+E).
plot_instance = InteractiveGraph(graph_data)
plt.show()

# Create an editable plot, which is an interactive plot with the additions
# that nodes and edges can be inserted or deleted, and labels and annotations
# can be created, edited, or deleted as well.
plot_instance = EditableGraph(graph_data)
plt.show()

# Netgraph uses Matplotlib for creating the visualisation.
# Node and edge artistis are derived from `matplotlib.patches.PathPatch`.
# Node and edge labels are `matplotlib.text.Text` instances.
# Standard matplotlib syntax applies.
fig, ax = plt.subplots(figsize=(10,10))
plot_instance = Graph([(0, 1)], node_labels=True, edge_labels=True, ax=ax)
plot_instance.node_artists[0].set_alpha(0.2)
plot_instance.edge_artists[(0, 1)].set_facecolor('red')
plot_instance.edge_label_artists[(0, 1)].set_style('italic')
plot_instance.node_label_artists[1].set_size(10)
ax.set_title("This is my fancy title.")
ax.set_facecolor('honeydew') # change background color
fig.canvas.draw() # force redraw to display changes
fig.savefig('test.pdf', dpi=300)
plt.show()
