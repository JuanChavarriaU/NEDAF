import pandas as pd
import unittest
from Model.transformationData import TransformationData
from Model.exploreData import exploreData
from View.NetworkVisualizationMod import NetworkVisualizationMod
from Model.NetworkAnalysis import NetworkAnalysis, networkStatistics, NetworkCommunities
import timeit
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
import statistics
import networkx as nx


class TestNetworkAnalysis(unittest.TestCase):

    def test_create_network_graph(self):
        data = pd.DataFrame({
            'source': [1, 2, 3],
            'destination': [2, 3, 1],
            'weight': [4.0, 5.0, 6.0]
        })
        G = NetworkAnalysis.create_network_graph(data)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 3)
        self.assertEqual(G[1][2]['weight'], 4.0)

    def test_compute_large_layout(self):
        G = nx.karate_club_graph()  # Un grafo de prueba estándar
        na = NetworkAnalysis()
        layout = na.compute_large_layout(G)
        self.assertEqual(len(layout), G.number_of_nodes())

    def test_compute_medium_layout(self):
        G = nx.karate_club_graph()
        na = NetworkAnalysis()
        layout = na.compute_medium_layout(G)
        self.assertEqual(len(layout), G.number_of_nodes())

    def test_compute_small_layout(self):
        G = nx.karate_club_graph()
        na = NetworkAnalysis()
        layout = na.compute_small_layout(G)
        self.assertEqual(len(layout), G.number_of_nodes())


class TestNetworkStatistics(unittest.TestCase):

    def setUp(self):
        self.G = nx.karate_club_graph()

    def test_numberofNodes(self):
        self.assertEqual(networkStatistics.numberofNodes(self.G), 34)

    def test_numberofEdges(self):
        self.assertEqual(networkStatistics.numberofEdges(self.G), 78)

    def test_maximumDegree(self):
        self.assertEqual(networkStatistics.maximumDegree(self.G), 17)

    def test_minumumDegree(self):
        self.assertEqual(networkStatistics.minumumDegree(self.G), 1)

    def test_averageDegree(self):
        self.assertAlmostEqual(round(networkStatistics.averageDegree(self.G), 3), 4.588)

    def test_assortativity(self):
        self.assertAlmostEqual(round(networkStatistics.assortativity(self.G),4), -0.4756)

    def test_numberOfTriangles(self):
        self.assertEqual(networkStatistics.numberOfTriangles(self.G), 135)

    def test_networkDensity(self):
        self.assertAlmostEqual(round(networkStatistics.networkDensity(self.G),3), 0.139)

    def test_networkDiameter(self):
        self.assertEqual(networkStatistics.networkDiameter(self.G), 5)

    def test_networkRadius(self):
        self.assertEqual(networkStatistics.networkRadius(self.G), 3)

    def test_networkAverageClustering(self):
        self.assertAlmostEqual(round(networkStatistics.networkAverageClustering(self.G), 3), 0.571)

    def test_networkAveragePathLength(self):
        self.assertAlmostEqual(round(networkStatistics.networkAveragePathLength(self.G),3), 2.408)

    def test_networkDegreeDistribution(self):
        degree_dist = networkStatistics.networkDegreeDistribution(self.G)
        self.assertEqual(len(degree_dist), 18)  # Tiene 18 grados diferentes

    def test_networkClusteringCoefficient(self):
        clustering_coeff = networkStatistics.networkClusteringCoefficient(self.G)
        self.assertIsInstance(clustering_coeff, dict)

class TestNetworkCommunities(unittest.TestCase):

    def setUp(self):
        self.G = nx.karate_club_graph()

    def test_networkCommunities(self):
        communities = NetworkCommunities.networkCommunities(self.G)
        self.assertGreaterEqual(len(communities), 2)

    def test_networkModularity(self):
        modularity = NetworkCommunities.networkModularity(self.G)
        self.assertGreaterEqual(modularity, 0)

    def test_NoOfCommunities(self):
        no_of_communities = NetworkCommunities.NoOfCommunities(self.G)
        self.assertGreaterEqual(no_of_communities, 2)

    def test_networkCommunitySize(self):
        community_sizes = NetworkCommunities.networkCommunitySize(self.G)
        self.assertTrue(all(size > 0 for size in community_sizes))

    def test_networkKeyNodes(self):
        key_nodes = NetworkCommunities.networkKeyNodes(self.G)
        self.assertTrue(isinstance(key_nodes, list))

    def test_communityLeaderNodes(self):
        leaders = NetworkCommunities.communityLeaderNodes(self.G)
        self.assertTrue(isinstance(leaders, list))

    def test_networkIsolates(self):
        isolates = NetworkCommunities.networkIsolates(self.G)
        self.assertEqual(len(isolates), 0)

    def test_networkDegreeCentrality(self):
        degree_centrality = NetworkCommunities.networkDegreeCentrality(self.G)
        self.assertIsInstance(degree_centrality, dict)

    def test_networkBetweennessCentrality(self):
        betweenness_centrality = NetworkCommunities.networkBetweennessCentrality(self.G)
        self.assertIsInstance(betweenness_centrality, dict)

    def test_networkClosenessCentrality(self):
        closeness_centrality = NetworkCommunities.networkClosenessCentrality(self.G)
        self.assertIsInstance(closeness_centrality, dict)

    def test_networkEigenvectorCentrality(self):
        eigenvector_centrality = NetworkCommunities.networkEigenvectorCentrality(self.G)
        self.assertIsInstance(eigenvector_centrality, dict)

    def test_networkPageRank(self):
        pagerank = NetworkCommunities.networkPageRank(self.G)
        self.assertIsInstance(pagerank, dict)


"""class TestTransformationData(unittest.TestCase):

    def test_normalize_data(self):
        # Datos de ejemplo
        data = {
            'source': [1, 2, 3, 4],
            'destination': [2, 3, 4, 1],
            'weight': [10, 20, 30, 40]
        }
        df = pd.DataFrame(data)
        
        # Crear una instancia de TransformationData
        transformation_data = TransformationData(df)
        
        # Llamar al método normalize_data
        transformation_data.normalize_data()

        # Verificar si los valores normalizados están dentro del rango esperado
        normalized_weights = transformation_data.dataframe['normalized_weight']
        
        self.assertGreaterEqual(normalized_weights.min(), 0, "Los valores normalizados no están en el rango esperado (min < 0)")
        self.assertLessEqual(normalized_weights.max(), 1, "Los valores normalizados no están en el rango esperado (max > 1)")



    def test_cut_missing_values(self):
        data = {
            'source': [1, 2, 3, 4],
            'destination': [2,None, 4, 1],
            'weight': [10, 20, 30, 40]
        }
        df = pd.DataFrame(data)

        transformation_data = TransformationData(df)
        
        transformation_data.cut_missing_values()

        result = transformation_data.get_data()

        expected = df.dropna()

        pd.testing.assert_frame_equal(result, expected)  

        
    if __name__ == '__main__':
        unittest.main()

  """

"""class TestExploreData(unittest.TestCase):
    
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

"""class TestPerformance():

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
    
    #large networks
    df3 = load_data("mtx")
    G = NetworkAnalysis.create_network_graph(df3)
    print(G)
    times = test_create_network_graph(df3)
    media= statistics.mean(times[0].values())
    
    times2 = test_compute_small_layout(G)
    media2 = statistics.mean(times2[0].values())
    
    times3 = test_load_data(load_data, 'csv')
    media3 = statistics.mean(times3[0].values())
    plot_times(times, f'Tiempo promedio de creacion de grafo: {media:.4f}s', "Rendimiento de la creacion de una red grande (5,749 nodos)",1) 
    plot_times(times2, f'Tiempo promedio de computacion de layout: {media2:.4f}s', "Rendimiento de la computación de layout de una red grande (5,749 nodos)", 3)
    plot_times(times3, f'Tiempo promedio de carga de datos: {media3:.4f}s', "Rendimiento de la carga de datos de una red grande (5,749 nodos)", 3)
"""