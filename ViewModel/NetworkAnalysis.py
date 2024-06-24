import networkx as nx 
import pandas as pd
import time
import graph_tool.all as gt
class NetworkAnalysis():
    
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        
        self.data = data
        
    def create_network_graph(self, data: pd.DataFrame) -> nx.Graph:
        """
        Crea un grafo a partir de un DataFrame.
    
        Parámetros:
        data (pd.DataFrame): Un DataFrame que contiene los datos de los bordes. Se espera que tenga al menos tres columnas, 
                             donde la primera columna es el nodo de origen, la segunda columna es el nodo de destino 
                             y la tercera columna es el peso de la arista.
    
        Devuelve:
        nx.Graph: Un grafo de NetworkX creado a partir de los datos del DataFrame.
        """
        G = nx.Graph()
        #start_time = time.perf_counter()
        for _, row in data.iterrows():
            source = row[data.columns[0]]
            target = row[data.columns[1]]
            weight = row[data.columns[2]]
            G.add_edge(source, target, weight=weight)
        #finish_time = time.perf_counter()
        #print(f"la generacion del grafo took:{finish_time-start_time}s")
        return G

    def create_network_graph_graph_tool(self, data: pd.DataFrame) -> gt.Graph:
        g = gt.Graph(directed=False)

        weight_map = g.new_edge_property("float")

        vertex_map = {} 

        for idx, row in data.iterrows():
            source, target, weight = row[data.columns[0]], row[data.columns[1]], row[data.columns[2]]

            if source not in vertex_map:
                vertex_map[source] = g.add_vertex()

            if target not in vertex_map:
                vertex_map[target] = g.add_vertex()

            edge = g.add_edge(vertex_map[source], vertex_map[target])
            weight_map[edge] = weight

        g.edge_properties["weight"] = weight_map

        return g        



class networkStatistics(NetworkAnalysis): 

    def networkDiameter(self, G: nx.Graph) -> int:
        """Return the diameter of the network"""
        return nx.diameter(G)
    
    def networkRadius(self, G: nx.Graph) -> int:
        """Calcula el radio de la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el radio.

        Devuelve:
        int: El radio del grafo.
        """
        return nx.radius(G)
    
    def networkAverageClustering(self, G: nx.Graph) -> float:
        """
        Calcula el coeficiente de agrupamiento promedio de la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el coeficiente de agrupamiento promedio.

        Devuelve:
        float: El coeficiente de agrupamiento promedio del grafo.
        """
        return nx.average_clustering(G)
    
    def networkAverageDegree(self, G: nx.Graph) -> dict:
        """
            Calcula la conectividad de grado promedio de la red.

            Parámetros:
            G (nx.Graph): El grafo del cual se calculará la conectividad de grado promedio.

            Devuelve:
            dict: Un diccionario donde las claves son los grados y los valores son la conectividad promedio de ese grado.
        """
        return nx.average_degree_connectivity(G)
    
    def networkAveragePathLength(self, G: nx.Graph) -> float:
        """
        Calcula la longitud promedio del camino más corto en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la longitud promedio del camino más corto.

        Devuelve:
        float: La longitud promedio del camino más corto del grafo.
        """
        return nx.average_shortest_path_length(G)
    
    def networkDegreeDistribution(self, G: nx.Graph) -> list:
        """
        Calcula la distribución de grados de la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la distribución de grados.

        Devuelve:
        list: Una lista donde el índice representa el grado y el valor en ese índice representa el número de nodos con ese grado.
        """
        return nx.degree_histogram(G)
    
    def networkClusteringCoefficient(self, G: nx.Graph) -> float | dict:
        """
        Calcula el coeficiente de agrupamiento de la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el coeficiente de agrupamiento.

        Devuelve:
        float | dict: Si el grafo es no dirigido, devuelve el coeficiente de agrupamiento promedio.
                      Si el grafo es dirigido, devuelve un diccionario con los coeficientes de agrupamiento de cada nodo.
        """
        return nx.clustering(G)
    

class NetworkCommunities(NetworkAnalysis):

    def networkCommunities(self, G: nx.Graph) -> list:
        """
            Detecta comunidades en la red utilizando el algoritmo de modularidad codiciosa.

            Parámetros:
            G (nx.Graph): El grafo del cual se detectarán las comunidades.

            Devuelve:
            list: Una lista de comunidades, donde cada comunidad es una lista de nodos.
            """
        return nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    
    def networkModularity(self, G: nx.Graph) -> float:
        """
        Calcula la modularidad de la red en base a las comunidades detectadas.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la modularidad.

        Devuelve:
        float: La modularidad del grafo.
        """
        return nx.algorithms.community.modularity(G, self.networkCommunities(G), weight='weight')
    
    def NoOfCommunities(self, G: nx.Graph) -> int:
        """
        Calcula el número de comunidades en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el número de comunidades.

        Devuelve:
        int: El número de comunidades en el grafo.
        """
        return len(self.networkCommunities(G))
    
    def networkCommunitySize(self, G: nx.Graph) -> list:
        """
        Calcula el tamaño de cada comunidad en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el tamaño de las comunidades.

        Devuelve:
        list: Una lista con los tamaños de cada comunidad.
        """
        return [len(c) for c in self.networkCommunities(G)]
    
    def networkCommunitySizeDistribution(self, G: nx.Graph):
        """
        Calcula la distribución del tamaño de las comunidades en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la distribución del tamaño de las comunidades.

        Devuelve:
        list: Una lista con los tamaños de cada comunidad.
        """
        return self.networkCommunitySize(G)
    
    #key nodes
    def networkKeyNodes(self, G: nx.Graph):
        """
        Identifica los nodos clave en la red que tienen un grado mayor a 10.

        Parámetros:
        G (nx.Graph): El grafo del cual se identificarán los nodos clave.

        Devuelve:
        list: Una lista de nodos clave con un grado mayor a 10.
        """
        return [n for n, d in G.degree() if d > 10]
    
    #isolates
    def networkIsolates(self, G: nx.Graph):
        """
        Identifica los nodos aislados en la red que tienen un grado igual a 0.

        Parámetros:
        G (nx.Graph): El grafo del cual se identificarán los nodos aislados.

        Devuelve:
        list: Una lista de nodos aislados con un grado igual a 0.
        """
        return [n for n, d in G.degree() if d == 0]
    
    #degree centrality
    def networkDegreeCentrality(self, G: nx.Graph) -> dict:
        """
        Calcula la centralidad de grado para cada nodo en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la centralidad de grado.

        Devuelve:
        dict: Un diccionario donde las claves son los nodos y los valores son sus centralidades de grado.
        """
        return nx.degree_centrality(G)
    
    #betweenness centrality
    def networkBetweennessCentrality(self, G: nx.Graph) -> dict:
        """
        Calcula la centralidad de intermediación para cada nodo en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la centralidad de intermediación.

        Devuelve:
        dict: Un diccionario donde las claves son los nodos y los valores son sus centralidades de intermediación.
        """
        return nx.betweenness_centrality(G, weight='weight')
    
    #closeness centrality
    def networkClosenessCentrality(self, G: nx.Graph) -> dict:
        """
        Calcula la centralidad de cercanía para cada nodo en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la centralidad de cercanía.

        Devuelve:
        dict: Un diccionario donde las claves son los nodos y los valores son sus centralidades de cercanía.
        """
        return nx.closeness_centrality(G, distance='weight')
    
    #eigenvector centrality
    def networkEigenvectorCentrality(self, G: nx.Graph) -> dict:
        """
        Calcula la centralidad de vector propio para cada nodo en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará la centralidad de vector propio.

        Devuelve:
        dict: Un diccionario donde las claves son los nodos y los valores son sus centralidades de vector propio.
        """
        return nx.eigenvector_centrality(G, weight='weight')
    
    #pagerank
    def networkPageRank(self, G: nx.Graph):
        """
        Calcula el PageRank para cada nodo en la red.

        Parámetros:
        G (nx.Graph): El grafo del cual se calculará el PageRank.

        Devuelve:
        dict: Un diccionario donde las claves son los nodos y los valores son sus PageRank.
        """
        return nx.pagerank(G, weight='weight')
    