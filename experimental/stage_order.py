import numpy as np
from graph import Graph
from typing import List
from copy import deepcopy
from globals import configs
import itertools

def get_stage_order(G: Graph):
    """
        Given a partitioned and merged graph, find the best order among stages
    """

    if G is None:
        return [0]

    if len(G) == 1:
        return [1]

    def optimal_path_given_start(next, kway=2):

        def search_path(next, distance_selection):
            
            path_cum_eweights = 0
            path = []
            path.append(next)
            selection = 0

            while len(path) != len(G.xadj) - 1:
                adjacents = G.adj[G.xadj[next]: G.xadj[next + 1]].tolist()
                connection_weight : List = deepcopy(G.eweights[G.xadj[next]: G.xadj[next + 1]].tolist())

                for node in path:
                    if node in adjacents:
                        connection_weight[adjacents.index(node)] = 0

                distance_rank = np.argsort(connection_weight)[::-1]


                selected_adj_node_id = distance_rank[distance_selection[selection]]
                offset = 1
                while connection_weight[selected_adj_node_id] == 0 and offset <= len(connection_weight):

                    selected_adj_node_id = distance_rank[(distance_selection[selection] + offset) % len(distance_rank) ]
                    offset += 1
                
                if connection_weight[selected_adj_node_id] == 0:
                    return None
                
                selection += 1
                next = adjacents[selected_adj_node_id]
                path_cum_eweights += connection_weight[selected_adj_node_id]
                
                path.append(next)

            return path, path_cum_eweights
        
        kway = min(kway, len(G) - 1)
            
        possible_distance_selection = [item for item in itertools.product([i for i in range(kway)], repeat=len(G)-1)]

        possible_paths = [search_path(next, distance_selection) for distance_selection in possible_distance_selection]

        cum_weights = [possible_path[0] for possible_path in possible_paths if possible_path is not None]

        
        optimal_path = possible_paths[cum_weights.index(max(cum_weights))]

        return optimal_path

    cumulated_eweight = []
    
    paths = []

    for start in range(len(G)):
        next = start

        optimal_path = optimal_path_given_start(next, configs.kway)
        if optimal_path is not None:
            paths.append(optimal_path[0])
            cumulated_eweight.append(optimal_path[1])

    optimal_path = paths[np.argsort(cumulated_eweight)[0]]

    return optimal_path