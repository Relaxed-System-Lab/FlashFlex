import numpy as np
import pymetis
from pymetis import Options
from graph import Graph
from typing import List
from copy import deepcopy
from globals import *
from strategy import *
from layer_partition import *
from stage_order import *
    
def recover_nodes(sub_parts, nodes_mapping):
    """
        nodes_mapping: [[orig], [current]]
    """
    recovered_parts = [[] for _ in range(max(sub_parts) + 1)]
    for i in range(len(sub_parts)):
        recovered_parts[sub_parts[i]].append(nodes_mapping[0][i])
    
    return recovered_parts


def get_connection(parts, G: Graph):

    connection = [[] for _ in range(max(parts))]
    connection_weights = [[] for _ in range(max(parts))]
    
    subgraph_nodes = []
    for i in range(max(parts)):
        part: List = get_nodes(parts, i)
        subgraph_nodes.append(part)
    
    for i in range(len(subgraph_nodes)):
        for j in range(len(subgraph_nodes)):
            if i == j:
                continue
            else:
                adjacent_pos = np.argwhere(np.isin(subgraph_nodes[j], G.adj[G.xadj[i]: G.xadj[i + 1]])).ravel()
                weights = np.array(G.eweights[G.xadj[i]: G.xadj[i + 1]])[adjacent_pos]
                if sum(adjacent_pos) != 0:
                    connection[i].append(j)

                    connection_weights[i].append(sum(weights))
    
    return connection, connection_weights


def restore(coarsened_partition: List, nodes_mapping, ):
    nodes_num = len(nodes_mapping[0])
    uncoarsened_partition = [0 for _ in range(nodes_num)]
    for i in range(len(coarsened_partition)):
        for node in nodes_mapping[i]:
            uncoarsened_partition[node] = coarsened_partition[i]
    
    return uncoarsened_partition


def create_subgraphs(G: Graph, parts):
    """
        Given global graph and partition, create all the sub graphs
    """

    global configs

    nparts = max(parts) + 1
    sub_graphs = []
    parts = np.array(parts)
    all_nodes_mapping = []

    for n in range(nparts):
        # retrieve nodes in this subgraph
        part: List = get_nodes(parts, n)

        nodes_mapping = [[], []]
        
        adj = []
        xadj = []
        eweights = []
        start = 0
        for i in part:
            nodes_mapping[0].append(i)
            nodes_mapping[1].append(part.index(i))
            xadj.append(start)
            for j in range(G.xadj[i], G.xadj[i + 1]):
                if G.adj[j] in part:
                    adj.append(part.index(G.adj[j]))
                    
                    eweights.append(G.eweights[j])
                    start += 1
        xadj.append(start)

        vweights = G.vweights[part]

        # get adj, xadj, eweights correctly
        # when creating new subgraphs, need to make sure
        G_n = Graph(adj=np.array(adj), xadj=np.array(xadj), eweights=np.array(eweights), vweights=vweights, options=configs.options)
        sub_graphs.append(G_n)
    
        all_nodes_mapping.append(nodes_mapping)

    return sub_graphs, all_nodes_mapping

def initial_partition(G):
    """
        partition and create a series of graphs
    """
    return partitioner(G)

def get_nodes(parts, n):

    return np.argwhere(np.array(parts) == n).ravel().tolist()


def partitioner(G: Graph, nparts):
    """
        using pymeits api to partition a given graph
    """
    assert nparts != 0
    if nparts >= len(G):
        nparts = len(G) 
        

    (edgecuts,parts)=pymetis.part_graph(nparts=nparts,
                                        adjncy=G.adj,
                                        xadj=G.xadj,
                                        eweights=G.eweights, 
                                        vweights=G.vweights, 
                                        options=G.options)
    
    for e in range(max(parts)):
        try:
            assert e in parts
        except AssertionError:
            parts_ids = sorted(list((set(parts))))
            updated_parts = []
            for part in parts:
                updated_parts.append(parts_ids.index(part))
            
            print("Updated parts:", updated_parts)
            parts = updated_parts
            
            break


    return parts    # parts is a list, representing which part does a vertex belongs to


def repartition(all_sub_recovered_parts, G, npipeline):

    if npipeline <= 1:
        return None

    merged_G, subsub_nodes_mapping = merge_graph(all_sub_recovered_parts, G)

    # updata parts
    next_coarsened_parts = partitioner(merged_G, npipeline)


    next_parts = uncoarsen(next_coarsened_parts, subsub_nodes_mapping)

    return next_parts

def create_adj_xadj_empty_weights(nparts):
    new_adj = np.array([[i for i in range(nparts) if i != j] for j in range(nparts)]).flatten()
    new_xadj = [i for i in range(0, (nparts - 1) * (nparts + 1), nparts - 1)]

    new_eweights = [0 for i in range(nparts * (nparts - 1))]

    return new_adj, new_xadj, new_eweights

def merge_subgraph(parts, G: Graph):
    """
        Given a graph and partition on it, merge every part as a node
    """
    nparts = max(parts) + 1

    if nparts == 1:
        return None
    # nparts = 5

    new_adj, new_xadj, new_eweights = create_adj_xadj_empty_weights(nparts)

    new_vweights = []

    all_nodes = []

    def get_pos(cur_node, adj_node):

        return new_adj[new_xadj[cur_node]: new_xadj[cur_node + 1]].tolist().index(adj_node)
    
    for i in range(max(parts) + 1):
        all_nodes.append(get_nodes(parts, i))

    for j in range(len(all_nodes)):    
        nodes = all_nodes[j]
        vweights = G.vweights[nodes]
        new_vweights.append(sum(vweights))

        # assume the graph is connected
        for node in nodes:
            adjacents = G.adj[G.xadj[node]: G.xadj[node + 1]]
            adjacent_weights = G.eweights[G.xadj[node]: G.xadj[node + 1]]
            for i in range(len(adjacents)):
                
                if parts[adjacents[i]] != parts[node]:
                    offset = get_pos(parts[node], parts[adjacents[i]])
                    # offset = i
                    new_eweights[new_xadj[j]  + offset] += adjacent_weights[i]

    G_merged = Graph(adj=new_adj, xadj=new_xadj, eweights=new_eweights, vweights=new_vweights)
    
    return G_merged


def merge_graph(all_recovered_parts, G: Graph):

    nodes_mapping = []
    

    new_adj = []
    new_xadj = []
    new_eweights = []
    new_vweights = []

    node_id = 0
    for i in range(len(all_recovered_parts)):
        node_mapping = []
        for j in range(len(all_recovered_parts[i])):
            recovered_parts = all_recovered_parts[i][j]
            node_mapping.append(node_id)
            node_id += 1

        nodes_mapping.append(node_mapping)

    nodes_mapping = [all_recovered_parts, nodes_mapping]
    

    def get_subpart_pos(cur_node, node, new_adj, new_xadj):

        def get_cur_pos(cur_node):
            pos = 0
            for i in range(len(all_recovered_parts)):
                for j in range(len(all_recovered_parts[i])):
                    if cur_node in all_recovered_parts[i][j]:
                        return pos
                    else:
                        pos += 1
        pos = get_cur_pos(cur_node)
        adj_pos = get_cur_pos(node)

        return new_adj[new_xadj[pos]: new_xadj[pos + 1]].tolist().index(adj_pos), pos

    nnodes = node_id

    new_adj, new_xadj, new_eweights = create_adj_xadj_empty_weights(nnodes)

    new_vweights = []

    for i in range(len(all_recovered_parts)):
        recovered_parts = all_recovered_parts[i]
        for sub_part in recovered_parts:
            orig_vweights = G.vweights[sub_part]

            new_vweights.append(sum(orig_vweights))

            # assume the graph is connected
            for node in sub_part:
                adjacents = G.adj[G.xadj[node]: G.xadj[node + 1]]
                adjacent_weights = G.eweights[G.xadj[node]: G.xadj[node + 1]]
                for i in range(len(adjacents)):
                    
                    if adjacents[i] not in sub_part:
                        pos, cur_part = get_subpart_pos(node, adjacents[i], new_adj, new_xadj)
                        new_eweights[new_xadj[cur_part] + pos] += adjacent_weights[i]


    G_merged = Graph(adj=new_adj, xadj=new_xadj, eweights=new_eweights, vweights=new_vweights)
    
    return G_merged, nodes_mapping    

def uncoarsen(parts, nodes_mapping):

    all_subsub_parts = []
    for i in range(len(nodes_mapping[0])):
        for j in range(len(nodes_mapping[0][i])):
            all_subsub_parts.append(nodes_mapping[0][i][j])

    uncoar_parts = []
    for i in range(len(parts)):
        for orig_node in all_subsub_parts[i]:
            uncoar_parts.append(parts[i])

    return uncoar_parts


    

def partition_pipeline(G, parts, npipeline, i_iter):
    """
        Given global partition, partition each subgraph
    """
    global configs

    args = get_args()

    all_sub_recovered_parts = []
    all_pipelines = []
    sub_graphs, all_nodes_mapping = create_subgraphs(G, parts)

    for j in range(npipeline):

        sub_graphs[j].options = configs.options
        

        sub_parts = partitioner(sub_graphs[j], configs.K[ j])

        recovered_parts = recover_nodes(sub_parts, all_nodes_mapping[j])


        all_sub_recovered_parts.append(recovered_parts)
        
        merged_graph = merge_subgraph(sub_parts, sub_graphs[j])

        path = get_stage_order(merged_graph)

        # strategies for all pipelines, integrate path into strategy at the same time
        strategy = gen_strategy(recovered_parts, path)

        # layer partition
        layer_partition, memory_view = create_layer_partition(strategy)

        # update K

        if layer_partition is not None:
            all_pipelines.append([strategy, layer_partition, memory_view])
        else:
            all_pipelines.append(None)

    
    return all_pipelines, all_sub_recovered_parts