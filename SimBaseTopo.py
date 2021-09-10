# generate schema from given
from pickle5 import pickle
from Schema import *
import networkx as nx
from ElementType import *
from groundtruth import *
from GraphGenerator import *
import random
import pandas as pd
from scipy.sparse import lil_matrix
# sc_topo = pickle.load(open('snb_gt_topological.pkl','rb',-1))
# compute similarity based on nodes
# object is triple
def computeSim(list1, list2, sim_matrix, method=''):
    # exact match
    SIM_MAX_VAL = sim_matrix[0][0]
    if method == 'jaccard':
        return jaccard(list1, list2)
    else:
        sim_i, sim_j = [], []
        for i in list1:
            max_sim = 0
            for j in list2:
                max_sim = max(max_sim, sim_matrix[i, j])
            sim_i.append(max_sim)
        for j in list2:
            max_sim = 0
            for i in list1:
                max_sim = max(max_sim, sim_matrix[i, j])
            sim_j.append(max_sim)
        sim_i = sum(sim_i) / len(sim_i) if len(sim_i) != 0 else SIM_MAX_VAL
        sim_j = sum(sim_j) / len(sim_j) if len(sim_j) != 0 else SIM_MAX_VAL
        return min(sim_i, sim_j)

#  The range of output matrix is the same as the input matrix
def getTopoEdgeSim(G, node_sim_matrix, edge_sim_matrix, emap, coef=[1, 1, 1]):
    # (source_nodes, edge, target_nodes)
    sim_matrix = np.zeros((len(G.edges), len(G.edges)))
    keys = list(emap.keys())
    # for i in range(len(keys)):
    #     sim_matrix[i,i] = edge_sim_matrix[0,0]
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            ei, ej = keys[i], keys[j]
            src_sim = node_sim_matrix[ei[0], ej[0]]
            tar_sim = node_sim_matrix[ei[1], ej[1]]
            # edge_sim = edge_sim_matrix[emap[ei], emap[ej]]
            sim_matrix[emap[ei], emap[ej]] = sim_matrix[emap[ej], emap[ei]] \
                = sum([src_sim, tar_sim]) / 2.0  # TODO can be other methods
    return sim_matrix


# assume nodes can be index from 0 to len(G.nodes)
# the range of result matrix is the same as the range of edge_sim_matrix
def getTopoNodeSim(G, node_sim_matrix, edge_sim_matrix, emap, **kwargs):
    sim_matrix = np.zeros((len(G.nodes), len(G.nodes)))
    # for i in range(len(G.nodes)):
    #     sim_matrix[i,i] = node_sim_matrix[0,0]
    method = kwargs.get('method','jaccard')

    for ni in range(len(G.nodes)):
        for nj in range(ni, len(G.nodes)):
            in_edges_i = [emap[i] for i in G.in_edges(ni,keys=True)]
            in_edges_j = [emap[i] for i in G.in_edges(nj,keys=True)]
            out_edges_i = [emap[i] for i in G.out_edges(ni,keys=True)]
            out_edges_j = [emap[i] for i in G.out_edges(nj,keys=True)]
            sim_matrix[ni, nj] = sim_matrix[nj, ni] = ( computeSim(in_edges_i,
                                    in_edges_j, edge_sim_matrix, method) + \
                computeSim(out_edges_i,out_edges_j,edge_sim_matrix,method) ) / 2
    return sim_matrix

def getNodeSimilarityMatrixByAttr(G,mode=''):
    print('getting node matrix with mode ', mode)
    node_sim_matrix = np.ones((len(G.nodes), len(G.nodes)))
    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            if mode=='l':
                attr_i = set(filter(lambda k:type(G.nodes[i][k])==LBL , G.nodes[i]))
                attr_j = set(filter(lambda k:type(G.nodes[j][k])==LBL , G.nodes[j]))
            elif mode=='p':
                attr_i = set(filter(lambda k:type(G.nodes[i][k])!=LBL , G.nodes[i]))
                attr_j = set(filter(lambda k:type(G.nodes[j][k])!=LBL , G.nodes[j]))
            else:
                attr_i, attr_j = G.nodes[i].keys(), G.nodes[j].keys()
            node_sim_matrix[i,j] = jaccard(attr_i, attr_j)
    return node_sim_matrix

def getEdgeSimilarityMatrixByAttr(G, mode=''):
    print('getting edge matrix with mode ', mode)
    edge_sim_matrix = np.ones((len(G.edges), len(G.edges)))
    for i in tqdm(range(len(G.edges))):
        for j in range(len(G.edges)):
            if mode=='l':
                attr_i = set(filter(lambda k:type(G.edges[list(G.edges)[i]][k])==LBL , G.edges[list(G.edges)[i]]))
                attr_j = set(filter(lambda k:type(G.edges[list(G.edges)[j]][k])==LBL , G.edges[list(G.edges)[j]]))
            elif mode=='p':
                attr_i = set(filter(lambda k:type(G.edges[list(G.edges)[i]][k])!=LBL , G.edges[list(G.edges)[i]]))
                attr_j = set(filter(lambda k:type(G.edges[list(G.edges)[j]][k])!=LBL , G.edges[list(G.edges)[j]]))
            else:
                attr_i, attr_j = G.edges[list(G.edges)[i]].keys(), G.edges[list(G.edges)[j]].keys()
            edge_sim_matrix[i,j] = jaccard(attr_i, attr_j)
    return edge_sim_matrix

def getSchemaFromSimMatrix(G, node_sim_matrix, edge_sim_matrix,**kwargs):
    theta = kwargs.get('theta',0.5)
    node_partitions, _ = clusteringBySim(node_sim_matrix,mode='threshold', theta=theta)
    node_partitions = {i:li for i,li in enumerate(node_partitions)}
    edge_partitions, _ = clusteringBySim(edge_sim_matrix,mode='threshold', theta=theta)
    edge_partitions = {i:li for i,li in enumerate(edge_partitions)}
    for i in edge_partitions:
        edge_partitions[i] = [list(G.edges)[j] for j in edge_partitions[i]]

    node2partition,edge2partition = getObject2PartitionDict(node_partitions), getObject2PartitionDict(edge_partitions)
    topo_sck1 = GetSchemaFromGraphSplitEdgeType(G, node2partition,edge2partition)
    remove_attr(topo_sck1)
    # topo_sck1.computeF1score(sc_gt,CONSIDER_EDGES=True,WEIGHTED=True,CONSIDER_EDGES_OF_SRC_TAR=True)
    return topo_sck1

# imcomplete method
# problem: too much memeory usage
def topo_based():
    emap, cnt = {}, 0
    for e in G.edges:
        emap[e] = cnt
        cnt += 1
    node_sim_matrix, edge_sim_matrix = np.ones((len(G.nodes), len(G.nodes))), np.ones((len(G.edges), len(G.edges)))
    k_node, k_edge, k = 1, 1, 1
    for level in range(k):
        topo_node_sim = topo_edge_sim = 0
        if level < k_node:
            topo_node_sim = getTopoNodeSim(G, node_sim_matrix, edge_sim_matrix, emap)

        if level < k_edge:
            topo_edge_sim = getTopoEdgeSim(G, node_sim_matrix, edge_sim_matrix, emap)

        node_sim_matrix += coef * topo_node_sim
        edge_sim_matrix += coef * topo_edge_sim
        coef *= coef