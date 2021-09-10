from pickle5 import pickle
from Schema import *
import networkx as nx
from ElementType import *
from groundtruth import *
from IPython.core.debugger import set_trace
from sklearn.cluster import DBSCAN
import random

from utils import *


def computeSimilarity(obj_list1, obj_list2, partition_similarity, mode='jaccard',**kwargs):
    # method1: jaccard(nbrs1,nbrs2) exact match
    if mode=='jaccard':
        return jaccard(obj_list1,obj_list2)
    else:
        sim_i, sim_j = [], []
        for i in obj_list1:
            max_sim = 0
            for j in obj_list2:
                max_sim = max(max_sim, partition_similarity[i][j])
            sim_i.append(max_sim)
        for j in obj_list2:
            max_sim = 0
            for i in obj_list1:
                max_sim = max(max_sim, partition_similarity[i][j])
            sim_j.append(max_sim)
        sim_i = sum(sim_i) / len(sim_i) if len(sim_i) != 0 else 1
        sim_j = sum(sim_j) / len(sim_j) if len(sim_j) != 0 else 1
        return min(sim_i, sim_j)
    # method2: consider similarity of partition

def generateRandomNodes(nt, theta=0.5):
    type2val = {int: 12, str: 'string_value', float: 0.4,
                LBL: LBL(), list: [3, 'sf'], bool: True}
    node = {}
    # if there's mandatory attributes, then generate mandatory attributes
    # and other attributes with a possibility
    if nt.mandatory_attrs:
        for k in nt.attributes:
            if k in nt.mandatory_attrs:
                node[k] = type2val[list(nt.attributes[k])[0]]
            elif random.uniform(0, 1) > theta:
                node[k] = type2val[list(nt.attributes[k])[0]]
    # if there's no mandatory attributes, generate all attributes
    else:
        for k in nt.attributes:
            node[k] = type2val[list(nt.attributes[k])[0]]
    return node


# TODO, bugs, the source or target of the gnerated edges might not exist in the node list
# node_map: {node_type: [list of nodes of this type]}
def generateEdgesForAllNodes(et, node_map, incoming_edges, outgoing_edges,src_ind,tar_ind,**kwargs):
    #     src = generateRandomNodes(et.src_type)
    #     tar = generateRandomNodes(et.target_type)
    edge = generateRandomNodes(et)
    edge_list = []
    rand_num_edge_per_node = kwargs.get('edge_num',10)

    for node in node_map[src_ind]:
        for i in range(random.randint(0, rand_num_edge_per_node)):
            res = SimpleRel(edge)
            res.nodes[0] = node
            res.nodes[1] = random.choice(node_map[tar_ind])
            edge_list.append(res)
            incoming_edges[res.nodes[1]['index']].append(res)
            outgoing_edges[res.nodes[0]['index']].append(res)

    return edge_list

def getSrcAndTargetOfEdgeType(sc,et):
    src = tar = -1
    for ind,nt in enumerate(sc.nodetypeset.nodetypes):
        if et.src_type is nt:
            src = ind
        if et.target_type is nt:
            tar = ind 
    return src, tar

def generateFromSchema(sc, seed=19,**kwargs):#,rand_s=500,rand_t=1000):
    random.seed(seed)
    generated_nodes = [[] for i in sc.nodetypeset.nodetypes]
    generated_edges = [[] for i in sc.edgetypeset.edgetypes]
    nodes = []
    edges = []
    rand_s = kwargs.get('rand_s',500)
    rand_t = kwargs.get('rand_t',1000)
    print('rand_s',rand_s,'rand_t',rand_t)
    for node_ind, nt in enumerate(sc.nodetypeset.nodetypes):
        for i in range(random.randint(rand_s, rand_t)):
            node = generateRandomNodes(nt)
            node['index'] = len(nodes)
            generated_nodes[node_ind].append(node)
            nodes.append(node)

    incoming_edges = {i: [] for i in range(len(nodes))}
    outgoing_edges = {i: [] for i in range(len(nodes))}
    for e_ind,et in enumerate(sc.edgetypeset.edgetypes):
        src_ind, tar_ind = getSrcAndTargetOfEdgeType(sc,et)
        edge_list = generateEdgesForAllNodes(et, generated_nodes, \
                                             incoming_edges, outgoing_edges,src_ind,tar_ind,**kwargs)
        generated_edges[e_ind] = edge_list
    nodes_list = []
    edges_list = []
    for li in generated_nodes:
        nodes_list += li
    for li in generated_edges:
        edges_list += li
    node_dict_list = {'node_list': nodes_list, 'incoming_edges': incoming_edges, 'outgoing_edges': outgoing_edges}
    return node_dict_list, edges_list

def remove_attr(sc_gen, attr_key='index'):
    for nt in sc_gen.node_types:
        if attr_key in nt.attributes:
            del nt.attributes[attr_key]
    for et in sc_gen.edge_types:
        if attr_key in et.src_type.attributes:
            del et.src_type.attributes[attr_key]
        if attr_key in et.target_type.attributes:
            del et.target_type.attributes[attr_key]


# topo based 
# add partiion of nodes and edges]

def getEdges(n, G, node_partition, edge_partition, return_str=True):
    # a set of edge type
    if type(G)==nx.DiGraph:
        in_edges = set([edge_partition[e] for e in G.in_edges(n)])
        out_edges = set([edge_partition[e] for e in G.out_edges(n)])
    elif type(G)==nx.MultiDiGraph:
        in_edges = set([edge_partition[e] for e in G.in_edges(n,keys=True)])
        out_edges = set([edge_partition[e] for e in G.out_edges(n,keys=True)])
    inc = 'inc:' + '|'.join([str(i) for i in sorted(list(in_edges))])
    node = 'node:' + str(node_partition[n])
    out = 'out:' + '|'.join([str(i) for i in sorted(list(out_edges))])
    if return_str:
        return inc + node + out
    else:
        return sorted(in_edges), node_partition[n], sorted(out_edges)


def getNodes(e, node_partition, edge_partition,return_str=True):
    s = 'src:' + str(node_partition[e[0]])
    t = 'tar:' + str(node_partition[e[1]])
    edge = 'edge:' + str(edge_partition[e])
    if return_str:
        return s + edge + t
    else:
        return [node_partition[e[0]]],edge_partition[e],[node_partition[e[1]]]


# k is the degree of nbrs considered
# input: similarity of each partition?
def topo_based(G, k=2,node_partition=None, edge_partition=None,theta=0.99):
    if node_partition==None:
        node_partition = {i: 0 for i in list(G.nodes)}
    if edge_partition == None:
        edge_partition = {i: 0 for i in list(G.edges)}
    new_node_partition = {i: 0 for i in list(G.nodes)}
    new_edge_partition = {i: 0 for i in list(G.edges)}

    for level in range(k):
        npart_map = {}
        epart_map = {}
        part_num = 0

        for n in G.nodes:
            nbrs = getEdges(n, G, node_partition, edge_partition)  # flatten
            if not nbrs in npart_map:
                npart_map[nbrs] = part_num #extact match of nbrs
                part_num += 1

            new_node_partition[n] = npart_map[nbrs]
        node_partition = new_node_partition
        # triple (sour)
        part_num = 0
        for e in G.edges:
            nbrs = getNodes(e, node_partition, edge_partition)
            if not nbrs in epart_map:
                epart_map[nbrs] = part_num
                part_num += 1
            new_edge_partition[e] = epart_map[nbrs]
        edge_partition = new_edge_partition
    return node_partition, edge_partition, npart_map, epart_map


# node_partition  {node_id: partition_num}
# edge_partition {(src,tar): partition_num}
def getSchemaFromPartition(G, node_partition, edge_partition):
    from Schema import Schema
    sc = Schema()
    type_to_nodes = {}  # partition: [list of node]
    type_to_edges = {}  # partition:[list of edge]
    partition_to_type = {}  # for node, partition_num: NodeType
    for n in node_partition:
        if not node_partition[n] in type_to_nodes:
            type_to_nodes[node_partition[n]] = []
        type_to_nodes[node_partition[n]].append(n)

    for e in edge_partition:
        if not edge_partition[e] in type_to_edges:
            type_to_edges[edge_partition[e]] = []
        type_to_edges[edge_partition[e]].append(e)

    for t in type_to_nodes:
        nt = NodeType(elem=G.nodes[type_to_nodes[t][0]])
        for n in type_to_nodes[t]:
            nt2 = NodeType(elem=G.nodes[n])
            nt.mergeWithAnotherType(nt2)
        sc.nodetypeset.nodetypes.append(nt)
        sc.nodetypeset.numOfType.append(1)
        partition_to_type[t] = nt

    for t in type_to_edges:
        src, tar = type_to_edges[t][0][0], type_to_edges[t][0][1]

        st = partition_to_type[node_partition[src]]
        tart = partition_to_type[node_partition[tar]]
        
        if type(G)==nx.DiGraph:
            et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
        elif type(G) == nx.MultiDiGraph:
            et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar, type_to_edges[t][0][2]])
        for e in type_to_edges[t]:
            src, tar = e[0], e[1]
            st = partition_to_type[node_partition[src]]
            tart = partition_to_type[node_partition[tar]]
            if type(G) == nx.DiGraph:
                et2 = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
            elif type(G) == nx.MultiDiGraph:
                key = e[2]
                et2 = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar, key])
            et.mergeWithAnotherType(et2)
        sc.edgetypeset.edgetypes.append(et)
        sc.edgetypeset.numOfType.append(1)

    return sc

def getInEdges(G,n):
    if type(G)==nx.DiGraph:
        return G.in_edges(n)
    elif type(G)==nx.MultiDiGraph:
        return G.in_edges(n,keys=True)

def getOutEdges(G,n):
    if type(G)==nx.DiGraph:
        return G.out_edges(n)
    elif type(G)==nx.MultiDiGraph:
        return G.out_edges(n,keys=True)

def getSchemaFromPartitionWithEdges(G, node_partition, edge_partition):
    from Schema import Schema
    sc = Schema()
    type_to_nodes = {}  # partition: [list of node]
    type_to_edges = {}  # partition:[list of edge]
    partition_to_type = {}  # for node, partition_num: NodeType
    for n in node_partition:
        if not node_partition[n] in type_to_nodes:
            type_to_nodes[node_partition[n]] = []
        type_to_nodes[node_partition[n]].append(n)

    for e in edge_partition:
        if not edge_partition[e] in type_to_edges:
            type_to_edges[edge_partition[e]] = []
        type_to_edges[edge_partition[e]].append(e)

    set_trace()
    for t in type_to_nodes:
        nt = NodeType(elem=G.nodes[type_to_nodes[t][0]])
        for inc in getInEdges(G,type_to_nodes[t][0]):
            nt.incoming_edges.add(edge_partition[inc])
        for out in getOutEdges(G,type_to_nodes[t][0]):
            nt.outgoing_edges.add(edge_partition[out])
        for n in type_to_nodes[t]:
            nt2 = NodeType(elem=G.nodes[n])
            for inc in getInEdges(G,n):
                nt2.incoming_edges.add(edge_partition[inc])
            for out in getOutEdges(G,n):
                nt2.outgoing_edges.add(edge_partition[out])
            nt.mergeWithAnotherType(nt2)
        sc.nodetypeset.nodetypes.append(nt)
        sc.nodetypeset.numOfType.append(1)
        partition_to_type[t] = nt
    for t in type_to_edges:
        src, tar = type_to_edges[t][0][0], type_to_edges[t][0][1]

        st = partition_to_type[node_partition[src]]
        tart = partition_to_type[node_partition[tar]]

        if type(G)==nx.DiGraph:
            et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
        elif type(G) == nx.MultiDiGraph:
            et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar, type_to_edges[t][0][2]])
        # et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
        for e in type_to_edges[t]:
            src, tar = e[0], e[1]
            st = partition_to_type[node_partition[src]]
            tart = partition_to_type[node_partition[tar]]
            if type(G) == nx.DiGraph:
                et2 = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
            elif type(G) == nx.MultiDiGraph:
                key = e[2]
                et2 = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar, key])
            # et2 = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[src, tar])
            et.mergeWithAnotherType(et2)
        sc.edgetypeset.edgetypes.append(et)
        sc.edgetypeset.numOfType.append(1)

    # change the edges back to
    for nt in sc.nodetypeset.nodetypes:
        nt.incoming_edges = list(nt.incoming_edges)
        new_incoming = []
        new_outgoing = []
        nt.outgoing_edges = list(nt.outgoing_edges)
        for i,inc in enumerate(nt.incoming_edges):
            ct = ContentType(attr_type=sc.edgetypeset.edgetypes[inc].attributes)
            new_incoming.append(ct)

        for i,out in enumerate(nt.outgoing_edges):
            ct = ContentType(attr_type=sc.edgetypeset.edgetypes[out].attributes)
            new_outgoing.append(ct)
        nt.incoming_edges = set(new_incoming)
        nt.outgoing_edges = set(new_outgoing)
    return sc


def getEdgesAttrFromPartition(f, n):
    e_per_n = set()
    for inc in list(f(n, data=True, keys=True)):
        e_per_n.add('|k|'.join(sorted(list(inc[3].keys()))))
    #         for j in inc[3].keys():
    #             e_per_n.add(j)
    e_per_n = ';'.join(list(sorted(e_per_n)))
    return e_per_n


def showEdgesOfaPartition(G, nodes):
    es = set()
    for n in nodes:
        e_per_n = getEdgesAttrFromPartition(G.in_edges, n)
        es.add(e_per_n)
    print('inc', es)
    es = set()
    for n in nodes:
        e_per_n = getEdgesAttrFromPartition(G.out_edges, n)
        es.add(e_per_n)
    print('out', es)

def getEtype(G, st, tart, e):
    if type(G) == nx.DiGraph:
        et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[e[0], e[1]])
    elif type(G) == nx.MultiDiGraph:
        et = EdgeType(src_type=st, target_type=tart, relation_type=G.edges[e[0], e[1], e[2]])
    return et


def GetSchemaFromGraphSplitEdgeType(G, node_partition, edge_partition):
    from Schema import Schema
    sc = Schema()
    type_to_nodes = {}  # partition: [list of node]
    type_to_edges = {}  # partition:[list of edge]
    partition_to_type = {}  # for node, partition_num: NodeType
    for n in node_partition:
        if not node_partition[n] in type_to_nodes:
            type_to_nodes[node_partition[n]] = []
        type_to_nodes[node_partition[n]].append(n)
    #     print(type_to_nodes.keys())
    for e in edge_partition:
        if not edge_partition[e] in type_to_edges:
            type_to_edges[edge_partition[e]] = []
        type_to_edges[edge_partition[e]].append(e)

    sc.nodetypeset.nodetypes = [None for t in type_to_nodes]
    sc.nodetypeset.numOfType = [0 for t in type_to_nodes]
    for t in type_to_nodes:
        nt = NodeType(elem=G.nodes[type_to_nodes[t][0]])
        for inc in getInEdges(G, type_to_nodes[t][0]):
            nt.incoming_edges.add(edge_partition[inc])
        for out in getOutEdges(G, type_to_nodes[t][0]):
            nt.outgoing_edges.add(edge_partition[out])
        for n in type_to_nodes[t]:
            nt2 = NodeType(elem=G.nodes[n])
            for inc in getInEdges(G, n):
                nt2.incoming_edges.add(edge_partition[inc])
            for out in getOutEdges(G, n):
                nt2.outgoing_edges.add(edge_partition[out])
            nt.mergeWithAnotherType(nt2)
        sc.nodetypeset.nodetypes[t] = nt
        sc.nodetypeset.numOfType[t] = 1
        partition_to_type[t] = nt
    #         print('nt ',t, 'incoming: ', nt.incoming_edges, 'out ',nt.outgoing_edges)
    #         showEdgesOfaPartition(G,type_to_nodes[t])
    e_old_to_new = {t: {} for t in type_to_edges}
    for t in type_to_edges:
        src_tar_map = {}
        for e in type_to_edges[t]:
            src, tar = node_partition[e[0]], node_partition[e[1]]
            if not (src, tar) in src_tar_map:
                src_tar_map[(src, tar)] = []
            src_tar_map[(src, tar)].append(e)
        # made edge type from all the src_tar_tuple
        #         print(src_tar_map.keys())
        for src, tar in src_tar_map:
            e_li = src_tar_map[(src, tar)]
            st = partition_to_type[src]
            tart = partition_to_type[tar]

            et = getEtype(G, st, tart, e_li[0])
            for e in e_li:
                et2 = getEtype(G, st, tart, e)
                et.mergeWithAnotherType(et2)

            sc.edgetypeset.edgetypes.append(et)
            sc.edgetypeset.numOfType.append(1)
            e_id = len(sc.edgetypeset) - 1
            e_old_to_new[t][(src, tar)] = e_id
    #     sc.edgetypeset.print()
    # change the edges back to
    for n_id, nt in enumerate(sc.nodetypeset.nodetypes):
        nt.incoming_edges = list(nt.incoming_edges)
        new_incoming = []
        new_outgoing = []
        nt.outgoing_edges = list(nt.outgoing_edges)
        #         print(nt.incoming_edges,nt.outgoing_edges)
        #         set_trace()
        for i, inc in enumerate(nt.incoming_edges):
            for src, tar in e_old_to_new[inc]:
                if n_id == tar:
                    real_eid = e_old_to_new[inc][(src, tar)]
                    ct = ContentType(attr_type=sc.edgetypeset.edgetypes[real_eid].attributes)
                    new_incoming.append(ct)

        for i, out in enumerate(nt.outgoing_edges):
            for src, tar in e_old_to_new[out]:
                if n_id == src:
                    real_eid = e_old_to_new[out][(src, tar)]
                    ct = ContentType(attr_type=sc.edgetypeset.edgetypes[real_eid].attributes)
                    new_outgoing.append(ct)
        nt.incoming_edges = set(new_incoming)
        nt.outgoing_edges = set(new_outgoing)
    return sc


# n2p, e2p = getObject2PartitionDict(node_partitions), getObject2PartitionDict(edge_partitions)
# testsc = test(G, n2p,e2p)

# org_sim should also include similarity of it self i.e. org_sim[i][i] = l+p
def update_partitions(new_partitions_list, new_part_sim_list, org_partitions, org_sim):
    partitions = {}
    new_partitions_sim = {}
    part_num = -1
    new2old = {}  # map new partition number to old partitions number
    # new: [[0,1,2], [3,4]]
    # old: [0,1]
    part_size = [len(i) for i in new_partitions_list]
    for i, new_partitions in enumerate(new_partitions_list):
        for partition in new_partitions:
            part_num += 1
            partitions[part_num] = partition
            new2old[part_num] = i

    for i in range(len(partitions)):
        for j in range(i, len(partitions)):
            if i == j:
                if not i in new_partitions_sim:
                    new_partitions_sim[i] = {j: org_sim[0][0]}
                new_partitions_sim[i][j] = org_sim[0][0]
            elif new2old[i] == new2old[j]:  # in the same partiiton originally

                prev_len = sum(part_size[:new2old[i]])
                # print(i, j, new2old[i], new2old[j])
                # print(i - prev_len, j - prev_len)
                new_partitions_sim[i][j] = org_sim[new2old[i]][new2old[j]] + \
                                           new_part_sim_list[new2old[i]][i - prev_len][j - prev_len]
            else:
                if not i in new_partitions_sim:
                    new_partitions_sim[i] = {}
                new_partitions_sim[i][j] = org_sim[new2old[i]][new2old[j]]
    obj2partition = {}
    for part in partitions:
        for n in partitions[part]:
            obj2partition[n] = part
    return partitions, new_partitions_sim, obj2partition
# tested
# new_partitions_list = [[[0,1],[2]],[[3],[4]]]
# new_part_sim_list = [{0:{0:0.3,1:0.1}},{0:{0:0.15,1:0.12}}]
# org_partitions = [[0,1,2],[3,4]]
# org_sim = {0:{0:0.7,1:0.2},1:{1:0.7}}
# update_partitions(new_partitions_list, new_part_sim_list, org_partitions, org_sim)


def clusteringBySim(sim_matrix, theta=0.5, mode='threshold'):
    ind = np.array([i for i in range(sim_matrix.shape[0])])

    if mode == 'threshold':
        sim_copy = sim_matrix.copy()
        sim_copy[sim_copy > theta] = 1
        sim_copy[sim_copy <= theta] = 0
        n_components, labels = connected_components(
            sim_copy, directed=False, return_labels=True)
    elif mode == 'boolean' or mode == 'bool':
        n_components, labels = connected_components(
            sim_matrix, directed=False, return_labels=True)
    elif mode == 'DBScan':
        clustering = DBSCAN(eps=3, min_samples=1, metric='precomputed').fit(sim_matrix)
        if -1 in clustering.labels_:
            print('-1 in db clustering')
        n_components, labels = len(set(clustering.labels_)), clustering.labels_

    new_partitions = [[] for i in range(n_components)]
    for i, l in enumerate(labels):
        new_partitions[l].append(i)
    if mode!='threshold':
        return new_partitions

    # else, the similarity matrix is used to compute partition similarity
    new_partition_sim = {i: {j: 0 for j in range(i, n_components)} for i in range(n_components)}
    for i in range(n_components):  # assume no -1 in labels
        index_i = ind[labels == i]
        for j in range(i, n_components):
            if i == j:
                new_partition_sim[i][j] = 1
                continue
            index_j = ind[labels == j]
            sim_list = []
            for ind_i in index_i:
                for ind_j in index_j:
                    sim_list.append(sim_matrix[ind_i, ind_j])
            new_partition_sim[i][j] = sum(sim_list) / len(sim_list)

    return new_partitions, new_partition_sim



def generateGraphFromSchema(gt_sc, l=0.4, p=0.4, **kwargs):
    nl, el = generateFromSchema(gt_sc, **kwargs)

    from Schema import Schema
    sc_gen = Schema()

    # mode = TOPO, because TOPO mode will fill the incoming_edges and outgoing_edges of nodetypes
    n_part, e_part = sc_gen.extractSchema(node_dict_list=nl, edge_dict_list=el,
                                  # IGNORE_LABEL=False, mode=kwargs.get('mode','l'), **kwargs)
                            IGNORE_LABEL = False, **kwargs)
    remove_attr(sc_gen)
    edge_partition_sim = getElemTypeSimilarityDict(sc_gen.edgetypeset.edgetypes, l, p)
    node_partition_sim = getElemTypeSimilarityDict(sc_gen.nodetypeset.nodetypes, l, p)

    G = nx.MultiDiGraph()
    nl_tup = [(i['index'], i) for i in nl['node_list']]
    el_tup = [(i.nodes[0]['index'], i.nodes[1]['index'], i) for i in el]
    G.add_nodes_from(nl_tup)

    node2partition, edge2partition = {}, {}
    for n in nl['node_list']:
        node2partition[n['index']] = n_part[n['index']]
    for ind, e in enumerate(el_tup):
        key = G.add_edge(e[0],e[1])
        G.edges[e[0],e[1],key].update(e[2])
        edge2partition[(e[0], e[1], key)] = e_part[ind]

    node_partitions = {}  # map partition number to a list of nodes that belong to the partition
    edge_partitions = {}
    for node in node2partition:
        partition_num = node2partition[node]
        if not partition_num in node_partitions:
            node_partitions[partition_num] = []
        node_partitions[partition_num].append(node)

    for edge in edge2partition:
        partition_num = edge2partition[edge]
        if not partition_num in edge_partitions:
            edge_partitions[partition_num] = []
        edge_partitions[partition_num].append(edge)
    # TODO remove attribute 'index' from each node

    return G, node2partition, edge2partition, node_partitions, \
           edge_partitions, node_partition_sim, edge_partition_sim, sc_gen


# objects can be random index,
# objects[ind] = obj
# map objects to 1..len(objects)
# return
# new_partitions is a list of partition
# each partition is a list of objects
def repartitionNode(objects, G, partition_similarity, node2partition, edge2partition,**kwargs):
    index_map = {obj: ind for ind, obj in enumerate(objects)}
    sim_matrix = np.zeros((len(objects), len(objects)), dtype=np.bool)
    theta = kwargs.get('theta',0.5)
    #     sim_matrix = np.zeros((len(objects),len(objects)), dtype=np.float16)
    # theta = partition_similarity[0][0]/2
    # print('theta in repartition node',theta)

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            o1, o2 = objects[i], objects[j]
            in_edges1, _, out_edges1 = getEdges(o1, G, node2partition, edge2partition, return_str=False)
            in_edges2, _, out_edges2 = getEdges(o2, G, node2partition, edge2partition, return_str=False)
            sim = computeSimilarity(in_edges1,
                                    in_edges2, partition_similarity, **kwargs) + \
                  computeSimilarity(out_edges1,
                                    out_edges2, partition_similarity, **kwargs)
            # print(i,j,sim,theta,sim/2>=theta)
            sim_matrix[i, j] = 1 if sim / 2 > theta else 0
    #     new_partitions, new_partition_similarity = clusteringBySim(sim_matrix,theta,mode='bool')
    new_partitions = clusteringBySim(sim_matrix, theta, mode='bool')
    new_partitions = [[objects[i] for i in part] for part in new_partitions]

    return new_partitions  # , new_partition_similarity,sim_matrix


def repartitionEdge(objects, node_part_sim, node2partition, edge2partition, **kwargs):
    # set_trace()
    index_map = {obj: ind for ind, obj in enumerate(objects)}
    #     sim_matrix = lil_matrix((len(objects),len(objects)))
    sim_matrix = np.zeros((len(objects), len(objects)), dtype=np.bool)
    theta = kwargs.get('theta', 0.5)
    theta = node_part_sim[0][0] * theta
    # print('theta in repartition edge',theta)

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            o1, o2 = objects[i], objects[j]
            src1, _, tar1 = getNodes(o1, node2partition, edge2partition, return_str=False)
            src2, _, tar2 = getNodes(o2, node2partition, edge2partition, return_str=False)
            src1, src2 = sorted((src1[0], src2[0]))
            tar1, tar2 = sorted((tar1[0], tar2[0]))
            sim_matrix[i, j] = 1 if (node_part_sim[src1][src2] + node_part_sim[tar1][tar2]) / 2.0 >= theta else 0

    #     new_partitions, new_partition_similarity = clusteringBySim(sim_matrix,theta,mode='bool')
    new_partitions = clusteringBySim(sim_matrix, theta, mode='bool')
    new_partitions = [[objects[i] for i in part] for part in new_partitions]

    return new_partitions  # , new_partition_similarity


# partition_similarity: initially in range [0,p+l] i.e. [0,1-t]
# is a map;  p_s[i][j] = sim  (i and j are corresponding partitoin number)

def topoPartitionBased(G, k, t=[0.2, 0.1], node_partitions=None, edge_partitions=None,
                       node_partition_sim=None, edge_partition_sim=None,
                       node2partition=None, edge2partition=None,**kwargs):
    print('compute node mode: ',kwargs.get('mode','jaccard'))
    if node_partitions == None and node_partition_sim == None:
        print('initilizing partitions for G')
        node_partitions, edge_partitions, node_partition_sim, edge_partition_sim, \
        node2partition, edge2partition = intializePartitions(G, t)
    # set_trace()
    for i in range(k):
        newN_partitions_list = []
        for part in tqdm(node_partitions):
            new_node_partitions = \
                repartitionNode(node_partitions[part], G, edge_partition_sim,
                                node2partition, edge2partition, **kwargs)
            newN_partitions_list.append(new_node_partitions)
        newE_part_list = []
        for part in tqdm(edge_partitions):
            new_edge_partitions = \
                repartitionEdge(edge_partitions[part], node_partition_sim,
                                node2partition, edge2partition, **kwargs)
            newE_part_list.append(new_edge_partitions)

        new_node_partitions, node_partition_sim, new_n2part = \
            updatePartitions(newN_partitions_list, node_partition_sim, t[i],
                             n2part=node2partition, e2part=edge2partition, G=G,
                             nbr_partition_sim=edge_partition_sim)
        new_edge_partitions, edge_partition_sim, new_e2part = \
            updatePartitions(newE_part_list, edge_partition_sim, t[i],
                             n2part=node2partition, e2part=edge2partition, G=G,
                             nbr_partition_sim=node_partition_sim)
        node2partition, node_partitions = new_n2part, new_node_partitions
        edge2partition, edge_partitions = new_e2part, new_edge_partitions

    return node_partitions, edge_partitions, node_partition_sim, edge_partition_sim


def nbrs(obj, **kwargs):
    n2part = kwargs.get('n2part')
    e2part = kwargs.get('e2part')
    G = kwargs.get('G')
    if type(obj) == tuple:  # edge
        t1, _, t2 = getNodes(obj, n2part, {obj: 0}, return_str=False)
    else:  # node
        t1, _, t2 = getEdges(obj, G, {obj: 0}, e2part, return_str=False)
    return t1, t2


def T_partition(p_i, p_j, partition_sim, **kwargs):
    # counts of partitions of nbr
    # nbr_t1_i[partition_of_nbr] = occcurence
    nbr_t1_i = {}
    nbr_t2_i = {}
    nbr_t1_j = {}
    nbr_t2_j = {}
    # if partiion_sim is node_partition_sim
    # then nbr_partition_sim should be edge_partition_sim, vice versa
    nbr_partition_sim = kwargs.get('nbr_partition_sim')
    for obj in p_i:
        nbrt1, nbrt2 = nbrs(obj, **kwargs)  # two set of object partition
        for partition in nbrt1:
            if not partition in nbr_t1_i:
                nbr_t1_i[partition] = 0
            nbr_t1_i[partition] += 1
        for partition in nbrt2:
            if not partition in nbr_t2_i:
                nbr_t2_i[partition] = 0
            nbr_t2_i[partition] += 1
    for obj in p_j:
        nbrt1, nbrt2 = nbrs(obj, **kwargs)  # two set of object partition
        for partition in nbrt1:
            if not partition in nbr_t1_j:
                nbr_t1_j[partition] = 0
            nbr_t1_j[partition] += 1
        for partition in nbrt2:
            if not partition in nbr_t2_j:
                nbr_t2_j[partition] = 0
            nbr_t2_j[partition] += 1
            # compute similarity of nbr_t1_i,nbr_t1_j
    # so the partition similarity of nbr should be used
    t1_sim = computeSimilarityOfPartitionCount(nbr_t1_i, nbr_t1_j, nbr_partition_sim)
    t2_sim = computeSimilarityOfPartitionCount(nbr_t2_i, nbr_t2_j, nbr_partition_sim)

    return (t1_sim + t2_sim) / 2


# tested
# new_partitions_list = [[[0,1],[2]],[[3],[4]]]
# new_part_sim_list = [{0:{0:0.3,1:0.1}},{0:{0:0.15,1:0.12}}]
# org_partitions = [[0,1,2],[3,4]]
# org_sim = {0:{0:0.7,1:0.2},1:{1:0.7}}
# update_partitions(new_partitions_list, new_part_sim_list, org_partitions, org_sim)
# org_sim should also include similarity of it self i.e. org_sim[i][i] = l+p
def updatePartitions(new_partitions_list, org_sim, ti, **kwargs):
    # set_trace()
    partitions = {}
    new_partitions_sim = {}
    part_num = -1
    new2old = {}  # map new partition number to old partitions number
    # new: [[0,1,2], [3,4]]
    # old: [0,1]
    for i, new_partitions in enumerate(new_partitions_list):
        for partition in new_partitions:
            part_num += 1
            partitions[part_num] = partition
            new2old[part_num] = i
    for i in range(len(partitions)):
        for j in range(i, len(partitions)):
            if not i in new_partitions_sim:
                new_partitions_sim[i] = {}
            if i == j:
                new_partitions_sim[i][j] = org_sim[0][0] + ti
            else:
                new_partitions_sim[i][j] = org_sim[new2old[i]][new2old[j]] + \
                                           ti * T_partition(partitions[i], partitions[j], org_sim, **kwargs)

    obj2partition = {}
    for part in partitions:
        for n in partitions[part]:
            obj2partition[n] = part
    return partitions, new_partitions_sim, obj2partition

if __name__=='__main__':
    G = nx.DiGraph() # a small graph used for testing
    G.add_nodes_from([
        (0, {'Person': 'lbl', 'name': 'Same', 'age': 12}),
        (1, {'Person': 'lbl', 'name': 'Tom'}),
        (2, {'Person': 'lbl', 'name': 'John', 'phone': 324})
    ])
    G.add_edges_from([
        (0, 1, {'Knows': 'lbl'}),
        (1, 2, {'Knows': 'lbl', 'at': 'school'})
    ])

    topoPartitionBased(G, k=1, t=[0.1])
