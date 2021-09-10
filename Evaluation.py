from Schema import *
from GraphGenerator import *
# check if each edge type have exactly one source type
# and one target type in nodetypeset
def validateEdgetypeOfSchema(sc,mode='is'):
    for et in sc.edgetypeset.edgetypes:
        src = tar = 0
        if sc.nodetypeset.nodetypes.count(et.src_type) == 1 and \
                sc.nodetypeset.nodetypes.count(et.target_type) == 1:
            continue
        for nt in sc.nodetypeset.nodetypes:
            if mode=='is':
                if nt is et.src_type:
                    src += 1
                if nt is et.target_type:
                    tar += 1
            elif mode.lower()=='l' or mode=='label' :
                if nt.equalLabels( et.src_type ):
                    src += 1
                if nt.equalLabels( et.target_type ):
                    tar += 1
        if src != 1 or tar != 1:
            return False
    return True


def findWithIs(li, e):
    if li.count(e) == 1:
        return li.index(e)
    for i, v in enumerate(li):
        if v is e:
            return i
    print(e, 'not found')


def getGraphFromSchema(sc):
    if validateEdgetypeOfSchema(sc) == False:
        print('the schema has problem with edge types, not matching node types')
        return
    g = nx.MultiGraph()
    g.add_nodes_from([(index, nt.attributes) for index, nt in enumerate(sc.nodetypeset.nodetypes)])
    edge_type_list = []
    for et in sc.edgetypeset.edgetypes:
        src = findWithIs(sc.nodetypeset.nodetypes, et.src_type)
        tar = findWithIs(sc.nodetypeset.nodetypes, et.target_type)
        edge_type_list.append((src, tar, et.attributes))
    g.add_edges_from(edge_type_list)
    return g


def node_subst_cost(n1, n2):
    cost = 0
    if set(n1).issuperset(set(n2)):
        cost = 0
    else:
        cost = len(set(n2) - set(n1)) / len(set(n2))
    return cost


def edge_subst_cost(e1, e2):
    if set(e1) == set(e2):
        return 0
    return 1

def computeGEDscore(gt_sc, sc):
    sc_G = getGraphFromSchema(gt_sc)
    scgen_G = getGraphFromSchema(sc)
    result = nx.algorithms.similarity.optimize_graph_edit_distance(scgen_G, sc_G, \
                                    node_subst_cost=node_subst_cost,edge_subst_cost=edge_subst_cost)
    print('proximating GED')
    for i in result:
        print('score: ',i)
    print('done')