
def normalizeCountDict(d):
    s = 0
    rm = []
    for i in d:
        if d[i]==0:
            rm.append(i)
        s += d[i]
    for k in rm:
        del d[k]
    for i in d:
        d[i] /= s
    return d


# compute nbr similarity
#TODO sophisticated
# sim(di,dj) = frequency*partition_simiarltiy
def computeSimilarityOfPartitionCount(di, dj, partition_sim):
    di = normalizeCountDict(di)
    dj = normalizeCountDict(dj)
    if len(set(di).intersection(set(dj)))==0: # no common nbr
        return 0
    result = 0
    in_i = set(di)-set(dj)
    in_j = set(dj)-set(di)
    for part in set(di).intersection(set(dj)):
        result += (di[part]+dj[part])
    for part in in_i:
        max_j = -1
        m = 0
        for j in dj:
            x,y = sorted((part,j))
            if partition_sim[x][y]>m:
                m = partition_sim[x][y]
                max_j = j
        if max_j!=-1 :
            result += (di[part]+dj[max_j])*m
    for part in in_j:
        max_i = -1
        m = 0
        for i in di:
            x,y = sorted((part,i))
            if partition_sim[x][y]>m:
                m = partition_sim[x][y]
                max_i = i
        if max_i!=-1:
            result += (di[max_i]+dj[part]) * m
    return result/2


def intializePartitions(G,t=[0]):
    node_partitions = {0: list(G.nodes)}
    edge_partitions = {0: list(G.edges)}
    # node_partition_sim = {0: {0: 1-sum(t) if sum(t)!=1 else 1}}
    # edge_partition_sim = {0: {0: 1-sum(t) if sum(t)!=1 else 1}}
    node_partition_sim = {0: {0: 1-sum(t) }}
    edge_partition_sim = {0: {0: 1-sum(t) }}
    node2partition = {i: 0 for i in list(G.nodes)}
    edge2partition = {i: 0 for i in list(G.edges)}
    return node_partitions, edge_partitions, node_partition_sim, edge_partition_sim, \
           node2partition, edge2partition

def getObject2PartitionDict(object_partitions):
    obj2part = {}
    for p in object_partitions:
        for obj in object_partitions[p]:
            obj2part[obj] = p
    return obj2part