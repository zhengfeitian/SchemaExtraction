from NewElementTypeSet import *
from ElementType import *
from tqdm import tqdm
import numpy as np
import time
import py2neo
from Evaluation import validateEdgetypeOfSchema

def findByEqualLabels(li,e):
    for i,n in enumerate(li):
        if n.equalLabels(e):
            return i
    return -1


class Schema():
    def __init__(self,semantic=ALOM):
        self.nodetypeset = NodeTypeSet(semantic=semantic)
        self.edgetypeset = EdgeTypeSet()
        self.nodes = self.node_types = self.nodetypeset.nodetypes
        self.edges = self.edge_types = self.edgetypeset.edgetypes
        self.semantic = semantic

    # get a mapping {n: [rel1, rel2]} where n is src or target of rel*
    def getNodeEdgeMap(self):
        srcN_edges_map = {k.toHashStr():[] for k in self.nodetypeset.nodetypes}
        tarN_edges_map = {k.toHashStr():[] for k in self.nodetypeset.nodetypes}
        for et in self.edgetypeset.edgetypes:
            srcN_edges_map[et.src_type.toHashStr()].append(et)
            tarN_edges_map[et.target_type.toHashStr()].append(et)
        return srcN_edges_map, tarN_edges_map

    def getListOfEdgeForNetworkX(self):
        node_map = {k.toHashStr():k for k in self.nodetypeset.nodetypes}
        res = [] # list of triple (src_ind, tar_ind, edge_attributes)
        for et in self.edgetypeset.edgetypes:
            src = node_map[et.src_type.toHashStr()]
            tar = node_map[et.target_type.toHashStr()]
            src = self.nodetypeset.nodetypes.index(src)
            tar = self.nodetypeset.nodetypes.index(tar)
            attrs = {k:list(et.attributes[k]) for k in et.attributes}
            res.append((src,tar,et.attributes))
        return res

    def mergeBySimAndEdge(self, idf=False,comb='AND',**kwargs):
        srcN_edges_map, tarN_edges_map = self.getNodeEdgeMap()
        old_to_new_typemap = \
            self.nodetypeset.mergeBySimAndEdge(srcN_edges_map,tarN_edges_map,comb=comb)
        self.edgetypeset.mergeByJaccardSim(old_to_new_typemap, idf=idf) # idf not used

    def isConform(self,graph=None, sample_size=None):
        start = time.time()
        if graph!=None:
            nodes = graph.nodes
            if sample_size!=None:
                node_ind = np.random.randint(len(nodes),size=sample_size)
            else:
                node_ind = range(len(nodes))

            cnt_none = 0
            for i in tqdm(node_ind):
                node = nodes.get(i)
                if node==None:
                    cnt_none+=1
                    continue
                if self.nodetypeset.atLeastOneMatch(node)==False:
                    print('execution time: ', time.time() - start)
                    return False
            print('none node: ',cnt_none)

            relationships = graph.relationships
            if sample_size!=None:
                rel_ind = np.random.randint(len(relationships),size=sample_size)
            else:
                rel_ind = range(len(relationships))
            cnt_none = 0
            for i in tqdm(rel_ind):
                relation = relationships.get(i)
                if relation==None:
                    cnt_none += 1
                    continue
                relation[type(relation).__name__] = LBL()
                # it's possible that a relationship have more than two nodes
                if self.edgetypeset.atLeastOneMatch(relation)==False:
                    print('execution time: ', time.time() - start)
                    return False
            print('none relation: ',cnt_none)
        print('execution time: ', start - time.time() )
        return True

    def print(self):
        print('semantic: ', self.semantic)
        print('node type')
        self.nodetypeset.print()
        print('edge type')
        self.edgetypeset.print()

    def resetNbrOfEdgeType(self):
        if not validateEdgetypeOfSchema(self,mode='l'):
            print('the edgetypes of merged sc cannot be relinekd')
            return
        for et in self.edgetypeset.edgetypes:
            src_ind = findByEqualLabels(self.nodetypeset.nodetypes,et.src_type)
            et.src_type = self.nodetypeset.nodetypes[src_ind]
            tar_ind = findByEqualLabels(self.nodetypeset.nodetypes,et.target_type)
            et.target_type = self.nodetypeset.nodetypes[tar_ind]

    def mergeByLabelsSet(self, IGNORE_SRC_TAR=False):
        self.nodetypeset.mergeByLabelSet()
        self.edgetypeset.mergeByLabelSet(IGNORE_NODES=IGNORE_SRC_TAR)
        self.resetNbrOfEdgeType()

    def mergeByPropertiesSet(self):
        self.nodetypeset.mergeByPropertySet()
        self.edgetypeset.mergeByPropertySet(IGNORE_NODES=True)

    def mergeByTopology(self,e_theta=0.99):
        self.nodetypeset.mergeByTopology()
        self.edgetypeset.mergeByEdgeSim(return_sim_matrix=False,coef=[1,0,1],theta=e_theta)

    # this is
    def mergeByLabelAndTopology(self):
        self.nodetypeset.mergeBySimilarity(mode=['T','L'])
        self.edgetypeset.mergeByLabelSet(IGNORE_NODES=False)

    def mergeByLabelAndProperty(self,n_theta=0.5):
        self.nodetypeset.mergeBySimilarity(mode=['P','L'],theta=n_theta)
        self.edgetypeset.mergeByAttributeSet(mode=['P','L'],IGNORE_NODES=True)

    def mergeByPropertyAndTopology(self):
        self.nodetypeset.mergeBySimilarity(mode=['T','P'])
        self.edgetypeset.mergeByPropertySet(IGNORE_NODES=False)

    def mergeByJaccardSim(self, idf=False,**kwargs):
        # theta=kwargs.get('theta',0.5)
        # e_theta = kwargs.get('e_theta',theta)
        # n_theta = kwargs.get('n_theta',theta)
        node_o2n_id = {}
        edge_o2n_id = {}
        old_to_new_typemap = self.nodetypeset.mergeByJaccardSim(idf=idf,
                                            oldId2newId=node_o2n_id,**kwargs)
        self.edgetypeset.mergeByJaccardSim(old_to_new_typemap, idf=idf,
                                           oldId2newId=edge_o2n_id,**kwargs) # idf not used
        return node_o2n_id, edge_o2n_id

    def mergeByDBScan(self, idf=False):
        old_to_new_typemap = self.nodetypeset.mergeByDBScan(idf=idf)
        self.edgetypeset.mergeByDBScan(old_to_new_typemap, idf=idf)  # idf not used

    # def mergeBy

    # CONSIDER_EDGES mean if the evaluation of node type consider incoming and outgoing edges
    # CONSIDER_EDGES_OF_SRC_TAR means if the equal in comparing edge type consider the incoming and outgoing
    # edges of source node type and target node type
    def computeF1score(self, gt_sc, igoreDatatype=False, CONSIDER_EDGES=True,
                       CONSIDER_EDGES_OF_SRC_TAR=False, **kwargs):
        n_p, n_r, n_f1 = self.nodetypeset.computeF1score(gt_sc.nodetypeset, ignoreDatatype=igoreDatatype,
                                        CONSIDER_EDGES=CONSIDER_EDGES, **kwargs)
        e_p, e_r, e_f1 = self.edgetypeset.computeF1score(gt_sc.edgetypeset, ignoreDatatype=igoreDatatype,
                                        CONSIDER_EDGES_OF_SRC_TAR=CONSIDER_EDGES_OF_SRC_TAR, **kwargs)
        return n_p, n_r, n_f1, e_p, e_r, e_f1


    # mode: Topo, label, property
    def extractSchema(self, graph=None, subgraph=None, sample_size=None,
                      node_cursor=None, edge_cursor=None,node_dict_list={},
                      edge_dict_list={},cursor=None,IGNORE_LABEL=False,mode=None,**kwargs):
        start_time = time.time()
        # extract set of attributes
        # assume no isolated nodes
        if cursor:
            for n, incoming_edges, outgoing_edges in tqdm(cursor):
                if IGNORE_LABEL == False:
                    n = preprocessPy2neoNode(n)
                self.nodetypeset.addWithEdges(n, incoming_edges, outgoing_edges)
                for path in incoming_edges:
                    if type(path)==py2neo.Path:
                        relation = path.relationships[0]
                        relation[type(relation).__name__] = LBL()
                        path = relation
                    self.edgetypeset.add(path)

        elif node_dict_list and edge_dict_list:
            node_partitions = []
            edge_partitions = []
            if mode=='TOPO' or mode=='label' or mode=='l' or mode=='p' or mode=='attr_jaccard':
                incoming_edges = node_dict_list['incoming_edges']
                outgoing_edges = node_dict_list['outgoing_edges']
                node_list = node_dict_list['node_list']
                for i, n in tqdm(enumerate(node_list)):
                    if mode=='TOPO':
                        n_partition_num = self.nodetypeset.addWithEdges(n,
                                            incoming_edges[n['index']], outgoing_edges[n['index']])
                    elif mode=='l' or mode=='label':
                        n_partition_num = self.nodetypeset.addWithLabels(n,
                                            incoming_edges[n['index']], outgoing_edges[n['index']])
                    elif mode=='p':
                        n_partition_num = self.nodetypeset.addWithEdges(n,
                                     incoming_edges[n['index']], outgoing_edges[n['index']], False)
                    elif mode=='attr_jaccard':
                        n_partition_num = self.nodetypeset.addWithEdges(n,
                                    incoming_edges[n['index']], outgoing_edges[n['index']], False)
                    node_partitions.append(n_partition_num)
            else:
                if type(node_dict_list)==dict:
                    node_dict_list = node_dict_list['node_list']
                for i, n in tqdm(enumerate(node_dict_list)):
                    n_partition_num = self.nodetypeset.add(n)
                    node_partitions.append(n_partition_num)
            for i, r in tqdm(enumerate(edge_dict_list)):
                # it's possible that a relationship have more than two nodes
                src_t = self.nodetypeset[node_partitions[r.nodes[0]['index']]]
                tar_t = self.nodetypeset[node_partitions[r.nodes[1]['index']]]
                if mode == 'label' or mode == 'l':
                    e_partition_num = self.edgetypeset.addWithLabels(r,src_t,tar_t)
                elif mode=='p':
                    e_partition_num = self.edgetypeset.addWithProperties(r, src_t, tar_t)
                else:
                    e_partition_num = self.edgetypeset.add(r, src_t, tar_t)
                edge_partitions.append(e_partition_num)

            if mode=='attr_jaccard':
                node_o2n, edge_o2n = self.mergeByJaccardSim(**kwargs)
                node_partitions = [node_o2n.get(item,item)  for item in node_partitions]
                edge_partitions = [edge_o2n.get(item, item) for item in edge_partitions]
            return node_partitions, edge_partitions

        elif subgraph!=None:
            nodes = list(subgraph.nodes)
            for i, n in enumerate(nodes):
                node = nodes[i]
                node_dict = dict(node)
                # convert labels to attribute
                labels = str(node.labels).split(':')[1:]
                for l in labels:
                    node_dict[l] = LBL()

                self.nodetypeset.add(node_dict)

            relationships = list(subgraph.relationships)
            for i, r in enumerate(relationships):
                relation = relationships[i]
                relation[type(relation).__name__] = LBL()
                # it's possible that a relationship have more than two nodes
                self.edgetypeset.add(relation)

        elif graph!=None:
            nodes = graph.nodes
            if sample_size!=None:
                node_ind = np.random.randint(len(nodes),size=sample_size)
            else:
                # node_ind = range(len(nodes))
                node_ind = list(graph.nodes)

            cnt_none = 0
            for i in tqdm(node_ind):
                node = nodes.get(i)
                if node==None:
                    cnt_none+=1
                    continue
                node_dict = dict(node)
                # convert labels to attribute
                labels = str(node.labels).split(':')[1:]
                for l in labels:
                    node_dict[l] = LBL()

                self.nodetypeset.add(node_dict)
            print('none node: ',cnt_none)

            relationships = graph.relationships
            if sample_size!=None:
                rel_ind = np.random.randint(len(relationships),size=sample_size)
            else:
                rel_ind = range(len(relationships))
            cnt_none = 0
            for i in tqdm(rel_ind):
                relation = relationships.get(i)
                if relation==None:
                    cnt_none += 1
                    continue
                relation[type(relation).__name__] = LBL()
                # it's possible that a relationship have more than two nodes
                self.edgetypeset.add(relation)
            print('none relation: ',cnt_none)

        elif node_cursor!=None and edge_cursor!=None:
            for cur in tqdm(node_cursor):
                node = cur['n']
                node_dict = preprocessPy2neoNode(node)
                self.nodetypeset.add(node_dict)

            for cur in tqdm(edge_cursor):
                relation = cur['r']
                relation[type(relation).__name__] = LBL()
                # it's possible that a relationship have more than two nodes
                self.edgetypeset.add(relation)


        print('Schema extraction time', time.time()-start_time)

    # update schema
    # graph contain updated nodes and relationships
    # as only the updated instances are given, new element types might be created
    # we do not know how the elements look before updated so the types the elements originall conformed to are unknown
    # then old element types might not be deleted when they should
    def updateSchema(self, subg, UPDATE_NODE=True, UPDATE_REL=True, **kwargs):
        if UPDATE_NODE:
            nodes = list(subg.nodes)
            for n in tqdm(nodes):
                node_dict = preprocessPy2neoNode(n)
                self.nodetypeset.update(node_dict, **kwargs)
            #     TODO

        if UPDATE_REL:
            # TODO
            pass






