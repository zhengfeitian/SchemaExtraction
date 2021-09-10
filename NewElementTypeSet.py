from neo4j import GraphDatabase
from py2neo import Node,Relationship
import json
from ElementType import *
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import numpy as np
from math import log, sqrt
from sklearn.cluster import DBSCAN
import py2neo
from tqdm import tqdm
from scipy.sparse import csr_matrix,lil_matrix
from IPython.core.debugger import set_trace


def addEdgesToNodetype(nt, incoming_edges, outgoing_edges):
    for inc in incoming_edges:
        et = ContentType(elem=inc)
        nt.incoming_edges.add(et)
    for out in outgoing_edges:
        et = ContentType(elem=out)
        nt.outgoing_edges.add(et)


# check if elem is conformed to elem_type
from ElementTypeSet import NodeTypeSet

def jaccard(s1,s2):
    if type(s1)==NodeType and type(s2)==NodeType:
        s1 = set(s1.getAttrs(return_key=True))
        s2 = set(s2.getAttrs(return_key=True))
    if type(s1)!=set:
        s1 = set(s1)
    if type(s2)!=set:
        s2 = set(s2)
    if len(s1.union(s2))==0:
        # print('two sets are empty when computing jaccard')
        return 1
    return len(s1.intersection(s2))/len(s1.union(s2))

# given a list of relationships [rel1,rel2,] with source or edges start with
# the same node
# compute similarity, i.e. structural similarity of two nodes
def computeListOfEdgesSim(rels_i, rels_j):
    if len(rels_j) == 0 and len(rels_i) == 0:
        return 1
    elif len(rels_i) == 0 and len(rels_j) != 0:
        return 0
    elif len(rels_i) != 0 and len(rels_j) == 0:
        return 0
    ejaccard = []
    for resi in rels_i:
        jaccard_li = []
        for resj in rels_j:
            edge_jaccard = jaccard(resi.attributes, resj.attributes)
            if hasattr(resi,'src_type'):
                src_node_sim = jaccard(resi.src_type,resj.src_type)
                tar_node_sim = jaccard(resi.target_type,resj.target_type)
            # TODO try the AND combination method
            # print('edge resi and resj similarity')
            # resi.print()
            # resj.print()
            # edge_jaccard = 1 if edge_jaccard>0.5 else 0
            # src_node_sim = 1 if src_node_sim>0.5 else 0
            # tar_node_sim = 1 if tar_node_sim>0.5 else 0
                if  (edge_jaccard + src_node_sim + tar_node_sim) / 3 > 0.5:
                    resi.print()
                    resj.print()
                    print('resi, resj similarity',  (edge_jaccard + src_node_sim + tar_node_sim) / 3 )
                jaccard_li.append( (edge_jaccard + src_node_sim + tar_node_sim) / 3 )
            else:
                jaccard_li.append(edge_jaccard)
        # if len(jaccard_li) > 0:
            # ejaccard.append( sum(jaccard_li)/len(jaccard_li))
        ejaccard.append(max(jaccard_li))
        # else:
        #     ejaccard.append(1)
    ejaccard = sum(ejaccard) / len(ejaccard)
    return ejaccard


def isTypeOf(elem,elem_type,strict=True):
    if strict and set(elem)==set(elem_type):
        for k in elem.keys():
            if not istypeof(elem[k],elem_type[k]):
                return False
        return True
    if strict==False and set(elem)>=set(elem_type):
        for k in elem_type.keys():
            if not istypeof(elem[k],elem_type[k]):
                return False
        return True

    return False

def getElemTypeSimilarityDict(element_types,l,p):
    # element_types: a list of element types
    # return two dimensional similarity dict {i:{j: similarity}}
    sim = {i:{j: 0 for j in range(i,len(element_types))} for i in range(len(element_types))}
    for i in range(len(element_types)):
        for j in range(i, len(element_types)):
            if i==j:
                sim[i][j] = l+p
            else:
                sim[i][j] = l * jaccard( element_types[i].getLabels(),
                                         element_types[j].getLabels())
                sim[i][j] += p * jaccard( element_types[i].getProperties(),
                                          element_types[j].getProperties())
    return sim

def findIndex(li, x):
    for i, e in enumerate(li):
        if e is x:
            return i
    return -1

class NodeTypeSet:
    def __init__(self, semantic=ALOM):
        # list of element type
        self.nodetypes = [] # list of node type
        self.numOfType = [] # keep count of the instnaces per type
        self.node_map = {} # map nodetypeHash to nodetype
        self.semantic = semantic

    def __getitem__(self, item):
        return self.nodetypes[item]

    def getIdf(self):
        attr_list = []
        for nt in self.nodetypes:
            attr_list += list(set(nt.attributes))
        idf_dict = {}
        for i in attr_list:
            if not i in idf_dict:
                idf_dict[i] = 0
            idf_dict[i] += 1
        for k in idf_dict:
            idf_dict[k] = log(len(self.nodetypes)/ idf_dict[k])
            # idf_dict[k] = 1 / sqrt(idf_dict[k])
        # print('idf: ', idf_dict)
        return idf_dict

    # return union of all attribute keys in all types
    def getAttributes(self):
        res = set()
        for nt in self.nodetypes:
            res = res.union(set(nt.attributes))
        return list(res)

    def plotVectorizedNodes(self, idf=False):
        attrs = self.getAttributes()
        res = [[0 for i in attrs] for j in self.nodetypes]
        for i,nt in enumerate(self.nodetypes):
            for attr_ind,attr in enumerate(attrs):
                if attr in nt.attributes:
                    res[i][attr_ind] = 1
        labels_for_vector = [nt.toHashStr(without_type=True) for nt in self.nodetypes]
        return res,labels_for_vector

    # merge the
    def mergeByLabelSet(self):
        return self.mergeByHash(mode=['L'])

    # mode is a list ['TOPO','Label','Property']
    def mergeByHash(self,mode=None):
        new_nodetypes = []
        topo_type_dict = {}
        old2new_ind,str2oid = {},{} # map the original index of node type to the new index of node type
        for oid, nt in enumerate(self.nodetypes):
            hashed_info = ''
            if 'TOPO' in mode or 'T' in mode:
                hashed_info = hashed_info + nt.getConnectedEdges() + 'ENDTOPO' # getConnectedEdges return string
            if 'Label' in mode or 'L' in mode:
                hashed_info = hashed_info + ':'.join(set(nt.getLabels())) + 'ENDL'
            if 'Property' in mode or 'P' in mode:
                hashed_info = hashed_info + ':'.join(set(nt.getProperties())) + 'ENDP'
            if not hashed_info in topo_type_dict:
                topo_type_dict[hashed_info] = nt
            else:
                # this change the original attributes dicitonary
                topo_type_dict[hashed_info].mergeWithAnotherType(nt)
            str2oid[hashed_info] = oid

        for k in topo_type_dict:
            if not topo_type_dict[k].mandatory_attrs:
                topo_type_dict[k].mandatory_attrs = set(topo_type_dict[k].attributes)
            new_nodetypes.append(topo_type_dict[k])
            nid = len(new_nodetypes) - 1
            oid = str2oid[k]
            old2new_ind[oid] = nid

        self.nodetypes,old_nts = new_nodetypes,self.nodetypes
        return old2new_ind, old_nts

    def mergeByLabelAndTopology(self):
        self.mergeByHash(mode = ['L','T'])

    # the schema schould be extraced with the addWithEdges method
    def mergeByTopology(self):
        self.mergeByHash(mode=['T'])
        # m

    # merge the
    # TODO the type of properties are ignored. i.e. 'name':str and 'name':int are considered the same
    def mergeByPropertySet(self):
        self.mergeByHash(mode=['P'])


    # merge types by Jaccard similarity
    # theta is the similarity threshold
    # return a dict that map the original node type to the new merged node type
    def mergeByJaccardSim(self,theta=0.5, idf=True, return_sim_matrix=False,**kwargs):
        print('merge node by ',kwargs.get('mode','attr'), ' theta:', theta)
        # oldId2newId = kwargs.get('oldId2newId',{})
        num_nt = len(self.nodetypes)
        idf_dict = None
        # sim_matrix = [[0 for i in range(num_nt)] for j in range(num_nt)]
        # res_matrix = [[0 for i in range(num_nt)] for j in range(num_nt)]
        sim_matrix = lil_matrix((num_nt,num_nt))
        res_matrix = lil_matrix((num_nt,num_nt))
        if idf:
            idf_dict = self.getIdf()
        for i in range(len(self.nodetypes)):
            nt_i = self.nodetypes[i]
            for j in range(i+1,len(self.nodetypes)):
                nt_j = self.nodetypes[j]
                if nt_i.jaccardSim(nt_j,idf_dict=idf_dict) > theta:
                    if return_sim_matrix:
                        res_matrix[i,j] = res_matrix[j,i] = \
                            nt_i.jaccardSim(nt_j,idf_dict=idf_dict,mode=kwargs.get('mode','attr'))
                    sim_matrix[i,j] = sim_matrix[j,i] = 1

                    # print('%d %d connect'%(i,j))

        # sim_matrix = csr_matrix(sim_matrix)
        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)
        if return_sim_matrix:
            return res_matrix
        return self.mergeByComponents(n_components, labels, **kwargs)

    # mode = ['T','P','L']
    def mergeBySimilarity(self,mode,theta=0.5):
        num_nt = len(self.nodetypes)
        idf_dict = None
        sim_matrix = lil_matrix((num_nt,num_nt))
        deno=3.0
        if set(mode)==set(['P','L']):
            deno = 2.0
        # when there's no label in all node types,
        # then the similarity measure is purely based on property set
        no_label_flag = 1

        for i in range(len(self.nodetypes)):
            nt_i = self.nodetypes[i]
            for j in range(i+1,len(self.nodetypes)):
                nt_j = self.nodetypes[j]
                if 'L' in mode:
                    sim_matrix[i,j] += jaccard(nt_i.getLabels(),nt_j.getLabels())
                    sim_matrix[j,i] = sim_matrix[i,j]
                    if len(nt_i.getLabels())>0 or len(nt_j.getLabels())>0:
                        no_label_flag=0
                    # print(i,j,'label sim: ',sim_matrix[i,j])
                if 'T' in mode:
                    inc_sim = computeListOfEdgesSim(list(nt_i.incoming_edges),list(nt_j.incoming_edges))
                    out_sim = computeListOfEdgesSim(list(nt_i.outgoing_edges),list(nt_j.outgoing_edges))
                    # print(i,j,'inc sim: ',inc_sim,'out sim:',out_sim)
                    sim_matrix[i,j] += (inc_sim+out_sim)/2
                    sim_matrix[j,i] = sim_matrix[i,j]
                if 'P' in mode:
                    sim_matrix[i,j] += jaccard(nt_i.getProperties(),nt_j.getProperties())
                    sim_matrix[j,i] = sim_matrix[i,j]
                    # print(i,j,'property sim: ',sim_matrix[i,j])

                #TODO if 'p' in mode

        if no_label_flag==1: # if there's label
            theta += 1
        else:
            sim_matrix /=deno
        sim_matrix[sim_matrix>=theta] = 1
        sim_matrix[sim_matrix<theta] = 0


        # sim_matrix = csr_matrix(sim_matrix)
        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)
        return self.mergeByComponents(n_components, labels)


    # method for combining similarity;  AND or AVG
    # ignore_content, if the properties and labels are ignored when merging
    def mergeBySimAndEdge(self,src_map,tar_map, theta=0.5, idf=False, return_sim_matrix=False,comb='AND',
                          IGNORE_CONTENT=False):
        num_nt = len(self.nodetypes)
        idf_dict = None
        sim_matrix = [[0 for i in range(num_nt)] for j in range(num_nt)]
        node_sim_matrix = [[0 for i in range(num_nt)] for j in range(num_nt)]
        src_edge_sim = [[0 for i in range(num_nt)] for j in range(num_nt)]
        tar_edge_sim = [[0 for i in range(num_nt)] for j in range(num_nt)]
        if idf:
            idf_dict = self.getIdf()
        for i in range(num_nt):
            nt_i = self.nodetypes[i]
            rels_i = src_map[nt_i.toHashStr()] # list of outgoing edges from nt_i
            tar_i = tar_map[nt_i.toHashStr()] # list of incoing edges of nt_i
            for j in range(i+1,len(self.nodetypes)):
                nt_j = self.nodetypes[j]
                rels_j = src_map[nt_j.toHashStr()]
                tar_j = tar_map[nt_j.toHashStr()]

                jaccardSim = nt_i.jaccardSim(nt_j)
                sim_matrix[i][j] = sim_matrix[j][i] = round(jaccardSim,2)
                node_sim_matrix[i][j] = node_sim_matrix[j][i] = jaccardSim
                src_edge_sim[i][j] = round(computeListOfEdgesSim(rels_i,rels_j),2)
                src_edge_sim[j][i] = round(computeListOfEdgesSim(rels_j,rels_i),2)
                src_edge_sim[i][j] = src_edge_sim[j][i] = min(src_edge_sim[i][j], src_edge_sim[j][i])

                tar_edge_sim[i][j] = round(computeListOfEdgesSim(tar_i,tar_j),2)
                tar_edge_sim[j][i] = round(computeListOfEdgesSim(tar_j,tar_i),2)
                tar_edge_sim[i][j] = tar_edge_sim[j][i] = min(tar_edge_sim[i][j], tar_edge_sim[j][i])
                # if sim_matrix[i][j] > theta and (src_edge_sim[i][j]<theta or tar_edge_sim[i][j]<theta):
                #     print('(%d,%d) node: %.2f src: %.2f tar: %.2f' %
                #           (i,j,sim_matrix[i][j], src_edge_sim[i][j], tar_edge_sim[i][j]))

                if IGNORE_CONTENT: # if content of a node are ignored, then
                    sim_matrix[i][j]=theta+0.0001

                if comb=='AND' and sim_matrix[i][j]>theta and src_edge_sim[i][j] > theta and tar_edge_sim[i][j]>theta:
                    sim_matrix[i][j] = sim_matrix[j][i] = 1
                elif comb=='AVG' and sim_matrix[i][j]+src_edge_sim[i][j]+tar_edge_sim[i][j] > 3*theta:
                    sim_matrix[i][j] = sim_matrix[j][i] = 1
                else:
                    sim_matrix[i][j] = sim_matrix[j][i] = 0

        if return_sim_matrix:
            return node_sim_matrix, src_edge_sim, tar_edge_sim

        sim_matrix = csr_matrix(sim_matrix)
        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)
        print('components ', n_components)
        return self.mergeByComponents(n_components, labels)

    # sim_matrix is csr matrix
    def mergeByComponents(self, n_components, labels, **kwargs):
        oldId2newId = kwargs.get('oldId2newId',{})
        oldType_to_newType_map = {i.toHashStr(): i for i in self.nodetypes}
        num_nt = len(self.nodetypes)
        if n_components == num_nt:
            print('no node cluster found')
            return oldType_to_newType_map
        else:
            print('%d clusters found out of %d nodes' % (n_components, num_nt))
        new_node_types = []
        old_nt = np.array(self.nodetypes)

        for i in range(n_components):
            nt = mergeContentType(old_nt[labels==i])
            new_node_types.append(nt)
            new_id = len(new_node_types)-1
            for id, lb in enumerate(labels):
                if lb==i:
                    oldId2newId[id] = new_id
            for ot in old_nt[labels==i]:
                oldType_to_newType_map[ot.toHashStr()] = nt

        self.nodetypes = new_node_types
        return oldType_to_newType_map

    def mergeByDBScan(self, idf=False):
        print('start merging node type by DBScan, with idf',idf)
        oldType_to_newType_map = {i.toHashStr(): i for i in self.nodetypes}
        dbScan = DBSCAN(min_samples=1,metric='cosine')
        attr_list = self.getAttributes()
        attrs = [list(set(nt.getAttrs())) for nt in self.nodetypes]
        X = []
        x = []
        sample_weight = []
        idf_dict = self.getIdf()
        for ind, listOfAttr in enumerate(attrs):
            if idf:
                x = [idf_dict[i] if i in listOfAttr else 0 for i in attr_list]
            else:
                x = [1 if i in listOfAttr else 0 for i in attr_list]
            sample_weight.append(self.numOfType[ind]) # use the number of elements as
            X.append(x)
        sample_weight = np.array(sample_weight)
        dbScan.fit(X, sample_weight=sample_weight)
        n_components = len(set(dbScan.labels_))
        labels = dbScan.labels_

        return self.mergeByComponents(n_components, labels)

    def __len__(self):
        return len(self.nodetypes)

    def print(self):
        print('number of node types ', len(self.nodetypes))
        for i, nt in enumerate(self.nodetypes):
            print('index: ',i,' number of elements: ', self.numOfType[i])
            nt.print()

    def nodeTypeExist(self, node):
        # node should be dict
        # key is proeprty
        # value is the literal value
        for node_t in self.nodetypes:
            if node_t.isTypeOf(node):
                return True
        return False

    # compute f1 score given the ground truth node type set
    # TODO extend the mandatory properties checking
    # TODO further optimize using hashing
    def computeF1score(self, gt_nodetypeset: NodeTypeSet, ignoreDatatype=False,CONSIDER_EDGES=True, **kwargs):
        ANALYSE_MODE=kwargs.get('ANALYSE_MODE',False)
        true_pos = false_pos = false_neg = 0
        tp_count = [0 for i in gt_nodetypeset.nodetypes]
        if kwargs.get('PRINT',False):
            print('computing node f1 score ignoredatatype',ignoreDatatype,'consider edges',CONSIDER_EDGES)
        for nt in self.nodetypes:
            exist = 0
            for gt_ind, gt_nt in enumerate(gt_nodetypeset.nodetypes):
                if ignoreDatatype:
                    if nt.equalAttributes(gt_nt):
                        exist = 1
                        tp_count[gt_ind] += 1
                else:
                    if nt.equal(gt_nt, CONSIDER_EDGES=CONSIDER_EDGES):
                        exist = 1
                        tp_count[gt_ind] += 1
            if exist == 1:
                true_pos += 1
            else:
                false_pos += 1
                if ANALYSE_MODE:
                    print('FALSE POSITIVE')
                    nt.print(True)
        for gt_nt in gt_nodetypeset.nodetypes:
            exist = 0
            for nt in self.nodetypes:
                if ignoreDatatype:
                    if gt_nt.equalAttributes(nt):
                        exist = 1
                        break
                else:
                    if gt_nt.equal(nt, CONSIDER_EDGES=CONSIDER_EDGES):
                        exist = 1
                        break
            if exist == 0:
                false_neg += 1
                if ANALYSE_MODE:
                    print('false negative')
                    gt_nt.print(True)

        weighted = kwargs.get('WEIGHTED',None)
        if weighted:
            if kwargs.get('PRINT', False):
                print('duplicate TP count: ', tp_count)
            tp_count = [1/i if i!=0 else 0 for i in tp_count]
            false_pos += sum([1 if i>0 else 0 for i in tp_count])-sum(tp_count)
            true_pos = sum(tp_count)

        if kwargs.get('PRINT',False):
            print('tp: ',true_pos, 'fn:' , false_neg, 'fp: ',false_pos)
        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        f1 = 0
        if precision+recall!=0:
            f1 = 2 * (precision*recall) / (precision+recall)

        if kwargs.get('PRINT',True):
            print('precision: ', precision)
            print('recall: ', recall)
            print('f1_score: ', f1)
        return precision, recall, f1

    # def computeF1scoreIgnoreMand(self,gt_nodetypeset: NodeTypeSet):
    #     true_pos = false_pos = false_neg = 0
    #     for nt in self.nodetypes:
    #         exist = 0
    #         for gt_nt in gt_nodetypeset.nodetypes:
    #             if set(gt_nt.attributes).issubset(set(nt.attributes)):
    #                 exist = 1
    #                 break
    #         if exist == 1:
    #             true_pos += 1
    #         else:
    #             false_pos += 1
    #     for gt_nt in gt_nodetypeset.nodetypes:
    #         exist = 0
    #         for nt in self.nodetypes:
    #             if set(gt_nt.attributes).issubset(set(nt.attributes)):
    #                     exist = 1
    #                 break
    #         if exist == 0:
    #             gt_nt.print()
    #             false_neg += 1
    #     print('TP: ',true_pos,'FN: ', false_neg,'FP: ',false_pos)
    #     precision = true_pos/(true_pos+false_pos)
    #     recall = true_pos/(true_pos+false_neg)
    #     f1 = 2 * (precision*recall) / (precision+recall)
    #     print('precision: ', precision)
    #     print('recall: ', recall)
    #     print('f1_score: ', f1)
    #     return precision, recall, f1

    def atLeastOneMatch(self, node, strict=True):
        return self.nodeTypeExist(node)

    def exactlyOneMatch(self, node, strict=True):
        find_type = 0
        for node_t in self.nodetypes:
            if node_t.isTypeOf(node):
                find_type += 1

        if find_type == 1:
            return True
        return False

    def combinatorialMatch(self, node):
        overmatch_types = []

        for node_t in self.nodetypes:
            # todo duplicate comparison
            # if it strictly conform to one type, match
            if node_t.isTypeOf(node,check_strict=True):
                return True

            if node_t.isTypeOf(node):
                overmatch_types.append(node_t.getAttrs())

        intersec_type = mergeType(overmatch_types)
        if isTypeOf(node, intersec_type, strict=True):
            return True
        else:
            return False

    # return the index of the node type the added node conforming to
    # attr_set should also contains labels {'label':LBL(),'prop':value}
    def add(self, attr_set):
        for ind,node_t in enumerate(self.nodetypes):
            # type with same set of attribute keys found
            if set(node_t.getAttrs()) == set(attr_set):
                t_attrs = node_t.getAttrs()
                for k in t_attrs.keys():
                    if istypeof(attr_set[k],t_attrs[k]):
                        pass
                    else:
                        t_attrs[k].add(type(attr_set[k]))
                self.numOfType[ind] += 1
                return ind
        # create new type
        # new_node_type = NodeType(semantic=self.semantic)
        # for k in attr_set.keys():
        #     new_node_type.attributes[k] = set([type(attr_set[k])])
        new_node_type = None
        if len(set(attr_set))==0:
            print('attr set =0', attr_set)
            new_node_type = NodeType(semantic=self.semantic)
        else:
            new_node_type = NodeType(elem=attr_set,semantic=self.semantic)

        self.nodetypes.append(new_node_type)
        self.numOfType.append(1)
        return len(self.nodetypes)-1


    def addWithLabels(self, attr_set, incoming_edges, outgoing_edges):
        element_labels = set(filter(lambda k:type(attr_set[k])==LBL , attr_set))
        for ind,node_t in enumerate(self.nodetypes):
            if set(node_t.getLabels()) == element_labels:
                addEdgesToNodetype(node_t,incoming_edges,outgoing_edges)
                t_attrs = node_t.getAttrs()
                for k in t_attrs.keys():
                    t_attrs[k].add(type(attr_set[k])) #TODO mandatory type can be inferred
                self.numOfType[ind] += 1
                return ind
        new_node_type = None
        if len(set(attr_set))==0:
            print('attr set =0', attr_set)
            new_node_type = NodeType(semantic=self.semantic)
        else:
            new_node_type = NodeType(elem=attr_set,semantic=self.semantic)
            addEdgesToNodetype(new_node_type,incoming_edges,outgoing_edges)

        self.nodetypes.append(new_node_type)
        self.numOfType.append(1)
        return len(self.nodetypes)-1

    # CONSIDER_EDGES when checking if correpsonding node type already exist
    # when CONSIDER_EDGES=False; it's equal to property-based method;
    # just that the node types contain incoming and outgoing edges
    def addWithEdges(self, attr_set, incoming_edges, outgoing_edges,CONSIDER_EDGES=True):
        if len(set(attr_set))==0:
            print('attr set =0', attr_set)
        new_node_type = NodeType(elem=attr_set,semantic=self.semantic)
        for e in incoming_edges:
            if hasattr(e, 'relationships') and len(e.relationships) > 1:
                print('number of relationships in path greater than 1')
            # TODO only attrs of edge are considered
            # nbr are not considered
            if type(e)==py2neo.Path:
                e = e.relationships[0]
                e[type(e).__name__] = LBL()
            et = ContentType(elem=e)
            new_node_type.incoming_edges.add(et)
        for e in outgoing_edges:
            if hasattr(e,'relationships') and len(e.relationships) > 1:
                print('number of relationships in path greater than 1')
            if type(e)==py2neo.Path:
                e = e.relationships[0]
                e[type(e).__name__] = LBL()
            et = ContentType(elem=e)
            new_node_type.outgoing_edges.add(et)

        for ind,node_t in enumerate(self.nodetypes):
            # type with same set of attribute keys found
            # TODO not ignore datatype
            if node_t.equal(new_node_type,ignore_datatype=True,CONSIDER_EDGES=CONSIDER_EDGES):
                return ind  # type already exist

        self.nodetypes.append(new_node_type)
        self.numOfType.append(1)
        return len(self.nodetypes)-1


    def removeNoElemType(self):
        start_len = len(self.numOfType)
        while 0 in self.numOfType:
            ind = self.numOfType.index(0)
            del self.numOfType[ind]
            del self.nodetypes[ind]
        print('remove ', start_len - len(self.numOfType), ' element')

    # given updated elements, update node types accordingly
    def update(self, attr_set, **kwargs):
        # TODO datatype of property type are ignored here
        created_attr = kwargs.get('created_attr', set())
        removed_attr = kwargs.get('removed_attr', set())
        original_attr = set(attr_set)
        original_attr -= created_attr
        original_attr =  removed_attr.union(original_attr)
        type_exist = 0
        for ind,node_t in enumerate(self.nodetypes):
            if node_t.hasAttrs(original_attr):
                self.numOfType[ind] -= 1
                if self.numOfType[ind] == 0:
                    self.removeNoElemType()
            if node_t.isTypeOf(attr_set):
                self.numOfType[ind] += 1
                type_exist = 1
        if type_exist:
            return
        print('updating type, type does not exist, creating type')
        self.add(attr_set)

    def tojson(self):
        # type_list is a list of types
        res = []
        for t in self.nodetypes:
            # new dict of type, serilized
            type_dict = {}
            t_attrs = t.getAttrs()
            for k in t_attrs:
                type_dict[k] = [i.__name__ for i in list(t_attrs[k])]
            res.append(type_dict)
        return json.dumps(res)

#%%
def mergeType(types : list):
    merged_type = {}
    for t in types:
        for k in t.keys():
            if merged_type.get(k)==None:
                merged_type[k] = set()
            merged_type[k] = merged_type[k].union(t[k])
    return merged_type

class EdgeTypeSet():
    def __init__(self):
        self.edgetypes = []
        self.numOfType = []

    def __getitem__(self, item):
        return self.edgetypes[item]

    def getAttributes(self):
        res = set()
        for et in self.edgetypes:
            res = res.union(set(et.attributes))
        return list(res)

    def mergeByLabelSet(self,IGNORE_NODES=False, **kwargs):
        self.mergeByAttributeSet(mode=['L'],IGNORE_NODES=IGNORE_NODES, **kwargs)
        # new_edgetypes = []
        # label_type_dict = {}
        #
        # for et in self.edgetypes:
        #     lbs = et.getLabels(EDGE_LABELS_ONLY=IGNORE_NODES)
        #     if not lbs in label_type_dict:
        #         label_type_dict[lbs] = et
        #     else:
        #         label_type_dict[lbs].mergeWithAnotherType(et)
        #
        # for lbs in label_type_dict:
        #     new_edgetypes.append(label_type_dict[lbs])
        # self.edgetypes = new_edgetypes

    def mergeByPropertySet(self,IGNORE_NODES=False):
        self.mergeByAttributeSet(mode=['P'],IGNORE_NODES=IGNORE_NODES)
        # new_edgetypes = []
        # prop_type_dict = {}
        #
        # for et in self.edgetypes:
        #     lbs = et.getProperties(EDGE_PROP_ONLY=IGNORE_NODES)
        #     if not lbs in prop_type_dict:
        #         prop_type_dict[lbs] = et
        #     else:
        #         prop_type_dict[lbs].mergeWithAnotherType(et)
        #
        # for lbs in prop_type_dict:
        #     new_edgetypes.append(prop_type_dict[lbs])
        # self.edgetypes = new_edgetypes

    def mergeByAttributeSet(self,mode=[],IGNORE_NODES=False, **kwargs):
        # old2new = kwargs.get('old2new',{})
        # old_nts = kwargs.get('old_nodetypes', [])
        print('merge edge by '+';'.join(mode),' ignore src and tar: ', IGNORE_NODES)
        new_edgetypes = []
        attr_type_dict = {}

        for i,et in enumerate(self.edgetypes):
            attrs = ''
            if 'L' in mode:
                attrs = attrs + et.getLabels(EDGE_LABELS_ONLY=IGNORE_NODES) + 'ENDLABEL'
            if 'P' in mode:
                attrs = attrs + et.getProperties(EDGE_PROP_ONLY=IGNORE_NODES)
            if not attrs in attr_type_dict:
                attr_type_dict[attrs] = et
            else:
                attr_type_dict[attrs].mergeWithAnotherType(et)

        for lbs in attr_type_dict:
            new_edgetypes.append(attr_type_dict[lbs])
        self.edgetypes = new_edgetypes


    def getListOfEdgeInfo(self):
        res = []
        for et in self.edgetypes:
            txt = 'edge: ,' + \
            ','.join(list(et.getAttrs(return_key=True))) + ',\n|src: ,' + \
            ','.join(list(et.src_type.getAttrs(return_key=True))) + ',\n|tar: ,' + \
            ','.join(list(et.target_type.getAttrs(return_key=True)))
            res.append(txt)
        return res
    # merge only if source(i), source(j) in the same cluster, and
    # target(i), target(j) in the same cluster
    # and Jarccard_sim ( edge(i),  edge(j) )  > threshold
    def mergeByJaccardSim(self, nodeMap, theta=0.5, idf=True,return_sim_matrix=False,**kwargs):
        num_et = len(self.edgetypes)
        # sim_matrix = [[0 for i in range(num_et)] for j in range(num_et)]
        oldid2newid = kwargs.get('oldId2newId',{})
        sim_matrix = lil_matrix((num_et,num_et))
        return_matrix = lil_matrix((num_et,num_et))
        print('merge edge by ',kwargs.get('mode','attr'), ' theta: ', theta)

        for i in range(num_et):
            src_i = self.edgetypes[i].src_type.toHashStr()
            tar_i = self.edgetypes[i].target_type.toHashStr()
            e_i = self.edgetypes[i]

            for j in range(i+1, num_et):
                src_j = self.edgetypes[j].src_type.toHashStr()
                tar_j = self.edgetypes[j].target_type.toHashStr()
                e_j = self.edgetypes[j]

                return_matrix[i,j] = return_matrix[j,i] = e_i.jaccardSim(e_j,mode=kwargs.get('mode','attr'))
                if nodeMap[src_i] == nodeMap[src_j] and \
                    nodeMap[tar_j]==nodeMap[tar_i] and \
                    e_i.jaccardSim(e_j) > theta:
                    sim_matrix[i,j] = sim_matrix[j,i] = 1

        if return_sim_matrix:
            return return_matrix
        # sim_matrix = csr_matrix(sim_matrix)
        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)

        if n_components == num_et:
            print('no edge cluster found')
            return
        else:
            print('%d clusters found out of %d nodes' % (n_components, num_et))

        new_edge_types = []
        old_et = np.array(self.edgetypes)
        for i in range(n_components):
            et = mergeContentType(old_et[labels==i])
            new_edge_types.append(et)
            new_id = len(new_edge_types)-1
            for id, lb in enumerate(labels):
                if lb==i:
                    oldid2newid[id] = new_id
        self.edgetypes = new_edge_types

#   TODO idf is not supported
    def mergeByDBScan(self, nodeMap, idf=False):
        print('start merging edge type by DBScan, with idf',idf)
        num_et = len(self.edgetypes)
        dbScan = DBSCAN(min_samples=1,metric='cosine')
        attr_list = self.getAttributes()
        attrs = [list(set(et.getAttrs())) for et in self.edgetypes]
        X = []
        x = []
        sample_weight = []
        for ind, listOfAttr in enumerate(attrs):
            if not idf:
                x = [1 if i in listOfAttr else 0 for i in attr_list]
            else:
                print('idf not supported')
                return
            sample_weight.append(self.numOfType[ind]) # use the number of elements as
            X.append(x)
        sample_weight = np.array(sample_weight)
        dbScan.fit(X, sample_weight=sample_weight)
        labels = dbScan.labels_

        sim_matrix = [[0 for i in self.edgetypes] for j in self.edgetypes]
        for i in range(num_et):
            src_i = self.edgetypes[i].src_type.toHashStr()
            tar_i = self.edgetypes[i].target_type.toHashStr()
            e_i = self.edgetypes[i]

            for j in range(i+1, num_et):
                src_j = self.edgetypes[j].src_type.toHashStr()
                tar_j = self.edgetypes[j].target_type.toHashStr()
                e_j = self.edgetypes[j]

                if nodeMap[src_i] == nodeMap[src_j] and \
                        nodeMap[tar_j]==nodeMap[tar_i] and \
                        labels[i]==labels[j]:
                    sim_matrix[i][j] = sim_matrix[j][i] = 1

        sim_matrix = csr_matrix(sim_matrix)
        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)

        if n_components == num_et:
            print('no edge cluster found')
            return
        else:
            print('%d clusters found out of %d nodes' % (n_components, num_et))

        new_edge_types = []
        old_et = np.array(self.edgetypes)
        for i in range(n_components):
            et = mergeContentType(old_et[labels==i])
            new_edge_types.append(et)
        self.edgetypes = new_edge_types

    def mergeByEdgeSim(self, return_sim_matrix=False, theta=0.5,
                       coef=[1,1,1],normalized_coef = True):
        if normalized_coef:
            coef = [i/sum(coef) for i in coef]
        num_et = len(self.edgetypes)
        sim_matrix = lil_matrix((num_et,num_et))
        src_matrix = lil_matrix((num_et,num_et))
        tar_matrix = lil_matrix((num_et,num_et))
        for i in range(num_et):
            src_i = self.edgetypes[i].src_type
            tar_i = self.edgetypes[i].target_type
            e_i = self.edgetypes[i]

            for j in range(i + 1, num_et):
                src_j = self.edgetypes[j].src_type
                tar_j = self.edgetypes[j].target_type
                e_j = self.edgetypes[j]
                # sim_matrix[i,j] = sim_matrix[j,i] = jaccard(src_i,src_j) + jaccard(tar_i,tar_j) + \
                #             jaccard(e_i.getAttrs(),e_j.getAttrs())
                src_matrix[i,j]= src_matrix[j,i] = jaccard(src_i,src_j)
                tar_matrix[i, j] = tar_matrix[j, i] = jaccard(tar_i, tar_j)
                sim_matrix[i, j] = sim_matrix[j, i] = jaccard(e_i.getAttrs(),e_j.getAttrs())
                sim_matrix[i, j] = sim_matrix[j, i] = \
                    coef[0] * src_matrix[i, j] + coef[1] * sim_matrix[i, j] + coef[2] * tar_matrix[i, j]

                sim_matrix[i,j] = sim_matrix[j,i] = 1 if sim_matrix[i,j]>theta else 0

        if return_sim_matrix:
            return src_matrix,tar_matrix,sim_matrix

        n_components, labels = connected_components(
            sim_matrix, directed=True, return_labels=True)

        if n_components == num_et:
            print('no edge cluster found')
            return
        else:
            print('%d clusters found out of %d edges' % (n_components, num_et))

        new_edge_types = []
        old_et = np.array(self.edgetypes)
        for i in range(n_components):
            et = mergeContentType(old_et[labels==i])
            new_edge_types.append(et)
        self.edgetypes = new_edge_types


    def computeF1score(self, gt_edgetypeset,ignoreDatatype=False,CONSIDER_EDGES_OF_SRC_TAR=False, **kwargs):
        ANALYSE_MODE = kwargs.get('ANALYSE_MODE', False)
        true_pos = false_pos = false_neg = 0
        tp_count = [0 for i in gt_edgetypeset.edgetypes]
        for et in self.edgetypes:
            exist = 0
            for gt_ind, gt_et in enumerate(gt_edgetypeset.edgetypes):
                if ignoreDatatype:
                    if et.equalAttributes(gt_et):
                        exist = 1
                        tp_count[gt_ind] += 1
                else:
                    if et.equal(gt_et,CONSIDER_EDGES=CONSIDER_EDGES_OF_SRC_TAR):
                        exist = 1
                        tp_count[gt_ind] += 1
            if exist == 1:
                true_pos += 1
            else:
                false_pos += 1
                if ANALYSE_MODE:
                    print('FALSE POSITIVE')
                    et.print()
        for gt_et in gt_edgetypeset.edgetypes:
            exist = 0
            for et in self.edgetypes:
                if ignoreDatatype:
                    if gt_et.equalAttributes(et):
                        exist = 1
                        break
                else:
                    if gt_et.equal(et,CONSIDER_EDGES=CONSIDER_EDGES_OF_SRC_TAR):
                        exist = 1
                        break
            if exist == 0:
                false_neg += 1
                if ANALYSE_MODE:
                    print('FALSE NEGATIVE')
                    gt_et.print()

        weighted = kwargs.get('WEIGHTED',None)
        if weighted:
            if kwargs.get('PRINT', False):
                print('duplicate TP count: ', tp_count)
            tp_count = [1/i if i!=0 else 0 for i in tp_count]
            false_pos += sum([1 if i>0 else 0 for i in tp_count])-sum(tp_count)
            true_pos = sum(tp_count)

        if kwargs.get('PRINT',False):
            print('tp: ',true_pos, 'fn:' , false_neg, 'fp: ',false_pos)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 0
        if precision+recall != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        if kwargs.get('PRINT',True):
            print('precision: ', precision)
            print('recall: ', recall)
            print('f1_score: ', f1)
        return precision, recall, f1

    def __len__(self):
        return len(self.edgetypes)

    def print(self):
        print('number of edge types ', len(self.edgetypes))
        for ind,et in enumerate(self.edgetypes):
            print('index ',ind)
            et.print()

    def atLeastOneMatch(self, relation, strict=True):
        for e_type  in self.edgetypes:
            if e_type.isTypeOf(relation):
                return True
        return False

    def exactlyOneMatch(self, relation, strict=True):
        find_type = 0
        for e_type in self.edgetypes:
            if e_type.isTypeOf(relation):
                find_type += 1
                if find_type > 1:
                    return False

        return True if find_type==1 else False

    def combinatorialMatch(self, relation):
        # the source and target nodes should already checked in the nodes comformance checking process
        return self.atLeastOneMatch(relation, True)

    def add(self, relation, src_t = None, tar_t = None):
        for i,e_type in enumerate(self.edgetypes):
            if e_type.isTypeOf(relation):
                self.numOfType[i] += 1
                return i
            elif e_type.equalAttributes(relation):
                e_type.addDataTypeFromElem(relation)
                self.numOfType[i] += 1
                return i

        # create new type and add
        new_edge_type = EdgeType(relation=relation)
        if src_t!=None and tar_t!=None:
            new_edge_type.src_type = src_t
            new_edge_type.target_type = tar_t
        self.edgetypes.append(new_edge_type)
        self.numOfType.append(1)
        return len(self.edgetypes)-1

    # add a relation, consider label equivalence
    # equivalence measure only the edges labels themself, no src labels
    def addWithLabels(self,relation,src_t, tar_t):

        element_labels = set(filter(lambda k:type(relation[k])==LBL , relation))
        for i,e_type in enumerate(self.edgetypes):
            if e_type.src_type==src_t  and e_type.target_type==tar_t and \
                    set(e_type.getLabels(EDGE_LABELS_ONLY=True, RETURN_STR=False)) == element_labels:
                e_type.addDataTypeFromElem(relation, IGNORE_NODES = True,ADD_TYPE_IF_NOT_EXIST=True)
                self.numOfType[i] += 1
                return i

        # create new type and add
        new_edge_type = EdgeType(relation=relation)
        new_edge_type.src_type, new_edge_type.target_type = src_t, tar_t
        self.edgetypes.append(new_edge_type)
        self.numOfType.append(1)
        return len(self.edgetypes)-1

    def addWithProperties(self,relation,src_t, tar_t):

        element_props = set(filter(lambda k:type(relation[k])!=LBL , relation))
        for i,e_type in enumerate(self.edgetypes):
            if e_type.src_type==src_t  and e_type.target_type==tar_t and \
                    set(e_type.getProperties(EDGE_PROP_ONLY=True, RETURN_STR=False)) == element_props:
                e_type.addDataTypeFromElem(relation, IGNORE_NODES = True,ADD_TYPE_IF_NOT_EXIST=True)
                self.numOfType[i] += 1
                return i

        # create new type and add
        new_edge_type = EdgeType(relation=relation)
        new_edge_type.src_type, new_edge_type.target_type = src_t, tar_t
        self.edgetypes.append(new_edge_type)
        self.numOfType.append(1)
        return len(self.edgetypes)-1

if __name__ == '__main__':
    # for test
    a = Node("Person", name="Alice", age=33)
    b = Node("Prof", status="teacher", uni='tu/e')
    c = Node('City', name='Gent', url='http.cal')
    workin = Relationship(a, 'workIn', c, start='12-02-2021')
    workinTwo = Relationship(a, 'workIn', c, end='2031')
    v = Node('Person', 'Prof', name='Jan', age=132, status='lectures', uni='ugent')
    # KNOWS = Relationship.type("KNOWS")
    nodetypeset = NodeTypeSet()
    edgetypeset = EdgeTypeSet()
    a = preprocessPy2neoNode(a)
    b = preprocessPy2neoNode(b)
    c = preprocessPy2neoNode(c)
    v = preprocessPy2neoNode(v)
    nodetypeset.add(a)
    nodetypeset.add(c)
    print('combine match', str(nodetypeset.combinatorialMatch(v)))
    edgetypeset.add(workin)
    edgetypeset.add(workinTwo)
    v1 = Node('Person', name='Jan', age=32)
    v2 = Node('City', name='London', url='www.london.org')
    e1 = Relationship(v1, 'workIn', v2, start='2020', end='2002')
    print(str(edgetypeset.exactlyOneMatch(e1, True)))
