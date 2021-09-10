#%%

from neo4j import GraphDatabase
from py2neo import *
import json
from ElementType import *



# if type of variable x is a type in the list
def istypeof(x,li):
    for t in li:
        if type(x) == t:
            return True
    return False

# check if elem is conformed to elem_type
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

# turn labels to attributes
def preprocessPy2neoNode(node):
    node_dict = dict(node)
    # convert labels to attribute
    labels = str(node.labels).split(':')[1:]
    for l in labels:
        node_dict[l] = LBL()
    return node_dict


class NodeTypeSet:
    def __init__(self):
        self.nodetypes = []
        self.numOfType = [] # keep count of the instnaces per type

    def nodeTypeExist(self, node, strict=True):
        # node should be dict
        # key is proeprty
        # value is the literal value
        for node_t in self.nodetypes:
            if isTypeOf(node, node_t, strict=strict):
                return True
        return False

    def atLeastOneMatch(self, node, strict=True):
        return self.nodeTypeExist(node,strict)

    def exactlyOneMatch(self, node, strict=True):
        find_type = 0
        for node_t in self.nodetypes:
            if isTypeOf(node, node_t, strict):
                find_type += 1

        if find_type == 1:
            return True

        return False

    def combinatorialMatch(self, node):
        overmatch_types = []

        for node_t in self.nodetypes:
            # todo duplicate comparison
            # if it strictly conform to one type, match
            if isTypeOf(node,node_t, strict=True):
                return True

            if isTypeOf(node,node_t, strict=False):
                overmatch_types.append(node_t)

        intersec_type = mergeType(overmatch_types)
        if isTypeOf(node, intersec_type, strict=True):
            return True
        else:
            return False

    def add(self, attr_set):
        for ind,node_t in enumerate(self.nodetypes):
            # type with same set of attribute keys found
            if set(node_t) == set(attr_set):
                for k in node_t.keys():
                    if istypeof(attr_set[k],node_t[k]):
                        pass
                    else:
                        node_t[k].add(type(attr_set[k]))
                    self.numOfType[ind] += 1
                return
        # create new type
        new_node_type = {}
        for k in attr_set.keys():
            new_node_type[k] = set([type(attr_set[k])])
        self.nodetypes.append(new_node_type)
        self.numOfType.append(1)

    def tojson(self):
        # type_list is a list of types
        res = []
        for t in self.nodetypes:
            # new dict of type, serilized
            type_dict = {}
            for k in t:
                type_dict[k] = [i.__name__ for i in list(t[k])]
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

def makeType(elem):
    typ = {}
    for k in elem.keys():
        typ[k] = set()
        typ[k].add(type(elem[k]))
    return typ

def addDataType(elem,elem_type):
    for k in elem.keys():
        if type(elem[k]) not in elem_type[k]:
            elem_type[k].add(type(elem[k]))

class EdgeTypeSet():
    def __init__(self):
        self.edgetypes = []
        self.source_node_type = []
        self.target_node_type = []

    def atLeastOneMatch(self, relation, strict=True):
        source_node = relation.nodes[0]
        target_node = relation.nodes[1]
        edge = dict(relation)
        edge[type(relation).__name__] = LBL()
        for i in range(len(self.edgetypes)):
            et = self.edgetypes[i]
            st = self.source_node_type[i]
            tt = self.target_node_type[i]

            if isTypeOf(edge, et, strict) and isTypeOf(source_node,st, strict) \
                    and isTypeOf(target_node, tt, strict):
                return True
        return False


    def exactlyOneMatch(self, relation, strict=True):
        source_node = relation.nodes[0]
        target_node = relation.nodes[1]
        edge = dict(relation)
        edge[type(relation).__name__] = LBL()
        find_type = 0
        for i in range(len(self.edgetypes)):
            et = self.edgetypes[i]
            st = self.source_node_type[i]
            tt = self.target_node_type[i]

            if isTypeOf(edge, et, strict) and isTypeOf(source_node,st, strict) \
                    and isTypeOf(target_node, tt, strict):
                find_type += 1
        return True if find_type==1 else False

    def combinatorialMatch(self, relation):
        # the source and target nodes should already checked in the nodes comformance checking process
        return self.atLeastOneMatch(relation, True)

    def add(self, relation):
        source_node = relation.nodes[0]
        target_node = relation.nodes[1]
        edge = dict(relation)
        edge[type(relation).__name__] = LBL()
        for i in range(len(self.edgetypes)):
            et = self.edgetypes[i]
            st = self.source_node_type[i]
            tt = self.target_node_type[i]

            if isTypeOf(edge, et) and isTypeOf(source_node,st) \
                    and isTypeOf(target_node, tt):
                return
            # add data type
            if set(edge)==set(et) and set(source_node)==set(st) \
                    and set(target_node)==set(tt):
                addDataType(edge,et)
                addDataType(source_node,st)
                addDataType(target_node,tt)
        # add new type
        new_edge_type = makeType(edge)
        new_source_node = makeType(source_node)
        new_target_node = makeType(target_node)
        self.edgetypes.append(new_edge_type)
        self.source_node_type.append(new_source_node)
        self.target_node_type.append(new_target_node)
    def print(self):
        print('Number of edge types', len(self.edgetypes))
        for i in range(len(self.edgetypes)):
            print('Source',self.source_node_type[i])
            print('edge',self.edgetypes[i])
            print('target',self.target_node_type[i])
            print('\n')





if __name__ == '__main__':
    # for test
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
    nodetypeset.combinatorialMatch(v)
    edgetypeset.add(workin)
    edgetypeset.add(workinTwo)
    v1 = Node('Person', name='Jan', age=32)
    v2 = Node('City', name='London', url='www.london.org')
    e1 = Relationship(v1, 'workIn', v2, start='2020', end='2002')
    edgetypeset.exactlyOneMatch(e1, True)
