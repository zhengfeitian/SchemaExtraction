import copy

ATTITUDE_CLOSE = 0
ATTITUDE_COMB = 1
ATTITUDE_OPEN = 2

ALOM = 'at_least_one_match'
EOM = 'exactly_one_match'
COMB = 'combinatorial'

# simple relationship class
class SimpleRel(dict):
    def __init__(self,*arg,**kw):
        self.nodes = [{},{}] #nodes[0] is src, nodes[1] is target

        super(SimpleRel, self).__init__(*arg, **kw)
    def __eq__(self,other):
        if set(self.nodes[0])==set(other.nodes[0]) and \
                set(self.nodes[1])==set(other.nodes[1]) and \
                set(self)==set(other):
            return True
        return False
    def getKey(self):
        return set(self)
    def mapToStr(self):
        s = ','.join(sorted(list(set(self.nodes[0]))))
        t = ','.join(sorted(list(set(self.nodes[1]))))
        e = ','.join(sorted(list(set(self))))
        return 'src:'+s+'|tar'+t+'|edge'+e

# if type of variable x is a type in the list
def istypeof(x, li):
    for t in li:
        if type(x) == t:
            return True
    return False

# given a dict, return true if it's element
def isElement(d):
    for k in d:
        if type(d[k])==set:
            return False
    return True

# merge a list of types
def mergeContentType(types):
    t = copy.deepcopy(types[0])
    for i in range(1,len(types)):
        t.mergeWithAnotherType(types[i]) #TODO
    return t

# turn labels to attributes
def preprocessPy2neoNode(node):
    if type(node)==dict:
        return node
    node_dict = dict(node)
    # convert labels to attribute
    labels = str(node.labels).split(':')[1:]
    for l in labels:
        node_dict[l] = LBL()
    return node_dict

#merge a list of edge types
def mergeEdgeType(eTypes):
    t = copy.deepcopy(eTypes[0])
    for i in range(1, len(eTypes)):
        src_t = eTypes[i].src_type
        tar_t = eTypes[i].target_type

        t.src_type.mergeWithAnotherType(src_t)
        t.target_type.mergeWithAnotherType(tar_t)
        t.mergeWithAnotherType(eTypes[i])
    return t

class LBL:
    def LBL(self):
        self.__name__ = 'LBL'


class ContentType:
    def __init__(self, *args, **kwargs):
        self.attributes = {}
        self.mandatory_attrs = set()
        self.attitude = kwargs.get('attitude', ATTITUDE_CLOSE)
        elem = kwargs.get('elem',None)
        semantic = kwargs.get('semantic', ALOM)
        content_type = kwargs.get('attr_type',None)
        if content_type!=None:
            if isElement(content_type):
                elem,content_type = content_type,None
        if semantic == COMB:
            self.attitude = ATTITUDE_COMB
        if elem!=None:
            self.initTypeFromElem(elem)
        if content_type!=None:
            self.attributes = content_type

    def removeLabelsFromAttr(self):
        remove_list = []
        for k in self.attributes:
            if list(self.attributes[k])[0]==LBL:
                remove_list.append(k)
        for k in remove_list:
            del self.attributes[k]
            self.mandatory_attrs.discard(k)
            print('delete ',k)
        # print('after delete')
        # self.print()


    # check if self contain a set of attr key
    def hasAttrs(self,d):
        if set(self.attributes) == set(d):
            return True
        return False
    def getAttrs(self,return_key=False):
        if return_key:
            return set(self.attributes)
        return self.attributes

    # compute jaccard similarity with antoher type
    # if idf_dict is given, compute the idf weighted simlarity
    def jaccardSim(self, t, idf_dict=None,mode='attr'):
        t_attrs = t.getAttrs()
        # treat labels as properties
        if mode=='attr':
            s1 = set(self.attributes)
            s2 = set(t.getAttrs())
        elif mode=='p':
            s1 = set(self.getProperties())
            s2 = set(t.getProperties())
        intersec = s1.intersection(s2)
        uni = s1.union(s2)
        if idf_dict != None:
            numerator = denominator = 0
            for i in intersec:
                numerator += idf_dict[i]
            for i in uni:
                denominator += idf_dict[i]
            return numerator/denominator
        return len(s1.intersection(s2))/len(s1.union(s2))

    def getLabels(self):
        res = set()
        for k in self.attributes:
            if LBL in self.attributes[k]:
                res.add(k)
        return sorted(list(res))

    def equalLabels(self, other):
        return set(self.getLabels())==set(other.getLabels())

    # return a sorted list of property keys
    def getProperties(self):
        res = set()
        for k in self.attributes:
            if not LBL in self.attributes[k]:
                res.add(k)
        return sorted(list(res))

    def initTypeFromElem(self,elem):
        elem = dict(elem)
        elemType = {}
        for k in elem.keys():
            elemType[k] = set([type(elem[k])])
        self.attributes = elemType

    # elem and self.attributes keys are same
    # extract type from elem and merge it with current type
    def addDataTypeFromElem(self,elem, ADD_TYPE_IF_NOT_EXIST=False):
        # for k in self.attributes: #old version
        for k in elem:
            if ADD_TYPE_IF_NOT_EXIST:
                if not k in self.attributes:
                    self.attributes[k] = set()
            if not type(elem[k]) in self.attributes[k]:
                self.attributes[k].add(type(elem[k]))

    # this should only be called when doing nodetypeset.merbylabel() as it changes the mandatory attr
    # merge current content type with another type
    # all attrs of t will be added
    # merge with another type t
    # if self does not have mandatory attr, the attr intersection of t and self are set as self.mandatory attr
    # if self has mandatory attr, attr will be remove if it does not appear in t's attrs
    def mergeWithAnotherType(self, t):
        # if there's no mandatory attrs in self
        # then take intersection as mandatory attrs
        for mand_attr in t.mandatory_attrs:
            if not mand_attr in self.attributes:
                print('cannot be merged because of mandatory type',self.attributes,t.attributes)
                return
        if not self.mandatory_attrs:
            self.mandatory_attrs = set(self.attributes).intersection(set(t.attributes))
        # if mandatory attributes exist, remove mand attrs not in t
        else:
            rm_attrs = set()
            for mand_attr in self.mandatory_attrs:
                if not mand_attr in t.attributes:
                    rm_attrs.add(mand_attr)
            self.mandatory_attrs -= rm_attrs


        for k in t.attributes:
            if k in self.attributes:
                self.attributes[k].union(t.attributes[k])
            else:
                self.attributes[k] = t.attributes[k].copy()


    # check if the elem has all the mandatory attributes
    def checkMandatoryAttr(self, elem):
        for attr in self.mandatory_attrs:
            if attr not in elem:
                return False
            if elem[attr] not in self.attributes[attr]:
                return False
        return True
    # if node is type of the nodetype
    def isTypeOf(self, elem, check_strict=False):
        if self.checkMandatoryAttr(elem) == False:
            return False

        elem_type = self.attributes
        strict = (self.attitude == ATTITUDE_CLOSE)
        if check_strict:
            strict = True

        if strict and set(elem) == set(elem_type):
            for k in elem.keys():
                if not istypeof(elem[k], elem_type[k]):
                    return False
            return True
        if strict == False and set(elem) >= set(elem_type):
            for k in elem_type.keys():
                if not istypeof(elem[k], elem_type[k]):
                    return False
            return True

        return False

    # check is content type and elem share the same  set of attribute
    # elem should be dictonary
    def equalAttributes(self, elem):
        if issubclass(type(elem),ContentType):
            return set(elem.attributes)==set(self.attributes)
        else:
            return set(elem)==set(self.attributes)

    # ct is another content type
    def equalProperties(self, ct):
        for k in self.attributes:
            # if k is not label
            if not LBL in self.attributes[k]:
                if not k in ct.attributes:
                    return False
                if self.attributes[k]!=ct.attributes[k]:
                    return False
        for k in ct.attributes:
            if not LBL in ct.attributes[k]:
                if not k in self.attributes:
                    return False
                if self.attributes[k]!=ct.attributes[k]:
                    return False
        return True

    def __str__(self):
        return str(self.attributes)

    def toHashStr(self):
        res = ''
        for k in sorted(self.attributes):
            types = '|'.join([i.__name__ for i in list(self.attributes[k])])
            res = res + k + ':' + types + ';'
        return res

    # TODO mandatory attribute not considered
    def __hash__(self):
        return hash(self.toHashStr())

    # if two types are equal
    # TODO extend to mandatory attributes
    def equal(self, ct, ignore_datatype=True):
        # all keys should be the same
        if set(ct.attributes)!=set(self.attributes):
        # if not set(ct.attributes).issubset(self.attributes):
            return False
        # all attributes type should be the same
        if not ignore_datatype:
            for k in self.attributes:
                if k in ct.attributes:
                    if self.attributes[k]!=ct.attributes[k]:
                        return False
                else:
                    return False
        # else:
        #     print('ignore attributes type only comparing attribute keys')
        return True

    def __eq__(self, other):
        if other==None:
            return False
        return self.equal(other, False)

    def print(self):
        print('------------Content Type----------------')
        d = {}
        for k in self.attributes:
            if k in self.mandatory_attrs:
                d[k] = '|'.join([i.__name__ for i in list(self.attributes[k])])
            else:
                d[k + '?'] = '|'.join([i.__name__ for i in list(self.attributes[k])])

        for k in d:
            print("{:<8} {:<15}".format(k,d[k]))


class NodeType(ContentType):
    def __init__(self, node=None, *args, **kwargs):

        self.outgoing_edges = set()   # list of ContentType
        self.incoming_edges = set()
        self.mandatory_edge_type = {}
        if type(node) == dict:
            super().__init__(elem=node, *args, **kwargs)
        # node = elem
        elif node!=None:
            node_dict = dict(node)
            if hasattr(node,'labels'):
                node_dict = preprocessPy2neoNode(node)
            super().__init__(elem=node_dict, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def print(self, PRINT_EDGES=False):
        super(NodeType, self).print()

        if PRINT_EDGES:
            print('incoming edges')
            for inc in self.incoming_edges:
                print(inc.attributes)
            print('outgoing  edges')
            for inc in self.outgoing_edges:
                print(inc.attributes)

    # get neighour in object graph,
    # level 1 is edges
    # mode='T' topology based, only return 1,0 indicating existence of edges
    def getConnectedEdges(self,mode='T',level=1):
        inc = sorted([str(i) for i in self.incoming_edges])
        out = sorted([str(i) for i in self.outgoing_edges])
        if mode=='T':
            inc = 'exist' if len(inc)>0 else 'notexist'
            out = 'exist' if len(out)>0 else 'notexist'
            return '|INC|'+ inc + '$' + '|OUT|' + out
        else:
            return '|INC|'.join(inc) + '$' + '|OUT|'.join(out)

    # TODO  not ignore data type
    def equal(self, nt, ignore_datatype=True,CONSIDER_EDGES=False):
        if CONSIDER_EDGES:
            if hasattr(self,'incoming_edges') and hasattr(nt,'incoming_edges'):
                if self.incoming_edges != nt.incoming_edges or \
                    self.outgoing_edges != nt.outgoing_edges:
                    return False
        return super().equal(nt, ignore_datatype=ignore_datatype)

    def NodeType(self, node):
        self.initTypeFromElem(node)

    def mergeWithAnotherType(self, t):
        super(NodeType, self).mergeWithAnotherType(t)
        if self.hasAttrs('incoming_edges') and self.hasAttrs('outgoing_edges') \
                and t.hasAttrs('incoming_edges') and t.hasAttrs('outgoing_edges'):
            self.incoming_edges = self.incoming_edges.union(t.incoming_edges)
            self.outgoing_edges = self.outgoing_edges.union(t.outgoing_edges)

    # return a str encoding attributes and their types
    # if two hashstr euqal, two types are equal
    # TODO mandatory attrs
    def toHashStr(self,without_type=False):
        if without_type:
            return ';'.join(set(self.attributes))
        return 'Node|' + super(NodeType, self).toHashStr()

    def isTypeOf(self, node, check_strict=False):
        if type(node) == dict:
            return super(NodeType, self).isTypeOf(node, check_strict)
        else:
            node_dict = dict(node)
            labels = str(node.labels).split(':')[1:]
            for l in labels:
                node_dict[l] = LBL()
            return super().isTypeOf(node_dict, check_strict)



class EdgeType(ContentType):
    def __init__(self,  relation=None, src_type=None, target_type=None, 
                 relation_type=None, *args, **kwargs):
        self.src_type = None
        self.target_type = None
        # print('init edge type')
        # relation = kwargs.get('relation',None)
        if relation!=None:
            # print('relation not None', relation)
            source = relation.nodes[0]
            target = relation.nodes[1]
            if type(source)==dict and type(target)==dict:
                self.src_type = NodeType(elem=source)
                self.target_type = NodeType(elem=target)
            elif type(source)==NodeType and type(target)==NodeType:
                self.src_type = source
                self.target_type = target
            else:
                self.src_type = NodeType(node=source)
                self.target_type = NodeType(node=target)
            super().__init__(elem=relation,*args, **kwargs)
        elif src_type!=None and target_type!=None and relation_type!=None:
            if type(src_type)==dict:
                src_type = NodeType(attr_type=src_type)
            if type(target_type)==dict:
                target_type = NodeType(attr_type=target_type)
            self.src_type = src_type
            self.target_type = target_type

            super(EdgeType, self).__init__(attr_type=relation_type)
        else:
            print('relation is None')
            super().__init__(*args, **kwargs)
        self.nodes = [self.src_type, self.target_type]

    def getLabels(self,EDGE_LABELS_ONLY=False,RETURN_STR=True):
        src_labels = 'source'+':'.join(self.src_type.getLabels())
        edg_labels = 'edge' + ':'.join(super(EdgeType, self).getLabels())
        tgt_labels = 'tartget' + ':'.join(self.target_type.getLabels())
        if EDGE_LABELS_ONLY:
            if RETURN_STR:
                return edg_labels
            else:
                return super(EdgeType, self).getLabels()
        return '|'.join([src_labels,edg_labels,tgt_labels])

    def getProperties(self,EDGE_PROP_ONLY=False, RETURN_STR=True):
        src_props = 'source'+':'.join(self.src_type.getProperties())
        edg_props = 'edge' + ':'.join(super(EdgeType, self).getProperties())
        tgt_props = 'tartget' + ':'.join(self.target_type.getProperties())
        if EDGE_PROP_ONLY and RETURN_STR:
            return edg_props
        if EDGE_PROP_ONLY and RETURN_STR==False:
            return super(EdgeType,self).getProperties()
        return '|'.join([src_props,edg_props,tgt_props])

    # if type of relation is self
    def isTypeOf(self, relation, check_strict=False):
        source = relation.nodes[0]
        target = relation.nodes[1]
        if type(source)==NodeType and type(target)==NodeType:
            if self.src_type.equal(source) \
                    and self.target_type.equal(target) \
                    and super().isTypeOf(elem=relation):
                return True
        elif self.src_type.isTypeOf(source) \
                and self.target_type.isTypeOf(target) \
                and super().isTypeOf(elem=relation):
            return True
        return False

    def equalAttributes(self, relation):
        s = t = None
        if issubclass(type(relation),EdgeType):
            s = relation.src_type
            t = relation.target_type
        else:
            s = relation.nodes[0]
            t = relation.nodes[1]
        if self.src_type.equalAttributes(s) and \
            self.target_type.equalAttributes(t) and \
            super().equalAttributes(relation):
            return True
        return False

    # check if two relation types are equal
    # TODO check mandatory attributes
    def equal(self, eType, ignore_datatype=False,CONSIDER_EDGES=True):
        if self.src_type.equal(eType.src_type,CONSIDER_EDGES=CONSIDER_EDGES) and \
            self.target_type.equal(eType.target_type,CONSIDER_EDGES=CONSIDER_EDGES) and \
            super(EdgeType, self).equal(eType,ignore_datatype):
            return True
        return False

    def addDataTypeFromElem(self, relation, IGNORE_NODES=False, ADD_TYPE_IF_NOT_EXIST=False):
        if IGNORE_NODES==False:
            s = relation.nodes[0]
            t = relation.nodes[1]
            self.src_type.addDataTypeFromElem(s)
            self.target_type.addDataTypeFromElem(t)
        super(EdgeType, self).addDataTypeFromElem(relation, ADD_TYPE_IF_NOT_EXIST)

    def mergeWithAnotherType(self, r):
        s = r.src_type
        t = r.target_type
        self.src_type.mergeWithAnotherType(s)
        self.target_type.mergeWithAnotherType(t)
        super(EdgeType, self).mergeWithAnotherType(r)

    def print(self):
        print('source')
        self.src_type.print()
        print('edge')
        super(EdgeType,self).print()
        print('target')
        self.target_type.print()

