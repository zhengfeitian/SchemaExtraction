from Schema import *
from ElementType import *
import pandas as pd
import networkx as nx
# from pickle5 import pickle
# nodes

# return networkx DiGraph
def get_taxonomy_data():
    df = pd.read_csv('data/taxonomy.csv',delimiter=';')
    nodes = []
    edges = []
    name_to_node = {}
    ground_truth = {}
    id_to_node = {}
    for ind, row in df.iterrows():
        n = {'id':row['ID']}
        prev = prevv = row['Cat1']
        ground_truth[n['id']] = 'Cat1'
        for col in df.columns[2:]:
            if pd.isnull(row[col]):
                n['name'] = prev
                if prev!=prevv:
                    source = name_to_node[prevv]
                    edges.append([source,n])
                name_to_node[prev] = n
                break
            else:
                if col=='Cat7':
                    source = name_to_node[prev]
                    edges.append([source,n])
                    name_to_node[row[col]]=n
                    ground_truth[n['id']] = col
                    break
                prevv = prev
                prev = row[col]
                ground_truth[n['id']] = col

        nodes.append(n)
        id_to_node[n['id']] = n
    G = nx.DiGraph()
    G.add_nodes_from([n['id'] for n in nodes])
    G.add_edges_from([(e[0]['id'],e[1]['id']) for e in edges])
    # relable the nodes such that their ids are consecutive
    relable_map = {}
    cnt = 0
    for ind in sorted(list(G.nodes)):
        relable_map[ind] = cnt
        cnt += 1
    G = nx.relabel.relabel_nodes(G, mapping=relable_map)
    return G

def snb_and_cpg():
    place = {'name':set([str]), 'url':set([str]), 'Place':set([LBL])}
    city = {'name':set([str]), 'url':set([str]), 'City':set([LBL]), 'Place':set([LBL])}
    country = {'name':set([str]), 'url':set([str]), 'Country':set([LBL]), 'Place':set([LBL])}
    continent = {'name':set([str]), 'url':set([str]), 'Continent':set([LBL]), 'Place':set([LBL])}

    organization = {'name':set([str]), 'url':set([str]), 'Organisation':set([LBL])}
    university = {'name':set([str]), 'url':set([str]), 'University':set([LBL]), 'Organisation':set([LBL])}
    company = {'name':set([str]), 'url':set([str]), 'Company':set([LBL]), 'Organisation':set([LBL])}

    tagclass = {'name':set([str]), 'url':set([str]),'TagClass':set([LBL])}
    tag = {'name':set([str]), 'url':set([str]), 'Tag':set([LBL])}

    person={'creationDate':set([int]), 'firstName':set([str]), 'lastName':set([str]),
            'gender':set([str]),'birthday':set([int]), 'email':set([list]),
            'speaks':set([list]), 'browserUsed':set([str]), 'locationIP':set([str]),
            'Person':set([LBL])}
    forum = {'creationDate':set([int]),'title':set([str]), 'Forum':set([LBL])}

    message = {'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Message':set([LBL])}
    comment = {'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Comment':set([LBL]), 'Message':set([LBL])}
    post = {'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]),'language':set([str]),
               'imageFile':set([str]), 'Post':set([LBL]), 'Message':set([LBL])}
    node_type = [place,city,country,continent,organization,university,company,
                 tag,tagclass,person,forum,message,comment,post]

    containerOf = EdgeType(src_type=forum,target_type=post,
                           relation_type={'CONTAINER_OF':set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person,
                         relation_type={'HAS_CREATOR':set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                          relation_type={'HAS_INTEREST':set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                        relation_type={'HAS_MEMBER':set([LBL]), 'joinDate':set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person,
                           relation_type={'HAS_MODERATOR':set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                           relation_type={'HAS_TAG':set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                      relation_type={'HAS_TAG':set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                      relation_type={'HAS_TYPE':set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                          relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                       relation_type={'IS_PART_OF':set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                        relation_type={'IS_PART_OF':set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                        relation_type={'IS_SUBCLASS_OF':set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                        relation_type={'KNOWS':set([LBL]), 'creationDate':set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                        relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                        relation_type={'REPLY_OF':set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                        relation_type={'STUDY_AT':set([LBL]), 'classYear':set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                        relation_type={'WORK_AT':set([LBL]), 'workFrom':set([int])})
    gt_edge_types = [containerOf,hasCreator,hasInterest,hasMember,hasModerator,hasTag,
                    hasTag2, hasType, isLocatedIn,isLocatedIn2,isLocatedIn3,isLocatedIn4,
                    isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt]
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [NodeType(attr_type=i) for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    #with open('schema_snb-1_all_nodes_rels_ground_truth.pickle', 'wb') as f:
     #   pickle.dump(snb_gt_sc, f, -1)

     # nodes
    # note: meta has optional attr 'tag':string
    meta = {'Meta':LBL,'lastDatabaseEdit':str,'roiInfo':str, 'roiHierarchy': str,
            'primaryRois':list, 'neuroglancerInfo':str, 'statusDefinitions': str,
            'preHPThreshold': float, 'superLevelRois': list, 'latestMutationId': int,
            'logo': str, 'postHighAccuracyThreshold':float, 'meshHost':str,'dataset':str,
            'postHPThreshold':float, 'neuroglancerMeta': str, 'medulla7column_Meta':LBL,
            'totalPreCount':int, 'totalPostCount':int, 'uuid':str}
    segment_neuron = {'bodyId':int, 'instance':str, 'post':int, 'pre':int, 'roiInfo':str,
                     'size':int, 'type':str, 'cropped':bool, 'Neuron':LBL,
           'Segment':LBL, 'mushroombody_Neuron':LBL, 'mushroombody_Segment':LBL,
           'status':str,'statusLabel':str,'alpha1':bool,'alpha2':bool,'alpha3':bool}
    synapseSet = {'SynapseSet':LBL,'mushroombody_SynapseSet':LBL}
    synapse = {'type':str, 'location':int, 'Synapse':LBL,'mushroombody_Synapse':LBL,
                        'confidence':float,'alpha1':bool,'alpha2':bool,'alpha3':bool}
    node_type = [meta, segment_neuron, synapseSet,synapse]
    for i,nt in enumerate(node_type):
        for k in nt:
            nt[k] = set([nt[k]])
        node_type[i] = NodeType(attr_type=nt)
    meta = node_type[0]
    segment_neuron = node_type[1]
    synapseSet = node_type[2]
    synapse = node_type[3]
    #edges
    ConnectsTo = EdgeType(src_type=segment_neuron, target_type=segment_neuron,
                         relation_type={'ConnectsTo':set([LBL]), 'weight':set([int]),
                                       'weightHP': set([int]), 'roiInfo':set([str])})
    ConnectsTo2 = EdgeType(src_type=synapseSet, target_type=synapseSet,
                          relation_type={'ConnectsTo':set([LBL])})
    SynapsesTo = EdgeType(src_type=synapse, target_type=synapse,
                         relation_type={'SynapsesTo':set([LBL])})
    Contains = EdgeType(src_type=segment_neuron, target_type=synapseSet,
                       relation_type={'Contains':set([LBL])})
    Contains2 = EdgeType(src_type=synapseSet, target_type=synapse,
                        relation_type={'Contains':set([LBL])})
    gt_edge_types = [ConnectsTo, ConnectsTo2, SynapsesTo, Contains, Contains2]

    mb6_gt_sc = Schema()
    mb6_gt_sc.nodetypeset.nodetypes = node_type
    mb6_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    mb6_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    # with open('schema_mb6_ground_truth.pickle','wb') as f:
    #     pickle.dump(mb6_gt_sc,f, -1)

    # cpg
    # with open('cpg/stl/base.json') as f:
    #     d = json.load(f)
    gt_cpg_sc = Schema()
    from ElementType import LBL
    gt_nodetypes = []
    node_map = {} #node name to NodeType map
    not_in_node_key = []
    for nt in d['nodeTypes']:
        attrs = {}
        for k in nt['keys']:
            lower_k = underscoreToCap(k)
            if lower_k in node_key_set:
                attrs[lower_k] = node_key_set[lower_k]
            else:
                not_in_node_key.append(lower_k)
        attrs[nt['name']] = LBL()
        node_type = NodeType(node=attrs)
        node_map[nt['name']] = node_type
        gt_nodetypes.append(node_type)

        # constructing edges
    edge_types = []
    for nt in d['nodeTypes']:
        src = node_map[nt['name']]
        if not 'outEdges' in nt:
            continue
        for outEdge in nt['outEdges']:
            for tar in outEdge['inNodes']:
                et = EdgeType(src_type=src, target_type=node_map[tar['name']],relation_type={outEdge['edgeName']:set([LBL])})
                edge_types.append(et)
    # not_in_node_key
    gt_cpg_sc.nodetypeset.nodetypes = gt_nodetypes
    gt_cpg_sc.nodetypeset.numOfType = [1 for i in gt_nodetypes]
    gt_cpg_sc.edgetypeset.edgetypes = edge_types
    gt_cpg_sc.edgetypeset.numOfType = [1 for i in edge_types]
    # with open('gt_cpg_sc_stl.pkl','wb') as f:
    #     pickle.dump(gt_cpg_sc,f)

    place = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    city = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    country = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    continent = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}

    organization = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    university = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    company = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}

    tagclass = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}
    tag = {'name': set([str]), 'url': set([str]), 'Label': set([LBL])}

    person = {'creationDate': set([int]), 'firstName': set([str]), 'lastName': set([str]),
              'gender': set([str]), 'birthday': set([int]), 'email': set([list]),
              'speaks': set([list]), 'browserUsed': set([str]), 'locationIP': set([str]),
              'Label': set([LBL])}
    forum = {'creationDate': set([int]), 'title': set([str]), 'Label': set([LBL])}

    message = {'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
               'content': set([str]), 'length': set([int]), 'Label': set([LBL])}
    comment = {'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
               'content': set([str]), 'length': set([int]), 'Label': set([LBL])}
    post = {'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
            'content': set([str]), 'length': set([int]), 'language': set([str]),
            'imageFile': set([str]), 'Label': set([LBL])}
    node_type = [place, city, country, continent, organization, university, company,
                 tag, tagclass, person, forum, message, comment, post]

    containerOf = EdgeType(src_type=forum, target_type=post,
                           relation_type={'CONTAINER_OF': set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person,
                          relation_type={'HAS_CREATOR': set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                           relation_type={'HAS_INTEREST': set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                         relation_type={'HAS_MEMBER': set([LBL]), 'joinDate': set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person,
                            relation_type={'HAS_MODERATOR': set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                      relation_type={'HAS_TAG': set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                       relation_type={'HAS_TAG': set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                       relation_type={'HAS_TYPE': set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                           relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                        relation_type={'IS_PART_OF': set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                         relation_type={'IS_PART_OF': set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                            relation_type={'IS_SUBCLASS_OF': set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                     relation_type={'KNOWS': set([LBL]), 'creationDate': set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                     relation_type={'LIKES': set([LBL]), 'creationDate': set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                       relation_type={'REPLY_OF': set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                       relation_type={'STUDY_AT': set([LBL]), 'classYear': set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                      relation_type={'WORK_AT': set([LBL]), 'workFrom': set([int])})
    gt_edge_types = [containerOf, hasCreator, hasInterest, hasMember, hasModerator, hasTag,
                     hasTag2, hasType, isLocatedIn, isLocatedIn2, isLocatedIn3, isLocatedIn4,
                     isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt]
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [NodeType(attr_type=i) for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id'] = set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]


# with open('schema_snb-1_all_nodes_rels_ground_truth.pickle', 'wb') as f:
#   pickle.dump(snb_gt_sc, f, -1)


    #with incoming and outgoing edges
    place = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'Place': set([LBL])})
    city = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'City': set([LBL]), 'Place': set([LBL])})
    city.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    city.outgoing_edges.add(ContentType(elem={'IS_PART_OF': LBL()}))

    country = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'Country': set([LBL]), 'Place': set([LBL])})
    country.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    country.incoming_edges.add(ContentType(elem={'IS_PART_OF': LBL()}))
    country.outgoing_edges.add(ContentType(elem={'IS_PART_OF': LBL()}))

    continent = NodeType(
        attr_type={'name': set([str]), 'url': set([str]), 'Continent': set([LBL]), 'Place': set([LBL])})
    continent.incoming_edges.add(ContentType(elem={'IS_PART_OF': LBL()}))

    organization = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'Organisation': set([LBL])})

    university = NodeType(
        attr_type={'name': set([str]), 'url': set([str]), 'University': set([LBL]), 'Organisation': set([LBL])})
    university.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    university.incoming_edges.add(ContentType(elem={'STUDY_AT': LBL(), 'classYear': 342}))

    company = NodeType(
        attr_type={'name': set([str]), 'url': set([str]), 'Company': set([LBL]), 'Organisation': set([LBL])})
    company.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    company.incoming_edges.add(ContentType(elem={'WORK_AT': LBL(), 'workFrom': 32}))

    tagclass = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'TagClass': set([LBL])})
    tagclass.incoming_edges.add(ContentType(elem={'HAS_TYPE': LBL()}))
    tagclass.incoming_edges.add(ContentType(elem={'IS_SUBCLASS_OF': LBL()}))
    tagclass.outgoing_edges.add(ContentType(elem={'IS_SUBCLASS_OF': LBL()}))

    tag = NodeType(attr_type={'name': set([str]), 'url': set([str]), 'Tag': set([LBL])})
    tag.outgoing_edges.add(ContentType(elem={'HAS_TYPE': LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_TAG': LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_INTEREST': LBL()}))

    person = NodeType(attr_type={'creationDate': set([int]), 'firstName': set([str]), 'lastName': set([str]),
                                 'gender': set([str]), 'birthday': set([int]), 'email': set([list]),
                                 'speaks': set([list]), 'browserUsed': set([str]), 'locationIP': set([str]),
                                 'Person': set([LBL])})
    person.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    person.outgoing_edges.add(ContentType(elem={'HAS_INTEREST': LBL()}))
    person.outgoing_edges.add(ContentType(elem={'STUDY_AT': LBL(), 'classYear': 4324}))
    person.outgoing_edges.add(ContentType(elem={'WORK_AT': LBL(), 'workFrom': 32}))
    person.outgoing_edges.add(ContentType(elem={'LIKES': LBL(), 'creationDate': 'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_CREATOR': LBL()}))
    person.incoming_edges.add(ContentType(elem={'HAS_MEMBER': LBL(), 'joinDate': 'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_MODERATOR': LBL()}))
    person.incoming_edges.add(ContentType(elem={'KNOWS': LBL(), 'creationDate': 'fs'}))
    person.outgoing_edges.add(ContentType(elem={'KNOWS': LBL(), 'creationDate': 'fs'}))

    forum = NodeType(attr_type={'creationDate': set([int]), 'title': set([str]), 'Forum': set([LBL])})
    forum.outgoing_edges.add(ContentType(elem={'HAS_MEMBER': LBL(), 'joinDate': 'fs'}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_MODERATOR': LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_TAG': LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'CONTAINER_OF': LBL()}))

    message = NodeType(attr_type={'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
                                  'content': set([str]), 'length': set([int]), 'Message': set([LBL])})
    message.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_TAG': LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_CREATOR': LBL()}))
    message.incoming_edges.add(ContentType(elem={'REPLY_OF': LBL()}))
    message.incoming_edges.add(ContentType(elem={'LIKES': LBL(), 'creationDate': 'fs'}))

    comment = NodeType(attr_type={'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
                                  'content': set([str]), 'length': set([int]), 'Comment': set([LBL]),
                                  'Message': set([LBL])})
    comment.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_TAG': LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_CREATOR': LBL()}))
    comment.incoming_edges.add(ContentType(elem={'REPLY_OF': LBL()}))
    comment.incoming_edges.add(ContentType(elem={'LIKES': LBL(), 'creationDate': 'fs'}))
    comment.outgoing_edges.add(ContentType(elem={'REPLY_OF': LBL()}))

    post = NodeType(attr_type={'creationDate': set([int]), 'browserUsed': set([str]), 'locationIP': set([str]),
                               'content': set([str]), 'length': set([int]), 'language': set([str]),
                               'imageFile': set([str]), 'Post': set([LBL]), 'Message': set([LBL])})
    post.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN': LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_TAG': LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_CREATOR': LBL()}))
    post.incoming_edges.add(ContentType(elem={'REPLY_OF': LBL()}))
    post.incoming_edges.add(ContentType(elem={'LIKES': LBL(), 'creationDate': 'fs'}))
    post.incoming_edges.add(ContentType(elem={'CONTAINER_OF': LBL()}))

    node_type = [place, city, country, continent, organization, university, company,
                 tag, tagclass, person, forum, message, comment, post]

    containerOf = EdgeType(src_type=forum, target_type=post,
                           relation_type={'CONTAINER_OF': set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person,
                          relation_type={'HAS_CREATOR': set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                           relation_type={'HAS_INTEREST': set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                         relation_type={'HAS_MEMBER': set([LBL]), 'joinDate': set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person,
                            relation_type={'HAS_MODERATOR': set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                      relation_type={'HAS_TAG': set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                       relation_type={'HAS_TAG': set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                       relation_type={'HAS_TYPE': set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                           relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                            relation_type={'IS_LOCATED_IN': set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                        relation_type={'IS_PART_OF': set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                         relation_type={'IS_PART_OF': set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                            relation_type={'IS_SUBCLASS_OF': set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                     relation_type={'KNOWS': set([LBL]), 'creationDate': set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                     relation_type={'LIKES': set([LBL]), 'creationDate': set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                       relation_type={'REPLY_OF': set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                       relation_type={'STUDY_AT': set([LBL]), 'classYear': set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                      relation_type={'WORK_AT': set([LBL]), 'workFrom': set([int])})
    gt_edge_types = [containerOf, hasCreator, hasInterest, hasMember, hasModerator, hasTag,
                     hasTag2, hasType, isLocatedIn, isLocatedIn2, isLocatedIn3, isLocatedIn4,
                     isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt]
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [i for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id'] = set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    sc_gt_withNodeEdgeProfile = snb_gt_sc


def get_taxonomy():
    cat1 = NodeType()
    cat1.outgoing_edges.add(ContentType())

    cat2 = NodeType()
    cat2.incoming_edges.add(ContentType())
    cat2.outgoing_edges.add(ContentType())

    cat3 = NodeType()
    cat3.incoming_edges.add(ContentType())
    cat3.outgoing_edges.add(ContentType())

    cat4 = NodeType()
    cat4.incoming_edges.add(ContentType())
    cat4.outgoing_edges.add(ContentType())

    cat5 = NodeType()
    cat5.incoming_edges.add(ContentType())
    cat5.outgoing_edges.add(ContentType())

    cat6 = NodeType()
    cat6.incoming_edges.add(ContentType())
    cat6.outgoing_edges.add(ContentType())

    cat7 = NodeType()
    cat7.incoming_edges.add(ContentType())

    node_type = [cat1,cat2,cat3,cat4,cat5,cat6,cat7]
    subcat12 = EdgeType(src_type=cat1, target_type=cat2,
                      relation_type={})
    subcat23 = EdgeType(src_type=cat2, target_type=cat3,
                        relation_type={})
    subcat34 = EdgeType(src_type=cat3, target_type=cat4,
                        relation_type={})
    subcat45 = EdgeType(src_type=cat4, target_type=cat5,
                        relation_type={})
    subcat56 = EdgeType(src_type=cat5, target_type=cat6,
                        relation_type={})
    subcat67 = EdgeType(src_type=cat6, target_type=cat7,
                        relation_type={})
    edge_type = [subcat12,subcat23,subcat34,subcat45,subcat56,subcat67]


    gt_edge_types = []
    from Schema import Schema
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [i for i in node_type]
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in edge_type]
    return snb_gt_sc


#
def snb_gt_LT():
    # from Schema import Schema
    # changed ldbc snb for [LT] remove the supertype label, i.e. city, country, continetnet, just maintain place
    # remove
    place = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Place':set([LBL])})
    city = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Place':set([LBL])})
    city.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    city.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    country = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Place':set([LBL])})
    country.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    country.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))
    country.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    continent = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Place':set([LBL])})
    continent.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))


    organization = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Organisation':set([LBL])})

    university = NodeType(attr_type={'name':set([str]), 'url':set([str]),  'Organisation':set([LBL])})
    university.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    university.incoming_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':342}))


    company = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Organisation':set([LBL])})
    company.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    company.incoming_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))


    tagclass = NodeType(attr_type={'name':set([str]), 'url':set([str]),'TagClass':set([LBL])})
    tagclass.incoming_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tagclass.incoming_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))
    tagclass.outgoing_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))

    tag = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Tag':set([LBL])})
    tag.outgoing_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))


    person=NodeType(attr_type={'creationDate':set([int]), 'firstName':set([str]), 'lastName':set([str]),
            'gender':set([str]),'birthday':set([int]), 'email':set([list]),
            'speaks':set([list]), 'browserUsed':set([str]), 'locationIP':set([str]),
            'Person':set([LBL])})
    person.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':4324}))
    person.outgoing_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))
    person.outgoing_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))
    person.outgoing_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))


    forum = NodeType(attr_type={'creationDate':set([int]),'title':set([str]), 'Forum':set([LBL])})
    forum.outgoing_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    message = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Message':set([LBL])})
    message.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    message.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    message.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))

    comment = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Comment':set([LBL]), 'Message':set([LBL])})
    comment.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    comment.outgoing_edges.add(ContentType(elem={'REPLY_OF':LBL()}))

    post = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]),'language':set([str]),
               'imageFile':set([str]), 'Post':set([LBL]), 'Message':set([LBL])})
    post.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    post.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    post.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    post.incoming_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    node_type = [place,city,country,continent,organization,university,company,
                 tag,tagclass,person,forum,message,comment,post]
    node_type_forGen = [city,country,continent,university,company,
                 tag,tagclass,person,forum,message,comment,post]

    containerOf = EdgeType(src_type=forum,target_type=post,
                           relation_type={'CONTAINER_OF':set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person,
                          relation_type={'HAS_CREATOR':set([LBL])})
    hasCreator_post = EdgeType(src_type=post, target_type=person,
                               relation_type={'HAS_CREATOR':set([LBL])})
    hasCreator_comment = EdgeType(src_type=comment, target_type=person,
                                  relation_type={'HAS_CREATOR':set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                           relation_type={'HAS_INTEREST':set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                         relation_type={'HAS_MEMBER':set([LBL]), 'joinDate':set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person,
                            relation_type={'HAS_MODERATOR':set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                      relation_type={'HAS_TAG':set([LBL])})
    hasTag_post = EdgeType(src_type=post, target_type=tag,
                           relation_type={'HAS_TAG':set([LBL])})
    hasTag_comment = EdgeType(src_type=comment, target_type=tag,
                              relation_type={'HAS_TAG':set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                       relation_type={'HAS_TAG':set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                       relation_type={'HAS_TYPE':set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                            relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2_post = EdgeType(src_type=post, target_type=country,
                                 relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2_comm = EdgeType(src_type=comment, target_type=country,
                                 relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                            relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                            relation_type={'IS_LOCATED_IN':set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                        relation_type={'IS_PART_OF':set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                         relation_type={'IS_PART_OF':set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                            relation_type={'IS_SUBCLASS_OF':set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                     relation_type={'KNOWS':set([LBL]), 'creationDate':set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                     relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    likes_post = EdgeType(src_type=person, target_type=post,
                          relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    likes_comment = EdgeType(src_type=person, target_type=comment,
                             relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                       relation_type={'REPLY_OF':set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                       relation_type={'STUDY_AT':set([LBL]), 'classYear':set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                      relation_type={'WORK_AT':set([LBL]), 'workFrom':set([int])})
    gt_edge_types = [containerOf,hasCreator,hasInterest,hasMember,hasModerator,hasTag,
                     hasTag2, hasType, isLocatedIn,isLocatedIn2,isLocatedIn3,isLocatedIn4,
                     isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt,
                     isLocatedIn2_post, isLocatedIn2_comm, likes_post, likes_comment,
                     hasCreator_post, hasCreator_comment, hasTag_post, hasTag_comment]
    from Schema import Schema
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [i for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    #with open('snb_gt_removeSuperTypeLabel_LT.pkl','wb') as f:
    #    pickle.dump(snb_gt_sc,f)
    sc = Schema()
    sc.nodetypeset.nodetypes = [i for i in node_type_forGen]
    for nt in sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    sc.nodetypeset.numOfType = [1 for i in node_type_forGen]
    sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    return snb_gt_sc, sc

def get_gt_northwind():
    customer_demo_graphics = NodeType(attr_type={'CustomerTypeID':set([int]), 'CustomerDesc':set([str])})
    customer_demo_graphics.outgoing_edges.add(ContentType(elem={'group':LBL()}))

    customer_demo = NodeType()
    customer_demo.incoming_edges.add(ContentType(elem={'group':LBL()}))
    customer_demo.incoming_edges.add(ContentType(elem={'has':LBL()}))

    customer = NodeType(attr_type={'CustomerID':set([int]), 'CompanyName':set([str]),
                                  'ContactName':set([str]), 'ContactTitle':set([str]),
                                  'Address':set([str]),'City':set([str]),'Region':set([str]),
                                  'PostalCode':set([str]), 'Country':set([str]), 'Phone':set([str]),
                                  'Fax':set([str])})
    customer.outgoing_edges.add(ContentType(elem={'has':LBL()}))
    customer.outgoing_edges.add(ContentType(elem={'place':LBL()}))

    supplier = NodeType(attr_type={'SupplierID':set([int]), 'CompanyName':set([str]),
                                  'ContactName':set([str]), 'ContactTitle':set([str]),
                                  'Address':set([str]),'City':set([str]),'Region':set([str]),
                                  'PostalCode':set([str]), 'Country':set([str]), 'Phone':set([str]),
                                  'Fax':set([str]), 'Homepage':set([str])})
    supplier.outgoing_edges.add(ContentType(elem={'supply':LBL()}))

    product = NodeType(attr_type={'ProductID':set([int]), 'ProductName':set([str]),
                                 'QuantityPerUnit':set([int]), 'UnitPrice':set([float]),
                                 'UnitsInStock':set([int]), 'UnitsOnOrder':set([int]),
                                 'ReorderLevel':set([int]), 'Discontinued':set([bool])})
    product.incoming_edges.add(ContentType(elem={'supply':LBL()}))
    product.incoming_edges.add(ContentType(elem={'describe':LBL()}))
    product.outgoing_edges.add(ContentType(elem={'order':LBL()}))

    category = NodeType(attr_type={'CategoryID':set([int]), 'CategoryName':set([str]),
                                  'Description':set([str]), 'Picture':set([str])})
    category.outgoing_edges.add(ContentType(elem={'describe':LBL()}))


    order_detail = NodeType(attr_type={'UnitPrice':set([float]), 'Quantity':set([int]),
                                      'Discount':set([float])})
    order_detail.incoming_edges.add(ContentType(elem={'order':LBL()}))
    order_detail.incoming_edges.add(ContentType(elem={'consistOf':LBL()}))

    order = NodeType(attr_type={'OrderID':set([int]), 'OrderDate':set([str]),
                               'RequiredDate':set([str]), 'ShippedDate':set([str]),
                               'ShipVia':set([str]), 'Freight':set([str]), 'ShipName':set([str]),
                               'ShipCity':set([str]), 'ShipRegion':set([str]), 'ShipRegion':set([str]),
                               'ShipPostalCode':set([str]), 'ShipCountry':set([str])})
    order.outgoing_edges.add(ContentType(elem={'consistOf':LBL()}))
    order.incoming_edges.add(ContentType(elem={'ship':LBL()}))
    order.incoming_edges.add(ContentType(elem={'handle':LBL()}))
    order.incoming_edges.add(ContentType(elem={'place':LBL()}))

    shipper = NodeType(attr_type={'ShipperID':set([int]), 'CompanyName':set([str]), 'Phone':set([str])})
    shipper.outgoing_edges.add(ContentType(elem={'ship':LBL()}))

    employee = NodeType(attr_type={'EmployeeID':set([int]), 'LastName':set([str]),
                                  'FirstName':set([str]), 'Title':set([str]),
                                  'TitleOfCourtesy':set([str]), 'BirthDate':set([str]),
                                  'HireDate':set([str]), 'Address':set([str]), 
                                  'City':set([str]), 'Region':set([str]), 'PostalCode':set([str]), 
                                  'Country':set([str]), 'HomePhone':set([str]), 'Extension':set([str]),
                                  'Photo':set([str]), 'Notes':set([str]), 'ReportsTo':set([str]),
                                  'PhotoPath':set([str])})
    employee.outgoing_edges.add(ContentType(elem={'handle':LBL()}))
    employee.outgoing_edges.add(ContentType(elem={'mgr':LBL()}))
    employee.incoming_edges.add(ContentType(elem={'mgr':LBL()}))
    employee.outgoing_edges.add(ContentType(elem={'salesArea':LBL()}))

    employee_territory = NodeType(attr_type={'RegionID':set([int]), 'RegionDescription':set([str])})
    employee_territory.incoming_edges.add(ContentType(elem={'salesArea':LBL()}))
    employee_territory.incoming_edges.add(ContentType(elem={'group':LBL()}))

    territory = NodeType(attr_type={'TerritoryID':set([int]), 'TerritoryDescription':set([str])})
    territory.incoming_edges.add(ContentType(elem={'group':LBL()}))
    territory.outgoing_edges.add(ContentType(elem={'group':LBL()}))

    region = NodeType(attr_type={'RegionID':set([int]), 'RegionDescription':set([str])})
    region.outgoing_edges.add(ContentType(elem={'group':LBL()}))

    node_type = [customer_demo_graphics, customer_demo, customer, order,
                supplier, category, product, order_detail, shipper, employee,
                employee_territory, territory, region]

    group0 = EdgeType(src_type=customer_demo_graphics,target_type=customer_demo,
                           relation_type={'group':set([LBL])})
    group1 = EdgeType(src_type=region,target_type=territory,
                           relation_type={'group':set([LBL])})
    group2 = EdgeType(src_type=territory,target_type=employee_territory,
                           relation_type={'group':set([LBL])})
    has = EdgeType(src_type=customer,target_type=customer_demo,
                           relation_type={'has':set([LBL])})
    place = EdgeType(src_type=customer,target_type=order,
                           relation_type={'place':set([LBL])})
    ship = EdgeType(src_type=shipper,target_type=order,
                           relation_type={'ship':set([LBL])})
    handle = EdgeType(src_type=employee,target_type=order,
                           relation_type={'handle':set([LBL])})
    mgr = EdgeType(src_type=employee,target_type=employee,
                           relation_type={'mgr':set([LBL])})
    salesArea = EdgeType(src_type=employee,target_type=employee_territory,
                           relation_type={'salesArea':set([LBL])})
    consistOf = EdgeType(src_type=order,target_type=order_detail,
                           relation_type={'consistOf':set([LBL])})
    orderEdge = EdgeType(src_type=product,target_type=order_detail,
                           relation_type={'order':set([LBL])})
    describe = EdgeType(src_type=category,target_type=product,
                           relation_type={'describe':set([LBL])})
    supply = EdgeType(src_type=supplier,target_type=product,
                           relation_type={'supply':set([LBL])})

    gt_edge_types = [group0, group1, group2, has, place, ship, handle, mgr, salesArea, consistOf, orderEdge,
                    describe, supply]

    from Schema import Schema
    nw_gt_sc = Schema()
    nw_gt_sc.nodetypeset.nodetypes = [i for i in node_type]

    nw_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    nw_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    nwsc_gt_withNodeEdgeProfile = nw_gt_sc
    return nw_gt_sc

def get_snb_with_incEdgesAndOutEdges():
    place = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Place':set([LBL])})
    city = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'City':set([LBL]), 'Place':set([LBL])})
    city.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    city.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    country = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Country':set([LBL]), 'Place':set([LBL])})
    country.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    country.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))
    country.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    continent = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Continent':set([LBL]), 'Place':set([LBL])})
    continent.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))


    organization = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Organisation':set([LBL])})

    university = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'University':set([LBL]), 'Organisation':set([LBL])})
    university.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    university.incoming_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':342}))


    company = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Company':set([LBL]), 'Organisation':set([LBL])})
    company.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    company.incoming_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))


    tagclass = NodeType(attr_type={'name':set([str]), 'url':set([str]),'TagClass':set([LBL])})
    tagclass.incoming_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tagclass.incoming_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))
    tagclass.outgoing_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))

    tag = NodeType(attr_type={'name':set([str]), 'url':set([str]), 'Tag':set([LBL])})
    tag.outgoing_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))


    person=NodeType(attr_type={'creationDate':set([int]), 'firstName':set([str]), 'lastName':set([str]),
            'gender':set([str]),'birthday':set([int]), 'email':set([list]),
            'speaks':set([list]), 'browserUsed':set([str]), 'locationIP':set([str]),
            'Person':set([LBL])})
    person.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':4324}))
    person.outgoing_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))
    person.outgoing_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))
    person.outgoing_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))


    forum = NodeType(attr_type={'creationDate':set([int]),'title':set([str]), 'Forum':set([LBL])})
    forum.outgoing_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    message = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Message':set([LBL])})
    message.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    message.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    message.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))

    comment = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]), 'Comment':set([LBL]), 'Message':set([LBL])})
    comment.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    comment.outgoing_edges.add(ContentType(elem={'REPLY_OF':LBL()}))

    post = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]),'language':set([str]),
               'imageFile':set([str]), 'Post':set([LBL]), 'Message':set([LBL])})
    post.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    post.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    post.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    post.incoming_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    node_type = [place,city,country,continent,organization,university,company,
                 tag,tagclass,person,forum,message,comment,post]

    containerOf = EdgeType(src_type=forum,target_type=post,
                           relation_type={'CONTAINER_OF':set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person,
                         relation_type={'HAS_CREATOR':set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                          relation_type={'HAS_INTEREST':set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                        relation_type={'HAS_MEMBER':set([LBL]), 'joinDate':set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person,
                           relation_type={'HAS_MODERATOR':set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                           relation_type={'HAS_TAG':set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                      relation_type={'HAS_TAG':set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                      relation_type={'HAS_TYPE':set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                          relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                       relation_type={'IS_PART_OF':set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                        relation_type={'IS_PART_OF':set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                        relation_type={'IS_SUBCLASS_OF':set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                        relation_type={'KNOWS':set([LBL]), 'creationDate':set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                        relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                        relation_type={'REPLY_OF':set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                        relation_type={'STUDY_AT':set([LBL]), 'classYear':set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                        relation_type={'WORK_AT':set([LBL]), 'workFrom':set([int])})
    gt_edge_types = [containerOf,hasCreator,hasInterest,hasMember,hasModerator,hasTag,
                    hasTag2, hasType, isLocatedIn,isLocatedIn2,isLocatedIn3,isLocatedIn4,
                    isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt]
    from Schema import Schema
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [i for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    sc_gt_withNodeEdgeProfile = snb_gt_sc
    sc_for_gen = Schema() # remove the type place, organization; for gneerate graph
    sc_for_gen.nodetypeset.nodetypes = [i for i in node_type[1:]]
    sc_for_gen.nodetypeset.nodetypes.pop(3)
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    sc_for_gen.nodetypeset.numOfType = [1 for i in node_type[2:]]
    sc_for_gen.edgetypeset.edgetypes = [i for i in gt_edge_types]
    return snb_gt_sc,sc_for_gen

def get_nolabel_SNB_gt():

    # changed ldbc snb for [PT] remove the label
    # remove 
    place = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    city = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    city.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    city.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    country = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    country.incoming_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    country.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))
    country.outgoing_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))

    continent = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    continent.incoming_edges.add(ContentType(elem={'IS_PART_OF':LBL()}))


    organization = NodeType(attr_type={'name':set([str]), 'url':set([str])})

    university = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    university.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    university.incoming_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':342}))


    company = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    company.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    company.incoming_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))


    tagclass = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    tagclass.incoming_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tagclass.incoming_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))
    tagclass.outgoing_edges.add(ContentType(elem={'IS_SUBCLASS_OF':LBL()}))

    tag = NodeType(attr_type={'name':set([str]), 'url':set([str])})
    tag.outgoing_edges.add(ContentType(elem={'HAS_TYPE':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    tag.incoming_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))


    person=NodeType(attr_type={'creationDate':set([int]), 'firstName':set([str]), 'lastName':set([str]),
            'gender':set([str]),'birthday':set([int]), 'email':set([list]),
            'speaks':set([list]), 'browserUsed':set([str]), 'locationIP':set([str])})
    person.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'HAS_INTEREST':LBL()}))
    person.outgoing_edges.add(ContentType(elem={'STUDY_AT':LBL(),'classYear':4324}))
    person.outgoing_edges.add(ContentType(elem={'WORK_AT':LBL(),'workFrom':32}))
    person.outgoing_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    person.incoming_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    person.incoming_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))
    person.outgoing_edges.add(ContentType(elem={'KNOWS':LBL(),'creationDate':'fs'}))


    forum = NodeType(attr_type={'creationDate':set([int]),'title':set([str])})
    forum.outgoing_edges.add(ContentType(elem={'HAS_MEMBER':LBL(),'joinDate':'fs'}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_MODERATOR':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    forum.outgoing_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    message = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int])})
    message.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    message.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    message.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    message.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))

    comment = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int])})
    comment.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    comment.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    comment.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    comment.outgoing_edges.add(ContentType(elem={'REPLY_OF':LBL()}))

    post = NodeType(attr_type={'creationDate':set([int]), 'browserUsed':set([str]),'locationIP':set([str]),
               'content':set([str]), 'length':set([int]),'language':set([str]),
               'imageFile':set([str])})
    post.outgoing_edges.add(ContentType(elem={'IS_LOCATED_IN':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_TAG':LBL()}))
    post.outgoing_edges.add(ContentType(elem={'HAS_CREATOR':LBL()}))
    post.incoming_edges.add(ContentType(elem={'REPLY_OF':LBL()}))
    post.incoming_edges.add(ContentType(elem={'LIKES':LBL(),'creationDate':'fs'}))
    post.incoming_edges.add(ContentType(elem={'CONTAINER_OF':LBL()}))

    node_type = [place,city,country,continent,organization,university,company,
                 tag,tagclass,person,forum,message,comment,post]
    gen_node_type = [city,country,continent,university,company,
                 tag,tagclass,person,forum,message,comment,post]

    containerOf = EdgeType(src_type=forum,target_type=post,
                           relation_type={'CONTAINER_OF':set([LBL])})
    hasCreator = EdgeType(src_type=message, target_type=person, 
                         relation_type={'HAS_CREATOR':set([LBL])})
    hasCreator_post = EdgeType(src_type=post, target_type=person,
                          relation_type={'HAS_CREATOR':set([LBL])})
    hasCreator_comment = EdgeType(src_type=comment, target_type=person,
                               relation_type={'HAS_CREATOR':set([LBL])})
    hasInterest = EdgeType(src_type=person, target_type=tag,
                          relation_type={'HAS_INTEREST':set([LBL])})
    hasMember = EdgeType(src_type=forum, target_type=person,
                        relation_type={'HAS_MEMBER':set([LBL]), 'joinDate':set([str])})
    hasModerator = EdgeType(src_type=forum, target_type=person, 
                           relation_type={'HAS_MODERATOR':set([LBL])})
    hasTag = EdgeType(src_type=message, target_type=tag,
                           relation_type={'HAS_TAG':set([LBL])})
    hasTag_post = EdgeType(src_type=post, target_type=tag,
                      relation_type={'HAS_TAG':set([LBL])})
    hasTag_comment = EdgeType(src_type=comment, target_type=tag,
                           relation_type={'HAS_TAG':set([LBL])})
    hasTag2 = EdgeType(src_type=forum, target_type=tag,
                      relation_type={'HAS_TAG':set([LBL])})
    hasType = EdgeType(src_type=tag, target_type=tagclass,
                      relation_type={'HAS_TYPE':set([LBL])})
    isLocatedIn = EdgeType(src_type=company, target_type=country,
                          relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2 = EdgeType(src_type=message, target_type=country,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2_post = EdgeType(src_type=post, target_type=country,
                            relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn2_comm = EdgeType(src_type=comment, target_type=country,
                                 relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn3 = EdgeType(src_type=person, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isLocatedIn4 = EdgeType(src_type=university, target_type=city,
                           relation_type={'IS_LOCATED_IN':set([LBL])})
    isPartOf = EdgeType(src_type=city, target_type=country,
                       relation_type={'IS_PART_OF':set([LBL])})
    isPartOf2 = EdgeType(src_type=country, target_type=continent,
                        relation_type={'IS_PART_OF':set([LBL])})
    isSubclassOf = EdgeType(src_type=tagclass, target_type=tagclass,
                        relation_type={'IS_SUBCLASS_OF':set([LBL])})
    knows = EdgeType(src_type=person, target_type=person,
                        relation_type={'KNOWS':set([LBL]), 'creationDate':set([str])})
    likes = EdgeType(src_type=person, target_type=message,
                        relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    likes_post = EdgeType(src_type=person, target_type=post,
                     relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    likes_comment = EdgeType(src_type=person, target_type=comment,
                          relation_type={'LIKES':set([LBL]), 'creationDate':set([str])})
    replyOf = EdgeType(src_type=comment, target_type=message,
                        relation_type={'REPLY_OF':set([LBL])})
    studyAt = EdgeType(src_type=person, target_type=university,
                        relation_type={'STUDY_AT':set([LBL]), 'classYear':set([int])})
    workAt = EdgeType(src_type=person, target_type=company,
                        relation_type={'WORK_AT':set([LBL]), 'workFrom':set([int])})
    gt_edge_types = [containerOf,hasCreator,hasInterest,hasMember,hasModerator,hasTag,
                    hasTag2, hasType, isLocatedIn,isLocatedIn2,isLocatedIn3,isLocatedIn4,
                    isPartOf, isPartOf2, isSubclassOf, knows, likes, replyOf, studyAt, workAt,
                     isLocatedIn2_post, isLocatedIn2_comm, likes_post, likes_comment,
                     hasCreator_post, hasCreator_comment, hasTag_post, hasTag_comment]
    from Schema import Schema
    snb_gt_sc = Schema()
    snb_gt_sc.nodetypeset.nodetypes = [i for i in node_type]
    for nt in snb_gt_sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    snb_gt_sc.nodetypeset.numOfType = [1 for i in node_type]
    snb_gt_sc.edgetypeset.edgetypes = [i for i in gt_edge_types]

    sc = Schema() # sc do not contain abstract type
    sc.nodetypeset.nodetypes = [i for i in gen_node_type]
    for nt in sc.nodetypeset.nodetypes:
        nt.attributes['id']=set([int])
    sc.nodetypeset.numOfType = [1 for i in gen_node_type]
    sc.edgetypeset.edgetypes = [i for i in gt_edge_types]
    return snb_gt_sc,sc