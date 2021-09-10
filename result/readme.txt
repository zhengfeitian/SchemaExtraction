snb_gt_LL.pkl         # the original LDBC snb schema
(a graph generated based on the above schema)
The generated graph: 10087 nodes, 76,458 edges
snb_LL_label_based.pkl    # the extracted schema based on label based method, no weighting
snb_LL_prop_based.pkl # the extracted shcema based on the property based method, no weighting
snb_LL_topo_based_k2.pkl # the extracted schema based on the topo-based method (k=2)


nw_gt_PP.pkl
nodes: 9512, edges: 58244
nw_PP_label_based.pkl
nw_PP_prop_based.pkl
nw_PP_topo_based_k2.pkl
nw_pp_topo_based_k1.pkl # node types also contain incoming_edges and outgoing_edges

snb_gt_nolabel.pkl    # the SNB with all labels removed
nodes: 10087, edges: 289,448
snb_NL_org_extracted.pkl # the schema extracted from node list and edge list with sc_gen.extractSchema(node_dict_list=nl,edge_dict_list=el,IGNORE_LABEL=False,mode='TOPO')
snb_NL_prop_based.pkl
snb_NL_topo_based_k2.pkl
snb_NL_topo_based_k1.pkl # within incoming edges and outgoing edges in the node types



snb_gt_noSuperTypeLabel.pkl
nodes: 10087, edges: 119949
snb_nSTL_label_based.pkl # already runned the mergebylabelset method
snb_nSTL_topo_based_k1.pkl
snb_nSTL_topo_label_k1.pkl #