import cProfile
import pandas as pd
from Schema import *
from NewElementTypeSet import *
from ElementType import *
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import CountVectorizer
# import plotly.graph_objects as go
# from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from pickle5 import pickle
import json
import os
import re
import pydot
import pprofile as pp
# import line_profiler

class MyRel(dict):
    def __init__(self,*arg,**kw):
        self.nodes = [{},{}]

        super(MyRel, self).__init__(*arg, **kw)
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

prof = pp.Profile()
with prof:
    with open('cpg/stl/parsed_stl.json') as f:
        node_dict_list = json.load(f)
    node_map = {}
    for node_dict in node_dict_list:
        node_map[node_dict['id']] = node_dict
    edge_dict_list = []
    directory = 'cpg/stl/out/'
    edges = {}
    dot_str = ''
    for filename in tqdm(os.listdir(directory)[:4]):
        if filename.endswith(".dot") :
            try:
                with open(directory + filename) as f:
                    dot_str = f.read()
                    li = dot_str.split('\n')
                    li[0] = re.sub(r'[^a-zA-Z0-9{\s]','',li[0])
                    dot_str = '\n'.join(li)
                g = pydot.graph_from_dot_data(dot_str)[0]
                for e in g.get_edges():
                    s = int(re.sub('"','',e.get_source()))
                    t = int(re.sub('"','',e.get_destination()))
                    lbls = re.sub('"','',g.get_edges()[0].get_attributes()['label']).split(':')
                    if len(lbls)>2:
                        print(lbls)
                    rel = MyRel({lbls[0]:LBL()})
                    rel.nodes[0] = node_map[s]
                    rel.nodes[1] = node_map[t]
                    edge_dict_list.append(rel)
            except:
                print(dot_str)
                print(filename)
prof.print_stats()