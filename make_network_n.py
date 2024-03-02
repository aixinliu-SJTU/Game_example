# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt#导入画图工具包，命名为plt
import networkx as nx#导入networkx包，命名为nx



def create_10_net():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5,6,7,8,9])
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)])
    return G