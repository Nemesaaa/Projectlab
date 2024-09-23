import time

start_time = time.time()

import sumolib
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random
random.seed(300)

import torch
import torch_geometric
import torch_geometric.data as Data
import torch_geometric.utils as pyg_utils

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

from os import walk
import os
import xml.etree.cElementTree as et

f = []

#Function to read in the network we work with into a networkx object with the nodes and edges, no features yet
def read_sumo_net(filename):
    net = sumolib.net.readNet(filename)
    G = nx.Graph()
    
    # Add nodes
    for node in net.getNodes():
        G.add_node(node.getID(), pos=(node.getCoord()))
    # Add edges
    for edge in net.getEdges():
        for lane in edge.getLanes():
            edge_id = lane.getEdge().getID()
            starting_node_id = net.getEdge(edge_id).getFromNode().getID()
            ending_node_id = net.getEdge(edge_id).getToNode().getID()
            G.add_edge(starting_node_id, ending_node_id, edge_id = edge_id)
    return G

#Function to add the features to the network graph we created already

def add_edge_features_from_xml(G, xml_filename, interval_begin):
    # Parse the XML file
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    #Find the interval corresponding to the interval_begin time 
    interval = root.find(f'.//interval[@begin="{interval_begin}"]')
    #Extract all the features of the edges
    edges = interval.findall('.//edge')
    for edge in edges:
        edge_id = edge.get('id')
        edge_features = {}
        edge_features['left'] = edge.get('left')
        #We can add other features here
        #Iterate through the edges in the existing NetworkX graph
        for xml_edge_id, xml_edge_data in G.edges.items():
            if G.get_edge_data(xml_edge_id[0],xml_edge_id[1])['edge_id'] == edge_id:
                G.edges[xml_edge_id].update(edge_features)
    return G

def nx_to_pyg(graph):
    # Convert NetworkX graph to PyTorch Geometric Data object
    pyg_data = Data.Data()
    #We have to number the nodes, because that is how Data object works
    # Mapping between string node IDs and numerical indices
    node_id_to_index = {node_id: i for i, node_id in enumerate(graph.nodes)}

    # Set node features
    num_nodes = graph.number_of_nodes()
    node_features = np.zeros((num_nodes, 2))  # Assuming num_features is known, this is important to change, if we want to change something, altough I do not think that will be the case for us
    for i, (node, features) in enumerate(graph.nodes(data=True)):
        node_features[i] = [features['pos'][0], features['pos'][1]]  # Add node features accordingly, this case the coordinates
    pyg_data.x = torch.tensor(node_features, dtype=torch.float)

    # Set edge features and edge indices
    edge_index = []
    edge_features = []
    for u, v, features in graph.edges(data=True):
        # Map string node IDs to numerical indices
        u_index = node_id_to_index[u]
        v_index = node_id_to_index[v]
        edge_index.append([u_index, v_index])
        edge_features.append([float(features['left'])])  # Add edge features accordingly, if we add more features, we have to change this line

    pyg_data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    pyg_data.edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return pyg_data


#Function to plot the graph
def plot_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10)
    plt.show()
Graph = read_sumo_net('s_gyor.net.xml')

G1 = read_sumo_net('s_gyor.net.xml')
G2 = add_edge_features_from_xml(G1,'gyor_forg_15_min_1_1.xml',"0.00")
pyg_data = nx_to_pyg(G2)
#print(pyg_data)

f = []
mypath=os.getcwd()
for (dirpath, dirnames, filenames) in walk(mypath):

    f.extend(filenames)
    break
for i in f:
    splited = i.split(".")
    if splited[1] !="xml":
        f.remove(i)
f.remove("s_gyor.net.xml")
print(f)

f_12 = []
f_1 = []
f_11 = []
f_14 = []

for i in f:
    if "_12_" in i:
        f_12.append(i)

    if "_11_" in i:
        f_11.append(i)

    if "_1_" in i:
        f_1.append(i)  

    if "_14_" in i:
        f_14.append(i)   

print(f_14)
print(f_12)
print(f_11)
print(f_1)

print(str(len(f_14)) +"  "+ str(len(f_1)) +"  "+str(len(f_12)) +"  "+str(len(f_11)) +"  "+ str(len(f)) +"  "+ str(len(f_1)+len(f_11)+len(f_12)+len(f_14)))


timesteps=[]
kezd = 1800
for i in range(14):
    timesteps.append(str(kezd+i*900)+".00")

data_f_14 = []
data_f_12 = []
data_f_11 = []
data_f_1 = []

data_types = [data_f_14, data_f_12, data_f_11, data_f_1]
data_sources = [f_14, f_12, f_11, f_1]

data_test=[]
data_train = []
for k in range(4):
    for i in data_sources[k]:
        for j in timesteps:
            G2 = add_edge_features_from_xml(G1,i,j)
            #print(i)
            #print(j)
            data_types[k].append(nx_to_pyg(G2))
        print("--- %s seconds ---" % (time.time() - start_time) + str(i))
#torch.save(data,'data.pth')

for i in range(4):
    current_sample = random.sample(data_types[i], int(len(data_types[i])*0.7))

    for k in current_sample:
        data_train.append(k)
        data_types[i].remove(k)

    for j in data_types[i]:
        data_test.append(j)

print(str(len(data_train)) + "leght of train data")
print(str(len(data_test)) + "lenght of test data")

torch.save(data_test,'data_test.pth')
torch.save(data_train,'data_train.pth')
    
print("--- %s seconds ---" % (time.time() - start_time))