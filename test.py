'''
import xml.etree.ElementTree as ET

# Load the XML data from a file
xml_file = "gyor_forg_15_min.xml"

# Parse the XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Accessing interval attributes
intervals = root.findall('.//interval')
for interval in intervals:
    interval_id = interval.attrib['begin']
    print("Begin:", interval_id)
    
    # Accessing edge attributes
    edges = interval.findall('.//edge')
    for edge in edges:
        edge_id = edge.attrib['id']
        left_value = edge.attrib['left']
        print("Edge ID:", edge_id)
        print("Left Value:", left_value)

    break
'''
""" import torch

data = torch.load('data.pth')
print(data[0]) """

import numpy as np
a=2350.014
print(a)
b = float(a)
c = int(b)
print(type(c))
print(int(np.linspace(11,202,10)))
