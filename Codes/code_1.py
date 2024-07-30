import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
import xml.etree.cElementTree as et


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0

import tensorflow as tf
import math





# Load the XML data from a file
xml_file = "gyor_forg_15_min.xml"

# Parse the XML
tree = et.parse(xml_file)
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

