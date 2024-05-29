import os
import sys
import subprocess
import sumolib
import traci

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))


# parse the net
net = sumolib.net.readNet('s_gyor.net.xml')

sumolib.open('s_gyor.sumocfg')