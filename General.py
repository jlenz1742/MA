import igraph as ig
import Tools
import numpy as np
import igraph_interface
import random
import Network_Generator
import Tracer_Tracking_Algorithm
import Plot

''' Input '''

# Dimensions

x_dimension = 3
y_dimension = 3
z_dimension = 3

length_honeycomb = 3
initial_diameter = 1.0

# Specify inlet and outlet pores

inlet_pores = [0, 1, 2, 3]
outlet_pores = [152, 153, 154, 155]

start_node_tracer = 0

# BC

total_inflow = 100
pressure_outlet_pores = 1

''' Build a new graph '''

graph_main = ig.Graph()
graph_main = Network_Generator.create_3d_graph(graph_main, x_dimension, y_dimension, z_dimension, length_honeycomb)


''' Plot '''

# Plot.plot_graph(graph_main)

''' Tracer Path '''

path = Tracer_Tracking_Algorithm.determine_path_of_a_tracer(graph_main, initial_diameter, outlet_pores, pressure_outlet_pores, inlet_pores, total_inflow, start_node_tracer)
# Plot.plot_path(graph_main, path)
print(path)

