import igraph as ig
import Tools
import numpy as np
import igraph_interface
import random
import Network_Generator
import Tracer_Tracking_Algorithm
import Plot
import math
import Import_Penetrating_Trees

''' Input '''

# Dimensions
x_dimension = 3
y_dimension = 3
z_dimension = 3

length_honeycomb = 1
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

''' Import Penetrating Tree '''

path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\Masterarbeit\database_penetrating_trees\arteryDB'
file = 0

graph_ = Import_Penetrating_Trees.get_penetrating_tree_from_pkl_file(path, file)



# Tüftle

coord_lim = Import_Penetrating_Trees.coordinates_limits(graph_)

delta_x = coord_lim['x_max'] - coord_lim['x_min']
delta_y = coord_lim['y_max'] - coord_lim['y_min']
delta_z = coord_lim['z_max'] - coord_lim['z_min']
print(delta_z)

x_0 = int((delta_x - 0.5 * length_honeycomb)/(1.5 * length_honeycomb)) + 1
y_0 = int(delta_y / (2 * length_honeycomb * math.cos(30 * 2 * math.pi / 360))) + 1
z_0 = int(delta_z / length_honeycomb) + 2


if (x_0 % 2) == 0:

    x_0 += 1

else:

    None

print(x_0, y_0, z_0)
print(coord_lim)
graph_test = ig.Graph()
graph_test = Network_Generator.create_3d_graph_test(graph_test, x_0, y_0, z_0, length_honeycomb, coord_lim)
print(ig.summary(graph_test))

# Plot.plot_graph(graph_test)

print(max(graph_test.vs['x_coordinate']))
print(max(graph_test.vs['y_coordinate']))
print(max(graph_test.vs['z_coordinate']))
print('fini')

