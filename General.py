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

length_honeycomb = 35
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

print(coord_lim)

# x_test = 635.2
# y_test = 1546.86
# z_test = 25.63


# list_distances = []
# list_edges = []
#
# for edge in range(graph_test.ecount()):
#
#     if graph_test.es[edge]['CanBeConnectedToCB'] == 1:
#
#         x_mp = graph_test.es[edge]['Coord_midpoint'][0]
#         y_mp = graph_test.es[edge]['Coord_midpoint'][1]
#         z_mp = graph_test.es[edge]['Coord_midpoint'][2]
#
#         distance = math.sqrt(math.pow(x_test - x_mp, 2) + math.pow(y_test - y_mp, 2) + math.pow(z_test - z_mp, 2))
#
#         list_distances.append(distance)
#         list_edges.append(edge)
#
#     else:
#
#         continue
#
# print(list_distances)
# print(list_edges)
#
# print('Min Distance: ', min(list_distances), ' Edge: ', list_edges[list_distances.index(min(list_distances))])
#
# print(graph_test.es[list_edges[list_distances.index(min(list_distances))]])
#
# graph_test.add_vertex(x_coordinate=x_test, y_coordinate=y_test, z_coordinate=z_test, PartOfCapBed=0,
#                       PartOfPenetratingTree=1)
#
# graph_test.add_edge(graph_test.es[list_edges[list_distances.index(min(list_distances))]].source, 858,
#                     connection_CB_Pene=1)
# graph_test.add_edge(graph_test.es[list_edges[list_distances.index(min(list_distances))]].target, 858,
#                     connection_CB_Pene=1)
#
# graph_test.delete_edges(list_edges[list_distances.index(min(list_distances))])


print('Number of Nodes Penetrating Tree: ', graph_.vcount())
print('Number of Nodes Capillary Bed: ', graph_test.vcount())
print('Neighborhood_old Penetrating Tree: ', graph_.neighborhood())
neighborhood_old = graph_.neighborhood()
neighborhood_new = []
for i in neighborhood_old:
    neighborhood_new.append([x + graph_test.vcount()-1 for x in i])

print('Neighborhood_new_Penetrating_Tree: ', neighborhood_new)

nodes_before_penetrating_tree = graph_test.vcount()


graph_test.add_vertices(graph_.vcount())

for vertex in range(graph_.vcount()):

    vertex_new = vertex + nodes_before_penetrating_tree

    graph_test.vs[vertex_new]['x_coordinate'] = graph_.vs[vertex]['x_coordinate']
    graph_test.vs[vertex_new]['y_coordinate'] = graph_.vs[vertex]['y_coordinate']
    graph_test.vs[vertex_new]['z_coordinate'] = graph_.vs[vertex]['z_coordinate']
    graph_test.vs[vertex_new]['attachmentVertex'] = graph_.vs[vertex]['attachmentVertex']
    graph_test.vs[vertex_new]['CapBedConnection'] = graph_.vs[vertex]['CapBedConnection']

    graph_test.vs[vertex_new]['PartOfPenetratingTree'] = 1

    neighbors = graph_.neighbors(vertex)
    neighbors_new = [x + nodes_before_penetrating_tree for x in neighbors]

    for neighbor in neighbors_new:

        if neighbor > vertex_new:

            graph_test.add_edge(vertex_new, neighbor, edge_length=0, diameter=0, PartOfPenetratingTree=1)

    if graph_.vs[vertex]['CapBedConnection'] == 1:

        list_distances = []
        list_edges = []

        for edge in range(graph_test.ecount()):

            if graph_test.es[edge]['CanBeConnectedToCB'] == 1:

                x_vertex = graph_test.vs[vertex_new]['x_coordinate']
                y_vertex = graph_test.vs[vertex_new]['y_coordinate']
                z_vertex = graph_test.vs[vertex_new]['z_coordinate']

                x_mp = graph_test.es[edge]['Coord_midpoint'][0]
                y_mp = graph_test.es[edge]['Coord_midpoint'][1]
                z_mp = graph_test.es[edge]['Coord_midpoint'][2]

                distance = math.sqrt(
                    math.pow(x_vertex - x_mp, 2) + math.pow(y_vertex - y_mp, 2) + math.pow(z_vertex - z_mp, 2))

                list_distances.append(distance)
                list_edges.append(edge)

            else:

                continue


        graph_test.add_edge(graph_test.es[list_edges[list_distances.index(min(list_distances))]].source, vertex_new,
                            connection_CB_Pene=1)
        graph_test.add_edge(graph_test.es[list_edges[list_distances.index(min(list_distances))]].target, vertex_new,
                            connection_CB_Pene=1)

        graph_test.delete_edges(list_edges[list_distances.index(min(list_distances))])

    else:

        continue


Plot.plot_graph(graph_test)

