import numpy as np
import pickle
import igraph
import math
import random


def random_choice_of_trees(number_of_penetrating_vein_trees, number_of_penetrating_artery_trees):

    # Number of available files

    number_arteries = 58
    number_veins = 103

    vein_files = []
    artery_files = []

    artery_trees_files = list(range(number_arteries))
    vein_trees_files = list(range(number_veins))

    for i in range(number_of_penetrating_vein_trees):

        vein_i = random.choice(vein_trees_files)
        vein_files.append(vein_i)
        vein_trees_files.remove(vein_i)

    for j in range(number_of_penetrating_artery_trees):

        artery_j = random.choice(artery_trees_files)
        artery_files.append(artery_j)
        artery_trees_files.remove(artery_j)

    needed_files = {'Files_arteries': artery_files, 'Files_veins': vein_files}

    return needed_files


def get_penetrating_tree_from_pkl_file(file_path, file_id):

    ''' Function reads pkl file and creates graph with the most important attributes. '''

    path_ = file_path
    file_name_edges = '\\' + str(file_id) + '_edgesDict.pkl'
    file_name_vertices = '\\' + str(file_id) + '_verticesDict.pkl'

    with open(path_ + file_name_edges, 'rb') as f:

        data_edge = pickle.load(f, encoding='latin1')

    with open(path_ + file_name_vertices, 'rb') as f:

        data_vertex = pickle.load(f, encoding='latin1')

    adjlist = np.array(data_edge['tuple'])
    g = igraph.Graph(adjlist.tolist())

    ''' IMPORTANT ATTRIBUTES '''

    g.es['edge_length'] = 0
    g.es['diameter'] = 0

    g.vs['x_coordinate'] = 0
    g.vs['y_coordinate'] = 0
    g.vs['z_coordinate'] = 0

    g.vs['attachmentVertex'] = data_vertex['attachmentVertex']
    g.vs['CapBedConnection'] = 0

    # STORE COORDINATES and CONNECTION TO CAPILLARY BED AS ATTRIBUTES

    for vertex in range(len(data_vertex['coords'])):

        g.vs[vertex]['x_coordinate'] = data_vertex['coords'][vertex][0]
        g.vs[vertex]['y_coordinate'] = data_vertex['coords'][vertex][1]
        g.vs[vertex]['z_coordinate'] = data_vertex['coords'][vertex][2]

        neighbors_current_vertex = g.neighbors(vertex)

        if len(neighbors_current_vertex) <= 2:
            g.vs[vertex]['CapBedConnection'] = 1

    # CALCULATE LENGTH OF EACH CONNECTION (SUMMATION OF SEGMENTS)

    for edge in range(g.ecount()):

        l_tot = 0
        sum_resistances = 0

        for i in range(len(data_edge['points'][edge]) - 1):
            start_node = data_edge['points'][edge][i]
            end_node = data_edge['points'][edge][i + 1]
            segment_length = math.sqrt(
                math.pow((end_node[0] - start_node[0]), 2) + math.pow((end_node[1] - start_node[1]), 2) + math.pow(
                    (end_node[2] - start_node[2]), 2))
            segment_resistance = segment_length / math.pow(data_edge['diameters'][edge][i], 4)

            l_tot += segment_length
            sum_resistances += segment_resistance

        g.es[edge]['edge_length'] = l_tot
        g.es[edge]['diameter'] = math.pow(l_tot / sum_resistances, 0.25)

    return g

def get_coordinates_limits_from_several_graphs(graphs):

    x_min = []
    x_max = []
    y_min = []
    y_max = []
    z_min = []
    z_max = []

    for graph in graphs:

        limites_temp = coordinates_limits(graph)

        x_min.append(limites_temp['x_min'])
        x_max.append(limites_temp['x_max'])
        y_min.append(limites_temp['y_min'])
        y_max.append(limites_temp['y_max'])
        z_min.append(limites_temp['z_min'])
        z_max.append(limites_temp['z_max'])

    coordinates_limits_several_graphs = {}

    coordinates_limits_several_graphs['x_min'] = min(x_min)
    coordinates_limits_several_graphs['x_max'] = max(x_max)
    coordinates_limits_several_graphs['y_min'] = min(y_min)
    coordinates_limits_several_graphs['y_max'] = max(y_max)
    coordinates_limits_several_graphs['z_min'] = min(z_min)
    coordinates_limits_several_graphs['z_max'] = max(z_max)

    return coordinates_limits_several_graphs

def coordinates_limits(_graph):

    ''' Function takes a igraph graph and returns max and min coordinates '''

    # Requirements: Graph need x_coordinate, y_coordinate, z_coordinate as attributes (Vertex attribute)

    coordinate_limits = {}

    coordinate_limits['x_min'] = min(_graph.vs['x_coordinate'])
    coordinate_limits['x_max'] = max(_graph.vs['x_coordinate'])
    coordinate_limits['y_min'] = min(_graph.vs['y_coordinate'])
    coordinate_limits['y_max'] = max(_graph.vs['y_coordinate'])
    coordinate_limits['z_min'] = min(_graph.vs['z_coordinate'])
    coordinate_limits['z_max'] = max(_graph.vs['z_coordinate'])

    return coordinate_limits


def get_number_of_combs(coord_limits, l_honeycomb):

    delta_x = coord_limits['x_max'] - coord_limits['x_min']
    delta_y = coord_limits['y_max'] - coord_limits['y_min']
    delta_z = coord_limits['z_max'] - coord_limits['z_min']

    x_0 = int((delta_x - 0.5 * l_honeycomb) / (1.5 * l_honeycomb)) + 1
    y_0 = int(delta_y / (2 * l_honeycomb * math.cos(30 * 2 * math.pi / 360))) + 1
    z_0 = int(delta_z / l_honeycomb) + 2

    if (x_0 % 2) == 0:

        x_0 += 1

    else:

        None

    number_of_combs = [x_0, y_0, z_0]

    return number_of_combs

