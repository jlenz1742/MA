import numpy as np
import pickle
import igraph
import math


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


