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

    print([i for i in data_edge])
    print([i for i in data_vertex])

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


path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\Masterarbeit\database_penetrating_trees\arteryDB'
file_id = 0

graph_ = get_penetrating_tree_from_pkl_file(path, file_id)

