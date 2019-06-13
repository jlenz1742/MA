import pandas as pd
import numpy as np
import math


def create_csv_files_from_graph(graph, p_veins, r_sphere, coords_sphere):

    ''' Function transforms graph to three csv files to proceed with the simulation '''

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  node_data.csv                                                   #
    #                                                                                                                  #
    ####################################################################################################################

    node_data_df = pd.DataFrame()

    node_data_df['x'] = [x / math.pow(10, 6) for x in graph.vs['x_coordinate']]
    node_data_df['y'] = [y / math.pow(10, 6) for y in graph.vs['y_coordinate']]
    node_data_df['z'] = [z / math.pow(10, 6) for z in graph.vs['z_coordinate']]
    node_data_df.to_csv('Export/node_data.csv', index=False, sep=';')

    ####################################################################################################################
    #                                                                                                                  #
    #                                           node_boundary_data.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    node_boundary_data_df = pd.DataFrame()

    nodeid = []
    pressure = []

    for vertex in range(graph.vcount()):

        if graph.vs[vertex]['attachmentVertex'] == 1:

            nodeid.append(vertex)

            # Boundary Veins

            if graph.vs[vertex]['Type'] == 1:

                pressure.append(p_veins)

            # Boundary Arteries

            elif graph.vs[vertex]['Type'] == 2:

                pressure.append(5000)

        else:

            None

    node_boundary_data_df['nodeId'] = nodeid

    boundary_type = [1] * len(nodeid)
    node_boundary_data_df['boundaryType'] = boundary_type

    node_boundary_data_df['p'] = pressure

    flux = [np.nan] * len(nodeid)
    node_boundary_data_df['flux'] = flux

    node_boundary_data_df.to_csv('Export/node_boundary_data.csv', index=False, sep=';')

    ####################################################################################################################
    #                                                                                                                  #
    #                                                edge_data.csv                                                     #
    #                                                                                                                  #
    ####################################################################################################################

    n_1 = []
    n_2 = []

    for edge in range(graph.ecount()):

        n_1.append(graph.es[edge].source)
        n_2.append(graph.es[edge].target)

    edge_data_df = pd.DataFrame()
    edge_data_df['n1'] = n_1
    edge_data_df['n2'] = n_2
    edge_data_df['D'] = graph.es['diameter']
    edge_data_df['L'] = graph.es['edge_length']
    # edge_data_df['Type'] = graph.es['Type']
    edge_data_df.to_csv('Export/edge_data.csv', index=False, sep=';')

    ####################################################################################################################
    #                                                                                                                  #
    #                                               activated_eids.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    # activated_edge_ids = []
    #
    # for edge in range(graph.ecount()):
    #
    #     p1 = graph.es[edge].source
    #     x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
    #     y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
    #     z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)
    #
    #     p2 = graph.es[edge].target
    #     x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
    #     y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
    #     z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)
    #
    #     x_3 = coords_sphere['x'] / math.pow(10, 6)
    #     y_3 = coords_sphere['y'] / math.pow(10, 6)
    #     z_3 = coords_sphere['z'] / math.pow(10, 6)
    #
    #     radius = r_sphere / math.pow(10, 6)
    #
    #     a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
    #     b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
    #     c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + \
    #         math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)
    #
    #     value = math.pow(b, 2) - 4 * a * c
    #
    #     print(value)
    #     if value >= 0:
    #
    #         activated_edge_ids.append(edge)
    #
    #     else:
    #
    #         continue
    #
    # print(activated_edge_ids)

    ####################################################################################################################
    #                                                                                                                  #
    #                                                reacting_eids.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    return
