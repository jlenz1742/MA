import pandas as pd
import numpy as np


def create_csv_files_from_graph(graph, p_veins):

    ''' Function transforms graph to three csv files to proceed with the simulation '''

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  node_data.csv                                                   #
    #                                                                                                                  #
    ####################################################################################################################

    node_data_df = pd.DataFrame()
    node_data_df['x'] = graph.vs['x_coordinate']
    node_data_df['y'] = graph.vs['y_coordinate']
    node_data_df['z'] = graph.vs['z_coordinate']
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

    return
