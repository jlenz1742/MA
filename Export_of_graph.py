import pandas as pd
import math
import os
import csv
import time
import json


def create_csv_files_from_graph(graph, p_veins, key_word_all, summary_information, key_number_modi):

    ''' Function transforms graph to three csv files to proceed with the simulation '''

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  Preparation                                                     #
    #                                                                                                                  #
    ####################################################################################################################

    time_str = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('Export/' + time_str)

    ####################################################################################################################
    #                                                                                                                  #
    #                                                    Read Me                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    with open('Export/' + time_str + '/read_me.txt', 'a+') as file:
        file.write(json.dumps(summary_information))

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  node_data.csv                                                   #
    #                                                                                                                  #
    ####################################################################################################################

    node_data_df = pd.DataFrame()

    node_data_df['x'] = [x / math.pow(10, 6) for x in graph.vs['x_coordinate']]
    node_data_df['y'] = [y / math.pow(10, 6) for y in graph.vs['y_coordinate']]
    node_data_df['z'] = [z / math.pow(10, 6) for z in graph.vs['z_coordinate']]
    node_data_df.to_csv('Export/' + time_str + '/node_data.csv', index=False)

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

    flux = ['nan'] * len(nodeid)
    node_boundary_data_df['flux'] = flux

    node_boundary_data_df.to_csv('Export/' + time_str + '/node_boundary_data.csv', index=False)

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
    edge_data_df.to_csv('Export/' + time_str + '/edge_data.csv', index=False)

    ####################################################################################################################
    #                                                                                                                  #
    #                                               activated_eids.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    activated_eids = []

    for edge in range(graph.ecount()):

        if graph.es[edge]['Activated'] == 1:

            activated_eids.append(edge)

    data = {'Activated': activated_eids}
    activated_eids_df = pd.DataFrame(data)
    activated_eids_df_new = activated_eids_df.T

    activated_eids_df_new.to_csv('Export/' + time_str + '/activated_eids_temp.csv', index=False)

    with open('Export/' + time_str + '/activated_eids_temp.csv', 'r') as f:
        with open('Export/' + time_str + '/activated_eids.csv', 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)

    os.remove('Export/' + time_str + '/activated_eids_temp.csv')

    ####################################################################################################################
    #                                                                                                                  #
    #                                                reacting_eids.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    if key_word_all == 0:

        reacting_eids = []

        for edge in range(graph.ecount()):

            if graph.es[edge]['Modifiable'] == 1:

                if key_number_modi == 0:

                    if graph.es[edge]['Type'] == 0 or graph.es[edge]['Type'] == 3:

                        reacting_eids.append(edge)

                elif key_number_modi == 1:

                    if graph.es[edge]['Type'] == 1:

                        reacting_eids.append(edge)

                elif key_number_modi == 2:

                    if graph.es[edge]['Type'] == 1 or graph.es[edge]['Type'] == 0 or graph.es[edge]['Type'] == 3:

                        reacting_eids.append(edge)

                elif key_number_modi == 3:

                    reacting_eids.append(edge)

        data_mod = {'Modifiable': reacting_eids}
        reacting_eids_df = pd.DataFrame(data_mod)
        reacting_eids_df_new = reacting_eids_df.T

        reacting_eids_df_new.to_csv('Export/' + time_str + '/reacting_eids_temp.csv', index=False)

        with open('Export/' + time_str + '/reacting_eids_temp.csv', 'r') as f:
            with open('Export/' + time_str + '/reacting_eids.csv', 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

        os.remove('Export/' + time_str + '/reacting_eids_temp.csv')

        with open('Export/' + time_str + '/reacting_eids.csv', newline='') as f:
            r = csv.reader(f)
            data = [line for line in r]

        with open('Export/' + time_str + '/reacting_eids.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['eid_list'])
            w.writerows(data)

    else:

        reacting_eids = [1,2,3]

        data_mod = {'Modifiable': reacting_eids}
        reacting_eids_df = pd.DataFrame(data_mod)
        reacting_eids_df_new = reacting_eids_df.T

        reacting_eids_df_new.to_csv('Export/' + time_str + '/reacting_eids_temp.csv', index=False)

        with open('Export/' + time_str + '/reacting_eids_temp.csv', 'r') as f:
            with open('Export/' + time_str + '/reacting_eids.csv', 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

        os.remove('Export/' + time_str + '/reacting_eids_temp.csv')

        with open('Export/' + time_str + '/reacting_eids.csv') as f:
            r = csv.reader(f)
            data = [line for line in r]

        with open('Export/' + time_str + '/reacting_eids.csv', 'w', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['all'])
            w.writerows(data)

    return
