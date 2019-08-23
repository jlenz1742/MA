import pandas as pd
import math
import os
import csv
import time
import json
import Plot


def create_csv_files_from_graph(graph, p_veins, p_arteries, key_word_all, summary_information, key_number_modi, radii):

    ''' Function transforms graph to three csv files to proceed with the simulation '''

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  Preparation                                                     #
    #                                                                                                                  #
    ####################################################################################################################

    time_str = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('Export/' + time_str)
    os.makedirs('Export/' + time_str + '\\' + 'mvn1_edit')
    os.makedirs('Export/' + time_str + '\\' + 'adjointMethod')

    ####################################################################################################################
    #                                                                                                                  #
    #                                                    Read Me                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    count_arterial_vessels = 0
    count_venous_vessels = 0
    count_capillary_bed_honeycomb = 0
    count_capillary_bed_connections = 0

    tot_volume_arterial_vessels = 0
    tot_volume_venous_vessels = 0
    tot_volume_capillary_bed_honeycomb = 0
    tot_volume_capillary_bed_connections = 0

    for edge in range(graph.ecount()):

        if graph.es[edge]['Type'] == 0:

            count_capillary_bed_honeycomb += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_capillary_bed_honeycomb += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 2:

            count_arterial_vessels += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_arterial_vessels += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 1:

            count_venous_vessels += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_venous_vessels += math.pi * math.pow(radius, 2) * len_edge

        elif graph.es[edge]['Type'] == 3:

            count_capillary_bed_connections += 1
            len_edge = graph.es[edge]['edge_length']
            radius = graph.es[edge]['diameter'] / 2

            tot_volume_capillary_bed_connections += math.pi * math.pow(radius, 2) * len_edge

    summary_information['# arterial vessel'] = count_arterial_vessels
    summary_information['# venous vessel'] = count_venous_vessels
    summary_information['# Capillary Bed vessels'] = count_capillary_bed_honeycomb
    summary_information['# Connection vessels'] = count_capillary_bed_connections

    summary_information['# Volume arterial vessel'] = tot_volume_arterial_vessels
    summary_information['# Volume venous vessel'] = tot_volume_venous_vessels
    summary_information['# Volume Capillary Bed vessels'] = tot_volume_capillary_bed_honeycomb
    summary_information['# Volume Connection vessels'] = tot_volume_capillary_bed_connections

    with open('Export/' + time_str + '/read_me.txt', "w") as file:
        json.dump(summary_information, file, indent=4)

    ####################################################################################################################
    #                                                                                                                  #
    #                                                  node_data.csv                                                   #
    #                                                                                                                  #
    ####################################################################################################################

    node_data_df = pd.DataFrame()

    node_data_df['x'] = [x / math.pow(10, 6) for x in graph.vs['x_coordinate']]
    node_data_df['y'] = [y / math.pow(10, 6) for y in graph.vs['y_coordinate']]
    node_data_df['z'] = [z / math.pow(10, 6) for z in graph.vs['z_coordinate']]
    node_data_df.to_csv('Export/' + time_str + '\\' + 'mvn1_edit' + '/node_data.csv', index=False)

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

                pressure.append(p_arteries)

        else:

            None

    node_boundary_data_df['nodeId'] = nodeid

    boundary_type = [1] * len(nodeid)
    node_boundary_data_df['boundaryType'] = boundary_type

    node_boundary_data_df['p'] = pressure

    flux = ['nan'] * len(nodeid)
    node_boundary_data_df['flux'] = flux

    node_boundary_data_df.to_csv('Export/' + time_str + '\\' + 'mvn1_edit' + '/node_boundary_data.csv', index=False)

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
    edge_data_df['Type'] = graph.es['Type']
    edge_data_df.to_csv('Export/' + time_str + '\\' + 'mvn1_edit' + '/edge_data.csv', index=False)

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

    activated_eids_df_new.to_csv('Export/' + time_str + '\\' + 'adjointMethod' + '/activated_eids_temp.csv', index=False)

    with open('Export/' + time_str + '\\' + 'adjointMethod' + '/activated_eids_temp.csv', 'r') as f:
        with open('Export/' + time_str + '\\' + 'adjointMethod' + '/activated_eids.csv', 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)

    os.remove('Export/' + time_str + '\\' + 'adjointMethod' + '/activated_eids_temp.csv')

    ####################################################################################################################
    #                                                                                                                  #
    #                                                reacting_eids.csv                                                 #
    #                                                                                                                  #
    ####################################################################################################################

    for radius in radii:

        os.makedirs('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius))
        name = 'modi_radius_' + str(radius)

        if key_word_all == 0:

            for modi in key_number_modi:

                reacting_eids = []

                for edge in range(graph.ecount()):

                    if graph.es[edge][name] == 1:

                        if modi == 0:

                            if graph.es[edge]['Type'] == 0 or graph.es[edge]['Type'] == 3:

                                reacting_eids.append(edge)

                        elif modi == 1:

                            if graph.es[edge]['Type'] == 1:

                                reacting_eids.append(edge)

                        elif modi == 2:

                            if graph.es[edge]['Type'] == 1 or graph.es[edge]['Type'] == 0 or graph.es[edge]['Type'] == 3:

                                reacting_eids.append(edge)

                        elif modi == 3:

                            reacting_eids.append(edge)

                data_mod = {'Modifiable': reacting_eids}
                reacting_eids_df = pd.DataFrame(data_mod)
                reacting_eids_df_new = reacting_eids_df.T

                reacting_eids_df_new.to_csv('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/temp.csv', index=False)

                with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/temp.csv', 'r') as f:

                    if modi == 0:

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_cb_only_specified_region.csv', 'w') as f1:
                            next(f)  # skip header line
                            for line in f:
                                f1.write(line)

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_cb_only_specified_region.csv',
                                  newline='') as f:
                            r = csv.reader(f)
                            data = [line for line in r]

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_cb_only_specified_region.csv', 'w',
                                  newline='') as f:
                            w = csv.writer(f)
                            w.writerow(['eid_list'])
                            w.writerows(data)

                    elif modi == 1:

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_only_specified_region.csv', 'w') as f1:
                            next(f)  # skip header line
                            for line in f:
                                f1.write(line)

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_only_specified_region.csv',
                                  newline='') as f:
                            r = csv.reader(f)
                            data = [line for line in r]

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_only_specified_region.csv', 'w',
                                  newline='') as f:
                            w = csv.writer(f)
                            w.writerow(['eid_list'])
                            w.writerows(data)

                    elif modi == 2:

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_cb_specified_region.csv', 'w') as f1:
                            next(f)  # skip header line
                            for line in f:
                                f1.write(line)

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_cb_specified_region.csv',
                                  newline='') as f:
                            r = csv.reader(f)
                            data = [line for line in r]

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_arteries_cb_specified_region.csv', 'w',
                                  newline='') as f:
                            w = csv.writer(f)
                            w.writerow(['eid_list'])
                            w.writerows(data)

                    elif modi == 3:

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_all_specified_region.csv', 'w') as f1:
                            next(f)  # skip header line
                            for line in f:
                                f1.write(line)

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_all_specified_region.csv', newline='') as f:
                            r = csv.reader(f)
                            data = [line for line in r]

                        with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_all_specified_region.csv', 'w', newline='') as f:
                            w = csv.writer(f)
                            w.writerow(['eid_list'])
                            w.writerows(data)

                os.remove('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/temp.csv')

        else:

            reacting_eids = [1, 2, 3]

            data_mod = {'Modifiable': reacting_eids}
            reacting_eids_df = pd.DataFrame(data_mod)
            reacting_eids_df_new = reacting_eids_df.T

            reacting_eids_df_new.to_csv('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_temp.csv', index=False)

            with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_temp.csv', 'r') as f:
                with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids.csv', 'w') as f1:
                    next(f)  # skip header line
                    for line in f:
                        f1.write(line)

            os.remove('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids_temp.csv')

            with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids.csv') as f:
                r = csv.reader(f)
                data = [line for line in r]

            with open('Export/' + time_str + '\\' + 'adjointMethod' + '\\' + 'radius_' + str(radius) + '/reacting_eids.csv', 'w', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['all'])
                w.writerows(data)

    ####################################################################################################################
    #                                                                                                                  #
    #                                                     Plots.csv                                                    #
    #                                                                                                                  #
    ####################################################################################################################

    # Modifiable

    if key_word_all == 0:

        for r in radii:
            a = 'modi_radius_' + str(r)
            Plot.plot_chosen_region(graph, a, 'Export/' + time_str + '\\' + a + '.png')

    # Activated

    Plot.plot_chosen_region(graph, 'Activated', 'Export/' + time_str + '\\' + 'activated_region' + '.png')

    # Total


    # Create Pickle File

    graph.write_pickle('Export/' + time_str + '\\' + 'graph.pkl')
    Plot.plot_graph(graph, 'Export/' + time_str + '\\' + 'graph_total.png')

    return
