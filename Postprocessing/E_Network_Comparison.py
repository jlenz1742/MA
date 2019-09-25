import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import os
import pandas as pd
import csv
import igraph as ig
from collections import OrderedDict

########################################################################################################################
#
# DIAMETER BOXPLOTS (R100 - ALL)
#
########################################################################################################################

# Box Plot Diameter All Vessel Types


def get_diameter_data_for_different_networks(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                diameter_changes_reacting_edges = t_new_array
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                diameter_changes_reacting_edges = t_new_array[reacting_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(diameter_changes_selection_FINAL)

            if radius == 'R_125':

                data_r_125.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(diameter_changes_selection_FINAL)

            if radius == 'R_175':

                data_r_175.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(diameter_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(diameter_changes_selection_FINAL)


    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_diameter_all_vessel_types(prozent):
    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_diameter_data_for_different_networks(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=1)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Only Modifiable Region included')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Diameter Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 17
    bottom = -2
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    print(upper_labels)
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .9, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_all_types.png')


# Box Plot Diameter Capillaries


def get_diameter_data_for_different_networks_capillaries(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                capillaries_eids = []

                for i in range(graph_.ecount()):

                    if graph_.es[i]['Type'] == 0 or graph_.es[i]['Type'] == 3:

                        capillaries_eids.append(i)

                diameter_changes_reacting_edges = t_new_array[capillaries_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 0 or graph_.es[reacting_eid]['Type'] == 3:
                        reacting_eids_new.append(reacting_eid)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
            diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_125':

                data_r_125.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_175':

                data_r_175.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(diameter_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(diameter_changes_selection_FINAL)

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_diameter_capillaries(prozent):

    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_diameter_data_for_different_networks_capillaries(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=1)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Only Modifiable Region included')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Diameter Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 17
    bottom = -2
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    print(upper_labels)
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .9, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_capillaries.png')


# Box Plot Diameter Arterioles and veins


def get_diameter_data_for_different_networks_art_ven(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:
            print(radius)

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                capillaries_eids = []

                for i in range(graph_.ecount()):

                    if graph_.es[i]['Type'] == 1 or graph_.es[i]['Type'] == 2:
                        capillaries_eids.append(i)

                diameter_changes_reacting_edges = t_new_array[capillaries_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                reacting_eids_new = []

                for reacting_eid in reacting_eids:

                    if graph_.es[reacting_eid]['Type'] == 1 or graph_.es[reacting_eid]['Type'] == 2:
                        reacting_eids_new.append(reacting_eid)

                diameter_changes_reacting_edges = t_new_array[reacting_eids_new]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
                diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_125':

                data_r_125.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_175':

                data_r_175.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(diameter_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(diameter_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(diameter_changes_selection_FINAL)

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_diameter_art_ven(prozent):
    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_diameter_data_for_different_networks_art_ven(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=1)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Only Modifiable Region included')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Diameter Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 17
    bottom = -2
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    print(upper_labels)
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .9, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Diameter_Change_R100_to_ALL\boxplot_diameter_art_ven.png')


########################################################################################################################
#
# DIAMETER BOXPLOTS - Direct Comparison (All Cap, All Arteries)
#
########################################################################################################################


def get_diameter_data_for_different_networks_arteries_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Arteries']
    # histogram_data_id = ['All']

    data = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:
            print(radius)

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                capillaries_eids = []

                for i in range(graph_.ecount()):

                    if graph_.es[i]['Type'] == 1 or graph_.es[i]['Type'] == 2:
                        capillaries_eids.append(i)

                diameter_changes_reacting_edges = t_new_array[capillaries_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                diameter_changes_reacting_edges = t_new_array[reacting_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
                diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            data.append(diameter_changes_selection_FINAL)

    return data


def plot_diameter_arteries_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_diameter_data_for_different_networks_arteries_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=1)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Only Modifiable Region included')
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative Diameter Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 40
    bottom = -8
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))

    print(upper_labels)
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_diameter_arterioles_only.png')


def get_diameter_data_for_different_networks_caps_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Cap']
    # histogram_data_id = ['All']

    data = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:
            print(radius)

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            t_1 = np.array(df_start['D'])
            t_2 = np.array(df_final['D'])

            t = t_2 / t_1
            t_new = []

            for i in range(len(t)):

                if (i % 2) == 0:
                    t_new.append(t[i])

            t_new_array = np.asarray(t_new)
            reacting_eids = []

            if radius == 'All':

                capillaries_eids = []

                for i in range(graph_.ecount()):

                    if graph_.es[i]['Type'] == 1 or graph_.es[i]['Type'] == 2:
                        capillaries_eids.append(i)

                diameter_changes_reacting_edges = t_new_array[capillaries_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            else:

                with open(current_input_path + 'reacting_eids.csv', newline='') as f:
                    reader = csv.reader(f)
                    row1 = next(reader)  # gets the first line
                    row2 = next(f)
                    reacting_eids = list(map(int, row2.split(',')))

                diameter_changes_reacting_edges = t_new_array[reacting_eids]
                diameter_changes_reacting_edges_percent = np.abs((diameter_changes_reacting_edges - 1) * 100)

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            diameter_changes_selection_index = np.where(
                diameter_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            diameter_changes_selection_FINAL = (
                diameter_changes_reacting_edges_percent[diameter_changes_selection_index[0]])

            data.append(diameter_changes_selection_FINAL)

    return data


def plot_diameter_caps_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_diameter_data_for_different_networks_caps_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=1)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Only Modifiable Region included')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Diameter Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 17
    bottom = -2
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))

    print(upper_labels)
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(
        r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_diameter_caps_only.png')


########################################################################################################################
#
# FLOW BOXPLOTS - Direct Comparison (All Cap, All Arteries)
#
########################################################################################################################


def get_flow_data_for_different_networks_arteries_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Arteries']
    # histogram_data_id = ['All']

    data = []
    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            for i in reversed(activated_eids):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            data.append(flow_changes_selection_FINAL)

    return data


def plot_flow_arteries_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_arteries_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))


    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_flow_arterioles.png')


def get_flow_data_for_different_networks_caps_direct_comp(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['All_Cap']
    # histogram_data_id = ['All']

    data = []
    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            for i in reversed(activated_eids):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            data.append(flow_changes_selection_FINAL)

    return data


def plot_flow_caps_direct_comp(prozent):

    random_dists = ['Network 1', 'Network 2', ' Network 3', 'Network 4']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_caps_direct_comp(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))


    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Direct_Comparison\boxplot_flow_caps.png')


########################################################################################################################
#
# FLOW BOXPLOTS (R100 - ALL)
#
########################################################################################################################

# Box Plot Flow All Vessel Types


def get_flow_data_for_different_networks(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            for i in reversed(activated_eids):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(flow_changes_selection_FINAL)

            if radius == 'R_125':

                data_r_125.extend(flow_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(flow_changes_selection_FINAL)

            if radius == 'R_175':

                data_r_175.extend(flow_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(flow_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(flow_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(flow_changes_selection_FINAL)

    print(len(data_r_100), data_r_100)
    print(len(data_r_125), data_r_125)
    print(len(data_r_150), data_r_150)
    print(len(data_r_175), data_r_175)
    print(len(data_r_200), data_r_200)
    print(len(data_r_300), data_r_300)
    print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_flow_all_vessel_types(prozent):
    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\boxplot_flow_all_types.png')


# Box Plot Flow Capilleries


def get_flow_data_for_different_networks_cap(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            edges_to_be_deleted = []

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

                    None

                else:

                    edges_to_be_deleted.append(edge)

            edges_to_be_deleted.extend(activated_eids)
            edges_to_be_deleted.sort()

            edges_to_be_deleted = list(OrderedDict.fromkeys(edges_to_be_deleted))

            for i in reversed(edges_to_be_deleted):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(flow_changes_selection_FINAL)

            if radius == 'R_125':

                data_r_125.extend(flow_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(flow_changes_selection_FINAL)

            if radius == 'R_175':

                data_r_175.extend(flow_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(flow_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(flow_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(flow_changes_selection_FINAL)

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_125), data_r_125)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_175), data_r_175)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_flow_cap(prozent):

    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_cap(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\boxplot_flow_cap.png')


# Box Plot Flow Arterioles and Venules


def get_flow_data_for_different_networks_art_ven(percent):

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All']
    # histogram_data_id = ['All']

    data_r_100 = []
    data_r_125 = []
    data_r_150 = []
    data_r_175 = []
    data_r_200 = []
    data_r_300 = []
    data_r_all = []

    for network in networks:

        graph_ = ig.Graph()

        if network == '0_Out_':

            graph_ = ig.Graph.Read_Pickle(path_0_graph)

        elif network == '3_Out_':

            graph_ = ig.Graph.Read_Pickle(path_3_graph)

        elif network == '4_Out_':

            graph_ = ig.Graph.Read_Pickle(path_4_graph)

        elif network == '6_Out_':

            graph_ = ig.Graph.Read_Pickle(path_6_graph)

        print(network)

        for radius in histogram_data_id:

            current_input_path = path + '\\' + network + '\\' + radius + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + radius + '\\' + 'out' + '\\'
            meshdata_start = current_output_path + 'meshdata_9999.csv'

            r = csv.reader(open(current_input_path + 'activated_eids.csv'))
            lines = list(r)
            activated_eids = lines[0]
            activated_eids = list(map(int, activated_eids))

            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(current_output_path):
                for file in f:

                    if file[0:8] == 'meshdata':
                        files.append(file)

            files.remove('meshdata_9999.csv')
            meshdata_final = current_output_path + files[0]

            df_start = pd.read_csv(meshdata_start)
            df_final = pd.read_csv(meshdata_final)

            f_start = np.array(df_start['tav_Fplasma'])
            f_final = np.array(df_final['tav_Fplasma'])

            f_start_new = []
            f_end_new = []

            for i in range(len(f_start)):

                if (i % 2) == 0:
                    f_start_new.append(f_start[i])
                    f_end_new.append((f_final[i]))

            f_start_new_array = np.asarray(f_start_new)
            f_end_new_array = np.asarray(f_end_new)

            f_ratio = f_start_new_array / f_end_new_array
            flow_changes_reacting_edges_percent = np.abs((f_ratio - 1) * 100)

            # Length

            l_start = np.array(df_start['L'])
            l_start_new = []

            for i in range(len(l_start)):

                if (i % 2) == 0:
                    l_start_new.append(l_start[i])

            l_start_new_array = np.asarray(l_start_new)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))
            # print(len(activated_eids))

            edges_to_be_deleted = []

            for edge in range(graph_.ecount()):

                if graph_.es[edge]['Type'] == 1 or graph_.es[edge]['Type'] == 2:

                    None

                else:

                    edges_to_be_deleted.append(edge)

            edges_to_be_deleted.extend(activated_eids)
            edges_to_be_deleted.sort()

            edges_to_be_deleted = list(OrderedDict.fromkeys(edges_to_be_deleted))

            for i in reversed(edges_to_be_deleted):

                flow_changes_reacting_edges_percent = np.delete(flow_changes_reacting_edges_percent, i)
                l_start_new_array = np.delete(l_start_new_array, i)

            # print(len(flow_changes_reacting_edges_percent))
            # print(len(l_start_new_array))

            # HIER KANN BEGRENZUNG EINGEBAUEN WERDEN

            grenze_prozentuela_anderung = percent
            flow_changes_selection_index = np.where(flow_changes_reacting_edges_percent > grenze_prozentuela_anderung)
            flow_changes_selction = (flow_changes_reacting_edges_percent[flow_changes_selection_index[0]])
            flow_changes_selection_FINAL = []

            counter = 0

            for x in flow_changes_selction:

                if x < 100:

                    counter += 1
                    flow_changes_selection_FINAL.append(x)

                else:

                    # print(x)
                    None

            # print(len(flow_changes_selection_FINAL))
            # print(counter)
            # print(diameter_changes_selection_FINAL)

            if radius == 'R_100':

                data_r_100.extend(flow_changes_selection_FINAL)

            if radius == 'R_125':

                data_r_125.extend(flow_changes_selection_FINAL)

            elif radius == 'R_150':

                data_r_150.extend(flow_changes_selection_FINAL)

            if radius == 'R_175':

                data_r_175.extend(flow_changes_selection_FINAL)

            elif radius == 'R_200':

                data_r_200.extend(flow_changes_selection_FINAL)

            elif radius == 'R_300':

                data_r_300.extend(flow_changes_selection_FINAL)

            elif radius == 'All':

                data_r_all.extend(flow_changes_selection_FINAL)

    # print(len(data_r_100), data_r_100)
    # print(len(data_r_125), data_r_125)
    # print(len(data_r_150), data_r_150)
    # print(len(data_r_175), data_r_175)
    # print(len(data_r_200), data_r_200)
    # print(len(data_r_300), data_r_300)
    # print(len(data_r_all), data_r_all)

    return [data_r_100, data_r_125, data_r_150, data_r_175, data_r_200, data_r_300, data_r_all]


def plot_flow_art_ven(prozent):

    random_dists = ['R100', 'R_125', ' R150', 'R_175', 'R200', 'R300', 'All']

    # N = 10
    #
    # norm = np.random.normal(1, 1, N)
    # print(norm)
    # logn = np.random.lognormal(1, 1, N)
    # expo = np.random.exponential(1, N)
    # gumb = np.random.gumbel(6, 4, N)
    # tria = np.random.triangular(2, 9, 11, N)

    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array
    # bootstrap_indices = np.random.randint(0, N, N)

    data = get_flow_data_for_different_networks_art_ven(prozent)
    # data = [
    #     norm,
    #     logn,
    #     expo,
    #     gumb,
    #     tria,
    # ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+', markersize=0.2)
    plt.setp(bp['medians'], color='green')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Vessels outside activated region included only')
    ax1.set_xlabel('Modifiable Regions')
    ax1.set_ylabel('Relative Flow Change in %')

    # Now fill the boxes with desired colors
    box_colors = ['lightgray', 'lightgray']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue

        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))

        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            # ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='Royalblue', marker='*', markeredgecolor='Royalblue')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 60
    bottom = -14
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(random_dists,
                        rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    lower_labels = []
    lower_labels.append(str(np.round(np.mean(data[0]), 2)))
    lower_labels.append(str(np.round(np.mean(data[1]), 2)))
    lower_labels.append(str(np.round(np.mean(data[2]), 2)))
    lower_labels.append(str(np.round(np.mean(data[3]), 2)))
    lower_labels.append(str(np.round(np.mean(data[4]), 2)))
    lower_labels.append(str(np.round(np.mean(data[5]), 2)))
    lower_labels.append(str(np.round(np.mean(data[6]), 2)))

    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .1, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='green')

        ax1.text(pos[tick], .05, lower_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k], color='Royalblue')

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, f'{N} Random Numbers',
    #          backgroundcolor=box_colors[0], color='black', weight='roman',
    #          size='x-small')
    # fig.text(0.80, 0.045, 'IID Bootstrap Resample',
    #          backgroundcolor=box_colors[1],
    #          color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')

    plt.savefig(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Flow_Change_R100_to_ALL\boxplot_flow_art_ven.png')


########################################################################################################################
#
# Distance Plot
#
########################################################################################################################


########################################################################################################################
# START
########################################################################################################################

prozent_grenze_diameter = 1
prozent_grenze_flow = 1

# # FLOW (R100 to ALL)
# plot_flow_all_vessel_types(prozent_grenze_flow)
# plot_flow_cap(prozent_grenze_flow)
# plot_flow_art_ven(prozent_grenze_flow)
#
#
# # DIAMETER (R100 to ALL)
# plot_diameter_all_vessel_types(prozent_grenze_diameter)
# plot_diameter_art_ven(prozent_grenze_diameter)
# plot_diameter_capillaries(prozent_grenze_diameter)
#
# # Flow (All Cap, All Art)
# plot_flow_arteries_direct_comp(prozent_grenze_flow)
# plot_flow_caps_direct_comp(prozent_grenze_flow)


# # Diameter (All Cap, All Art)
# plot_diameter_arteries_direct_comp(prozent_grenze_diameter)
plot_diameter_caps_direct_comp(prozent_grenze_diameter)
