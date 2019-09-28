import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import csv
import igraph as ig
import os
from matplotlib.ticker import PercentFormatter


def adjoint_data_plot(path, target_path_):

    size_figure = (15, 15)
    fig = plt.figure(figsize=size_figure)
    plt.suptitle("")

    ####################################################################################################################
    #                                                                                                                  #
    #                                                   Rho Plot                                                       #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 1)

    path_adj_data = path
    df = pd.read_csv(path_adj_data)
    it = np.asarray(df.it)
    cost = np.asarray(df.totalCost)
    plt.semilogy(it, cost)

    plt.title("Cost Function", fontsize=10)
    plt.grid(True)

    ####################################################################################################################
    #                                                                                                                  #
    #                                               Regularization Rho n                                               #
    #                                                                                                                  #
    ####################################################################################################################

    plt.subplot(2, 1, 2)

    path_adj_data = path
    df = pd.read_csv(path_adj_data)
    it = np.asarray(df.it)
    regularization_rho_n = np.asarray(df.regularization_rho_n)
    rho = np.asarray(df.rho)
    plt.plot(it, rho, label='Rho', linestyle='dashed')
    plt.plot(it, regularization_rho_n, label='Rho_n')

    plt.title("Regularization Rho n", fontsize=10)
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05 * max(regularization_rho_n))

    plt.savefig(target_path_ + '\\' + 'Adjoint_plot.png')
    plt.clf()


def compute_flow_change(start_file, end_file, activated_ids, target_path_):

    df_start = pd.read_csv(start_file)
    df_final = pd.read_csv(end_file)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    # FLOW

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

    f_ratio = f_start_new_array/f_end_new_array

    f_activated = f_ratio[activated_ids]

    # LENGTH

    l_start = np.array(df_start['L'])
    l_start_new = []

    for i in range(len(l_start)):

        if (i % 2) == 0:
            l_start_new.append(l_start[i])

    l_start_new_array = np.asarray(l_start_new)

    l_activated = l_start_new_array[activated_ids]

    # Gewichtete Flussänderungsberechnung

    q_0 = l_activated * np.abs(f_start_new_array[activated_ids])
    q_0_sum = np.sum(q_0)

    dir_ = np.ones(len(activated_ids))
    dir_tool = np.where(f_start_new_array[activated_ids] < 0)

    dir_[dir_tool] = -1

    q_1 = l_activated * dir_ * f_end_new_array[activated_ids]
    q_1_sum = np.sum(q_1)

    relative_flow_change = q_1_sum/q_0_sum

    if 1.29 < (q_1_sum/q_0_sum) < 1.31:

        with open(target_path_ + '\Check_Flow_Change.txt', 'w') as f:
            f.write("Succesfull, Flow Change is: %f" % relative_flow_change)

    else:

        print('No Successs: ', relative_flow_change)


def compute_diameter_change(start_file, end_file, activated_ids):
    df_start = pd.read_csv(start_file)
    df_final = pd.read_csv(end_file)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    d_ratio = d_start_new_array / d_end_new_array

    d_ratio_activated = d_ratio[activated_ids]

    print(d_ratio_activated)
    mean_alpha = np.mean(d_ratio_activated)
    std_alpha = np.std(d_ratio_activated)

    d_ratio_non_activated = np.delete(d_ratio, activated_ids)
    print()


def plot_3_d_diameter(meshdata_start, meshdata_final, target_path_):
    df_start = pd.read_csv(meshdata_start)
    df_final = pd.read_csv(meshdata_final)


    x = df_start['x']
    x_min = min(x)
    x_max = max(x)

    y = df_start['y']
    y_min = min(y)
    y_max = max(y)

    z = df_start['z']
    z_min = min(z)
    z_max = max(z)

    f_plasma_1 = np.array(df_start['Fplasma'])
    f_plasma_2 = np.array(df_final['Fplasma'])

    f_plasma_ratio = f_plasma_2 / f_plasma_1

    test = []

    for i in range(len(f_plasma_ratio)):

        if f_plasma_ratio[i] > 2:

            test.append(1)

        elif f_plasma_ratio[i] < 0:

            test.append(1)

        else:

            test.append(f_plasma_ratio[i])

    f_plasma_ratio_processed = np.asarray(test)

    print('Max and Min Plasma Flow Processed: ', max(f_plasma_ratio_processed), min(f_plasma_ratio_processed))

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
            t_new.append(t[i])

    t_new_array = np.asarray(t_new)
    reacting_ids = np.where(t_new_array != 1)

    print(f_plasma_ratio_processed[reacting_ids])

    # generate a list of (x,y,z) points
    points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)

    # set up a list of segments
    segs_temp = np.concatenate([points[:-1], points[1:]], axis=1)
    segs = []

    for i in range(len(segs_temp)):

        if (i % 2) == 0:
            segs.append(segs_temp[i])

    # make the collection of segments

    max_diameter_change = max(t_new_array)-1
    min_diameter_change = np.absolute(np.absolute(min(t_new_array)) - 1)

    a = np.where(1.005 > t_new_array)
    b = np.where(0.995 < t_new_array)
    c = np.intersect1d(a, b)

    transparent_array = np.copy(np.asarray(segs)[c, ])
    transparent_diameters_ratio = np.copy(t_new_array[c, ])
    transparent_diameters = np.copy(np.asarray(t_2_new)[c, ])
    non_transparent_segs = []
    non_transparent_diameters_ratio = []
    non_transparent_diameters = []

    indices_non_transparent = list(range(len(t_new_array)))

    for i in c:

        indices_non_transparent.remove(i)

    for i in indices_non_transparent:

        non_transparent_segs.append(segs[i])
        non_transparent_diameters_ratio.append((t_new_array[i]))
        non_transparent_diameters.append(t_2_new[i])

    if max_diameter_change > min_diameter_change:

        min_change = 1-max_diameter_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.3, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=min_change, vmax=max(t_new_array)))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=min_change, vmax=max(t_new_array)))
    else:

        max_change = 1 + min_diameter_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.3, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=min(t_new_array), vmax=max_change))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=min(t_new_array), vmax=max_change))

    lb.set_array(np.asarray(non_transparent_diameters_ratio))
    lc.set_array(transparent_diameters_ratio)  # color the segments by our parameter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    plt.xticks(fontsize=7, rotation=0)
    plt.yticks(fontsize=7, rotation=0)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_zlabel('Z Coordinate', rotation=0)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Bonus: To get rid of the grid as well:
    ax.grid(True)

    ax.add_collection3d(lb)
    ax.add_collection3d(lc)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.colorbar(lb, label='Relative Diameter Change')
    plt.savefig(target_path_ + '\\' + 'diameter_change_3D.png')
    plt.clf()


def plot_3_d_plasma(meshdata_start, meshdata_final, target_path_):

    df_start = pd.read_csv(meshdata_start)
    df_final = pd.read_csv(meshdata_final)

    # DIAMETER

    d_start = np.array(df_start['D'])
    d_final = np.array(df_final['D'])

    d_start_new = []
    d_end_new = []

    for i in range(len(df_start)):

        if (i % 2) == 0:
            d_start_new.append(d_start[i])
            d_end_new.append((d_final[i]))

    d_start_new_array = np.asarray(d_start_new)
    d_end_new_array = np.asarray(d_end_new)

    # FLOW

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

    f_ratio = f_start_new_array/f_end_new_array

    x = df_start['x']
    x_min = min(x)
    x_max = max(x)

    y = df_start['y']
    y_min = min(y)
    y_max = max(y)

    z = df_start['z']
    z_min = min(z)
    z_max = max(z)

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    t = t_2 / t_1
    t_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_2_new.append(t_2[i] / (4 / math.pow(10, 6)))
            t_new.append(t[i])

    t_new_array = np.asarray(t_new)

    # generate a list of (x,y,z) points
    points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)

    # set up a list of segments
    segs_temp = np.concatenate([points[:-1], points[1:]], axis=1)
    segs = []

    for i in range(len(segs_temp)):

        if (i % 2) == 0:
            segs.append(segs_temp[i])

    # make the collection of segments

    max_f_ratio_change = max(f_ratio)-1
    min_f_ratio_change = np.absolute(np.absolute(min(f_ratio)) - 1)

    a = np.where(1.02 > t_new_array)
    b = np.where(0.98 < t_new_array)
    c = np.intersect1d(a, b)

    transparent_array = np.copy(np.asarray(segs)[c, ])
    transparent_diameters_ratio = np.copy(t_new_array[c, ])
    transparent_flow_ratio = np.copy(f_ratio[c, ])
    transparent_diameters = np.copy(np.asarray(t_2_new)[c, ])
    non_transparent_segs = []
    non_transparent_diameters_ratio = []
    non_transparent_diameters = []
    non_transpartent_flow_ratio = []

    indices_non_transparent = list(range(len(t_new_array)))

    for i in c:

        indices_non_transparent.remove(i)

    for i in indices_non_transparent:

        non_transparent_segs.append(segs[i])
        non_transparent_diameters_ratio.append((t_new_array[i]))
        non_transparent_diameters.append(t_2_new[i])
        non_transpartent_flow_ratio.append(f_ratio[i])

    if max_f_ratio_change > min_f_ratio_change:

        min_change = 1-max_f_ratio_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.5, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))
    else:

        max_change = 1 + min_f_ratio_change

        lc = Line3DCollection(transparent_array, linewidths=transparent_diameters, alpha=0.5, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

        lb = Line3DCollection(non_transparent_segs, linewidths=non_transparent_diameters, alpha=1, cmap=plt.get_cmap('coolwarm', 10),
                              norm=mpl.colors.Normalize(vmin=0.5, vmax=1.5))

    lc.set_array(transparent_flow_ratio)  # color the segments by our parameter
    lb.set_array(np.asarray(non_transpartent_flow_ratio))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    plt.xticks(fontsize=7, rotation=0)
    plt.yticks(fontsize=7, rotation=0)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_zlabel('Z Coordinate', rotation=0)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Bonus: To get rid of the grid as well:
    ax.grid(True)


    ax.add_collection3d(lb)
    ax.add_collection3d(lc)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.colorbar(lb, label='Relative Flow Change')
    plt.savefig(target_path_ + '\\' + 'flow_change_3D.png')
    plt.clf()


def plot_diameter_change_per_total_volume(graph, meshdata_start, meshdata_final, target_path_):

    df_start = pd.read_csv(meshdata_start)
    df_final = pd.read_csv(meshdata_final)

    graph_ = ig.Graph.Read_Pickle(graph)

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    length = np.array(df_final['L'])

    t = t_2 / t_1
    t_new = []
    l_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_new.append(t[i])
            t_2_new.append(t_2[i])
            l_new.append(length[i])

    t_new_array = np.asarray(t_new)
    t_2_new_array = np.asarray(t_2_new)
    l_new_array = np.asarray(l_new)

    diameter_change = np.asarray(np.absolute((t_new_array-1)*100))
    print(diameter_change[0:10])

    diameter_change_capillaries = []
    volume_capillaries = []
    diameter_change_arteries = []
    volume_arteries = []
    diameter_change_veins = []
    volume_veins = []

    for edge in range(graph_.ecount()):

        if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

            diameter_change_capillaries.append(diameter_change[edge])
            volume_capillaries.append(l_new_array[edge]*math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

        elif graph_.es[edge]['Type'] == 1:

            diameter_change_veins.append(diameter_change[edge])
            volume_veins.append(l_new_array[edge] * math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

        elif graph_.es[edge]['Type'] == 2:

            diameter_change_arteries.append((diameter_change[edge]))
            volume_arteries.append(l_new_array[edge] * math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

    diameter_change_capillaries = np.asarray(diameter_change_capillaries)
    diameter_change_arteries = np.asarray(diameter_change_arteries)
    diameter_change_veins = np.asarray(diameter_change_veins)
    volume_capillaries = np.asarray(volume_capillaries)
    volume_veins = np.asarray(volume_veins)
    volume_arteries = np.asarray(volume_arteries)

    # index = np.where(np.asarray(diameter_change) > 0.1)
    # index_list = list(index[0])
    # # print(index_list)
    # diameter_change_new = diameter_change[index_list]

    prozent_grenze = 0.25
    index_cap = np.where(np.asarray(diameter_change_capillaries) > prozent_grenze)
    index_list_cap = list(index_cap[0])
    diameter_change_cap_new = diameter_change_capillaries[index_list_cap]
    volume_cap_new = volume_capillaries[index_list_cap]

    index_art = np.where(np.asarray(diameter_change_arteries) > prozent_grenze)
    index_list_art = list(index_art[0])
    diameter_change_art_new = diameter_change_arteries[index_list_art]
    volume_art_new = volume_arteries[index_list_art]

    index_ve = np.where(np.asarray(diameter_change_veins) > prozent_grenze)
    index_list_ve = list(index_ve[0])
    diameter_change_ve_new = diameter_change_veins[index_list_ve]
    volume_ve_new = volume_veins[index_list_ve]

    # NEW

    # Generate a normal distribution, center at x=0 and y=5

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(10, 5))
    bins = 20

    volume_total = np.sum(volume_capillaries) + np.sum(volume_arteries) + np.sum(volume_veins)
    # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(diameter_change_cap_new, bins=bins, weights=np.ones(len(diameter_change_cap_new)) / len(diameter_change_capillaries), density=False, color='grey', alpha=0.5, label='Capillaries', ec="k")
    axs[0].hist(diameter_change_cap_new, bins=bins, weights=volume_cap_new / volume_total, density=False, color='grey', alpha=0.5, label='Capillaries', ec="k")
    # axs[0].set_title('Capillaries')
    axs[0].legend()
    axs[0].grid()
    axs[0].set(ylabel='Volume Percentage of affected Vessels')

    axs[1].hist(diameter_change_art_new, bins=bins, weights=volume_art_new / volume_total, density=False, color='red', alpha=0.5, label='Arterioles', ec="k")
    # axs[1].set_title('Arterioles')
    axs[1].legend()
    axs[1].grid()

    axs[2].hist(diameter_change_ve_new, bins=bins, weights=volume_ve_new / volume_total, density=False, color='blue', alpha=0.5, label='Venules', ec="k")
    # axs[2].set_title('Venules')
    axs[2].set_ylim()
    plt.legend()
    plt.grid()

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.xlabel(' ')
    fig.text(0.5, 0.02, 'Change of Diameter in %', ha='center')
    plt.savefig(target_path_ + '\\' + 'diameter_change_histogram_per_tot_volume.png')
    plt.clf()

    return

    # OLD

    # # print(diameter_change)
    # names = ['Capillaries', 'Veins', 'Arteries']
    # colors = ['#E69F00', '#56B4E9', '#D55E00']
    # n, bins, patches = plt.hist([diameter_change_cap_new, diameter_change_ve_new, diameter_change_art_new]
    #                             , bins='auto', density=False, color=colors, stacked=True, alpha=0.75, label=names)
    #
    # plt.xlabel('Diameter Change in %')
    # plt.ylabel('Number of Vessels')
    # plt.xlim(0, 15)
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(target_path_ + '\\' + 'diameter_change_histogram.png')
    # plt.clf()


def plot_diameter_change_per_volume_of_vessel_type(graph, meshdata_start, meshdata_final, target_path_):

    df_start = pd.read_csv(meshdata_start)
    df_final = pd.read_csv(meshdata_final)

    graph_ = ig.Graph.Read_Pickle(graph)

    t_1 = np.array(df_start['D'])
    t_2 = np.array(df_final['D'])

    length = np.array(df_final['L'])

    t = t_2 / t_1
    t_new = []
    l_new = []
    t_2_new = []

    for i in range(len(t)):

        if (i % 2) == 0:
            t_new.append(t[i])
            t_2_new.append(t_2[i])
            l_new.append(length[i])

    t_new_array = np.asarray(t_new)
    t_2_new_array = np.asarray(t_2_new)
    l_new_array = np.asarray(l_new)

    diameter_change = np.asarray(np.absolute((t_new_array-1)*100))
    print(diameter_change[0:10])

    diameter_change_capillaries = []
    volume_capillaries = []
    diameter_change_arteries = []
    volume_arteries = []
    diameter_change_veins = []
    volume_veins = []

    for edge in range(graph_.ecount()):

        if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

            diameter_change_capillaries.append(diameter_change[edge])
            volume_capillaries.append(l_new_array[edge]*math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

        elif graph_.es[edge]['Type'] == 1:

            diameter_change_veins.append(diameter_change[edge])
            volume_veins.append(l_new_array[edge] * math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

        elif graph_.es[edge]['Type'] == 2:

            diameter_change_arteries.append((diameter_change[edge]))
            volume_arteries.append(l_new_array[edge] * math.pi * 0.25 * math.pow(t_2_new_array[edge], 2))

    diameter_change_capillaries = np.asarray(diameter_change_capillaries)
    diameter_change_arteries = np.asarray(diameter_change_arteries)
    diameter_change_veins = np.asarray(diameter_change_veins)
    volume_capillaries = np.asarray(volume_capillaries)
    volume_veins = np.asarray(volume_veins)
    volume_arteries = np.asarray(volume_arteries)

    # index = np.where(np.asarray(diameter_change) > 0.1)
    # index_list = list(index[0])
    # # print(index_list)
    # diameter_change_new = diameter_change[index_list]

    prozent_grenze = 0.25
    index_cap = np.where(np.asarray(diameter_change_capillaries) > prozent_grenze)
    index_list_cap = list(index_cap[0])
    diameter_change_cap_new = diameter_change_capillaries[index_list_cap]
    volume_cap_new = volume_capillaries[index_list_cap]

    index_art = np.where(np.asarray(diameter_change_arteries) > prozent_grenze)
    index_list_art = list(index_art[0])
    diameter_change_art_new = diameter_change_arteries[index_list_art]
    volume_art_new = volume_arteries[index_list_art]

    index_ve = np.where(np.asarray(diameter_change_veins) > prozent_grenze)
    index_list_ve = list(index_ve[0])
    diameter_change_ve_new = diameter_change_veins[index_list_ve]
    volume_ve_new = volume_veins[index_list_ve]

    # NEW

    # Generate a normal distribution, center at x=0 and y=5

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(10, 5))
    bins = 20

    volume_total = np.sum(volume_capillaries) + np.sum(volume_arteries) + np.sum(volume_veins)
    # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(diameter_change_cap_new, bins=bins, weights=np.ones(len(diameter_change_cap_new)) / len(diameter_change_capillaries), density=False, color='grey', alpha=0.5, label='Capillaries', ec="k")
    axs[0].hist(diameter_change_cap_new, bins=bins, weights=volume_cap_new / np.sum(volume_capillaries), density=False, color='grey', alpha=0.5, label='Capillaries', ec="k")
    # axs[0].set_title('Capillaries')
    axs[0].legend()
    axs[0].grid()
    axs[0].set(ylabel='Volume Percentage of affected Vessels')

    axs[1].hist(diameter_change_art_new, bins=bins, weights=volume_art_new / np.sum(volume_arteries), density=False, color='red', alpha=0.5, label='Arterioles', ec="k")
    # axs[1].set_title('Arterioles')
    axs[1].legend()
    axs[1].grid()

    axs[2].hist(diameter_change_ve_new, bins=bins, weights=volume_ve_new / np.sum(volume_veins), density=False, color='blue', alpha=0.5, label='Venules', ec="k")
    # axs[2].set_title('Venules')
    axs[2].set_ylim()
    plt.legend()
    plt.grid()

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.xlabel(' ')
    fig.text(0.5, 0.02, 'Change of Diameter in %', ha='center')
    plt.savefig(target_path_ + '\\' + 'diameter_change_histogram_per_volume_of_given_types.png')
    plt.clf()

    return

    # OLD

    # # print(diameter_change)
    # names = ['Capillaries', 'Veins', 'Arteries']
    # colors = ['#E69F00', '#56B4E9', '#D55E00']
    # n, bins, patches = plt.hist([diameter_change_cap_new, diameter_change_ve_new, diameter_change_art_new]
    #                             , bins='auto', density=False, color=colors, stacked=True, alpha=0.75, label=names)
    #
    # plt.xlabel('Diameter Change in %')
    # plt.ylabel('Number of Vessels')
    # plt.xlim(0, 15)
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(target_path_ + '\\' + 'diameter_change_histogram.png')
    # plt.clf()


def plot_3d_cube(flows, cube_flow_max, xticks, yticks, z_coord, graph):

    print(flows[0])
    print(flows[1])
    print(flows[2])
    print(flows[3])

    x_art = []
    y_art = []
    x_ven = []
    y_ven = []

    for v in range(graph.vcount()):

        if graph.vs[v]['Type'] == 1:

            x_ven.append(graph.vs[v]['x_coordinate'])
            y_ven.append(graph.vs[v]['y_coordinate'])

        elif graph.vs[v]['Type'] == 2:

            x_art.append(graph.vs[v]['x_coordinate'])
            y_art.append(graph.vs[v]['y_coordinate'])

    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(right=0.9)
    dmin, dmax = 0.8, cube_flow_max

    # subplot number 1
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.title.set_text(str(min(z_coord[0])) + ' < z < ' + str(max(z_coord[0])))
    plt.xticks(xticks[0], size=6)
    plt.yticks(yticks[0], size=6)
    # plt.scatter(x=x_art, y=y_art, s=0.5, c='r')
    # plt.scatter(x=x_ven, y=y_ven, s=0.5, c='b')
    plt.imshow(flows[0], vmin=dmin, vmax=dmax, extent=[min(xticks[0]), max(xticks[0]), min(yticks[0]), max(yticks[0])])

    # subplot number 2
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.title.set_text(str(min(z_coord[1])) + ' < z < ' + str(max(z_coord[1])))
    plt.xticks(xticks[1], size=6)
    plt.yticks(yticks[1], size=6)
    # plt.scatter(x=x_art, y=y_art, s=0.5, c='r')
    # plt.scatter(x=x_ven, y=y_ven, s=0.5, c='b')
    plt.imshow(flows[1], vmin=dmin, vmax=dmax, extent=[min(xticks[1]), max(xticks[1]), min(yticks[1]), max(yticks[1])])

    # subplot number 3
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.title.set_text(str(min(z_coord[2])) + ' < z < ' + str(max(z_coord[2])))
    plt.xticks(xticks[2], size=6)
    plt.yticks(yticks[2], size=6)
    # plt.scatter(x=x_art, y=y_art, s=0.5, c='r')
    # plt.scatter(x=x_ven, y=y_ven, s=0.5, c='b')
    plt.imshow(flows[2], vmin=dmin, vmax=dmax, extent=[min(xticks[2]), max(xticks[2]), min(yticks[2]), max(yticks[2])])

    # subplot number 4
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.title.set_text(str(min(z_coord[3])) + ' < z < ' + str(max(z_coord[3])))
    plt.xticks(xticks[3], size=6)
    plt.yticks(yticks[3], size=6)
    # plt.scatter(x=x_art, y=y_art, s=0.5, c='r')
    # plt.scatter(x=x_ven, y=y_ven, s=0.5, c='b')
    plt.imshow(flows[3], vmin=dmin, vmax=dmax, extent=[min(xticks[3]), max(xticks[3]), min(yticks[3]), max(yticks[3])])

    # subplot for colorbar

    ax_cbar = fig.add_axes([0.1, 0.1, 0.82, 0.05])
    plt.colorbar(cax=ax_cbar, orientation='horizontal', label='Flow Rate [$m^3/s$]')
    plt.show()


def distance_diameter_check():

    return


def flow_rate_cube(graph, cube_side_length, meshdata_file, meshdata_final):

    Final_Flows_Output = []
    Final_Edge_Eids_Output = []

    df_start = pd.read_csv(meshdata_file)
    df_final = pd.read_csv(meshdata_final)

    # x segmentation

    x_min = min(graph.vs['x_coordinate'])
    x_max = max(graph.vs['x_coordinate'])

    delta_x = x_max - x_min
    number_of_cubes_x = math.ceil(delta_x / cube_side_length)

    regions_x = []
    labels_x = []

    for k in range(number_of_cubes_x):

        start = x_min + k * cube_side_length
        end = start + cube_side_length

        labels_x.append(np.round((start+end)/2, 2))
        regions_x.append([start, end])

    # y segmentation

    y_min = min(graph.vs['y_coordinate'])
    y_max = max(graph.vs['y_coordinate'])

    delta_y = y_max - y_min

    number_of_cubes_y = math.ceil(delta_y / cube_side_length)

    regions_y = []
    labels_y = []

    for k in range(number_of_cubes_y):

        start = y_min + k * cube_side_length
        end = start + cube_side_length

        labels_y.append(np.round((start+end)/2, 2))
        regions_y.append([start, end])

    # z segmentation

    z_min = min(graph.vs['z_coordinate'])
    z_max = max(graph.vs['z_coordinate'])

    delta_z = z_max - z_min

    number_of_cubes_z = math.ceil(delta_z / cube_side_length)

    regions_z = []
    labels_z = []

    for k in range(number_of_cubes_z):

        start = z_min + k * cube_side_length
        end = start + cube_side_length

        labels_z.append(np.round((start+end)/2, 2))
        regions_z.append([start, end])

    # Initialize Dictionary for cubes

    cube_coordinates = []
    cube_position = []

    for x in range(number_of_cubes_x):

        for y in range(number_of_cubes_y):

            for z in range(number_of_cubes_z):

                # print(x, y, z)
                cube_coordinates.append([regions_x[x], regions_y[y], regions_z[z]])
                cube_position.append([x, y, z])


    cube_dictionary_edge_ids = {}
    cube_dictionary_flow_rates = {}
    cube_dictionary_length = {}

    for k in range(len(cube_coordinates)):

        cube_dictionary_edge_ids[str(k)] = []
        cube_dictionary_flow_rates[str(k)] = []
        cube_dictionary_length[str(k)] = []

    for _edge_ in range(graph.ecount()):

        if graph.es[_edge_]['Type'] == 0 or graph.es[_edge_]['Type'] == 3:

            print('Edge ', _edge_, ' out of ', graph.ecount())
            source_node = graph.es[_edge_].source
            target_node = graph.es[_edge_].target

            x_source = graph.vs[source_node]['x_coordinate']
            y_source = graph.vs[source_node]['y_coordinate']
            z_source = graph.vs[source_node]['z_coordinate']
            p_source = [x_source, y_source, z_source]

            x_target = graph.vs[target_node]['x_coordinate']
            y_target = graph.vs[target_node]['y_coordinate']
            z_target = graph.vs[target_node]['z_coordinate']
            p_target = [x_target, y_target, z_target]

            points_in_between = find_points_on_line(p_source, p_target, 5)

            for cube in range(len(cube_coordinates)):

                x_lower = cube_coordinates[cube][0][0]
                x_upper = cube_coordinates[cube][0][1]
                y_lower = cube_coordinates[cube][1][0]
                y_upper = cube_coordinates[cube][1][1]
                z_lower = cube_coordinates[cube][2][0]
                z_upper = cube_coordinates[cube][2][1]

                for point in points_in_between:

                    x_point = point[0]
                    y_point = point[1]
                    z_point = point[2]

                    if x_lower < x_point < x_upper:

                        if y_lower < y_point < y_upper:

                            if z_lower < z_point < z_upper:

                                x = np.absolute(df_start['tav_Fplasma'][2 * _edge_]/df_final['tav_Fplasma'][2 * _edge_])

                                if x < 5:

                                    list_so_far_ids = cube_dictionary_edge_ids[str(cube)]
                                    list_so_far_ids.append(_edge_)
                                    cube_dictionary_edge_ids[str(cube)] = list_so_far_ids

                                    flows_so_far = cube_dictionary_flow_rates[str(cube)]
                                    flows_so_far.append(np.absolute(df_start['tav_Fplasma'][2 * _edge_]/df_final['tav_Fplasma'][2 * _edge_]))
                                    print(np.absolute(df_start['tav_Fplasma'][2 * _edge_]/df_final['tav_Fplasma'][2 * _edge_]))
                                    cube_dictionary_flow_rates[str(cube)] = flows_so_far

                                    length_so_far = cube_dictionary_length[str(cube)]
                                    length_so_far.append(graph.es[_edge_]['edge_length'])
                                    cube_dictionary_length[str(cube)] = length_so_far

                                    break

    for cube in range(len(cube_coordinates)):

        length_cube = np.array(cube_dictionary_length[str(cube)])
        flows_cube = np.array(cube_dictionary_flow_rates[str(cube)])

        length_total = np.sum(length_cube)

        flow_in_cube = np.sum((length_cube*flows_cube))/length_total

        Final_Flows_Output.append(flow_in_cube)
        Final_Edge_Eids_Output.append(cube_dictionary_edge_ids[str(cube)])

    data_package = {'Coordinates': cube_coordinates, 'Eids': Final_Edge_Eids_Output, 'Flow_Rates': Final_Flows_Output,
                    'Position': cube_position, 'cubes_y': number_of_cubes_y, 'cubes_x': number_of_cubes_x,
                    'cubes_z': number_of_cubes_z}

    # print('CUBE:')
    # print(cube_coordinates)
    #
    # print('EIDS: ')
    # print(Final_Edge_Eids_Output)
    #
    # print('FLOWS: ')
    # print(Final_Flows_Output)
    #
    # print('POSITION: ')
    # print(cube_position)

    return data_package


def flow_rate_cube_diameters(graph, cube_side_length, meshdata_file, meshdata_final):

    Final_Flows_Output = []
    Final_Edge_Eids_Output = []

    df_start = pd.read_csv(meshdata_file)
    df_final = pd.read_csv(meshdata_final)

    # x segmentation

    x_min = min(graph.vs['x_coordinate'])
    x_max = max(graph.vs['x_coordinate'])

    delta_x = x_max - x_min
    number_of_cubes_x = math.ceil(delta_x / cube_side_length)

    regions_x = []
    labels_x = []

    for k in range(number_of_cubes_x):

        start = x_min + k * cube_side_length
        end = start + cube_side_length

        labels_x.append(np.round((start+end)/2, 2))
        regions_x.append([start, end])

    # y segmentation

    y_min = min(graph.vs['y_coordinate'])
    y_max = max(graph.vs['y_coordinate'])

    delta_y = y_max - y_min

    number_of_cubes_y = math.ceil(delta_y / cube_side_length)

    regions_y = []
    labels_y = []

    for k in range(number_of_cubes_y):

        start = y_min + k * cube_side_length
        end = start + cube_side_length

        labels_y.append(np.round((start+end)/2, 2))
        regions_y.append([start, end])

    # z segmentation

    z_min = min(graph.vs['z_coordinate'])
    z_max = max(graph.vs['z_coordinate'])

    delta_z = z_max - z_min

    number_of_cubes_z = math.ceil(delta_z / cube_side_length)

    regions_z = []
    labels_z = []

    for k in range(number_of_cubes_z):

        start = z_min + k * cube_side_length
        end = start + cube_side_length

        labels_z.append(np.round((start+end)/2, 2))
        regions_z.append([start, end])

    # Initialize Dictionary for cubes

    cube_coordinates = []
    cube_position = []

    for x in range(number_of_cubes_x):

        for y in range(number_of_cubes_y):

            for z in range(number_of_cubes_z):

                # print(x, y, z)
                cube_coordinates.append([regions_x[x], regions_y[y], regions_z[z]])
                cube_position.append([x, y, z])


    cube_dictionary_edge_ids = {}
    cube_dictionary_flow_rates = {}
    cube_dictionary_length = {}

    for k in range(len(cube_coordinates)):

        cube_dictionary_edge_ids[str(k)] = []
        cube_dictionary_flow_rates[str(k)] = []
        cube_dictionary_length[str(k)] = []

    for _edge_ in range(graph.ecount()):

        if graph.es[_edge_]['Type'] == 0 or graph.es[_edge_]['Type'] == 3:

            print('Edge ', _edge_, ' out of ', graph.ecount())
            source_node = graph.es[_edge_].source
            target_node = graph.es[_edge_].target

            x_source = graph.vs[source_node]['x_coordinate']
            y_source = graph.vs[source_node]['y_coordinate']
            z_source = graph.vs[source_node]['z_coordinate']
            p_source = [x_source, y_source, z_source]

            x_target = graph.vs[target_node]['x_coordinate']
            y_target = graph.vs[target_node]['y_coordinate']
            z_target = graph.vs[target_node]['z_coordinate']
            p_target = [x_target, y_target, z_target]

            points_in_between = find_points_on_line(p_source, p_target, 5)

            for cube in range(len(cube_coordinates)):

                x_lower = cube_coordinates[cube][0][0]
                x_upper = cube_coordinates[cube][0][1]
                y_lower = cube_coordinates[cube][1][0]
                y_upper = cube_coordinates[cube][1][1]
                z_lower = cube_coordinates[cube][2][0]
                z_upper = cube_coordinates[cube][2][1]

                for point in points_in_between:

                    x_point = point[0]
                    y_point = point[1]
                    z_point = point[2]

                    if x_lower < x_point < x_upper:

                        if y_lower < y_point < y_upper:

                            if z_lower < z_point < z_upper:

                                x = np.absolute(df_start['D'][2 * _edge_]/df_final['D'][2 * _edge_])

                                if x < 1.5:

                                    list_so_far_ids = cube_dictionary_edge_ids[str(cube)]
                                    list_so_far_ids.append(_edge_)
                                    cube_dictionary_edge_ids[str(cube)] = list_so_far_ids

                                    flows_so_far = cube_dictionary_flow_rates[str(cube)]
                                    flows_so_far.append(np.absolute(df_start['D'][2 * _edge_]/df_final['D'][2 * _edge_]))
                                    cube_dictionary_flow_rates[str(cube)] = flows_so_far

                                    length_so_far = cube_dictionary_length[str(cube)]
                                    length_so_far.append(graph.es[_edge_]['edge_length'])
                                    cube_dictionary_length[str(cube)] = length_so_far

                                    break

    for cube in range(len(cube_coordinates)):

        length_cube = np.array(cube_dictionary_length[str(cube)])
        flows_cube = np.array(cube_dictionary_flow_rates[str(cube)])

        length_total = np.sum(length_cube)

        flow_in_cube = np.sum((length_cube*flows_cube))/length_total

        Final_Flows_Output.append(flow_in_cube)
        Final_Edge_Eids_Output.append(cube_dictionary_edge_ids[str(cube)])

    data_package = {'Coordinates': cube_coordinates, 'Eids': Final_Edge_Eids_Output, 'Flow_Rates': Final_Flows_Output,
                    'Position': cube_position, 'cubes_y': number_of_cubes_y, 'cubes_x': number_of_cubes_x,
                    'cubes_z': number_of_cubes_z}

    # print('CUBE:')
    # print(cube_coordinates)
    #
    # print('EIDS: ')
    # print(Final_Edge_Eids_Output)
    #
    # print('FLOWS: ')
    # print(Final_Flows_Output)
    #
    # print('POSITION: ')
    # print(cube_position)

    return data_package


def flow_versus_depth_comparison(network_ids, segments, path_eval):

    data_sets = []

    for j in network_ids:

        path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
                + str(j) + '\\graph.pkl'

        graph = ig.Graph.Read_Pickle(path)

        meshdata_file = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem\\' + str(j) + '_Baseflow\\out\\meshdata_249.csv'

        df_start = pd.read_csv(meshdata_file)

        z_min = min(graph.vs['z_coordinate'])
        z_max = max(graph.vs['z_coordinate'])

        delta_z = z_max - z_min

        segment_length = delta_z / segments

        regions = []
        labels_x = []
        averaged_flow_rates_per_region = []

        for i in range(segments):

            start = z_min + i * segment_length
            end = start + segment_length

            labels_x.append(np.round((start+end)/2, 2))
            regions.append([start, end])

        for region in range(len(regions)):

            reg = regions[region]
            z_lower = reg[0]
            z_upper = reg[1]

            flows_in_region = []
            eids_in_region = []
            corresponding_length = []

            for i in range(graph.ecount()):

                if graph.es[i]['Type'] == 0 or graph.es[i]['Type'] == 3:

                    source_node = graph.es[i].source
                    target_node = graph.es[i].target

                    z_coord_source_node = graph.vs[source_node]['z_coordinate']
                    z_coord_target_node = graph.vs[target_node]['z_coordinate']

                    if z_lower < z_coord_source_node < z_upper:

                        flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                        eids_in_region.append(i)
                        corresponding_length.append(graph.es[i]['edge_length'])
                        graph.es[i]['RegionID'] = region

                    elif z_lower < z_coord_target_node < z_upper:

                        flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                        eids_in_region.append(i)
                        corresponding_length.append(graph.es[i]['edge_length'])
                        graph.es[i]['RegionID'] = region

                    elif z_coord_source_node > z_upper and z_coord_target_node < z_lower:

                        flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                        eids_in_region.append(i)
                        corresponding_length.append(graph.es[i]['edge_length'])
                        graph.es[i]['RegionID'] = region

                    elif z_coord_source_node < z_lower and z_coord_target_node > z_upper:

                        flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                        eids_in_region.append(i)
                        corresponding_length.append(graph.es[i]['edge_length'])
                        graph.es[i]['RegionID'] = region

                else:

                    None

            flows_as_array = np.array(flows_in_region)
            length_as_array = np.array(corresponding_length)
            total_length = np.sum(corresponding_length)

            averaged_flow_in_region = np.sum((flows_as_array * length_as_array)) / total_length

            averaged_flow_rates_per_region.append(averaged_flow_in_region)

        data_sets.append([labels_x, averaged_flow_rates_per_region])

    for i in range(len(data_sets)):

        plt.plot(data_sets[i][0], data_sets[i][1], label='Network ID: ' + str(network_ids[i]))

    plt.ylabel('(Length) Averaged Flow Rates')
    plt.xlabel('Depth [$\mu$m]')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.legend()
    plt.grid(True)
    plt.savefig(path_eval + '\\Depth_Vs_Flow_Comparison.png')

    return None


def find_points_on_line(p_1, p_2, number_of_points):

    point_1 = np.array(p_1)
    point_2 = np.array(p_2)

    points_all = []

    vector = point_2 - point_1
    delta_vector = vector/number_of_points

    for k in range(number_of_points+1):

        point_new = point_1 + delta_vector * k
        points_all.append(point_new)

    return points_all


def define_activated_region(graph, coords_sphere, r_sphere):

    edges_in_current_region = []

    for edge in range(graph.ecount()):

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                edges_in_current_region.append(edge)

            elif u_2 < 0 and u_1 > 1:

                edges_in_current_region.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                edges_in_current_region.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                edges_in_current_region.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                edges_in_current_region.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                edges_in_current_region.append(edge)

            else:

                continue

        else:

            continue

    return edges_in_current_region


def plot_distance():

    path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint'

    path_0_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
    path_3_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\3\graph.pkl'
    path_4_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\4\graph.pkl'
    path_6_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\6\graph.pkl'

    coord = {'x': 200, 'y': 400, 'z': 400}

    radius_min = 50.0
    radius_max = 400.0
    r_steps = 50
    delta_radius = radius_max - radius_min
    step_radius = delta_radius / r_steps

    radii = np.arange(radius_min, radius_max, step_radius)

    # print(radii)
    # print(len(radii))
    # for radius in radii:
    #     print(radius)
    #     current_selection = define_activated_region(graph_, coord, radius)
    #     print(len(current_selection))
    #     print(current_selection)

    networks = ['0_Out_', '3_Out_', '4_Out_', '6_Out_']
    # networks = ['0_Out_']
    histogram_data_id = ['R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300', 'All', 'All_Arteries', 'All_Cap']

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

        for r in histogram_data_id:

            print(r)

            case_id = r
            # current_input_path = path + '\\' + network + '\\' + r + '\\' + 'in\\adjointMethod' + '\\'
            current_output_path = path + '\\' + network + '\\' + r + '\\' + 'out' + '\\'
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

            diameter_change_capillaries = []
            diameter_change_veins = []
            diameter_change_arterioles = []
            diameter_change_all = []

            for radius in radii:

                # print(radius)
                current_selection = define_activated_region(graph_, coord, radius)

                current_selection_arterioles = []
                current_selection_venules = []
                current_selection_capillaries = []

                for edge in current_selection:

                    if graph_.es[edge]['Type'] == 0 or graph_.es[edge]['Type'] == 3:

                        current_selection_capillaries.append(edge)

                    elif graph_.es[edge]['Type'] == 1:

                        current_selection_venules.append(edge)

                    elif graph_.es[edge]['Type'] == 2:

                        current_selection_arterioles.append(edge)

                diameter_change_for_selection = t_new_array[current_selection]
                diameter_change_for_selection_cap = t_new_array[current_selection_capillaries]
                diameter_change_for_selection_art = t_new_array[current_selection_arterioles]
                diameter_change_for_selection_ven = t_new_array[current_selection_venules]

                diameter_change_for_selection_percent = np.abs(diameter_change_for_selection - 1)*100
                diameter_change_for_selection_percent_cap = np.abs(diameter_change_for_selection_cap - 1) * 100
                diameter_change_for_selection_percent_art = np.abs(diameter_change_for_selection_art - 1) * 100
                diameter_change_for_selection_percent_ven = np.abs(diameter_change_for_selection_ven - 1) * 100

                diameter_change_all.append(np.mean(diameter_change_for_selection_percent))
                diameter_change_capillaries.append(np.mean(diameter_change_for_selection_percent_cap))
                diameter_change_veins.append(np.mean(diameter_change_for_selection_percent_ven))
                diameter_change_arterioles.append(np.mean(diameter_change_for_selection_percent_art))

            plt.plot(radii, diameter_change_all, label='All Vessel Types')
            plt.plot(radii, diameter_change_capillaries, label='Capillaries')
            plt.plot(radii, diameter_change_arterioles, label='Arterioles')
            plt.plot(radii, diameter_change_veins, label='Venules')

            plt.ylabel('Relative Diameter Change in %')
            plt.xlabel('Distance from Activation Center')
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.legend()

            plt.gca().yaxis.grid(True)
            # print(r + '\\' + 'D_Change_vs_Radius.png')
            plt.savefig(path + '\\' + network + '\\' + case_id + '\\' + 'D_Change_vs_Radius.png')
            plt.clf()

            # print(diameter_change_all)
            # print(diameter_change_arterioles)
            # print(diameter_change_veins)
            # print(diameter_change_capillaries)

    return None


# INPUT ----------------------------------------------------------------------------------------------------------------

ids = ['All', 'All_Arteries', 'All_Cap', 'R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300']
# ids = ['All', 'All_Cap', 'R_100', 'R_125', 'R_150', 'R_175', 'R_200', 'R_300']
# ids = ['All']

path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\0_out_'
path_graph = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Networks\0\graph.pkl'
start_file = '\out\meshdata_9999.csv'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:

        if file[0:8] == 'meshdata':

            files.append(file)

files.remove('meshdata_9999.csv')
end_file = '\out\\' + files[0]
print(end_file)

for idx in ids:

    print(idx)

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path + '\\' + idx):
        for file in f:

            if file[0:8] == 'meshdata':
                files.append(file)

    files.remove('meshdata_9999.csv')
    end_file = '\out\\' + files[0]
    print(end_file)

    simulation_path = path + '\\' + idx

    corresponding_network = simulation_path + r'\in\adjointMethod'
    _path_adjoint_file = simulation_path + r'\out\adjointdata.csv'
    _start_file = simulation_path + start_file
    _end_file_ = simulation_path + end_file

    path_x = corresponding_network + r'\activated_eids.csv'

    r = csv.reader(open(path_x))
    lines = list(r)
    activated_eids = lines[0]
    activated_eids = list(map(int, activated_eids))

    # adjoint_data_plot(_path_adjoint_file, simulation_path)
    #
    # compute_flow_change(_start_file, _end_file_, activated_eids, simulation_path)
    # plot_3_d_plasma(_start_file, _end_file_, simulation_path)
    # plot_3_d_diameter(_start_file, _end_file_, simulation_path)

    # HISTOGRAM

    if idx == '':

         continue

    else:

        plot_diameter_change_per_total_volume(path_graph, _start_file, _end_file_, simulation_path)
        plot_diameter_change_per_volume_of_vessel_type(path_graph, _start_file, _end_file_, simulation_path)


# Diameter Change versus Distance from Activation Center

# plot_distance()

# 3D Plot Special - Flows  ---------------------------------------------------------------------------------------------

# for i in range(1):
#
#     print(i)
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#             + str(i) + '\\graph.pkl'
#
#     path1 = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Sim_1\All\out\meshdata_249.csv'
#
#     path2 = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Adjoint\Sim_1\All\out\meshdata_39999.csv'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#     cube_info = flow_rate_cube(graph_, 150, path1, path2)
#
#     cube_position = cube_info['Position']
#     cube_Coordinates = cube_info['Coordinates']
#     cube_flows = cube_info['Flow_Rates']
#     max_cube_flow = max(cube_flows)
#
#     for i in range(len(cube_flows)):
#
#         print(cube_flows[i])
#
#     z_list = [0, 1, 2, 3]              # max 4
#
#     data_to_plot_tot = []
#     x_coord_list_tot = []
#     y_coord_list_tot = []
#     z_coord_list_tot = []
#
#     for z in z_list:
#
#         data_to_plot = []
#         data_to_plot_position = []
#         coordinates = []
#
#         for y in range(cube_info['cubes_y']-1, -1, -1):
#
#             y_line_cubes = []
#
#             for x in range(cube_info['cubes_x']):
#
#                 y_line_cubes.append([x, y, z])
#
#             data_to_plot_position.append(y_line_cubes)
#
#         for j in data_to_plot_position:
#
#             current_y_line = j
#             line_to_plot = []
#
#             for cube in current_y_line:
#                 index = cube_position.index(cube)
#                 line_to_plot.append(cube_flows[index])
#                 coordinates.append(cube_Coordinates[index])
#
#             data_to_plot.append(line_to_plot)
#
#         x_coordinates = []
#         y_coordinates = []
#         z_coordinates = []
#
#         for coord in coordinates:
#
#             x_coordinates.append(coord[0])
#             y_coordinates.append(coord[1])
#             z_coordinates.append(coord[2])
#
#         x_coord_list = []
#         y_coord_list = []
#         z_coord_list = []
#
#         for x in x_coordinates:
#
#             x_coord_list.append(x[0])
#             x_coord_list.append(x[1])
#
#         for y in y_coordinates:
#             y_coord_list.append(y[0])
#             y_coord_list.append(y[1])
#
#         for z in z_coordinates:
#             z_coord_list.append(z[0])
#             z_coord_list.append(z[1])
#
#         x_coord_list = list(dict.fromkeys(x_coord_list))
#         y_coord_list = list(dict.fromkeys(y_coord_list))
#         z_coord_list = list(dict.fromkeys(z_coord_list))
#
#         x_coord_list_tot.append(x_coord_list)
#         y_coord_list_tot.append(y_coord_list)
#         z_coord_list_tot.append(z_coord_list)
#
#         data_to_plot_tot.append(data_to_plot)
#
#     print(data_to_plot_tot)
#
#     plot_3d_cube(data_to_plot_tot, max_cube_flow, x_coord_list_tot, y_coord_list_tot, z_coord_list_tot, graph_)





































