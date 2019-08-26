import igraph as ig
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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


def volume_fractions(graph):

    x_min = min(graph.vs['x_coordinate'])
    x_max = max(graph.vs['x_coordinate'])
    y_min = min(graph.vs['y_coordinate'])
    y_max = max(graph.vs['y_coordinate'])
    z_min = min(graph.vs['y_coordinate'])
    z_max = max(graph.vs['y_coordinate'])

    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta_z = z_max - z_min

    volume_cuboid = (delta_x * delta_y * delta_z)*math.pow(10, -18)

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

    fraction_all_vessels = (tot_volume_arterial_vessels + tot_volume_venous_vessels +
                            tot_volume_capillary_bed_connections + tot_volume_capillary_bed_honeycomb)/volume_cuboid

    fraction_arteries = tot_volume_arterial_vessels / volume_cuboid

    fraction_venous = tot_volume_venous_vessels / volume_cuboid

    fraction_capillaries = (tot_volume_capillary_bed_honeycomb + tot_volume_capillary_bed_connections) / volume_cuboid

    fraction_dict = {'All': fraction_all_vessels, 'Veins': fraction_venous, 'Arteries': fraction_arteries, 'Capillaries': fraction_capillaries}

    return fraction_dict


def volume_fractions_plot(frac_cap, frac_v, frac_a, frac_all, path):

    fig, axarr = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Volume Fractions", fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    axarr[0, 0].hist(frac_all, 8, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[0, 0].axvline(np.mean(frac_all), color='k', linestyle='dashed', linewidth=1)
    axarr[0, 0].set_title('Volume Fraction All')
    mu_all = np.mean(frac_all)
    sigma_all = np.std(frac_all)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_all,),
        r'$\sigma=%.5f$' % (sigma_all,)))

    axarr[0, 0].text(0.05, 0.95, textstr, transform=axarr[0, 0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    axarr[0, 1].hist(frac_cap, 8, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[0, 1].set_title('Volume Fraction Capillaries')
    axarr[0, 1].axvline(np.mean(frac_cap), color='k', linestyle='dashed', linewidth=1)
    mu_cap = np.mean(frac_cap)
    sigma_cap = np.std(frac_cap)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_cap,),
        r'$\sigma=%.5f$' % (sigma_cap,)))

    axarr[0, 1].text(0.05, 0.95, textstr, transform=axarr[0, 1].transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

    axarr[1, 0].hist(frac_a, 8, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[1, 0].set_title('Volume Fraction Arteries')
    axarr[1, 0].axvline(np.mean(frac_a), color='k', linestyle='dashed', linewidth=1)
    mu_a = np.mean(frac_a)
    sigma_a = np.std(frac_a)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_a,),
        r'$\sigma=%.5f$' % (sigma_a,)))

    axarr[1, 0].text(0.75, 0.95, textstr, transform=axarr[1, 0].transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

    axarr[1, 1].hist(frac_v, 8, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[1, 1].set_title('Volume Fraction Veins')
    axarr[1, 1].axvline(np.mean(frac_v), color='k', linestyle='dashed', linewidth=1)
    mu_v = np.mean(frac_v)
    sigma_v = np.std(frac_v)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_v,),
        r'$\sigma=%.5f$' % (sigma_v,)))

    axarr[1, 1].text(0.75, 0.95, textstr, transform=axarr[1, 1].transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig(path + '\\Volume_Fractions_Histogram.png')


def topology_characteristic_plot(path_a, path_v, path_cap, graph, network_id):

    # Sort

    edges_v = []
    edges_a = []
    edges_cap = []

    for edge in range(graph.ecount()):

        if graph.es[edge]['Type'] == 1:

            edges_v.append(edge)

        elif graph.es[edge]['Type'] == 2:

            edges_a.append(edge)

        else:

            edges_cap.append(edge)

    # Veins

    diameter_v = []
    length_v = []

    for edge in edges_v:

        diameter_v.append(graph.es[edge]['diameter']*math.pow(10, 6))
        length_v.append(graph.es[edge]['edge_length']*math.pow(10, 6))

    fig, axarr = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle("Veins Characteristics", fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    axarr[0].hist(diameter_v, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[0].axvline(np.mean(diameter_v), color='k', linestyle='dashed', linewidth=1)
    axarr[0].set_title('Diameter Distribution')
    mu_d_v = np.mean(diameter_v)
    sigma_d_v = np.std(diameter_v)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_d_v,),
        r'$\sigma=%.5f$' % (sigma_d_v,)))

    axarr[0].text(0.6, 0.95, textstr, transform=axarr[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    axarr[1].hist(length_v, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[1].set_title('Length Distribution')
    axarr[1].axvline(np.mean(length_v), color='k', linestyle='dashed', linewidth=1)
    mu_l_v = np.mean(length_v)
    sigma_l_v = np.std(length_v)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_l_v,),
        r'$\sigma=%.5f$' % (sigma_l_v,)))

    axarr[1].text(0.6, 0.95, textstr, transform=axarr[1].transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig(path_v + '\\Characteristics_Veins_Network_' + str(network_id) + '.png')
    plt.clf()
    plt.close(fig)

    # Arteries

    diameter_a = []
    length_a = []

    for edge in edges_a:
        diameter_a.append(graph.es[edge]['diameter'] * math.pow(10, 6))
        length_a.append(graph.es[edge]['edge_length'] * math.pow(10, 6))

    fig, axarr = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle("Arteries Characteristics", fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    axarr[0].hist(diameter_a, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[0].axvline(np.mean(diameter_a), color='k', linestyle='dashed', linewidth=1)
    axarr[0].set_title('Diameter Distribution')
    mu_d_a = np.mean(diameter_a)
    sigma_d_a = np.std(diameter_a)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_d_a,),
        r'$\sigma=%.5f$' % (sigma_d_a,)))

    axarr[0].text(0.6, 0.95, textstr, transform=axarr[0].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    axarr[1].hist(length_a, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[1].set_title('Length Distribution')
    axarr[1].axvline(np.mean(length_a), color='k', linestyle='dashed', linewidth=1)
    mu_l_a = np.mean(length_a)
    sigma_l_a = np.std(length_a)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_l_a,),
        r'$\sigma=%.5f$' % (sigma_l_a,)))

    axarr[1].text(0.6, 0.95, textstr, transform=axarr[1].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig(path_a + '\\Characteristics_Arteries_Network_' + str(network_id) + '.png')
    plt.clf()
    plt.close(fig)

    # Caps

    diameter_c = []
    length_c = []

    for edge in edges_cap:
        diameter_c.append(graph.es[edge]['diameter'] * math.pow(10, 6))
        length_c.append(graph.es[edge]['edge_length'] * math.pow(10, 6))

    fig, axarr = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle("Capillaries Characteristics", fontsize=16)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    axarr[0].hist(diameter_c, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[0].axvline(np.mean(diameter_c), color='k', linestyle='dashed', linewidth=1)
    axarr[0].set_title('Diameter Distribution')
    mu_d_c = np.mean(diameter_c)
    sigma_d_c = np.std(diameter_c)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_d_c,),
        r'$\sigma=%.5f$' % (sigma_d_c,)))

    axarr[0].text(0.6, 0.95, textstr, transform=axarr[0].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    axarr[1].hist(length_c, 20, density=False, facecolor='blue', alpha=0.3, ec="k", histtype='stepfilled')
    axarr[1].set_title('Length Distribution')
    axarr[1].axvline(np.mean(length_c), color='k', linestyle='dashed', linewidth=1)
    mu_l_c = np.mean(length_c)
    sigma_l_c = np.std(length_c)

    textstr = '\n'.join((
        r'$\mu=%.5f$' % (mu_l_c,),
        r'$\sigma=%.5f$' % (sigma_l_c,)))

    axarr[1].text(0.6, 0.95, textstr, transform=axarr[1].transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig(path_cap + '\\Characteristics_Capillaries_Network_' + str(network_id) + '.png')
    plt.clf()
    plt.close(fig)

    return


def plot_tree_with_regions(graph, region_id):

    # Graph need to have coordinates of each node as vertex attribute

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for edge in range(graph.ecount()):

        x = []
        y = []
        z = []

        x.append(graph.vs[graph.es[edge].source]['x_coordinate'])
        x.append(graph.vs[graph.es[edge].target]['x_coordinate'])

        y.append(graph.vs[graph.es[edge].source]['y_coordinate'])
        y.append(graph.vs[graph.es[edge].target]['y_coordinate'])

        z.append(graph.vs[graph.es[edge].source]['z_coordinate'])
        z.append(graph.vs[graph.es[edge].target]['z_coordinate'])

        if graph.es[edge]['RegionID'] == region_id:

            ax.plot(x, y, z, color='green')

        else:

            ax.plot(x, y, z, color='lightgray')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


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
        corresponding_flows = []
        corresponding_eids = []
        averaged_flow_rates_per_region = []

        for i in range(segments):

            start = z_min + i * segment_length
            end = start + segment_length

            labels_x.append(np.round((start+end)/(2*delta_z), 2))
            regions.append([start, end])

        for region in range(len(regions)):

            reg = regions[region]
            z_lower = reg[0]
            z_upper = reg[1]

            flows_in_region = []
            eids_in_region = []

            for i in range(graph.ecount()):

                source_node = graph.es[i].source
                target_node = graph.es[i].target

                z_coord_source_node = graph.vs[source_node]['z_coordinate']
                z_coord_target_node = graph.vs[target_node]['z_coordinate']

                if z_lower < z_coord_source_node < z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    graph.es[i]['RegionID'] = region

                elif z_lower < z_coord_target_node < z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    graph.es[i]['RegionID'] = region

                elif z_coord_source_node > z_upper and z_coord_target_node < z_lower:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    graph.es[i]['RegionID'] = region

                elif z_coord_source_node < z_lower and z_coord_target_node > z_upper:

                    flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                    eids_in_region.append(i)
                    graph.es[i]['RegionID'] = region

            corresponding_flows.append(flows_in_region)
            corresponding_eids.append(eids_in_region)

            averaged_flow_rates_per_region.append(np.average(flows_in_region))

        data_sets.append([labels_x, averaged_flow_rates_per_region])

    for i in range(len(data_sets)):

        plt.plot(data_sets[i][0], data_sets[i][1], label='Network ID: ' + str(network_ids[i]))

    plt.ylabel('Averaged Flow Rates')
    plt.xlabel('Depth / Total Depth')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.legend()
    plt.grid(True)
    plt.savefig(path_eval + '\\Depth_Vs_Flow_Comparison.png')

    return None


def flow_versus_depth(graph, segments, meshdata_file, path_eval, network):

    df_start = pd.read_csv(meshdata_file)

    z_min = min(graph.vs['z_coordinate'])
    z_max = max(graph.vs['z_coordinate'])

    delta_z = z_max - z_min

    segment_length = delta_z / segments

    regions = []
    labels_x = []
    corresponding_flows = []
    corresponding_eids = []
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

        for i in range(graph.ecount()):

            source_node = graph.es[i].source
            target_node = graph.es[i].target

            z_coord_source_node = graph.vs[source_node]['z_coordinate']
            z_coord_target_node = graph.vs[target_node]['z_coordinate']

            if z_lower < z_coord_source_node < z_upper:

                flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                eids_in_region.append(i)
                graph.es[i]['RegionID'] = region

            elif z_lower < z_coord_target_node < z_upper:

                flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                eids_in_region.append(i)
                graph.es[i]['RegionID'] = region

            elif z_coord_source_node > z_upper and z_coord_target_node < z_lower:

                flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                eids_in_region.append(i)
                graph.es[i]['RegionID'] = region

            elif z_coord_source_node < z_lower and z_coord_target_node > z_upper:

                flows_in_region.append(np.absolute(df_start['tav_Fplasma'][2 * i]))
                eids_in_region.append(i)
                graph.es[i]['RegionID'] = region

        corresponding_flows.append(flows_in_region)
        corresponding_eids.append(eids_in_region)

        averaged_flow_rates_per_region.append(np.average(flows_in_region))
        # plot_tree_with_regions(graph, region)

    plt.plot(labels_x, averaged_flow_rates_per_region)
    plt.xticks(labels_x, labels_x, rotation=45,  size=7)
    plt.ylabel('Averaged Flow Rates')
    plt.xlabel('Depth in $\mu$m')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig(path_eval + '\\Flow_Versus_Depth_Network_' + str(network) + '.png')
    plt.clf()

    return None


def flow_rate_cube(graph, cube_side_length, meshdata_file):

    df_start = pd.read_csv(meshdata_file)

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

    for x in range(number_of_cubes_x):

        for y in range(number_of_cubes_y):

            for z in range(number_of_cubes_z):

                print(x, y, z)
                cube_coordinates.append([regions_x[x], regions_y[y], regions_z[z]])

    cube_dictionary_edge_ids = {}
    cube_dictionary_flow_rates = {}

    for k in range(len(cube_coordinates)):

        cube_dictionary_edge_ids[str(k)] = []
        cube_dictionary_flow_rates[str(k)] = []

    for _edge_ in range(graph.ecount()):

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

                            list_so_far_ids = cube_dictionary_edge_ids[str(cube)]
                            list_so_far_ids.append(_edge_)
                            cube_dictionary_edge_ids[str(cube)] = list_so_far_ids

                            flows_so_far = cube_dictionary_flow_rates[str(cube)]
                            flows_so_far.append(np.absolute(df_start['tav_Fplasma'][2 * _edge_]))
                            cube_dictionary_flow_rates[str(cube)] = flows_so_far
                            break

    print(cube_coordinates)
    print(cube_dictionary_edge_ids)
    print(cube_dictionary_flow_rates)

    return


########################################################################################################################
#                                                   Flow Rate per Cube                                                 #
########################################################################################################################

for i in range(1):

    path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
            + str(i) + '\\graph.pkl'

    path1 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem\\' \
            + str(i) + '_Baseflow\\out\\meshdata_249.csv'

    path2 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem\\' \
             'Evaluation'

    graph_ = ig.Graph.Read_Pickle(path)
    flow_rate_cube(graph_, 200, path1)

########################################################################################################################
#                                                Topology Characteristics                                              #
########################################################################################################################

# for i in range(20):
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#            + str(i) + '\\graph.pkl'
#
#     target_path_arteries = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small' \
#                            r'\Topologie\Arteries_Characteristics'
#
#     target_path_veins = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Topologie' \
#                         r'\Veins_Charachteristics'
#
#     target_path_caps = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Topologie' \
#                        r'\Capillaries_Characteristics'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#
#     topology_characteristic_plot(target_path_arteries, target_path_veins, target_path_caps, graph_, i)

########################################################################################################################
#                                                    Volume Fractions                                                  #
########################################################################################################################

# target_path_topo = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\02_Network_Study_Small\Topologie'
# fraction_all = []
# fraction_veins = []
# fraction_arteries = []
# fraction_capillaries = []
#
# for i in range(20):
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#            + str(i) + '\\graph.pkl'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#     fractions = volume_fractions(graph_)
#
#     fraction_all.append(fractions['All'])
#     fraction_veins.append(fractions['Veins'])
#     fraction_arteries.append(fractions['Arteries'])
#     fraction_capillaries.append(fractions['Capillaries'])
#
# print(fraction_capillaries)
# print(fraction_arteries)
# print(fraction_veins)
# print(fraction_all)
#
# volume_fractions_plot(fraction_capillaries, fraction_veins, fraction_arteries, fraction_all, target_path_topo)

########################################################################################################################
#                                               Depth versus Flow Rate                                                 #
########################################################################################################################

# for i in range(20):
#
#     path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Networks\\' \
#            + str(i) + '\\graph.pkl'
#
#     path1 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem\\' \
#            + str(i) + '_Baseflow\\out\\meshdata_249.csv'
#
#     path2 = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem\\' \
#             'Evaluation'
#
#     graph_ = ig.Graph.Read_Pickle(path)
#     flow_versus_depth(graph_, 20, path1, path2, i)

########################################################################################################################
#                                          Network Comparison (Depth vs. Flow)                                         #
########################################################################################################################

# target_path = 'D:\\00 Privat\\01_Bildung\\01_ETH Zürich\MSc\\00_Masterarbeit\\02_Network_Study_Small\\Flow_Problem' \
#         '\\Evaluation'
#
# number_of_segments = 15
# network_ids = [1, 5, 9, 12, 19]
#
# flow_versus_depth_comparison(network_ids, number_of_segments, target_path)
