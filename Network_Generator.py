import math
import Beta_Distribution
import Chose_activated_region


def create_3d_graph(graph, x, y, z, l_honeycomb, coord, diameter_info):

    for level in range(z):

        graph = create_plane(graph, x, y, level, l_honeycomb, coord)

    for edge in range(graph.ecount()):

        value = Beta_Distribution.get_value_from_beta_distribution(diameter_info['mean'], diameter_info['std'])

        if value > diameter_info['max']:

            graph.es[edge]['diameter'] = diameter_info['max']

        elif value < diameter_info['min']:

            graph.es[edge]['diameter'] = diameter_info['min']

        else:

            graph.es[edge]['diameter'] = value

    return graph


def create_plane_geometrical_help(graph, x, y, z, l_honeycomb, coord_lim):

    horizontal_lines = 4 * y + 1
    vertices_per_line = x + 1
    total_vertices = horizontal_lines * vertices_per_line
    graph.add_vertices(total_vertices)

    vector_ = {'x_displacement': l_honeycomb * 0.5 , 'y_displacement': l_honeycomb * 0.2}

    ''' Add Edges '''

    start_id = graph.vcount() - total_vertices
    end_vertices = list(range(graph.vcount() - vertices_per_line, graph.vcount()))

    if z == 0:

        graph.vs[0]['x_coordinate'] = coord_lim['x_min'] + vector_['x_displacement']
        graph.vs[0]['y_coordinate'] = coord_lim['y_min'] + vector_['y_displacement']
        graph.vs[0]['z_coordinate'] = coord_lim['z_min']

    else:

        None

    for line in range(horizontal_lines):

        vertices_in_current_line = list(range(start_id, start_id + vertices_per_line))

        for i in range(len(vertices_in_current_line)):

            # Create Coordinates

            graph.vs[vertices_in_current_line[i]]['z_coordinate'] = graph.vs[0]['z_coordinate'] + z * l_honeycomb*0.5

            # Add Vertex Attribute -> Vein_Point = 1 -> venous penetrating tree starts here!

            if (line % 2) == 0:

                graph.vs[vertices_in_current_line[i]]['vein_point'] = 1

            else:

                graph.vs[vertices_in_current_line[i]]['vein_point'] = 0

            if (line % 2) == 0:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = graph.vs[0]['y_coordinate'] + math.sin(60 * 2 * math.pi / 360) * l_honeycomb * line / 2 # Set y-coordinate of each vertex

            else:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = graph.vs[0]['y_coordinate'] + math.cos(30 * 2 * math.pi / 360) * l_honeycomb / 2 + (line-1) * math.cos(30 * 2 * math.pi / 360) * l_honeycomb /2

            if (line % 4) == 0:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5)

            elif (line % 2) != 0:

                if (vertices_in_current_line[i] % 2) == 0:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i - 0.25 * l_honeycomb


                else:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5) + 0.25 * l_honeycomb

            else:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i - 0.5 * l_honeycomb


                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5) + 0.5 * l_honeycomb

            # Create Connections

            if vertices_in_current_line[i] in end_vertices:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                    graph.es[graph.ecount()-1]['CanBeConnectedToPenetratingTree'] = 1

            elif (line % 2) != 0:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)
                graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

            else:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)
                graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

                if (line % 4) == 0:

                    if (vertices_in_current_line[i] % 2) == 0:

                        graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                        graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

                    else:

                        None

                else:

                    if (vertices_in_current_line[i] % 2) != 0:

                        if vertices_in_current_line[i] == start_id + vertices_per_line - 1:

                            None

                        else:

                            graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

                    else:

                        None

        start_id += vertices_per_line

    if z == 0:

        None

    elif (z % 2) != 0:

        nodes_slanting_to_the_left = []
        start_nodes = list(range(x + 1, 2 * (x + 1)))

        for node in start_nodes:

            if (node % 2) == 0:

                for i in range(y):
                    nodes_slanting_to_the_left.append(node + i * 4 * (x + 1))

            else:

                 for i in range(y):
                    nodes_slanting_to_the_left.append(node + i * 4 * (x + 1) + 2 * (x + 1))

        nodes_slanting_to_the_left.sort()
        nodes_slanting_to_the_left = [l + total_vertices * (z - 1) for l in nodes_slanting_to_the_left]
        nodes_slanting_to_left_upper_level = [x + total_vertices for x in nodes_slanting_to_the_left]

        for j in range(len(nodes_slanting_to_the_left)):

            graph.add_edge(nodes_slanting_to_the_left[j], nodes_slanting_to_left_upper_level[j])
            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

    else:

        nodes_slanting_to_the_right = []
        start_nodes = list(range(x + 1, 2 * (x + 1)))

        for node in start_nodes:

            if (node % 2) == 0:

                for i in range(y):
                    nodes_slanting_to_the_right.append(node + i * 4 * (x + 1) + 2 * (x + 1))

            else:

                for i in range(y):
                    nodes_slanting_to_the_right.append(node + i * 4 * (x + 1))

        nodes_slanting_to_the_right.sort()
        nodes_slanting_to_the_right = [l + total_vertices*(z-1) for l in nodes_slanting_to_the_right]
        nodes_slanting_to_the_right_upper_level = [k + total_vertices for k in nodes_slanting_to_the_right]

        for j in range(len(nodes_slanting_to_the_right)):

            graph.add_edge(nodes_slanting_to_the_right[j], nodes_slanting_to_the_right_upper_level[j])
            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

    for edge in range(graph.ecount()):

        if graph.es[edge]['CanBeConnectedToPenetratingTree'] == 1:

            x_source = graph.vs[graph.es[edge].source]['x_coordinate']
            x_target = graph.vs[graph.es[edge].target]['x_coordinate']

            x_mp = (x_target - x_source) * 0.5 + x_source
            y_mp = graph.vs[graph.es[edge].source]['y_coordinate']
            z_mp = graph.vs[graph.es[edge].source]['z_coordinate']

            graph.es[edge]['Coord_midpoint'] = (x_mp, y_mp, z_mp)

    h = math.cos(30 * 2 * math.pi / 360) * l_honeycomb
    graph.vs['artery_point'] = 0

    for column in range(x):

        if (column % 2) == 0:

            for midpoint_vertex in range(y):

                x_coordinate_midpoint = 0.5 * l_honeycomb + 1.5 * l_honeycomb * column + vector_['x_displacement']
                y_coordinate_midpoint = h + 2 * h * midpoint_vertex + vector_['y_displacement']
                z_coordinate_midpoint = 0

                graph.add_vertex(x_coordinate=x_coordinate_midpoint, y_coordinate=y_coordinate_midpoint,
                                 z_coordinate=z_coordinate_midpoint, artery_point=1)

        else:

            for midpoint_vertex in range(y-1):

                x_coordinate_midpoint = 0.5 * l_honeycomb + 1.5 * l_honeycomb * column + vector_['x_displacement']
                y_coordinate_midpoint = 2 * h + 2 * h * midpoint_vertex + vector_['y_displacement']
                z_coordinate_midpoint = 0

                graph.add_vertex(x_coordinate=x_coordinate_midpoint, y_coordinate=y_coordinate_midpoint,
                                 z_coordinate=z_coordinate_midpoint, artery_point=1)


    graph.es['PartOfCapBed'] = 1
    graph.vs['PartOfCapBed'] = 1
    graph.es['PartOfPenetratingTree'] = 0
    graph.vs['PartOfPenetratingTree'] = 0
    graph.es['connection_CB_Pene'] = 0
    graph.vs['attachmentVertex'] = 0
    graph.vs['CapBedConnection'] = 0

    return graph


def trim_geometrical_help(graph_geometrical_help, coord_lim):

    x_min = coord_lim['x_min']
    x_max = coord_lim['x_max']
    y_min = coord_lim['y_min']
    y_max = coord_lim['y_max']

    print(x_min, x_max, y_min, y_max)

    edges_to_be_deleted = []

    for edge in range(graph_geometrical_help.ecount()):

        start_point = graph_geometrical_help.es[edge].source
        end_point = graph_geometrical_help.es[edge].target

        x_p_start = graph_geometrical_help.vs[start_point]['x_coordinate']
        x_p_end = graph_geometrical_help.vs[end_point]['x_coordinate']

        y_p_start = graph_geometrical_help.vs[start_point]['y_coordinate']
        y_p_end = graph_geometrical_help.vs[end_point]['y_coordinate']

        if x_p_start < x_min or x_p_start > x_max or y_p_start > y_max or y_p_start < y_min:

            edges_to_be_deleted.append(edge)

        elif x_p_end < x_min or x_p_end > x_max or y_p_end > y_max or y_p_end < y_min:

            edges_to_be_deleted.append(edge)

    edges_to_be_deleted_new = []

    for i in range(len(edges_to_be_deleted)):

        edge_id_new = edges_to_be_deleted[i] - i
        edges_to_be_deleted_new.append(edge_id_new)

    for edge in edges_to_be_deleted_new:

        graph_geometrical_help.delete_edges([edge])

    vertices_to_be_deleted = []

    for vertex in range(graph_geometrical_help.vcount()):

        x_vertex = graph_geometrical_help.vs[vertex]['x_coordinate']
        y_vertex = graph_geometrical_help.vs[vertex]['y_coordinate']

        if x_vertex < x_min or x_vertex > x_max or y_vertex > y_max or y_vertex < y_min:

            vertices_to_be_deleted.append(vertex)

        else:

            continue

    vertices_to_be_deleted_new = []

    for i in range(len(vertices_to_be_deleted)):

        vertex_id_new = vertices_to_be_deleted[i] - i
        vertices_to_be_deleted_new.append(vertex_id_new)

    for vertex in vertices_to_be_deleted_new:

        graph_geometrical_help.delete_vertices([vertex])

    return graph_geometrical_help


def create_plane(graph, x, y, z, l_honeycomb, coord_lim):

    horizontal_lines = 4 * y + 1
    vertices_per_line = x + 1
    total_vertices = horizontal_lines * vertices_per_line
    graph.add_vertices(total_vertices)

    ''' Add Edges '''

    start_id = graph.vcount() - total_vertices
    end_vertices = list(range(graph.vcount() - vertices_per_line, graph.vcount()))

    if z == 0:

        graph.vs[0]['x_coordinate'] = coord_lim['x_min']
        graph.vs[0]['y_coordinate'] = coord_lim['y_min']
        graph.vs[0]['z_coordinate'] = coord_lim['z_min']

    else:

        None

    for line in range(horizontal_lines):

        vertices_in_current_line = list(range(start_id, start_id + vertices_per_line))

        for i in range(len(vertices_in_current_line)):

            # Create Coordinates

            graph.vs[vertices_in_current_line[i]]['z_coordinate'] = graph.vs[0]['z_coordinate'] + z * l_honeycomb*0.5

            if (line % 2) == 0:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = graph.vs[0]['y_coordinate'] + math.sin(60 * 2 * math.pi / 360) * l_honeycomb * line / 2 # Set y-coordinate of each vertex

            else:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = graph.vs[0]['y_coordinate'] + math.cos(30 * 2 * math.pi / 360) * l_honeycomb / 2 + (line-1) * math.cos(30 * 2 * math.pi / 360) * l_honeycomb /2

            if (line % 4) == 0:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5)

            elif (line % 2) != 0:

                if (vertices_in_current_line[i] % 2) == 0:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i - 0.25 * l_honeycomb


                else:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5) + 0.25 * l_honeycomb

            else:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + 1.5 * l_honeycomb * i - 0.5 * l_honeycomb


                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = graph.vs[0]['x_coordinate'] + l_honeycomb * (1.5 * i - 0.5) + 0.5 * l_honeycomb

            # Create Connections

            if vertices_in_current_line[i] in end_vertices:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                    graph.es[graph.ecount()-1]['CanBeConnectedToPenetratingTree'] = 1

            elif (line % 2) != 0:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)
                graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

            else:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)
                graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 0

                if (line % 4) == 0:

                    if (vertices_in_current_line[i] % 2) == 0:

                        graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                        graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

                    else:

                        None

                else:

                    if (vertices_in_current_line[i] % 2) != 0:

                        if vertices_in_current_line[i] == start_id + vertices_per_line - 1:

                            None

                        else:

                            graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
                            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

                    else:

                        None

        start_id += vertices_per_line

    ####################################################################################################################
    #                                                                                                                  #
    #                                    Connections between layers (z-direction)                                      #
    #                                                                                                                  #
    ####################################################################################################################

    if z == 0:

        None

    elif (z % 2) != 0:

        nodes_slanting_to_the_left = []
        start_nodes = list(range(x + 1, 2 * (x + 1)))

        for node in start_nodes:

            if (node % 2) == 0:

                for i in range(y):
                    nodes_slanting_to_the_left.append(node + i * 4 * (x + 1))

            else:

                 for i in range(y):
                    nodes_slanting_to_the_left.append(node + i * 4 * (x + 1) + 2 * (x + 1))

        nodes_slanting_to_the_left.sort()
        nodes_slanting_to_the_left = [l + total_vertices * (z - 1) for l in nodes_slanting_to_the_left]
        nodes_slanting_to_left_upper_level = [x + total_vertices for x in nodes_slanting_to_the_left]

        for j in range(len(nodes_slanting_to_the_left)):

            graph.add_edge(nodes_slanting_to_the_left[j], nodes_slanting_to_left_upper_level[j])
            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

    else:

        nodes_slanting_to_the_right = []
        start_nodes = list(range(x + 1, 2 * (x + 1)))

        for node in start_nodes:

            if (node % 2) == 0:

                for i in range(y):
                    nodes_slanting_to_the_right.append(node + i * 4 * (x + 1) + 2 * (x + 1))

            else:

                for i in range(y):
                    nodes_slanting_to_the_right.append(node + i * 4 * (x + 1))

        nodes_slanting_to_the_right.sort()
        nodes_slanting_to_the_right = [l + total_vertices*(z-1) for l in nodes_slanting_to_the_right]
        nodes_slanting_to_the_right_upper_level = [k + total_vertices for k in nodes_slanting_to_the_right]

        for j in range(len(nodes_slanting_to_the_right)):

            graph.add_edge(nodes_slanting_to_the_right[j], nodes_slanting_to_the_right_upper_level[j])
            graph.es[graph.ecount() - 1]['CanBeConnectedToPenetratingTree'] = 1

    ####################################################################################################################
    #                                                                                                                  #
    #                                           Coordinates of edge midpoints                                          #
    #                                                                                                                  #
    ####################################################################################################################

    for edge in range(graph.ecount()):

        if graph.es[edge]['CanBeConnectedToPenetratingTree'] == 1:

            x_source = graph.vs[graph.es[edge].source]['x_coordinate']
            x_target = graph.vs[graph.es[edge].target]['x_coordinate']

            x_mp = (x_target - x_source) * 0.5 + x_source
            y_mp = graph.vs[graph.es[edge].source]['y_coordinate']
            z_mp = graph.vs[graph.es[edge].source]['z_coordinate']

            graph.es[edge]['Coord_midpoint'] = (x_mp, y_mp, z_mp)

    graph.es['PartOfCapBed'] = 1
    graph.vs['PartOfCapBed'] = 1
    graph.es['PartOfPenetratingTree'] = 0
    graph.vs['PartOfPenetratingTree'] = 0
    graph.es['connection_CB_Pene'] = 0
    graph.vs['attachmentVertex'] = 0
    graph.vs['CapBedConnection'] = 0

    graph.es['edge_length'] = l_honeycomb / math.pow(10, 6)

    graph.es['Type'] = 0
    graph.vs['Type'] = 0

    return graph


def add_penetrating_tree_to_tot(graph_to_add, graph_penetrating_tot, start_point):

    # Find vectorial displacement

    index_inlet = graph_to_add.vs.find(attachmentVertex=1).index

    index_inlet_x_coordinate = graph_to_add.vs[index_inlet]['x_coordinate']
    index_inlet_y_coordinate = graph_to_add.vs[index_inlet]['y_coordinate']
    index_inlet_z_coordinate = graph_to_add.vs[index_inlet]['z_coordinate']

    delta_x = index_inlet_x_coordinate - start_point[0]
    delta_y = index_inlet_y_coordinate - start_point[1]
    delta_z = index_inlet_z_coordinate - start_point[2]

    # print(graph_to_add.vs[index_inlet])

    neighborhood_old = graph_to_add.neighborhood()
    neighborhood_new = []
    for i in neighborhood_old:
        neighborhood_new.append([x + graph_penetrating_tot.vcount() - 1 for x in i])

    nodes_before_penetrating_tree = graph_penetrating_tot.vcount()

    graph_penetrating_tot.add_vertices(graph_to_add.vcount())

    for vertex in range(graph_to_add.vcount()):

        vertex_new = vertex + nodes_before_penetrating_tree

        graph_penetrating_tot.vs[vertex_new]['x_coordinate'] = graph_to_add.vs[vertex]['x_coordinate'] - delta_x
        graph_penetrating_tot.vs[vertex_new]['y_coordinate'] = graph_to_add.vs[vertex]['y_coordinate'] - delta_y
        graph_penetrating_tot.vs[vertex_new]['z_coordinate'] = graph_to_add.vs[vertex]['z_coordinate'] - delta_z
        graph_penetrating_tot.vs[vertex_new]['attachmentVertex'] = graph_to_add.vs[vertex]['attachmentVertex']
        graph_penetrating_tot.vs[vertex_new]['CapBedConnection'] = graph_to_add.vs[vertex]['CapBedConnection']
        graph_penetrating_tot.vs[vertex_new]['Type'] = graph_to_add.vs[vertex]['Type']

        graph_penetrating_tot.vs[vertex_new]['PartOfPenetratingTree'] = 1

        neighbors = graph_to_add.neighbors(vertex)
        neighbors_new = [x + nodes_before_penetrating_tree for x in neighbors]

        for neighbor in neighbors_new:

            if neighbor > vertex_new:

                # Damit attribute übernommen werden können
                edge_id_origin_tree = graph_to_add.get_eid(vertex, neighbor-nodes_before_penetrating_tree)

                graph_penetrating_tot.add_edge(vertex_new, neighbor,
                                               edge_length=graph_to_add.es[edge_id_origin_tree]['edge_length'],
                                               diameter=graph_to_add.es[edge_id_origin_tree]['diameter'],
                                               PartOfPenetratingTree=1,
                                               Type=graph_to_add.es[edge_id_origin_tree]['Type'])

            else:

                continue


    return graph_penetrating_tot


def add_penetrating_tree_to_cap_bed(graph_penetrating_tree, graph_capillary_bed, length_honeycomb):

    number_of_vertex_to_be_connected = 0

    for v in range(graph_penetrating_tree.vcount()):

        if graph_penetrating_tree.vs[v]['CapBedConnection'] == 1:

            number_of_vertex_to_be_connected += 1

    print('Number of Vertices to be connected: ', number_of_vertex_to_be_connected)
    print('Total number of vertices: ', graph_penetrating_tree.vcount())

    zaehler = 0

    neighborhood_old = graph_penetrating_tree.neighborhood()
    neighborhood_new = []
    for i in neighborhood_old:
        neighborhood_new.append([x + graph_capillary_bed.vcount() - 1 for x in i])

    nodes_before_penetrating_tree = graph_capillary_bed.vcount()

    graph_capillary_bed.add_vertices(graph_penetrating_tree.vcount())

    for vertex in range(graph_penetrating_tree.vcount()):

        vertex_new = vertex + nodes_before_penetrating_tree

        graph_capillary_bed.vs[vertex_new]['x_coordinate'] = graph_penetrating_tree.vs[vertex]['x_coordinate']
        graph_capillary_bed.vs[vertex_new]['y_coordinate'] = graph_penetrating_tree.vs[vertex]['y_coordinate']
        graph_capillary_bed.vs[vertex_new]['z_coordinate'] = graph_penetrating_tree.vs[vertex]['z_coordinate']
        graph_capillary_bed.vs[vertex_new]['attachmentVertex'] = graph_penetrating_tree.vs[vertex]['attachmentVertex']
        graph_capillary_bed.vs[vertex_new]['CapBedConnection'] = graph_penetrating_tree.vs[vertex]['CapBedConnection']
        graph_capillary_bed.vs[vertex_new]['Type'] = graph_penetrating_tree.vs[vertex]['Type']

        graph_capillary_bed.vs[vertex_new]['PartOfPenetratingTree'] = 1

        neighbors = graph_penetrating_tree.neighbors(vertex)
        neighbors_new = [x + nodes_before_penetrating_tree for x in neighbors]

        for neighbor in neighbors_new:

            if neighbor > vertex_new:

                edge_id_origin_tree = graph_penetrating_tree.get_eid(vertex, neighbor - nodes_before_penetrating_tree)

                graph_capillary_bed.add_edge(vertex_new, neighbor,
                                             edge_length=graph_penetrating_tree.es[edge_id_origin_tree]['edge_length'],
                                             diameter=graph_penetrating_tree.es[edge_id_origin_tree]['diameter'],
                                             PartOfPenetratingTree=1,
                                             Type=graph_penetrating_tree.es[edge_id_origin_tree]['Type'])

        if graph_penetrating_tree.vs[vertex]['CapBedConnection'] == 1:

            if graph_penetrating_tree.vs[vertex]['attachmentVertex'] == 0:

                zaehler += 1
                list_distances = []
                list_edges = []

                x_vertex = graph_capillary_bed.vs[vertex_new]['x_coordinate']
                y_vertex = graph_capillary_bed.vs[vertex_new]['y_coordinate']
                z_vertex = graph_capillary_bed.vs[vertex_new]['z_coordinate']

                coord_sphere = {'x': x_vertex, 'y': y_vertex, 'z': z_vertex}

                possible_connections_edge_ids = []
                _radius_ = length_honeycomb

                while len(possible_connections_edge_ids) == 0:

                    possible_connections_edge_ids = Chose_activated_region.define_possible_connections(graph_capillary_bed,
                                                                                                   coord_sphere, _radius_)

                    _radius_ = length_honeycomb * 1.5

                print('Possible connections exist.')
                print('Length: ', len(possible_connections_edge_ids))
                print('Zähler: ', zaehler)

                for _edge_ in possible_connections_edge_ids:

                        source_vertex = graph_capillary_bed.es[_edge_].source
                        target_vertex = graph_capillary_bed.es[_edge_].target

                        x_source = graph_capillary_bed.vs[source_vertex]['x_coordinate']
                        y_source = graph_capillary_bed.vs[source_vertex]['y_coordinate']
                        z_source = graph_capillary_bed.vs[source_vertex]['z_coordinate']
                        x_target = graph_capillary_bed.vs[target_vertex]['x_coordinate']
                        y_target = graph_capillary_bed.vs[target_vertex]['y_coordinate']
                        z_target = graph_capillary_bed.vs[target_vertex]['z_coordinate']

                        x_mp = (x_source + x_target) / 2
                        y_mp = (y_source + y_target) / 2
                        z_mp = (z_source + z_target) / 2

                        distance = math.sqrt(math.pow(x_vertex - x_mp, 2) + math.pow(y_vertex - y_mp, 2) +
                                             math.pow(z_vertex - z_mp, 2))

                        list_distances.append(distance)
                        list_edges.append(_edge_)

                start_node_first_edge = graph_capillary_bed.es[list_edges[list_distances.index(
                    min(list_distances))]].source

                edge_length_start_node = math.sqrt(math.pow(graph_capillary_bed.vs[start_node_first_edge]['x_coordinate'] -
                                                            graph_capillary_bed.vs[vertex_new]['x_coordinate'], 2) +
                                                   math.pow(graph_capillary_bed.vs[start_node_first_edge]['y_coordinate'] -
                                                            graph_capillary_bed.vs[vertex_new]['y_coordinate'], 2) +
                                                   math.pow(graph_capillary_bed.vs[start_node_first_edge]['z_coordinate'] -
                                                            graph_capillary_bed.vs[vertex_new]['z_coordinate'], 2))

                start_node_second_edge = graph_capillary_bed.es[list_edges[list_distances.index(
                    min(list_distances))]].target

                edge_length_end_node = math.sqrt(math.pow(graph_capillary_bed.vs[start_node_second_edge]['x_coordinate'] -
                                                          graph_capillary_bed.vs[vertex_new]['x_coordinate'], 2) +
                                                 math.pow(graph_capillary_bed.vs[start_node_second_edge]['y_coordinate'] -
                                                          graph_capillary_bed.vs[vertex_new]['y_coordinate'], 2) +
                                                 math.pow(graph_capillary_bed.vs[start_node_second_edge]['z_coordinate'] -
                                                          graph_capillary_bed.vs[vertex_new]['z_coordinate'], 2))

                graph_capillary_bed.add_edge(start_node_first_edge, vertex_new, connection_CB_Pene=1,
                                             diameter=4 / math.pow(10, 6), edge_length=edge_length_start_node / math.pow(10,
                                                                                                                         6),
                                             Type=3)

                graph_capillary_bed.add_edge(start_node_second_edge, vertex_new, connection_CB_Pene=1,
                                             diameter=4 / math.pow(10, 6), edge_length=edge_length_end_node / math.pow(10,
                                                                                                                       6),
                                             Type=3)

                graph_capillary_bed.delete_edges(list_edges[list_distances.index(min(list_distances))])

            else:

                continue

        else:

            continue

    return graph_capillary_bed
