
def edit_penetrating_tree(penetrating_tree, length_x, length_y, trim_factor):

    x_min = 0 - length_x * trim_factor
    x_max = length_x * (1 + trim_factor)
    y_min = 0 - length_y * trim_factor
    y_max = length_y * (1 + trim_factor)

    edges_to_be_deleted = []
    vertices_to_be_deleted = []

    for edge in range(penetrating_tree.ecount()):

        start_point = penetrating_tree.es[edge].source
        end_point = penetrating_tree.es[edge].target

        x_p_start = penetrating_tree.vs[start_point]['x_coordinate']
        x_p_end = penetrating_tree.vs[end_point]['x_coordinate']

        y_p_start = penetrating_tree.vs[start_point]['y_coordinate']
        y_p_end = penetrating_tree.vs[end_point]['y_coordinate']

        if x_p_start < x_min or x_p_start > x_max or y_p_start > y_max or y_p_start < y_min:

            edges_to_be_deleted.append(edge)

            # if (len(penetrating_tree.neighbors(end_point)) - 1) < 2:
            #
            #     penetrating_tree.vs[end_point]['CapBedConnection'] = 1

        elif x_p_end < x_min or x_p_end > x_max or y_p_end > y_max or y_p_end < y_min:

            edges_to_be_deleted.append(edge)

            # if (len(penetrating_tree.neighbors(start_point)) - 1) < 2:
            #
            #     penetrating_tree.vs[start_point]['CapBedConnection'] = 1

    edges_to_be_deleted_new = []

    for i in range(len(edges_to_be_deleted)):

        edge_id_new = edges_to_be_deleted[i] - i
        edges_to_be_deleted_new.append(edge_id_new)

    for edge in edges_to_be_deleted_new:

        penetrating_tree.delete_edges([edge])

    for vertex in range(penetrating_tree.vcount()):

        x_vertex = penetrating_tree.vs[vertex]['x_coordinate']
        y_vertex = penetrating_tree.vs[vertex]['y_coordinate']

        if x_vertex < x_min or x_vertex > x_max or y_vertex > y_max or y_vertex < y_min:

            vertices_to_be_deleted.append(vertex)

        else:

            continue

    vertices_to_be_deleted_new = []

    for i in range(len(vertices_to_be_deleted)):
        vertex_id_new = vertices_to_be_deleted[i] - i
        vertices_to_be_deleted_new.append(vertex_id_new)

    for vertex in vertices_to_be_deleted_new:
        penetrating_tree.delete_vertices([vertex])

    for v in range(penetrating_tree.vcount()):

        neighbors_current_vertex = penetrating_tree.neighbors(v)

        if len(neighbors_current_vertex) <= 1:
            penetrating_tree.vs[v]['CapBedConnection'] = 1

    return penetrating_tree


def delete_unconnected_parts(graph_penetrating):

    attachment_vertices = []

    for vertex in range(graph_penetrating.vcount()):

        if graph_penetrating.vs[vertex]['attachmentVertex'] == 1:

            attachment_vertices.append(vertex)

    vertices_to_deleted = []

    for vertex in range(graph_penetrating.vcount()):

        control_variable = 0

        for attachmentVertex in attachment_vertices:

            shortest_path = graph_penetrating.get_shortest_paths(attachmentVertex, vertex)

            if len(shortest_path[0]) != 0:

                control_variable = 1

                break

            else:

                continue

        if control_variable == 0:

            neighbors = graph_penetrating.neighbors(vertex)

            for neighbor in neighbors:

                graph_penetrating.delete_edges(graph_penetrating.get_eid(vertex, neighbor))

            vertices_to_deleted.append(vertex)

    graph_penetrating.delete_vertices(vertices_to_deleted)

    return graph_penetrating


def control_function(graph_):

    attachment_vertices = []

    for vertex in range(graph_.vcount()):

        if graph_.vs[vertex]['attachmentVertex'] == 1:

            attachment_vertices.append(vertex)

    for vertex in range(graph_.vcount()):

        control_variable = 0

        for attachmentVertex in attachment_vertices:

            shortest_path = graph_.get_shortest_paths(attachmentVertex, vertex)

            if len(shortest_path) == 0:

                control_variable = 1

        if control_variable == 1:

            print('Vertex ', vertex, ' not connected.')

        neighbours_vertex = graph_.neighbors(vertex)

        if len(neighbours_vertex) <= 1:

            # print(neighbours_vertex)

            if graph_.vs[vertex]['CapBedConnection'] != 1:

                print('CapBedConnection Id wrong.')

    print('Check Penetrating Tree over.')

    return
