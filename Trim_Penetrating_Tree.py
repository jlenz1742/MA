
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

        elif x_p_end < x_min or x_p_end > x_max or y_p_end > y_max or y_p_end < y_min:

            edges_to_be_deleted.append(edge)

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

    return penetrating_tree


def delete_unconnected_trees(graph_main):

    vertices_to_be_deleted = []
    edges_to_be_deleted = []

    for vertex in range(graph_main.vcount()):

        if graph_main.vs[vertex]['Type'] == 2 or graph_main.vs[vertex]['Type'] == 3:

            if len(graph_main.neighbors(vertex)) <= 1:

                vertices_to_be_deleted.append(vertex)
                print('da hamamdsf')

    vertices_to_be_deleted_new = []

    for i in range(len(vertices_to_be_deleted)):
        vertex_id_new = vertices_to_be_deleted[i] - i
        vertices_to_be_deleted_new.append(vertex_id_new)

    for vertex in vertices_to_be_deleted_new:
        graph_main.delete_vertices([vertex])

    return graph_main



