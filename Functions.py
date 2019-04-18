import igraph as ig
import math

""" Variables """

l_honeycomb = 1                           # Side length of a hexagon
l_gap = 1                                 # Distance between two layers
diameter = 1                              # Diameter of edges
x_dimens = 5                                     # Must be odd
y_dimens = 4                                     # Can be either odd or even
z = 3

""" Graph """

def generate_2D_network(x, y, start_id_vertices, start_id_edges):

    graph = ig.Graph()

    # Vertices Attributes

    graph.vs['x_coordinate'] = float()
    graph.vs['y_coordinate'] = float()
    graph.vs['z_coordinate'] = float()

    ''' Add Vertices '''

    horizontal_lines = 2 * y + 1
    vertices_per_line = x + 1
    total_vertices = horizontal_lines * vertices_per_line

    graph.add_vertices(total_vertices)

    ''' Add Edges '''

    start_id = 0
    end_vertices = list(range(total_vertices - vertices_per_line, total_vertices))

    for line in range(horizontal_lines):

        vertices_in_current_line = list(range(start_id, start_id + vertices_per_line))

        for i in range(len(vertices_in_current_line)):

            # Create Coordinates

            graph.vs[vertices_in_current_line[i]]['y_coordinate'] = math.sin(60 * 2 * math.pi / 360) * l_honeycomb * line # Set y-coordinate of each vertex

            current_x_coordinate = 0

            if (line % 2) == 0:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = 1.5 * l_honeycomb * i

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = l_honeycomb * (1.5 * i - 0.5)

            else:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = 1.5 * l_honeycomb * i - 0.5 * l_honeycomb

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = l_honeycomb * (1.5 * i - 0.5) + 0.5 * l_honeycomb

            # Create Connections

            if vertices_in_current_line[i] in end_vertices:

                if (vertices_in_current_line[i] % 2) == 0:
                    graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)

            else:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)

                if (line % 2) == 0:

                    if (vertices_in_current_line[i] % 2) == 0:

                        graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)

                    else:

                        None

                else:

                    if (vertices_in_current_line[i] % 2) != 0:

                        if vertices_in_current_line[i] == start_id + vertices_per_line - 1:

                            None

                        else:

                            graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)

                    else:

                        None

        start_id += vertices_per_line

    graph.vs["original_id"] = list(range(start_id_vertices, start_id_vertices + graph.vcount()))
    graph.es["original_id"] = list(range(start_id_edges, start_id_edges + graph.ecount()))
    return graph

x = generate_2D_network(x_dimens, y_dimens, 0, 0)


