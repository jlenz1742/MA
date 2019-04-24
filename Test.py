import igraph as ig
import math
from Plot import plot_graph


graph_main = ig.Graph()
x_dimension = 3
y_dimension = 3
z_dimension = 3
lenght_honeycomb = 3

graph_main.vs['x_coordinate'] = float()
graph_main.vs['y_coordinate'] = float()
graph_main.vs['z_coordinate'] = float()
graph_main.vs['vortex_id'] = int()

# Edge Attributes

graph_main.es['edge_id'] = int()

def create_graph(graph, x, y, z, l_honeycomb):

    horizontal_lines = 4 * y + 1
    vertices_per_line = x + 1
    total_vertices = horizontal_lines * vertices_per_line
    graph.add_vertices(total_vertices)

    ''' Add Edges '''

    start_id = graph.vcount() - total_vertices
    end_vertices = list(range(graph.vcount() - vertices_per_line, graph.vcount()))

    for line in range(horizontal_lines):

        vertices_in_current_line = list(range(start_id, start_id + vertices_per_line))

        for i in range(len(vertices_in_current_line)):

            # Create Coordinates

            graph.vs[vertices_in_current_line[i]]['z_coordinate'] = z * l_honeycomb

            if (line % 2) == 0:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = math.sin(60 * 2 * math.pi / 360) * l_honeycomb * line / 2 # Set y-coordinate of each vertex

            else:

                graph.vs[vertices_in_current_line[i]]['y_coordinate'] = math.cos(30 * 2 * math.pi / 360) * l_honeycomb / 2 + (line-1) * math.cos(30 * 2 * math.pi / 360) * l_honeycomb /2

            if (line % 4) == 0:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = 1.5 * l_honeycomb * i

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = l_honeycomb * (1.5 * i - 0.5)

            elif (line % 2) != 0:

                if (vertices_in_current_line[i] % 2) == 0:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = 1.5 * l_honeycomb * i - 0.25 * l_honeycomb

                else:

                   graph.vs[vertices_in_current_line[i]]['x_coordinate'] = l_honeycomb * (1.5 * i - 0.5) + 0.25 * l_honeycomb


            else:

                if (vertices_in_current_line[i] % 2) == 0:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = 1.5 * l_honeycomb * i - 0.5 * l_honeycomb

                else:

                    graph.vs[vertices_in_current_line[i]]['x_coordinate'] = l_honeycomb * (1.5 * i - 0.5) + 0.5 * l_honeycomb

            # Create Connections

            if vertices_in_current_line[i] in end_vertices:

                if (vertices_in_current_line[i] % 2) == 0:
                    graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)

            elif (line % 2) != 0:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)       # Verbindung gegen oben

            else:

                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)       # Verbindung gegen oben

                if (line % 4) == 0:

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


    return graph



for level in range(z_dimension):

    graph_main = create_graph(graph_main, x_dimension, y_dimension, level, lenght_honeycomb)

plot_graph(graph_main)

# for i in range(graph_main.vcount()):

#   print(graph_main.vs[i])