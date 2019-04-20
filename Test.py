import igraph as ig

import igraph as ig
import math

""" Variables """

l_honeycomb = 1                           # Side length of a hexagon
l_gap = 1                                 # Distance between two layers
diameter = 1                              # Diameter of edges
x = 3                                     # Must be odd
y = 3                                     # Can be either odd or even
z = 2

""" Graph """

graph = ig.Graph()

# Vertices Attributes

graph.vs['x_coordinate'] = float()
graph.vs['y_coordinate'] = float()
graph.vs['z_coordinate'] = float()
graph.vs['vortex_id'] = int()

# Edge Attributes

graph.es['edge_id'] = int()

''' Add Vertices '''

horizontal_lines = 4 * y + 1

print('number of horinzontal lines: ', horizontal_lines)

vertices_per_line = x + 1

print('number of vertices per line: ', vertices_per_line)

total_vertices = horizontal_lines * vertices_per_line

print('total vertices per Line: ', total_vertices)

graph.add_vertices(total_vertices)

''' Add Edges '''

start_id = 0
end_vertices = list(range(total_vertices - vertices_per_line, total_vertices))

for line in range(horizontal_lines):

    print('iteration through line: ', line)

    vertices_in_current_line = list(range(start_id, start_id + vertices_per_line))

    print('vertices in current line: ', vertices_in_current_line)

    for i in range(len(vertices_in_current_line)):

        # Create Coordinates

        if (line % 2) == 0:

            graph.vs[vertices_in_current_line[i]]['y_coordinate'] = math.sin(60 * 2 * math.pi / 360) * l_honeycomb * line # Set y-coordinate of each vertex

        else:

            graph.vs[vertices_in_current_line[i]]['y_coordinate'] = math.sin(60 * 2 * math.pi / 360) * l_honeycomb * (line-1) + math.cos(30 * 2 * math.pi / 360) * l_honeycomb * 0.5


        current_x_coordinate = 0

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

        print(graph.vs[vertices_in_current_line[i]])

        # Create Connections

        if vertices_in_current_line[i] in end_vertices:

            if (vertices_in_current_line[i] % 2) == 0:
                graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + 1)

        elif (line % 2) != 0:

            print(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)
            graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)       # Verbindung gegen oben

        else:

            graph.add_edge(vertices_in_current_line[i], vertices_in_current_line[i] + x + 1)       # Verbindung gegen oben

            if (line % 4) == 0:

                print('da: ', vertices_in_current_line[i])
                if (vertices_in_current_line[i] % 2) == 0:


                    print(vertices_in_current_line[i], vertices_in_current_line[i] + 1)
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


nodes_slanting_to_the_left = []
nodes_slanting_to_the_right = []
#
start_nodes = list(range(x+1, 2*(x+1)))

for node in start_nodes:

    print('node: ', node)
    if (node % 2) == 0:

        for i in range(y):

            nodes_slanting_to_the_left.append(node + i * 4 * (x+1))
            nodes_slanting_to_the_right.append(node + i * 4 * (x+1) + 2 * (x+1))

    else:

        for i in range(y):

            nodes_slanting_to_the_left.append(node + i * 4 * (x+1) + 2 * (x+1))
            nodes_slanting_to_the_right.append(node + i * 4 * (x+1))


    print(node)

nodes_slanting_to_the_left.sort()
nodes_slanting_to_the_right.sort()
print(nodes_slanting_to_the_left)
print(nodes_slanting_to_the_right)


for level in range(z):

    print(level)


#graphslanting_to_the_left = []

#print(graph.vs[16])
#print(ig.summary(graph))
# layout = graph.layout("kk")
# ig.plot(graph, layout = layout)
#
# print(graph)