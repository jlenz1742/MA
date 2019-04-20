import igraph as ig
import math


graph_main = ig.Graph()
x = 3
y = 3
z = 2

graph_main.vs['x_coordinate'] = float()
graph_main.vs['y_coordinate'] = float()
graph_main.vs['z_coordinate'] = float()
graph_main.vs['vortex_id'] = int()

# Edge Attributes

graph_main.es['edge_id'] = int()

def create_graph(graph):

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



                return graph