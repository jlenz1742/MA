import igraph as ig

''' INPUT '''

l_honeycomb = 1                           # Side length of a hexagon
l_gap = 1                                 # Distance between two layers
diameter = 1                              # Diameter of edges
x = 3
y = 3
z = 3

graph = ig.Graph()

''' Set Graph Attributes '''

# WARNING: ID changes if vertices are deleted or added -> Therefore, an ID-attribute is added!!

# Vertices Attributes

graph.vs['x_coordinate'] = float()
graph.vs['y_coordinate'] = float()
graph.vs['z_coordinate'] = float()
graph.vs['vortex_id'] = int()

# Edge Attributes

graph.es['edge_id'] = int()

# Calculate number of vertices

total_number_of_lines = (y+1) * 2
sandwich_lines = total_number_of_lines - 2
number_of_vertices_per_line = (x+1)

sandwich_vertices = sandwich_lines * number_of_vertices_per_line
bottom_and_top_vertices = x*2

total_vertices = bottom_and_top_vertices + sandwich_vertices
graph.add_vertices(total_vertices)


# Create connections between vertices


number_of_horizontal_edges = (x*y) + x

current_start_vertex = 0
current_stop_vertex = current_start_vertex + 1

for i in range(number_of_horizontal_edges):

    graph.add_edge(current_start_vertex, current_stop_vertex)
    current_start_vertex += 2
    current_stop_vertex = current_start_vertex + 1

next_id = current_stop_vertex - 1

print(next_id)
print(graph)

if (x % 2) == 0:

    print('Even')

    starting_points_bottom = list(range(0, x))
    starting_points_bottom.append(starting_points_bottom[-1] + x)

    for bottom_vertex in starting_points_bottom:

        if bottom_vertex == 0:

            current_start_vertex = bottom_vertex

            source_vertex = bottom_vertex
            target_vertex = next_id
            next_existing_vertex = bottom_vertex

            for i in range(2*y):

                if (i % 2) == 0:

                    graph.add_edge(source_vertex, target_vertex)

                    source_vertex = target_vertex
                    next_id += 1
                    next_existing_vertex += 2*x
                    target_vertex = next_existing_vertex

                else:

                    graph.add_edge(source_vertex, target_vertex)
                    source_vertex = next_existing_vertex
                    target_vertex = next_id

        elif bottom_vertex == starting_points_bottom[-1]:

            current_start_vertex = bottom_vertex

            source_vertex = bottom_vertex
            target_vertex = next_id
            next_existing_vertex = bottom_vertex

            for i in range(2 * y):

                if (i % 2) == 0:

                    graph.add_edge(source_vertex, target_vertex)

                    source_vertex = target_vertex
                    next_id += 1
                    next_existing_vertex += 2 * x
                    target_vertex = next_existing_vertex

                else:

                    graph.add_edge(source_vertex, target_vertex)
                    source_vertex = next_existing_vertex
                    target_vertex = next_id

        else:

            source_vertex = bottom_vertex

            for i in range(2*y + 1):

                if (i % 2) == 0:

                    graph.add_edge(source_vertex, source_vertex + x - 1)
                    source_vertex += (x-1)

                else:

                    graph.add_edge(source_vertex, source_vertex + x + 1)
                    source_vertex += (x + 1)


else:

    print('Odd')

    starting_points_bottom = list(range(0, x+1))

    print(starting_points_bottom)

    for bottom_vertex in starting_points_bottom:

        if bottom_vertex == 0:

            current_start_vertex = bottom_vertex

            source_vertex = bottom_vertex
            target_vertex = next_id
            next_existing_vertex = bottom_vertex

            for i in range(2*y):

                if (i % 2) == 0:

                    graph.add_edge(source_vertex, target_vertex)

                    source_vertex = target_vertex
                    next_id += 1
                    next_existing_vertex += 2*x
                    target_vertex = next_existing_vertex

                else:

                    graph.add_edge(source_vertex, target_vertex)
                    source_vertex = next_existing_vertex
                    target_vertex = next_id

        elif bottom_vertex == starting_points_bottom[-1]:

            current_start_vertex = bottom_vertex

            source_vertex = bottom_vertex
            target_vertex = next_id
            next_existing_vertex = bottom_vertex

            for i in range(2 * y):

                if (i % 2) == 0:

                    graph.add_edge(source_vertex, target_vertex)

                    source_vertex = target_vertex
                    next_id += 1
                    next_existing_vertex += 2 * x
                    target_vertex = next_existing_vertex

                else:

                    graph.add_edge(source_vertex, target_vertex)
                    source_vertex = next_existing_vertex
                    target_vertex = next_id

        else:

            print('da')
            source_vertex = bottom_vertex

            for i in range(2*y + 1):

                graph.add_edge(source_vertex, source_vertex + x )
                source_vertex += x



print(graph)



