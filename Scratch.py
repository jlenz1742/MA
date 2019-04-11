import igraph as ig


''' INPUT '''

l_honeycomb = 1                           # Side length of a hexagon
l_gap = 1                                 # Distance between two layers
x = 3
y = 3
z = 3

graph = ig.Graph()
number_ver = 10
graph.add_vertices(10)

''' Set Graph Attributes '''

# WARNING: ID changes if vertices are deleted or added -> Therefore, an ID-attribute is added!!

# Vertices Attributes

graph.vs['x_coordinate'] = float()
graph.vs['y_coordinate'] = float()
graph.vs['z_coordinate'] = float()
graph.vs['vortex_id'] = int()

# Edge Attributes

graph.es['edge_id'] = int()
graph.add_vertex(10)
graph.add_vertex(199)


b=graph.vs.find(name=10)
print(b)


