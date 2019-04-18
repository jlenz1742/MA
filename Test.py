import igraph as ig


x = ig.Graph()

x.add_vertices(3)

x.add_edge(0,1)
x.add_edge(1,2)
print(x)
x.add_edge(0,2)
x.delete_vertices(1)
print(x)