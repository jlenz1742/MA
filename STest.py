import igraph as ig


graph = ig.Graph()

graph.add_vertices(4)

graph.vs[0]['x_coordinate'] = 0
graph.vs[0]['y_coordinate'] = 0
graph.vs[0]['z_coordinate'] = 1

graph.add_edge(0, 1, original_id=0)
graph.add_edge(1, 2, original_id=1)
graph.add_edge(2, 3, original_id=2)
graph.add_edge(3, 0, original_id=3)

graph.vs["original_id"] = list(range(graph.vcount()))


def split_edge(edge_id):

    print('Current edge id: ', edge_id)

    source_vertex = graph.es['original_id' == edge_id].source
    target_vertex = graph.es['original_id' == edge_id].target

    print(source_vertex, target_vertex)

    graph.delete_edges('original_id' == edge_id)

    graph.add_vertices(1)
    id_new_vertex = graph.vcount() - 1

    graph.add_edge(source_vertex, id_new_vertex, original_id=graph.ecount()-1)
    graph.add_edge(id_new_vertex, target_vertex, original_id=graph.ecount()-1)

    return

print(graph.ecount())

for edge in range(graph.ecount()):

    split_edge(edge)



for edge in range(graph.ecount()):

    print(graph.es[edge])
