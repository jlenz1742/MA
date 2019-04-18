import igraph as ig
from Functions import generate_2D_network


x = generate_2D_network(3,3,0,0)

y = generate_2D_network(3,3, x.vcount(), x.ecount())

print(y)

def merge_graphs(graph_main, graph_sub):

    return None


def modify_2D_network(graph, control_variable_):

    # Control_variable = 0: Split edges which are directed slightly to the right(lowermost plane, maybe top plane)
    # Control_variable = 1: Split edges which are directed slightly to the left (Maybe top plane)
    # Control_variable = 2: Split edges (Planes in between)



    print(graph)

    return

modify_2D_network(x, 1)

