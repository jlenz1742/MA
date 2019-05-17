import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_graph(graph):

    # Graph need to have coordinates of each node as vertex attribute

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for edge in range(graph.ecount()):

        x = []
        y = []
        z = []

        x.append(graph.vs[graph.es[edge].source]['x_coordinate'])
        x.append(graph.vs[graph.es[edge].target]['x_coordinate'])

        y.append(graph.vs[graph.es[edge].source]['y_coordinate'])
        y.append(graph.vs[graph.es[edge].target]['y_coordinate'])

        z.append(graph.vs[graph.es[edge].source]['z_coordinate'])
        z.append(graph.vs[graph.es[edge].target]['z_coordinate'])

        try:
            if graph.es[edge]['connection_CB_Pene'] == 1:

                ax.plot(x, y, z, color='blue')

            elif graph.es[edge]['PartOfPenetratingTree'] == 1:

                ax.plot(x, y, z, color='green')

            else:

                ax.plot(x, y, z, color='red')

        except:

            ax.plot(x, y, z, color='red')

    for vertex in range(graph.vcount()):

        if graph.vs[vertex]['attachmentVertex'] == 1:
            ax.scatter(graph.vs[vertex]['x_coordinate'], graph.vs[vertex]['y_coordinate'], color='black')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()

def plot_geometrical_help(graph):

    # Graph need to have coordinates of each node as vertex attribute

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for edge in range(graph.ecount()):

        x = []
        y = []
        z = []

        x.append(graph.vs[graph.es[edge].source]['x_coordinate'])
        x.append(graph.vs[graph.es[edge].target]['x_coordinate'])

        y.append(graph.vs[graph.es[edge].source]['y_coordinate'])
        y.append(graph.vs[graph.es[edge].target]['y_coordinate'])

        z.append(graph.vs[graph.es[edge].source]['z_coordinate'])
        z.append(graph.vs[graph.es[edge].target]['z_coordinate'])

        if graph.es[edge]['connection_CB_Pene'] == 1:

            ax.plot(x, y, z, color='blue')

        elif graph.es[edge]['PartOfPenetratingTree'] == 1:

            ax.plot(x, y, z, color='green')

        else:

            ax.plot(x, y, z, color='red')

    for vertex in range(graph.vcount()):

        if graph.vs[vertex]['artery_point'] == 1:

            ax.scatter(graph.vs[vertex]['x_coordinate'], graph.vs[vertex]['y_coordinate'], color='green')

        elif graph.vs[vertex]['vein_point'] == 1:

            ax.scatter(graph.vs[vertex]['x_coordinate'], graph.vs[vertex]['y_coordinate'], color='blue')

    plt.show()

def plot_path(graph, path):

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(len(path)-1):

        x = []
        y = []
        z = []


        x.append(graph.vs[path[i]]['x_coordinate'])
        x.append(graph.vs[path[i+1]]['x_coordinate'])

        y.append(graph.vs[path[i]]['y_coordinate'])
        y.append(graph.vs[path[i+1]]['y_coordinate'])

        z.append(graph.vs[path[i]]['z_coordinate'])
        z.append(graph.vs[path[i+1]]['z_coordinate'])

        ax.plot(x, y, z, color='red')

    plt.show()
