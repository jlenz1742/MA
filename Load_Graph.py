import igraph as ig
import Plot


graph_ = ig.Graph.Read_Pickle(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Coding_JLEN\MA\Export\20190819_151428\test.pkl')

Plot.plot_penetrating_tree(graph_)








