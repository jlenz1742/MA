import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Realistic_network_mouse\mvn1_edit\edge_data.csv'

df = pd.read_csv(path)

diameters = df['D'].tolist()

mu = np.mean(diameters)
median = np.median(diameters)
sigma = np.std(diameters)


########################################################################################################################
#                                                                                                                      #
#                                                  Plot (Histogram)                                                    #
#                                                                                                                      #
########################################################################################################################

# fig, ax = plt.subplots()
# textstr = '\n'.join((
#     r'$\mu=%.8f$' % (mu, ),
#     r'$\mathrm{median}=%.8f$' % (median, ),
#     r'$\sigma=%.8f$' % (sigma, )))
#
# ax.hist(diameters, 50)
# # these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#
# # place a text box in upper left in axes coords
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
#
# plt.show()





