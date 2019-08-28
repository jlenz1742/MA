import datetime
import os
import time
import json
import pandas as pd
import glob
import re
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import random
import igraph as ig

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_cube():

    plt.figure(facecolor='w',figsize=(10, 40))
    # generate random data
    x = np.random.randint(0,500,(11,11))

    print(x[1])
    dmin,dmax = 0,200
    plt.imshow(x, vmin=dmin, vmax=dmax)

    # create the colorbar
    # the aspect of the colorbar is set to 'equal', we have to set it to 'auto',
    # otherwise twinx() will do weird stuff.
    # ref: Draw colorbar with twin scales - stack overflow -
    # URL: https://stackoverflow.com/questions/27151098/draw-colorbar-with-twin-scales
    cbar = plt.colorbar()
    pos = cbar.ax.get_position()
    ax1 = cbar.ax
    ax1.set_aspect('auto')

    # resize the colorbar
    pos.x1 -= 0.08

    # arrange and adjust the position of each axis, ticks, and ticklabels
    ax1.set_position(pos)
    ax1.yaxis.set_ticks_position('right') # set the position of the first axis to right
    ax1.yaxis.set_label_position('right') # set the position of the fitst axis to right
    ax1.set_ylabel(u'Flow Rate [$m^3/s$]')

    # Save the figure
    plt.show()

plot_3d_cube()