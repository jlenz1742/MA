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


def find_points_on_line(p_1, p_2, number_of_points):

    point_1 = np.array(p_1)
    point_2 = np.array(p_2)

    points_all = []

    vector = point_2 - point_1
    delta_vector = vector/number_of_points

    # length = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2) + math.pow(vector[2], 2)

    for i in range(number_of_points+1):


        point_new = point_1 + delta_vector * i
        points_all.append(point_new)

    return points_all


cube_x = 10
cube_y = 10
cube_z = 9

cube_dictionary = {}

for k in range(cube_x * cube_y * cube_z):

    cube_dictionary[str(k)] = k

print(cube_dictionary)
