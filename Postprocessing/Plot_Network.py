import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd

df = pd.read_csv(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\gamma_0_1\out\meshdata_9999.csv')
df_final = pd.read_csv(r'D:\00 Privat\01_Bildung\01_ETH Zürich\MSc\00_Masterarbeit\Data_Simulation\wo_rbcs\gamma_0_1\out\meshdata_5469999.csv')

headers = list(df.columns.values)
# print(headers)

x = df['x']
x_min = min(x)
x_max = max(x)

y = df['y']
y_min = min(y)
y_max = max(y)

z = df['z']
z_min = min(z)
z_max = max(z)

# t = df['D']

t_1 = np.array(df['D'])
t_2 = np.array(df_final['D'])

t = t_2/t_1
# print(min(t), max(t))

# generate a list of (x,y,z) points
points = np.array([x, y, z]).transpose().reshape(-1, 1, 3)
# print(points.shape)  # Out: (len(x),1,3)

# set up a list of segments
segs = np.concatenate([points[:-1],points[1:]],axis=1)
# print(segs)
# print(segs.shape)  # Out: ( len(x)-1, 2, 3 )
                  # see what we've done here -- we've mapped our (x,y,z)
                  # points to an array of segment start/end coordinates.
                  # segs[i,0,:] == segs[i-1,1,:]

# make the collection of segments
lc = Line3DCollection(segs, cmap=plt.get_cmap('Greys'))
lc.set_array(t)  # color the segments by our parameter

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
plt.colorbar(lc)
plt.show()