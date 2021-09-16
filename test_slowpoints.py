# %load_ext autoreload
# %autoreload 2

import tensorflow as tf
import scipy
import numpy as np
from slowpoints import find_slow_points, visualize_slow_points
import pdb
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import json
import pandas as pd
from sklearn.decomposition import PCA
import pickle

"""Toy example
# 2-d example from the beginning of the paper
def F0(x):
    return jnp.array([(1 - x[0]**2) * x[1], x[0]/2. - x[1]])
x0=[np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1)])
                                   for i in range(20)]
slow_points, J_eig = find_slow_points(F0, x0)
                        """

with open('weight_history_0.json') as f:
    weight_history = json.load(f)

def RNN_RHS(x, W, bias, activation_fn, u=None, Wu=None):
    if u is None:
        u = np.zeros(x.shape)
        Wu = np.zeros(W.shape)

    return -1 * x + activation_fn(jnp.matmul(W, x) + jnp.matmul(Wu, u) + bias)

# Generate 500 initial conditions in the (-1, 1) hypercube
x0 = []
for i in range(500):
    x0.append(np.random.normal(-1, 1, size=(256,)))

W = np.array(weight_history['trained weights'][-1])
b = np.array(weight_history['bias'])

slow_points, J_eig = find_slow_points(lambda x: RNN_RHS(x, W, b, jax.nn.relu), x0)

with open("slow_points.txt", "wb") as fp:   #Pickling
    pickle.dump(slow_points, fp)
with open("J_eig.txt", "wb") as fp:   #Pickling
    pickle.dump(J_eig, fp)
"""
F = lambda x: RNN_RHS(x, W, b, jax.nn.relu)

df = pd.read_csv('activations2.csv')
df=df.drop(columns=['Unnamed: 0'])
activations=df.to_numpy()
#print('shape1:',activations.shape)
# Append this 256 dimensional data to include both the trajectory and the slow points, and do PCA altogether.
pre_pca = np.append(activations, x0)
#print('shape2:',pre_pca.shape)
pre_pca = np.append(pre_pca, slow_points)
#print('shape3:',pre_pca.shape)
pre_pca = np.reshape(pre_pca,(-1,256))

pca=PCA(n_components=2)
pca.fit(pre_pca)

pre_pca=pd.DataFrame(pre_pca)
result=pd.DataFrame(pca.transform(pre_pca), columns=['PCA%i' % i for i in range(2)], index=pre_pca.index)

trajectories=result[:1500].to_numpy()
x0=result[1500:1501].to_numpy()
slow_points=result[1501:].to_numpy()

fig, ax = plt.subplots()
ax0 = visualize_slow_points(fig, F, x0, trajectories, slow_points=slow_points, J_eig=J_eig, dim=None)
fig.savefig('toy_slowpoint.png', bbox_inches='tight', pad_inches=0)


F = lambda x: RNN_RHS(x, W, b, jax.nn.relu)

fig, ax = plt.subplots()

if slow_points is None:
    slow_points, J_eig = find_slow_points(F, x0)

# Marker styles
marker = {'fixed': 'X', 'saddle': '1'}
color = {'fixed': 'blue', 'saddle': 'red'}

#if x0[0].size > 3:
#    if dim is None:
#        dim = 3
#else:
#    dim = x0[0].size
dim = x0[0].size

if dim == 2:
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0

    for i in range(len(trajectories)):
        # Add on the trajectories
        ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], color='green', alpha=0.5)


    for i, slow_point in enumerate(slow_points):

        if slow_point[0] < xmin:
            xmin = slow_point[0]
        elif slow_point[0] > xmax:
            xmax = slow_point[0]

        if slow_point[1] < ymin:
            ymin = slow_point[1]
        elif slow_point[1] > ymax:
            ymax = slow_point[1]

        if np.all(J_eig[i][0] < 0):
            sp_type = 'fixed'
        else:
            sp_type = 'saddle'
        ax.scatter(slow_point[0], slow_point[1], marker=marker[sp_type], color=color[sp_type],
                   s=500)


    ax.set_xlim([xmin - 1e-1, xmax + 1e-1])
    ax.set_ylim([ymin - 1e-1, ymax + 1e-1])

elif dim == 3:
    pass


ax.set_xlabel(r'$x_1$', fontsize=16)
ax.set_ylabel(r'$x_2$', fontsize=16)

fig.savefig('toy_slowpoint.pdf', bbox_inches='tight', pad_inches=0)
"""
