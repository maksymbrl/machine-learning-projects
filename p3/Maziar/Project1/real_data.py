# -*- coding: utf-8 -*-
from helper_functions import *
from regression_class import *

from imageio import imread
from mpl_toolkits.mplot3d import Axes3D

# TODO: remove later
import numpy as np

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

def init_ml(terrain, row_size, col_size):
    row_len, col_len = np.shape(terrain)
    
    row = np.linspace(0, 1, row_size)
    col = np.linspace(0, 1, col_size)
    
    colmat, rowmat = np.meshgrid(col, row)
    
    z = terrain[:row_size, :col_size]
    
    row_arr = rowmat.ravel()
    col_arr = colmat.ravel()
    z_arr = z.ravel()
    
    return row_arr, col_arr, z_arr

test_size_row = 500
test_size_col = 200

degree = 5

x, y , z = init_ml(terrain1, test_size_row, test_size_col)
row_len, col_len = len(x), len(y)
X = design_poly_matrix(x, y, degree)

ols = OLSClass (x, y, z, degree)
X = design_poly_matrix(x, y, degree)
#beta1 = ols.Ridge(X, 0.1, z)
beta2 = ols.Lasso(X, 0.01, z)
row_len, col_len = np.shape(terrain1)
    
row = np.linspace(0, 1, test_size_row)
col = np.linspace(0, 1, test_size_col)

colmat, rowmat = np.meshgrid(col, row)

#z_predict = ols.predict(X, 'Ridge')#
z_predict2 = ols.predict(X, 'Lasso')
z_p_reshaped = z_predict.reshape(test_size_row, test_size_col)
z_p_reshaped2 = z_predict2.reshape(test_size_row, test_size_col)



fig = plt.figure()
    
ax = fig.add_subplot(1, 3, 1, projection='3d')
surf = ax.plot_surface(colmat, rowmat, terrain1[:test_size_row, :test_size_col], cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Original')

#ax = fig.add_subplot(1, 3, 2, projection='3d')
#surf = ax.plot_surface(colmat, rowmat, z_p_reshaped, cmap=cm.viridis, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Maziar')

ax = fig.add_subplot(1, 3, 3, projection='3d')
surf = ax.plot_surface(colmat, rowmat, z_p_reshaped2, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Maziar')