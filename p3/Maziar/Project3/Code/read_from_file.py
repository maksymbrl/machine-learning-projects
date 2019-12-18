import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Trying to set the seed
np.random.seed(0)
random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
population_path = cwd + '/../Data/ASV_table.tsv'
metadata_path = cwd + '/../Data/Metadata_table.tsv'
nanDict = {}
"""
df.columns (identifier)
df.values (population size)

population_size.shape -> (72, 14991)
"""
population_size = pd.read_csv(population_path, delimiter='\s+', encoding='utf-8')

population_size_std = [x.std() for x in population_size.values.T]

# 319 of the bioms have std > 100

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(population_size_std[:320], label='std')
ax.set_title('std values of population of each biom')
plt.show()

"""
df.columns (properties)
df.values (values)

metadata.shape -> (71, 41)
"""
metadata = pd.read_csv(metadata_path, delimiter='\s+', encoding='utf-8')
first, last = 3, -1
metadata_std = [x.std() for x in metadata.values.T[first: last]]
metadata_col = list(metadata.columns)

metadata_col_sorted = [x for _, x in sorted(zip(metadata_std, metadata_col))]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(metadata_std, label='std')
ax.set_title('std values of metadata of each feature')
plt.show()

corr_list = []

for i in metadata_col_sorted[:10]:
    corr_list.append([])
    for j in metadata_col_sorted[:10]:
        corr_list[-1].append(metadata.get(i).corr(metadata.get(j)))

heat_map = sb.heatmap(corr_list)
plt.show()

# The ones we should consider:

l = ["Latitude", "Longitude", "Altitude", "Area", "Depth", "Temperature", "Secchi", "O2", "CH4", "pH", "TIC", "SiO2", "KdPAR"]

toDrop = [x for x in metadata.columns if x not in l]
metadata = metadata.drop(toDrop, axis=1)

