import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import scale
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split

class DataProcessing:

    def __init__(self):
        self.url                 = "https://github.com/laurent0001/tsv/raw/master/CDOM.gradient.mat.tsv"
        self.url_mesh            = "https://github.com/laurent0001/tsv/raw/master/Bray_distances_by_CDOM_gradient_meshgrid.tsv"
        self.url_CDOM_diag_mesh  = "https://github.com/laurent0001/tsv/blob/master/CDOM.diag.mesh.tsv?raw=true"
        self.url_CDOM_sorted     = "https://github.com/laurent0001/tsv/raw/master/CDOM.tsv"
        # Load bacterial community data
        self.url_ASV             = "https://github.com/laurent0001/Project-3/blob/master/ASV_table.tsv?raw=true"
        self.url_ASV_ranged      = "https://github.com/laurent0001/Project-3/blob/master/ASV_table_ranged.tsv?raw=true"
        self.url_metadata        = "https://github.com/laurent0001/Project-3/raw/master/Metadata_table.tsv?raw=true"
        self.url_metadata_scaled = "https://github.com/laurent0001/Project-3/raw/master/Metadata_table_scaled.tsv?raw=true"

    def GetMainData(self, *args):
        '''
        Data Preparation step
        '''
        CDOM = pd.read_csv(self.url, sep="\t") #BCC pairwise distances with CDOM values for both sites for each row
        #CDOM_sites = pd.read_csv(url_sites, sep="\t") #Sites matching the order of BCC pairwise distances with CDOM values of both sites for each row
        CDOM.mesh = pd.read_csv(self.url_mesh, sep="\t")
        CDOM_diag_mesh = pd.read_csv(self.url_CDOM_diag_mesh, sep="\t")
        CDOM_sorted = pd.read_csv(self.url_CDOM_sorted, sep="\t")
        CDOM_diag_mesh.columns = ["CDOM.x1", "CDOM.x2", "CDOM.mid"]

        ASV = pd.read_csv(self.url_ASV, sep="\t")
        ASV_ranged = pd.read_csv(self.url_ASV_ranged, sep="\t")
        metadata = pd.read_csv(self.url_metadata, sep="\t")
        metadata_scaled = pd.read_csv(self.url_metadata_scaled, sep="\t")
        X_ASV = ASV_ranged
        X_ASV.columns = [''] * len(X_ASV.columns)
        X_ASV = X_ASV.to_numpy()
        #y_CDOM = metadata.iloc[:, 27][:, np.newaxis]

        # split data into train and test sets
        y_CDOM = metadata.iloc[:, 27] #Requires 1d array

        data = CDOM, CDOM_sorted, CDOM_diag_mesh, \
               ASV, ASV_ranged, \
               metadata, metadata_scaled, \
               X_ASV, y_CDOM
        return data
