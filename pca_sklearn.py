print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


filename = 'dataset/dcglobal_feas.txt'
data = np.loadtxt(filename, dtype=np.float32)
y = data[:, 0]
X = data[:, 1:]

n_components = [200, 2000, 5000]
for i in n_components:
    pca = PCA(n_components=i)
    X_r = pca.fit(X).transform(X)

    pca_variance_ratio = sum(pca.explained_variance_ratio_)

    print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
    print('sum:' % pca_variance_ratio)

