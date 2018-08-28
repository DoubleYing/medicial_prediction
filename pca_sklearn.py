print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# n_components=0.9999999999 must be between 1 and n_features=247151 with svd_solver='arpack'
# n_components=0.9999999999 must be between 1 and n_features=247151 with svd_solver='randomized'
# n_components='mle' is only supported if n_samples >= n_features
def file_pca_mle(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    X = data[:, 1:]
    pca = PCA(n_components='mle', svd_solver='full')
    X_r = pca.fit(X).transform(X)
    print(X)
    print(X_r)
    explained_variance_ratio = sum(pca.explained_variance_ratio_)
    print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))
    print('sum of explained variance ratio:', explained_variance_ratio)
    n_component = len(pca.explained_variance_ratio_)
    print('best n component is=', n_component)

    out_ratio_filename = filename + '_' + str(n_component) + '_pca_ratio.txt'
    np.savetxt(out_ratio_filename, pca.explained_variance_ratio_)

    out_filename = filename + '_' + str(n_component) + '_pca.txt'
    np.savetxt(out_filename, X_r)

def file_pca(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    X = data[:, 1:]
    pca = PCA(n_components=0.9999999999)
    X_r = pca.fit(X).transform(X)
    print(X)
    print(X_r)
    explained_variance_ratio = sum(pca.explained_variance_ratio_)
    print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))
    print('sum of explained variance ratio:', explained_variance_ratio)
    n_component = len(pca.explained_variance_ratio_)
    print('best n component is=', n_component)

    out_ratio_filename = filename + '_' + str(n_component) + '_pca_ratio.txt'
    np.savetxt(out_ratio_filename, pca.explained_variance_ratio_)

    out_filename = filename + '_' + str(n_component) + '_pca.txt'
    np.savetxt(out_filename, X_r)

def file_pca_feas_num(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    X = data[:, 1:]

    # n_components = [200, 500, 1000, 1500, 2000]
    n_components = [10, 50, 100]
    ratios = []
    for i in n_components:
        pca = PCA(n_components=i)
        X_r = pca.fit(X).transform(X)

        pca_variance_ratio = sum(pca.explained_variance_ratio_)

        print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
        print('sum:' , pca_variance_ratio)
        out_ratio_filename = filename + '_' + str(i) + '_pca_ratio.txt'
        np.savetxt(out_ratio_filename, pca.explained_variance_ratio_)
        out_filename = filename+'_'+str(i)+'_pca.txt'
        np.savetxt(out_filename, X_r)
        ratios.append(pca_variance_ratio)
    ratios = np.array(ratios)
    np.savetxt(filename+'pca_ratios.txt', ratios)

# filenames = ['dataset/dc_feas.txt']
filenames = ['dataset/dcglobal_feas.txt', 'dataset/falff_feas.txt', 'dataset/reho_feas.txt']
for filename in filenames:
    file_pca(filename)