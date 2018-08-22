from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold, cross_val_score


dc_feas_filename = 'dataset/dc_feas.txt'
dcglobal_feas_filename = 'dataset/dcglobal_feas.txt'
reho_feas_filename = 'dataset/reho_feas.txt'
falff_feas_filename = 'dataset/falff_feas.txt'
dc_feas_data = np.loadtxt(dc_feas_filename, dtype=np.float32)
dcglobal_feas_data = np.loadtxt(dcglobal_feas_filename, dtype=np.float32)
reho_feas_data = np.loadtxt(reho_feas_filename, dtype=np.float32)
falff_feas_data = np.loadtxt(falff_feas_filename, dtype=np.float32)
print(dcglobal_feas_data.shape)
# exit()
data = np.zeros([dcglobal_feas_data.shape[0], dcglobal_feas_data.shape[1], 4], dtype=float,order='C')
data[:,:,0] = dc_feas_data
data[:,:,1] = dcglobal_feas_data
data[:,:,2] = reho_feas_data
data[:,:,3] = falff_feas_data
print('load data end...')

k_fold = KFold(n_splits=10)
train = []
test = []
accuracys = []
for train_indices, test_indices in k_fold.split(data):
    train = data[train_indices]
    test = data[test_indices]

    train_labels = train[:, 0, 0].reshape(train.shape[0], 1, 1)
    train_feas = train[:, 1:, :]
    test_labels = test[:, 0, 0].reshape(test.shape[0], 1, 1)
    test_feas = test[:, 1:, :]
    clf = svm.LinearSVC()
    re = clf.fit(train_feas, train_labels)
    print('svm_multi fit end')
    predict = 0
    for i in range(test_feas.shape[0]):
        a = clf.predict([test_feas[i]])
        if a[0] == test_labels[i]:
            predict+=1
    accuracy = float(predict/test_feas.shape[0])
    accuracys.append(accuracy)
    print(accuracy)
    exit()
print(np.mean(accuracys))

