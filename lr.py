from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


filename = 'dataset/reho_feas.txt'
data = np.loadtxt(filename, dtype=np.float32)
k_fold = KFold(n_splits=10)
train = []
test = []
accuracys = []
threshold = 0
for train_indices, test_indices in k_fold.split(data):
    train = data[train_indices]
    test = data[test_indices]
    train_labels = train[:, 0].reshape(train.shape[0], 1)
    train_feas = train[:, 1:]
    test_labels = test[:, 0].reshape(test.shape[0], 1)
    test_feas = test[:, 1:]

    # clf = svm.SVR()
    clf = LinearRegression()
    re = clf.fit(train_feas, train_labels)
    predict = 0
    y = clf.predict(test_feas)
    for i in range(test_feas.shape[0]):
        if y[i][0] <= threshold:
            y[i][0] = -1
        else:
            y[i][0] = 1
        if y[i][0]  == test_labels[i]:
            predict+=1

    # print(test_feas[:, 1].shape, y.shape)
    # plt.show()
    accuracy = float(predict/test_feas.shape[0])
    accuracys.append(accuracy)
    print(accuracy)
print(np.mean(accuracys))


'''
k-fold=10
dc: 0.574047619047619
dcglobal:0.666904761904762
falff:0.628095238095238
reho:0.5973809523809523
'''