from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold, cross_val_score


filename = 'dataset/falff_feas.txt'
data = np.loadtxt(filename, dtype=np.float32)
k_fold = KFold(n_splits=10)
train = []
test = []
accuracys = []
for train_indices, test_indices in k_fold.split(data):
    train = data[train_indices]
    test = data[test_indices]
    train_labels = train[:, 0].reshape(train.shape[0], 1)
    train_feas = train[:, 1:]
    test_labels = test[:, 0].reshape(test.shape[0], 1)
    test_feas = test[:, 1:]
    clf = svm.LinearSVC()
    re = clf.fit(train_feas, train_labels)
    predict = 0
    for i in range(test_feas.shape[0]):
        a = clf.predict([test_feas[i]])
        if a[0] == test_labels[i]:
            predict+=1
    accuracy = float(predict/test_feas.shape[0])
    accuracys.append(accuracy)
    print(accuracy)
print(np.mean(accuracys))

'''
dc: 0.5883333333333333
dcglobal:0.7061904761904761
reho:0.5528571428571428
falff:0.6221428571428571
'''