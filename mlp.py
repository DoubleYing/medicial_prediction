from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

filename = 'dataset/reho_feas.txt'
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
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, activation='tanh')
    re = clf.fit(train_feas, train_labels)
    y = clf.predict(test_feas)
    predict = 0
    for i in range(test_feas.shape[0]):
        if y[i] == test_labels[i]:
            predict+=1
    accuracy = float(predict/test_feas.shape[0])
    accuracys.append(accuracy)
    print(accuracy)
print(np.mean(accuracys))

'''
dc: 0.45738095238095233
dcglobal:0.4969047619047619
reho:
falff:
'''