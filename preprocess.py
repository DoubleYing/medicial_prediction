import numpy as np
from sklearn import preprocessing

test = np.loadtxt('dataset/test.txt')
test_feas = test[:, 1:]
test_labels = test[:, 0]
test_feas_scaled = preprocessing.scale(test_feas)
print(test_feas_scaled)