from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston#波士顿房屋价格预测
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#集成学习ensemble库中的随机森林回归RandomForestRegressor

#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
#20个弱分类器，深度为4
scores = []
for i in range(X.shape[1]):#分别让每个特征与响应变量做模型分析并得到误差率
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     print(score)
     scores.append((round(np.mean(score), 3), names[i]))
print (sorted(scores, reverse=True))#对每个特征的分数排序