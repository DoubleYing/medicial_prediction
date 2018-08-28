# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# 特征降维
def pca_feature(data):
    pca = PCA(n_components=500)
    pca_data = pca.fit_transform(data)
    print(np.shape(data), np.shape(pca_data))
    # pca_1.txt:n_components=None，默认保留204个特征 pca_500:n_components=500 pca_1000:n_components=1000
    return pca_data


# 数据预处理
def delete_feas(data):
    label = data[:, 0]
    negative = 0
    positive = 0
    # 计算每个类别的样本数
    for i in label:
        if i == -1:
            negative = negative + 1
        else:
            positive = positive + 1
    print("negative label:", negative)
    # output 204
    print("positive label:", positive)
    # output 204
    # 去除label，只保留特征
    data = np.delete(data, 0, axis=1)
    # 计算特征总数
    print("original feature number:", len(data))
    # output
    # 去除无效特征，即去除所有样本具有相同特征值的特征
    max_feature = np.max(data, axis=0)
    min_feature = np.min(data, axis=0)
    index = []
    for i, j, f in zip(max_feature, min_feature, range(len(max_feature))):
        if i == j:
            index.append(f)
    data = np.delete(data, index, axis=1)
    # 输出无效特征数和最终保留特征数
    print("invalid feature number:", len(index))
    print("valid feature number:", len(data[0]))

    np.savetxt("dataset/data.txt", data)
    np.savetxt("dataset/labels.txt", label)
    return label, data

def max_min_feature(data):
    scale = MinMaxScaler()
    data = scale.fit_transform(data)
    return data

def normalization(data):
    scale = StandardScaler()
    data = scale.fit_transform(data)
    return data


# 读取文件，并拼接所有特征
def read_file(file1, file2, file3, file4):
    data1 = np.loadtxt(file1, dtype=np.float32)
    data2 = np.loadtxt(file2, dtype=np.float32)
    data3 = np.loadtxt(file3, dtype=np.float32)
    data4 = np.loadtxt(file4, dtype=np.float32)
    data = np.concatenate((data1, data2[:, 1:], data3[:, 1:], data4[:, 1:]), axis=-1)
    return data


def train(feature, label):
    label = transport_labels(label)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, random_state=1000)
    xgb_train = xgb.DMatrix(X_train, label=Y_train)
    xgb_test = xgb.DMatrix(X_test, label=Y_test)
    # xgboost参数
    params = {
        'booster': 'gbtree',
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        # 'nthread':7,# cpu 线程数 默认最大
        'eta': 0.07,  # 如同学习率
        'min_child_weight': 6,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'max_depth': 3,  # 构建树的深度，越大越容易过拟合
        'gamma': 0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        # 'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'reg_alpha': 0.1,  # L1 正则项参数
        # 'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
        'objective': 'binary:logistic',  # logitraw',  # 二分类的逻辑回归问题，输出的结果为wTx
        # 'num_class':10, # 类别数，多分类与 multisoftmax 并用
        'seed': 1000,  # 随机种子
        'eval_metric': 'error'
    }

    plst = list(params.items())
    num_rounds = 10000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    # model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=1000)  # , pred_margin=1
    # 交叉验证
    cv_model = xgb.cv(plst, xgb_train, num_rounds, metrics={"error"}, nfold=5,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(1000)])
    num_boost_rounds = len(cv_model)
    model = xgb.train(dict(plst, silent=1), xgb_train, evals=watchlist, num_boost_round=num_boost_rounds)
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    # importance = xgb.plot_importance(model)
    # print(importance)
    # plt.show()
    importance = model.get_fscore()
    feature_num = [i for i in range(len(importance))]
    print(feature_num)
    print(importance)
    plt.bar(feature_num, importance.values())
    plt.show()
    print(type(importance))
    x1 = cv_model[['train-error-mean']]
    x2 = cv_model[['test-error-mean']]
    y = [i for i in range(num_boost_rounds)]
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title('训练集和验证集的错误率-迭代次数曲线图')
    plt.plot(y, x1, color='blue', label='训练集')
    plt.plot(y, x2, color='green', label='验证集')
    plt.xlabel('迭代次数')
    plt.ylabel('平均错误率')
    plt.legend()
    plt.show()

    # model.save_model('./model/xgb.model') # 用于存储训练出的模型
    # print("best best_ntree_limit", model.best_ntree_limit)
    # 预测数据
    y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    # 输出test的平均准确率
    print('accs=%f' % (sum(1 for i in range(len(y_pred)) if int(y_pred[i] >= 0.5) == Y_test[i]) / float(len(y_pred))))


# xgb参数搜索
def xgboost_grid_search(feature, label):
    param_name = 'learning_rate'
    xgboost = xgb.XGBClassifier(
                                max_depth=3,
                                # learning_rate=0.07,
                                n_estimators=35,
                                silent=True,
                                objective="binary:logistic",
                                booster='gbtree',
                                gamma=0,
                                min_child_weight=6,
                                subsample=0.8,
                                colsample_bytree=0.7,
                                reg_alpha=0.1,
                                seed=1000)
    parameters = {
        # param_name: [0.001, 0.01, 0.1, 0.3, 0.5],
        param_name: list(np.arange(0.01,0.1,0.01))
    }
    label = transport_labels(label)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, random_state=1000)
    clf = GridSearchCV(xgboost, parameters, scoring='average_precision', cv=5)
    clf.fit(X_train, Y_train)
    print(clf.best_params_)
    # print(clf.cv_results_)
    draw_grid_scores(clf.cv_results_, [i[param_name] for i in clf.cv_results_['params']], title='xgboost参数调优', x_name='param-'+param_name,
                     y_name='score')
    # output: {'C': 13, 'gamma': 1e-05, 'kernel': 'sigmoid'}
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    print(y_pred, Y_test)
    print('accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))

# svm参数搜索
def svm_grid_search(feature, label):
    svc = SVC(kernel='sigmoid')
    parameters = {
            # 'C': [0.001, 0.01, 0.1, 10, 50, 75, 100, 125, 150],
            'C': list(range(50, 70, 2))
            # 'C': list(range(130, 150)),
        }

    label = transport_labels(label)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, random_state=1000)
    clf = GridSearchCV(svc, parameters, scoring='accuracy', cv=5)
    clf.fit(X_train, Y_train)
    print(clf.best_params_)
    draw_grid_scores(clf.cv_results_, [i['C'] for i in clf.cv_results_['params']], title='svm参数调优', x_name='param-C', y_name='score')
    # output: {'C': 13, 'gamma': 1e-05, 'kernel': 'sigmoid'}
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    print(y_pred, Y_test)
    print('accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))

# 画grid search
def draw_grid_scores(grid_scores, grid_names, title, x_name, y_name):
    std_test_score = grid_scores['std_test_score']
    mean_test_score = grid_scores['mean_test_score']
    std_train_score = grid_scores['std_train_score']
    mean_train_score = grid_scores['mean_train_score']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(title)
    plt.plot(grid_names, mean_test_score, color='blue', label='mean_test_score')
    plt.plot(grid_names, mean_train_score, color='green', label='mean_train_score')
    plt.plot(grid_names, std_test_score, color='red', label='std_test_score')
    plt.plot(grid_names, std_train_score, color='black', label='std_train_score')
    plt.scatter(grid_names, mean_test_score, color='blue')
    plt.scatter(grid_names, mean_train_score, color='green')
    plt.scatter(grid_names, std_train_score, color='red')
    plt.scatter(grid_names, std_train_score, color='black')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(loc='lower right')
    plt.show()


def emsembal_train(feature, label):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from mlxtend.classifier import EnsembleVoteClassifier,StackingClassifier
    label = transport_labels(label)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, random_state=1000)
    xgb_train = xgb.DMatrix(X_train, label=Y_train)
    xgb_test = xgb.DMatrix(X_test, label=Y_test)
    clf1 = SVC(C=60, kernel='sigmoid',probability=True)
    clf2 = RandomForestClassifier(random_state=0)
    clf3 = LogisticRegression(random_state=0)
    clf4 = xgb.XGBClassifier(max_depth=8, learning_rate=0.07,
                 n_estimators=35, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 gamma=0, min_child_weight=6,
                 subsample=0.8, colsample_bytree=0.7,
                 reg_alpha=0.1, seed=1000)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf3, clf4], voting='soft')

    eclf.fit(X_train, Y_train)
    y_pred = eclf.predict(X_test)
    # print(y_pred, Y_test)
    print('eclf accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))

    # sclf = StackingClassifier(classifiers=[clf1, clf4, clf3],
    #                           use_probas=True,
    #                           average_probas=False,
    #                           meta_classifier=clf3)
    # sclf.fit(X_train, Y_train)
    # y_pred = sclf.predict(X_test)
    # print('sclf accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))


    clf1.fit(X_train, Y_train)
    y_pred = clf1.predict(X_test)
    print('svm accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))

    # clf3.fit(X_train, Y_train)
    # y_pred = clf3.predict(X_test)
    # print('lr accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))

    # clf4.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], eval_metric='error')
    # y_pred = clf4.predict(X_test)
    # print('xgb accs=%f' % (sum(1 for i in range(len(y_pred)) if y_pred[i] == Y_test[i]) / float(len(y_pred))))


def transport_labels(labels):
    res = []
    for i in labels:
        res.append(i if i==1 else 0)
    return np.array(res)

if __name__ == '__main__':
    file1 = "dataset/dc_feas.txt"
    file2 = "dataset/dcglobal_feas.txt"
    file3 = "dataset/falff_feas.txt"
    file4 = "dataset/reho_feas.txt"
    file5 = "dataset/data.txt"
    file6 = 'dataset/max_min_data.txt'
    file7 = 'dataset/std_data.txt'
    file8 = 'dataset/normal_data.txt'
    file9 = 'dataset/max_min_pca.txt'
    file10 = 'dataset/std_pca.txt'
    file11 = 'dataset/normal_pca.txt'

    # file_pca1 = "dataset/dc_pca_500.txt"
    # file_pca2 = "dataset/dcglobal_pca_500.txt"
    # file_pca3 = "dataset/falff_pca_500.txt"
    # file_pca4 = "dataset/reho_pca_500.txt"
    #
    # data1 = np.loadtxt(file_pca1, dtype=np.float32)
    # data2 = np.loadtxt(file_pca2, dtype=np.float32)
    # data3 = np.loadtxt(file_pca3, dtype=np.float32)
    # data4 = np.loadtxt(file_pca4, dtype=np.float32)
    # feature = data4
    # feature = np.concatenate((data1, data2, data3, data4), axis=-1)

    # 给每个文件归一化后降维
    # data = np.loadtxt(file4)
    # labels, data = max_min_feature(data)
    # pca_feature(data)

    # 预处理，删除过于特征
    # read_file(file1, file2,file3,file4)
    # Y, X = delete_feas(read_file(file1, file2, file3, file4))

    # 归一化处理
    # data = np.loadtxt(file5)
    # max_min_data = max_min_feature(data)
    # np.savetxt('max_min_data.txt', max_min_data)
    # std_data = normalization(data)
    # np.savetxt('std_data.txt', std_data)
    # normal_data = normalization(max_min_data)
    # np.savetxt('normal_data.txt', normal_data)

    # 降维pca
    # data = np.loadtxt(file6)
    # max_min_pca = pca_feature(data)
    # np.savetxt("max_min_pca.txt", max_min_pca)
    # data = np.loadtxt(file7)
    # std_pca = pca_feature(data)
    # np.savetxt("std_pca.txt", std_pca)
    # data = np.loadtxt(file8)
    # normal_pca = pca_feature(data)
    # np.savetxt("normal_pca.txt", normal_pca)

    # 载入3个预处理的文件（file9.file10.file11）中的一个
    feature = np.loadtxt(file11)
    label = np.loadtxt("dataset/labels.txt")
    print(np.shape(feature))

    # ensembal训练
    emsembal_train(feature, label)
    # svmc参数调优
    # svm_grid_search(feature, label)

    # xgboost调优
    # xgboost_grid_search(feature, label)

    # xgboost训练
    # train(feature, label)

