from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import normalize, MinMaxScaler
import xgboost as xgb
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def xgboost_sklearn(X_train, X_test, y_train, y_test):
    start_time = time.time()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  ##test_size测试集合所占比例
    # xgb矩阵赋值
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)
    ##参数
    params = {
        'booster': 'gbtree',
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        # 'nthread':7,# cpu 线程数 默认最大
        'eta': 0.1,  # 如同学习率
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'max_depth': 6,  # 构建树的深度，越大越容易过拟合
        'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        # 'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'alpha':0.1, # L1 正则项参数
        # 'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
        'objective': 'binary:logistic', #二分类的逻辑回归问题，输出的结果为wTx
        # 'num_class':10, # 类别数，多分类与 multisoftmax 并用
        'seed': 1000,  # 随机种子
        'eval_metric': 'error'
    }
    plst = list(params.items())
    num_rounds = 10000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]

    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    cv_model = xgb.cv(plst, xgb_train, num_rounds, metrics={"error"}, nfold=5,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(1000)])
    num_boost_rounds = len(cv_model)
    model = xgb.train(plst, xgb_train, evals=watchlist, num_boost_round=num_boost_rounds)
    # model.save_model('./model/xgb.model') # 用于存储训练出的模型

    # 画图
    draw_feature_importance(model)
    draw_cv_error(cv_model, num_boost_rounds)

    print("best best_ntree_limit", model.best_ntree_limit)
    y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

    y_pred = trans_y_pred(y_pred)
    aucc = cal_aucc(y_pred, y_test)
    print ('aucc=%f' % aucc)
    # 输出运行时长
    cost_time = time.time() - start_time
    print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")
    return model.best_ntree_limit, aucc, cost_time, y_pred


def draw_feature_importance(model):
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    importance = model.get_fscore()
    feature_num = [i for i in range(len(importance))]
    print(feature_num)
    print(importance)
    plt.bar(feature_num, importance.values())
    plt.show()
    print(type(importance))


def draw_cv_error(cv_model, num_boost_rounds):
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

def multi_xgboost():
    filename1 = 'dataset/reho_feas.txt_203_pca.txt'
    filename2 = 'dataset/falff_feas.txt_203_pca.txt'
    filename3 = 'dataset/dc_feas.txt_302_pca.txt'
    filename4 = 'dataset/dcglobal_feas.txt_203_pca.txt'
    labels_fliename = 'dataset/labels.txt'
    X1 = np.loadtxt(filename1, dtype=np.float32)
    X2 = np.loadtxt(filename2, dtype=np.float32)
    X3 = np.loadtxt(filename3, dtype=np.float32)
    X4 = np.loadtxt(filename4, dtype=np.float32)
    y = np.loadtxt(labels_fliename, dtype=np.float32)
    y = transport_labels(y)
    X = np.hstack((X1, X2, X3, X4))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    xgboost_sklearn(X_train, X_test, y_train, y_test)

def multi_xgboost_pca(data, n_split = 10):
    # 归一化
    X = normalize(data[:, 1:], norm='l2')
    X_neg = X[0:102,:]
    X_pos = X[102:, :]
    y = data[:, 0]
    y_neg = y[0:102]
    y_pos = y[102:]

    k_fold = KFold(n_splits=n_split)
    re_auccs = []
    for train_indices, test_indices in k_fold.split(y_neg):
        y_train = np.hstack((y_neg[train_indices], y_pos[train_indices]))
        y_test = np.hstack((y_neg[test_indices], y_pos[test_indices]))

        X_train = np.vstack((X_neg[train_indices], X_pos[train_indices]))
        X_test = np.vstack((X_neg[test_indices], X_pos[test_indices]))
        best_ntree_limit, aucc, cost_time, y_pred = xgboost_sklearn(X_train, X_test, y_train, y_test)

        re_auccs.append(aucc)
    re_auccs.append(sum(re_auccs) / len(re_auccs))
    re_auccs = np.array(re_auccs)
    np.savetxt('output/four_feas_merge_auccs.txt', re_auccs)
    return

def multi_xgboost_pca2(data, n_split = 10):
    # 归一化
    # min_max_scaler = MinMaxScaler()
    # X = min_max_scaler.fit_transform(data[:, 1:])
    X = data[:, 1:]
    y = data[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

    best_ntree_limit, aucc, cost_time, y_pred = xgboost_sklearn(X_train, X_test, y_train, y_test)
    print(best_ntree_limit, aucc, cost_time, y_pred)
    return



def vote_xgboost_pca(X1, X2, X3, X4, y, data, n_split = 6):
    samp_num, feas_num = X1.shape

    # 归一化
    Xs = []
    for i in range(4):
        Xs.append(normalize(data[:, 1+i*feas_num:1+(i+1)*feas_num], norm='l2'))

    # 划分数据集
    k_fold = KFold(n_splits=n_split)
    re_auccs = []
    for train_indices, test_indices in k_fold.split(y):
        y_train = y[train_indices]
        y_test = y[test_indices]
        y_preds = []
        auccs = []
        for i in range(len(Xs)):
            X_train = Xs[i][train_indices]
            X_test = Xs[i][test_indices]
            best_ntree_limit, aucc, cost_time, y_pred = xgboost_sklearn(X_train, X_test, y_train, y_test)
            y_preds.append(y_pred)
            auccs.append(aucc)
        re = vote(np.array(y_preds))
        re_aucc = cal_aucc(re, y_test)
        re_auccs.append(re_aucc)
    re_auccs.append(sum(re_auccs)/len(re_auccs))
    re_auccs = np.array(re_auccs)
    np.savetxt('output/four_feas_sep_auccs.txt', re_auccs)
    return



def vote_xgboost():
    filename1 = 'dataset/reho_feas.txt'
    filename2 = 'dataset/falff_feas.txt'
    filename3 = 'dataset/dc_feas.txt'
    filename4 = 'dataset/dcglobal_feas.txt'
    labels_fliename = 'dataset/labels.txt'
    X1 = np.loadtxt(filename1, dtype=np.float32)
    X2 = np.loadtxt(filename2, dtype=np.float32)
    X3 = np.loadtxt(filename3, dtype=np.float32)
    X4 = np.loadtxt(filename4, dtype=np.float32)
    y = np.loadtxt(labels_fliename, dtype=np.float32)
    # vote_multi_xgboost(X1[:, 1:], X2[:, 1:], X3[:, 1:], X4[:, 1:], y)


# 由多个特征投票决定
def vote(y_preds):
    fea_num, data_num = y_preds.shape
    vote_ys = sum(y_preds)

    for i in range(data_num):
        if vote_ys[i] > fea_num/2:
            vote_ys[i] = 1
        elif vote_ys[i] < fea_num/2:
            vote_ys[i] = 0
        else:
            vote_ys[i] = np.random.randint(0,2)
    return vote_ys


# 转换labels，从-1到0
def transport_labels(labels):
    res = []
    for i in labels:
        res.append(i if i==1 else 0)
    return np.array(res)

# 计算准确率
def cal_aucc(y_pred, y_test):
    aucc = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_test[i]) / float(len(y_pred))
    return aucc

# 将y_pred转换为0，1
def trans_y_pred(y_pred):
    y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])
    return y_pred

# 读取pca后的特征文件
def read_pca_feas_file():
    filename1 = 'dataset/reho_feas.txt_203_pca.txt'
    filename2 = 'dataset/falff_feas.txt_203_pca.txt'
    filename3 = 'dataset/dc_feas.txt_203_pca.txt'
    filename4 = 'dataset/dcglobal_feas.txt_203_pca.txt'
    labels_fliename = 'dataset/labels.txt'
    X1 = np.loadtxt(filename1, dtype=np.float32)
    X2 = np.loadtxt(filename2, dtype=np.float32)
    X3 = np.loadtxt(filename3, dtype=np.float32)
    X4 = np.loadtxt(filename4, dtype=np.float32)
    y = np.loadtxt(labels_fliename, dtype=np.float32)
    y = transport_labels(y)
    return X1, X2, X3, X4, y


# 读取完整的特征文件
def read_full_feas_file():
    filename1 = 'dataset/reho_feas.txt'
    filename2 = 'dataset/falff_feas.txt'
    filename3 = 'dataset/dc_feas.txt'
    filename4 = 'dataset/dcglobal_feas.txt'
    labels_fliename = 'dataset/labels.txt'
    X1 = np.loadtxt(filename1, dtype=np.float32)
    X2 = np.loadtxt(filename2, dtype=np.float32)
    X3 = np.loadtxt(filename3, dtype=np.float32)
    X4 = np.loadtxt(filename4, dtype=np.float32)
    y = np.loadtxt(labels_fliename, dtype=np.float32)
    y = transport_labels(y)
    return X1[:, 1:], X2[:, 1:], X3[:, 1:], X4[:, 1:], y



if __name__ == '__main__':
    n_split = 10

    #
    # X = np.loadtxt('dataset/pca_1.txt')
    # y = np.loadtxt('dataset/labels.txt')
    # y = transport_labels(y)
    # data = np.hstack((y.reshape(-1, 1), X))
    #

    # 读取数据文件
    X1, X2, X3, X4, y = read_pca_feas_file()
    # X1, X2, X3, X4, y = read_full_feas_file()

    # 联合
    data = np.hstack((y.reshape(-1, 1), X1, X2, X3, X4))

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(data[:, 1:])

    data = np.hstack((data[:, 0].reshape(-1,1), X))

    multi_xgboost_pca2(data, n_split)

# dcglobal_500: aucc:0.561976  error: 0.439024
# dc_500: aucc:0.561976  error: 0.439024
# falff_500: aucc:0.561976  error: 0.439024
# reho_500: aucc:0.561976  error: 0.439024
# multi: aucc:0.561976  error: 0.439024
