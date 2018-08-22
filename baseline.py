import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score

def load_data_indices(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    k_fold = KFold(n_splits=10)
    train = []
    test = []
    for train_indices, test_indices in k_fold.split(data):
        train.append(train_indices)
        test.append(test_indices)
    return train, test, data

def transform_labels(labels):
    re = []
    for i in labels:
        if i[0] == 1:
            re.append([0,1])
        else:
            re.append([1,0])
    return np.array(re)


if __name__ == '__main__':
    filename = 'dataset/test.txt'
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
        # train_labels = transform_labels(train_labels)
        test_labels = test[:, 0].reshape(test.shape[0], 1)
        test_feas = test[:, 1:]
        # test_labels = np.array([[0, 1] if i[0] == 0 else [0, 1] for i in test_labels])
        # test_labels = transform_labels(test_labels)


        # 定义线性模型
        x = tf.placeholder(tf.float32, [None, train_feas.shape[1]])
        y_ = tf.placeholder(tf.float32, [None, 1])
        b = tf.Variable(tf.random_uniform([2]))
        W = tf.Variable(tf.random_uniform([(train_feas.shape)[1], 2], -1.0, 1.0))

        z = tf.matmul(x, W) + b
        y = tf.nn.softmax(z)

        # 定义损失函数
        loss = tf.reduce_mean(tf.square(y - y_))
        optimizer = tf.train.AdamOptimizer(0.001)
        train = optimizer.minimize(loss)

        # 初始化所以变量
        init = tf.initialize_all_variables()

        # 启动图
        sess = tf.Session()
        sess.run(init)

        # 拟合平面
        for step in range(2001):
            sess.run(train, feed_dict={x: train_feas, y_: train_labels})
            if step % 20 == 0:
                total_cross_entropy = sess.run(loss, feed_dict={x: train_feas, y_: train_labels})
                print("loss: ", total_cross_entropy)

        # 评估模型
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        # correct_prediction = 1 if (y <= 0 and y_ == 0) or (y > 0 and y == 1) else 0
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        accuracys.append(accuracy)
        # print("accuracy train", sess.run(accuracy, feed_dict={x: train_feas, y_: train_labels}))
        print("accuracy test", sess.run(accuracy, feed_dict={x: test_feas, y_: test_labels}))
        # exit()
print(float(np.sum(accuracys)/10))

