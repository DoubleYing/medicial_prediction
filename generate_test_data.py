import numpy as np
# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(204, 100)) # 随机输入
y_data = np.random.randint(0, 2, (204, 1))
print(x_data.shape, y_data.shape)
x_data = np.c_[y_data, x_data]
print(x_data.shape, y_data.shape)
np.savetxt('dataset/test.txt', x_data)