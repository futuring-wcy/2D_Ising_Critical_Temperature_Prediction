import numpy as np

# 创建一个示例的二维数组
array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])

# 使用 numpy.random.shuffle 方法打乱第二维
# 注意：numpy.random.shuffle 会就地修改数组，不返回新数组
# 因此，在此之前最好创建数组的副本
shuffled_array = array.copy()
np.random.shuffle(shuffled_array.T)  # 使用 .T 转置操作以打乱第二维

print("随机打乱第二维后的数组：")
print(shuffled_array)

# 使用 numpy.random.permutation 方法创建新数组并打乱第二维
permutation = np.random.permutation(array.shape[1])
shuffled_array = array[:, permutation]

print("随机打乱第二维后的数组：")
print(shuffled_array)
