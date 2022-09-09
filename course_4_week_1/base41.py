import numpy as np

import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
        把数据集X的图像边界全部使用0来扩充pad个宽度和高度。

        参数：
            X - 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
            pad - 整数，每个图像在垂直和水平维度上的填充量
        返回：
            X_paded - 扩充后的图像数据集，维度为（样本数，图像高度 + 2*pad，图像宽度 + 2*pad，图像通道数）

    """
    X_paded = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_paded


def conv_single_step(a_slice_prev, W, b):
    """
        在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
        这里切片大小和过滤器大小相同

        参数：
            a_slice_prev - 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
            W - 权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
            b - 偏置参数，包含在了一个矩阵中，维度为（1,1,1）

        返回：
            Z - 在输入数据的片X上卷积滑动窗口（w，b）的结果。
    """

    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
        实现卷积函数的前向传播

        参数：
            A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
            W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
            b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
            hparameters - 包含了"stride"与 "pad"的超参数字典。

        返回：
            Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
            cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)



