#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Part1():
    def basic_sigmod(self,x):
        """
        Compute sigmoid of x.
        Arguments:
        x -- A scalar
        Return:
        s -- sigmoid(x)
        """
        s = 1.0 / (1 + 1 / np.exp(x))

        return s

    # GRADED FUNCTION: sigmoid_derivative
    def sigmoid_derivative(self, x):
        """
        Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
        You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

        Arguments:
        x -- A scalar or numpy array

        Return:
        ds -- Your computed gradient.
        """

        s = 1.0 / (1 + 1 / np.exp(x))
        ds = s * (1 - s)

        return ds

    # GRADED FUNCTION: image2vector
    def image2vector(self, image):
        """
        Argument:
        image -- a numpy array of shape (length, height, depth)

        Returns:
        v -- a vector of shape (length*height*depth, 1)
        """

        v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

        return v

    # GRADED FUNCTION: normalizeRows
    def normalizeRows(self, x):
        """
        Implement a function that normalizes each row of the matrix x (to have unit length).

        Argument:
        x -- A numpy matrix of shape (n, m)

        Returns:
        x -- The normalized (by row) numpy matrix. You are allowed to modify x.
        """

        # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)  #计算单位向量的长度

        # Divide x by its norm.
        x = x / x_norm  #将x向量单位化，用于numpy的广播

        return x

    # GRADED FUNCTION: softmax
    def softmax(self, x):
        """Calculates the softmax for each row of the input x.

        Your code should work for a row vector and also for matrices of shape (n, m).

        Argument:
        x -- A numpy matrix of shape (n,m)

        Returns:
        s -- A numpy matrix equal to the softmax of x, of shape (n,m)
        """

        # Apply exp() element-wise to x. Use np.exp(...).
        x_exp = np.exp(-1.0 * x)
        # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
        s = x_exp / x_sum
        return s

    # GRADED FUNCTION: L1
    def L1(self, yhat, y):
        """
        Arguments:
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)

        Returns:
        loss -- the value of the L1 loss function defined above
        """

        ### START CODE HERE ### (≈ 1 line of code)
        loss = np.sum(np.abs(yhat - y))
        ### END CODE HERE ###

        return loss

    # GRADED FUNCTION: L2
    def L2(self, yhat, y):
        """
        Arguments:
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)

        Returns:
        loss -- the value of the L2 loss function defined above
        """

        ### START CODE HERE ### (≈ 1 line of code)
        loss = np.sum(np.power((y - yhat), 2))
        ### END CODE HERE ###

        return loss

    # GRADED FUNCTION: initialize_with_zeros

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros((dim, 1))
        b = 0

        assert (w.shape == (dim,1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b

# p1 = Part1()
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
# print(image.shape[0])
# print(image.shape[1])
# print(image.shape[2])
#
# print(str(p1.image2vector(image)))

# x = np.array([
# #     [0, 3, 4],
# #     [1, 6, 4]])
# # print("normalizeRows(x) = " + str(p1.normalizeRows(x)))

# x = np.array([
#     [9, 2, 5, 0, 0],
#     [7, 5, 0, 0 ,0]])
# # print("softmax(x) = " + str(p1.softmax(x)))
#
# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
# W = np.random.rand(3, len(x1))
# print("W:"+ str(W))
# mul = np.multiply(x1, W)
# print("mul:" + str(mul))
# dot = np.dot( W, x1)
# print("dot:" + str(dot))
#
#
# yhat = np.array([.9, 0.2, 0.1, .4, .9])
# y = np.array([1, 0, 0, 1, 1])
# print("L1 = " + str(p1.L1(yhat,y)))
