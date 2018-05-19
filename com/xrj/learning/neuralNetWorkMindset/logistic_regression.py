#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
import pylab
from scipy import ndimage
from com.xrj.learning.neuralNetWorkMindset import lr_utils, basic_function as basic_func

class logic_regrssion():
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

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        """
        m = X.shape[1]

        z = np.dot(w.T, X) + b
        A = basic_func.basic_sigmod(z)
        assert (A.shape == Y.shape)
        cost = -1.0 * np.sum((np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A)))) / m
        dz  = A - Y
        dzT = dz.reshape(dz.shape[0], -1).T

        dw  = np.dot(X, dzT) / m
        db  = np.sum(dz) / m

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)

        grads = {"dw" : dw,
                 "db" : db}

        return  grads, cost

    # GRADED FUNCTION: optimize
    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            costs.append(cost)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if print_cost and (i) % 100 == 0 :
                print("Cost for every iterate 100 steps, %i times cal cost: %f" %(i, cost))

            if i == num_iterations - 1 :
                grads = {"dw" : dw,
                         "db" : db}
                params = {"w" : w,
                          "b" : b}
                return params, grads, costs

    # GRADED FUNCTION: predict
    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        Y_hat = basic_func.basic_sigmod(np.dot(w.T, X) + b)
        Y_prediction = np.where(Y_hat > 0.5 , 1, 0)
        return Y_prediction

    # GRADED FUNCTION: model
    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """
        # initialize parameters with zeros 样本的数量
        dim_train_w = X_train.shape[0]

        # Gradient descent 通过已经计算好的训练数据集算出权重w和偏差b，而不要去计算test数据集的w和b
        w, b = self.initialize_with_zeros(dim_train_w)
        params, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        # Retrieve parameters w and b from dictionary "parameters" 计算出来的w和b用来计算Y_predict
        w = params["w"]
        b = params["b"]

        # Predict test/train set examples
        Y_train_predict = self.predict(w, b, X_train)
        Y_test_prdict = self.predict(w, b, X_test)

        # Print train/test Errors
        Y_train_accuracy = 100 - np.mean(np.abs(Y_train - Y_train_predict) * 100)
        Y_test_accuracy = 100 - np.mean(np.abs(Y_test - Y_test_prdict) * 100)
        print ("Train accuracy is %f" % Y_train_accuracy)
        print ("Test accuracy is %f" % Y_test_accuracy)

        d = {"w" : w,
             "b" : b,
             "costs" : costs,
             "grads" : grads,
             "Y_train_accuracy" : Y_train_accuracy,
             "Y_test_accuracy" : Y_test_accuracy,
             "Y_train_predict" : Y_train_predict,
             "Y_test_prdict" : Y_test_prdict,
             "num_iterations" : num_iterations,
             "learning_rate" : learning_rate}
        return d

lr = logic_regrssion()

# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# grads, cost = lr.propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# params, grads, costs = lr.optimize(w, b, X, Y, num_iterations= 200, learning_rate = 0.009, print_cost = True)

# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))

# w = np.array([[0.1124579],[0.23106775]])
# b = -0.3
# X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#
# print ("predictions = " + str(lr.predict(w, b, X)))

"""
注意：预处理时一定需要对数据集进行标准化，一般是对每个示例中减去整个NUMPY数组的平均值，然后将每个示例除以整个NUMPY数组的标准偏差。
但是对于图像的话方便一些，直接将每个实例除以255

"""
train_set_x_org, train_set_y_org, test_set_x_org, test_set_y_org, classes = lr_utils.lr_utils().load_dataset()
m_train = train_set_x_org.shape[0]
train_set_x =  train_set_x_org.reshape(m_train, -1).T / 255
train_set_y = train_set_y_org

m_test = test_set_x_org.shape[0]
test_set_x = test_set_x_org.reshape(m_test, -1).T / 255
test_set_y = test_set_y_org

#get the result
# d = lr.model(train_set_x, train_set_y_org, test_set_x, test_set_y_org, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
# Y_train_predict = d["Y_train_predict"]
# Y_test_prdict = d["Y_test_prdict"]

# num_px = 64
# for index in range(np.size(Y_test_prdict, axis=1)) :
#     plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
#     pylab.show()
#     print("y = " + str(test_set_y_org[0, index]) + ", you predicted that it is a \"" + classes[
#         Y_test_prdict[0, index]].decode("utf-8") + "\" picture.")

# Plot learning curve (with costs)
# costs = d["costs"]
# costs = np.squeeze(costs)
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()

#Choice of learning rate
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for ind in learning_rates:
    print ("learning rate is: " + str(ind))
    models[str(ind)] = lr.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = ind, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

# test your own example - TODO
# ## START CODE HERE ## (PUT YOUR IMAGE NAME)
# my_image = "isacatornot.jpg"   # change this to the name of your image file
# ## END CODE HERE ##
#
# # We preprocess the image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
#
# plt.imshow(image)
# print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")