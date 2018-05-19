#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from com.xrj.learning.neuralNetWorkMindset import planar_utils as putils, testCases, basic_function as bf

np.random.seed(1) # set a seed so that the results are consistent

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess

# GRADED FUNCTION: layer_sizes
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer
    n_h = 4  #hard core
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


# GRADED FUNCTION: initialize_parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = bf.basic_sigmod(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# GRADED FUNCTION: compute_cost
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log((1-A2)), (1-Y))
    cost = -(1.0 / m) * np.sum(logprobs)
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))

    return cost


# GRADED FUNCTION: backward_propagation
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1.0 / m ) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m ) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1,2))
    dW1 = (1.0 / m ) * np.dot(dZ1, X.T)
    db1 = (1.0 / m ) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# GRADED FUNCTION: predict
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.where(A2 > 0.5, 1, 0)

    return predictions

if __name__ == '__main__':
    # X, Y = putils.load_planar_dataset()
    # Visualize the data:
    # plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
    # plt.show()

    # shape_X = np.shape(X)
    # shape_Y = np.shape(Y)
    # m = np.size(X, axis=1)  # training set size
    # print('The shape of X is: ' + str(shape_X))
    # print('The shape of Y is: ' + str(shape_Y))
    # print('I have m = %d training examples!' % (m))

    # # Train the logistic regression classifier
    # clf = sklearn.linear_model.LogisticRegressionCV();
    # clf.fit(X.T, Y.T);
    # # Plot the decision boundary for logistic regression
    # putils.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    # plt.title("Logistic Regression")
    #
    # # Print accuracy
    # LR_predictions = clf.predict(X.T)
    # print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
    #        '% ' + "(percentage of correctly labelled datapoints)")

    # X_assess, Y_assess = testCases.layer_sizes_test_case()
    # (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    # print("The size of the input layer is: n_x = " + str(n_x))
    # print("The size of the hidden layer is: n_h = " + str(n_h))
    # print("The size of the output layer is: n_y = " + str(n_y))

    # n_x, n_h, n_y = testCases.initialize_parameters_test_case()
    # parameters = initialize_parameters(n_x, n_h, n_y)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # X_assess, parameters = testCases.forward_propagation_test_case()
    # A2, cache = forward_propagation(X_assess, parameters)
    # # Note: we use the mean here just to make sure that your output matches ours.
    # print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    # A2, Y_assess, parameters = testCases.compute_cost_test_case()
    # print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

    # parameters, cache, X_assess, Y_assess = testCases.backward_propagation_test_case()
    # grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    # print("dW1 = " + str(grads["dW1"]))
    # print("db1 = " + str(grads["db1"]))
    # print("dW2 = " + str(grads["dW2"]))
    # print("db2 = " + str(grads["db2"]))

    # parameters, grads = testCases.update_parameters_test_case()
    # parameters = update_parameters(parameters, grads)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # X_assess, Y_assess = nn_model_test_case()
    # parameters = nn_model(X_assess, Y_assess, 4, num_iterations=5000, print_cost=True)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # np.random.seed(1)
    # X, Y = putils.load_planar_dataset()
    # plt.figure(figsize=(50, 50))
    # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    # for n_h in hidden_layer_sizes:
    #     parameters = nn_model(X, Y, n_h)
    #     putils.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    #     predictions = predict(parameters, X)
    #     accuracy = float((np.dot(Y, predictions.T)) + np.dot(1 - Y, 1 - predictions.T)) * 100 / float(Y.size)
    #     print("Accuracy for %d hidden units = " % n_h + str(accuracy))
    #     plt.show()

    # parameters, X_assess = testCases.predict_test_case()
    # predictions = predict(parameters, X_assess)
    # print("predictions mean = " + str(np.mean(predictions)))

    # # Build a model with a n_h-dimensional hidden layer
    # X, Y = putils.load_planar_dataset()
    # parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    # # Plot the decision boundary
    # putils.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show()
    # # Print accuracy
    # predictions = predict(parameters, X)
    # print('Accuracy: %d' % float(
    #     (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    X, Y = putils.load_planar_dataset()
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        putils.plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

