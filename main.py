import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def sigmoid_function(z):
    """Sigmoid or logistic function to get the estimated probability that label is 1 on input x.
    :param z:
            1) transposed vector of thetas * vector of features.
            2) row of features * vector of thetas.

    !!! For second case you need to vectorize it before use: "np.vectorize(sigmoid_function)".
    """

    return 1 / (1 + math.exp(-z))


vectorized_sigmoid_function = np.vectorize(sigmoid_function)


def cost_function(x, params, y, m):
    """Standard cost function for logistic regression.
    :param x: matrix of features.
    :param params: vector of thetas.
    :param y: vector of labels.
    :param m: number of training examples
    """

    x_theta = np.matmul(x, params)
    hypothesis = vectorized_sigmoid_function(x_theta)

    return np.sum(np.log(hypothesis) * y + np.log(1 - hypothesis) * (1 - y)) * (-1 / m)


# Data loading

df_train = pd.read_csv("train.csv").values
df_test = pd.read_csv("test.csv").values

train_features, train_labels = np.c_[np.ones(df_train.shape[0]), df_train[:, :14]], df_train[:, 14:]
test_features, test_labels = np.c_[np.ones(df_test.shape[0]), df_test[:, :14]], df_test[:, 14:]


# Init vector of thetas by ones

thetas = np.ones((df_train.shape[1], 1))


# Cost function for initial thetas

cost = cost_function(train_features, thetas, train_labels, df_train.shape[0])

print("Cost with initial thetas: {}".format(cost))


# Minimizing cost function

learning_rate = 0.00001
previous_cost = cost

# Break loop when cost function decreases less than e = 10^-3

for step in range(1000):
    buffer_thetas = np.ones((df_train.shape[1], 1))

    for j in range(df_train.shape[1]):
        buffer_thetas[[j]] = learning_rate * np.sum(
            (vectorized_sigmoid_function(np.matmul(train_features, thetas)) - train_labels) * train_features[:, j])

    thetas -= buffer_thetas
    current_cost = cost_function(train_features, thetas, train_labels, df_train.shape[0])

    if previous_cost - current_cost < 0.001:
        break

    print("Cost on {} step: {:.8f}".format(step, current_cost))

    previous_cost = current_cost


print("------------------------")


# Evaluating an accuracy of the model.
# Because this is a classification problem, we suppose pred = 1 if pred > 0.5 else 0

y_pred = []
y_true = []

for i in range(test_features.shape[0]):
    pred = 1 if float(sigmoid_function(np.matmul(test_features[i], thetas))) >= 0.5 else 0
    true = int(test_labels[i])

    y_pred.append(pred)
    y_true.append(true)

    print("Predicted - {}, Label - {}".format(pred, true))


print("------------------------")
print("The accuracy is: {:.2f}".format(accuracy_score(y_pred=y_pred, y_true=y_true)))
