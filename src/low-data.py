"""SMALL DATA EXPERIMENT:

How would this network perform with 10 times less data?
How would it perform compared to an SVM?
"""

import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU
import numpy as np
import random
import time

np.random.seed(123456)
random.seed(256)


# def load_reduce_data(filename="../data/mnist.pkl.gz", percentage=10):
#     f = gzip.open(filename, 'rb')
#     training_data, validation_data, test_data = pickle.load(
#         f, encoding="latin1")
#     f.close()

#     m_train = training_data[0].shape[0]
#     ind_train = list(range(m_train))
#     random.shuffle(ind_train)
#     ind_train = ind_train[0:m_train // percentage]

#     m_val = validation_data[0].shape[0]
#     ind_val = list(range(m_val))
#     random.shuffle(ind_val)
#     ind_val = ind_val[0:m_val // percentage]

#     m_test = test_data[0].shape[0]
#     ind_test = list(range(m_test))
#     random.shuffle(ind_test)
#     ind_test = ind_test[0:m_test // percentage]

#     training_data = (training_data[0][ind_train, :],
#                      training_data[1][ind_train])
#     validation_data = (validation_data[0][ind_val, :],
#                        validation_data[1][ind_val])
#     test_data = (test_data[0][ind_test, :],
#                  test_data[1][ind_test])

#     return training_data, validation_data, test_data


training_data, validation_data, test_data = network3.load_data_shared(percentage=10)
mini_batch_size = 30


net = Network([
    ConvPoolLayer(input_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(input_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

start_time = time.time()
net.SGD(training_data, 120, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
end_time = time.time()

print(f'Total time elapsed: {end_time - start_time} seconds')



"""RESULTS:

100% of data
------------
mini_batch_size = 10
net.SGD(training_data, 50, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
Best validation accuracy of 99.16% obtained at iteration 139999 (epoch 27)
Corresponding test accuracy of 99.19%
Total time elapsed: 3364.949262857437 seconds


10% of data
-----------
mini_batch_size = 10
net.SGD(training_data, 50, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
Best validation accuracy of 97.50% obtained at iteration 5499 (epoch 10)
Corresponding test accuracy of 96.40%
Total time elapsed: 485.6690628528595 seconds

mini_batch_size = 30
net.SGD(training_data, 120, mini_batch_size, 0.01, validation_data, test_data, lmbda=0.5)
Best validation accuracy of 97.88% obtained at iteration 19919 (epoch 118)
Corresponding test accuracy of 96.36%
Total time elapsed: 748.222781419754 seconds

1% of data
----------
mini_batch_size = 10
(training_data, 500, mini_batch_size, 0.001, validation_data, test_data, lmbda=1)
Best validation accuracy of 88.00% obtained at iteration 24999
Corresponding test accuracy of 90.00%
Total time elapsed: 300.659471988678 seconds

"""
