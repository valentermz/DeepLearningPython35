"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier and grid search for hyperparameter tuning."""

import mnist_loader
import numpy as np
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

np.random.seed(123456)
random.seed(256)


def trim_data(training_data, validation_data, test_data, percentage=100):

    m_train = training_data[0].shape[0]
    ind_train = list(range(m_train))
    random.shuffle(ind_train)
    ind_train = ind_train[0:(percentage * m_train) // 100]

    m_val = validation_data[0].shape[0]
    ind_val = list(range(m_val))
    random.shuffle(ind_val)
    ind_val = ind_val[0:(percentage * m_val) // 100]

    m_test = test_data[0].shape[0]
    ind_test = list(range(m_test))
    random.shuffle(ind_test)
    ind_test = ind_test[0:(percentage * m_test) // 100]

    training_data = (training_data[0][ind_train, :],
                     training_data[1][ind_train])
    validation_data = (validation_data[0][ind_val, :],
                       validation_data[1][ind_val])
    test_data = (test_data[0][ind_test, :],
                 test_data[1][ind_test])

    return training_data, validation_data, test_data


# Set the percentage of data to be used
percentage = 10

# Load and trim the data
training_data, validation_data, test_data = trim_data(
    *mnist_loader.load_data(), percentage=percentage)


# Combine trainging + validation, rename to X and y

X_train = np.vstack((training_data[0], validation_data[0]))
y_train = np.concatenate((training_data[1], validation_data[1]))

X_test = test_data[0]
y_test = test_data[1]


start_time = time.time()


# Train with pre-specified hyperparameters:
# ---------------------------------------------------------------------------

# clf = SVC(kernel='rbf', C=10, gamma=.024)
# clf.fit(X_train, y_train)


# Grid Search
# ---------------------------------------------------------------------------

# Set the grid to search from:
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.022, 0.23, 0.024],
                     'C': [3, 10]}]

score = 'accuracy'
print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                   scoring=score, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()


end_time = time.time()

print(f'Total time elapsed: {end_time - start_time} seconds')


"""RESULTS:

percentage = 1
-----------------------------------------------------------------------------

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.018, 0.020, 0.022, 0.024, 0.026],
                     'C': [3, 10, 30, 100]}]

Best parameters set found on development set:
{'C': 3, 'gamma': 0.02, 'kernel': 'rbf'}

Detailed classification report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      1.00      1.00        10
           2       0.83      0.83      0.83        12
           3       0.92      0.92      0.92        13
           4       0.92      1.00      0.96        11
           5       1.00      0.80      0.89         5
           6       1.00      1.00      1.00         9
           7       0.70      0.78      0.74         9
           8       1.00      0.90      0.95        10
           9       1.00      1.00      1.00        10

   micro avg       0.93      0.93      0.93       100
   macro avg       0.94      0.92      0.93       100
weighted avg       0.93      0.93      0.93       100

Overall test accuracy: 93.0%
Total time elapsed: 24.52613377571106 seconds


percentage = 10
-----------------------------------------------------------------------------

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.020, 0.022, 0.024],
                     'C': [1, 3, 10]}]

Best parameters set found on development set:
{'C': 10, 'gamma': 0.024, 'kernel': 'rbf'}

Detailed classification report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00       107
           1       0.98      1.00      0.99       116
           2       0.95      0.94      0.95       108
           3       0.95      0.95      0.95       111
           4       0.97      0.97      0.97        96
           5       0.96      0.95      0.96        85
           6       0.99      1.00      0.99        79
           7       0.93      0.96      0.95       110
           8       0.96      0.98      0.97        82
           9       0.97      0.91      0.94       106

   micro avg       0.97      0.97      0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

Overall test accuracy: 96.6%
Total time elapsed: 477.4482045173645 seconds


percentage = 100
-----------------------------------------------------------------------------

Trained on the parameters chosen by grid search on small data:
{'C': 10, 'gamma': 0.024, 'kernel': 'rbf'}

Detailed classification report:

              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.98      0.99      0.98      1010
           4       0.99      0.98      0.99       982
           5       0.99      0.98      0.98       892
           6       0.99      0.99      0.99       958
           7       0.98      0.98      0.98      1028
           8       0.98      0.98      0.98       974
           9       0.98      0.98      0.98      1009

   micro avg       0.99      0.99      0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

Overall test accuracy: 98.56%
Total time elapsed: 543.1020934581757 seconds

"""
