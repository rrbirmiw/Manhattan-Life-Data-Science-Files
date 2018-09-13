""" demo.py contains code to compare the performance of _our_ classifier
    against that of the sklearn equivalent -- LinearSVC -- using the same
    hyperparameters

    For both classifiers we set Lambda=1, rho = 0.5, max_iter = 1000 and epsilon =0.0001

    Output:
    -classification accuracy on the IRIS dataset for each classifier

    Usage:
    >>> python demo.py

Written by Rahul Birmiwal
2018
"""

""" OTHER EXAMPLES
1. L2 / Ridge Regression http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html#sphx-glr-auto-examples-linear-model-plot-ridge-coeffs-py
2. LASSO http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py
3. Logistic Regression / Probabilities http://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py

4. Multiclass (See main below )
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn import datasets
import linearsvm as my_clfs

if __name__ == '__main__':
    # Load and standardize the data
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create _our_ multiclass SVM
    clf = my_clfs.Weighted_Multiclass_SVM(lamb=1.0, rho=0.5, loss_fn = 'squared_hinge')
    clf.fit(X_train.T, y_train)
    (my_predictions, my_score) = clf.score(X_test.T, y_test)

    # Create using sci-kit learn's LinearSVC, using same hyperparameters
    sklearn_clf = LinearSVC(loss='squared_hinge', C=1.0, tol=0.0001, max_iter=1000,
                            fit_intercept=False) #same hyperparameters
    sklearn_clf.fit(X_train, y_train)
    sk_score = sklearn_clf.score(X_test, y_test)
    sk_predictions = sklearn_clf.predict(X_test)

    print("Accuracy using our implementation ", my_score)
    print("Accuracy using scikit learn equivalent ", sk_score)
