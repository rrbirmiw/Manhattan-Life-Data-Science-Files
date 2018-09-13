
"""LinearSVM is a class implementation of the Linear Support Vector Machine
   The Linear SVM minimizes a  hinge loss function h(z) over beta:
   J = (1/N)*sum[  h(yx^Tbeta)   ] + lambda * Regularization(beta)

   Here, we implement 2 different hinge losses specifiable by the user:
    1. the Huberized (Smooth) Hinge Loss: http://qwone.com/~jason/writing/smoothHinge.pdf
    2. the squared hinge loss:
        http://lear.inrialpes.fr/people/harchaoui/teaching/2013-2014/ensl/m2/lecture2.pdf, (page 5)

   This module has two classes:
    1. LinearSVM()
    2. Weighted_Multiclass_SVM()

    - LinearSVM implements binary classification for y in {-1,+1}; cannot be used in multiclass settings
    - Weighted_Multiclass_SVM implements multiclass classification for y in {0,1,2....K} for K classes
      by using the one-v-rest approach. That is, it instantiates K _LinearSVM_ binary classifiers
      and uses maximal-margin heuristic to predict

    A Note on Weighted OVR Classifiers:
        https://hal.inria.fr/hal-00835810/document (page 5)
        the w-OVR classifier applies a balancing parameter, rho, to the postiive (negative) loss terms
        Harchaoui, et. al, in the above paper, shows that tuning rho greatly affects model performance
        Setting rho=0.5 (default) achieves the canonical (1/N) normalization term

    Dependencies:
    - numpy
    - loss_functions.py: this module uses loss_functions.py, a module that contains the aforementioned
                         two loss functions and their associated gradients
    Usage:
    - IMPORTANT: These classifiers require the data to be standardized, and the training
                 data matrix be in N x D format (N: size, D: dimensions)
    - IMPORTANT: Valid loss function parameters are:
        1. 'huberized_hinge'
        2. 'squared_hinge'

        >>> iris = datasets.load_iris()
        >>> X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.3, random_state=42)
        >>> scaler = StandardScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test = scaler.transform(X_test)

        >>> clf = Weighted_Multiclass_SVM(lamb=1.0, rho=0.7, loss_function='huberized_hinge')
        >>> clf.fit(X_train.T, y_train, accuracy_epsilon=0.00001)
        >>> clf.score(X_test.T, y_test)

Written by Rahul Birmiwal
2018
"""


import numpy as np
import loss_functions as lf


class LinearSVM():
    """class for a binary linear support vector machine, with user-defined
        loss function, lambda, and balancing parameter, rho """

    def __init__(self,loss_function = 'huberized_hinge', lamb=0.1, rho=0.5):
        """
        Args:
            -loss_function (str): either 'huberized_hinge' or 'squared_hinge'
            -lambda (float): Regularization parameter, default 0.1
            -rho (float): balancing parameter in [0,1], default 0.5
        """
        self.betaCoef = None
        self.X_train = None
        self.y_train = None
        if (loss_function == 'huberized_hinge'):
            self.loss_function = lf.huber_loss_obj
            self.gradient_function = lf.huber_loss_grad
        elif (loss_function == 'squared_hinge'):
            self.loss_function = lf.squared_hingeloss_obj
            self.gradient_function = lf.squared_hingeloss_grad
        self.rho = rho
        self.N = 0
        self.D = 0
        self.lamb = lamb
        self.betas = []

    def obj(self, beta):
        """ Weighted OVR applies a weight rho/N+ to the terms with y = +1
            and applies weight (1-rho)/N- to terms with y = -1

            Uses loss_functions.py
        """
        rho = self.rho
        positive_indices = self.y_train == 1
        negative_indices = self.y_train == - 1

        terms = self.y_train * np.matmul(self.X_train.T, beta) #hadamard product

        vFunc = np.vectorize(self.loss_function)
        loss_terms = vFunc(terms)
        positive_terms = loss_terms[positive_indices]
        neg_terms = loss_terms[negative_indices]

        J = (rho/len(positive_terms))*np.sum(positive_terms) + ((1-rho)/len(neg_terms))*np.sum(neg_terms)
        J +=  self.lamb*np.linalg.norm(beta)**2
        return J
    def grad(self, beta):
        """gradient of objective function above
           uses loss_functions.py
        """

        rho = self.rho
        positive_indices = self.y_train == 1
        negative_indices = self.y_train == - 1
        terms = self.y_train * np.matmul(self.X_train.T, beta) #hadamard product
        vFunc = np.vectorize(self.gradient_function)
        positive_terms = vFunc(terms[positive_indices])

        neg_terms = vFunc(terms[negative_indices])
        J = (rho/len(positive_terms))*np.matmul(self.X_train[:,positive_indices],
                                                self.y_train[positive_indices] * (positive_terms))
        J += ((1-rho)/len(neg_terms))*np.matmul(self.X_train[:,negative_indices],
                                                self.y_train[negative_indices] * (neg_terms))
        J +=  2*self.lamb*beta
        return J
    def bt_line_search(self,x, t, alpha=0.5, beta=0.9, max_iter=1000):
        """implementation of backtracking line search
           Source code adapted from Ms. Corinne Jones, University of Washington, 2018
        """
        lamb = self.lamb
        grad_x = self.grad(x)  # Gradient at x
        norm_grad_x = np.linalg.norm(grad_x)  # Norm of the gradient at x
        found_t = False
        epsilon = 0.001
        i = 0  # Iteration counter
        while (found_t is False and i < max_iter):
            term_A = self.obj(x - t*grad_x)
            term_B = self.obj(x) - alpha*t*norm_grad_x**2
            cond = term_A - epsilon < term_B
            if cond:
                found_t = True
            elif i == max_iter - 1:
                raise('Maximum number of iterations of backtracking reached')
            else:
                t *= beta
                i += 1
        return t

    def fast_gradient_descent(self,t_init, max_iter,epsilon):
        """ Performs gradient descent using the fast (accelerated) gradient method to a given
        accuracy, epsilon
        https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf

        Args:
            -t_init (float): initial step size
            -max_iter (int): maximum number of iterations
            -epsilon (float): stopping criterion for the descent algorithm (along with iter)
        Returns:
            None, but stores the beta coefficients in _this_ classifier
        """
        lamb = self.lamb
        beta_init = np.zeros(self.D)
        theta_init = np.zeros(self.D)
        b = beta_init
        theta = theta_init

        grad_theta = self.grad(theta)
        grad_beta = self.grad(b)
        self.betas = [b]
        t = t_init
        iter = 0
        while  iter < max_iter and np.linalg.norm(grad_beta) > epsilon:
            t = self.bt_line_search(b, t,lamb)
            #updates to beta/theta
            prior_b = b
            b = theta - t*grad_theta
            theta = b + (iter/(iter+3))*(b-prior_b)
            grad_theta = self.grad(theta)
            grad_beta = self.grad(b)
            iter += 1
            self.betas.append(b)
        self.betaCoef =  np.array(self.betas[-1])
        return

    def fit(self, X_train,y_train, t_init = 1.0, epsilon=0.0001, max_iter = 1000):
        """ Function analog to scikit-learn classifier fit(),
            fits X_train, y_train to this classifier to given epsilon
        Args:
        - X_train: standardized X data in N x D format
        - y_train: train labels in y in {-1, +1}
        -t_init (float): initial step size
        -max_iter (int): maximum number of iterations
        -epsilon (float): stopping criterion for the descent algorithm (along with iter)
        """
        self.X_train=X_train
        self.y_train = y_train
        self.N,self.D = X_train.shape[1],X_train.shape[0]

        # run optimizer
        self.fast_gradient_descent(t_init, max_iter, epsilon)

    def predict(self, X_test):
        """ hypothesis function for the support vector machine
            analog to sklearn predict() """
        return np.sign(np.matmul(X_test.T, self.betaCoef))

    def score(self, X_test,y_test):
        """ function analog to scikit-learn score()
        Args:
        - X_test: standardized X test data in N x D format
        - y_test: test labels in y in {-1, +1}

        Returns:
        - classification accuracy
        """

        predictions = self.predict(X_test)
        return np.mean(y_test==predictions)



class Weighted_Multiclass_SVM():
    """ class implementation of a multiclass ensemble using linear support vector machines
        analog to sklearn.svm.LinearSVC, which handles multiclassification using one-v-rest
    """
    def __init__(self, lamb, rho, loss_fn  ):
        """
        NOTE: Each of the args lamb,rho,loss_fn are applied to each of the K individual SVMs
        Args:
            -loss_fn (str): either 'huberized_hinge' or 'squared_hinge' for each clf
            -lambda (float): Regularization parameter, default 0.1 for each clf
            -rho (float): balancing parameter in [0,1], default 0.5 for each clf
        """

        self.loss_fn = loss_fn
        self.classifiers = []
        self.rho = rho
        self.mapping = {}
        self.lamb = lamb

    def hypothesis(self,X, beta):
        return np.matmul(X.T, beta) # the margin

    def fit(self, X_train, y_train, accuracy_epsilon=0.0001):
        """ Function analog to scikit-learn classifier fit(),
            fits X_train, y_train to this classifier to given epsilon
        Args:
        - X_train: standardized X data in N x D format
        - y_train: train labels in y in {-1, +1}
        - accuracy_epsilon (float): stopping criterion for the descent algorithm (along with iter)
                              default 0.0001
        """
        class_labels = np.unique(y_train)
        counter = 0
        for classNum in class_labels:
            #the jth classifier will predict y== +1 as classNum
            self.mapping[counter] = classNum
            counter += 1

            # recreate the training labels such that y = + 1 if y==classNum
            # else, -1
            z_train = np.where( y_train == classNum, 1, -1)
            clf = LinearSVM(self.loss_fn, self.lamb, self.rho )
            clf.fit(X_train, z_train, epsilon=accuracy_epsilon)
            self.classifiers.append(clf)

    def predict(self, X_test):
        """ predict each row in the N x D testing matrix of data using
            maximal-margin classification scheme
            analog to sklearn predict()
            """
        predictions = []
        for i in np.arange(0, X_test.shape[1]):
            x_i = X_test[:,i]
            all_margins = []
            for clf in self.classifiers:
                margin = self.hypothesis(x_i, clf.betaCoef) #1 x d times d x 1
                all_margins.append( margin )
            predictions.append(   self.mapping[np.argmax(all_margins)] ) #plus 1 because zero-indexed here
        return predictions

    def score(self, X_test, y_test):
        """ function analog to sklearn score()
        Returns:
            -(predictions, score): (tuple): predictions is a vector of predictions
                                            score is classification accuracy
        """
        predictions = self.predict(X_test)
        return (predictions, np.mean(predictions==y_test))
