
"""
logistic regression with strongly convex regularizer
"""

import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

sqnorm = lambda x: np.linalg.norm(x, ord=2)**2

def regularizer(w: np.ndarray):
    return sqnorm(w)

def regularizer_grad(w):
    return 2*w

def regularizer_hess(w):
    d_0 = w.shape[0]
    return 2*np.eye(d_0)

def logreg_loss_non_reg(w, X, y, la):
    assert la > 0
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    w = w.flatten()
    y = y.flatten()
    l = np.log(1 + np.exp(-np.multiply(X@w,y)))
    return np.mean(l)

def logreg_loss_distributed(w, X_ar, y_ar, la):
    assert la > 0
    n_workers = len(X_ar)
    cum_los = 0
    for i in range(n_workers):
        cum_los += logreg_loss_non_reg(w, X_ar[i], y_ar[i], la)
    return cum_los/n_workers + la * regularizer(w)

def logreg_grad(w, X, y, la):
    assert la > 0
    assert (y.shape[0] == X.shape[0])
    assert (w.shape[0] == X.shape[1])

    X_y = np.multiply(X, y[:, np.newaxis])
    denominator = 1 + np.exp(X_y@w)
    denominator = np.squeeze(np.asarray(denominator))
    loss_grad = - np.mean(X_y/denominator[:,np.newaxis], axis = 0)
    loss_grad = np.squeeze(np.asarray(loss_grad))
    assert len(loss_grad) == len(w)
    return loss_grad + la * regularizer_grad(w)

def logreg_grad_distributed(w, X_ar, y_ar, la):
    assert la > 0
    n_workers = len(X_ar)
    cum_grad = 0
    for i in range(n_workers):
        cum_grad += logreg_grad(w, X_ar[i], y_ar[i], la)
    return cum_grad/n_workers

def logreg_hess_non_reg(w, X_0, y_0):
    n_0, d_0 = X_0.shape
    return (1 / (4*n_0)) * (X_0.T @ X_0)

def logreg_hess_non_reg_distributed(w, X_ar, y_ar):
    n_workers = len(X_ar)
    cum_hess = 0
    for i in range(n_workers):
        cum_hess += logreg_hess_non_reg(w, X_ar[i], y_ar[i])
    return cum_hess/n_workers

def logreg_hess_distributed(w, X_ar, y_ar, la):
    n_workers = len(X_ar)
    cum_hess = 0
    for i in range(n_workers):
        for j in range(X[i].shape[0]):
            X_y_w = y_ar[i][j]*X_ar[i][j]@ w
            cum_hess += ((np.exp(-0.5*X_y_w) + np.exp(0.5*X_y_w))**(-2)) * (X_ar[i][j]@X_ar[i][j].T)/(X[i].shape[0])
    return cum_hess/n_workers +la*regularizer_hess(w)

#computes partial derivative (for QSGD, RR-QSGD)
def logreg_part_grad(w, X, y, la, ids):
    """
    Returns full gradient
    :param w:
    :param X:
    :param y:
    :param la:
    :return:
    """
    assert la > 0
    assert (y.shape[0] == X.shape[0])
    assert (w.shape[0] == X.shape[1])

    loss_part_grad = np.zeros(w.shape)

    numerator = np.multiply(X[:,ids], y[:, np.newaxis])
    denominator = 1 + np.exp(np.multiply(X@w, y))

    matrix = numerator/denominator[:,np.newaxis]

    loss_part_grad[ids] = - np.mean(matrix, axis = 0)

    assert len(loss_part_grad) == len(w)
    return loss_part_grad + la * regularizer_part_grad(w,ids)

def regularizer_part_grad(w, ids):
    reg_part_grad = np.zeros(w.shape)
    reg_part_grad[ids] = 2*w[ids]
    return reg_part_grad

##
#Sparse computations
##

def logreg_loss_non_reg_sparse(w, X, y, la):
    assert la > 0
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    w = w.flatten()
    y = y.flatten()
    #l = np.log(1 + np.exp(-X.dot(w) * y))
    l = np.log(1 + np.exp(-np.multiply(X@w,y)))
    return np.mean(l)

def logreg_loss_distributed_sparse(w, X_ar, y_ar, la):
    assert la > 0
    n_workers = len(X_ar)
    cum_los = 0
    for i in range(n_workers):
        cum_los += logreg_loss_non_reg(w, X_ar[i], y_ar[i], la)
    return cum_los/n_workers + la * regularizer(w)

def logreg_grad_old(w, X, y, la):

    assert la > 0
    assert (y.shape[0] == X.shape[0])
    assert (w.shape[0] == X.shape[1])

    numerator = np.multiply(X, y[:, np.newaxis])
    denominator = 1 + np.exp(np.multiply(X@w,y))
    loss_grad = - np.mean(numerator/denominator[:,np.newaxis], axis = 0)

    assert len(loss_grad) == len(w)
    return loss_grad + la * regularizer_grad(w)

