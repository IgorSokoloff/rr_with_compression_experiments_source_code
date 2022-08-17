"""
A script for the data preprocessing
It takes the given dataset and outomes the partition
"""

import numpy as np
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from numpy.linalg import norm
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import math
import datetime
from IPython import display
from scipy.optimize import minimize
from logreg_functions_strongly_convex import *
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy import linalg
from numpy.random import RandomState

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')

parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg",
                    help='loss function ')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--hetero', action='store', dest='hetero', type=int, default=0, help='hetero setting')
parser.add_argument('--is_minimize', action='store', dest='is_minimize', type=int, default=1, help='minimize or not')
args = parser.parse_args()

dataset = args.dataset
loss_func = args.loss_func
num_workers = args.num_workers
hetero = args.hetero
is_minimize = args.is_minimize
#debug section
'''
num_workers = 20
dataset = 'a9a_hetero'
dataset = 'w8a_hetero'
#dataset = 'mushrooms_hetero'
loss_func = 'log-reg'
hetero = 1
is_minimize = 0
'''

if loss_func is None:
    raise ValueError("loss_func has to be specified")

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

def sort_dataset_by_label(X, y):
    sort_index = np.argsort(y)
    X_sorted = X[sort_index].copy()
    y_sorted = y[sort_index].copy()
    return X_sorted, y_sorted

currentDT = datetime.datetime.now()
print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))

data_name = dataset + ".txt"
user_dir = os.path.expanduser('~/')
RAW_DATA_PATH = os.getcwd() +'/data/'

project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

enc_labels = np.nan
data_dense = np.nan
    
#if not (os.path.isfile(data_path + 'X.npy') and os.path.isfile(data_path + 'y.npy')):
if os.path.isfile(RAW_DATA_PATH + data_name):
    data, labels = load_svmlight_file(RAW_DATA_PATH + data_name)
    enc_labels = labels.copy()
    data_dense = data.todense()
    if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
        min_label = min(np.unique(enc_labels))
        max_label = max(np.unique(enc_labels))
        enc_labels[enc_labels == min_label] = -1
        enc_labels[enc_labels == max_label] = 1
    #print (enc_labels.shape, enc_labels[-5:])
else:
    raise ValueError("cannot load " + data_name)

assert (type(data_dense) == np.matrix or type(data_dense) == np.ndarray)
assert (type(enc_labels) == np.ndarray)

if np.sum(np.isnan(enc_labels)) > 0:
    raise ValueError("nan values of labels")

if np.sum(np.isnan(data_dense)) > 0:
    raise ValueError("nan values in data matrix")

print ("Data shape: ", data_dense.shape)

if hetero:
    data_dense, enc_labels = sort_dataset_by_label(data_dense, enc_labels)

X_0 = np.float64(data_dense)
y_0 = enc_labels
assert len(X_0.shape) == 2
assert len(y_0.shape) == 1
data_len = enc_labels.shape[0]
nan_check([X_0,y_0])
np.save(data_path + 'X', X_0)
np.save(data_path + 'y', y_0)

#partition of data for each worker

chunk = data_len//num_workers

X = []
y = []
#sr_X = []
#sc_X = []
X_0_inds = np.arange(X_0.shape[0])

for i in range(num_workers):
    inds = X_0_inds[i*chunk:] if (i==num_workers-1) else X_0_inds[i*chunk:(i+1)*chunk]
    X.append(X_0[inds])
    #sr_X.append(sparse.csr_matrix(X[i]))
    #sc_X.append(sparse.csc_matrix(X[i]))
    y.append(y_0[inds])

nan_check(y)
nan_check(X)

for i in range (len(X)):
    print (f"worker {i} has {X[i].shape[0]} datasamples; class 1: {X[i][np.where(y[i] == 1)].shape[0]}; class -1: {X[i][np.where(y[i]==-1)].shape[0]}")
    #print (f"sparse:worker {i} has {sr_X[i].shape[0]} datasamples; class 1: {sr_X[i][np.where(y[i] == 1)].shape[0]}; class -1: {sr_X[i][np.where(y[i]==-1)].shape[0]}")
    np.save(data_path + 'X_{0}_nw{1}_{2}'.format(dataset, num_workers, i), np.float64(X[i]))
    #sparse.save_npz(data_path + 'sr-X_{0}_nw{1}_{2}'.format(dataset, num_workers, i), sr_X[i])
    #sparse.save_npz(data_path + 'sc-X_{0}_nw{1}_{2}'.format(dataset, num_workers, i), sc_X[i])
    np.save(data_path + 'y_{0}_nw{1}_{2}'.format(dataset, num_workers, i), y[i].flatten())

n_0, d_0 = X_0.shape

any_vector = np.zeros(d_0)

hess_f_non_reg = logreg_hess_non_reg_distributed(any_vector, X, y)

desired_cond_number = 1e+4
L_non_reg = np.float64(linalg.eigh(a=hess_f_non_reg, eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])
la = L_non_reg/(2*(desired_cond_number + 1))
#la = np.float64(linalg.eigh(a=hess_f_non_reg, eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])
mu = 2*la
L_0 = np.float64(linalg.eigh(a=hess_f_non_reg + la*regularizer_hess(any_vector), eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])

Li = np.zeros(num_workers, dtype=np.float64)
L = []
n = np.zeros(num_workers, dtype=int)
d = np.zeros(num_workers, dtype=int)
for i in range(num_workers):
    n[i], d[i] = X[i].shape
    hess_f_i = logreg_hess_non_reg(any_vector, X[i], y[i]) + la*regularizer_hess(any_vector)
    Li[i] = np.float64(linalg.eigh(a=hess_f_i, eigvals_only=True, turbo=True, type=1, eigvals=(d[i]-1, d[i]-1))[0])
    L.append([])
    for j in range(n[i]):
        L[i].append(linalg.eigh(a= np.outer(X[i][j],X[i][j])/4 + la*regularizer_hess(any_vector), eigvals_only=True, turbo=True, type=1, eigvals=(d[i]-1, d[i]-1))[0])
np.save(data_path + 'la', la)
np.save(data_path + 'L_0', L_0)
np.save(data_path + 'Li', Li)
np.save(data_path + 'L', L)
np.save(data_path + 'mu', mu)
dim = X_0.shape[1]
x_0 = np.zeros(dim, dtype=np.float64)

if not os.path.isfile(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)):
    x_0 = np.random.normal(loc=0.0, scale=1.0, size=dim)
    np.save(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset), np.float64(x_0))
else:
    x_0 = np.array(np.load(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)), dtype=np.float64)
print(f"la: {la}")
print(f"actual condition number: {L_0/mu}")

if is_minimize:
    x_star_path = data_path + 'x_star_{0}_{1}.npy'.format(loss_func, dataset)
    f_star_path = data_path + 'f_star_{0}_{1}.npy'.format(loss_func, dataset)

    print("optimization")

    grad = lambda w: logreg_grad_distributed(w, X, y, la)
    f = lambda w: logreg_loss_distributed(w, X, y, la)
    hess = lambda w: logreg_hess_distributed(w, X, y, la) + la*regularizer_hess(w)

    #minimize_result = minimize(fun=f, x0=x_0, jac=grad, method="CG", tol=1e-16, options={"maxiter": 100_000_000})
    minimize_result = minimize(fun=f, x0=x_0, jac=grad, method="Newton-CG", tol=1e-16, options={"maxiter": 100_000_000})
    x_star, f_star = minimize_result.x, minimize_result.fun
    np.save(x_star_path, np.float64(x_star))
    np.save(f_star_path, np.float64(f_star))

    print(f"x_star[:10]: {x_star[:10]}")
    print(f"f_star: {f_star}")

