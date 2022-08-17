
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
from scipy import sparse
from numpy.random import RandomState
from logreg_functions_strongly_convex import *

myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x)
sqnorm = lambda x: norm(x, ord=2) ** 2


def stopping_criterion(epoch, n_epoch, arg_res, eps):
    return (epoch <= n_epoch) and (arg_res >=eps)

'''
def stopping_criterion_epochs(epoch, n_epoch):
    return (epoch <= n_epoch)
'''

def rand_k_compressor(x, k_rk, rs_rand, randk_probs):
    output = np.zeros(x.shape)
    dim = x.shape[0]
    inds = rs_rand.choice(a=np.arange(dim), size=k_rk, replace=False, p=randk_probs)
    output[inds] = x[inds]
    return (dim/k_rk)* output

def sgd_reshuffled_estimator(X, x, y, la, inds):
    return logreg_grad(x, X[inds], y[inds], la)

#TODO:
def estimator (X_ar, x, y_ar, la, n_w, k_rk, rs_randk, inds_ar, randk_probs, h_ar_j, alpha):
    diana_est = np.zeros(x.shape[0])
    for i in range(n_w):
        s_grad = sgd_reshuffled_estimator(X_ar[i], x, y_ar[i], la, inds_ar[i]).copy()
        m_i = rand_k_compressor(s_grad - h_ar_j[i], k_rk, rs_randk, randk_probs).copy()
        diana_est += h_ar_j[i] + m_i
        h_ar_j[i] = h_ar_j[i] + alpha*m_i
    return diana_est/n_w, h_ar_j


def run_algorithm(x_0, x_star, f_star, X_ar, y_ar, la, stepsize, batch_ar, eps, k_rk, n_w, experiment_name, project_path, dataset, n_epochs, n_ls_in_epoch, alpha):
    currentDT = datetime.datetime.now()
    print(currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print(experiment_name + f" with k={k_rk} has started")
    print ("step_size: ", step_size)
    NUM_BITS_PER_FLOAT = 64
    bits_sent_per_worker = NUM_BITS_PER_FLOAT*k_rk
    rs_randk = RandomState(12345)
    rs_perm = RandomState(12346)
    dim = x_0.shape[0]
    n_samples_total = sum([X_ar[i].shape[0] for i in range(n_w)])
    randk_probs = np.full(dim, 1 / dim)
    h_ar = np.zeros((n_ls_in_epoch, n_w, dim), dtype=np.float64)
    X_reshuffled_indices = [rs_perm.permutation(np.arange(X_ar[i].shape[0])) for i in range (n_w)]
    f_x = logreg_loss_distributed(x_0, X_ar, y_ar, la)
    func_diff_ar = [f_x - f_star]
    bits_od_ar = [0]
    comms_ar = [0]
    epochs_ar = [0]
    arg_res_ar = [sqnorm(x_0 - x_star)] #argument residual \sqnorm{x^t - x_star}
    x = x_0.copy()
    PRINT_EVERY = 10000000
    j_ls = 0
    while stopping_criterion(epochs_ar[-1], n_epochs, arg_res_ar[-1], eps):
        if j_ls == n_ls_in_epoch:
            j_ls = 0
        inds_ar = [X_reshuffled_indices[i][j_ls*batch_ar[i]:] if (j_ls==n_ls_in_epoch-1) else X_reshuffled_indices[i][j_ls*batch_ar[i]:(j_ls+1)*batch_ar[i]] for i in range (n_w)]
        g, h_ar[j_ls] = estimator(X_ar, x, y_ar, la, n_w, k_rk, rs_randk, inds_ar, randk_probs, h_ar[j_ls], alpha)
        x = x - stepsize*g
        f_x = logreg_loss_distributed(x, X_ar, y_ar, la)
        func_diff_ar.append(f_x - f_star)

        bits_od_ar.append(bits_od_ar[-1]+bits_sent_per_worker)
        arg_res_ar.append(sqnorm(x - x_star))
        comms_ar.append(comms_ar[-1] + 1)
        batch_total = sum(list(map(len, inds_ar)))
        epochs_ar.append(epochs_ar[-1] + (batch_total/n_samples_total))
        j_ls += 1

        if comms_ar[-1]%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print_last_point_metrics(bits_od_ar,  epochs_ar, comms_ar, arg_res_ar, func_diff_ar)
            save_data(bits_od_ar,  epochs_ar, comms_ar, arg_res_ar, func_diff_ar, x.copy(), k_rk, experiment_name, project_path, dataset)

    save_data(bits_od_ar, epochs_ar, comms_ar, arg_res_ar, func_diff_ar, x.copy(), k_rk, experiment_name, project_path, dataset)
    print(experiment_name + f" with k={k_rk} finished")
    print("End-point:")
    print_last_point_metrics(bits_od_ar,  epochs_ar, comms_ar, arg_res_ar, func_diff_ar)


def save_data(bits_od_ar, epochs_ar, comms_ar, arg_res_ar, func_diff_ar, x_solution, k_rk, experiment_name, project_path, dataset):
    print("data saving")
    experiment = '{0}_{1}'.format(experiment_name, k_rk)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'bits_od' + '_' + experiment, np.array(bits_od_ar, dtype=np.float64))
    np.save(logs_path + 'epochs' + '_' +  experiment, np.array(epochs_ar, dtype=np.float64))
    np.save(logs_path + 'comms' + '_' +    experiment, np.array(comms_ar, dtype=np.float64))
    np.save(logs_path + 'args_diff' + '_' + experiment, np.array(arg_res_ar, dtype=np.float64))
    np.save(logs_path + 'funcs_diff' + '_' +         experiment, np.array(func_diff_ar, dtype=np.float64))
    np.save(logs_path + 'solution' + '_' +          experiment, x_solution)

def print_last_point_metrics(bits_od_ar,  epochs_ar, comms_ar, arg_res_ar, func_diff_ar):
    print(f"comms: {comms_ar[-1]}; epochs:{epochs_ar[-1]} bits_od: {bits_od_ar[-1]}; arg_res: {arg_res_ar[-1]}; func_diff: {func_diff_ar[-1]}")

parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_epochs', action='store', dest='max_epochs', type=int, default=10, help='Maximum number of epochs')
parser.add_argument('--max_bits', action='store', dest='max_bits', type=float, default=None, help='Maximum number of bits transmitted from worker to server')
parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--n_w', action='store', dest='n_w', type=int, default=20, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=float, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-7, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='Dataset name for saving logs')
parser.add_argument('--prb', action='store', dest='prb', type=float, default=0.1, help='Proportion of batchsize')
parser.add_argument('--stepsize_type', action='store', dest='stepsize_type', type=str, default="theoretical", help='Stepsize type')
parser.add_argument('--stepsize_base', action='store', dest='stepsize_base', type=float, default=None, help='Stepsize')
args = parser.parse_args()

n_epochs = args.max_epochs
n_bits = args.max_bits
k_rk = args.k
n_w = args.n_w
eps = args.tol
dataset = args.dataset
factor = args.factor
prb = args.prb
stepsize_type = args.stepsize_type
stepsize_base = args.stepsize_base

loss_func = "log-reg"
'''
n_w = 20
k_rk = 50
n_epochs = 1
dataset = "mushrooms_hetero"
loss_func = "log-reg"
factor = 1.0
eps = 1e-8
prb = 0.1
stepsize_type = "theoretical"
stepsize_base = None
'''
assert stepsize_type in ["theoretical", "custom"]

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"
print(project_path)
data_path = project_path + "data_{0}/".format(dataset)

X_ar = []
y_ar = []
for i in range(n_w):
    X_ar.append(np.load(data_path + 'X_{0}_nw{1}_{2}.npy'.format(dataset, n_w, i)))
    y_ar.append(np.load(data_path + 'y_{0}_nw{1}_{2}.npy'.format(dataset, n_w, i)))
    assert (X_ar[i].dtype == 'float64')
    assert (len(X_ar[i].shape) == 2)

la = np.load(data_path + 'la.npy')
L_0 = np.load(data_path + 'L_0.npy')
Li = np.load(data_path + 'Li.npy')
L = np.load(data_path + 'L.npy')
mu = np.load(data_path + 'mu.npy')

x_0 = np.array(np.load(data_path + 'w_init_{0}_{1}.npy'.format(loss_func, dataset)), dtype=np.float64)
x_star_path = data_path + 'x_star_{0}_{1}.npy'.format(loss_func, dataset)
f_star_path = data_path + 'f_star_{0}_{1}.npy'.format(loss_func, dataset)
x_star = np.float64(np.load(x_star_path))
f_star = np.float64(np.load(f_star_path))

dim = X_ar[0].shape[1]

omega = dim/k_rk - 1
alpha = 1/(omega + 1)
n_its_in_epoch = int(1/prb)
batch_ar = np.array([int(prb*X_ar[i].shape[0]) for i in range(n_w)])

if stepsize_type=="theoretical":
    stepsize_base = min(alpha/(2*n_its_in_epoch*mu), 1/(np.max(L)*(1 + 6*omega/n_w)))

step_size = np.float64(stepsize_base*factor)


experiment_name = "diana-rr_nw{0}_b{1}_{2}x".format(n_w, myrepr(prb), myrepr(factor))
print ('------------------------------------------------------------------------------------------------')
start_time = time.time()
run_algorithm(x_0, x_star, f_star, X_ar, y_ar, la, step_size, batch_ar, eps,  k_rk, n_w, experiment_name, project_path, dataset, n_epochs, n_its_in_epoch, alpha)
time_diff = time.time() - start_time
print(f"Computation time: {time_diff} sec")
print ('------------------------------------------------------------------------------------------------')


