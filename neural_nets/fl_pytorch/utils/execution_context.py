#!/usr/bin/env python3

# Import PyTorch root package import torch
import torch

import threading, random, time
import numpy as np

from . import thread_pool

class ExecutionContext: 
    ''' Private execution data. Please do not manipulate directly, only with dedicated API'''
    pass

def initExecutionContext():
    """Initialize thread specific execution context"""
    c = ExecutionContext()
    c.eval_thread_pool = thread_pool.ThreadPool()         #: Threads responsible for model evaluation
    c.local_training_threads = thread_pool.ThreadPool()   #: Threads responsible for local training
    c.remote_training_threads = thread_pool.ThreadPool()  #: Threads responsible for remote training
    c.saver_thread = thread_pool.ThreadPool()             #: Threads responsible for work involved saving results into filesystem

    c.experimental_options = {}                           #: Dicltionary with experimental options to experiments with optimization algorithms
    c.np_random = np.random.RandomState()                 #: Thread specific numpy random generator
    c.random = random.Random()                            #: Thread specific Python random generator
    c.context_init_time = time.time()                     #: Time when executio context has been initialized
    return c

def resetExecutionContext(c):
    """Reset thread specific execution context"""
    for th in c.remote_training_threads.threads:
        th.external_socket.rawSendString("finish_work")

    c.remote_training_threads.stop()
    c.local_training_threads.stop()
    c.eval_thread_pool.stop()
    c.saver_thread.stop()


#============================================================================================================================================================================
torch_global_lock = threading.Lock()               #: Internal global locks to mitigate problems with PyTorch initialization with global torch state
#============================================================================================================================================================================
# Global shared callbacks for tasks
simulation_start_fn = None                         #: Callback for start simulation "callback(H:server_state)"
is_simulation_need_ealystop_fn = None              #: Predicate which can be setuped for checking that simulation should be canceled. "callback(H:server_state)"
simulation_progress_steps_fn   = None              #: Callback for track the current progress of training model, e.g. in GUI          "callback(progress:float, H:server_state)"
simulation_finish_fn  = None                       #: Callback for fucntion which will be called once as simulation is finished.      "callback(H:server_state)"
