# Federated Optimization Algorithms with Random Reshuffling and Gradient Compression

This is a Python 3 implementation of Q-RR, DIANA-RR, Q-NASTYA, NASTYA-DIANA, QSGD, DIANA, FedCOM and FedPAQ algorithms from the [paper](https://arxiv.org/abs/2206.07021).

 ## Reference
 In case you find the method or code useful for your research, please consider citing

 ```
@article{sadiev2022federated,
  title={Federated Optimization Algorithms with Random Reshuffling and Gradient Compression},
  author={Sadiev, Abdurakhmon and Malinovsky, Grigory and Gorbunov, Eduard and Sokolov, Igor and Khaled, Ahmed and Burlachenko, Konstantin and Richt{\'a}rik, Peter},
  journal={arXiv preprint arXiv:2206.07021},
  year={2022}
}

 ```
 ## Repository overview
 - Folder "data" contains LibSVM datasets (phishing, mushrooms, and a9a) that were used for experiments. They were downloaded from the official [library](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

 - Method scripts:
 - - "data\_preprocessing.py" - script for the data preprocessing before launching any experiment. It performs data partitioning per workers, computes and saves necessary Lipshitz constants that are used by each method to get theoretical stepsize. Moreover, it estimates the solution of the problem via scipy solver (this is needed to track $f(x^t) - f(x^{\star})$ during the optimization procedure);
 - - "logreg_functions\_strongly\_convex.py" - contains auxiliary functions for logistic regression problem;
 - -  The rest of scripts corresponds to the methods that were used in the paper (Q-RR, DIANA-RR, Q-NASTYA, NASTYA-DIANA, QSGD, DIANA, FedCOM and FedPAQ).

## How to launch scripts

Command examples below are expected to be executed in the bash\shell prompt in the project folder (that contains all scripts and "data" folder). In all commands, for simplicity, we set default parameters that were in our experiments. All commands were tested on MacOS and Linux.

### Preprocessing
Before running experiments one need to do a praparation step via running a script "data\_preprocessing.py":

```
python3 data_preprocessing.py --dataset mushrooms --num_workers 20 --loss_func log-reg --hetero 1 --is_minimize 1
```

**Parameters:**
- --dataset:
- --num_workers - number of workers participation in the training.
- --loss_func - a loss function type. In our experiments, we used only the Logistic Regression problem. 
- --hetero: 0 or 1. Parameter 0 corresponds to the data partitioning in the default order. If one set 1, the whole dataset is firstly
sorted in ascending order of labels, and then equally split among workers. For more detail see Appendix A in the [paper](https://arxiv.org/abs/2206.07021). That is, the choice \[--hetero 1\] relates to the more heterogeneous setting than default splitting.
- --is_minimize: 0 or 1. For the very first run, one needs to set \[--is_minimize 1\] to estimate the solution of the problem.
If one wishes to change the setting (eg., increase number of workers) there is no need to solve the optimization problem again, therefore one can choose \[--is_minimize 0\] for faster preprocessing.

### Launching experiments


 ## License
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
