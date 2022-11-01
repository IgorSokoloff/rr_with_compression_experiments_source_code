# Federated Optimization Algorithms with Random Reshuffling and Gradient Compression

This is a repository with Python 3 source code for the experimental part of the paper Federated Optimization Algorithms with Random Reshuffling and Gradient Compression [arXiv:2206.07021](https://arxiv.org/abs/2206.07021).

# The structure of the repository

Path | Description
--- | --- |
[logistic_regression](logistic_regression)  |  Experiments for Training L2 Regulirized Logistic Regression in LIBSVM datasets across 20 clients.
[neural_nets](neural_nets)  | Experiments for Training Deep Neural Network model: ResNet-18 on CIFAR-10 across 10 clients.
----

# Reference
In case you find the methods or code useful for your research, please consider citing

```
@article{sadiev2022federated,
  title={Federated Optimization Algorithms with Random Reshuffling and Gradient Compression},
  author={Sadiev, Abdurakhmon and Malinovsky, Grigory and Gorbunov, Eduard and Sokolov, Igor and Khaled, Ahmed and Burlachenko, Konstantin and Richt{\'a}rik, Peter},
  journal={arXiv preprint arXiv:2206.07021},
  year={2022}
}
```

# Information about Authors

* [Abdurakhmon Sadiev](https://www.researchgate.net/profile/Abdurakhmon-Sadiev)
* [Grigory Malinovsky](https://grigory-malinovsky.github.io/)
* [Eduard Gorbunov](https://eduardgorbunov.github.io/)
* [Igor Sokolov](https://cemse.kaust.edu.sa/people/person/igor-sokolov)
* [Ahmed Khaled](https://rka97.github.io/)
* [Prof. Peter Richtarik](https://richtarik.org/) 

# Paper Abstract
Gradient compression is a popular technique for improving communication complexity of stochastic first-order methods in distributed training of machine learning models. 
However, the existing works consider only with-replacement sampling of stochastic gradients. In contrast, it is well-known in practice and recently confirmed
in theory that stochastic methods based on without-replacement sampling, e.g., Random Reshuffling (<span style="color:rgb(213,40,16)">RR</span>) method, perform better than ones that sample the gradients with-replacement. In this work, we close this gap in the literature and provide the first analysis of methods with gradient compression and without-replacement sampling. 

We first develop a distributed variant of random reshuffling with gradient compression (<span style="color:rgb(213,40,16)">Q-RR</span>), and show how to reduce the variance coming from gradient quantization through the use of control iterates. 

Next, to have a better fit to Federated Learning applications, we incorporate local computation and propose a variant of <span style="color:rgb(213,40,16)">Q-RR</span> called <span style="color:rgb(213,40,16)">Q-NASTYA</span>. <span style="color:rgb(213,40,16)">Q-NASTYA</span> uses local gradient steps and different local and global stepsizes.

Next, we show how to reduce compression variance in this setting as well. Finally, we prove the convergence results for the proposed methods and outline several settings in which they improve upon existing algorithms.

----

<table style="text-align:center;">
<tr>
<td style="padding:15px;text-align:center;vertical-align:middle;"> <img height="100px" src="https://github.com/IgorSokoloff/rr_with_compression_experiments_source_code/blob/main/imgs/KAUST-logo.png"/> </td> 
<td style="padding:15px;text-align:center;vertical-align:middle;"> <img height="75px" src="https://github.com/IgorSokoloff/rr_with_compression_experiments_source_code/blob/main/imgs/MBZUAI_Logo.png"/> </td>
<td style="padding:15px;text-align:center;vertical-align:middle;"> <img height="100px" src="https://github.com/IgorSokoloff/rr_with_compression_experiments_source_code/blob/main/imgs/mipt-logo.png"/> </td> 
<td style="padding:15px;text-align:center;vertical-align:middle;"> <img height="75px" src="https://github.com/IgorSokoloff/rr_with_compression_experiments_source_code/blob/main/imgs/princeton-university-logo.png"/> </td>
</tr>
</table>

# Reference Materials

We would like to provide several resources of information regarding technologies that has been used during creating experimental part. If you have Matlab or R background you may found this usefull.

### Python Relative Material

* https://docs.python.org/3/tutorial/ - Python Tutorial. Originally written by Guido van Rossum (original author of this language).
* https://docs.python.org/3.8/reference/index.html - Python Language Reference.
* https://docs.python.org/3/library/stdtypes.html - Built-in types.
* https://book.pythontips.com/en/latest/index.html - Python tricks.

### Numpy Relative Material

* https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html - Tutorial for people familiar with MATLAB switch to Numpy.
* http://docs.scipy.org/doc/numpy/reference/ - Numpy reference.
* http://docs.scipy.org/doc/numpy/reference/routines.math.html - The full list of mathematical functions provided by Numpy.
* http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html - The full list of Numpy functions for manipulating arrays.
* https://numpy.org/doc/stable/user/basics.broadcasting.html - Numpy broadcasting explanation.

### Matplotlib Relative Material

* https://matplotlib.org/2.0.2/api/pyplot_api.html - Matplotlib documentation.
* https://colab.research.google.com/notebooks/charts.ipynb - Matplotlib examples.

### PyTorch Relative Material

* https://pytorch.org/docs/stable/index.html - PyTorch index for several topics
* https://pytorch.org/tutorials/beginner/ptcheat.html -  The PyTorch Cheat Sheet
* https://pytorch.org/docs/stable/torch.html -  PyTorch API

### FL_PyTorch Relative Material

* https://github.com/burlachenkok/flpytorch - Opensource software suite based on PyTorch to support efficient simulated Federated Learning experiments.
