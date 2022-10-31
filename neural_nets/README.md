# How to Launch experiments with Neural Nets for our paper.

## Prerequisites

The experiments for Neural Nets are constructed via modifying FL_PyTorch [https://arxiv.org/abs/2202.03099](https://arxiv.org/abs/2202.03099), [GitHub link](https://github.com/burlachenkok/flpytorch).

This simulator is constructed based on the PyTorch computation framework. The first step is preparing the environment. 
Preparation of environment described in [README.md](https://github.com/burlachenkok/flpytorch/blob/main/README.md) files of this publicly available repository.

If you have installed [conda](https://docs.conda.io/en/latest/) environment and package manager then you should perform only the following steps for preparing the environment.

```
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy cudatoolkit"=11.1" h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
conda activate fl
```

Our experiments have been carried out utilizing computation in NVIDIA GPUs.

Our modification of the simulator is located in `./fl_pytorch`. Use this version that we're providing instead of the Open Source version.

## Place with Execution Command Lines

Change the working directory to `"./fl_pytorch"`. The next BASH scripts contain a command line for launching the computation work:
* `cmdline_for_experiment_all_layers_public.sh` 
* `cmdline_for_experiment_last_linear_layer_only_public.sh`

If you want to use [WandB](https://wandb.ai/settings) online tool to track the progress of the numerical experiments please specify:
* `--wandb-key "xxxxxxxxxxx" ` with a key from your wandb profile: [https://wandb.ai/settings](https://wandb.ai/settings
* `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use.
You should replace `--wandb-project-name "vvvvvvvvvv"` with a project name that you're planning to use or leave the default name. Both of these keys should be replaced manually if you're interested in WandB support.

## Visualization of the Results

Result binary files can be loaded into the simulator `fl_pytorch\fl_pytorch\gui\start.py`. After this plots can be visualized in *Analysis* tab. Recommendations on how to achieve this are available in [TUTORIAL.md](https://github.com/burlachenkok/flpytorch/blob/main/TUTORIAL.md) provided with [flpytorch](https://github.com/burlachenkok/flpytorch) simulator.
