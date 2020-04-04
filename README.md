# Stochastic Differentiable NAS Benchmarking
Please note that this is essentially a fork of [the official repository for GDAS](https://github.com/D-X-Y/AutoDL-Projects) and that repository contains a description for the software required to run this code and documentation for most of its contents.

The only difference in use from the above repository is that I have changed the significance of the command line arguments passed to the search script. Instead of (dataset, track_batch_norm, random_seed) the three required arguments passed to the search script on the command line are now (bilevel, resource_constraint, random_seed). The dataset has been fixed as CIFAR10 and tracking the batch norm is disabled. 
A 1 for the first argument will enable bilevel training, optimizing the architecture weights separately on the validation data.
A 1 for the second argument will enable a resource constraint, adding a regularization term that is a function of the compuational and memory costs of the model.

The syntax for running the base experiments therefore is:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh 0 0 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/SNAS.sh 0 0 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/ProxylessNAS.sh 0 0 -1
```
