# A Generative Adversarial Framework for Bounding Confounded Causal Effects
This is the official implementation of [A Generative Adversarial Framework for Bounding Confounded Causal Effects](). 
Neural networks are written using PyTorch-1.4.0. (Make sure the PyTorch version is correct)

## Objective
We develop a bounding method for estimating the average causal
effect (ACE) under unidentfiable situations due to hidden confounders.

## Overview
Directories:
+ **data** contains the Adult and Dutch datasets.
+ **synthetic** contains the experimental code on synthetic data.
+ **adult** contains the experimental code on Adult dataset.
+ **dutch** contains the experimental code on Dutch dataset.

## Reproduction
To reproduce the experiments on synthetic data using linear model:
1. Go to synthetic/linear_model directory.
2. Run "python main.py".

Other experiments are similar.
