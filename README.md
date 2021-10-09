# ExactBayesianNetworkPriors
Code to generate figures from Zavatone-Veth and Pehlevan, "Exact marginal prior distributions of finite Bayesian neural networks" (*NeurIPS* 2021).

## Description

This directory contains [MATLAB](https://www.mathworks.com/products/matlab.html) code to reproduce the figures in [our paper](https://arxiv.org/abs/2104.11734). It has been tested in versions 9.5 (R2018b) and 9.8 (R2020a), and requires the [`meijerG`](https://www.mathworks.com/help/symbolic/meijerg.html) function from the [Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html). 

## Getting started

The figures may be generated as follows:

- Figure 1: First run the script `DeepLinearNetworkPrior.m` and then run the script `PlotPriorResults.m`. 
- Figure 2: Run the script `PlotBottleneckPrior.m`.
- Figure 3: First run the script `DeepReluNetworkPrior.m` and then run the script `PlotPriorResults.m`. 
- Figure 4: Run the script `PlotEdgeworthApproximation.m`.

Evaluation of the theoretical ReLU network prior is quite slow, as the computationally-expensive [`meijerG`](https://www.mathworks.com/help/symbolic/meijerg.html) function must be evaluated many times (see Appendix E of our paper for details). To obtain a cruder, faster approximation, increase the threshold set in Line 62 of `DeepReluNetworkPrior.m` from `eps` to a larger value. 
