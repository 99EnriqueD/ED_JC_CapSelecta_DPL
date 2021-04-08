# Neural-Symbolic Capita Selecta project

We present two fashion applications, budget and outfit, powered by [DeepProbLog](https://bitbucket.org/problog/deepproblog/src). These were developed for the [Capita Selecta AI course](https://www.onderwijsaanbod.kuleuven.be/syllabi/e/H05N0AE.htm#activetab=doelstellingen_idp4361008) at the KU Leuven. The report for the project can be found in the root of the repository.

# Overview

All scripts to train and evaluate models are in the root of the repository.

## data
All data needed to train and evaluate all models. This includes FashionMNIST data, DeepFashion data (contact authors for more information about integrating this here), and custom generated data files.

## data_generators
Scripts that create custom txt files that are needed to orchestrate data for training and testing.

## DeepProbLog 
Neccessary DeepProbLog language files.

## metrics
Logged metric data and scripts

## pl_files
DeepProbLog fashion application code. 

## neural_networks
Neural networks used for both DeepProbLog programs as well as baselines.

## util
Miscellaneous files.
