# Investigating Parameter Sharing in Neural Networks

This project explores technics for sharing parameters within neural networks starting with investigating BatchEnsembles, were the result of several neural networks are being considered. For further exploration three optimizers are being compared and evaluated. Furthermore, the actual sharing of parameters within a neural network as well as an ensemble is being execuded.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Experiments](#experiments)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)

---

## Overview
Ensemble methods are powerful tools for improving the accuracy and reliability of machine learning models. However, they often require a lot of computing power and memory, which can make them difficult to scale. BatchEnsemble provides a solution to this problem by sharing parameters between ensemble members. It uses a common base matrix with small adjustments (low-rank perturbations) that enables efficient parallel processing while maintaining high prediction accuracy. However, as the number of ensemble members increases or tasks become more complex, resource requirements can become a challenge. This study focuses on reducing the number of parameters in neural networks to improve computational efficiency without sacrificing performance. Inspired by BatchEnsemble, we explore new ways of sharing parameters within a single neural network, such as sharing layers or groups of layers. We are applying the benefits of BatchEnsemble to other types of models, including convolutional networks trained on datasets such as CIFAR-10. We are also testing optimization methods such as the IVON optimizer to see how they affect training and performance. The results of this study should provide practical insights into building scalable and resource-efficient machine learning models.

---

## Dataset
The CIFAR-10 dataset contains 60,000 color images, were each image has a size of 32x32.
It is divided into 10 different classes with 6,000 images per class.
The dataset contains 50,000 training images and 10,000 test images.

The CIFAR-20 dataset is an agumented version of the CIFAR-100 dataset. It uses 20 superclasses instead of the full 100 classes from CIFAR-100.

---

## Experiments
The deep neural networks used in this project are on the base of VGG11.
The project implemented following experiments:
- **BatchEnsemble with Parameter Testing**: Testing dependencies between alpha and gamma initiation and testing ensemble size depending on learning rate
- **Comparing Optimizers within Single Neural Network and BatchEnsemble**: Compare Adam, SGD and IVON optimizer within a single deep neural network to the usage of a BatchEnsemble
- **Paramter Sharing within Neural Networks**: Explore parameter sharing between individual networks, between ensemble members and a combination of both

---

## Setup and Installation
### Prerequisites
- Python 3.10+
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/stefanielnz/P7_BatchEnsemble
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

So that everyone could do the tasks evenly and as we do not all have the same resources available, everyone worked differently.
We had to run parameter and optimizer testing on the DTU HPC as it was not possible to run local on our laptops. 
While using the DTU HPC clusters we divided the code into different tasks and .py-files.
We were able to run parameter sharing on a PC. So it was not necessary to use the DTU HPC for this experiments.

Each .py- and .ipynb-file can be normally executed if the requirements are correctly installed and the computational power is enough.

In `data/` are several .csv-files containing results from different experiments from parameter and optimizer testing which are the base for the plots.
Files:
- ParameterTesting_Optimizer.py: loads CIFAR-10, implements simple and BatchEnsemble models, contains training and validation loops based on selection of SGD, Adam and IVON, allows saving of training and validation results for each epoch to csv
- Parameter_Testing_Optimizer_plots: contains code that loads data from all result csv files and plots them in suitable ways

The two notebooks do not produce the plots mentioned in the paper. These are handcrafted in Matlab for easy configuration.
- CIFAR20_ParameterSharing.ipynb: extends the concept of ParamSharing to CIFAR-20 with three different sharing regimens (within model, within ensemble, both combined) in the 4 sharing intensities (no sharing, early layer sharing, deep layer sharing, all layer sharing)
- ParameterSharing.ipynb: trains the VGG11 model on CIFAR-10 with three different sharing regimens (within model, within ensemble, both combined) in the 4 sharing intensities (no sharing, early layer sharing, deep layer sharing, all layer sharing)
