# Investigating Parameter Sharing in Neural Networks

This project explores technics for sharing parameters within neural networks starting with investigating BatchEnsembles, were the result of several neural networks are being considered. For further exploration three optimizers are bein compared and evaluated. Furthermore, the actual sharing of parameters within a neural network as well as an ensemble is being execuded.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Experiments](#experiments)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)

---

## Overview


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
- **BatchEnsemble with Parameter Testing**: Using BatchEnsemble containing of several deep neural networks and test out different values for paramters alpha and gamma 
- **Comparing Optimizers within Single Neural Network and BatchEnsemble**: Compare Adam, SGD and IVON optimizer within a single deep neural network to the usage of a BatchEnsemble
- **Paramter Sharing within Neural Networks**: Explore parameter sharing between individual networks, between ensemble members and a combination of both

---

## Setup and Installation
### Prerequisites
- Python 10+
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
Due to the use of DTU HPC clusters the code is split into different tasks and .py-files.

Each .py-file can be normally executed if the requirements are correctly installed and the computational power is enough.

In `data/` are several .csv-files containing results from different experiments which are the base for the plots.
Files:
- ParameterTesting_Optimizer.py: loads CIFAR10, implements simple and batchensemble models, contains training and validation loops based on selection of SGD, Adam and IVON, allows saving of training and validation results for each epoch to csv
- Parameter_Testing_Optimizer_plots: contains code that loads data from all result csv files and plots them in suitable ways

The two notebooks do not produce the plots mentioned in the paper. These are handcrafted in Matlab for easy configuration.
- CIFAR20_ParamSharing.ipynb: extends the concept of ParamSharing to CIAFR20 with three different sharing regimens (within model, within ensemble, both combined) in the 4 sharing intensities (no sharing, early layer sharing, deep layer sharing, all layer sharing)
- ParamSharing.ipynb: trains the VGG11 model on CIFAR10 with three different sharing regimens (within model, within ensemble, both combined) in the 4 sharing intensities (no sharing, early layer sharing, deep layer sharing, all layer sharing)
