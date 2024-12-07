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

CIFAR-20

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
- Python 3.7+
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
