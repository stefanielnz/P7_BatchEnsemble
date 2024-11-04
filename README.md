# Parameter Sharing in Neural Networks: 02456 Deep Learning Course project

**BatchEnsemble** is a method for training ensembles of neural networks with shared parameters, allowing for efficient training and improved generalization.

## Getting Started

After reading the BatchEnsemble paper, you can start by exploring the provided code examples:

* **experiments/skafte/00_batchensemble_mlp.py:** An example experiment demonstrating how to train a BatchEnsemble MLP on the Skafte synthetic dataset from <https://github.com/SkafteNicki/john>.
* **src/parameter_sharing/models/batchensemble.py:** A Python implementation of a BatchEnsemble layer using a mixin class. This implementation provides a flexible and modular approach to incorporating BatchEnsemble into your neural network layers.

This repository uses poetry and a package structure to manage dependencies and project structure. To get started:

1. **Install Poetry:** If you don't have Poetry installed, follow the instructions at <https://python-poetry.org/docs/#installation>
2. **Python Version:** This project requires Python 3.10 or above. You can use `pyenv` to manage different Python versions if needed.
3. **Create a Virtual Environment:** Navigate to the project root and run `poetry install` to create a virtual environment and install the required dependencies.
4. **Activate the Environment:** Activate the virtual environment using `poetry shell`.

After installation and activation you can run individual scritps with eg. `python experiments/skafte/00_batchensemble_mlp.py`.

## Resources

* **BatchEnsemble Paper:** <https://arxiv.org/abs/2002.06715>
* **Alternative BatchEnsemble Implementation:** <https://github.com/giannifranchi/LP_BNN/tree/main>
* **BatchEnsemble for BNNs using Rank-1 factors:** <https://arxiv.org/abs/2005.07186>
* **IVON Optimizer:** <https://github.com/team-approx-bayes/ivon>
* **02456 Deep Learning Course Repository:** <https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch>

## Next Steps

* **Explore Convolutional Layers:** Implement BatchEnsemble for convolutional layers and evaluate its performance on image datasets.
* **Compare with IVON:** Investigate variational training of BatchEnsemble layers using the IVON optimizer.
* **Implement Parameter Sharing Within Single Network:** Explore how to apply parameter sharing within a single network, potentially using techniques like weight tying or shared layers.
* **Investigate Bayesian Neural Networks:** Explore the use of BatchEnsemble for training Bayesian neural networks and analyze its impact on uncertainty estimation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
