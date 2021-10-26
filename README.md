#  RELEVANCE-BASED PRUNING
This is the repository with the implementation of the proposed method: Relevance-Based Pruning proposed in Nieto et al. See full paper here https://rdcu.be/cyFP3

Extreme learning machines (ELMs) suffer from the presence of unrepresentative features. This means that the number of hidden nodes must increase for the algorithm to achieve good classification results.

The following Gif shows how the amount of unrepresentative features (JF) impacts on the classification performance of an ELM. The number of training examples is 100, coincident with the overfit of the algorithm.

<img src="images/JF_gif.gif" width="800">

In our work, we explain why the overfits occur and propose a method for tuning the number of hidden nodes (M) without the need of retraining the whole network in each step.

We show that our method is available to search for the optimal number of hidden nodes, without jeopardizing classification performance. We mainly focus on the classification of electroencephalography signals, but the method can prove powerful for dealing with unrepresentative features.

## MatLab implementation
We offer an implementation of the traditional ELM and the proposed RBP method in MatLab, along with an example of the aplication of RBP in synthetic data.


## Python implementation
We also offer a python implementation of the traditional ELM and the proposed RBP method.

We develop a Colaboratory demo tutorial, that explains step by step of all the implemented functions.


## Citing this work

Please cite this work.
```bibtex
@article{nieto2021extreme,
  title={Extreme Learning Machine design for dealing with unrepresentative features},
  author={Nieto, Nicol{\'a}s and Ibarrola, Francisco J and Peterson, Victoria and Rufiner, Hugo L and Spies, Ruben},
  journal={Neuroinformatics},
  pages={1--10},
  year={2021},
  publisher={Springer}
}


```
