# Project: The Bayesian Learning Rule

This project is based on the paper titled "The Bayesian Learning Rule" by Mohammad Emtiyaz Khan and Håvard Rue, published in 2023. The project was carried out by Alexandre François, Sacha Braun, and Hippolyte Pilchen, all reachable at forename.lastname@polytechnique.edu. It was completed as part of the Bayesian Machine Learning class taught by Rémi Bardenet and Julyan Arbel for the Master 2 MVA program.

## Abstract

The Bayesian Learning Rule (BLR) is a principle that unifies a wide array of Bayesian and non-Bayesian algorithms by focusing on the minimization of expected loss within a geometric framework defined by a selected posterior distribution. This framework incorporates natural gradients, which provide insights into the curvature of the loss function, facilitating the optimization process. By adopting a Bayesian approach and applying appropriate approximations, it becomes possible to derive established algorithms, with their complexity being directly linked to the chosen posterior's complexity. Our contribution starts with an analysis of the increased solutions stability and robustness provided by the BLR, and an extension of the unifying perspective of the BLR from which we derived Stochastic Average Gradient. Additionally, we apply BLR principles to refine existing algorithms, notably NoisyNet.


## Experiment Highlights and Contributions

This document outlines our original contributions and experimental highlights, focusing on the Bayesian versions of algorithms to showcase their smoothing effects and stability. Our research delves into three key experiments, each with its unique focus and contribution to the field of machine learning and Bayesian inference.

### Experiment 1: Smoothing Effect and Robustness

We propose an experiment to showcase the smoothing effect and the robustness of solutions provided by the Bayesian versions of algorithms. This experiment is designed to highlight how Bayesian approaches can stabilize the learning process and yield more reliable solutions across a range of conditions. To remake experiments, you can run notebooks flat_regions.ipynb and MLP_esp.ipynb. The first one tries to give some insights on the role of the Bayesian approach and tries to understand to what extent it can provide more robust solutions. The second one illustrates these principles through a toy example of a from-scracth MLP that performs classification, and compares to more conventional approaches in terms of robustness and stability. 


<p align="center">
  <img src="/noisyNet/results/blr.gif" alt="Robustness_BLR" width="50%" height="auto">
  <br>
  <em>Smoothing effects and stability of the BLR</em>
</p>

### Experiment 2: Stochastic Average Gradient Descent (SAG)

As an additional contribution, we introduce a novel learning algorithm based on the universality of the Bayesian Linear Regression (BLR). This algorithm, derived for Stochastic Average Gradient Descent (SAG), has not been previously described. Our experiment aims to underline the versatility and applicability of BLR in optimizing gradient descent methods in a Bayesian framework.

### Experiment 3: Connection between BLR and NoisyNet

Our research establishes a direct connection between BLR and NoisyNet, a well-known reinforcement learning algorithm. NoisyNet is designed to minimize the loss incurred by a neural network perturbed by Gaussian noise. The original NoisyNet algorithm utilizes parameter updates derived from Euclidean geometry. However, upon reviewing the literature, specifically Khan et al. (2023), we argue that employing natural gradients could significantly enhance the update process. This experiment proposes an alternative optimizer tailored for this neural network, highlighting the potential improvements in efficiency and effectiveness. In light of our review and findings, we propose the use of natural gradients as a more effective method for updating parameters in neural networks, especially those influenced by Bayesian Linear Regression and algorithms like NoisyNet. Our proposed optimizer aims to refine the updating process, making it more aligned with the underlying statistical properties of the model.

## Conclusion

Through these experiments, we aim to demonstrate the inherent benefits of Bayesian methods in machine learning, from enhancing stability and robustness to offering novel approaches to algorithm optimization. Our work underscores the importance of continuing to explore and integrate Bayesian principles into machine learning to uncover more effective and efficient solutions, but also to derive new algorithms that will rely on information geometry. 


## Requirements



## Usage




