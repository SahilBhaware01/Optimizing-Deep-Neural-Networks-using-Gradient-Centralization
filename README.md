# Optimizing-Deep-Neural-Networks-using-Gradient-Centralization
Optimizing Deep Neural Networks: The Role of Gradient Centralization in Training and Generalization

Deep learning advancements in visual recognition and natural language processing are driven by improved neural network architectures, large datasets, and better optimization methods. Training deep neural networks (DNNs) is bottlenecked by parameter updates to minimize the loss function, which affects performance and generalization. Optimization techniques like Stochastic Gradient Descent (SGD), Adam, and RMSProp address issues like vanishing gradients and local minima. Adaptive methods such as Adagrad and SGDM further enhance training efficiency. Batch and weight normalization techniques stabilize learning, speed up training, and improve model generalization by normalizing inputs and weights.



OPTIMIZATION TECHNIQUES:

1. Stochastic Gradient Descent(SGD)

2. Gradient Descent with Momentum (GD with momentum)

3. Adaptive Gradient Algorithm (Adagrad)

4. Root Mean Square Propagation (RMSProp)

5. Adaptive Moment Estimation (Adam)



DATASET USED: CIFAR-100



Proposed Technique:


Introduction to Gradient Centralization: Gradient Centralization (GC) is an optimization technique that focuses on centralizing gradients relative to weights, enhancing training dynamics. It is easy to integrate into existing gradient-based optimizers with minimal code changes, offering both theoretical and empirical benefits.

GC improves the optimization process by centralizing gradients, regularizing weight and feature spaces, and creating a more consistent training regimen. It also enhances model generalization, helping prevent overfitting and improving performance on unseen data.

This study aims to compare Deep Neural Networks trained with and without GC, focusing on general image classification. It will evaluate GC’s impact on training efficiency, generalization, and performance metrics, providing insights into its effect on

 <img width="587" alt="Screenshot 2025-03-10 at 11 50 28 PM" src="https://github.com/user-attachments/assets/03ae9ef5-70c1-4628-a637-88d51236ec0b" />

