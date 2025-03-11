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



RESULTS:

1. SGD
<img width="371" alt="Screenshot 2025-03-10 at 11 55 15 PM" src="https://github.com/user-attachments/assets/10833a46-e41f-43b6-a878-ccced932582a" />

2. SGDM
<img width="369" alt="Screenshot 2025-03-10 at 11 55 51 PM" src="https://github.com/user-attachments/assets/c9d7082e-87a9-4492-8e70-0a8975fd1427" />

3. Adagrad
<img width="369" alt="Screenshot 2025-03-10 at 11 56 05 PM" src="https://github.com/user-attachments/assets/42c11502-2b76-4c8c-b841-1c91c3d5e89b" />

4. Adam
<img width="373" alt="Screenshot 2025-03-10 at 11 56 25 PM" src="https://github.com/user-attachments/assets/d2b43e45-f77e-4a75-a6c9-789020aa9441" />

5. RMSProp
<img width="368" alt="Screenshot 2025-03-10 at 11 56 38 PM" src="https://github.com/user-attachments/assets/9078d644-4a90-4d58-932a-364f709a3839" />



