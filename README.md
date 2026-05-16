# TelCo Churn Prediction Deep Learning (4 layer Neural Network from Scratch)

A neural network built using 4 layers (2 hidden) to find the most likely features to predict customer cancellation.

The engine uses NumPy instead of a library like Scikit-learn or Tensorflow. The purpose of this Neural Network is to simply take the features presented in the data file, and determine the top 5 most likely categories to predict customer cancellation.

# Project Overview
### Data Inputs
* Dataset: TelCo Customer Churn data
* Feature Dimension: 45-input feature matrix (processed via bulk one-hot encoding in Pandas)

### Network Architecture
The engine utilizes a 4-layer neural network to compress high-dimensional customer profiles into a single binary prediction path. The network layers are configured as follows:
* Input Layer: 45 features
* Hidden Layer 1: 20 neurons (ReLU activation)
* Hidden Layer 2: 7 neurons (ReLU activation)
* Output Layer: 1 neuron (Sigmoid activation)

### The Math
* Activations: The Sigmoid function was used to force the final output to be a percentage between 0 and 1. This is the probability of churning. The ReLU function was also used as well to handle the hidden layers.

* Optimizations: Through the manual writing of calculus for forward and backward pass, I learned how the neural network would update its derivatives of weights, biases, activation values and its z values.

### Data Preprocessing (Normalization)
The data had to be normalized using mean and standard deviation. The reason for this is that the category `SeniorCitizen` has values of 0 or 1. However, the category `MonthlyCharges` could be 20 to 120+ in range of values. Without normalization, the math would've been considered "brittle". Unscaled large numbers like `MonthlyCharges` cause massive gradient updates that dominate the network. In turn, this would force the weights for smaller features (like `SeniorCitizen`) to scale down or oscillate wildly. Thus, normalizing every feature via a strict Z-score normalization to have a mean of 0 and a standard deviation of 1 prevents this "brittle" math and ensures stable weight updates across all categories.


### Challenges & Discoveries
I found that without feeding np.random a seed of 42, the categories that it would print as its final result were not predictably the same every run due to random weight initialization. I deliberately removed this ambiguity by adding a random seed stabilizer.  Once this was added, the categories consistently printed the same each time meaning the training path was perfectly repeatable.

### Installation
Ensure you have `numpy` and `pandas` installed, then run the script:
```
python churnPrediction.py
```
