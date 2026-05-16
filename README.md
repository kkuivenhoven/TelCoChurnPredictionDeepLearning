# TelCo Churn Prediction Deep Learning (4 layer Neural Network from Scratch)

A neural network built using 4 layers (2 hidden) to find the most likely features to predict customer cancellation.

The engine uses NumPy instead of a library like Scikit-learn or Tensorflow. The purpose of this Neural Network is to simply take the features presented in the data file and determine the top 5 most likely categories to predict customer cancellation.

## Project Overview
# Data Inputs
- TelCo Customer Churn data

### The Brain
A four layer network with 20 neurons to process the relationships between features.

### The Math
The function sigmoid was used to force the final output to be a percentage between 0 and 1. This is the probability of churning. The ReLU function was also used as well to handle the hidden layers.

Through the manual writing of calculus for forward and backward pass, I learned how the neural network would update its derivatives of weights, biases, activation values and its z values.

### Data Preprocessing (Normalization)
The data had to be normalized using mean and standard deviation. The reason for this is that the category SeniorCitizen has values of 0 or 1. However, the category MonthlyCharges could be 5 to 50 in range of values. Without normalization, the math would've been considered "brittle". Thus, normalizing every feature to have a mean of 0 and a standard deviation prevents "brittle" math and weights that could've been really small for categories like MonthlyCharges otherwise.


### Challenges & Discoveries
I found that without feeding np.random a seed of 42, the categories that it would print as it's final result were not predictably the same. Once this was added as a sort of "stabilizer", the categories consistently printed the same each time.

### Installation
To run the script:
```
python churnPrediction.py
```
