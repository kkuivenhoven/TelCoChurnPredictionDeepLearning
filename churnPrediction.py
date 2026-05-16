# 4-Layer Deep Neural Network (MLP) from scratch
# Binary classification: Customer Churn Prediction (0 = Retained, 1 = Churned/lost)
#
# Topology: [45 input features] -> [20 hidden] => [7 hidden] -> [1 output probability]
#
# Correct order of operations:
# 1. Forward propagation:
# 	- Linearly combine inputs and weights, then apply activation functions layer by layer.
# 	- Z1, A1 (ReLU) -> Z2, A2 (ReLU) -> Z3, A3 (ReLU) -> Z4, A4 (Sigmoid/Y-hat)
# 2. Compute Cost:
# 	- Run Binary Cross-Entropy Loss to measure the mathematical delta between the final
#     output vector (A4) and the true ground-truth labels (Y).
# 3. Backward Propagation:	
#	- Execute the chain rule backward through all 4 layers to compute partial derivatives.
# 	- Track loss gradients down to dW and db for every single layer matrix
# 4. Update parameters:
#	- Apply gradient descent updates to shift the weight and bias matrices in the opposite
#	  direction of the cost gradient (W = W - alpha * dW)
# 5. Iterative Optimization Loop:
#	- Repeat steps 1-4 across 3,000 iterations using a fixed random seed stabilizer until
#	  the model converges and weight distributions settle.

import numpy as np
import pandas as pd

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s

def predict(parameters, X, layer_dims):
	_cache, size = forward_propagation(X, layer_dims, parameters)
	last_A_value = _cache["A" + str(size)]	
	predictions = (last_A_value > 0.5).astype(int)
	return predictions

def forward_propagation(X_norm, layer_dims, parameters):
	# Forward pass belongs here - think of it as like taking a practice exam
	_cache = {}	
	_cache["A" + str(0)] = X_norm
	for l in range(1, len(layer_dims) - 1):
		# The "Forward Pass" Architecture
		_cache["Z" + str(l)] = np.dot(parameters["W" + str(l)], _cache["A" + str(l-1)]) + parameters["b" + str(l)]
		# ReLU function
		_cache["A" + str(l)] = np.maximum(0, _cache["Z" + str(l)])
    
	size = ((len(layer_dims)) - 1)
	_cache["Z" + str(size)] = np.dot(parameters["W" + str(size)], _cache["A" + str(size-1)]) + parameters["b" + str(size)]
	_cache["A" + str(size)] = sigmoid(_cache["Z" + str(size)])
	return _cache, size

np.random.seed(42)

dataFrame = pd.read_csv('WA-Customer-Churn.csv')

dataFrame["TotalCharges"] = pd.to_numeric(dataFrame["TotalCharges"].str.strip(), errors='coerce').fillna(0).astype(float)
# pd.to_numeric(...) handles conversion
# errors = 'coerce' turns anything invalid (like ' ', '', or non-numeric text) into NaN
# the result is a float column automatically (since NaN requires float)

dataFrame = dataFrame.drop('customerID', axis=1)
# below encodes every string column at once and keeps the numerical columns (Charges/Tenure) safe.
# This is a "bulk move" rather than calling out each column at once
dataFrame = pd.get_dummies(dataFrame)
dataFrame = dataFrame.astype(float)

Y_df = dataFrame["Churn_Yes"] # Use Churn_Yes (not Churn_No) since want the model to predict the
						      # probability of the "positive" event (churning)

dataFrame = dataFrame.drop('Churn_No', axis=1)
dataFrame = dataFrame.drop('Churn_Yes', axis=1)

true_network_features = dataFrame.columns.tolist()

# Shape is (m, nx) where m is rows (customers) and nx is columns (features)
# for this, need (nx, m)

# Transpose such that each column represents a customer and each row represents a feature
Y_df = Y_df.values.reshape(-1, 1)
Y_df = Y_df.T
X_df = dataFrame.T
X_df = X_df.to_numpy()

# need to normalize (level the playing field) since have features with completely different
# "energy levels". I.e. monthly charges ranging from 20 to 120, but SeniorCitizen is 0 or 1
# and Gender_male is 0 or 1.
# IF we do NOT normalize, the W weights connected to MonthlyCharges will have to be tiny, while
# the weights for Gender will have to be huge. This results in "brittle" math or scale sensitivity. 
# We need every feature to have a mean of 0 and a standard deviation of 1.

# Our X at this point is (45, 7043) - Axis 0 is 45 features while Axis 1 is 7043 customers
mu = np.mean(X_df, axis = 1, keepdims=True)
std_dev = np.std(X_df, axis = 1, keepdims=True)

# Z-score normalization. "Tames" every feature. We had to normalize the data to prevent a 
# form a bias forming such as a "Protein Bias" that we discovered in our previous project.
X_norm = ((X_df - mu)/std_dev)

print(X_norm.shape)
# [Input, Hidden1, Hidden2, Output]
layer_dims = [45, 20, 7, 1]
learning_rate = 0.01

# initialize the parameters
parameters = {}
cache = {}
for l in range(1, len(layer_dims)):
	parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * learning_rate
	parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

for i in range(3000):
	# Forward pass belongs here - think of it as like taking a practice exam
	cache, size = forward_propagation(X_norm, layer_dims, parameters)

    # --- COMPUTE COST ---
    # We need to see how wrong A3 is compared to Y.
	m = X_df.shape[1] # m is the # of the columns in the matrix X_df
	inner = (Y_df * np.log(cache["A" + str(size)])) + (1 - Y_df)*(np.log(1-cache["A" + str(size)]))
	cost = (-1)*(1/m)*np.sum(inner)
    
	grads = {}
    # --- BACKWARD PASS ---
    # Finding dW and db using the chain rule.
	grads["dZ" + str(size)] = (cache["A" + str(size)] - Y_df)
	for l in range(len(layer_dims) - 1, 0, -1):
		# A. Calculate gradients for current layer (l)
		# use the error signal dZ_l to find how much W and b need to change
		grads["dW" + str(l)] = (1/m) * np.dot(grads["dZ" + str(l)], cache["A" + str(l-1)].T)
		grads["db" + str(l)] = (1/m) * np.sum(grads["dZ" + str(l)], axis = 1, keepdims=True)
		
		# B. The HAND-OFF (prepare for the layer behind)
		if l > 1:
			# Step 1: pass the "blame" backward across the weights
			# i.e. calculate the blame for the previous layer
			grads["dA" + str(l-1)] = np.dot(parameters["W" + str(l)].T, grads["dZ" + str(l)])

			# Step 2: apply the "activation gate" (the ReLU derivative)
			# step a1: create the "gate" (the ReLU derivative)
			# this creates a matrix of 1s and 0s
			# 1.0 if Z > 0, else 0.0
			relu_derivative_gate = (cache["Z" + str(l-1)] > 0).astype(float)

			# step a2: multiply the Blame by the Gate
			# this is the element-wise multiplication
			# only the "blame" where the gate was 1.0 survivies
			dZ_prev = grads["dA" + str(l-1)] * relu_derivative_gate

			# Obtain the "error signal" for the previous layer (dZ[l-1])
			grads["dZ" + str(l-1)] = dZ_prev
		
    
    # --- STEP E: UPDATE PARAMETERS ---
    # Your circular logic: W = W - learning_rate * dW
	for l in range(1, len(layer_dims)):
		parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
		parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

	if i % 100 == 0:
		print(f"Cost after iteration {i}: {cost}")


predictions = predict(parameters, X_norm, layer_dims)
accuracy = np.mean(predictions == Y_df)

print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

parameter_W1 = parameters["W1"]
parameter_W1 = np.abs(parameter_W1)
mean_values = np.mean(parameter_W1, axis=0)
max_index = np.argmax(mean_values)

all_sorted_indices = np.argsort(mean_values)
top_5_indices = all_sorted_indices[-5:][::-1]
print("Top 5 Most Predictive Features:")
for rank, index in enumerate(top_5_indices, 1):
	print(f"{rank}. {true_network_features[index]} (Score: {mean_values[index]:.6f})")
