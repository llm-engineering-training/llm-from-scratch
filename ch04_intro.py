from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from importlib.metadata import version

# Below we are going to access the weight and bias matrix
layer = torch.nn.Linear(in_features=128, out_features=256)
#print("layer weight:\n", layer.weight)
#print("layer bias:\n", layer.bias)

#the weight and bias matrix are initially set using the Kaiming uniform or HE method
#parameters are what get optimized
""" for name, param in layer.named_parameters():
    print(name, param.shape) """

#Non-linear activation. Model so far above is constrained by linearity
# ReLU-Rectified Linear Unit - cheap to compute. In ReLU some nuerons die hidden 
# #layer default, zero for all negative values - used everywhere
# Leaky RelU adds a small value for negative values
# tanh(x) zero centered - low dimensional cases - good for zero centered
# sigmoid() - not recommended in hidden layers. Suffers from slow convergence - used for binary classification
# GELU - performs well - default in large scale applications

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

#print("Model:\n", model)

# for custom forward methods - this function was never tested or ran
""" def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    return x """


# The loss function and why it is essential to the training process
# loss function provides feedback to the model. 
# Tells the optimizaer on how to adjust weights for improvements
# loss - predicted minus actual
#   always a scaler, gets lower as the model improves
#   RMSE - Regression mean-squared error
#   Categprical Cross Entropy - loss function used for categorical data
# Backward Pass - use loss to compute gradients
# Optimizer Step -  
# Gradients is a partial derivative of loss with respect to each parameter
# change in loss / unit change in each weight
# Loss function must be batch-compatible and task-appropriate
#


# .backward() -  work backwrds and figure out how every weight contributed to the error
#   computes gradients of the loss function
# pytorch autograd - automatic differentiation
# 
# 
# 
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
c = a * b
d = c + 1
loss = d**2
loss.backward()
#a.grad
#print("gradient of b:\n", b.grad)
# For the simple example above
#   loss is (a * b + 1)^2
#   gradient w.r.t. a is 2*(a * b +1) \ c.b
#   gradient w.r.t. b is 2*(a * b +1) \ c.a
# when required_grad=True Pytorch tracks all operations
# 
# 1- forward pass - linear layer, attention blocks etc
# 2 -  compute scalar value - the loss
# 3 - backward pass
# 4 -  parameter updates in the optimization -
# 5 - zero out the gradients for every training loop
# 

# For the simple example above
# Gradient is a measure of change - we calculate this using the chain-rule of partial derivatives
# How are gradients computed
# 
# 
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3,5)
        self.fc2 = nn.Linear(5,2)

    def forward(self, x):
        x =F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
x_input = torch.randn(1,3)
target = torch.tensor([[0.0, 1.0]]) # these are the actuals
output = model(x_input)
loss = F.mse_loss(output, target)

loss.backward()

for name, param in model.named_parameters():
    print(f"{name} grad:\n{param.grad}")

# Break in gradient flow
#
#
#
#
#
