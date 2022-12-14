# This code refers to GPytorch tutorial:
# https://docs.gpytorch.ai/en/v1.5.1/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

##########import##########
import math
from math import floor
import time
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.io import loadmat

import numpy as np
import torch

import warnings; warnings.filterwarnings('ignore')


##########LOAD DATA##########
data_name = 'skillcraft'
print('Data:', data_name)
data = torch.Tensor(loadmat('./data/uci/%s/%s.mat' % (data_name, data_name))['data'])

X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()


##########CUDA##########
print("Device is CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


##########MODEL##########
# Define the DNN+(d->1000->500->50->2->100->1)
data_dim = train_x.size(-1)

model = torch.nn.Sequential(
    torch.nn.Linear(data_dim, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1)
)

if torch.cuda.is_available():
    model = model.cuda()


##########TRAIN##########
training_iterations = 60

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loss
mll = torch.nn.L1Loss()


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Find optimal model hyperparameters
        model.train()
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()


start = time.time()
train()
end = time.time()
print("Training time: %.2f" % (end - start))


##########TEST##########
model.eval()
with torch.no_grad():
    fit = model(train_x)
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds - test_y))))


##########PLOT##########
# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(30, 3))
#
#     # # Get upper and lower confidence bounds
#     # lower, upper = observed_pred.confidence_region()
#     # Plot training data as black line
#     ax.plot(range(0, len(train_y)), train_y.numpy(), 'k', linewidth=0.5)
#     # Plot training data as green line
#     ax.plot(range(0, len(train_y)), fit.numpy(), 'g', linewidth=0.3)
#     # Plot testing data as red line
#     ax.plot(range(len(train_y), len(train_y) + len(test_y)), test_y.numpy(), 'r', linewidth=0.5)
#     # Plot predictive means as blue line
#     ax.plot(range(len(train_y), len(train_y) + len(test_y)), preds.numpy(), 'b', linewidth=0.3)
#     # # Shade between the lower and upper confidence bounds
#     # ax.fill_between(range(len(train_y), len(train_y) + len(test_y)), lower.numpy(), upper.numpy(), alpha=0.5)
#     # ax.set_ylim([-3, 3])
#     ax.legend(['Training Data', 'Testing Data', 'Predictive'])
#     plt.show()