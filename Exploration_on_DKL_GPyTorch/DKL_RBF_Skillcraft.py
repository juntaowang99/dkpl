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
import gpytorch

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
# Define the DKL feature extractor(d->1000->500->50->2)
data_dim = train_x.size(-1)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))


feature_extractor = LargeFeatureExtractor()


# Define the DKL GP Regression Model(RBF kernel)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


##########TRAIN##########
training_iterations = 60

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()


start = time.time()
train()
end = time.time()
print("Training time: %.2f" % (end-start))


##########TEST##########
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    fit = model(train_x)
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))


##########PLOT##########
# observed_pred = likelihood(preds)
# observed_fit = likelihood(fit)

# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(30, 3))
#
#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black line
#     ax.plot(range(0, len(train_y)), train_y.numpy(), 'k', linewidth=0.5)
#     # Plot training data as green line
#     ax.plot(range(0, len(train_y)), fit.mean.numpy(), 'g', linewidth=0.3)
#     # Plot testing data as red line
#     ax.plot(range(len(train_y), len(train_y)+len(test_y)), test_y.numpy(), 'r', linewidth=0.5)
#     # Plot predictive means as blue line
#     ax.plot(range(len(train_y), len(train_y)+len(test_y)), observed_pred.mean.numpy(), 'b', linewidth=0.3)
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(range(len(train_y), len(train_y)+len(test_y)), lower.numpy(), upper.numpy(), alpha=0.5)
#     # ax.set_ylim([-3, 3])
#     ax.legend(['Training Data', 'Fitted data', 'Testing Data', 'Predictive Mean', 'Confidence'])
#     plt.show()