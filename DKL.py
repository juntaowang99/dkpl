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
data_name = 'gas'
print('Data:', data_name)
data = torch.tensor(loadmat('./data/uci/%s/%s.mat' % (data_name, data_name))['data'], dtype=torch.float64)

X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()


##########MODEL##########
# Define the DKL feature extractor(d->1000->500->50->2)
data_dim = train_x.size(-1)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.Sigmoid())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.Sigmoid())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.Sigmoid())
        self.add_module('linear4', torch.nn.Linear(50, 1))


class MaternKernel1D32(torch.nn.Module):
    def __init__(self, length_scale=0.1):
        super(MaternKernel1D32, self).__init__()
        self.nu = 3 / 2
        self.length_scale = torch.autograd.Variable(torch.DoubleTensor([length_scale]), requires_grad=True)

    def forward(self, x1, x2):
        r = torch.cdist(x1.reshape(-1, 1), x2.reshape(-1, 1))
        term = np.sqrt(3) * r / self.length_scale
        return (1 + term) * torch.exp(-term)


class MaternKernel1D52(torch.nn.Module):
    def __init__(self, length_scale=0.5):
        super(MaternKernel1D52, self).__init__()
        self.nu = 5 / 2
        self.length_scale = torch.nn.Parameter(torch.DoubleTensor([length_scale]), requires_grad=True)
        # self.length_scale = length_scale

    def forward(self, x1, x2):
        r = torch.cdist(x1.reshape(-1, 1), x2.reshape(-1, 1))
        term = np.sqrt(5) * r / self.length_scale
        return (1 + term + (term**2)/3) * torch.exp(-term)


class DeepKernelGP(torch.nn.Module):
    def __init__(self, kernel, feature_extractor, sigma_noise=1.0):
        super(DeepKernelGP, self).__init__()
        self.kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        self.feature_extractor = feature_extractor
        self.is_fitted = False

        self.train_x = None
        self.train_y = None
        self.test_x = None

        self.project_x_train = None
        self.project_x_test = None

        self.K_train_train = None
        self.K_train_noise = None
        self.K_train_noise_inv = None
        self.K_test_test = None
        self.K_train_test = None

        self.posterior_mean = None
        self.posterior_cov = None

        # self.prior_mean = torch.autograd.Variable(torch.DoubleTensor([mean]), requires_grad=True)
        self.prior_var = torch.nn.Parameter(torch.DoubleTensor([sigma_noise**2]), requires_grad=True)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def fit(self, x_train, y_train):
        if self.train_x is None:
            self.train_x = x_train
        if self.train_y is None:
            self.train_y = y_train.reshape(-1, 1)

        self.project_x_train = self.feature_extractor(self.train_x)
        self.project_x_train = self.scale_to_bounds(self.project_x_train)  # Make the NN values "nice"

        self.K_train_train = self.kernel(self.project_x_train, self.project_x_train).evaluate()
        self.K_train_noise = self.K_train_train + self.prior_var * torch.eye(len(self.project_x_train))
        self.K_train_noise_inv = torch.linalg.inv(self.K_train_noise)

        self.is_fitted = True

    def nll_loss(self):
        term_data_fit = torch.mm(torch.mm(self.train_y.T, self.K_train_noise_inv), self.train_y)
        term_complex_penalty = torch.log(torch.linalg.det(self.K_train_noise))
        term_const = torch.log(2 * torch.DoubleTensor([torch.pi]))
        nll_loss = (0.5 * term_data_fit + 0.5 * term_complex_penalty)/len(self.project_x_train) + 0.5 * term_const

        if nll_loss.item == torch.nan:
            print("false here")

        return nll_loss

    def predict(self, x_test):
        if not self.is_fitted:
            raise Exception("Haven't been fitted")

        self.test_x = x_test
        self.project_x_test = self.feature_extractor(self.test_x)

        self.K_test_test = self.kernel(self.project_x_test, self.project_x_test).evaluate()
        self.K_train_test = self.kernel(self.project_x_train, self.project_x_test).evaluate()

        term = torch.mm(self.K_train_test.T, self.K_train_noise_inv)
        self.posterior_mean = torch.mm(term, self.train_y)
        self.posterior_cov = self.K_test_test - torch.mm(term, self.K_train_test)

        return self.posterior_mean, self.posterior_cov


deep_net = LargeFeatureExtractor().to(torch.float64)
matern32 = MaternKernel1D32().to(torch.float64)
matern52 = MaternKernel1D52().to(torch.float64)
dkl = DeepKernelGP(kernel=matern52, feature_extractor=deep_net).to(torch.float64)


##########TRAIN##########
epochs = 60

# Find optimal model hyperparameters
dkl.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': dkpl.feature_extractor.parameters()},
#     {'params': dkpl.kernel.parameters()},
#     {'params': dkpl.prior_var}
# ], lr=0.01)
optimizer = torch.optim.Adam(dkl.parameters(), lr=0.001)

forward_time = []
backward_time = []
epoch_time = []
def train():
    iterator = tqdm(range(epochs))
    for i in iterator:
        epoch_start = time.time()
        # Zero backprop gradients
        optimizer.zero_grad()

        forward_start = time.time()
        # Fit GP
        dkl.fit(train_x, train_y)
        # Calc loss and backprop derivatives
        loss = dkl.nll_loss()
        forward_end = time.time()
        forward_time.append(forward_end - forward_start)

        iterator.set_postfix(loss=loss.item())

        backward_start = time.time()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(dkl.parameters(), 10)
        optimizer.step()
        backward_end = time.time()
        backward_time.append(backward_end-backward_start)

        epoch_end = time.time()
        epoch_time.append(epoch_end-epoch_start)


start = time.time()
train()
end = time.time()
print("Training time: %.2f" % (end-start))


# ##########TEST##########
# dkl.eval()
# predict_time = []
# with torch.no_grad():
#     predict_start = time.time()
#     pred_mean, pred_cov = dkl.predict(test_x)
#     predict_end = time.time()
#     predict_time.append(predict_end - predict_start)
#
# print('Test MAE: {}'.format(torch.mean(torch.abs(pred_mean - test_y))))