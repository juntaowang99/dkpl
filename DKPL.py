##########import##########
import math
from math import floor
import time
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import cm

import scipy
from scipy.io import loadmat
import scipy.sparse.linalg

import numpy as np
import torch
import gpytorch

##########LOAD DATA##########
data_name = 'gas'
print('Data:', data_name)
data = torch.tensor(loadmat('./data/uci/%s/%s.mat' % (data_name, data_name))['data'], dtype=torch.float64)

X = data[:, :-1]
# Scaling data
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1

y = data[:, -1]

# Split train and test data sets
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
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 1))


class DeepKernelPacketGP(torch.nn.Module):
    def __init__(self,
                 kernel: gpytorch.kernels.MaternKernel,
                 feature_extractor,
                 sigma_noise=1.0):
        super(DeepKernelPacketGP, self).__init__()
        self.kernel = kernel
        self.feature_extractor = feature_extractor
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        # self.prior_mean = torch.autograd.Variable(torch.DoubleTensor([mean]), requires_grad=True)
        self.prior_var = torch.nn.Parameter(torch.DoubleTensor([sigma_noise**2]), requires_grad=True)
        self.is_fitted = False

        self.train_x = None
        self.train_y = None
        self.test_x = None

        self.project_x_train = None
        self.project_x_test = None

        self.phi_project_x_train = None
        self.A_project_x_train = None
        self.phi_A_noise = None
        self.phi_A_noise_inv = None

        self.K_test_test = None
        self.K_train_test = None
        self.phi_project_x_test_T = None

        self.posterior_mean = None
        self.posterior_cov = None

    def fit(self, x_train, y_train):
        if self.train_x is None:
            self.train_x = x_train
        if self.train_y is None:
            self.train_y = y_train.reshape(-1, 1)

        self.project_x_train = self.feature_extractor(self.train_x)
        self.project_x_train = self.scale_to_bounds(self.project_x_train)  # Make the NN values "nice"

        # Kernel Packet
        self.A_project_x_train, self.phi_project_x_train = self.compute_basis(self.project_x_train, int(2 * self.kernel.nu + 2), self.kernel.lengthscale)
        self.A_project_x_train, self.phi_project_x_train = self.A_project_x_train.to_dense(), self.phi_project_x_train.to_dense()

        # self.phi_A_noise = self.phi_project_x_train + torch.sparse.mm((self.prior_var * torch.eye(len(self.project_x_train))).to_sparse(), self.A_project_x_train)
        # self.phi_A_noise_inv = torch.tensor(
        #     scipy.sparse.linalg.inv(
        #         scipy.sparse.coo_matrix(
        #             (self.phi_A_noise._values(),
        #              (self.phi_A_noise._indices()[0],
        #               self.phi_A_noise._indices()[1])),
        #             self.phi_A_noise.size()).tocsc()).toarray(),
        #     dtype=torch.float64,
        #     requires_grad=True)

        self.phi_A_noise = self.phi_project_x_train + self.prior_var * self.A_project_x_train
        self.phi_A_noise_inv = torch.tensor(
            scipy.sparse.linalg.inv(
                scipy.sparse.csc_matrix(self.phi_A_noise.detach().numpy())
            ).toarray(),
            dtype=torch.float64,
            requires_grad=True)
        # self.phi_A_noise_inv = torch.linalg.inv(self.phi_A_noise)

        self.is_fitted = True

    def nll_loss(self):
        term_data_fit = torch.mm(torch.mm(torch.mm(self.train_y.T, self.A_project_x_train), self.phi_A_noise_inv), self.train_y)

        term_complex_penalty = torch.log(torch.linalg.det(self.phi_A_noise)) - torch.log(torch.linalg.det(self.A_project_x_train))

        term_const = torch.log(2 * torch.DoubleTensor([torch.pi]))
        nll_loss = (0.5 * term_data_fit + 0.5 * term_complex_penalty)/len(self.project_x_train) + 0.5 * term_const

        return nll_loss

    def predict(self, x_test):
        if not self.is_fitted:
            raise Exception("Haven't been fitted")

        self.test_x = x_test
        self.project_x_test = self.feature_extractor(self.test_x)

        self.K_test_test = self.kernel(self.project_x_test, self.project_x_test).evaluate()
        self.K_train_test = self.kernel(self.project_x_train, self.project_x_test).evaluate()
        self.phi_project_x_test_T = torch.mm(self.K_train_test.T, self.A_project_x_train)

        term = torch.mm(self.phi_project_x_test_T, self.phi_A_noise_inv)
        self.posterior_mean = torch.mm(term, self.train_y)
        self.posterior_cov = self.K_test_test - torch.mm(term, self.K_train_test)

        return self.posterior_mean, self.posterior_cov

    def compute_basis(self, x, k, rho):
        # initialization
        n = x.shape[-2]

        row_A = np.array([])
        col_A = np.array([])
        values_A = np.array([])
        row_A_left_sided = np.array([])
        col_A_left_sided = np.array([])
        values_A_left_sided = np.array([])
        row_A_right_sided = np.array([])
        col_A_right_sided = np.array([])
        values_A_right_sided = np.array([])

        row_phi_x_x = np.array([])
        col_phi_x_x = np.array([])
        values_phi_x_x = np.array([])
        row_phi_x_x_left_sided = np.array([])
        col_phi_x_x_left_sided = np.array([])
        values_phi_x_x_left_sided = np.array([])
        row_phi_x_x_right_sided = np.array([])
        col_phi_x_x_right_sided = np.array([])
        values_phi_x_x_right_sided = np.array([])

        # calculate intermediate basis phi_[(k+1)/2 : (n-(k-1)/2)]
        for i in range(int((k - 1) / 2), int(n - (k - 1) / 2)):
            # # standardize x by x + t, where t = -(xmin + xmax)/2
            # # t = - (x[int(i - (k - 1) / 2)] + x[int(i + (k - 1) / 2)]) / 2
            # # x_tran = x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1] - (x[int(i - (k - 1) / 2)] + x[int(i + (k - 1) / 2)]) / 2
            # # x_tran = x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1]
            # x_temp = (x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1] - (x[int(i - (k - 1) / 2)] + x[int(i + (k - 1) / 2)]) / 2) ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            #
            # ### calculate A
            # c = (k - 2) ** 0.5 / rho
            # delta_positive_part = x_temp * torch.exp(c * (x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1] - x[int(i - (k - 1) / 2)]))
            # delta_negative_part = x_temp * torch.exp(- c * (x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1] - x[int(i - (k - 1) / 2)]))
            # A_matrix = torch.cat((delta_positive_part.T, delta_negative_part.T, torch.tensor([[1] + [0] * (k-1)])),
            #                      dim=-2)
            # zero_matrix = torch.tensor([[0] * (k - 1) + [1]], dtype=torch.float64).T
            # A_temp = torch.linalg.solve(A_matrix, zero_matrix)

            #####################
            # standardize x by x + t, where t = -(xmin + xmax)/2
            t = - (x[int(i - (k - 1) / 2)] + x[int(i + (k - 1) / 2)]) / 2
            x_tran = x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1] + t  # transformed x
            # x_tran = x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1]
            x_temp = x_tran ** torch.tensor(range(0, int((k - 3) / 2 + 1)))

            # calculate A
            c = (k - 2) ** 0.5 / rho
            delta_positive_part = x_temp * torch.exp(c * x_tran)
            delta_negative_part = x_temp * torch.exp(-c * x_tran)
            A_matrix = torch.cat((delta_positive_part.T, delta_negative_part.T, torch.tensor([[1] + [0] * (k - 1)])),
                                 dim=-2)
            zero_matrix = torch.tensor([[0] * (k - 1) + [1]], dtype=torch.float64).T
            A_temp = torch.linalg.solve(A_matrix, zero_matrix)
            ######################

            row_A = np.append(row_A, range(int(i - (k - 1) / 2), int(i + (k - 1) / 2 + 1)))
            col_A = np.append(col_A, [i] * k)
            values_A = np.append(values_A, A_temp.data.T.numpy()[0])

            ### calculate phi_x_x
            x_mid = x[int(i - (k - 1) / 2):int(i + (k - 1) / 2) + 1]
            k_x_x = self.kernel(x_mid).evaluate()
            phi_x_x_temp = torch.mm(k_x_x, A_temp)

            row_phi_x_x = np.append(row_phi_x_x, range(int(i - (k - 1) / 2), int(i + (k - 1) / 2 + 1)))
            col_phi_x_x = np.append(col_phi_x_x, [i] * k)
            values_phi_x_x = np.append(values_phi_x_x, phi_x_x_temp.data.T.numpy()[0])

        # calculate the boundary basis phi_[1 : (k-1)/2] & x[n-(k-3)/2 : n]
        for i in range(0, int((k - 1) / 2)):
            # ### calculate boundary of A
            # # compute x_left_temp
            # s_left = int(i + (k - 1) / 2) + 1
            # # t_left = - x[s_left-1]
            # # x_left_tran = x[0:0+s_left] - (x[0] + x[0+s_left-1])/2 # transformed x_left
            # # x_left_temp_l = x_left_tran ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            # x_left_temp_l = (x[0:0+s_left] - (x[0] + x[0+s_left-1])/2) ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            # # x_left_temp_r = x_left_tran ** torch.tensor(range(0, int(s_left - (k + 3) / 2 + 1)))
            # x_left_temp_r = (x[0:0+s_left] - (x[0] + x[0+s_left-1])/2) ** torch.tensor(range(0, int(s_left - (k + 3) / 2 + 1)))
            #
            # # compute x_right_temp
            # s_right = k - 1 - i
            # # t_right = - x[n - s_right]
            # # x_right_tran = x[(n - s_right):n] + t_right  # transformed x_right
            # # x_right_tran = x[(n - s_right):n] - (x[n - s_right]+x[n-1])/2  # transformed x_right
            # x_right_temp_l = (x[(n - s_right):n] - (x[n - s_right]+x[n-1])/2) ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            # x_right_temp_r = (x[(n - s_right):n] - (x[n - s_right]+x[n-1])/2) ** torch.tensor(range(0, int(s_right - (k + 3) / 2 + 1)))
            #
            # # calculate left boundary A[:,i]
            # delta_positive_part_left_sided = x_left_temp_l * torch.exp(c * (x[0:0+s_left] - x[0]))
            # delta_negative_part_left_sided = x_left_temp_r * torch.exp(-c * (x[0:0+s_left] - x[0]))
            # A_matrix_left_sided = torch.cat(
            #     (delta_positive_part_left_sided.T,
            #      delta_negative_part_left_sided.T,
            #      torch.tensor([[1] + [0] * (s_left - 1)])),
            #     dim=-2)
            # zero_matrix_left = torch.tensor([[0] * (s_left - 1) + [1]], dtype=torch.float64).T
            # A_temp_left_sided = torch.linalg.solve(A_matrix_left_sided, zero_matrix_left)
            #
            # row_A_left_sided = np.append(row_A_left_sided, range(0, 0+s_left))
            # col_A_left_sided = np.append(col_A_left_sided, [i] * s_left)
            # values_A_left_sided = np.append(values_A_left_sided, A_temp_left_sided.data.T.numpy()[0])
            #
            # # calculate right boundary A[:,n-(k-1)/2+i]
            # delta_positive_part_right_sided = x_right_temp_r * torch.exp(c * (x[(n - s_right):n] - x[n - s_right]))
            # delta_negative_part_right_sided = x_right_temp_l * torch.exp(-c * (x[(n - s_right):n] - x[n - s_right]))
            # A_matrix_right_sided = torch.cat(
            #     (delta_positive_part_right_sided.T,
            #      delta_negative_part_right_sided.T,
            #      torch.tensor([[1] + [0] * (s_right - 1)])),
            #     dim=-2)
            # zero_matrix_right = torch.tensor([[0] * (s_right - 1) + [1]], dtype=torch.float64).T
            # A_temp_right_sided = torch.linalg.solve(A_matrix_right_sided, zero_matrix_right)
            #
            # row_A_right_sided = np.append(row_A_right_sided, range((n - s_right), n))
            # col_A_right_sided = np.append(col_A_right_sided, [n-(k-1)/2+i] * s_right)
            # values_A_right_sided = np.append(values_A_right_sided, A_temp_right_sided.data.T.numpy()[0])

            #############
            # compute x_left_temp
            s_left = int(i + (k - 1) / 2) + 1
            t_left = - x[s_left - 1]
            x_left_tran = x[0:0 + s_left] + t_left  # transformed x_left
            x_left_temp_l = x_left_tran ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            x_left_temp_r = x_left_tran ** torch.tensor(range(0, int(s_left - (k + 3) / 2 + 1)))

            # calculate left boundary A[:,i]
            delta_positive_part_left_sided = x_left_temp_l * torch.exp(c * x_left_tran)
            delta_negative_part_left_sided = x_left_temp_r * torch.exp(-c * x_left_tran)
            A_matrix_left_sided = torch.cat(
                (delta_positive_part_left_sided.T,
                 delta_negative_part_left_sided.T,
                 torch.tensor([[1] + [0] * (s_left - 1)])),
                dim=-2)
            zero_matrix_left = torch.tensor([[0] * (s_left - 1) + [1]], dtype=torch.float64).T
            A_temp_left_sided = torch.linalg.solve(A_matrix_left_sided, zero_matrix_left)

            row_A_left_sided = np.append(row_A_left_sided, range(0, 0 + s_left))
            col_A_left_sided = np.append(col_A_left_sided, [i] * s_left)
            values_A_left_sided = np.append(values_A_left_sided, A_temp_left_sided.data.T.numpy()[0])

            # compute x_right_temp
            s_right = k - 1 - i
            t_right = - x[n - s_right]
            x_right_tran = x[(n - s_right):n] + t_right  # transformed x_right
            x_right_temp_l = x_right_tran ** torch.tensor(range(0, int((k - 3) / 2 + 1)))
            x_right_temp_r = x_right_tran ** torch.tensor(range(0, int(s_right - (k + 3) / 2 + 1)))

            # calculate right boundary A[:,n-(k-1)/2+i]
            delta_positive_part_right_sided = x_right_temp_r * torch.exp(c * x_right_tran)
            delta_negative_part_right_sided = x_right_temp_l * torch.exp(-c * x_right_tran)
            A_matrix_right_sided = torch.cat(
                (delta_positive_part_right_sided.T,
                 delta_negative_part_right_sided.T,
                 torch.tensor([[1] + [0] * (s_right - 1)])),
                dim=-2)
            zero_matrix_right = torch.tensor([[0] * (s_right - 1) + [1]], dtype=torch.float64).T
            A_temp_right_sided = torch.linalg.solve(A_matrix_right_sided, zero_matrix_right)

            row_A_right_sided = np.append(row_A_right_sided, range((n - s_right), n))
            col_A_right_sided = np.append(col_A_right_sided, [n - (k - 1) / 2 + i] * s_right)
            values_A_right_sided = np.append(values_A_right_sided, A_temp_right_sided.data.T.numpy()[0])
            #######################

            ### calculate boundary of phi_x_x
            # left boundary basis
            x_mid_left_sided = x[0:0+s_left]
            k_x_x_left_sided = self.kernel(x_mid_left_sided).evaluate()
            phi_x_x_temp_left_sided = torch.mm(k_x_x_left_sided, A_temp_left_sided)

            row_phi_x_x_left_sided = np.append(row_phi_x_x_left_sided, range(0, 0+s_left))
            col_phi_x_x_left_sided = np.append(col_phi_x_x_left_sided, [i] * s_left)
            values_phi_x_x_left_sided = np.append(values_phi_x_x_left_sided,
                                                  phi_x_x_temp_left_sided.data.T.numpy()[0])

            # right boundary basis
            x_mid_right_sided = x[(n - s_right):n]
            k_x_x_right_sided = self.kernel(x_mid_right_sided).evaluate()
            phi_x_x_temp_right_sided = torch.mm(k_x_x_right_sided, A_temp_right_sided)

            row_phi_x_x_right_sided = np.append(row_phi_x_x_right_sided, range((n - s_right), n))
            col_phi_x_x_right_sided = np.append(col_phi_x_x_right_sided, [n-(k-1)/2+i] * s_right)
            values_phi_x_x_right_sided = np.append(values_phi_x_x_right_sided,
                                                   phi_x_x_temp_right_sided.data.T.numpy()[0])

        A = torch.sparse_coo_tensor(
            np.array([np.append(np.append(row_A_left_sided, row_A), row_A_right_sided),
                      np.append(np.append(col_A_left_sided, col_A), col_A_right_sided)]),
            np.append(np.append(values_A_left_sided, values_A), values_A_right_sided),
            (n, n))

        phi_x_x = torch.sparse_coo_tensor(
            np.array([np.append(np.append(row_phi_x_x_left_sided, row_phi_x_x), row_phi_x_x_right_sided),
                      np.append(np.append(col_phi_x_x_left_sided, col_phi_x_x), col_phi_x_x_right_sided)]),
            np.append(np.append(values_phi_x_x_left_sided, values_phi_x_x), values_phi_x_x_right_sided),
            (n, n))

        return A, phi_x_x

deep_net = LargeFeatureExtractor().to(torch.float64)
dkpl = DeepKernelPacketGP(kernel=gpytorch.kernels.MaternKernel(nu=2.5), feature_extractor=deep_net).to(torch.float64)

##########TRAIN##########
epochs = 60

# Find optimal model hyperparameters
dkpl.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': dkpl.feature_extractor.parameters()},
#     {'params': dkpl.kernel.parameters()},
#     {'params': dkpl.prior_var}
# ], lr=0.01)
optimizer = torch.optim.Adam(dkpl.parameters(), lr=0.001)

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
        dkpl.fit(train_x, train_y)
        # Calc loss and backprop derivatives
        loss = dkpl.nll_loss()
        forward_end = time.time()
        forward_time.append(forward_end-forward_start)

        iterator.set_postfix(loss=loss.item())

        backward_start = time.time()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(dkl.parameters(), 10)
        optimizer.step()
        backward_end = time.time()
        backward_time.append(backward_end - backward_start)

        epoch_end = time.time()
        epoch_time.append(epoch_end-epoch_start)

start = time.time()
train()
end = time.time()
print("Training time: %.2f" % (end-start))

# ##########TEST##########
# dkpl.eval()
# predict_time = []
# with torch.no_grad():
#     predict_start = time.time()
#     pred_mean, pred_cov = dkpl.predict(test_x)
#     predict_end = time.time()
#     predict_time.append(predict_end-predict_start)
#
# print('Test MAE: {}'.format(torch.mean(torch.abs(pred_mean - test_y))))