import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import wishart
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvnorm
from scipy.signal import savgol_filter
plt.ion()

class MultivariateRegression():
    """
    Find dynamic matrices and switching points by applying linear regression of parts of the
    system until error becomes to big
    """
    def __init__(self):
        # Set number of time steps T, truncation level L and state dimension dims
        self.T = 200
        self.L = 10
        self.dims = 2
        self.epsilon = 0.01 #percent
        self.tolerance = 0
        # assign modes from 0 to K-1 (easier indexing since 0 is 1 in python)
        self.K = 4
        self.lambd = 0.00004
        self.thresh = 0.005 #for mode similarity check

        # the cached statistics
        self.Y = []

        self.hyperparameters = {}
        self.hyperparameters[
            'rho'] = 0.7  # used for the Bernoulli trial when calculating the number of tables that serve a specific dish
        self.hyperparameters['kappa'] = 0.3  # self transition parameter
        self.hyperparameters['alpha'] = 5  # determines how much beta is spread out
        self.hyperparameters['gamma'] = 12  # determines how much pi is spread out
        self.hyperparameters['K'] = np.eye(self.dims)
        self.hyperparameters['M'] = np.zeros((self.dims, self.dims))
        self.hyperparameters['n_0'] = 2
        self.hyperparameters['S_0'] = 1e-5 * np.eye(self.dims)
        self.hyperparameters['r_0'] = 1e-5
        self.hyperparameters['R_0'] = 1e-5 * np.eye(2)
        self.hyperparameters['nit'] = 50
        self.hyperparameters['a'] = 1
        self.hyperparameters['b'] = 1
        self.x = np.zeros((self.dims,self.T))
        self.y = np.zeros((self.dims,self.T))
        self.z = np.zeros(self.T, dtype=np.int32)

        # initialise mode sequence z
        z = np.zeros(self.T, dtype=np.int32)
        z[0] = np.random.randint(0, self.L)
        z_real = np.zeros(self.T, dtype=np.int32)
        for k in range(0, self.T):
            if k < 15:
                z_real[k] = 1
            elif k >= 15 and k < 30:
                z_real[k] = 1
            elif k >= 30 and k < 40:
                z_real[k] = 1
            elif k >= 40 and k < 50:
                z_real[k] = 1
            elif k >= 50 and k < 75:
                z_real[k] = 1
            elif k >= 75 and k < 90:
                z_real[k] = 3
            else:
                z_real[k] = 2
            # z_real[k] = np.random.choice(values, 1, p=list(prob)) #real test sequence (generated with pi_init distribution)
            z[k] = np.random.randint(0, self.L)  # initialized sequence for sampler (randomly generated)
        z = np.copy(z_real)

        # initialise dynamical parameters
        A_list = {}
        Sigma_list = {}
        R_list = {}
        fixed = True
        # random generation
        for k in range(0, self.L):
            if self.dims == 1:
                A_list[k] = (np.random.rand(1) * 1)
                Sigma_list[k] = (np.random.rand(1) * 0.1)
                R_list[k] = (np.random.rand(1) * 0.01)
                A_list[1] = np.array([1.1])
                A_list[2] = np.array([0.9])
                A_list[3] = np.array([1.6])
                A_list[4] = np.array([0.8])



            else:
                # generate positive definite matrices (maybe better way to do it?)
                A_list[k] = np.random.rand(self.dims, self.dims)+0.15
                A_list[k][0,0] = 0
                wish = wishart.rvs(self.dims + 1, np.eye(self.dims))
                Sigma_list[k] = wish / np.linalg.norm(wish) * 1
                wish = wishart.rvs(self.dims + 1, np.eye(self.dims))
                R_list[k] = wish / np.linalg.norm(wish) * 1
                # fixed sigmas and R (to test algorithm with not too crazy of matrices)
                if fixed:
                    # tune this to make it "harder" for the algorithm to sample correctly
                    # (Sigma is hidden noise between x, R is observation noise)
                    R_list[k] = np.eye(self.dims) * 0.0001
                    Sigma_list[k] = np.eye(self.dims) * 0.001

                    A_list[1] = np.array([[0, 1], [0.05 , 0.05]])
                    A_list[2] = np.array([[0, 1], [-0.05 , -0.05]])
                    A_list[3] = np.array([[0, 1], [0.002, 0.0001]])

        # generate state sequence x in correspondence with the dynamical parameters and set pseudo observations psi
        x = np.zeros((self.dims,self.T))
        y = np.zeros((self.dims,self.T))
        # set random value for initial state of x
        for dd in range(0,self.dims):
            x[dd,0] = np.random.normal(5,0.5)
        for t in range(1,self.T):
            # use norm for 1D, and mvnorm for n-D
            # x_t+1  = A_zt * x_t
            if self.dims == 1:
                #x[:,t] = np.random.normal(A_list[z_real[t]]*x[:,t-1],Sigma_list[z_real[t]])
                x[:,t] = np.random.normal(A_list[z_real[t]]*x[:,t-1],0.002)

            else:
                #x[:,t] = np.random.multivariate_normal(A_list[z_real[t]].dot(x[:,t-1]),Sigma_list[z_real[t]])
                x[:,t] = np.random.multivariate_normal(A_list[z_real[t]].dot(x[:,t-1]),np.eye(self.dims) * 0.001)

        x1 = np.sin(0.02*np.arange(0, self.T, 1))*2+0.5
        x2 = np.sin(0.02*np.arange(0, self.T, 1))*2+0.5
        x[0, :] = x1[:]
        x[1, :] = x2[:]
        psi = np.copy(x)

        #
        for t in range(0,self.T):
            # add noise to observations
            if self.dims == 1:
                y[:, t] = x[:, t] + np.random.normal(0, R_list[z_real[t]])
            else:
                y[:, t] = x[:, t] + np.random.multivariate_normal(np.zeros(self.dims), R_list[z_real[t]])

        self.y = y
        self.x = x
        self.z = z
        self.A_list = A_list
        self.Sigma_list = Sigma_list
        self.R_list = R_list
        
        ############
        # END OF INIT METHOD
        ############

    # def smoothing_filter(self, y):
    #     f_sz = 7
    #     x = np.zeros((self.dims, self.T))
    #     for i in range(0, self.T):
    #         if (i < np.trunc(f_sz / 2)):
    #             offset = int(np.trunc(f_sz / 2) - i)
    #         elif (i > self.T - np.trunc(f_sz / 2)):
    #             offset = -int(np.trunc(f_sz / 2) + i + - (self.T - 1))
    #         else:
    #             offset = 0
    #         if (i == 197):
    #             a= 0
    #         x[:, i] = np.sum(y[:, i - int(np.trunc(f_sz / 2)) + offset : i + int(np.trunc(f_sz / 2)) + offset + 1], axis = 1)/f_sz
    #     return x


    def find_similar_modes(self, z, A_est, B, modecnt):
        """
        Check if the regressor matrix is similir to an already found matrix and reassign modes of z accordingly
        :param z:
        :param A_est:
        :param B:
        :param modecnt:
        :return:
        """
        unique_z = np.unique(z)[1::]
        ii = 1
        found = False
        while ii < len(unique_z):
            distance = np.linalg.norm(A_est[ii] - B) / np.linalg.norm(A_est[ii])
            if distance < self.thresh:
                z[np.where(z == modecnt)] = ii
                found = True
            ii += 1
        return found, z

    def error(self, x_curr, x_est):
        """
        Calculate relative error of the total sub interval and the final value
        :param x_curr:
        :param x_est:
        :return:
        """
        i = 0
        total_error = 0
        while i < np.shape(x_curr)[1]:
            total_error += (np.linalg.norm(x_curr[:, i] - x_est[:, i]) / np.linalg.norm(x_curr[:, i]))
            i += 1
        total_error /= len(x_curr)
        end_error = np.linalg.norm(x_curr[:, -1] - x_est[:, -1]) / np.linalg.norm(x_curr[:, -1])
        return total_error, end_error

    def regression(self, x):
        """
        Multivariate ridge regression
        """
        X = x[:,:-1].T
        Xd = x[:,1:].T
        B = np.linalg.inv(X.T.dot(X) + self.lambd*np.eye(self.dims)).dot(X.T).dot(Xd)
        return B

    def inference(self):
        """
        Inference Algorithm yielding the dynamic matrices, number modes and mode sequence z
        """

        # Filter requirements.
        self.x = savgol_filter(self.y,7,3)
        T = self.T
        A_est = {}
        mode_cnt = 1
        z = np.zeros(T, dtype=np.int32)
        k = 0
        # iterate from every point of time ...
        while k < T-1:
            error_cnt = 0
            # ... and apply regression until time step l
            best_l = 0
            # take a minimum number of time steps to ensure regularity of Data matrix
            offset = self.dims**2 + 5
            offset = 2
            for l in range(k+offset, T + 1):
                current_x = self.x[:, k:l]
                B = self.regression(current_x)
                x_est = np.zeros((self.dims, l - k - 1))
                for t in range(k+1, l):
                    # calculate estimated value at time step l with computed regressor matrix
                    if t == k+1:
                        x_est[:, t-k-1] = B.T.dot(current_x[:, t - k - 1])
                    else:
                        x_est[:, t-k-1] = B.T.dot(x_est[:, t - k - 2])
                # stop current iteration if deviation from real value becomes too big
                # calculate both total demonstration error and final estimator value
                total_error, end_error = self.error(current_x[:, 1::], x_est)
                if (end_error > self.epsilon*2 or total_error > self.epsilon) and l - k > 2:
                    error_cnt += 1
                    # store index of time step when error value exceeded threshold
                    if error_cnt == 1:
                        best_l = l-1
                else:
                    # reset error count if exceeding of error does not occur consecutively (might be outlier/noise)
                    error_cnt = 0
                    best_l = 0
                    z[l - 1] = mode_cnt

                # right now error cnt circumvented, could be used for fine-tuning (always enters this branch if error
                # is too high )
                if error_cnt > self.tolerance or l == T:
                    if best_l == 0:
                        best_l = l
                    # compute regressor matrix A with "best fit" data
                    current_x = self.x[:, k:best_l]
                    B = self.regression(current_x)
                    # this if clause could be removed if better implementation is found (matrix has to be assigned at last time step)
                    if l == T:
                        A_est[mode_cnt] = B
                        z[T - 1] = mode_cnt
                        return A_est, z, mode_cnt, self.x, self.y
                    # check if current regressor matrix is similar to other matrices, and change z accordingly if so
                    found, z = self.find_similar_modes(z, A_est, B, mode_cnt)
                    if found:
                        break
                    # if B is a newly encountered matrix (not similar to others), store in matrix list and increment mode count
                    else:
                        A_est[mode_cnt] = B
                        if l < T:
                            mode_cnt += 1
                        break

            k = best_l - 1
            #k = l-2
        return A_est, z, mode_cnt, self.x, self.y


    def test(self):
        #plt.ion()
        A_est, z, mode_cnt, x, y = self.inference()

        x_est = np.zeros((self.dims, self.T))
        x_est[:,0] = self.x[:,0]
        for t in range(1,self.T):
            x_est[:,t] = A_est[z[t]].T.dot(x_est[:,t-1])
        tt = np.linspace(0, self.T - 1, self.T)
        if self.dims == 1:
            plt.figure()
            plt.title("dim 1")
            plt.plot(tt, self.x[0, :], label="real values", color="blue")
            plt.plot(tt, x_est[0, :], label="estimated values", color="green", linestyle="dashed")

        else:
            plt.figure()
            plt.title("dim 1")
            plt.plot(tt, self.y[0, :], label="observed values", color="red")
            plt.plot(tt, self.x[0, :], label="real values", color="blue")
            plt.plot(tt, x_est[0,:], label="estimated values", color="green", linestyle="dashed")
            plt.legend(loc='best')

            plt.figure()
            plt.title("dim 2")
            plt.plot(tt, self.y[1, :], label="observed values", color="red")
            plt.plot(tt, self.x[1, :], label="real values", color="blue")
            plt.plot(tt, x_est[1, :], label="estimated values", color="green", linestyle="dashed")
            plt.legend(loc='best')
        a = 0



"""
Testing of all (sampling) functions
"""
#tester = MultivariateRegression()
#tester.test()
