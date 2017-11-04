import unittest
import numpy as np
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import norm
from scipy.stats import wishart
import matplotlib.pyplot as plt
import Heuristic
from scipy.signal import savgol_filter
import pickle
import time


from slds_max import *


class SetupTest(unittest.TestCase):
    """
    Set Up the necessary initialization procedures and attributes

    CAUTION: The initialization procedures only set up the initial distributions etc - any prior sampling
            for the testing of the testing modules has to be done within the testing modules themselves
            (e.g., if the Kalman filter needs resampled z values, that needs to be done in there)

    Attributes:
     -------
        T (int):
            length of episode (number of time steps)
        dims (int):
            dimension of state variable
        C (np.array):
            Observation matrix, dimensions: dims * dims
        fixed (bool):
            Toggle random generation or customized generation of dynamic matrices
        z_real (np.array):
            real mode sequence, dimension: dims * T
        z (np.array):
            initialized mode sequence, dimension: dims * T
        Sigmas_real (dictionary):
            contains the dynamic noise matrices corresponding to the mode; each matrix is a 2D-array
            with dimensions dims * dims
        As_real (dictionary):
            contains the real dynamic transition matrices corresponding to the mode; each matrix is a 2D-array
            with dimensions dims * dims
        Sigmas (dictionary):
            contains the (initialized/sampled dynamic noise matrices corresponding to the mode; each matrix is a 2D-array
            with dimensions dims * dims
        As (dictionary):
            contains the initialized/sampled dynamic transition matrices corresponding to the mode; each matrix is a 2D-array
            with dimensions dims * dims
        R (np.array):
            Observation noise matrix, dimensions: dims * dims
        x (np.array):
            Hidden states, dimensions: dims * T
        psi (np.array):
            Pseudo observations - they equal x in the SLDS framework, dimensions: dims * T
        y (np.array):
            Observations, dimensions: dims * T

    Functions:
    -------
        init_z:
            initialize z_real
        init_z:
            initialize z
        init_dyn_params_real:
            initialize As_real, Sigmas_real and R_real
        init_dyn_params:
            initialize As, Sigmas and R
        init_x_psi_y_real:
            initialize x, psi and y
        plotting_initializations:
            plots everything related to initializing
        plotting_modes:
            plots a comparison of the actual and sampled mode sequence
    """

    def setUp(self):
        # load SLDS object: access all parameters and functions from the SLDS framework through this object
        self.SLDS_obj = SLDS()
        self.L = self.SLDS_obj.L
        self.T = 200
        self.dims = 2
        self.C = np.eye(self.dims)
        self.fixed = True

        # Initialization for synthetic data from methods
        self.z_real = self.init_z_real()
        self.As_real, self.Sigmas_real, self.R = self.init_dyn_params_real()
        self.x_real, self.psi_real, self.y = self.init_x_psi_y_real()

        heur = Heuristic.MultivariateRegression()
        A_est, z, mode_cnt, xx, yy = heur.inference()
        x1 = np.sin(0.02*np.arange(0, self.T, 1))*2+0.5
        x2 = np.sin(0.02*np.arange(0, self.T, 1))*2+0.5
        self.y[0, :] = xx[0,:]
        self.y[1, :] = xx[1,:]

        self.x, self.psi = self.init_x_psi()

        print("Real params: \n")
        print("gamma: " + str(self.SLDS_obj.gamma) + " kappa: " + str(self.SLDS_obj.kappa) + " rho: " + str(self.SLDS_obj.rho))

        # initialization for SLDS
        self.z = self.init_z()




        # psi = np.copy(xx)
        # unique_z = np.unique(z)

        #self.z = z

        # pass data to SLDS object
        self.SLDS_obj.set_observations(self.y, self.C)
        self.SLDS_obj.init_model(self.z, self.x)
        self.SLDS_obj.init_params_data()


        # Further initializations from SLDS object
        self.As, self.Sigmas = self.init_dyn_params()
        #self.As[1] = self.As_real[1]
        #self.As[2] = self.As_real[2]
        #self.As[3] = self.As_real[3]

        #from heuristic initialization - has to stand here
        # for ii in unique_z:
        #    if ii == 0:
        #        continue
        #    self.As[ii] = A_est[ii]

        self.pi_init = self.SLDS_obj.param_pi_beta_sampler['pi']
        self.beta_init = self.SLDS_obj.param_pi_beta_sampler['beta']

    def init_z_real(self):
        """
        Initialize mode sequence z

        Returns
        -------
        z_real:
            2D-array (dims * T)
        """
        T = self.T
        L = self.L
        values = np.arange(0, L)
        z_real = np.zeros(T, dtype=np.int32)
        z_real[0] = 1
        self.SLDS_obj.init_sample_pi_beta()
        params = self.SLDS_obj.param_pi_beta_sampler
        pi = params['pi']
        beta = params['beta']
        for t in range(1, T):
            t1 = 20
            t2 = t1 + 20
            t3 = t2 + 20
            t4 = t3 + 20
            t5 = t4 + 20
            t6 = t5 + 20
            t7 = t6 + 20
            t8 = t7 + 20

            if t < t1:
                z_real[t] = 1
            elif t >= t1 and t < t2:
                z_real[t] = 3
            elif t >= t2 and t < t3:
                z_real[t] = 1
            elif t >= t4 and t < t5:
                z_real[t] = 2
            elif t >= t5 and t < t6:
                z_real[t] = 2
            elif t >= t6 and t < t7:
                z_real[t] = 1
            elif t >= t7 and t < t8:
                z_real[t] = 3
            else:
                z_real[t] = 1
            #prob = pi[:, z_real[t - 1]]
            #z_real[t] = np.random.choice(values, 1, p=list(prob)) #real test sequence (generated with pi_init distribution)
        return z_real

    def init_z(self):
        """
        Initialize mode sequence z

        Returns
        -------
        z:
            2D-array (dims * T)
        """
        z = np.zeros(self.T, dtype=np.int32)
        for t in range(0,self.T):
            z[t] = np.random.randint(0, self.L)  # initialized sequence for sampler (randomly generated)
        z[0] = self.z_real[0]
        return z

    def init_dyn_params_real(self):
        """
        Initialize the real dynamic parameters As_real, Sigmas_real and R

        Returns
        -------
            As:
                Dictionary of arrays(dims * dims) (indexed by mode number)
            Sigmas:
                Dictionary of arrays (dims * dims) (indexed by mode number)
            R:
                2D-array (dims * dims)
        """
        L = self.SLDS_obj.L
        dims = self.dims

        As = {}
        Sigmas = {}
        # initialize R
        R = np.eye(dims) * 0.001

        # random generation (here the exact generation of all matrices can be specified)
        for k in range(0, L):
            if dims == 1:
                As[k] = (np.random.rand(1) * 1)
                Sigmas[k] = (np.random.rand(1) * 0.005)
            else:
                # generate random state transition matrix
                As[k] = (np.random.rand(dims, dims) * 1)

                """# generate random state transition matrix
                As[k] = np.random.rand(dims, dims) - 0.5
                # make matrix positive/negative definite
                dg = np.diagonal(As[k])
                As[k][np.diag_indices_from(As[k])] = np.absolute(dg) + 0.1 * np.ones(np.size(dg))"""

                wish = wishart.rvs(dims + 1, np.eye(dims))
                Sigmas[k] = wish / np.linalg.norm(wish) * 1
                # fixed sigmas and R (to test algorithm with not too crazy of matrices)
                if self.fixed:
                    # tune this to make it "harder" for the algorithm to sample correctly
                    # (Sigma is hidden noise between x, R is observation noise)
                    #### This is "state" transition noise, not observation noise!!! so ideally keep it low ####
                    Sigmas[k] = np.eye(dims) * 0.000000000001
                    # if needed, the A matrices can be initialized to a meaningful value (not just some random
                    #  invWishart sample)--->
                    As[1] = np.array([[0, 1], [0.7, 0.36]])
                    As[2] = np.array([[0, 1], [0.4, 0.56]])
                    As[3] = np.array([[0.5, 0.5], [0.32, 0.67]])
        return As, Sigmas, R

    def init_dyn_params(self):
        """
        Initialize dynamic parameters A, Sigma and R

        Returns
        -------
            As_real:
                Dictionary of arrays(dims * dims) (indexed by mode number)
            Sigmas_real:
                Dictionary of arrays (dims * dims) (indexed by mode number)
            R_real:
                2D-array (dims * dims)
        """

        L = self.SLDS_obj.L
        dims = self.dims
        params = self.SLDS_obj.param_dyn_param_sampler
        As = self.As_real.copy()
        Sigmas = self.Sigmas_real.copy()

        for k in range(0, L):
            S_hh = params['K']  # S _ phi dash, phi dash
            S_wh = np.dot(params['M'], params['K'])  # S _ phi, phi dash
            S_ww = np.dot(np.dot(params['M'], params['K']), params['M'].T)  # S _ phi, phi
            S_wdh = S_ww - np.dot(S_wh, np.linalg.inv(S_hh).dot(
                S_wh.T))  # S _ phi | phi dash (The definition is not in the algo summary but in the pages before)
            # Max Comment: Why not inverse? Solve gives me different values than inverse

            Sigma = invwishart.rvs(params['n_0'], params['S_0'])


            A_M = np.dot(np.linalg.inv(S_hh.T), S_wh.T).T  # mean of matrix normal distribution
            A_vecM = np.reshape(A_M, (np.prod(np.shape(A_M))))  # transform Matrix A_M to vector

            # hier war S_hh und Sigma vertauscht
            A_vec = multivariate_normal.rvs(A_vecM, np.kron(np.linalg.inv(S_hh),
                                                            Sigma))  # sample from matrix normal distribution (def. is some pages before in the paper)
            # hier war Transpose zu viel
            A = np.reshape(A_vec, np.shape(A_M))  # convert sampled to matrix

            As[k] = A
            Sigmas[k] = Sigma


        return As, Sigmas



    def init_x_psi_y_real(self):
        """
        Initialize states x and pseudo observations psi

        Returns
        -------
        x:
            2D-array (dims * T)
        psi:
            2D-array (dims * T)
        y:
            2D-array (dims * T)
        """
        dims = self.dims
        T = self.T
        As = self.As_real
        Sigmas = self.Sigmas_real
        R = self.R
        z_real = self.z_real

        x = np.zeros((dims, T))
        y = np.zeros((dims, T))
        # set random value for initial state of x
        for dd in range(0, dims):
            x[dd, 0] = np.random.normal(1, 0.0001)
        for t in range(1, T):
            # use norm for 1D, and mvnorm for n-D
            # x_t+1  = A_zt * x_t
            if dims == 1:
                x[:, t] = np.random.normal(As[z_real[t]] * x[:, t - 1], Sigmas[z_real[t]])
            else:
                x[:, t] = np.random.multivariate_normal(As[z_real[t]].dot(x[:, t - 1]), Sigmas[z_real[t]])
        psi = np.copy(x)

        # generate observations y: y = C*x + R (with C = eye(dims)
        for t in range(0, T):
            # add noise to observations
            if dims == 1:
                y[:, t] = x[:, t] + np.random.normal(0, R)
            else:
                y[:, t] = x[:, t] + np.random.multivariate_normal(np.zeros(dims), R)
        return x, psi, y

    def init_x_psi(self):
        x = savgol_filter(self.y, 7, 3)
        psi = np.copy(x)
        return x, psi

    def plotting_initializations(self):
        """
        Plots initialization relevant values, such as pi or beta distribution
        """
        saveFigures = True
        L = self.L
        beta = self.beta_init
        SLDS_obj = self.SLDS_obj
        ind = np.arange(1, L + 1)
        width = 0.6
        fig, ax = plt.subplots(2,2)
        ax[0, 0].bar(ind - width / 2, beta[:L], width, color='b')
        ax[0, 0].set_title("initial beta distribution")
        ax[0, 0].set_xlabel("Mode index")
        ax[0, 0].set_ylabel("Probability")
        dirich = np.zeros(L)
        dirich[:] = np.random.dirichlet((np.zeros(L) + SLDS_obj.gamma) / L, 1)
        ax[0, 1].bar(ind - width / 2, dirich[:L], width, color='b')
        ax[0, 1].set_title("initial pi distribution, one sample, no sticky")
        ax[0, 1].set_xlabel("Mode index")
        ax[0, 1].set_ylabel("Probability")

        tt = np.linspace(0, self.T - 1, self.T)
        ax[1, 0].plot(tt, self.x[0, :], label="real state values - dim 1", color="blue")
        ax[1, 0].set_title("dim 1")
        ax[1, 0].set_xlabel("Time step")
        ax[1, 0].set_ylabel("Value")
        plt.show()

    def plotting_modes(self):
        """
        Plots the true and sampled mode sequence
        """
        plt.rc('font', family='serif')
        myfigsize = (6, 2)

        plt.figure(frameon=False, figsize=myfigsize)
        T = self.T
        plt.xlabel("Time")
        plt.ylabel("Mode")
        tt = range(1, T)
        y = self.z_real[1::]
        y_new = y
        uniquey = np.unique(y)
        max_y = np.size(uniquey)
        for i in range(0, max_y):
            idx = np.where(y == uniquey[i])
            y_new[idx] = i + 1
        plt.scatter(tt, y_new, color='k', s=8, label='real sequence', marker='|')
        axes = plt.gca()
        axes.set_xlim([min(tt)-1, max(tt)+1])
        plt.yticks(range(1, max_y + 1))
        plt.tight_layout()
        plt.savefig('real_seq_' + str(time.time()) + '.png')

        plt.figure(frameon=False, figsize=myfigsize)
        plt.xlabel("Time")
        plt.ylabel("Mode")
        y = self.z[1::]
        y_new = y
        uniquey = np.unique(y)
        max_y = np.size(uniquey)
        for i in range(0, max_y):
            idx = np.where(y == uniquey[i])
            y_new[idx] = i+1
        plt.scatter(tt, y_new, color='k', s=8, label='sampled sequence', marker='|')
        axes = plt.gca()
        axes.set_xlim([min(tt)-1, max(tt)+1])
        plt.yticks(range(1, max_y+1))
        plt.tight_layout()
        plt.savefig('sampled_seq_' + str(time.time()) + '.png')

        plt.show()

    def plotting_reconstruct(self):
        """
        Plots the true state sequence and the reconstructed state sequence using the sampled dyn matrices
        """
        plt.rc('font', family='serif')
        myfigsize = (6, 3)

        x_est = np.zeros((self.dims, self.T))
        x_est[:, 0] = self.x[:, 0]
        for t in range(1, self.T):
            x_est[:, t] = self.As[self.z[t]].T.dot(x_est[:, t - 1])

        plt.figure(frameon=False, figsize=myfigsize)
        tt = np.linspace(0, self.T - 1, self.T)
        #fig, ax = plt.subplots()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.plot(tt, self.x[0, :], label="real state sequence $x_{1:T}$", color="blue", lw=1.6)
        plt.plot(tt, x_est[0, :], label="reconstructed state sequence $\hat{x_{1:T}}$", color="green", linestyle="dashed", lw=1.6)
        plt.tight_layout()
        plt.savefig('nolose_' + str(time.time()) + '.png')

        #fig2, ax2 = plt.subplots()
        plt.figure(frameon=False, figsize=myfigsize)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.plot(tt, self.x[1, :], label="real state sequence $x_{1:T}$", color="blue", lw=1.6)
        plt.plot(tt, x_est[1, :], label="reconstructed state sequence $\hat{x_{1:T}}$", color="green", linestyle="dashed", lw=1.6)
        plt.tight_layout()
        plt.savefig('nolose2_' + str(time.time()) + '.png')
        plt.show()




class DynSamplerTestCase(SetupTest):
    """
    Tests for the sampler of all dynamic parameters A, Sigma and R

    Functions
    -------


    """

    def test_sample_A(self):
        SLDS_obj = self.SLDS_obj
        x = SLDS_obj.x

        As_hat = {}
        Sigmas_hat = {}
        N = 40
        for i in range(0, N):
            # set real z for sampling transition matrices
            z = self.z_real
            SLDS_obj.z = z

            # sample transition matrices
            As_s, Sigmas_s = SLDS_obj.sample_dyn_params(SLDS_obj.z, SLDS_obj.x, self.As, self.Sigmas)

            # compare norms of true and estimated transition matrices
            for n in As_s.keys():
                if i == 0:
                    As_hat[n] = np.ndarray((N,) + np.shape(self.As_real[n]))
                    Sigmas_hat[n] = np.ndarray((N,) + np.shape(self.Sigmas_real[n]))

                As_hat[n][i, :, :] = As_s[n]
                Sigmas_hat[n][i, :, :] = Sigmas_s[n]

        dist_A = np.ndarray(len(As_s.keys()))
        dist_Sigma = np.ndarray(len(As_s.keys()))
        id = 0
        for n in As_s.keys():
            A_hat = np.mean(As_hat[n], 0)
            Sigma_hat = np.mean(Sigmas_hat[n], 0)
            assert np.shape(np.shape(A_hat))[0] == 2

            A = self.As_real[n]
            Sigma = self.Sigmas_real[n]

            print("Estimated A:\n", A_hat, "; \nTrue A:\n", A, "\n")
            print("Estimated Sigma:\n", Sigma_hat, "; True Sigma:\n", Sigma, "\n")

            diff_A = np.linalg.norm(A - A_hat)
            dist_A[id] = diff_A
            diff_Sigma = np.linalg.norm(Sigma - Sigma_hat)
            dist_Sigma[id] = diff_Sigma

            print("Norm difference in A: ", diff_A, "; and in Sigma: ", diff_Sigma, "\n")
            id = id + 1

        print(dist_A)
        self.assertTrue(np.mean(dist_A) < 0.2)
        self.assertTrue(np.mean(dist_Sigma) < 2)
        return

        # COMPARE APPLIED A
        # initialize estimated and true x
        x_hat = np.ndarray((SLDS_obj.xdim, SLDS_obj.T))
        x_true = np.ndarray((SLDS_obj.xdim, SLDS_obj.T))
        x_hat[:, 0] = x[:, 0]
        x_true[:, 0] = x[:, 0]

        print("Keysa: ", self.As.keys(), "\n")
        print("Keysa: ", As_hat.keys(), "\n")

        # estimate state sequences
        for t in range(1, self.T):
            # true without noise
            x_true[:, t] = self.As[z[t]].dot(x[:, t - 1])
            # prediction
            x_hat[:, t] = As_hat[z[t]].dot(x[:, t - 1])

        diff_x = np.linalg.norm(x_true - x_hat)

        print("Difference in x: ", diff_x, "\n")

        self.assertTrue(true)


class KalmanFilterTestCase(SetupTest):
    """
    Tests for the Kalman filter that samples both x and z

    Functions
    -------
    """

    def test_whatever(self):
        self.assertTrue(1)
        # continue as you want


class BlockSamplerTestCase(SetupTest):
    """
    Tests for the block sampler of the mode sequence z and also re-samples pi and beta

    Functions
    -------

    test_z_sampling:
        Tests whether the block sampler can sample at least X percent of the modes correctly
        \as of now, the tests works if supplied with correct A matrices (or slightly disturbed ones) and low noise

    """

    def test_z_sampling(self):
        """
        test if the z sample produces enough matches
        """

        # choose the desired threshold (e.g., 40% matched modes)
        threshold = self.T * 0.4
        SLDS_obj = self.SLDS_obj
        num_iter = 14
        pi = self.pi_init
        self.plotting_initializations()
        for ii in range(0, num_iter):
            # Step 3: set pseudo observations
            psi = SLDS_obj.x
            # Step 4: block sample z and also return transition counts n
            z_t, n = SLDS_obj.block_sample_z(self.z, psi, pi, self.As, self.Sigmas)
            z = z_t
            #dumps = {"zs": z, "zr": self.z_real}
            #pickle.dump(dumps, open("mydump" + str(time.time()) + ".p", "wb"))
            # Step 5: re sample pi and beta distribution
            pi_t, beta_t, m, w = SLDS_obj.sample_pi_beta(n)
            gam, ak, rh = SLDS_obj.sample_hyperparameters(z, n, m, w)
            As_s, Sigmas_s = SLDS_obj.sample_dyn_params(SLDS_obj.z, SLDS_obj.x, self.As, self. Sigmas)
            self.As = As_s
            self.Sigmas = Sigmas_s
            SLDS_obj.rho = rh
            SLDS_obj.gamma = gam
            SLDS_obj.alpha = (1 - SLDS_obj.rho) * ak
            SLDS_obj.kappa = SLDS_obj.rho * ak
            pi = pi_t
            beta = beta_t
            # check how many matches exist
            matches = len(list(filter(lambda x: x, z == self.z_real)))
            print("real sequence of z: " + str(self.z_real))
            print("Sampled sequence of z: " + str(self.z))
            print("Matches: " + str(matches))
            print("gamma: " + str(SLDS_obj.gamma) + " kappa: " + str(SLDS_obj.kappa) + " rho: " + str(SLDS_obj.rho))
            print("\n")
            self.plotting_modes()
            self.plotting_reconstruct()
            if (ii >= 3):
                a = 0
                self.plotting_modes()
                self.plotting_reconstruct()

            #plt.close("all")
        self.assertTrue(matches > threshold)


# executes all tests - needs to be at the end of the entire .py test file
if __name__ == '__main__':
    unittest.main()