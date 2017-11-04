import numpy as np
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import norm
from scipy.stats import gamma

class SLDS:
    """
    Implementation of the HDP SLDS algorithm of [1]

    .. [1] Fox, Emily B. "Bayesian nonparametric learning of complex
    dynamical phenomena." Diss. Massachusetts Institute of Technology,
    2009.
    """


    def __init__(self):
        self.L = 40
        self.xdim = -1
        self.initialized = 0
        self.has_observations = 0
        # Add the data independent parameters here
        self.a = 50  #hyperparameter for sampling alpha and kappa (initialization on page 160)
        self.b = 1 # hyperparameter for sampling alpha and kappa
        self.c = 6 # hyperparameter for sampling rho
        self.d = 2# hyperparameter for sampling rho
        self.e = 50 # hyperparameter for sampling gamma
        self.f = 1 # hyperparameter for sampling gamma
        self.rho = np.random.beta(self.c, self.d)  # used for the Bernoulli trial when calculating the number of tables that serve a specific dish
        ak = np.random.gamma(self.a, self.b)
        self.kappa = self.rho*ak  # self transition parameter
        self.alpha = (1 - self.rho)*ak  # determines how much beta is spread out
        self.gamma = np.random.gamma(self.e, self.f)  # determines how much pi is spread out

        self.Y_cached = {}



    def set_observations(self, y, C):
        """
        Sets the observations for the slds model.

        Parameters
        ----------
        y : ndarray
            2D array containing the observations of the modes
            The dimensionality is [state_dim, T].
        C : ndarray
            Observation matrix as 2D array
        """

        self.initialized = 0
        self.has_observations = 1
        self.y = y
        self.C = C
        self.T = np.shape(y)[1]
        if self.xdim == -1:
            self.xdim = np.shape(y)[0]


    def init_model(self, z, x):
        """
        Initializes the slds model.
        """
        assert self.has_observations == 1

        self.xdim = np.shape(x)[0]
        self.z = z
        self.x = x
        self.initialized = 1
        self.P_0 = np.identity(self.xdim)
        self.mu = np.zeros(self.xdim)


    def learn_model(self):
        """
        Learns the slds model.
        """

        # intialize_model
        if ~self.initialized:
            pass
            #TODO
            #self.init_model(z, x)

        x = self.x
        z = self.z
        y = self.y
        C = self.C

        # initialize priors
        self.init_params_data()
        # Step 2: sample x and z by using a kalman filter
        x, z = self.kalman_sample_x_z(z, y, pi, As, Sigmas, R)
        # Step 3: set pseudo observations
        psi = x
        # Step 4: block sample z and also return transition counts n
        z, n = self.block_sample_z(z, psi, pi, As, Sigmas)
        # Step 5: re sample pi and beta distribution
        pi, beta, m = self.sample_pi_beta(n)
        gam , ak = self.sample_hyperparameters(z, n, m)
        self.gamma = gam
        self. alpha = (1 - self.rho)*(self.alpha + self.kappa)
        self.kappa = self.rho*(self.alpha + self.kappa)
        # Step 6
        self.sample_dyn_params(z, x)
        # Step 7
        self.sample_measure_noise_covmat(x, y, C)


    # initialization methods

    def init_sample_pi_beta(self):
        """
        First step:
            Samples initial distribution of mode transition probability prior beta,
            uses stick breaking construction to spread the weights around the origin
            (dirichlet construction would also work, however, this would spread the weights over the entire range of L)
        Second step:
            Samples initial distribution of mode transition probability matrix pi[j,k]
            - j indexes probability of going TO mode j
            - k indexes the corresponding  mode k to which the probabilities belong to
        """
        v = np.random.beta(1, self.gamma, self.L)
        beta_init = np.zeros(self.L)
        v_iter = 1 - v
        beta_init[0] = v[0]
        for k in range(1,self.L):
            # use inner for-loop instead of reduce -> better performance (?)
            acc = 1
            for q in range(0,k):
                acc *= v_iter[q]
            beta_init[k] = v[k] * acc
        # calculate difference to 1 -> sum over beta will not equal exactly 1 due to truncation
        # then add the remainder on top of some weight (pick the first one arbitrarily, the maginitude is small anyway of the remainder)
        difference = 1 - sum(beta_init)
        beta_init += difference / self.L


        pi_init = np.zeros((self.L, self.L))
        for k in range(0, self.L):
            v = np.zeros(self.L)
            v = beta_init * self.gamma
            #v[k] = v[k] + self.kappa
            # use Dirichlet construction method (converges to DP as L goes to infinity)
            pi_init[:, k] = np.random.dirichlet(v, 1)

        self.param_pi_beta_sampler = {
            'pi': pi_init,
            'beta': beta_init
        }

    def init_params_data(self):
        """
        Initializes parameters that depend on the data.
        """

        self.init_sample_dyn_params()
        self.init_sample_pi_beta()
        # ADD YOUR data dependent parameter initializations HERE


    def init_sample_dyn_params(self):
        """
        Initializes parameters for sampling of dynamical params.
        This initialization is data dependent.
        """
        self.param_dyn_param_sampler = {
            'K': np.eye(self.xdim)*1.0,
            'M': np.zeros((self.xdim, self.xdim)),
            'n_0': 2,
            'S_0': 0.2 * np.eye(self.xdim),
            'r_0': 1e-4,
            'R_0': 1e-4 * np.eye(2),
            'sample_A': 1,
            'ridge_A': 1e-5
        }

    def kalman_sample_x_z(self, z, y, pi, As, Sigmas, R):
        """
        Samples the state sequence z,
        also updates and returns the transition counts n_jk (n: customer in rest. j chooses dish k)

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        y : ndarray
            2D array containing the states/pseudo-observations for each time step
            The dimensionality is [state_dim, T].
        pi: ndarray
            2D array containing the probabilities of transitioning to mode j from current mode k.
            The columns represent the current mode, and the rows the trans. prob.
            The dimensionality is [L, L]
        As : dict
            Dictionary containing for each mode in use the sampled dynamical system matrix A
        Sigmas : dict
            Dictionary containing for each mode in use the sampled noise matrix Sigma
        R : 2D array


        Returns
        -------
        x : ndarray
            2D array containing the current states over time. Columns indexes time step, row indexes dimension of state
            dimension of array is [dim, T]
        z : ndarray
            1D array containing the newly sampled mode assignments for the nodes

        """
        t_sz = self.T
        dim = self.xdim
        P_0 = self.P_0
        C = self.C

        Lambda_f = np.zeros((t_sz, dim, dim))
        Lambda_f[:, :, :] = P_0

        Lambda = np.zeros((t_sz, dim, dim))
        Lambda[:, :, :] = np.identity((dim))

        M = np.zeros((t_sz, dim, dim))
        M[:, :, :] = np.identity((dim))

        J = np.zeros((t_sz, dim, dim))
        J[:, :, :] = np.identity((dim))

        L = np.zeros((t_sz, dim, dim))
        L[:, :, :] = np.identity((dim))

        theta = np.zeros((dim, t_sz))

        theta_f = np.zeros((dim, t_sz))

        # kalman f
        # init

        # Forward Kalman Filter
        # input: Dynamic Matrix A, Lambda_f, Covariance Noise CVN
        # output:
        for t in range(1, t_sz - 1):
            # Compute
            M[t, :, :] = np.dot(np.linalg.inv(As[z[t]]),
                                np.dot(np.linalg.inv(Lambda_f[t, :, :]), np.linalg.inv(As[z[t]])))
            J[t, :, :] = np.dot(M[t, :, :], np.linalg.inv(M[t, :, :] + np.linalg.inv(Sigmas[str(z[t + 1])])))
            L[t, :, :] = np.identity(dim) - J[t, :, :]

            # Predict
            Lambda[t - 1, :, :] = np.dot(L[t, :, :], np.dot(M[t, :, :], np.transpose(L[t, :, :]))) + \
                                  np.dot(J[t, :, :], np.dot(Sigmas[z[t]], np.transpose(J[t, :, :])))
            theta[:, t - 1] = np.dot(L[t, :, :], np.dot(np.linalg.inv(np.transpose(As[z[t]])), \
                                                        (theta_f[:, t] + np.dot(theta_f[:, t],\
                                                                                np.dot(As[z[t]], mu)))))

            # Update
            Lambda_f[t, :, :] = Lambda_f[t - 1, :, :] + np.dot(np.transpose(C), np.dot(np.linalg.inv(R), C))
            theta_f[:, t] = theta[:, t - 1] + np.dot(np.transpose(C), np.dot(np.linalg.inv(R), y[:, t]))
            # kalman b
            # init
        J_b = np.zeros((t_sz, dim, dim))
        J_b[:, :, :] = np.identity((dim))
        L_b = np.zeros((t_sz, dim, dim))
        L_b[:, :, :] = np.identity((dim))
        Lambda_b = np.zeros((t_sz, dim, dim))
        # aa = np.dot(np.dot(np.transpose(C),np.linalg.inv(R)),y[:,t_sz-1])
        theta_b = np.zeros((dim, t_sz))
        Lambda_b[:, :, :] = np.identity((dim))
        Lambda_k = np.zeros((t_sz, dim, dim))
        theta_k = np.zeros((dim, t_sz))
        for t in range(t_sz - 1, 0, -1):
            if t == t_sz - 1:
                Lambda_b[t_sz - 1, :, :] = np.dot(np.transpose(C), np.dot(np.linalg.inv(R), C))
                theta_b[:, t_sz - 1] = np.dot(np.dot(np.transpose(C), np.linalg.inv(R)), y[:, t_sz - 1])
            else:
                # Compute
                J_b[t + 1, :, :] = np.dot(Lambda_b[t + 1, :, :],
                                          np.linalg.inv(Lambda_b[t + 1, :, :] + Sigmas[str(z[t + 1])]))
                L_b[t + 1, :, :] = np.identity(dim) - J_b[t + 1, :, :]

                # predict
                tmp1 = np.dot(L_b[t + 1, :, :], np.dot(Lambda_b[t + 1, :, :], np.transpose(L_b[t + 1, :, :])))
                tmp2 = np.dot(J[t + 1, :, :],
                              np.dot(np.linalg.inv(Sigmas[str(z[t + 1])]), np.transpose(J_b[t + 1, :, :])))
                Lambda = np.dot(np.transpose(As[str(z[t + 1])]), np.dot((tmp1 + tmp2), As[str(z[t + 1])]))
                theta = np.dot(np.transpose(As[str(z[t + 1])]),
                               np.dot(L_b[t + 1, :, :], (theta_b[:, t + 1] - np.dot(Lambda_b[t + 1, :, :], mu))))
                # update
                Lambda_b[t, :, :] = Lambda + np.dot(np.transpose(C), np.dot(np.linalg.inv(R), C))
                theta_b[:, t] = theta + np.dot(np.transpose(C), np.dot(np.linalg.inv(R), y[:, t]))

                # combine Filters
                # Eq 4.27
                # for t in range(0,t_sz):
            Lambda_k[t, :, :] = np.linalg.inv(Sigmas[z[t]] + np.dot(As[z[t]],\
                                                                             np.dot(np.linalg.inv(Lambda_f[t, :, :]),\
                                                                                    np.transpose(As[z[t]]))))
            theta_k[:, t] = np.dot(np.linalg.inv(Sigmas[z[t]] + np.dot(As[z[t]],\
                                                                                np.dot(Lambda_f[t, :, :], np.transpose(\
                                                                                    As[z[t]])))),\
                                   np.dot(As[z[t]], np.dot(np.linalg.inv(Lambda_f[t, :, :]), theta_f[:, t])))

            f_k = np.zeros(L_sz)
            for k in range(0, L_sz):
                tmp = np.dot(np.transpose((-0.5) * theta_k[:, t]),
                             np.dot(np.linalg.inv(Lambda_k[t, :, :]), theta_k[:, t])) + (0.5) * np.dot(
                    np.transpose(theta_k[:, t] + theta_b[:, t]),
                    np.dot(np.linalg.inv(Lambda_k[t, :, :] + Lambda_b[t, :, :]), (theta_k[:, t] + theta_b[:, t])))
                f_k[k] = np.dot(np.linalg.norm(Lambda_k[t, :, :]),
                                np.linalg.norm(np.linalg.inv(Lambda_k[t, :, :] + Lambda_b[t, :, :]))) * np.exp(tmp)

                ######################################
            probabilities = np.zeros(L_sz)
            for k in range(0, L_sz):
                probabilities[k] = (pi[z[t - 1], k]) * (pi[k, z[t + 1]]) * (f_k[k])
            probabilities = probabilities / np.sum(probabilities)
            values = np.arange(0, L_sz)
            # sample new z[t]
            z[t] = np.random.choice(values, 1, p=list(probabilities))
            mean = np.zeros(dim)
            variance = np.zeros((dim, dim))
            mean = np.linalg.inv(np.linalg.inv(Sigmas[z[t]]) + Lambda_b[t, :, :]).dot(
                np.linalg.inv(Sigmas[z[t]]).dot(As[z[t]]).dot(x[:, t - 1]) + theta_b[:, t])
            variance = np.linalg.inv(np.linalg.inv(Sigmas[z[t]]) + Lambda_b[t, :, :])
            x[:, t] = np.random.multivariate_normal(mean, variance)
        return x, z


    # kalman_sample_x_z new
    def sample_state(self, A, z, R, C, Sigma, y, pi):
        """
        Samples the state sequence z,
        also updates and returns the transition counts n_jk (n: customer in rest. j chooses dish k)

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        y : ndarray
            2D array containing the states/pseudo-observations for each time step
            The dimensionality is [state_dim, T].
        pi: ndarray
            2D array containing the probabilities of transitioning to mode j from current mode k.
            The columns represent the current mode, and the rows the trans. prob.
            The dimensionality is [L, L]
        As : dict
            Dictionary containing for each mode in use the sampled dynamical system matrix A
        Sigmas : dict
            Dictionary containing for each mode in use the sampled noise matrix Sigma
        R : 2D array


        Returns
        -------
        x : ndarray
            2D array containing the current states over time. Columns indexes time step, row indexes dimension of state
            dimension of array is [dim, T]
        z : ndarray
            1D array containing the newly sampled mode assignments for the nodes

        """

        horL = np.size(A, 0)
        xdim = self.xdim
        mu = np.zeros(xdim)  # noise mean

        # Forward Kalman Filter and Computation of Lambda_k and theta_k
        # input: Dynamic Matrix A, Covariance Noise Sigma, z, mu, y, ...?
        # initializes: Lambda_f[t_sz,dim,dim], Lambda[t,:,:], M[t,:,:], J[t,:,:], L[t,:,:]
        # output: Lambda_f, theta_f, Lambda_k and theta_k
        # output variables
        Lambda_f = np.ndarray((self.T+1, xdim, xdim))
        theta_f = np.zeros((xdim, self.T+1))

        # temporary variables
        M = np.ndarray((self.T+1, xdim, xdim))
        J = np.ndarray((self.T+1, xdim, xdim))
        L = np.ndarray((self.T+1, xdim, xdim))
        Lambda = np.ndarray((self.T, xdim, xdim))
        theta = np.zeros((xdim, self.T))

        # initialize
        Lambda_f[0, :, :] = np.eye(xdim)  # TODO should be P_0

        # work forwards in time
        for t in range(0, self.T+1):
            t0 = t-1  # t0 is used to index A and Sigma as indices 0..T-1 are used, in the paper however 1..T.

            # Compute
            M[t, :, :] = np.linalg.solve(np.dot(np.dot(A[z[t0+1]].T, Lambda_f[t, :, :]), A[z[t0+1]]), np.eye(xdim))
            J[t, :, :] = np.linalg.solve((M[t, :, :] + np.linalg.inv(Sigma[z[t0+1]])).T, M[t, :, :].T).T
            L[t, :, :] = np.identity(xdim) - J[t, :, :]

            if t == 0:
                continue

            # Predict
            Lambda_s1 = L[t - 1, :, :].dot(M[t - 1, :, :]).dot(np.transpose(L[t - 1, :, :]))
            Lambda_s2 = np.dot(np.linalg.solve(Sigma[z[t0]], J[t, :, :].T), J[t, :, :].T)
            Lambda[t - 1, :, :] = Lambda_s1 + Lambda_s2
            A_inv = np.linalg.inv(A[z[t0], :, :])
            theta[:, t - 1] = L[t - 1, :, :].dot(A_inv.T.dot(theta_f[:, t - 1] + theta_f[:, t - 1].dot(A_inv).dot(mu)))
            # Update
            Lambda_f[t, :, :] = Lambda[t - 1, :, :] + np.transpose(C).dot(np.linalg.inv(R)).dot(C)
            theta_f[:, t] = theta[:, t - 1] + np.transpose(C).dot(np.linalg.inv(R).dot(y[:, t0]))

        # Backward Kalman Filter
        # input: Dynamic Matrix A, Covariance Noise Sigma
        # initializes: Lambda_b[t_sz,dim,dim], Lambda[t,:,:], J_b[t,:,:], L_b[t,:,:]
        # output: Lambda_b, theta_b

        # output variables
        Lambda_b = np.array((self.T, xdim, xdim))
        theta_b = np.array((xdim, self.T))

        # temporary variables
        J_b = np.array((self.T, xdim, xdim))
        L_b = np.array((self.T, xdim, xdim))

        # intitialize
        Lambda_b[self.T, :, :] = C.T.dot(np.linalg.solve(R,C))
        theta_b[:, self.T] = C.T.dot(R, y[:, self.T - 1])

        z_hat = np.array(self.T)

        for t in range(self.T - 1, -1, -1):
            t0 = t-1

            # Compute
            J_b[t + 1, :, :] = Lambda_b[t + 1, :, :].dot(np.linalg.solve(
                (Lambda_b[t+1, :, :] + np.linalg.inv(Sigma[z[t0+1]])).T, Lambda_b[t + 1, :, :].T).T)
            L_b[t + 1, :, :] = np.identity(xdim) - J_b[t + 1, :, :]

            # Predict
            # tmp1 and tmp2 are summands of lambda
            tmp1 = L_b[t + 1, :, :].dot(Lambda_b[t + 1, :, :]).dot(np.transpose(L_b[t + 1, :, :]))
            tmp2 = J[t + 1, :, :].dot(np.linalg.inv(Sigma[z[t0+1]])).dot(np.transpose(J_b[t + 1, :, :]))
            Lambda = A[z[t0+1]].dot(tmp1 + tmp2).dot(A[z[t0 + 1]])
            theta = A[z[t0+1]].dot(L_b[t + 1, :, :]).dot(theta_b[:, t + 1] - Lambda_b[t + 1, :, :].dot(mu))

            # update
            if t > 0:
                Lambda_b[t, :, :] = Lambda + np.transpose(C).dot(np.linalg.inv(R)).dot(C)
                theta_b[:, t] = theta + np.transpose(C).dot(np.linalg.inv(R)).dot(y[:, t0])
            else: #t == 0
                Lambda_b[t, :, :] = Lambda
                theta_b[:, t] = theta

        for t in range(self.T+1, 0, -1):
            t0 = t-1

            f_k = np.array(hor_L)
            for k in range(0, hor_L):

                # Eq 4.27 at page 155
                l_inv = np.linalg.inv(Lambda_f[t - 1])
                Lambda_k = np.linalg.inv(Sigma[k-1] + np.dot(A[z[t0]], np.dot(l_inv, A[z[t0]].T)))

                theta_k = np.dot(Lambda_k[k][t, :, :], np.dot(A[z[t0]], np.dot(l_inv, theta_f[:, t - 1])))


                expo = -0.5 * theta_k.T.dot(np.linalg.solve(Lambda_k, theta_k)) \
                       +0.5 * (Lambda_k + Lambda_b[t, :, :]).T.dot(np.linalg.solve(Lambda_k + Lambda_b[t, :, :], \
                                                                                   theta_k + theta_b[:, t]))

                f_k[k] = np.sqrt(np.linalg.norm(Lambda_k / (Lambda_k + Lambda_b[t]))) * np.exp(expo)

            if t == 1:
                z_distr = 1 * pi[:, z[t0]] * f_k
            elif t == self.T+1:
                z_distr = pi[z[t0 - 1], :] * 1 * f_k
            else:
                z_distr = pi[z[t0 - 1], :] * pi[:, z[t0 + 1]] * f_k

            z_distr /= np.sum(z_distr)
            z_values = np.arange(0, horL)
            z_hat[t0] = np.random.choice(z_values, 1, p=z_distr)

        x = np.array(xdim, self.T)
        for t in range(0, self.T):
            tP = t + 1  # here t indices of implementation (0..T-1) are used instead of the paper form (1..T)

            siginv = np.linalg.inv(Sigma[z_hat[t]])
            x_mean = np.linalg.solve(siginv + Lambda_b[tP, :, :], siginv.dot(A[z_hat[t]]).dot(x[:, t-1]
                                                                                              + theta_b[:, tP]))
            x_var = np.linalg.inv(siginv + Lambda_b[tP, :, :])
            x[:, t] = np.random.multivariate_normal(x_mean, x_var)

        return z_hat, x

    def smoothing_filter(self, y):
        f_sz = 5
        x = np.zeros((self.dims, self.T))
        for i in range(0,self.T):
            if (i < np.trunc(f_sz/2)):
                offset = (np.trunc(f_sz/2) - i)
            if (i > T - np.trunc(f_sz/2)):
                offset = -(np.trunc(f_sz/2) - i)
            else:
                offset = 0

            x[:,i] = y[:, i-np.trunc(f_sz/2) + offset:i+np.trunc(f_sz/2)+offset+1]
        return x



    # SLDS methods (excluding initialization methods)
    def backward_message(self, z, psi, pi, As, Sigmas):
        """
        Calculates the backward messages needed for the subsequent sampling of z
        (See E. Fox thesis page 158 algorithm 14 step 1 a) for reference)

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        psi : ndarray
            2D array containing the states/pseudo-observations for each time step
            The dimensionality is [state_dim, T].
        pi: ndarray
            2D array containing the probabilities of transitioning to mode j from current mode k.
            The columns represent the current mode and the rows the trans. prob.
            The dimensionality is [L, L]
        As : dict
            Dictionary containing for each mode in use the sampled dynamical system matrix A
        Sigmas : dict
            Dictionary containing for each mode in use the sampled noise matrix Sigma

        Returns
        -------
        message : ndarray
                2D array containing the backward messages m_{t+1,t}(k)
                The dimensionality is [L, T]

        """
        L = self.L
        T = self.T
        dims = self.xdim
        message = np.zeros((L, T))
        acc = 0
        sub = 0
        message[:, T - 1] = 1
        unique_z = np.unique(z)
        for t in range(T - 2, -1, -1):
            for k in range(0, L):
                acc = 0
                for l in range(0, L):
                    if np.linalg.norm(As[l]) == 0:
                        acc = acc
                    else:
                        # check if the mode is used, then use the corresponding matirx - if not, use the initial prior matrix for A
                        # -> As will be filled with matrices sampled with respect to the mode -
                        # we need some index that "stores" the generic prior to be accessed
                        #
                        # if l in unique_z:
                        #     pd = norm(A[str(l)] * psi[t], Sigma[str(l)]).pdf(psi[t + 1])
                        # else:
                        if dims == 1:
                            pd = norm(As[l] * psi[:, t], Sigmas[l]).pdf(psi[:, t + 1])
                            acc = acc + message[l, t + 1] * pi[l, k] * pd
                        else:
                            pd = mvnorm(As[l].dot(psi[:, t]), Sigmas[l]).pdf(psi[:, t + 1])
                            acc = acc + message[l, t + 1] * pi[l, k] * pd

                message[k, t] = acc
            if np.sum(message[:,t] == 0):
                message[:,t] = 1
            message[:, t] = message[:, t] / np.sum(message[:, t])

        return message

    def block_sample_z(self, z, psi, pi, As, Sigmas):
        """
        Samples the state sequence z,
        also updates and returns the transition counts n_jk (n: customer in rest. j chooses dish k)
        (See E. Fox thesis page 158 algorithm 14 step 1 b) for reference)

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        psi : ndarray
            2D array containing the states/pseudo-observations for each time step
            The dimensionality is [state_dim, T].
        pi: ndarray
            2D array containing the probabilities of transitioning to mode j from current mode k.
            The columns represent the current mode, and the rows the trans. prob.
            The dimensionality is [L, L]
        As : dict
            Dictionary containing for each mode in use the sampled dynamical system matrix A
        Sigmas : dict
            Dictionary containing for each mode in use the sampled noise matrix Sigma

        Returns
        -------
        z : ndarray
            1D array containing the newly sampled mode assignments for the nodes
        n : ndarrax
            2D array containing the transition counts from mode j to mode k for the entire time sequence
            The dimensionality is [L, L]
        """
        T = self.T
        L = self.L
        dims = self.xdim
        n = np.zeros((L, L), dtype=np.int32)
        messages = self.backward_message(z, psi, pi, As, Sigmas)
        for t in range(1, T):
            f = np.zeros(L)
            probabilities = np.zeros(L)
            for k in range(0, L):
                # f[k] = np.random.multivariate_normal(A[k]*psi[t-1],Sigma[k])*calculate_messages(k,t)
                # check if A matrix is 0 (otherwise norm throws error)

                # calculate likelihood for generating observation in the respective mode k
                pd = 0
                try:
                    # if state dim is 1, mvnorm throws error (1D case "norm" has to be used as far as I know
                    if dims == 1:
                        pd = norm(As[k] * psi[:, t - 1], Sigmas[k]).pdf(psi[:, t])
                    else:
                        pd = mvnorm(As[k].dot(psi[:, t - 1]), Sigmas[k]).pdf(psi[:, t])
                except:
                    # If mean and sigma are 0, also throws error - catch it and just set f = 0 for this k
                    f[k] = 0
                if messages[k, t] == 0 or pd == 0 or pi[z[t - 1], k] == 0:
                    # Kind of another exception catching to force f to be 0 if one of the factors is zero
                    f[k] = 0
                    probabilities[k] = 0
                else:
                    # calculate probability of transitioning into mode k given mode z[t-1]
                    # use log for numerical reasons
                    f[k] = np.log(pd) + np.log(messages[k, t])
                    probabilities[k] = np.log(pi[z[t - 1], k]) + (f[k])
                    probabilities[k] = np.exp(probabilities[k])
            probabilities = probabilities / np.sum(probabilities)
            values = np.arange(0, L)
            # sample new z[t]
            z[t] = np.random.choice(values, 1, p=list(probabilities))
            if t != 0:
                # update n_jk to reflect new transition
                # n_jk stands for all transition from j to k within the entire time series)
                n[z[t - 1], z[t]] = n[z[t - 1], z[t]] + 1
            # add y[t] to the cached statistics

            ### INVESTIGATE USE OF CACHED STATISTICS AND IF NECESSARY
            #self.Y_cached[z[t]]['t'] = y[:, t]

        return z, n

    def sample_pi_beta(self, n):
        """
        Samples the mode transition probability matrix pi and the prior probability distribution beta
        (See E. Fox thesis algorithm 10 step 3,4 and 5 (sampling of pi) for reference

        Parameters
        ----------
        n : ndarray
            2D array containing the transition counts from mode j to mode k for the entire time sequence
            The dimensionality is [L, L]

        Returns
        -------
        pi : ndarray
            2D array containing the probabilities of transitioning to mode j from current mode k.
            The columns represent the current mode, and the rows the trans. prob.
            The dimensionality is [L, L]
        beta : ndarray
            1D array containing the prior probabilities for each mode
            The dimensionality is [L]
        m : ndarray
            2D array containing the transition counts from mode j to mode k for the entire time sequence
            The dimensionality is [L, L] (difference to n: tries to infer the number of tables in restaurant that led
            to transition (e.g., 20x transition from 4 to 5, but only 4 tables with transition counts 8,6,4,2 each)
        """
        params = self.param_pi_beta_sampler
        pi = params['pi']
        beta = params['beta']
        L = self.L
        kappa = self.kappa
        alpha = self.alpha
        rho = self.rho
        gamma = self.gamma
        m = np.zeros((L, L))

        # calculate counts of m (number of tables in current restaurant that consider a dish -
        # for example: 26 customers eating dish K could be eating it at 1,2,..,m tables, which has to be inferred
        for j in range(0, L):
            for k in range(0, L):
                for nn in range(1, n[j, k] + 1):
                    a = 0
                    if j == k:
                        a = kappa
                    p = (alpha * beta[k] + a) / (nn + alpha * beta[k] + a)
                    xx = np.random.binomial(1, p)
                    if xx == 1:
                        m[j, k] += 1
        w = np.zeros(L)
        # calculate w (overriding variables to account for sticky behaviour)
        for j in range(0, L):
            w[j] = np.random.binomial(m[j, j], rho * (rho + beta[j] * (1 - rho)) ** -1)
        m_bar = m - np.diag(w)

        # Alg 10 step 4: re-sample beta using the dirichlet distribution approximation
        vec = gamma/L + np.sum(m_bar, axis=0)
        beta[:] = np.random.dirichlet(vec, 1)

        # Alg 10 Step 5.1: re-sample pi using the dirichlet distribution
        for k in range(0, L):
            #v = np.zeros(L)
            v = beta*alpha + n[k, :]
            v[k] += kappa
            # re-sample pi using the dirichlet distribution approximation
            pi[:, k] = np.random.dirichlet(v, 1)

        return pi, beta, m, w

    def sample_hyperparameters(self, z, n, m, w):
        # alpha = (1-rho)(alpha + kappa)
        #  kappa = rho(alpha + kappa),
        """
        Samples the hyperparameters gamma, (alpha + kappa) and rho

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        n : ndarray
            2D array containing the transition counts from mode j to mode k for the entire time sequence
            The dimensionality is [L, L]
        m : ndarray
            2D array containing the transition counts from mode j to mode k for the entire time sequence
            The dimensionality is [L, L] (difference to n: tries to infer the number of tables in restaurant that led
            to transition (e.g., 20x transition from 4 to 5, but only 4 tables with transition counts 8,6,4,2 each)

        Returns
        -------
        ak : float
            transformed parameter consisting of spread parameter for pi distribution along with sticky parameter
        gamma : float
            spread parameter for beta distribution
        """

        unique_z = np.unique(z)
        sz = np.size(unique_z)

        sum_S = 0
        sum_R = 0
        for j in unique_z:
            sum_S += np.random.binomial(1, np.sum(n[j, :])/(np.sum(n[j, :]) + self.alpha + self.kappa))
            sum_R += np.log(np.random.beta(self.alpha + self.kappa + 1, np.sum(n[j, :])))
        A = self.a + np.sum(m) - sum_S
        B = self.b - sum_R
        ak = np.random.gamma(A, B)

        Kbar = 0
        Kbar = sz
        for k in unique_z:
            if np.sum(m[:, k]) == 0 and m[k, k] > 0:
                Kbar -= 1
        eta = np.random.beta(self.gamma + 1, np.sum(m))
        xi = np.random.binomial(1, np.sum(m)/(np.sum(m) + self.gamma))
        gamma = np.random.gamma(self.e + Kbar - xi, self.f - np.log(eta))

        rho = np.random.beta(np.sum(w) + self.c, np.sum(m) - np.sum(w) + self.d)
        return gamma, ak, rho


    def sample_dyn_params(self, z, x, A, Sigma):
        """
        Samples the dynamical parameters \A and \Sigma for every mode being used.

        Parameters
        ----------
        z : ndarray
            1D array containing the mode assignments for the nodes
        x : ndarray
            2D array containing the states of the modes
            The dimensionality is [state_dim, T].

        Returns
        -------
        As : dict
            Dictionary containing for each mode in use the sampled dynamical system matrix A
        Sigma: dict
            Dictionary containing for each mode in use the sampled noise matrix \Sigma
        """

        uniquez = np.unique(z)
        params = self.param_dyn_param_sampler
        As = A.copy()
        Sigmas = Sigma.copy()

        """
        for k in uniquez:
            ind = np.where(k == z)[0]
            if ind[0] == 0:  # remove time step 0 as there is no information about A
                ind = ind[1:]

            N_k = np.size(ind)
            ind_prev = np.array(ind) - 1
            Psi = x[:, ind]  # 15.1
            PsiHat = x[:, ind_prev]  # 15.1

            S_hh = np.dot(PsiHat, PsiHat.T) + params['K']  # S _ phi dash, phi dash
            S_wh = np.dot(Psi, PsiHat.T) + np.dot(params['M'], params['K'])  # S _ phi, phi dash
            S_ww = np.dot(Psi, Psi.T) + np.dot(np.dot(params['M'], params['K']), params['M'].T)  # S _ phi, phi
            S_wdh = S_ww - np.dot(S_wh, np.linalg.inv(S_hh).dot(S_wh.T))  # S _ phi | phi dash

            Sigma = invwishart.rvs(N_k + params['n_0'], S_wdh + params['S_0'])

            # Sample transition matrix A
            if params['sample_A']:
                A_mean = np.dot(np.linalg.inv(S_hh.T), S_wh.T).T  # mean of matrix normal distribution
                # transform Matrix A_mean to vector
                A_vec_mean = np.reshape(A_mean, (np.prod(np.shape(A_mean))))

                # sample from matrix normal distribution (def. is some pages before in the paper)
                A_vec = multivariate_normal.rvs(A_vec_mean, np.kron(np.linalg.inv(S_hh), Sigma))
                # convert sample into matrix form
                A_hat = np.reshape(A_vec, np.shape(A_mean))
            else:
                reg = params['ridge_A']
                A_hat = np.linalg.solve(np.dot(PsiHat, PsiHat.T) + reg * np.eye(self.xdim), np.dot(PsiHat, Psi.T)).T
            """
        for k in range(0, self.L):
            if k in uniquez:
                ind = np.where(k == z)[0]
                if ind[0] == 0:  # remove time step 0 as there is no information about A
                    ind = ind[1:]

                N_k = np.size(ind)
                ind_prev = np.array(ind) - 1
                Psi = x[:, ind]  # 15.1
                PsiHat = x[:, ind_prev]  # 15.1

                S_hh = np.dot(PsiHat, PsiHat.T) + params['K']  # S _ phi dash, phi dash
                S_wh = np.dot(Psi, PsiHat.T) + np.dot(params['M'], params['K'])  # S _ phi, phi dash
                S_ww = np.dot(Psi, Psi.T) + np.dot(np.dot(params['M'], params['K']), params['M'].T)  # S _ phi, phi
                S_wdh = S_ww - np.dot(S_wh, np.linalg.inv(S_hh).dot(S_wh.T))  # S _ phi | phi dash

                Sigma = invwishart.rvs(N_k + params['n_0'], S_wdh + params['S_0'])

                # Sample transition matrix A
                if params['sample_A']:
                    A_mean = np.dot(np.linalg.inv(S_hh.T), S_wh.T).T  # mean of matrix normal distribution
                    # transform Matrix A_mean to vector
                    A_vec_mean = np.reshape(A_mean, (np.prod(np.shape(A_mean))))

                    # sample from matrix normal distribution (def. is some pages before in the paper)
                    A_vec = multivariate_normal.rvs(A_vec_mean, np.kron(np.linalg.inv(S_hh), Sigma))
                    # convert sample into matrix form
                    A_hat = np.reshape(A_vec, np.shape(A_mean))
                else:
                    reg = params['ridge_A']
                    A_hat = np.linalg.solve(np.dot(PsiHat, PsiHat.T) + reg * np.eye(self.xdim), np.dot(PsiHat, Psi.T)).T
                As[k] = A_hat
                Sigmas[k] = Sigma
            else:
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

    def sample_measure_noise_covmat(self, x, y, C):
        #assert np.shape(x,1) == np.shape(y,1)
        params = self.param_dyn_param_sampler

        diff = y - np.dot(C, x)
        S_R = np.dot(diff, diff.T)

        # assume R is shared among nodes
        R = invwishart.rvs(self.T + params['r_0'], S_R + params['R_0'])
        return R

