# ============================================================
# Group information
# ============================================================

group_names     = ['Selin Yucebiyik', 'Maria Paula Sanchez', 'Antonis Adamou', 'Ara Mokree']
cid_numbers     = ['01868843', '06045575', '06067135', '06036735']
oral_assessment = [0, 1]  # indices of group members doing the oral assessment

# ============================================================
# Imports (restricted to allowed packages)
# ============================================================

import numpy as np
import random
from datetime import datetime
from scipy.spatial.distance import cdist

# NOTE: In the coursework environment, objective_func will be provided.
# If you want to run this file standalone, uncomment and adjust the import below
# according to the notebook / package you were given.
# from MLCE_CWBO2025.virtual_bioprocess import objective_func


# ============================================================
# Helper: encode categorical "celltype" into numeric value
# ============================================================

celltype_map = {'celltype_1': 0, 'celltype_2': 1, 'celltype_3': 2}

def encode_X(X_raw):
    """
    Convert a list of points with 'celltype_*' strings into a numeric numpy array.
    Each point has the form:
        [temperature, pH, f1, f2, f3, 'celltype_*']
    """
    X_num = []
    for row in X_raw:
        temp, pH, f1, f2, f3, cell = row
        X_num.append([temp, pH, f1, f2, f3, celltype_map[cell]])
    return np.array(X_num, dtype=float)


# ============================================================
# Gaussian Process surrogate with RBF kernel (using numpy + scipy only)
# ============================================================

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Squared exponential (RBF) kernel:
        k(x, x') = variance * exp(-0.5 * ||(x - x') / lengthscale||^2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = cdist(X1 / lengthscale, X2 / lengthscale, 'sqeuclidean')
    return variance * np.exp(-0.5 * sqdist)


class GPSurrogate:
    """
    Simple Gaussian Process surrogate model with:
    - RBF kernel
    - Gaussian noise on observations
    - Cholesky-based inference
    """

    def __init__(self, lengthscale=1.0, variance=1.0, noise=1e-6):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.X = None
        self.Y = None
        self.L = None      # Cholesky factor of K
        self.alpha = None  # K^{-1} y via Cholesky

    def fit(self, X, Y):
        """
        Fit GP to data.
        X: (n, d) array of inputs
        Y: (n,) array of outputs
        """
        self.X = np.atleast_2d(X)
        self.Y = np.array(Y).flatten()

        K = rbf_kernel(self.X, self.X, self.lengthscale, self.variance)
        K += self.noise * np.eye(len(self.X))

        # Cholesky decomposition K = L L^T
        self.L = np.linalg.cholesky(K)

        # alpha = K^{-1} y using Cholesky
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y))

    def predict(self, X_star):
        """
        Predict GP mean and standard deviation at test points X_star.
        Returns:
            mu:  (m,) mean predictions
            std: (m,) standard deviations
        """
        X_star = np.atleast_2d(X_star)

        # Cross-kernel between test and training
        K_star = rbf_kernel(X_star, self.X, self.lengthscale, self.variance)

        # Predictive mean
        mu = K_star.dot(self.alpha)

        # Predictive variance
        v = np.linalg.solve(self.L, K_star.T)
        K_ss = rbf_kernel(X_star, X_star, self.lengthscale, self.variance)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)

        # Numerical safety
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)
        return mu, std


# ============================================================
# Acquisition function: Upper Confidence Bound (UCB)
# ============================================================

def ucb_acquisition(mu, std, kappa=2.0):
    """
    Upper Confidence Bound acquisition:
        a(x) = mu(x) + kappa * sigma(x)
    Larger kappa -> more exploration.
    """
    return mu + kappa * std


# ============================================================
# Batch selection using acquisition function
# ============================================================

class AcquisitionSelection:
    """
    Select a batch of points from X_searchspace using a GP surrogate and UCB.
    """

    def __init__(self, X_searchspace_raw, gp_model, X_obs_raw, Y_obs, batch, kappa=2.0):
        self.batch = batch

        # Encode observed data to numeric form for GP
        X_obs = encode_X(X_obs_raw)
        gp_model.fit(X_obs, Y_obs)

        # Encode search space once for prediction
        self.X_searchspace_raw = X_searchspace_raw
        X_search = encode_X(X_searchspace_raw)

        # Predict on all candidate points in the search space
        mu, std = gp_model.predict(X_search)

        # Compute acquisition values (UCB)
        acq = ucb_acquisition(mu, std, kappa=kappa)

        # Select indices of top "batch" acquisition values
        best_indices = np.argsort(acq)[-batch:]  # largest acquisition values

        # Store the selected batch in raw format (with 'celltype_*' strings)
        self.X_batch_raw = [self.X_searchspace_raw[i] for i in best_indices]


# ============================================================
# Bayesian Optimisation (Batch BO) class
# ============================================================

class BO:
    """
    Batch Bayesian Optimisation using:
    - GP surrogate with RBF kernel
    - UCB acquisition
    - Categorical celltype encoded as integer

    Parameters
    ----------
    X_initial : list of initial points (each point is [temp, pH, f1, f2, f3, 'celltype_*'])
    X_searchspace : list of all candidate points with same format as above
    iterations : int, number of BO iterations
    batch : int, batch size (max 5 as per coursework)
    objective_func : callable, virtual bioprocess experiment
    """

    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func):
        start_time = datetime.timestamp(datetime.now())

        self.X_searchspace = X_searchspace
        self.iterations    = iterations
        self.batch         = batch
        self.objective_func = objective_func

        # Store all observed data (raw format for objective_func)
        self.X = list(X_initial)
        self.Y = self.objective_func(self.X)

        # Time bookkeeping: one non-zero per batch, zeros for the rest
        self.time  = [datetime.timestamp(datetime.now()) - start_time]
        self.time += [0] * (len(self.X) - 1)
        start_time = datetime.timestamp(datetime.now())

        # GP surrogate model (hyperparameters can be tuned if desired)
        gp = GPSurrogate(lengthscale=10.0, variance=1.0, noise=1e-6)

        # Main BO loop
        for self.iteration in range(iterations):
            # Select next batch of points using acquisition
            acq_selector = AcquisitionSelection(
                X_searchspace_raw=self.X_searchspace,
                gp_model=gp,
                X_obs_raw=self.X,
                Y_obs=self.Y,
                batch=self.batch,
                kappa=2.0
            )

            # New batch of candidate experiments (raw format)
            X_next = acq_selector.X_batch_raw

            # Evaluate the objective function on the new batch
            Y_next = self.objective_func(X_next)

            # Update collected data
            self.X.extend(X_next)
            self.Y = np.concatenate([self.Y, Y_next])

            # Update time bookkeeping
            self.time += [datetime.timestamp(datetime.now()) - start_time]
            self.time += [0] * (len(Y_next) - 1)
            start_time = datetime.timestamp(datetime.now())


# ============================================================
# BO Execution Block
# ============================================================

# Define initial design (4 starting experiments)
X_initial = [
    [33, 6.25, 10, 20, 20, 'celltype_1'],
    [38, 8.00, 20, 10, 20, 'celltype_3'],
    [37, 6.80,  0, 50,  0, 'celltype_1'],
    [36, 6.00, 20, 20, 10, 'celltype_2']
]

# Define search space grid
temp      = np.linspace(30, 40, 5)
pH        = np.linspace(6,  8,  5)
f1        = np.linspace(0, 50, 5)
f2        = np.linspace(0, 50, 5)
f3        = np.linspace(0, 50, 5)
celltypes = ['celltype_1', 'celltype_2', 'celltype_3']

X_searchspace = [
    [a, b, c, d, e, f]
    for a in temp
    for b in pH
    for c in f1
    for d in f2
    for e in f3
    for f in celltypes
]

# Only run BO here if objective_func is available in the current environment
if __name__ == "__main__":
    if 'objective_func' in globals():
        # 15 iterations, batch size 5 (as specified in the coursework)
        BO_model = BO(
            X_initial=X_initial,
            X_searchspace=X_searchspace,
            iterations=15,
            batch=5,
            objective_func=objective_func
        )

        # Example: print best observed value (optional)
        best_idx = int(np.argmax(BO_model.Y))
        print("Best observed titre:", BO_model.Y[best_idx])
        print("Best conditions:", BO_model.X[best_idx])
    else:
        print("objective_func is not defined. "
              "Import it from the provided MLCE_CWBO2025 package to run BO.")