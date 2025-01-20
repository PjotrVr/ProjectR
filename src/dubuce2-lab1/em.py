import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal

from dists import GMDist, sample_gauss_2d
from utils import visualize_pdf, visualize_2d_gmm

class GMM1d:
    def __init__(self, K):
        self.K = K
        self.pi = np.ones(K) / K
        self.mu = np.random.randn(K)
        self.sigma2 = np.ones(self.K)

    def init_from_samples(self, x):
        x = x.reshape(-1)
        indices = np.random.choice(len(x), size=self.K, replace=False)
        self.mu = x[indices]
        
    def e_step(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        weighted_pdfs = self.pi * norm.pdf(x, loc=self.mu, scale=np.sqrt(self.sigma2))
        gamma = weighted_pdfs / np.sum(weighted_pdfs, axis=1, keepdims=True)
        return gamma

    def m_step(self, x, gamma):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        Nk = np.sum(gamma, axis=0)
        self.pi = Nk / len(x)
        self.mu = np.sum(gamma * x, axis=0) / Nk
        self.sigma2 = np.sum(gamma * (x - self.mu)**2, axis=0) / Nk
        self.sigma2 = np.maximum(self.sigma2, 1e-6)
    
    def log_likelihood(self, x):
        weighted_pdfs = self.pi * norm.pdf(x.reshape(-1, 1), loc=self.mu, scale=np.sqrt(self.sigma2))
        total_prob = np.sum(weighted_pdfs, axis=1)
        log_total_prob = np.log(total_prob + 1e-10)
        log_likelihood = np.mean(log_total_prob)
        return log_likelihood

    def p_xz(self, x, k):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        return norm.pdf(x, loc=self.mu[k], scale=np.sqrt(self.sigma2[k]))
        
    def p_x(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        weighted_pdfs = self.pi * norm.pdf(x, loc=self.mu, scale=np.sqrt(self.sigma2))
        return np.sum(weighted_pdfs, axis=1) 
    

class GMM:
    def __init__(self, D, K):
        self.K = K
        self.D = D
        self.pi = np.ones(K) / K
        self.mu = np.random.randn(K, D)
        self.sigma2 = np.array([np.eye(D) for _ in range(K)])

    def init_from_samples(self, x):
        indices = np.random.choice(len(x), size=self.K, replace=False)
        self.mu = x[indices]
    
    def e_step(self, x):
        pdfs = np.array([
            multivariate_normal.pdf(x, mean=self.mu[k], cov=self.sigma2[k])
            for k in range(self.K)
        ]).T  # (N, K)

        weighted_pdfs = self.pi * pdfs  # (N, K)
        gamma = weighted_pdfs / np.sum(weighted_pdfs, axis=1, keepdims=True)
        return gamma

    def m_step(self, x, gamma):
        Nk = np.sum(gamma, axis=0)
        self.pi = Nk / len(x)
        self.mu = (gamma.T @ x) / Nk[:, np.newaxis]  # (K, D)

        for k in range(self.K):
            diff = x - self.mu[k]  # (N, D)
            weighted_diff = gamma[:, k][:, np.newaxis] * diff  # (N, D)
            self.sigma2[k] = (weighted_diff.T @ diff) / Nk[k]  # (D, D)
            self.sigma2[k] += 1e-6 * np.eye(self.D)  # Avoid zeros on diagonal

    def log_likelihood(self, x):
        pdfs = np.array([
            multivariate_normal.pdf(x, mean=self.mu[k], cov=self.sigma2[k])
            for k in range(self.K)
        ]).T  # (N, K)

        total_prob = np.sum(self.pi * pdfs, axis=1)  # (N,)
        log_total_prob = np.log(total_prob + 1e-10)
        log_likelihood = np.mean(log_total_prob)  # Avoid log(0)
        return log_likelihood

    def p_xz(self, x, k):
        return multivariate_normal.pdf(x, mean=self.mu[k], cov=self.sigma2[k])
        
    def p_x(self, x):
        pdfs = np.array([
            multivariate_normal.pdf(x, mean=self.mu[k], cov=self.sigma2[k])
            for k in range(self.K)
        ]).T  # (N, K)

        return np.sum(self.pi * pdfs, axis=1)  # (N,)


# 1d data
np.random.seed(30)
K = 3
L = 1_000_000
dist = GMDist.random(K)
X = dist.sample(L)

gmm = GMM1d(K)
gmm.init_from_samples(X)

epochs = 50
for i in range(epochs):
    gamma = gmm.e_step(X)
    gmm.m_step(X, gamma)
    
    ll = gmm.log_likelihood(X)
    print(f"Epoch: {i}/{epochs}, LL: {ll:.3f}")
    
visualize_pdf(gmm, dist, X)

# 2d data
np.random.seed(30)
X, y = sample_gauss_2d(num_classes=2, num_samples_per_class=1000)
gmm = GMM(D=2, K=2)
gmm.init_from_samples(X)

epochs = 100
for i in range(epochs):
    gamma = gmm.e_step(X)
    gmm.m_step(X, gamma)

    ll = gmm.log_likelihood(X)
    print(f"Epoch: {i}/{epochs}, LL: {ll:.3f}")
    
visualize_2d_gmm(gmm, X, y)