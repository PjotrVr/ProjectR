from abc import ABC, abstractmethod

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MixtureDist(ABC):
    def __init__(self, pi):
        self.pi = np.array(pi)
        self.K = len(self.pi)

    @abstractmethod
    def sample(self, L):
        pass

    @abstractmethod
    def p_xz(self, x, k):
        pass

    def p_x(self, x):
        return np.sum([self.pi[k] * self.p_xz(x, k) for k in range(self.K)], axis=0)

class GMDist(MixtureDist):
    def __init__(self, pi, mu, sigma2):
        super().__init__(pi)
        self.mu = np.array(mu)
        self.sigma2 = np.array(sigma2)

    @classmethod
    def random(cls, K):
        mu = np.random.normal(size=K)
        sigma2 = np.exp(np.random.normal(size=K) - 1)
        pi = np.random.dirichlet(5 * np.ones(K))
        return cls(pi, mu, sigma2)

    def sample(self, L):
        choices = np.random.choice(np.arange(self.K), size=L, p=self.pi)
        return np.random.normal(self.mu[choices], np.sqrt(self.sigma2[choices]))

    @staticmethod
    def normal_pdf(x, mu, sigma2):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

    def p_xz(self, x, k):
        return self.normal_pdf(x, self.mu[k], self.sigma2[k])

class UMDist(MixtureDist):
    def __init__(self, pi, a, b):
        super().__init__(pi)
        self.a = np.array(a)
        self.b = np.array(b)

    @classmethod
    def random(cls, K):
        a = np.random.uniform(-1, 3, size=K)
        b = np.random.uniform(3, 5, size=K)
        pi = np.random.dirichlet(5 * np.ones(K))
        return cls(pi, a, b)

    def sample(self, L):
        choices = np.random.choice(np.arange(self.K), size=L, p=self.pi)
        return np.random.uniform(self.a[choices], self.b[choices])

    @staticmethod
    def uniform_pdf(x, a, b):
        return (x >= a) * (x <= b) / (b - a)

    def p_xz(self, x, k):
        return self.uniform_pdf(x, self.a[k], self.b[k])


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
        gamma = weighted_pdfs / np.sum(gamma, axis=1, keepdims=True)
        return gamma
        

    def m_step(self, x, gamma):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        self.pi = np.mean(gamma, axis=0)
        self.mu = np.sum(gamma * x, axis=0) / np.sum(gamma, axis=0)
        self.sigma2 = np.sum(gamma * (x - self.mu)**2, axis=0) / np.sum(gamma, axis=0)
        self.sigma2 = np.maximum(self.sigma2, 1e-6)
    
    def log_likelihood(self, x):
        N = x.shape[0]
        weighted_pdfs = self.pi * norm.pdf(x.reshape(-1, 1), loc=self.mu, scale=np.sqrt(self.sigma2))
        total_prob = np.sum(weighted_pdfs, axis=1)
        log_total_prob = np.log(total_prob + 1e-10)
        log_likelihood = np.sum(log_total_prob)
        return log_likelihood / N

    def p_xz(self, x, k):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        return norm.pdf(x, loc=self.mu[k], scale=np.sqrt(self.sigma2[k]))
        
    def p_x(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        weighted_pdfs = self.pi * norm.pdf(x, loc=self.mu, scale=np.sqrt(self.sigma2))
        return np.sum(weighted_pdfs, axis=1) 
    

np.random.seed(30)
K = 3
L = 1_000_000
dist = GMDist.random(K)
X = dist.sample(L)

gmm = GMM1d(K)
num_iterations = 50
# gmm.init_from_samples(X)

for i in range(num_iterations):
    gamma = gmm.e_step(X)
    gmm.m_step(X, gamma)
    
    ll = gmm.log_likelihood(X)
    print(f"Epoch: {i}/{num_iterations}, LL: {ll:.3f}")
    
linsp = np.linspace(X.min(), X.max(), 1000)
plt.plot(linsp, dist.p_x(linsp), label="True PDF")
plt.plot(linsp, gmm.p_x(linsp), label="Estimated PDF")
plt.hist(X, bins=1000, density=True, alpha=0.5, label="Data")
plt.legend()
plt.show()