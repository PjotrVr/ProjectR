from abc import ABC, abstractmethod
import numpy as np

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
    

class Random2DGaussian:
    horizontal_min = 0
    horizontal_max = 10
    vertical_min = 0
    vertical_max = 10
    scale = 5

    def __init__(self):
        horizontal_range, vertical_range = self.horizontal_max - self.horizontal_min, self.vertical_max - self.vertical_min
        mean = (self.horizontal_min, self.vertical_min)
        mean += np.random.random_sample(2) * (horizontal_range, vertical_range)

        # Variances for principle directions (horizontal/vertical)
        eigvals = np.random.random_sample(2)
        eigvals *= (horizontal_range / self.scale, vertical_range / self.scale)
        eigvals **= 2

        # Pick random rotation [0, 1> * 2pi = [0, 2pi>
        theta = np.random.random_sample() * np.pi * 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Covariance matrix
        Sigma = R.T @ np.diag(eigvals) @ R

        self.get_sample = lambda n: np.random.multivariate_normal(mean, Sigma, n)

# One Gaussian source per class
def sample_gauss_2d(num_classes, num_samples_per_class):
    # Create Gaussians
    Gs, ys = [], []
    for i in range(num_classes):
        Gs.append(Random2DGaussian())
        ys.append(i)

    # Sample dataset
    X = np.vstack([G.get_sample(num_samples_per_class) for G in Gs])
    y = np.hstack([[y] * num_samples_per_class for y in ys])
    return X, y

# One class can have multiple Gaussian components
def sample_gmm_2d(num_components, num_classes, num_samples_per_class):
    # Create Gaussian components and assign them random class idx
    Gs, ys = [], []
    for _ in range(num_components):
        Gs.append(Random2DGaussian())
        ys.append(np.random.randint(num_classes))

    # Sample dataset
    X = np.vstack([G.get_sample(num_samples_per_class) for G in Gs])
    y = np.hstack([[y] * num_samples_per_class for y in ys])
    return X, y