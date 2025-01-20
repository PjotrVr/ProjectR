import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize_pdf(gmm, dist, X):
    linsp = np.linspace(X.min(), X.max(), 1000)
    plt.plot(linsp, dist.p_x(linsp), label="True PDF")
    plt.plot(linsp, gmm.p_x(linsp), label="Estimated PDF")
    plt.hist(X, bins=1000, density=True, alpha=0.5, label="Data")
    plt.legend()
    plt.show()


# Plotting functions from internet
def plot_ellipse(ax, mean, cov, color, n_std=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, fc='none', lw=2)
    ax.add_patch(ellipse)

def visualize_2d_gmm(gmm, X, y):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
    ax[0].set_title("Data Points with True Labels")
    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")
    
    gamma = gmm.e_step(X)
    cluster_assignments = np.argmax(gamma, axis=1)
    ax[1].scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='tab10', s=10, alpha=0.7)
    ax[1].set_title("Data Points with GMM Clusters")
    ax[1].set_xlabel("X1")
    ax[1].set_ylabel("X2")
    
    for k in range(gmm.K):
        plot_ellipse(ax[1], gmm.mu[k], gmm.sigma2[k], color='red')
    
    plt.tight_layout()
    plt.show()

