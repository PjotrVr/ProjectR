import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import data

# Globals
np.random.seed(100)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "imgs"
os.makedirs(IMG_DIR, exist_ok=True)

class PTDeep(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, device="cpu"):
        super().__init__()
        torch.manual_seed(100)
        self.activation = activation()
        self.flat = nn.Flatten()
        self.layers = nn.Sequential()
        
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        for i, (dim_in, dim_out) in enumerate(zip(layers, layers[1:])):
            self.layers.add_module(f"fc{i}", nn.Linear(dim_in, dim_out, device=self.device))
            if i != len(layers) - 1:
                self.layers.add_module(f"act{i}", activation())
        
        self.layers.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self.flat(X)
        X = self.layers(X)
        return X

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        loss = -torch.sum(Yoh_ * torch.log(probs + 1e-12)) / X.shape[0]
        return loss

    def count_params(self):
        total = 0
        for name, param in self.named_parameters():
            print(f"{name} ... {list(param.shape)}, numel={param.numel()}")
            total += param.numel()
        return total

    @torch.no_grad()
    def predict(self, X):
        probs = self.forward(X)
        Y = probs.argmax(axis=1)
        return Y

def evaluate(model, X, Y_):
    Y = model.predict(X)
    M, accuracy, recall, precision = data.eval_perf_multi(Y.detach().cpu().numpy().astype(np.int32), Y_.detach().cpu().numpy().astype(np.int32))
    return M, accuracy, recall, precision

def class_to_onehot(y):
    C = y.max() + 1
    yoh = torch.zeros(y.shape[0], C, device=y.device)
    yoh[range(y.shape[0]), y] = 1
    return yoh
    
def load_mnist(dirname="/tmp/mnist"):
    mnist_train = torchvision.datasets.MNIST(dirname, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dirname, train=False, download=True)

    X_train, y_train = mnist_train.data, mnist_train.targets
    X_test, y_test = mnist_test.data, mnist_test.targets
    X_train, X_test = X_train.float().div_(255.0), X_test.float().div_(255.0)

    return X_train.to(DEVICE), y_train.to(DEVICE), X_test.to(DEVICE), y_test.to(DEVICE)

def train_without_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=1000, lr=1e-1, weight_decay=1e-3, show_freq=100):
    model = PTDeep(config, device=DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_hist = []
    test_loss_hist = []
    best_model = deepcopy(model)
    best_loss = float("inf")

    for i in range(num_epochs):
        optimizer.zero_grad()
        loss = model.get_loss(X_train, yoh_train)
        loss.backward()
        optimizer.step()
        if i % show_freq == 0:
            with torch.no_grad():
                test_loss = model.get_loss(X_test, yoh_test)
                test_loss_hist.append(test_loss.item())
                if best_loss > test_loss:
                    best_loss = test_loss
                    best_model = deepcopy(model)
                print(f"Step: {i}/{num_epochs}, Train Loss: {loss:.2f}, Test Loss: {test_loss:.2f}")
                
    return best_model, train_loss_hist, test_loss_hist

def train_with_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=1000, lr=1e-1, weight_decay=1e-3, show_freq=100, patience=3, tolerance=1e-3):
    model = PTDeep(config, device=DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_hist = []
    test_loss_hist = []
    best_model = deepcopy(model)
    best_loss = float("inf")
    curr_patience = 0

    for i in range(num_epochs):
        optimizer.zero_grad()
        loss = model.get_loss(X_train, yoh_train)
        loss.backward()
        optimizer.step()
        train_loss_hist.append(loss.item())
        if i % show_freq == 0:
            with torch.no_grad():
                test_loss = model.get_loss(X_test, yoh_test)
                test_loss_hist.append(test_loss.item())
                if best_loss - test_loss > tolerance:
                    best_loss = test_loss
                    best_model = deepcopy(model)
                    curr_patience = 0
                else:
                    curr_patience += 1
                print(f"Step: {i + 1}/{num_epochs}, Train Loss: {loss:.2f}, Test Loss: {test_loss:.2f}")
                if curr_patience >= patience:
                    print(f"Patience reached {curr_patience}")
                    print("Early stop")
                    break

    return best_model, train_loss_hist, test_loss_hist

def visualize_weights(w, filename):
    w = w.reshape(C, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        axes[i].set_title(f"Weights for digit {i}")
        axes[i].imshow(w[i])
        axes[i].axis("off")

    plt.savefig(os.path.join(IMG_DIR, filename), dpi=300)

def part1(config, X_train, yoh_train, X_test, yoh_test):
    # Hyperparams
    lr = 1e-1
    
    print("Training without early stop:")
    model, train_loss_hist, test_loss_hist = train_without_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=100, lr=lr, weight_decay=1e-3)
    M, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")
    visualize_weights(model.layers[0].weight.detach().cpu().numpy(), "weight_reg_1e-3.png")

    print("\nTraining with early stop:")
    model, train_loss_hist, test_loss_hist = train_with_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=1000, lr=lr, weight_decay=1e-3)
    M, accuracy, recall, precision = evaluate(model, X_test, yoh_test)
    print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")
    visualize_weights(model.layers[0].weight.detach().cpu().numpy(), "weight_reg_1e-3_early_stop.png")

    print("\nTraining overfitting:")
    model, train_loss_hist, test_loss_hist = train_without_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=2000, lr=lr, weight_decay=0.0)
    M, accuracy, recall, precision = evaluate(model, X_test, yoh_test)
    print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")
    visualize_weights(model.layers[0].weight.detach().cpu().numpy(), "weight_overfit.png")

    print("\nTraining big regularization:")
    model, train_loss_hist, test_loss_hist = train_without_early_stop(config, X_train, yoh_train, X_test, yoh_test, num_epochs=1000, lr=lr, weight_decay=1e-1)
    M, accuracy, recall, precision = evaluate(model, X_test, yoh_test)
    print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")
    visualize_weights(model.layers[0].weight.detach().cpu().numpy(), "weight_reg_1e-1.png")    


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    N = X_train.shape[0]
    D = X_train.shape[1] * X_train.shape[2]
    C = y_train.max().add_(1).item()

    yoh_train = class_to_onehot(y_train)
    yoh_test = class_to_onehot(y_test)
    h = 100
    configs = [(D, C), (D, h, C), (D, h, h, C), (D, h, h, h, C)]

    part1(configs[0], X_train, yoh_train, X_test, yoh_test)

