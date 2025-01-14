import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

import os
from copy import deepcopy
from math import log
from datetime import datetime

import data

# Globals
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "imgs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))

    def forward(self, x):
        return x @ self.weight.T + self.bias

class FCN(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, device="cpu"):
        super().__init__()
        self._layers = layers
        self.activation = activation()
        self.flat = nn.Flatten()
        self.layers = nn.Sequential()

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        for i, (dim_in, dim_out) in enumerate(zip(layers, layers[1:])):
            self.layers.add_module(f"fc{i}", Linear(dim_in, dim_out, device=self.device))
            if i != len(layers) - 2:
                self.layers.add_module(f"act{i}", activation())

        self.layers.append(nn.Softmax(dim=1))

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
        preds = probs.argmax(axis=1)
        return preds

def save_state(filename, model, optimizer, train_loss_history, val_loss_history):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    }
    
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(state, filename)
    print(f"Saved to {filename}")

def load_state(filename, model, optimizer):
    state = torch.load(filename)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    train_loss_history = state["train_loss_history"]
    val_loss_history = state["val_loss_history"]
    print(f"Loaded from {filename}")

    return model, optimizer, train_loss_history, val_loss_history

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

def train(model, X_train, yoh_train, X_val, yoh_val, optimizer=optim.SGD, scheduler=False, epochs=10, lr=1e-4, weight_decay=1e-3, show_freq=1, batch_size=100, lr_decay=1-1e-4, seed=100, dirname=None, verbose=True):
    torch.manual_seed(seed)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    if batch_size == -1 or batch_size >= X_train.shape[0] or batch_size == None:
        batch_size = X_train.shape[0]

    train_loss_history, val_loss_history = [], []
    best_model = deepcopy(model)
    best_loss = float("inf")

    for i in range(epochs):
        model.train()
        shuffle_idx = torch.randperm(X_train.shape[0])
        X_train_shuffled = X_train[shuffle_idx]
        yoh_train_shuffled = yoh_train[shuffle_idx]

        for j in range(0, X_train.shape[0], batch_size):
            X_train_batch = X_train_shuffled[j:j+batch_size]
            yoh_train_batch = yoh_train_shuffled[j:j+batch_size]

            optimizer.zero_grad()
            loss = model.get_loss(X_train_batch, yoh_train_batch)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss_history.append(loss.item())

        with torch.no_grad():
            model.eval()
            val_loss = model.get_loss(X_val, yoh_val)
            val_loss_history.append(val_loss.item())

        if i % show_freq == 0:
            if best_loss > val_loss:
                best_loss = val_loss
                best_model = deepcopy(model)

            if verbose:
                print(f"Step: {i}/{epochs}, Train Loss: {loss:.2f}, Val Loss: {val_loss:.2f}")

    if dirname:        
        checkpoint_name = f"{type(best_model).__name__}_{str(best_model._layers)[1:-1].replace(', ', '_')}"
        checkpoint_name += f"_seed={seed}_optim={type(optimizer).__name__}_epochs={epochs}_lr={lr}_weight_decay={weight_decay}"
        checkpoint_name += f"_bs={'full' if batch_size == X_train.shape[0] else batch_size}"
        checkpoint_name += f"_scheduler={type(scheduler).__name__}_lr_decay={lr_decay:.4f}" if scheduler else "_scheduler=none"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        checkpoint_name += f"_{timestamp}.pth"

        filename = os.path.join(dirname, checkpoint_name)
        save_state(filename, best_model, optimizer, train_loss_history, val_loss_history)

    return best_model, train_loss_history, val_loss_history

def visualize_weights(w, filename):
    w = w.reshape(C, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        axes[i].set_title(f"Weights for digit {i}")
        axes[i].imshow(w[i], cmap="gray")
        axes[i].axis("off")

    plt.savefig(os.path.join(IMG_DIR, filename), dpi=300)

def moving_average_with_warmup(data, window_size=50):
    moving_avg = []
    for i in range(len(data)):
        current_window = data[max(0, i - window_size + 1):i + 1]
        moving_avg.append(sum(current_window) / len(current_window))
    return moving_avg

def part1(config, X_train, yoh_train, X_test, yoh_test):
    torch.manual_seed(100)
    print("=" * 20 + " Part 1 " + "=" * 20)
    N = X_train.shape[0]
    for reg in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
        print(f"Training with regularization: {reg}")
        model = FCN(config, device=DEVICE)
        model, _, _ = train(model, X_train, yoh_train, X_test, yoh_test, epochs=3000, lr=1e-1, weight_decay=reg, batch_size=N, show_freq=100, dirname=os.path.join(CHECKPOINT_DIR, "part1"))
        _, accuracy, recall, precision = evaluate(model, X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}\n")
        visualize_weights(model.layers[0].weight.detach().cpu().numpy(), f"weights_reg_{reg}.png")

def part2(configs, X_train, yoh_train, X_test, yoh_test):
    torch.manual_seed(100)
    print("=" * 20 + " Part 2 " + "=" * 20)
    models = []
    accuracies = []
    N = X_train.shape[0]
    for config, epochs, lr in zip(configs, [2000, 2000, 5000, 5000], [1e-1, 1e-1, 1e-2, 1e-2]):
        print(f"Training config {config}")
        model = FCN(config, device=DEVICE)
        model, _, _ = train(
            model, 
            X_train, 
            yoh_train, 
            X_test, 
            yoh_test, 
            epochs=epochs, 
            lr=lr, 
            batch_size=N, 
            show_freq=200, 
            dirname=os.path.join(CHECKPOINT_DIR, "part2")
        )

        _, accuracy, recall, precision = evaluate(model, X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}\n")
        
        accuracies.append(accuracy)
        models.append(model)

    # Show best and worst digits of the best model
    best_model = models[np.argmax(accuracies)]
    probs = best_model.forward(X_test)
    losses = torch.sum(-yoh_test * torch.log(probs + 1e-12), dim=1)
    sorted_losses, sorted_indices = torch.sort(losses)

    lowest_loss_indices = sorted_indices[:10]
    lowest_loss_values = sorted_losses[:10]
    highest_loss_indices = sorted_indices[-10:]
    highest_loss_values = sorted_losses[-10:]

    lowest_loss_samples = X_test[lowest_loss_indices]
    highest_loss_samples = X_test[highest_loss_indices]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        loss = lowest_loss_values[i].detach().cpu().item()
        sample = lowest_loss_samples[i].detach().cpu().numpy()
        axes[i].set_title(f"Loss {loss}")
        axes[i].imshow(sample, cmap="gray")
        axes[i].axis("off")

    plt.savefig(os.path.join(IMG_DIR, "best_digits.png"), dpi=300)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        loss = highest_loss_values[i].detach().cpu().item()
        sample = highest_loss_samples[i].detach().cpu().numpy()
        axes[i].set_title(f"Loss {loss}")
        axes[i].imshow(sample, cmap="gray")
        axes[i].axis("off")

    plt.savefig(os.path.join(IMG_DIR, "worst_digits.png"), dpi=300)    

def part3(config, X_train, y_train, X_test, y_test):
    torch.manual_seed(100)
    print("=" * 20 + " Part 3 " + "=" * 20)
    yoh_train = class_to_onehot(y_train)
    yoh_val = class_to_onehot(y_val)
    
    model = FCN(config, device=DEVICE) 
    model, _, _ = train(model, X_train, yoh_train, X_val, yoh_val, epochs=1000, lr=1e-1, weight_decay=1e-3, batch_size=X_train.shape[0], show_freq=50, dirname=os.path.join(CHECKPOINT_DIR, "part3"))
    
    _, accuracy, recall, precision = evaluate(model, X_train, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_val, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

def part4(config, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=100):
    torch.manual_seed(100)
    print("=" * 20 + " Part 4 " + "=" * 20)
    yoh_train = class_to_onehot(y_train)
    yoh_val = class_to_onehot(y_val)

    model = FCN(config, device=DEVICE)
    model, _, _ = train(
        model, 
        X_train, 
        yoh_train, 
        X_val, 
        yoh_val, 
        epochs=5, 
        lr=1e-1, 
        weight_decay=1e-3, 
        batch_size=batch_size, 
        show_freq=1, 
        dirname=os.path.join(CHECKPOINT_DIR, "part4")
    )
    
    _, accuracy, recall, precision = evaluate(model, X_train, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_val, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

def part5(config, X_train, y_train, X_val, y_val, X_test, y_test):
    torch.manual_seed(100)
    print("=" * 20 + " Part 5 " + "=" * 20)
    yoh_train = class_to_onehot(y_train)
    yoh_val = class_to_onehot(y_val)

    model = FCN(config, device=DEVICE)
    model, _, _ = train(
        model, 
        X_train, 
        yoh_train, 
        X_val, 
        yoh_val, 
        optimizer=optim.Adam, 
        epochs=50, 
        lr=1e-4, 
        weight_decay=1e-3, 
        batch_size=100, 
        show_freq=1, 
        dirname=os.path.join(CHECKPOINT_DIR, "part5")
    )

    _, accuracy, recall, precision = evaluate(model, X_train, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_val, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

def part6(config, X_train, y_train, X_val, y_val, X_test, y_test):
    torch.manual_seed(100)
    print("=" * 20 + " Part 5 " + "=" * 20)
    yoh_train = class_to_onehot(y_train)
    yoh_val = class_to_onehot(y_val)

    model = FCN(config, device=DEVICE)
    model, _, _ = train(
        model, 
        X_train, 
        yoh_train, 
        X_val, 
        yoh_val, 
        optimizer=optim.Adam, 
        scheduler=True,
        epochs=50, 
        lr=1e-4, 
        weight_decay=1e-3, 
        batch_size=100, 
        show_freq=1, 
        dirname=os.path.join(CHECKPOINT_DIR, "part6")
    )

    _, accuracy, recall, precision = evaluate(model, X_train, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_val, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

def part7(config, X_train, y_train):
    torch.manual_seed(100)
    print("=" * 20 + " Part 7 " + "=" * 20)
    model = FCN(config, device=DEVICE)
    print("Not trained model loss:", model.get_loss(X_train, class_to_onehot(y_train)).item())
    print("Random guess loss:", -log(1 / 10))

def part8(X_train, y_train, X_val, y_val, X_test, y_test):
    print("=" * 20 + " Part 8 " + "=" * 20)
    linear = SVC(kernel="linear", random_state=100)
    rbf = SVC(kernel="rbf", random_state=100)

    # Train
    linear.fit(X_train, y_train)
    rbf.fit(X_train, y_train)

    # Evaluate
    print("Linear kernel")
    y_train_preds = linear.predict(X_train)
    _, accuracy, recall, precision = data.eval_perf_multi(y_train_preds, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    y_val_preds = linear.predict(X_val)
    _, accuracy, recall, precision = data.eval_perf_multi(y_val_preds, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    y_test_preds = linear.predict(X_test)
    _, accuracy, recall, precision = data.eval_perf_multi(y_test_preds, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    print("RBF kernel")
    y_train_preds = rbf.predict(X_train)
    _, accuracy, recall, precision = data.eval_perf_multi(y_train_preds, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    y_val_preds = rbf.predict(X_val)
    _, accuracy, recall, precision = data.eval_perf_multi(y_val_preds, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    y_test_preds = rbf.predict(X_test)
    _, accuracy, recall, precision = data.eval_perf_multi(y_test_preds, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    N = X_train.shape[0]
    D = X_train.shape[1] * X_train.shape[2]
    C = y_train.max().add_(1).item()
    h = 100
    configs = [[D, C], [D, h, C], [D, h, h, C], [D, h, h, h, C]]
    yoh_train = class_to_onehot(y_train)
    yoh_test = class_to_onehot(y_test)
    
    # Warning: Comment out parts that you don't need immediately
    # This is very slow on CPU
    part1(configs[0], X_train, yoh_train, X_test, yoh_test)
    part2(configs, X_train, yoh_train, X_test, yoh_test)

    test_ratio = 0.2
    train_size = int(X_train.shape[0] * (1 - test_ratio))
    X_val, y_val = X_train[train_size:], y_train[train_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    part3(configs[1], X_train, X_val, y_val, y_train, X_test, y_test)
    part4(configs[1], X_train, y_train, X_val, y_val, X_test, y_test, batch_size=1)
    part5(configs[1], X_train, y_train, X_val, y_val, X_test, y_test)
    part6(configs[1], X_train, y_train, X_val, y_val, X_test, y_test)
    part7(configs[1], X_train, y_train)

    X_train_np = X_train.detach().cpu().numpy().reshape(X_train.shape[0], -1)
    y_train_np = y_train.detach().cpu().numpy()
    X_val_np = X_val.detach().cpu().numpy().reshape(X_val.shape[0], -1)
    y_val_np = y_val.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy().reshape(X_test.shape[0], -1)
    y_test_np = y_test.detach().cpu().numpy()
    part8(X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np)