import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from mnist_shootout import Linear, train, evaluate, moving_average_with_warmup, load_mnist, class_to_onehot

# Globals
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchNorm(nn.Module):
    def __init__(self, device="cpu", eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.device = device
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(1, device=device))
        self.beta = nn.Parameter(torch.zeros(1, device=device))

        self.register_buffer("running_mean", torch.zeros(1, device=device))
        self.register_buffer("running_var", torch.ones(1, device=device))
        
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"expected shape 2D input, got {x.dim()}D input")
        
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            variance = x.var(dim=0, unbiased=False, keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance.detach()
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        x_normalized = self.gamma * x_normalized + self.beta
        return x_normalized
    
class FCN_BN(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, BN=True, device="cpu"):
        super().__init__()
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
                if BN:
                    self.layers.add_module(f"bn{i}", BatchNorm(device=self.device))
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
    
if __name__ == "__main__":
    train_ratio = 0.8
    X_train, y_train, X_test, y_test = load_mnist()
    train_size = int(X_train.shape[0] * train_ratio)
    X_val, y_val = X_train[train_size:], y_train[train_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    yoh_train = class_to_onehot(y_train)
    yoh_val = class_to_onehot(y_val)
    yoh_test = class_to_onehot(y_test)

    # Train
    model = FCN_BN([784, 100, 10], device=X_train.device.type)
    model, train_loss_history, val_loss_history = train(
        model, 
        X_train, 
        yoh_train, 
        X_val, 
        yoh_val, 
        optimizer=optim.SGD, 
        num_epochs=10, 
        lr=1e-1, 
        weight_decay=1e-3, 
        show_freq=1, 
        batch_size=100
    )

    # Metrics
    _, accuracy, recall, precision = evaluate(model, X_train, y_train)
    print("Train set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_val, y_val)
    print("Validation set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    _, accuracy, recall, precision = evaluate(model, X_test, y_test)
    print("Test set")
    print(f"=> Accuracy: {accuracy:.2f}, Recall: {recall}, Precision: {precision}")

    #model.count_params()
    # Visualize training
    train_loss_history_smooth = moving_average_with_warmup(train_loss_history)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    axes[0].plot(train_loss_history, color="blue", alpha=0.2)
    axes[0].plot(train_loss_history_smooth, color="blue", label="Train Loss")
    axes[0].legend(fontsize=12)
    axes[0].set_title("Train", fontsize=16)
    axes[0].set_xlabel("Steps", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].grid(True)

    axes[1].plot(val_loss_history, label="Validation", color="red")
    axes[1].legend(fontsize=12)
    axes[1].set_title("Validation Loss", fontsize=16)
    axes[1].set_xlabel("Steps", fontsize=14)
    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    
