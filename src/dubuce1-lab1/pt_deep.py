import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
import matplotlib.pyplot as plt

class PTDeep(nn.Module):
    def __init__(self, layers, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.params = nn.ParameterDict()
        for i, (in_dim, out_dim) in enumerate(zip(layers, layers[1:])):
            self.params[f"W{i}"] = nn.Parameter(torch.randn((in_dim, out_dim), dtype=torch.float32))
            self.params[f"b{i}"] = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X):
        num_layers = len(self.params) // 2
        for i in range(num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            X = torch.matmul(X, W) + b
            if i < num_layers - 1:
                X = self.activation(X)
        return torch.softmax(X, dim=1)

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

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0.0):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    torch.manual_seed(100)
    # inicijalizacija optimizatora
    # ...
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...
    show_freq = param_niter // 10
    num_layers = len(model.params) // 2
    for i in range(param_niter):
        optimizer.zero_grad()
        l2_reg = 0.0
        for j in range(num_layers):
            l2_reg += torch.sum(model.params[f"W{j}"]**2)
        loss = model.get_loss(X, Yoh_) + param_lambda * l2_reg
        loss.backward()
        optimizer.step()
        if i % show_freq == show_freq - 1:
            print(f"Step: {i}, Loss: {loss}")

def eval(model, X):
    """Arguments:
        - model: type: PTDeep
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X_torch = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model.forward(X_torch)
    return probs.detach().numpy()

def deep_decfun(model):
    def classify(X):
        probs = eval(model, X)
        return probs.argmax(axis=1)
        
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    torch.manual_seed(100)
    #X, Y_ = data.sample_gauss_2d(3, 100)
    #X, Y_ = data.sample_gmm_2d(4, 2, 40)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    X_torch = torch.tensor(X, dtype=torch.float32)
    Y_torch = torch.tensor(Y_, dtype=torch.float32)
    Yoh_torch = torch.tensor(data.class_to_onehot(Y_), dtype=torch.float32)

    dims = [2, 10, 10, 2]
    ptd = PTDeep(dims)

    train(ptd, X_torch, Yoh_torch, 10_000, 0.1, 1e-4)

    probs = eval(ptd, X)
    Y = probs.argmax(axis=1).reshape(-1, 1)

    # ispiši performansu (preciznost i odziv po razredima)
    M, accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(f"Ukupna tocnost: {accuracy}, Odziv: {recall}, Preciznost: {precision}")

    # iscrtaj rezultate, decizijsku plohu
    decfun = deep_decfun(ptd)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(decfun, bbox, 0.5)
    #data.graph_surface(decfun, bbox, None)
    data.graph_data(X, Y_.reshape(-1), Y.reshape(-1))
    plt.show()

