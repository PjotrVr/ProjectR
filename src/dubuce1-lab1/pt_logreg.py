import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
        """

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C), dtype=torch.float32)
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        s = torch.mm(X, self.W) + self.b
        return torch.softmax(s, dim=1)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        # ...
        logits = torch.mm(X, self.W) + self.b
        exp_shifted = torch.exp(logits - torch.max(logits, dim=1, keepdims=True).values)
        probs = exp_shifted / exp_shifted.sum(dim=1, keepdims=True)
        log_probs = probs.log()
        loss = -torch.sum(log_probs * Yoh_) / X.shape[0]
        return loss
    
def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0.0):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    # ...
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...
    show_freq = param_niter // 10
    for i in range(param_niter):
        optimizer.zero_grad()
        l2_reg = torch.sum(model.W**2)
        loss = model.get_loss(X, Yoh_) + param_lambda * l2_reg
        loss.backward()
        optimizer.step()
        if i % show_freq == show_freq - 1:
            print(f'step: {i}, loss: {loss}')

def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
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

def logreg_decfun(model):
    def classify(X):
        probs = eval(model, X)
        return probs.argmax(axis=1)
        
    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    torch.manual_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)
    X_torch = torch.tensor(X, dtype=torch.float32)
    Y_torch = torch.tensor(Y_, dtype=torch.float32)
    Yoh_torch = torch.tensor(data.class_to_onehot(Y_), dtype=torch.float32)

    # definiraj model:
    ptlr = PTLogreg(X_torch.shape[1], Yoh_torch.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X_torch, Yoh_torch, 1000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = probs.argmax(axis=1).reshape(-1, 1)
    
    # ispiši performansu (preciznost i odziv po razredima)
    M, accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(f"Ukupna tocnost: {accuracy}, Odziv: {recall}, Preciznost: {precision}")
    
    # # iscrtaj rezultate, decizijsku plohu
    decfun = logreg_decfun(ptlr)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))

    #graph_surface(decfun, bbox, 0.5)
    data.graph_surface(decfun, bbox, None)
    data.graph_data(X, Y_.reshape(-1), Y.reshape(-1))
    plt.show()
