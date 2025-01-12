import numpy as np
import matplotlib.pyplot as plt
import data

def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.05, param_lambda=1e-3):
    num_classes = np.max(Y_) + 1
    h = 5
    W1 = np.random.randn(X.shape[1], h)
    b1 = np.zeros((h,))
    W2 = np.random.randn(h, num_classes)
    b2 = np.zeros((num_classes,))

    Y_oh = data.class_to_onehot(Y_)

    for i in range(int(param_niter)):
        # Forward
        s1 = X @ W1 + b1
        h1 = np.maximum(s1, 0)
        s2 = h1 @ W2 + b2
        
        # Loss
        P = softmax(s2)
        log_P = np.log(P)
        l2_reg = 0.5 * param_lambda * (np.sum(W1**2) + np.sum(W2**2))
        loss = -1 / X.shape[0] * np.sum(log_P[np.arange(X.shape[0]), Y_]) + l2_reg

        # Backward
        dL_ds2 = (P - Y_oh) / X.shape[0]
        dL_dW2 =  h1.T @ dL_ds2 + param_lambda * W2
        dL_db2 = dL_ds2.sum(0)
        dL_dh1 = dL_ds2 @ W2.T
        dL_ds1 = dL_dh1 * (s1 > 0)
        dL_dW1 = X.T @ dL_ds1 + param_lambda * W1
        dL_db1 = dL_ds1.sum(0)

        # Update GD
        W1 = W1 - param_delta * dL_dW1
        b1 = b1 - param_delta * dL_db1
        W2 = W2 - param_delta * dL_dW2
        b2 = b2 - param_delta * dL_db2

    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    s1 = X @ W1 + b1
    h1 = np.maximum(s1, 0)
    s2 = h1 @ W2 + b2
    return s2

def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        scores = fcann2_classify(X, W1, b1, W2, b2)
        return softmax(scores)[:, 1]

    return classify

if __name__ == "__main__":
    np.random.seed(100)
    K, C, N, h = 6, 2, 10, 5
    X, Y_ = data.sample_gmm_2d(K, C, N)

    W1, b1, W2, b2 = fcann2_train(X, Y_)
    scores = fcann2_classify(X, W1, b1, W2, b2)
    centered_scores = scores - scores[:, 0].reshape(-1, 1)
    Y = softmax(scores).argmax(axis=1)
    classify = fcann2_decfun(W1, b1, W2, b2)

    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(classify, rect)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()
