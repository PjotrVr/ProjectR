import torch
import torch.optim as optim

def gen_points(a, b, N=10):
    X = torch.rand(N)
    Y = a*X + b
    return X, Y

def pt_linreg_train(X, Y, epochs=1000, lr=0.1):
    a = torch.rand(1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)
    optimizer = optim.SGD([a, b], lr=lr)
    show_freq = epochs // 10

    for i in range(epochs):
        optimizer.zero_grad()
        Y_ = a*X + b
        diff = Y - Y_
        loss = diff.pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % show_freq == show_freq - 1:
            print(f'step: {i}, loss: {loss}, Y_: {Y_}, a: {a}, b: {b}')

    return a, b

def grad_check():
    a = torch.rand(1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)
    X, Y = gen_points(2, 3)
    Y_ = a*X + b
    diff = Y - Y_
    loss = diff.pow(2).mean()
    loss.backward()
    print(f'a.grad:{a.grad}, b.grad:{b.grad}')
    dL_da = -2 * (diff * X).mean()
    dL_db = -2 * (diff).mean()
    print(f'dL_da:{dL_da}, dL_db:{dL_db}')
    assert torch.allclose(a.grad, dL_da)
    assert torch.allclose(b.grad, dL_db)
    #print('grad check a:', torch.allclose(a.grad, dL_da))
    #print('grad check b:', torch.allclose(b.grad, dL_db))


if __name__ == '__main__':
    grad_check()
    X, Y = gen_points(2, 3)
    a, b = pt_linreg_train(X, Y)
    print(f'a:{a}, b:{b}')